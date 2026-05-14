from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import torch


def _normalize_to_unit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)

    result = np.zeros_like(values, dtype=np.float64)
    valid_values = values[finite]
    min_value = valid_values.min()
    max_value = valid_values.max()
    if max_value - min_value < eps:
        result[finite] = 0.0
    else:
        result[finite] = (valid_values - min_value) / (max_value - min_value + eps)
    return result


def _inverse_minmax_confidence(error: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    normalized = _normalize_to_unit(error, eps=eps)
    return np.clip(1.0 - normalized, 0.0, 1.0)


def compute_soft_curriculum_weights(
    horizon_count: int,
    current_epoch: int,
    total_epochs: int,
) -> np.ndarray:
    """Return the fixed soft-curriculum horizon weights used as DDASC's floor."""
    progress = 1.0 if total_epochs <= 0 else current_epoch / max(total_epochs, 1)
    if progress <= 0.5:
        min_weight = 0.4 + 0.6 * (progress / 0.5)
    else:
        min_weight = 1.0
    return np.linspace(1.0, min_weight, num=horizon_count, dtype=np.float64)


def compute_horizon_signal_sums(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    real_value: torch.Tensor,
    null_val: float = 0.0,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Accumulate validation signals over the last horizon dimension.

    Tensors are expected to use the distillation layout [B, 1, N, H].
    """
    with torch.no_grad():
        valid_mask = (torch.abs(real_value.detach() - null_val) > eps).float()
        reduce_dims = tuple(range(valid_mask.ndim - 1))
        valid_count = valid_mask.sum(dim=reduce_dims)
        student_error_sum = (torch.abs(student_pred.detach() - real_value.detach()) * valid_mask).sum(dim=reduce_dims)
        teacher_student_gap_sum = (
            torch.abs(student_pred.detach() - teacher_pred.detach()) * valid_mask
        ).sum(dim=reduce_dims)
        teacher_error_sum = (torch.abs(teacher_pred.detach() - real_value.detach()) * valid_mask).sum(dim=reduce_dims)

    return {
        "valid_count": valid_count.cpu().numpy().astype(np.float64),
        "student_error_sum": student_error_sum.cpu().numpy().astype(np.float64),
        "teacher_student_gap_sum": teacher_student_gap_sum.cpu().numpy().astype(np.float64),
        "teacher_error_sum": teacher_error_sum.cpu().numpy().astype(np.float64),
    }


def reduce_horizon_signal_sums(signal_sums: Dict[str, np.ndarray], eps: float = 1e-6) -> Dict[str, np.ndarray]:
    valid_count = np.asarray(signal_sums["valid_count"], dtype=np.float64)
    denominator = np.maximum(valid_count, 1.0)
    student_error = np.asarray(signal_sums["student_error_sum"], dtype=np.float64) / denominator
    teacher_student_gap = np.asarray(signal_sums["teacher_student_gap_sum"], dtype=np.float64) / denominator
    teacher_error = np.asarray(signal_sums["teacher_error_sum"], dtype=np.float64) / denominator

    if (valid_count <= 0).any() and (valid_count > 0).any():
        max_valid_error = teacher_error[valid_count > 0].max()
        teacher_error = teacher_error.copy()
        teacher_error[valid_count <= 0] = max_valid_error

    teacher_confidence = _inverse_minmax_confidence(teacher_error, eps=eps)
    teacher_confidence = np.where(valid_count > 0, teacher_confidence, 0.0)

    return {
        "student_error": student_error,
        "teacher_student_gap": teacher_student_gap,
        "teacher_confidence": teacher_confidence,
        "valid_count": valid_count,
    }


@dataclass
class DynamicCurriculumConfig:
    alpha: float = 0.45
    beta: float = 0.35
    gamma: float = 0.20
    eta: float = 0.70
    ema: float = 0.90
    warmup: int = 5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class DynamicCurriculumScheduler:
    """
    Dynamic Difficulty-Aware Soft Curriculum (DDASC).

    The scheduler keeps the existing soft curriculum as a floor and only
    increases horizon weights when validation signals indicate readiness.
    """

    def __init__(
        self,
        horizon_count: int,
        total_epochs: int,
        config: Optional[DynamicCurriculumConfig] = None,
        eps: float = 1e-6,
    ):
        self.horizon_count = int(horizon_count)
        self.total_epochs = max(int(total_epochs), 1)
        self.config = config or DynamicCurriculumConfig()
        self.eps = eps
        self.student_error_ema: Optional[np.ndarray] = None
        self.teacher_student_gap_ema: Optional[np.ndarray] = None
        self.teacher_confidence_ema: Optional[np.ndarray] = None

    def base_weights(self, current_epoch: int) -> np.ndarray:
        return compute_soft_curriculum_weights(self.horizon_count, current_epoch, self.total_epochs)

    def current_weights(self, current_epoch: int) -> np.ndarray:
        base = self.base_weights(current_epoch)
        if current_epoch <= self.config.warmup or self.student_error_ema is None:
            return base
        return self._dynamic_weights(base)

    def state_for_epoch(self, current_epoch: int) -> Dict[str, Optional[np.ndarray]]:
        base = self.base_weights(current_epoch)
        if current_epoch <= self.config.warmup or self.student_error_ema is None:
            return {
                "base_weights": base,
                "weights": base,
                "difficulty": None,
                "readiness": None,
            }
        difficulty, readiness = self._difficulty_and_readiness()
        return {
            "base_weights": base,
            "weights": self._dynamic_weights(base, readiness=readiness),
            "difficulty": difficulty,
            "readiness": readiness,
        }

    def update(
        self,
        student_error: np.ndarray,
        teacher_student_gap: np.ndarray,
        teacher_confidence: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        student_error = self._as_horizon_array(student_error)
        teacher_student_gap = self._as_horizon_array(teacher_student_gap)
        teacher_confidence = np.clip(self._as_horizon_array(teacher_confidence), 0.0, 1.0)

        self.student_error_ema = self._ema_update(self.student_error_ema, student_error)
        self.teacher_student_gap_ema = self._ema_update(self.teacher_student_gap_ema, teacher_student_gap)
        self.teacher_confidence_ema = self._ema_update(self.teacher_confidence_ema, teacher_confidence)

        difficulty, readiness = self._difficulty_and_readiness()
        return {
            "student_error_ema": self.student_error_ema.copy(),
            "teacher_student_gap_ema": self.teacher_student_gap_ema.copy(),
            "teacher_confidence_ema": self.teacher_confidence_ema.copy(),
            "difficulty": difficulty,
            "readiness": readiness,
        }

    def final_state(self) -> Dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "student_error_ema": self._optional_list(self.student_error_ema),
            "teacher_student_gap_ema": self._optional_list(self.teacher_student_gap_ema),
            "teacher_confidence_ema": self._optional_list(self.teacher_confidence_ema),
        }

    def _as_horizon_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape[0] != self.horizon_count:
            raise ValueError(f"Expected {self.horizon_count} horizon values, got {values.shape[0]}")
        return values

    def _ema_update(self, previous: Optional[np.ndarray], current: np.ndarray) -> np.ndarray:
        if previous is None:
            return current.copy()
        ema = float(np.clip(self.config.ema, 0.0, 0.999))
        return ema * previous + (1.0 - ema) * current

    def _difficulty_and_readiness(self) -> tuple[np.ndarray, np.ndarray]:
        if self.student_error_ema is None or self.teacher_student_gap_ema is None or self.teacher_confidence_ema is None:
            difficulty = np.zeros(self.horizon_count, dtype=np.float64)
            readiness = np.ones(self.horizon_count, dtype=np.float64)
            return difficulty, readiness

        difficulty = (
            self.config.alpha * self.student_error_ema
            + self.config.beta * self.teacher_student_gap_ema
            + self.config.gamma * (1.0 - self.teacher_confidence_ema)
        )
        readiness = 1.0 - _normalize_to_unit(difficulty, eps=self.eps)
        readiness = np.clip(readiness, 0.0, 1.0)
        return difficulty, readiness

    def _dynamic_weights(self, base: np.ndarray, readiness: Optional[np.ndarray] = None) -> np.ndarray:
        if readiness is None:
            _, readiness = self._difficulty_and_readiness()
        weights = base + self.config.eta * (1.0 - base) * readiness
        weights = np.clip(weights, base, 1.0)
        weights = self._enforce_short_to_long_monotonic(weights)
        return weights

    @staticmethod
    def _enforce_short_to_long_monotonic(weights: np.ndarray) -> np.ndarray:
        weights = weights.copy()
        for idx in range(1, weights.shape[0]):
            weights[idx] = min(weights[idx], weights[idx - 1])
        return weights

    @staticmethod
    def _optional_list(values: Optional[np.ndarray]):
        return None if values is None else values.astype(float).tolist()
