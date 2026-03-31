import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import masked_mae


def _build_valid_mask(real_value: torch.Tensor, null_val: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    """构建有效位置掩码，避免无效值污染蒸馏统计。"""
    if isinstance(null_val, float) and math.isnan(null_val):
        valid_mask = ~torch.isnan(real_value)
    else:
        valid_mask = torch.abs(real_value - null_val) > eps
    return valid_mask.float()


def _masked_reduce_mean(value: torch.Tensor, mask: torch.Tensor, dim, keepdim: bool = True) -> torch.Tensor:
    """安全地计算带掩码平均值。"""
    numerator = (value * mask).sum(dim=dim, keepdim=keepdim)
    denominator = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numerator / denominator


def _normalize_to_confidence(error_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    将误差映射为 [0, 1] 的连续可信度。
    误差越小，可信度越高。
    """
    flat_error = error_map.reshape(-1)
    min_error = flat_error.min()
    max_error = flat_error.max()
    if float(max_error - min_error) < eps:
        return torch.ones_like(error_map)
    normalized_error = (error_map - min_error) / (max_error - min_error + eps)
    return (1.0 - normalized_error).clamp(0.0, 1.0)


def compute_confidence_score(
    teacher_pred: torch.Tensor,
    real_value: torch.Tensor,
    null_val: float = 0.0,
    confidence_power: float = 1.0,
):
    """
    连续可信度估计：
    - 先按节点维、horizon 维统计教师误差
    - 再将误差映射成 [0, 1] 的连续可信度
    - 最终得到节点-预测步联合可信度分数

    teacher_pred / real_value: [B, 1, N, H]
    """
    valid_mask = _build_valid_mask(real_value.detach(), null_val=null_val)
    teacher_error = torch.abs(teacher_pred.detach() - real_value.detach())

    node_error = _masked_reduce_mean(teacher_error, valid_mask, dim=(0, 1, 3), keepdim=True)  # [1,1,N,1]
    horizon_error = _masked_reduce_mean(teacher_error, valid_mask, dim=(0, 1, 2), keepdim=True)  # [1,1,1,H]

    node_confidence = _normalize_to_confidence(node_error)
    horizon_confidence = _normalize_to_confidence(horizon_error)

    confidence_score = (node_confidence * horizon_confidence).clamp(0.0, 1.0)
    if confidence_power != 1.0:
        confidence_score = confidence_score.pow(confidence_power)

    confidence_score = confidence_score * valid_mask
    mean_confidence = confidence_score.sum().item() / valid_mask.sum().clamp_min(1.0).item()

    return {
        "node_confidence": node_confidence,
        "horizon_confidence": horizon_confidence,
        "confidence_score": confidence_score,
        "mean_confidence": mean_confidence,
    }


def compute_curriculum_map(horizon_count: int, current_epoch: int, total_epochs: int, device) -> torch.Tensor:
    """多步预测课程蒸馏：由短期到长期逐步开放 horizon。"""
    if total_epochs <= 0:
        visible_horizon = horizon_count
    else:
        progress = current_epoch / total_epochs
        if progress <= 1 / 3:
            visible_horizon = max(1, horizon_count // 3)
        elif progress <= 2 / 3:
            visible_horizon = max(1, (2 * horizon_count) // 3)
        else:
            visible_horizon = horizon_count

    curriculum_map = torch.zeros((1, 1, 1, horizon_count), device=device)
    curriculum_map[..., :visible_horizon] = 1.0
    return curriculum_map


def compute_relation_matrix(node_feature: torch.Tensor) -> torch.Tensor:
    """保留旧接口，兼容已有代码与分析脚本。"""
    if node_feature.size(-1) > 1:
        pooled_feature = node_feature.mean(dim=-1)
    else:
        pooled_feature = node_feature.squeeze(-1)

    node_embedding = pooled_feature.transpose(1, 2)
    node_embedding = F.normalize(node_embedding, dim=-1)
    relation_matrix = torch.bmm(node_embedding, node_embedding.transpose(1, 2))
    return relation_matrix


class RegressionDistillationLoss(nn.Module):
    """
    v4 默认目标：
    - hard supervision
    - confidence-aware absolute distillation
    - low-confidence trend distillation
    - curriculum distillation
    """

    def __init__(
        self,
        hard_weight=0.7,
        soft_weight=0.3,
        feature_weight=0.0,
        relation_weight=0.0,
        temperature=3.0,
        enable_confidence_filter=True,
        enable_curriculum=True,
        confidence_power=1.0,
        trend_weight=0.5,
    ):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.feature_weight = feature_weight
        self.relation_weight = relation_weight
        self.temperature = temperature
        self.enable_confidence_filter = enable_confidence_filter
        self.enable_curriculum = enable_curriculum
        self.confidence_power = confidence_power
        self.trend_weight = trend_weight

    def forward(
        self,
        student_pred,
        teacher_pred,
        real_value,
        student_feature,
        teacher_feature,
        current_epoch,
        total_epochs,
        null_val=0.0,
    ):
        hard_loss = masked_mae(student_pred, real_value, null_val)
        valid_mask = _build_valid_mask(real_value, null_val=null_val)

        confidence_items = compute_confidence_score(
            teacher_pred=teacher_pred,
            real_value=real_value,
            null_val=null_val,
            confidence_power=self.confidence_power,
        )

        if self.enable_confidence_filter:
            confidence_score = confidence_items["confidence_score"]
        else:
            confidence_score = valid_mask

        if self.enable_curriculum:
            curriculum_map = compute_curriculum_map(
                horizon_count=student_pred.size(-1),
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                device=student_pred.device,
            )
        else:
            curriculum_map = torch.ones((1, 1, 1, student_pred.size(-1)), device=student_pred.device)

        absolute_mask = confidence_score * curriculum_map
        if absolute_mask.sum().detach().item() <= 0:
            absolute_mask = valid_mask

        soft_student = student_pred / self.temperature
        soft_teacher = teacher_pred.detach() / self.temperature
        absolute_element = F.smooth_l1_loss(soft_student, soft_teacher, reduction="none")
        absolute_loss = (absolute_element * absolute_mask).sum() / absolute_mask.sum().clamp_min(1.0)
        absolute_loss = absolute_loss * (self.temperature ** 2)

        if student_pred.size(-1) > 1:
            trend_student = soft_student[..., 1:] - soft_student[..., :-1]
            trend_teacher = soft_teacher[..., 1:] - soft_teacher[..., :-1]
            trend_valid_mask = valid_mask[..., 1:] * valid_mask[..., :-1]
            trend_confidence = 0.5 * (confidence_score[..., 1:] + confidence_score[..., :-1])
            trend_curriculum = curriculum_map[..., 1:] * curriculum_map[..., :-1]
            trend_mask = (1.0 - trend_confidence).clamp(0.0, 1.0) * trend_valid_mask * trend_curriculum

            if not self.enable_confidence_filter:
                trend_mask = torch.zeros_like(trend_mask)

            if trend_mask.sum().detach().item() > 0:
                trend_element = F.smooth_l1_loss(trend_student, trend_teacher, reduction="none")
                trend_loss = (trend_element * trend_mask).sum() / trend_mask.sum().clamp_min(1.0)
                trend_loss = trend_loss * (self.temperature ** 2)
            else:
                trend_loss = torch.zeros(1, device=student_pred.device, dtype=student_pred.dtype).squeeze()
        else:
            trend_loss = torch.zeros(1, device=student_pred.device, dtype=student_pred.dtype).squeeze()
            trend_mask = torch.zeros_like(student_pred[..., :0])

        combined_soft_loss = (absolute_loss + self.trend_weight * trend_loss) / (1.0 + self.trend_weight)

        if self.feature_weight > 0.0:
            feature_loss = F.mse_loss(student_feature, teacher_feature.detach())
        else:
            feature_loss = torch.zeros(1, device=student_pred.device, dtype=student_pred.dtype).squeeze()

        if self.relation_weight > 0.0:
            teacher_relation = compute_relation_matrix(teacher_feature.detach())
            student_relation = compute_relation_matrix(student_feature)
            relation_loss = F.mse_loss(student_relation, teacher_relation)
        else:
            relation_loss = torch.zeros(1, device=student_pred.device, dtype=student_pred.dtype).squeeze()

        total_loss = (
            self.hard_weight * hard_loss
            + self.soft_weight * combined_soft_loss
            + self.feature_weight * feature_loss
            + self.relation_weight * relation_loss
        )

        visible_horizon = int(curriculum_map.sum().item())
        mean_node_confidence = confidence_items["node_confidence"].mean().item()
        mean_horizon_confidence = confidence_items["horizon_confidence"].mean().item()
        mean_trend_ratio = (
            trend_mask.sum().item() / trend_valid_mask.sum().clamp_min(1.0).item()
            if student_pred.size(-1) > 1
            else 0.0
        )

        return total_loss, {
            "hard_loss": hard_loss.item(),
            "soft_loss": combined_soft_loss.item(),
            "absolute_loss": absolute_loss.item(),
            "trend_loss": trend_loss.item(),
            "feature_loss": feature_loss.item(),
            "relation_loss": relation_loss.item(),
            "visible_horizon": visible_horizon,
            "mean_node_weight": mean_node_confidence,
            "mean_horizon_weight": mean_horizon_confidence,
            "confidence_keep_ratio": confidence_items["mean_confidence"],
            "trend_ratio": mean_trend_ratio,
            "confidence_enabled": int(self.enable_confidence_filter),
            "curriculum_enabled": int(self.enable_curriculum),
        }
