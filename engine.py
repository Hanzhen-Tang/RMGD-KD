import time

import torch
import torch.nn as nn
import torch.optim as optim

import util
from losses import RegressionDistillationLoss


def prepare_batch(x, y, device):
    """Convert a numpy batch into model-ready tensors."""
    inputs = torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device).transpose(1, 3)
    targets = targets[:, 0, :, :]
    return inputs, targets


def count_parameters(model):
    """Count trainable parameters."""
    if model is None:
        return 0
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class TeacherTrainer:
    """Trainer wrapper for the teacher model."""

    def __init__(self, model, scaler, learning_rate, weight_decay, clip=5.0):
        self.model = model
        self.scaler = scaler
        self.clip = clip
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = util.masked_mae

    def _shared_step(self, inputs, targets, training: bool):
        if training:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        padded_inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        outputs = self.model(padded_inputs).transpose(1, 3)
        real = targets.unsqueeze(1)
        pred = self.scaler.inverse_transform(outputs)
        loss = self.loss(pred, real, 0.0)

        if training:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        mape = util.masked_mape(pred, real, 0.0).item()
        rmse = util.masked_rmse(pred, real, 0.0).item()
        return {
            "loss": loss.item(),
            "mae": loss.item(),
            "mape": mape,
            "rmse": rmse,
            "pred": pred.detach(),
            "real": real.detach(),
        }

    def train_batch(self, inputs, targets):
        return self._shared_step(inputs, targets, training=True)

    @torch.no_grad()
    def eval_batch(self, inputs, targets):
        return self._shared_step(inputs, targets, training=False)


class DistillationTrainer:
    """Trainer wrapper for student distillation."""

    def __init__(
        self,
        teacher_model,
        student_model,
        supports,
        scaler,
        learning_rate,
        weight_decay,
        hard_weight=0.7,
        soft_weight=0.3,
        trend_weight=0.5,
        feature_weight=0.0,
        relation_weight=0.0,
        temperature=3.0,
        enable_confidence_filter=True,
        enable_curriculum=True,
        confidence_power=1.0,
        curriculum_mode="standard",
        clip=5.0,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.supports = supports
        self.scaler = scaler
        self.clip = clip
        self.current_epoch = 1
        self.total_epochs = 1
        self.use_feature_alignment = feature_weight > 0.0 or relation_weight > 0.0

        self.distill_loss = RegressionDistillationLoss(
            hard_weight=hard_weight,
            soft_weight=soft_weight,
            trend_weight=trend_weight,
            feature_weight=feature_weight,
            relation_weight=relation_weight,
            temperature=temperature,
            enable_confidence_filter=enable_confidence_filter,
            enable_curriculum=enable_curriculum,
            confidence_power=confidence_power,
            curriculum_mode=curriculum_mode,
        )

        if self.use_feature_alignment:
            teacher_feature_dim = getattr(self.teacher_model, "feature_dim")
            student_feature_dim = getattr(self.student_model, "feature_dim")
            self.feature_adapter = nn.Conv2d(student_feature_dim, teacher_feature_dim, kernel_size=(1, 1)).to(
                next(self.student_model.parameters()).device
            )
        else:
            self.feature_adapter = None

        params = list(self.student_model.parameters())
        if self.feature_adapter is not None:
            params += list(self.feature_adapter.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def set_epoch(self, current_epoch: int, total_epochs: int):
        self.current_epoch = current_epoch
        self.total_epochs = max(total_epochs, 1)

    def _shared_step(self, inputs, targets, training: bool):
        if training:
            self.student_model.train()
            self.optimizer.zero_grad()
        else:
            self.student_model.eval()

        real = targets.unsqueeze(1)
        teacher_inputs = nn.functional.pad(inputs, (1, 0, 0, 0))

        if not training:
            step_start = time.perf_counter()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(teacher_inputs, return_features=True)

        student_outputs = self.student_model(inputs, self.supports, return_features=True)

        teacher_pred = self.scaler.inverse_transform(teacher_outputs["prediction"].transpose(1, 3))
        student_pred = self.scaler.inverse_transform(student_outputs["prediction"].transpose(1, 3))

        if self.feature_adapter is not None:
            student_feature_for_loss = self.feature_adapter(student_outputs["hidden_state"])
        else:
            student_feature_for_loss = student_outputs["hidden_state"]

        total_loss, loss_items = self.distill_loss(
            student_pred=student_pred,
            teacher_pred=teacher_pred,
            real_value=real,
            student_feature=student_feature_for_loss,
            teacher_feature=teacher_outputs["hidden_state"],
            current_epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            null_val=0.0,
        )

        if training:
            total_loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.clip)
                if self.feature_adapter is not None:
                    torch.nn.utils.clip_grad_norm_(self.feature_adapter.parameters(), self.clip)
            self.optimizer.step()

        latency = (time.perf_counter() - step_start) * 1000.0 if not training else 0.0

        mae = util.masked_mae(student_pred, real, 0.0).item()
        mape = util.masked_mape(student_pred, real, 0.0).item()
        rmse = util.masked_rmse(student_pred, real, 0.0).item()
        return {
            "loss": total_loss.item(),
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "pred": student_pred.detach(),
            "real": real.detach(),
            "teacher_pred": teacher_pred.detach(),
            "student_hidden": student_outputs["hidden_state"].detach(),
            "teacher_hidden": teacher_outputs["hidden_state"].detach(),
            "latency_ms": latency,
            **loss_items,
        }

    def train_batch(self, inputs, targets):
        return self._shared_step(inputs, targets, training=True)

    @torch.no_grad()
    def eval_batch(self, inputs, targets):
        return self._shared_step(inputs, targets, training=False)
