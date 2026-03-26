import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import masked_mae


def _normalize_weight_map(weight_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a weight map so its mean is close to 1."""
    return weight_map / (weight_map.mean().detach() + eps)


def _build_valid_mask(real_value: torch.Tensor, null_val: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    """Build a binary mask for valid regression targets."""
    if torch.isnan(torch.tensor(null_val, device=real_value.device)):
        valid_mask = ~torch.isnan(real_value)
    else:
        valid_mask = torch.abs(real_value - null_val) > eps
    return valid_mask.float()


def _masked_reduce_mean(value: torch.Tensor, mask: torch.Tensor, dim, keepdim: bool = True) -> torch.Tensor:
    """Compute a masked mean with safe denominator handling."""
    numerator = (value * mask).sum(dim=dim, keepdim=keepdim)
    denominator = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numerator / denominator


def compute_reliability_map(teacher_pred: torch.Tensor, real_value: torch.Tensor, null_val: float = 0.0) -> dict:
    """
    Build node-wise and horizon-wise reliability weights from teacher error.

    teacher_pred / real_value: [B, 1, N, H]
    """
    valid_mask = _build_valid_mask(real_value.detach(), null_val=null_val)
    teacher_error = torch.abs(teacher_pred.detach() - real_value.detach())

    node_error = _masked_reduce_mean(teacher_error, valid_mask, dim=(0, 1, 3), keepdim=True)  # [1, 1, N, 1]
    node_weight = _normalize_weight_map(torch.exp(-node_error))

    horizon_error = _masked_reduce_mean(teacher_error, valid_mask, dim=(0, 1, 2), keepdim=True)  # [1, 1, 1, H]
    horizon_weight = _normalize_weight_map(torch.exp(-horizon_error))

    reliability_map = _normalize_weight_map(node_weight * horizon_weight)
    return {
        "node_weight": node_weight,
        "horizon_weight": horizon_weight,
        "reliability_map": reliability_map,
        "valid_mask": valid_mask,
    }


def compute_curriculum_map(horizon_count: int, current_epoch: int, total_epochs: int, device) -> torch.Tensor:
    """
    Curriculum schedule:
    - first 1/3 epochs: first 1/3 horizons
    - middle 1/3 epochs: first 2/3 horizons
    - last 1/3 epochs: all horizons
    """
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
    """
    Compute node relation matrix from node features.

    node_feature: [B, C, N, 1] or [B, C, N, T]
    output: [B, N, N]
    """
    if node_feature.size(-1) > 1:
        pooled_feature = node_feature.mean(dim=-1)
    else:
        pooled_feature = node_feature.squeeze(-1)

    node_embedding = pooled_feature.transpose(1, 2)  # [B, N, C]
    node_embedding = F.normalize(node_embedding, dim=-1)
    relation_matrix = torch.bmm(node_embedding, node_embedding.transpose(1, 2))
    return relation_matrix


class RegressionDistillationLoss(nn.Module):
    """Regression distillation loss for lightweight traffic forecasting."""

    def __init__(
        self,
        hard_weight=0.6,
        soft_weight=0.2,
        feature_weight=0.1,
        relation_weight=0.1,
        temperature=3.0,
        enable_reliability=True,
        enable_curriculum=True,
    ):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.feature_weight = feature_weight
        self.relation_weight = relation_weight
        self.temperature = temperature
        self.enable_reliability = enable_reliability
        self.enable_curriculum = enable_curriculum

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

        reliability_items = compute_reliability_map(teacher_pred, real_value, null_val=null_val)
        if self.enable_reliability:
            reliability_map = reliability_items["reliability_map"]
        else:
            reliability_map = torch.ones_like(student_pred)

        if self.enable_curriculum:
            curriculum_map = compute_curriculum_map(
                horizon_count=student_pred.size(-1),
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                device=student_pred.device,
            )
        else:
            curriculum_map = torch.ones((1, 1, 1, student_pred.size(-1)), device=student_pred.device)

        weighted_map = reliability_map * curriculum_map
        if weighted_map.sum().detach().item() <= 0:
            weighted_map = torch.ones_like(weighted_map)
        weighted_map = _normalize_weight_map(weighted_map)

        soft_student = student_pred / self.temperature
        soft_teacher = teacher_pred.detach() / self.temperature
        soft_element = F.smooth_l1_loss(soft_student, soft_teacher, reduction="none")
        soft_loss = (soft_element * weighted_map).mean() * (self.temperature ** 2)

        feature_loss = F.mse_loss(student_feature, teacher_feature.detach())

        teacher_relation = compute_relation_matrix(teacher_feature.detach())
        student_relation = compute_relation_matrix(student_feature)
        relation_loss = F.mse_loss(student_relation, teacher_relation)

        total_loss = (
            self.hard_weight * hard_loss
            + self.soft_weight * soft_loss
            + self.feature_weight * feature_loss
            + self.relation_weight * relation_loss
        )

        return total_loss, {
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "feature_loss": feature_loss.item(),
            "relation_loss": relation_loss.item(),
            "visible_horizon": int(curriculum_map.sum().item()),
            "mean_node_weight": reliability_items["node_weight"].mean().item(),
            "mean_horizon_weight": reliability_items["horizon_weight"].mean().item(),
            "reliability_enabled": int(self.enable_reliability),
            "curriculum_enabled": int(self.enable_curriculum),
        }
