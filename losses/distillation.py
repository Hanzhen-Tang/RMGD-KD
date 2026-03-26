import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import masked_mae


def _build_valid_mask(real_value: torch.Tensor, null_val: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    """构建有效位置掩码，避免无效值污染蒸馏。"""
    if torch.isnan(torch.tensor(null_val, device=real_value.device)):
        valid_mask = ~torch.isnan(real_value)
    else:
        valid_mask = torch.abs(real_value - null_val) > eps
    return valid_mask.float()


def _masked_reduce_mean(value: torch.Tensor, mask: torch.Tensor, dim, keepdim: bool = True) -> torch.Tensor:
    """安全地计算带掩码平均值。"""
    numerator = (value * mask).sum(dim=dim, keepdim=keepdim)
    denominator = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numerator / denominator


def _topk_binary_mask(score: torch.Tensor, keep_ratio: float, dim_size: int) -> torch.Tensor:
    """根据 top-k 规则生成二值筛选掩码。"""
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    keep_count = max(1, int(math.ceil(dim_size * keep_ratio)))
    flat_score = score.reshape(-1)
    topk_indices = torch.topk(flat_score, k=keep_count, largest=True).indices
    flat_mask = torch.zeros_like(flat_score)
    flat_mask[topk_indices] = 1.0
    return flat_mask.view_as(score)


def compute_confidence_filter(
    teacher_pred: torch.Tensor,
    real_value: torch.Tensor,
    keep_ratio: float = 0.7,
    null_val: float = 0.0,
):
    """
    可信度筛选蒸馏：
    - 教师误差越小，说明该位置知识越可信
    - 仅保留 top-k 的节点与 horizon 参与软蒸馏

    teacher_pred / real_value: [B, 1, N, H]
    """
    valid_mask = _build_valid_mask(real_value.detach(), null_val=null_val)
    teacher_error = torch.abs(teacher_pred.detach() - real_value.detach())

    node_error = _masked_reduce_mean(teacher_error, valid_mask, dim=(0, 1, 3), keepdim=True)  # [1,1,N,1]
    horizon_error = _masked_reduce_mean(teacher_error, valid_mask, dim=(0, 1, 2), keepdim=True)  # [1,1,1,H]

    node_confidence = -node_error
    horizon_confidence = -horizon_error

    node_mask = _topk_binary_mask(node_confidence, keep_ratio=keep_ratio, dim_size=node_confidence.size(2))
    horizon_mask = _topk_binary_mask(horizon_confidence, keep_ratio=keep_ratio, dim_size=horizon_confidence.size(3))

    confidence_mask = node_mask * horizon_mask * valid_mask
    kept_ratio = confidence_mask.sum().item() / valid_mask.sum().clamp_min(1.0).item()

    return {
        "node_mask": node_mask,
        "horizon_mask": horizon_mask,
        "confidence_mask": confidence_mask,
        "kept_ratio": kept_ratio,
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
    """保留关系矩阵接口，兼容旧代码路径。"""
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
    v3 方法默认目标：
    - hard supervision
    - confidence-filtered soft distillation
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
        confidence_keep_ratio=0.7,
    ):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.feature_weight = feature_weight
        self.relation_weight = relation_weight
        self.temperature = temperature
        self.enable_confidence_filter = enable_confidence_filter
        self.enable_curriculum = enable_curriculum
        self.confidence_keep_ratio = confidence_keep_ratio

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

        confidence_items = compute_confidence_filter(
            teacher_pred=teacher_pred,
            real_value=real_value,
            keep_ratio=self.confidence_keep_ratio,
            null_val=null_val,
        )
        if self.enable_confidence_filter:
            confidence_mask = confidence_items["confidence_mask"]
        else:
            confidence_mask = valid_mask

        if self.enable_curriculum:
            curriculum_map = compute_curriculum_map(
                horizon_count=student_pred.size(-1),
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                device=student_pred.device,
            )
        else:
            curriculum_map = torch.ones((1, 1, 1, student_pred.size(-1)), device=student_pred.device)

        soft_mask = confidence_mask * curriculum_map
        if soft_mask.sum().detach().item() <= 0:
            soft_mask = valid_mask

        soft_student = student_pred / self.temperature
        soft_teacher = teacher_pred.detach() / self.temperature
        soft_element = F.smooth_l1_loss(soft_student, soft_teacher, reduction="none")
        soft_loss = (soft_element * soft_mask).sum() / soft_mask.sum().clamp_min(1.0)
        soft_loss = soft_loss * (self.temperature ** 2)

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
            + self.soft_weight * soft_loss
            + self.feature_weight * feature_loss
            + self.relation_weight * relation_loss
        )

        visible_horizon = int(curriculum_map.sum().item())
        kept_node_ratio = confidence_items["node_mask"].mean().item()
        kept_horizon_ratio = confidence_items["horizon_mask"].mean().item()

        return total_loss, {
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "feature_loss": feature_loss.item(),
            "relation_loss": relation_loss.item(),
            "visible_horizon": visible_horizon,
            "mean_node_weight": kept_node_ratio,
            "mean_horizon_weight": kept_horizon_ratio,
            "confidence_keep_ratio": confidence_items["kept_ratio"],
            "confidence_enabled": int(self.enable_confidence_filter),
            "curriculum_enabled": int(self.enable_curriculum),
        }
