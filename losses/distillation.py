import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import masked_mae


def _normalize_weight_map(weight_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """把权重图归一化到均值约为 1，避免整体损失量级漂移。"""
    return weight_map / (weight_map.mean().detach() + eps)


def compute_reliability_map(teacher_pred: torch.Tensor, real_value: torch.Tensor) -> dict:
    """
    根据教师预测误差构造可靠性权重。
    输入张量维度统一为:
    teacher_pred / real_value: [B, 1, N, H]
    """
    teacher_error = torch.abs(teacher_pred.detach() - real_value.detach())

    # 节点可靠性：节点误差越小，说明教师在该节点越可信。
    node_error = teacher_error.mean(dim=(0, 1, 3), keepdim=True)  # [1, 1, N, 1]
    node_weight = torch.exp(-node_error)
    node_weight = _normalize_weight_map(node_weight)

    # 预测步可靠性：某 horizon 误差越小，说明该未来步知识更稳定。
    horizon_error = teacher_error.mean(dim=(0, 1, 2), keepdim=True)  # [1, 1, 1, H]
    horizon_weight = torch.exp(-horizon_error)
    horizon_weight = _normalize_weight_map(horizon_weight)

    reliability_map = _normalize_weight_map(node_weight * horizon_weight)
    return {
        "node_weight": node_weight,
        "horizon_weight": horizon_weight,
        "reliability_map": reliability_map,
    }


def compute_curriculum_map(horizon_count: int, current_epoch: int, total_epochs: int, device) -> torch.Tensor:
    """
    多步课程蒸馏:
    - 前 1/3 训练: 蒸馏前 1/3 horizon
    - 中间 1/3: 蒸馏前 2/3 horizon
    - 后 1/3: 蒸馏全部 horizon
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
    由节点隐藏表示构造节点关系矩阵。
    node_feature: [B, C, N, 1] 或 [B, C, N, T]
    返回: [B, N, N]
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
    """面向交通预测的多粒度蒸馏损失。"""

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
        # 1. 硬标签监督，保证学生直接学习真实交通状态。
        hard_loss = masked_mae(student_pred, real_value, null_val)

        # 2. 可靠性加权 + 课程蒸馏，避免学生过度学习教师噪声。
        reliability_items = compute_reliability_map(teacher_pred, real_value)
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

        weighted_map = _normalize_weight_map(reliability_map * (curriculum_map + 1e-6))

        soft_student = student_pred / self.temperature
        soft_teacher = teacher_pred.detach() / self.temperature
        soft_element = F.smooth_l1_loss(soft_student, soft_teacher, reduction="none")
        soft_loss = (soft_element * weighted_map).mean() * (self.temperature ** 2)

        # 3. 特征蒸馏，让学生学习教师更高层的时空语义。
        feature_loss = F.mse_loss(student_feature, teacher_feature.detach())

        # 4. 图关系蒸馏，让学生学习教师的空间拓扑关系知识。
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
