import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConv


class SimpleGCNStudent(nn.Module):
    """学生模型：用更浅的时空 GCN 近似教师模型。"""

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        hidden_dim=32,
        out_dim=12,
        dropout=0.3,
        support_len=2,
        gcn_order=2,
        graph_layers=2,
        input_seq_len=12,
    ):
        super().__init__()
        self.feature_dim = hidden_dim
        self.input_seq_len = input_seq_len
        self.graph_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.residual_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))

        for _ in range(graph_layers):
            self.graph_layers.append(
                GraphConv(hidden_dim, hidden_dim, dropout=dropout, support_len=support_len, order=gcn_order)
            )
            self.norm_layers.append(nn.BatchNorm2d(hidden_dim))

        # 这里沿时间维做一次压缩，输出 [B, hidden_dim, N, 1]
        self.temporal_readout = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, input_seq_len))
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))

        # num_nodes 保留在构造函数中，便于后续扩展到节点嵌入
        self.num_nodes = num_nodes

    def forward(self, inputs, supports, return_features: bool = False):
        assert inputs.ndim == 4, f"学生模型输入维度错误，当前为 {inputs.shape}"
        assert inputs.size(3) == self.input_seq_len, (
            f"学生模型时间长度需为 {self.input_seq_len}，当前为 {inputs.size(3)}"
        )

        x = self.input_proj(inputs)
        x = F.relu(self.temporal_conv(x))
        x = self.residual_proj(x)

        for graph_layer, norm_layer in zip(self.graph_layers, self.norm_layers):
            residual = x
            x = graph_layer(x, supports)
            x = F.relu(x + residual)
            x = norm_layer(x)

        hidden_state = F.relu(self.temporal_readout(x))
        hidden_state = self.dropout(hidden_state)
        prediction = self.forecast_head(hidden_state)

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
        }

