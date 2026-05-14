import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityMLPBlock(nn.Module):
    """Compact residual MLP block for the STID-style student."""

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.proj_in = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.proj_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.proj_in(x))
        x = self.dropout(self.proj_out(x))
        return F.relu(self.norm(x + residual))


class SimpleSTIDStudent(nn.Module):
    """
    Lightweight STID-style MLP student.

    It encodes recent observations with shared temporal-position embeddings and
    adds a node identity embedding before compact residual MLP blocks.
    """

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        hidden_dim=32,
        out_dim=12,
        dropout=0.3,
        mlp_layers=2,
        input_seq_len=12,
    ):
        super().__init__()
        self.feature_dim = hidden_dim
        self.input_seq_len = input_seq_len
        self.num_nodes = num_nodes

        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.position_embedding = nn.Parameter(torch.empty(1, hidden_dim, 1, input_seq_len))
        self.node_embedding = nn.Parameter(torch.empty(1, hidden_dim, num_nodes, 1))
        self.temporal_readout = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, input_seq_len))
        self.mlp_layers = nn.ModuleList(
            [IdentityMLPBlock(hidden_dim, dropout=dropout) for _ in range(max(mlp_layers, 1))]
        )
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))

        nn.init.xavier_uniform_(self.position_embedding)
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(self, inputs, supports=None, return_features: bool = False):
        assert inputs.ndim == 4, f"Expected [B, C, N, T] input, got {inputs.shape}"
        assert inputs.size(2) == self.num_nodes, f"Expected {self.num_nodes} nodes, got {inputs.size(2)}"
        assert inputs.size(3) == self.input_seq_len, (
            f"Expected input length {self.input_seq_len}, got {inputs.size(3)}"
        )

        x = self.input_proj(inputs)
        x = F.relu(x + self.position_embedding)
        x = F.relu(self.temporal_readout(x))
        x = x + self.node_embedding

        for layer in self.mlp_layers:
            x = layer(x)

        hidden_state = self.dropout(x)
        prediction = self.forecast_head(hidden_state)

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
        }

