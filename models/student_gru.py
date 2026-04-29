import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGRUStudent(nn.Module):
    """Lightweight recurrent student for v6 generalization experiments."""

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        hidden_dim=32,
        out_dim=12,
        dropout=0.3,
        recurrent_layers=2,
        input_seq_len=12,
    ):
        super().__init__()
        self.feature_dim = hidden_dim
        self.input_seq_len = input_seq_len
        self.num_nodes = num_nodes

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))

    def forward(self, inputs, supports=None, return_features: bool = False):
        assert inputs.ndim == 4, f"Expected [B, C, N, T] input, got {inputs.shape}"
        assert inputs.size(3) == self.input_seq_len, (
            f"Expected input length {self.input_seq_len}, got {inputs.size(3)}"
        )

        batch_size, _, num_nodes, seq_len = inputs.shape
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size * num_nodes, seq_len, -1)
        x = F.relu(self.input_proj(x))

        _, hidden = self.gru(x)
        hidden = hidden[-1]
        hidden = self.norm(hidden)
        hidden = hidden.view(batch_size, num_nodes, self.feature_dim)
        hidden_state = hidden.permute(0, 2, 1).unsqueeze(-1).contiguous()
        hidden_state = self.dropout(hidden_state)
        prediction = self.forecast_head(hidden_state)

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
        }

