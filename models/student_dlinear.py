import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDLinearStudent(nn.Module):
    """Very small decomposition-linear student for traffic forecasting."""

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        hidden_dim=32,
        out_dim=12,
        dropout=0.3,
        input_seq_len=12,
        moving_kernel=3,
    ):
        super().__init__()
        self.feature_dim = hidden_dim
        self.input_seq_len = input_seq_len
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.moving_kernel = max(int(moving_kernel), 1)

        self.seasonal_linear = nn.Linear(input_seq_len, out_dim)
        self.trend_linear = nn.Linear(input_seq_len, out_dim)
        self.output_proj = nn.Linear(in_dim, 1)
        self.feature_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, supports=None, return_features: bool = False):
        assert inputs.ndim == 4, f"Expected [B, C, N, T] input, got {inputs.shape}"
        assert inputs.size(2) == self.num_nodes, f"Expected {self.num_nodes} nodes, got {inputs.size(2)}"
        assert inputs.size(3) == self.input_seq_len, (
            f"Expected input length {self.input_seq_len}, got {inputs.size(3)}"
        )

        batch_size, _, num_nodes, _ = inputs.shape
        x = inputs.permute(0, 2, 1, 3).contiguous()
        trend = self._moving_average(x)
        seasonal = x - trend

        forecast = self.seasonal_linear(seasonal) + self.trend_linear(trend)
        forecast = forecast.permute(0, 1, 3, 2).contiguous()
        prediction = self.output_proj(forecast).permute(0, 2, 1, 3).contiguous()

        hidden = self.feature_proj(forecast)
        hidden = self.norm(hidden)
        hidden = hidden.mean(dim=2)
        hidden_state = hidden.permute(0, 2, 1).unsqueeze(-1).contiguous()
        hidden_state = self.dropout(hidden_state)

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
        }

    def _moving_average(self, x):
        if self.moving_kernel <= 1:
            return x

        pad_left = (self.moving_kernel - 1) // 2
        pad_right = self.moving_kernel - 1 - pad_left
        batch_size, num_nodes, channels, seq_len = x.shape
        flat = x.view(batch_size * num_nodes * channels, 1, seq_len)
        padded = F.pad(flat, (pad_left, pad_right), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.moving_kernel, stride=1)
        return trend.view(batch_size, num_nodes, channels, seq_len)

