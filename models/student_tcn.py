import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """A compact residual temporal block used by the lightweight TCN student."""

    def __init__(self, channels, dropout=0.3, dilation=1):
        super().__init__()
        self.filter_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, dilation),
            dilation=(1, dilation),
        )
        self.gate_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, dilation),
            dilation=(1, dilation),
        )
        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        filtered = torch.tanh(self.filter_conv(x))
        gated = torch.sigmoid(self.gate_conv(x))
        out = filtered * gated

        # Keep the same temporal length as the residual branch.
        out = out[..., : residual.size(-1)]
        out = self.dropout(self.residual_proj(out))
        out = self.norm(out + residual)
        return F.relu(out)


class SimpleTCNStudent(nn.Module):
    """Lightweight temporal-convolution student for v6 generalization experiments."""

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        hidden_dim=32,
        out_dim=12,
        dropout=0.3,
        temporal_layers=2,
        input_seq_len=12,
    ):
        super().__init__()
        self.feature_dim = hidden_dim
        self.input_seq_len = input_seq_len
        self.num_nodes = num_nodes

        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.temporal_layers = nn.ModuleList(
            [
                TemporalConvBlock(
                    hidden_dim,
                    dropout=dropout,
                    dilation=2 ** layer_idx,
                )
                for layer_idx in range(temporal_layers)
            ]
        )
        self.temporal_readout = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, input_seq_len))
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))

    def forward(self, inputs, supports=None, return_features: bool = False):
        assert inputs.ndim == 4, f"Expected [B, C, N, T] input, got {inputs.shape}"
        assert inputs.size(3) == self.input_seq_len, (
            f"Expected input length {self.input_seq_len}, got {inputs.size(3)}"
        )

        x = F.relu(self.input_proj(inputs))
        for layer in self.temporal_layers:
            x = layer(x)

        hidden_state = F.relu(self.temporal_readout(x))
        hidden_state = self.dropout(hidden_state)
        prediction = self.forecast_head(hidden_state)

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
        }
