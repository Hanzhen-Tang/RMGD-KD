import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSelfAttentionLayer(nn.Module):
    """Self-attention over one selected dimension of [B, T, N, C]."""

    def __init__(self, model_dim, feed_forward_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm_attn = nn.LayerNorm(model_dim)
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, dim):
        x = x.transpose(dim, -2)
        original_shape = x.shape
        length = original_shape[-2]

        flat_x = x.reshape(-1, length, self.model_dim)
        residual = flat_x
        query = self._to_attention_heads(self.q_proj(flat_x))
        key = self._to_attention_heads(self.k_proj(flat_x))
        value = self._to_attention_heads(self.v_proj(flat_x))

        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(-1, length, self.model_dim)
        attn = self.out_proj(attn)
        flat_x = self.norm_attn(residual + self.dropout_attn(attn))

        residual = flat_x
        flat_x = self.norm_ffn(residual + self.dropout_ffn(self.feed_forward(flat_x)))
        x = flat_x.reshape(original_shape)
        return x.transpose(dim, -2)

    def _to_attention_heads(self, x):
        batch_size, length, _ = x.shape
        x = x.view(batch_size, length, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()


class STAEformerTeacher(nn.Module):
    """
    STAEformer teacher adapted to this project's teacher interface.

    The architecture follows the STAEformer idea: input/time/adaptive
    embeddings, temporal attention, spatial attention, and mixed output
    projection for multi-horizon traffic forecasting.
    """

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        input_seq_len=12,
        out_dim=12,
        steps_per_day=288,
        input_embedding_dim=16,
        tod_embedding_dim=16,
        dow_embedding_dim=0,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=32,
        feed_forward_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.input_seq_len = input_seq_len
        self.out_dim = out_dim
        self.steps_per_day = steps_per_day
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim if in_dim >= 2 else 0
        self.dow_embedding_dim = dow_embedding_dim if in_dim >= 3 else 0
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.use_mixed_proj = use_mixed_proj

        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )
        self.feature_dim = self.model_dim

        self.input_proj = nn.Linear(in_dim, input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, self.tod_embedding_dim)
        else:
            self.tod_embedding = None
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        else:
            self.dow_embedding = None
        if self.spatial_embedding_dim > 0:
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_embedding)
        else:
            self.node_embedding = None
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(input_seq_len, num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None

        self.temporal_layers = nn.ModuleList(
            [
                FactorizedSelfAttentionLayer(
                    self.model_dim,
                    feed_forward_dim=feed_forward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.spatial_layers = nn.ModuleList(
            [
                FactorizedSelfAttentionLayer(
                    self.model_dim,
                    feed_forward_dim=feed_forward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        if use_mixed_proj:
            self.output_proj = nn.Linear(input_seq_len * self.model_dim, out_dim)
        else:
            self.temporal_proj = nn.Linear(input_seq_len, out_dim)
            self.output_proj = nn.Linear(self.model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, return_features: bool = False):
        assert inputs.ndim == 4, f"Expected [B, C, N, T] input, got {inputs.shape}"
        assert inputs.size(2) == self.num_nodes, f"Expected {self.num_nodes} nodes, got {inputs.size(2)}"

        if inputs.size(-1) < self.input_seq_len:
            inputs = F.pad(inputs, (self.input_seq_len - inputs.size(-1), 0, 0, 0))
        elif inputs.size(-1) > self.input_seq_len:
            inputs = inputs[..., -self.input_seq_len:]

        x_raw = inputs.permute(0, 3, 2, 1).contiguous()
        batch_size = x_raw.size(0)
        features = [self.input_proj(x_raw[..., : self.in_dim])]

        if self.tod_embedding is not None:
            tod_index = (x_raw[..., 1].clamp(0.0, 1.0 - 1e-6) * self.steps_per_day).long()
            features.append(self.tod_embedding(tod_index))

        if self.dow_embedding is not None:
            dow_index = x_raw[..., 2].clamp(0, 6).long()
            features.append(self.dow_embedding(dow_index))

        if self.node_embedding is not None:
            node_emb = self.node_embedding.view(1, 1, self.num_nodes, self.spatial_embedding_dim)
            features.append(node_emb.expand(batch_size, self.input_seq_len, -1, -1))

        if self.adaptive_embedding is not None:
            features.append(
                self.adaptive_embedding.view(1, self.input_seq_len, self.num_nodes, self.adaptive_embedding_dim)
                .expand(batch_size, -1, -1, -1)
            )

        x = torch.cat(features, dim=-1)
        x = self.dropout(x)

        for layer in self.temporal_layers:
            x = layer(x, dim=1)
        for layer in self.spatial_layers:
            x = layer(x, dim=2)

        hidden_state = x.mean(dim=1).permute(0, 2, 1).unsqueeze(-1).contiguous()

        if self.use_mixed_proj:
            out = x.transpose(1, 2).reshape(batch_size, self.num_nodes, self.input_seq_len * self.model_dim)
            prediction = self.output_proj(out).transpose(1, 2).unsqueeze(-1).contiguous()
        else:
            out = x.permute(0, 3, 2, 1).contiguous()
            out = self.temporal_proj(out).permute(0, 3, 2, 1).contiguous()
            prediction = self.output_proj(out).permute(0, 1, 2, 3).contiguous()

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
            "adaptive_adj": None,
        }

