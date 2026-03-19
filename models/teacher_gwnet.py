import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConv


class GWNetTeacher(nn.Module):
    """教师模型：基本保留原始 Graph WaveNet 结构。"""

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.supports = supports
        self.feature_dim = end_channels

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        receptive_field = 1
        self.supports_len = len(supports) if supports is not None else 0

        if gcn_bool and addaptadj:
            if self.supports is None:
                self.supports = []
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10, device=device))
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes, device=device))
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1)
                self.nodevec2 = nn.Parameter(initemb2)
            self.supports_len += 1
        else:
            self.nodevec1 = None
            self.nodevec2 = None

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                if self.gcn_bool:
                    self.gconv.append(GraphConv(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def _build_supports(self):
        adaptive_adj = None
        if self.gcn_bool and self.addaptadj and self.supports is not None and self.nodevec1 is not None:
            adaptive_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            return self.supports + [adaptive_adj], adaptive_adj
        return self.supports, adaptive_adj

    def forward(self, inputs, return_features: bool = False):
        assert inputs.ndim == 4, f"教师模型输入维度错误，当前为 {inputs.shape}"
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = F.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs

        x = self.start_conv(x)
        skip = 0
        supports, adaptive_adj = self._build_supports()

        for layer_idx in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[layer_idx](residual))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](residual))
            x = filter_out * gate_out

            skip_term = self.skip_convs[layer_idx](x)
            if isinstance(skip, int):
                skip = skip_term
            else:
                skip = skip[:, :, :, -skip_term.size(3):] + skip_term

            if self.gcn_bool and supports is not None:
                x = self.gconv[layer_idx](x, supports)
            else:
                x = self.residual_convs[layer_idx](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[layer_idx](x)

        skip = F.relu(skip)
        hidden_state = F.relu(self.end_conv_1(skip))
        prediction = self.end_conv_2(hidden_state)

        if not return_features:
            return prediction

        return {
            "prediction": prediction,
            "hidden_state": hidden_state,
            "adaptive_adj": adaptive_adj,
        }

