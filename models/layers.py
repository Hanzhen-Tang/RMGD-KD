import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    """按节点维度做图卷积传播。"""

    def forward(self, x, adj):
        x = torch.einsum("bcnt,nm->bcmt", (x, adj))
        return x.contiguous()


class Linear1x1(nn.Module):
    """1x1 卷积，相当于逐节点逐时间步的通道映射。"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.proj(x)


class GraphConv(nn.Module):
    """复用 GWNet 中的多阶图卷积。"""

    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        expanded_in = (order * support_len + 1) * c_in
        self.mlp = Linear1x1(expanded_in, c_out)
        self.dropout = dropout

    def forward(self, x, supports):
        out = [x]
        for adj in supports:
            x1 = self.nconv(x, adj)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adj)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)

