"""兼容原始 GWNet 写法的统一工具入口。"""

import torch

from utils.data_utils import DataLoader, StandardScaler, load_dataset
from utils.graph_utils import load_adj, load_pickle
from utils.metrics import masked_mae, masked_mape, masked_mse, masked_rmse, metric


def load_checkpoint(path, map_location=None):
    """
    兼容 PyTorch 2.6 及以上版本的 checkpoint 加载。

    本项目保存的 checkpoint 除了 model_state_dict，
    还包含 numpy 标量和训练配置，因此需要显式关闭
    weights_only 模式。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch。
        return torch.load(path, map_location=map_location)


__all__ = [
    "DataLoader",
    "StandardScaler",
    "load_dataset",
    "load_adj",
    "load_pickle",
    "masked_mae",
    "masked_mape",
    "masked_mse",
    "masked_rmse",
    "metric",
    "load_checkpoint",
]
