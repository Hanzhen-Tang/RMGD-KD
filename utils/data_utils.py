import os
from typing import Dict, Optional

import numpy as np


class DataLoader:
    """简单的数据迭代器，保持与原始 GWNet 代码一致。"""

    def __init__(self, xs, ys, batch_size, pad_with_last_sample: bool = True):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """对流量值做标准化，仅对第 0 个通道（速度值）生效。"""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std if std > 0 else 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(
    dataset_dir: str,
    batch_size: int,
    valid_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """读取 train/val/test 的 npz 文件，并构建数据迭代器。"""

    valid_batch_size = valid_batch_size or batch_size
    test_batch_size = test_batch_size or batch_size

    data = {}
    for category in ["train", "val", "test"]:
        category_path = os.path.join(dataset_dir, f"{category}.npz")
        category_data = np.load(category_path)
        data[f"x_{category}"] = category_data["x"]
        data[f"y_{category}"] = category_data["y"]

    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(),
        std=data["x_train"][..., 0].std(),
    )

    for category in ["train", "val", "test"]:
        data[f"x_{category}"][..., 0] = scaler.transform(data[f"x_{category}"][..., 0])

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
    data["scaler"] = scaler
    return data

