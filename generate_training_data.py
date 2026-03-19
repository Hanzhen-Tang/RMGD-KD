from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def generate_graph_seq2seq_io_data(
    df,
    x_offsets,
    y_offsets,
    add_time_in_day=True,
    add_day_in_week=False,
):
    """
    根据原始时序表生成监督学习样本。

    返回：
    x: [样本数, 输入长度, 节点数, 输入特征数]
    y: [样本数, 预测长度, 节点数, 输出特征数]
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = np.tile(df.index.dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)

    x_offsets = np.sort(np.arange(-(args.seq_length_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(args.y_start, args.seq_length_y + 1, 1))

    x, y = generate_graph_seq2seq_io_data(
        df=df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=not args.disable_time_in_day,
        add_day_in_week=args.add_day_in_week,
    )

    print(f"总样本维度: x={x.shape}, y={y.shape}")

    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_ratio)
    num_val = round(num_samples * args.val_ratio)
    num_test = num_samples - num_train - num_val

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_x = locals()[f"x_{split_name}"]
        split_y = locals()[f"y_{split_name}"]
        print(f"{split_name}: x={split_x.shape}, y={split_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{split_name}.npz"),
            x=split_x,
            y=split_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def build_parser():
    parser = argparse.ArgumentParser(description="生成 METR-LA / PEMS-BAY 的训练样本。")
    parser.add_argument("--traffic_df_filename", type=str, required=True, help="原始 h5 文件路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录，例如 data/METR-LA。")
    parser.add_argument("--seq_length_x", type=int, default=12, help="输入长度。")
    parser.add_argument("--seq_length_y", type=int, default=12, help="预测长度。")
    parser.add_argument("--y_start", type=int, default=1, help="预测从第几个未来步开始。")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例。")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例。")
    parser.add_argument("--add_day_in_week", action="store_true", help="是否加入星期特征。")
    parser.add_argument("--disable_time_in_day", action="store_true", help="是否禁用时间特征。")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    generate_train_val_test(args)
