import json
import os
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_REAL_COLOR = "#4D4D4D"
PAPER_PRED_COLOR = "#0072B2"


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_history(history: Dict, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def plot_training_curves(history: Dict, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train MAE")
    plt.plot(epochs, history["val_loss"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def prepare_curve_for_plot(
    real_values,
    pred_values,
    invalid_value: float = 0.0,
    eps: float = 1e-6,
    filter_invalid: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对交通预测曲线做可视化前清洗。

    公开交通数据里经常把缺失值记成 0。
    这些点在评估时通常会被 mask，但如果直接画图，
    会让真实曲线出现大量不自然的“掉到 0”的尖刺。
    """
    real_values = np.asarray(real_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)
    x_index = np.arange(real_values.shape[0])

    if not filter_invalid:
        return x_index, real_values, pred_values

    valid_mask = np.abs(real_values - invalid_value) > eps
    if valid_mask.sum() < max(10, real_values.shape[0] // 20):
        return x_index, real_values, pred_values

    return x_index[valid_mask], real_values[valid_mask], pred_values[valid_mask]


def plot_prediction_curve(
    real_values,
    pred_values,
    save_path: str,
    title: str,
    invalid_value: float = 0.0,
    filter_invalid: bool = True,
):
    ensure_dir(os.path.dirname(save_path))
    x_index, real_values, pred_values = prepare_curve_for_plot(
        real_values=real_values,
        pred_values=pred_values,
        invalid_value=invalid_value,
        filter_invalid=filter_invalid,
    )

    plt.figure(figsize=(12, 4.5))
    plt.plot(x_index, real_values, label="Real", linewidth=2.2, color=PAPER_REAL_COLOR)
    plt.plot(x_index, pred_values, label="Prediction", linewidth=2.0, color=PAPER_PRED_COLOR)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Speed / Flow")
    plt.legend(frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()


def prepare_heatmap_matrix(
    matrix,
    max_nodes: Optional[int] = 64,
    zero_diagonal: bool = False,
):
    matrix = np.asarray(matrix, dtype=float)
    if max_nodes is not None and matrix.ndim == 2:
        max_nodes = min(max_nodes, matrix.shape[0], matrix.shape[1])
        matrix = matrix[:max_nodes, :max_nodes]

    if zero_diagonal and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        matrix = matrix.copy()
        np.fill_diagonal(matrix, np.nan)

    return matrix


def plot_heatmap(
    matrix,
    save_path: str,
    title: str,
    cmap: str = "RdYlBu_r",
    max_nodes: Optional[int] = 64,
    zero_diagonal: bool = False,
    robust: bool = True,
):
    ensure_dir(os.path.dirname(save_path))
    matrix = prepare_heatmap_matrix(matrix, max_nodes=max_nodes, zero_diagonal=zero_diagonal)

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        vmin, vmax = None, None
    elif robust:
        vmin = np.percentile(finite_values, 5)
        vmax = np.percentile(finite_values, 95)
        if np.isclose(vmin, vmax):
            vmin, vmax = finite_values.min(), finite_values.max()
    else:
        vmin, vmax = finite_values.min(), finite_values.max()

    plt.figure(figsize=(7.8, 6.6))
    im = plt.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()
