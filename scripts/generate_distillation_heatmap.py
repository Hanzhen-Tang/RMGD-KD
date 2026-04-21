import argparse
import csv
import os
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import util
from engine import prepare_batch
from losses.distillation import compute_confidence_score
from model import GWNetTeacher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate teacher-error or confidence heatmaps for distillation analysis."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["teacher_error", "confidence", "both"],
        help="Which heatmap to generate.",
    )
    parser.add_argument(
        "--node_limit",
        type=int,
        default=48,
        help="Maximum number of nodes to display in the heatmap.",
    )
    parser.add_argument(
        "--node_select",
        type=str,
        default="top_error",
        choices=["top_error", "first"],
        help="How to choose nodes when node_limit is smaller than the total node count.",
    )
    parser.add_argument(
        "--confidence_power",
        type=float,
        default=1.0,
        help="Confidence power used in compute_confidence_score.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="distill_heatmap",
        help="Prefix for saved figures and CSV files.",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="outputs/figures",
        help="Directory to save heatmap figures.",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="outputs/reports",
        help="Directory to save exported heatmap matrices.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def build_teacher_model(args, ckpt, device, supports):
    teacher_supports = None if ckpt.get("aptonly", False) else supports
    model = GWNetTeacher(
        device=device,
        num_nodes=ckpt["num_nodes"],
        dropout=ckpt["dropout"],
        supports=teacher_supports,
        gcn_bool=ckpt["gcn_bool"],
        addaptadj=ckpt["addaptadj"],
        aptinit=None if ckpt["randomadj"] or teacher_supports is None else teacher_supports[0],
        in_dim=ckpt["in_dim"],
        out_dim=ckpt["seq_length"],
        residual_channels=ckpt["nhid"],
        dilation_channels=ckpt["nhid"],
        skip_channels=ckpt["nhid"] * 8,
        end_channels=ckpt["nhid"] * 16,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def aggregate_maps(
    model,
    dataloader,
    scaler,
    device,
    confidence_power: float,
) -> Dict[str, np.ndarray]:
    error_sum = None
    error_count = None
    confidence_sum = None
    confidence_count = None

    for x, y in dataloader["test_loader"].get_iterator():
        inputs, targets = prepare_batch(x, y, device)
        real = targets.unsqueeze(1)

        with torch.no_grad():
            outputs = model(F.pad(inputs, (1, 0, 0, 0)), return_features=True)
            teacher_pred = scaler.inverse_transform(outputs["prediction"].transpose(1, 3))

        valid_mask = (torch.abs(real) > 1e-6).float()
        teacher_error = torch.abs(teacher_pred - real)

        batch_error_sum = (teacher_error * valid_mask).sum(dim=(0, 1))
        batch_error_count = valid_mask.sum(dim=(0, 1))

        confidence_items = compute_confidence_score(
            teacher_pred=teacher_pred,
            real_value=real,
            null_val=0.0,
            confidence_power=confidence_power,
        )
        confidence_score = confidence_items["confidence_score"]
        batch_confidence_sum = confidence_score.sum(dim=(0, 1))
        batch_confidence_count = valid_mask.sum(dim=(0, 1))

        if error_sum is None:
            error_sum = batch_error_sum
            error_count = batch_error_count
            confidence_sum = batch_confidence_sum
            confidence_count = batch_confidence_count
        else:
            error_sum += batch_error_sum
            error_count += batch_error_count
            confidence_sum += batch_confidence_sum
            confidence_count += batch_confidence_count

    error_map = error_sum / error_count.clamp_min(1.0)
    confidence_map = confidence_sum / confidence_count.clamp_min(1.0)

    return {
        "teacher_error": error_map.cpu().numpy(),
        "confidence": confidence_map.cpu().numpy(),
    }


def select_nodes(matrix: np.ndarray, node_limit: int, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
    total_nodes = matrix.shape[0]
    if node_limit <= 0 or node_limit >= total_nodes:
        indices = np.arange(total_nodes)
        return matrix, indices

    if strategy == "first":
        indices = np.arange(node_limit)
    else:
        node_score = matrix.mean(axis=1)
        indices = np.argsort(node_score)[::-1][:node_limit]
        indices = np.sort(indices)

    return matrix[indices, :], indices


def save_matrix_csv(matrix: np.ndarray, node_indices: np.ndarray, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    horizon_labels = [f"H{i}" for i in range(1, matrix.shape[1] + 1)]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Node"] + horizon_labels)
        for node_idx, row in zip(node_indices, matrix):
            writer.writerow([int(node_idx)] + [float(value) for value in row])


def plot_node_horizon_heatmap(
    matrix: np.ndarray,
    node_indices: np.ndarray,
    save_path: str,
    title: str,
    cmap: str,
):
    ensure_dir(os.path.dirname(save_path))

    fig_width = max(7.5, 0.52 * matrix.shape[1] + 4.0)
    fig_height = max(6.0, 0.18 * matrix.shape[0] + 2.8)
    plt.figure(figsize=(fig_width, fig_height))

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size > 0:
        vmin = np.percentile(finite_values, 5)
        vmax = np.percentile(finite_values, 95)
        if np.isclose(vmin, vmax):
            vmin = finite_values.min()
            vmax = finite_values.max()
    else:
        vmin, vmax = None, None

    im = plt.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Node Index")
    plt.xticks(
        ticks=np.arange(matrix.shape[1]),
        labels=[f"H{i}" for i in range(1, matrix.shape[1] + 1)],
    )
    y_tick_step = max(1, len(node_indices) // 12)
    y_positions = np.arange(0, len(node_indices), y_tick_step)
    plt.yticks(y_positions, labels=[str(int(node_indices[pos])) for pos in y_positions])
    plt.tight_layout()
    plt.savefig(save_path, dpi=260)
    plt.close()


def export_heatmap(
    matrix: np.ndarray,
    node_limit: int,
    node_select: str,
    title: str,
    cmap: str,
    fig_path: str,
    csv_path: str,
):
    matrix_view, node_indices = select_nodes(matrix, node_limit=node_limit, strategy=node_select)
    save_matrix_csv(matrix_view, node_indices, csv_path)
    plot_node_horizon_heatmap(matrix_view, node_indices, fig_path, title=title, cmap=cmap)
    return matrix_view.shape[0]


def main():
    args = parse_args()
    ensure_dir(args.fig_dir)
    ensure_dir(args.csv_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    ckpt = util.load_checkpoint(args.teacher_checkpoint, map_location=device)
    model = build_teacher_model(args, ckpt, device, supports)

    aggregated = aggregate_maps(
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        device=device,
        confidence_power=args.confidence_power,
    )

    outputs = []
    if args.mode in ("teacher_error", "both"):
        fig_path = os.path.join(args.fig_dir, f"{args.exp_name}_teacher_error_heatmap.png")
        csv_path = os.path.join(args.csv_dir, f"{args.exp_name}_teacher_error_heatmap.csv")
        shown_nodes = export_heatmap(
            matrix=aggregated["teacher_error"],
            node_limit=args.node_limit,
            node_select=args.node_select,
            title="Teacher Prediction Error Across Nodes and Horizons",
            cmap="YlOrRd",
            fig_path=fig_path,
            csv_path=csv_path,
        )
        outputs.append(f"teacher_error_figure={fig_path}")
        outputs.append(f"teacher_error_csv={csv_path}")
        outputs.append(f"teacher_error_nodes={shown_nodes}")

    if args.mode in ("confidence", "both"):
        fig_path = os.path.join(args.fig_dir, f"{args.exp_name}_confidence_heatmap.png")
        csv_path = os.path.join(args.csv_dir, f"{args.exp_name}_confidence_heatmap.csv")
        shown_nodes = export_heatmap(
            matrix=aggregated["confidence"],
            node_limit=args.node_limit,
            node_select=args.node_select,
            title="Confidence Scores Across Nodes and Horizons",
            cmap="Blues",
            fig_path=fig_path,
            csv_path=csv_path,
        )
        outputs.append(f"confidence_figure={fig_path}")
        outputs.append(f"confidence_csv={csv_path}")
        outputs.append(f"confidence_nodes={shown_nodes}")

    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()
