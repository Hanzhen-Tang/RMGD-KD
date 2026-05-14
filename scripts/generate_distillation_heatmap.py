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
from losses.distillation import compute_confidence_score, compute_curriculum_map
from model import GWNetTeacher

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "FangSong",
    "STSong",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate teacher-error or confidence heatmaps for distillation analysis."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--teacher_checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["none", "teacher_error", "confidence", "both"],
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
        "--orientation",
        type=str,
        default="normal",
        choices=["normal", "transpose"],
        help=(
            "Figure layout. normal shows nodes on the y-axis and horizons on the x-axis; "
            "transpose shows horizons on the y-axis and nodes on the x-axis."
        ),
    )
    parser.add_argument(
        "--plot_curriculum",
        action="store_true",
        help="Also plot the curriculum weights across training epochs and horizons.",
    )
    parser.add_argument(
        "--curriculum_mode",
        type=str,
        default="soft",
        choices=["standard", "short", "wide", "soft"],
        help="Curriculum mode to visualize when --plot_curriculum is used.",
    )
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=50,
        help="Total epoch count used for curriculum visualization.",
    )
    parser.add_argument(
        "--curriculum_display_epochs",
        type=int,
        default=0,
        help=(
            "Number of leading epochs to display in the curriculum figure. "
            "Use 0 to auto-trim the unchanged plateau."
        ),
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
        writer.writerow(["节点编号"] + horizon_labels)
        for node_idx, row in zip(node_indices, matrix):
            writer.writerow([int(node_idx)] + [float(value) for value in row])


def plot_node_horizon_heatmap(
    matrix: np.ndarray,
    node_indices: np.ndarray,
    save_path: str,
    title: str,
    cmap: str,
    orientation: str,
    colorbar_label: str,
):
    ensure_dir(os.path.dirname(save_path))

    plot_matrix = matrix
    if orientation == "transpose":
        plot_matrix = matrix.T
        fig_width = max(10.5, 0.22 * matrix.shape[0] + 4.2)
        fig_height = max(4.8, 0.35 * matrix.shape[1] + 2.5)
    else:
        fig_width = max(7.5, 0.52 * matrix.shape[1] + 4.0)
        fig_height = max(6.0, 0.18 * matrix.shape[0] + 2.8)
    plt.figure(figsize=(fig_width, fig_height))

    finite_values = plot_matrix[np.isfinite(plot_matrix)]
    if finite_values.size > 0:
        vmin = np.percentile(finite_values, 5)
        vmax = np.percentile(finite_values, 95)
        if np.isclose(vmin, vmax):
            vmin = finite_values.min()
            vmax = finite_values.max()
    else:
        vmin, vmax = None, None

    im = plt.imshow(
        plot_matrix,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        origin="lower" if orientation == "transpose" else "upper",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04, label=colorbar_label)
    plt.title(title)
    if orientation == "transpose":
        plt.xlabel("节点编号")
        plt.ylabel("预测步长")
        x_tick_step = max(1, len(node_indices) // 12)
        x_positions = np.arange(0, len(node_indices), x_tick_step)
        plt.xticks(x_positions, labels=[str(int(node_indices[pos])) for pos in x_positions])
        plt.yticks(
            ticks=np.arange(matrix.shape[1]),
            labels=[f"H{i}" for i in range(1, matrix.shape[1] + 1)],
        )
    else:
        plt.xlabel("预测步长")
        plt.ylabel("节点编号")
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
    orientation: str,
    colorbar_label: str,
):
    matrix_view, node_indices = select_nodes(matrix, node_limit=node_limit, strategy=node_select)
    save_matrix_csv(matrix_view, node_indices, csv_path)
    plot_node_horizon_heatmap(
        matrix_view,
        node_indices,
        fig_path,
        title=title,
        cmap=cmap,
        orientation=orientation,
        colorbar_label=colorbar_label,
    )
    return matrix_view.shape[0]


def save_curriculum_csv(matrix: np.ndarray, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    horizon_labels = [f"H{i}" for i in range(1, matrix.shape[1] + 1)]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["训练轮次"] + horizon_labels)
        for epoch_idx, row in enumerate(matrix, start=1):
            writer.writerow([epoch_idx] + [float(value) for value in row])


def plot_curriculum_heatmap(
    mode: str,
    total_epochs: int,
    fig_path: str,
    csv_path: str,
    display_epochs: int,
):
    ensure_dir(os.path.dirname(fig_path))
    device = torch.device("cpu")
    horizon_count = 12
    total_epochs = max(1, total_epochs)
    rows = []
    for epoch_idx in range(total_epochs):
        weights, _ = compute_curriculum_map(
            horizon_count=horizon_count,
            current_epoch=epoch_idx,
            total_epochs=total_epochs,
            device=device,
            mode=mode,
        )
        rows.append(weights.view(-1).cpu().numpy())
    matrix = np.stack(rows, axis=0)
    save_curriculum_csv(matrix, csv_path)

    if display_epochs > 0:
        display_count = min(display_epochs, total_epochs)
    else:
        changing = np.where(np.any(np.abs(matrix - 1.0) > 1e-6, axis=1))[0]
        display_count = int(changing[-1] + 2) if changing.size else total_epochs
        display_count = min(max(display_count, 6), total_epochs)

    display_matrix = matrix[:display_count, :]

    fig_width = max(7.5, min(10.5, 0.32 * display_count + 3.5))
    finite_values = display_matrix[np.isfinite(display_matrix)]
    if finite_values.size:
        color_vmin = float(finite_values.min())
        color_vmax = float(finite_values.max())
        if np.isclose(color_vmin, color_vmax):
            color_vmin, color_vmax = 0.0, 1.0
    else:
        color_vmin, color_vmax = 0.0, 1.0

    plt.figure(figsize=(fig_width, 4.8))
    im = plt.imshow(
        display_matrix.T,
        cmap="YlGnBu",
        aspect="auto",
        vmin=color_vmin,
        vmax=color_vmax,
        origin="lower",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04, label="课程权重")
    plt.title(f"{mode} 模式下的课程权重随训练轮次变化")
    plt.xlabel("训练轮次")
    plt.ylabel("预测步长")
    x_step = max(1, display_count // 8)
    x_positions = np.arange(0, display_count, x_step)
    plt.xticks(x_positions, labels=[str(int(pos + 1)) for pos in x_positions])
    plt.yticks(
        ticks=np.arange(horizon_count),
        labels=[f"H{i}" for i in range(1, horizon_count + 1)],
    )
    if display_count < total_epochs:
        plt.axvline(display_count - 1.5, color="#4A5568", linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=260)
    plt.close()


def main():
    args = parse_args()
    if args.mode == "none" and not args.plot_curriculum:
        raise ValueError("Nothing to plot. Use --mode teacher_error/confidence/both or add --plot_curriculum.")
    if args.mode != "none" and not args.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint is required unless --mode none is used.")

    ensure_dir(args.fig_dir)
    ensure_dir(args.csv_dir)

    outputs = []
    if args.mode != "none":
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

        if args.mode in ("teacher_error", "both"):
            fig_path = os.path.join(args.fig_dir, f"{args.exp_name}_teacher_error_heatmap.png")
            csv_path = os.path.join(args.csv_dir, f"{args.exp_name}_teacher_error_heatmap.csv")
            shown_nodes = export_heatmap(
                matrix=aggregated["teacher_error"],
                node_limit=args.node_limit,
                node_select=args.node_select,
                title="教师预测误差热力图（节点-预测步长）",
                cmap="YlOrRd",
                fig_path=fig_path,
                csv_path=csv_path,
                orientation=args.orientation,
                colorbar_label="教师预测误差",
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
                title="可信度评分热力图（节点-预测步长）",
                cmap="Blues",
                fig_path=fig_path,
                csv_path=csv_path,
                orientation=args.orientation,
                colorbar_label="可信度得分",
            )
            outputs.append(f"confidence_figure={fig_path}")
            outputs.append(f"confidence_csv={csv_path}")
            outputs.append(f"confidence_nodes={shown_nodes}")

    if args.plot_curriculum:
        fig_path = os.path.join(
            args.fig_dir,
            f"{args.exp_name}_{args.curriculum_mode}_curriculum_heatmap.png",
        )
        csv_path = os.path.join(
            args.csv_dir,
            f"{args.exp_name}_{args.curriculum_mode}_curriculum_heatmap.csv",
        )
        plot_curriculum_heatmap(
            mode=args.curriculum_mode,
            total_epochs=args.total_epochs,
            fig_path=fig_path,
            csv_path=csv_path,
            display_epochs=args.curriculum_display_epochs,
        )
        outputs.append(f"curriculum_figure={fig_path}")
        outputs.append(f"curriculum_csv={csv_path}")

    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()
