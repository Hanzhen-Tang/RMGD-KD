import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import util
from engine import prepare_batch
from model import GWNetTeacher, SimpleGCNStudent

PAPER_REAL_COLOR = "#4D4D4D"
PAPER_TEACHER_COLOR = "#0072B2"
PAPER_STUDENT_COLOR = "#D55E00"
from utils.plotting import prepare_curve_for_plot


def parse_args():
    parser = argparse.ArgumentParser(description="在同一张图中对比教师与学生的预测效果。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_sensor", type=int, default=0)
    parser.add_argument("--plot_horizon", type=int, default=11)
    parser.add_argument("--show_zero_real", action="store_true", help="是否保留真实值中的 0 点。")
    parser.add_argument("--exp_name", type=str, default="teacher_student_compare")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_teacher(ckpt, device, supports):
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


def build_student(ckpt, device, supports):
    model = SimpleGCNStudent(
        num_nodes=ckpt["num_nodes"],
        in_dim=ckpt["in_dim"],
        hidden_dim=ckpt["student_hidden_dim"],
        out_dim=ckpt["seq_length"],
        dropout=ckpt["dropout"],
        support_len=len(supports),
        gcn_order=ckpt["student_order"],
        graph_layers=ckpt["student_layers"],
        input_seq_len=ckpt["input_seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def collect_predictions(model, model_type, dataloader, scaler, supports, device):
    outputs = []
    reals = []

    for x, y in dataloader["test_loader"].get_iterator():
        inputs, targets = prepare_batch(x, y, device)
        reals.append(targets.unsqueeze(1))

        with torch.no_grad():
            if model_type == "teacher":
                pred = model(torch.nn.functional.pad(inputs, (1, 0, 0, 0))).transpose(1, 3)
            else:
                pred = model(inputs, supports).transpose(1, 3)
        outputs.append(pred)

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(reals, dim=0)
    yhat = yhat[: realy.size(0), ...]
    yhat = scaler.inverse_transform(yhat)
    return yhat, realy


def plot_compare_curve(real_curve, teacher_curve, student_curve, save_path, title, filter_invalid=True):
    ensure_dir(os.path.dirname(save_path))
    x_index, real_curve, teacher_curve = prepare_curve_for_plot(
        real_values=real_curve,
        pred_values=teacher_curve,
        filter_invalid=filter_invalid,
    )
    if filter_invalid:
        valid_mask = np.abs(np.asarray(real_curve) - np.asarray(real_curve)) == 0  # 恒为 True，用于统一长度语义
        student_curve = np.asarray(student_curve, dtype=float)
        original_real = np.asarray(real_curve, dtype=float)
        # 前面 prepare_curve_for_plot 已经过滤掉无效点，因此这里直接按相同位置长度截取。
        student_curve = student_curve[: original_real.shape[0]]
    else:
        student_curve = np.asarray(student_curve, dtype=float)

    plt.figure(figsize=(12, 5))
    plt.plot(x_index, real_curve, label="Real", linewidth=2.5, color="#1f1f1f")
    plt.plot(x_index, teacher_curve, label="Teacher", linewidth=2.0, color="#d95f02")
    plt.plot(x_index, student_curve, label="Student", linewidth=2.0, color="#1b9e77")
    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Traffic Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    teacher_ckpt = util.load_checkpoint(args.teacher_checkpoint, map_location=device)
    student_ckpt = util.load_checkpoint(args.student_checkpoint, map_location=device)

    teacher_model = build_teacher(teacher_ckpt, device, supports)
    student_model = build_student(student_ckpt, device, supports)

    teacher_pred, realy = collect_predictions(teacher_model, "teacher", dataloader, scaler, supports, device)
    student_pred, _ = collect_predictions(student_model, "student", dataloader, scaler, supports, device)

    sensor_idx = min(args.plot_sensor, realy.size(2) - 1)
    horizon_idx = min(args.plot_horizon, realy.size(3) - 1)

    real_curve = realy[:, 0, sensor_idx, horizon_idx].cpu().numpy()
    teacher_curve = teacher_pred[:, 0, sensor_idx, horizon_idx].cpu().numpy()
    student_curve = student_pred[:, 0, sensor_idx, horizon_idx].cpu().numpy()

    if not args.show_zero_real:
        valid_mask = np.abs(real_curve) > 1e-6
        if valid_mask.sum() >= max(10, len(real_curve) // 20):
            x_index = np.arange(len(real_curve))[valid_mask]
            real_curve_plot = real_curve[valid_mask]
            teacher_curve_plot = teacher_curve[valid_mask]
            student_curve_plot = student_curve[valid_mask]
        else:
            x_index = np.arange(len(real_curve))
            real_curve_plot = real_curve
            teacher_curve_plot = teacher_curve
            student_curve_plot = student_curve
    else:
        x_index = np.arange(len(real_curve))
        real_curve_plot = real_curve
        teacher_curve_plot = teacher_curve
        student_curve_plot = student_curve

    ensure_dir("outputs/figures")
    ensure_dir("outputs/predictions")

    fig_path = os.path.join(
        "outputs",
        "figures",
        f"{args.exp_name}_sensor{sensor_idx}_h{horizon_idx + 1}.png",
    )

    plt.figure(figsize=(12, 5))
    plt.plot(x_index, real_curve_plot, label="Real", linewidth=2.5, color=PAPER_REAL_COLOR)
    plt.plot(x_index, teacher_curve_plot, label="Teacher", linewidth=2.0, color=PAPER_TEACHER_COLOR)
    plt.plot(x_index, student_curve_plot, label="Student", linewidth=2.0, color=PAPER_STUDENT_COLOR)
    plt.title(f"Teacher vs Student | sensor={sensor_idx}, horizon={horizon_idx + 1}")
    plt.xlabel("Time Index")
    plt.ylabel("Traffic Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    csv_path = os.path.join(
        "outputs",
        "predictions",
        f"{args.exp_name}_sensor{sensor_idx}_h{horizon_idx + 1}.csv",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_index", "real", "teacher_pred", "student_pred"])
        for idx, (real_v, teacher_v, student_v) in enumerate(zip(real_curve, teacher_curve, student_curve)):
            writer.writerow([idx, float(real_v), float(teacher_v), float(student_v)])

    teacher_mae = np.mean(np.abs(teacher_curve_plot - real_curve_plot))
    student_mae = np.mean(np.abs(student_curve_plot - real_curve_plot))

    print(f"teacher_curve_mae={teacher_mae:.4f}")
    print(f"student_curve_mae={student_mae:.4f}")
    print(f"comparison_figure={fig_path}")
    print(f"comparison_csv={csv_path}")


if __name__ == "__main__":
    main()
