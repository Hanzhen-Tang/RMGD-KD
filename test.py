import argparse
import csv
import os
import time

import numpy as np
import torch

import util
from engine import count_parameters, prepare_batch
from losses.distillation import compute_relation_matrix
from model import GWNetTeacher, SimpleGCNStudent
from utils.plotting import plot_heatmap, plot_prediction_curve


def parse_args():
    parser = argparse.ArgumentParser(description="测试教师或学生模型。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument("--plot_sensor", type=int, default=0, help="要可视化的传感器编号。")
    parser.add_argument("--plot_horizon", type=int, default=11, help="要可视化的 horizon，11 表示第 12 步。")
    parser.add_argument("--plot_adaptive_adj", action="store_true", help="是否绘制教师自适应邻接矩阵。")
    parser.add_argument("--plot_relation", action="store_true", help="是否绘制节点关系热力图。")
    parser.add_argument("--show_zero_real", action="store_true", help="是否保留真实值中的 0 点。默认会隐藏缺失 0。")
    parser.add_argument("--heatmap_max_nodes", type=int, default=64, help="热力图最多显示多少个节点。")
    parser.add_argument("--exp_name", type=str, default="eval")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_model(args, ckpt, device, supports):
    if args.model_type == "teacher":
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
    else:
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    ckpt = util.load_checkpoint(args.checkpoint, map_location=device)
    model = build_model(args, ckpt, device, supports)
    model_params = count_parameters(model)

    outputs = []
    reals = []
    adaptive_adj = None
    last_relation = None
    latencies = []

    for x, y in dataloader["test_loader"].get_iterator():
        inputs, targets = prepare_batch(x, y, device)
        reals.append(targets.unsqueeze(1))

        with torch.no_grad():
            start_time = time.perf_counter()
            if args.model_type == "teacher":
                model_outputs = model(torch.nn.functional.pad(inputs, (1, 0, 0, 0)), return_features=True)
                pred = model_outputs["prediction"].transpose(1, 3)
                adaptive_adj = model_outputs["adaptive_adj"]
            else:
                model_outputs = model(inputs, supports, return_features=True)
                pred = model_outputs["prediction"].transpose(1, 3)
            latencies.append((time.perf_counter() - start_time) * 1000.0)

        outputs.append(pred)
        last_relation = compute_relation_matrix(model_outputs["hidden_state"]).mean(dim=0).cpu().numpy()

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(reals, dim=0)
    yhat = yhat[: realy.size(0), ...]

    yhat_denorm = scaler.inverse_transform(yhat)
    realy_denorm = realy

    amae, amape, armse = [], [], []
    num_horizon = yhat_denorm.size(-1)
    for horizon_idx in range(num_horizon):
        pred = yhat_denorm[:, :, :, horizon_idx]
        real = realy_denorm[:, :, :, horizon_idx]
        mae, mape, rmse = util.metric(pred, real)
        print(
            f"[{args.model_type}] horizon={horizon_idx + 1:02d}, "
            f"MAE={mae:.4f}, MAPE={mape:.4f}, RMSE={rmse:.4f}"
        )
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

    print(
        f"[{args.model_type}] average -> "
        f"MAE={np.mean(amae):.4f}, MAPE={np.mean(amape):.4f}, RMSE={np.mean(armse):.4f}, "
        f"params={model_params:,}, latency={np.mean(latencies):.2f}ms/batch"
    )

    ensure_dir("outputs/predictions")
    ensure_dir("outputs/figures")

    sensor_idx = min(args.plot_sensor, yhat_denorm.size(2) - 1)
    horizon_idx = min(args.plot_horizon, yhat_denorm.size(3) - 1)
    real_curve = realy_denorm[:, 0, sensor_idx, horizon_idx].cpu().numpy()
    pred_curve = yhat_denorm[:, 0, sensor_idx, horizon_idx].cpu().numpy()

    figure_path = os.path.join(
        "outputs",
        "figures",
        f"{args.exp_name}_{args.model_type}_sensor{sensor_idx}_h{horizon_idx + 1}.png",
    )
    plot_prediction_curve(
        real_curve,
        pred_curve,
        figure_path,
        title=f"{args.model_type} prediction vs real | sensor={sensor_idx}, horizon={horizon_idx + 1}",
        filter_invalid=not args.show_zero_real,
    )

    csv_path = os.path.join("outputs", "predictions", f"{args.exp_name}_{args.model_type}_sensor{sensor_idx}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_index", "real", "pred"])
        for idx, (real_value, pred_value) in enumerate(zip(real_curve, pred_curve)):
            writer.writerow([idx, float(real_value), float(pred_value)])

    print(f"预测曲线已保存到: {figure_path}")
    print(f"预测数值已保存到: {csv_path}")

    if args.model_type == "teacher" and args.plot_adaptive_adj and adaptive_adj is not None:
        heatmap_path = os.path.join("outputs", "figures", f"{args.exp_name}_teacher_adaptive_adj.png")
        plot_heatmap(
            adaptive_adj.cpu().numpy(),
            heatmap_path,
            "Teacher Adaptive Adjacency",
            cmap="YlOrRd",
            max_nodes=args.heatmap_max_nodes,
            zero_diagonal=False,
            robust=True,
        )
        print(f"教师自适应邻接矩阵已保存到: {heatmap_path}")

    if args.plot_relation and last_relation is not None:
        relation_path = os.path.join("outputs", "figures", f"{args.exp_name}_{args.model_type}_relation.png")
        plot_heatmap(
            last_relation,
            relation_path,
            f"{args.model_type} node relation",
            cmap="Spectral_r",
            max_nodes=args.heatmap_max_nodes,
            zero_diagonal=True,
            robust=True,
        )
        print(f"节点关系热力图已保存到: {relation_path}")


if __name__ == "__main__":
    main()
