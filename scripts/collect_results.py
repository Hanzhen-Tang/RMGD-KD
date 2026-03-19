import argparse
import csv
import os
import time

import numpy as np
import torch

import util
from engine import count_parameters, prepare_batch
from model import GWNetTeacher, SimpleGCNStudent


def parse_args():
    parser = argparse.ArgumentParser(description="汇总多个教师/学生模型的测试结果到 CSV 和 Markdown 表。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_csv", type=str, default="outputs/reports/result_summary.csv")
    parser.add_argument("--output_md", type=str, default="outputs/reports/result_summary.md")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="格式: name,model_type,checkpoint_path，例如 Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def build_model(model_type, ckpt, device, supports):
    if model_type == "teacher":
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


def evaluate_model(model, model_type, dataloader, scaler, supports, device):
    outputs = []
    reals = []
    latencies = []

    for x, y in dataloader["test_loader"].get_iterator():
        inputs, targets = prepare_batch(x, y, device)
        reals.append(targets.unsqueeze(1))

        with torch.no_grad():
            start = time.perf_counter()
            if model_type == "teacher":
                pred = model(torch.nn.functional.pad(inputs, (1, 0, 0, 0))).transpose(1, 3)
            else:
                pred = model(inputs, supports).transpose(1, 3)
            latencies.append((time.perf_counter() - start) * 1000.0)
        outputs.append(pred)

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
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

    return {
        "MAE": float(np.mean(amae)),
        "MAPE": float(np.mean(amape)),
        "RMSE": float(np.mean(armse)),
        "LatencyMS": float(np.mean(latencies)),
    }


def write_csv(rows, output_csv):
    ensure_dir(os.path.dirname(output_csv))
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_md):
    ensure_dir(os.path.dirname(output_md))
    if not rows:
        return

    headers = list(rows[0].keys())
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [str(row[h]) for h in headers]
            f.write("| " + " | ".join(values) + " |\n")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    rows = []
    for run_spec in args.run:
        name, model_type, checkpoint_path = [item.strip() for item in run_spec.split(",", 2)]
        ckpt = util.load_checkpoint(checkpoint_path, map_location=device)
        model = build_model(model_type, ckpt, device, supports)
        metrics = evaluate_model(model, model_type, dataloader, scaler, supports, device)
        params = count_parameters(model)
        compression_ratio = ckpt.get("compression_ratio", "")

        row = {
            "Name": name,
            "ModelType": model_type,
            "Checkpoint": checkpoint_path,
            "Params": params,
            "CompressionRatio": compression_ratio,
            "MAE": round(metrics["MAE"], 4),
            "MAPE": round(metrics["MAPE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "LatencyMS": round(metrics["LatencyMS"], 4),
        }
        rows.append(row)
        print(row)

    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"csv_saved={args.output_csv}")
    print(f"md_saved={args.output_md}")


if __name__ == "__main__":
    main()
