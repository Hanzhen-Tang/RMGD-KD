import argparse
import copy
import os
import time

import numpy as np
import torch

import util
from engine import TeacherTrainer, prepare_batch
from model import GWNetTeacher
from utils.plotting import plot_training_curves, save_history


def parse_args():
    parser = argparse.ArgumentParser(description="训练 GWNet 教师模型。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--gcn_bool", action="store_true")
    parser.add_argument("--aptonly", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--randomadj", action="store_true")
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="checkpoints/teacher")
    parser.add_argument("--exp_name", type=str, default="metr_teacher")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    num_nodes = dataloader["x_train"].shape[2]
    in_dim = dataloader["x_train"].shape[3]
    seq_length = dataloader["y_train"].shape[1]

    if args.aptonly:
        supports = None
    adjinit = None if args.randomadj or supports is None else supports[0]

    model = GWNetTeacher(
        device=device,
        num_nodes=num_nodes,
        dropout=args.dropout,
        supports=supports,
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,
        aptinit=adjinit,
        in_dim=in_dim,
        out_dim=seq_length,
        residual_channels=args.nhid,
        dilation_channels=args.nhid,
        skip_channels=args.nhid * 8,
        end_channels=args.nhid * 16,
    ).to(device)

    trainer = TeacherTrainer(
        model=model,
        scaler=scaler,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ensure_dir(args.save_dir)
    history = {
        "train_loss": [],
        "train_mape": [],
        "train_rmse": [],
        "val_loss": [],
        "val_mape": [],
        "val_rmse": [],
    }
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    print(f"教师训练开始: num_nodes={num_nodes}, in_dim={in_dim}, horizon={seq_length}, device={device}")
    print(f"传感器数量: {len(sensor_ids)}, 映射大小: {len(sensor_id_to_ind)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        dataloader["train_loader"].shuffle()

        train_losses, train_mapes, train_rmses = [], [], []
        for batch_idx, (x, y) in enumerate(dataloader["train_loader"].get_iterator(), start=1):
            inputs, targets = prepare_batch(x, y, device)
            metrics = trainer.train_batch(inputs, targets)
            train_losses.append(metrics["loss"])
            train_mapes.append(metrics["mape"])
            train_rmses.append(metrics["rmse"])

            if batch_idx % args.print_every == 0 or batch_idx == 1:
                print(
                    f"[Teacher][Epoch {epoch:03d}][Iter {batch_idx:03d}] "
                    f"loss={metrics['loss']:.4f}, mape={metrics['mape']:.4f}, rmse={metrics['rmse']:.4f}"
                )

        val_losses, val_mapes, val_rmses = [], [], []
        for x, y in dataloader["val_loader"].get_iterator():
            inputs, targets = prepare_batch(x, y, device)
            metrics = trainer.eval_batch(inputs, targets)
            val_losses.append(metrics["loss"])
            val_mapes.append(metrics["mape"])
            val_rmses.append(metrics["rmse"])

        mean_train_loss = float(np.mean(train_losses))
        mean_train_mape = float(np.mean(train_mapes))
        mean_train_rmse = float(np.mean(train_rmses))
        mean_val_loss = float(np.mean(val_losses))
        mean_val_mape = float(np.mean(val_mapes))
        mean_val_rmse = float(np.mean(val_rmses))

        history["train_loss"].append(mean_train_loss)
        history["train_mape"].append(mean_train_mape)
        history["train_rmse"].append(mean_train_rmse)
        history["val_loss"].append(mean_val_loss)
        history["val_mape"].append(mean_val_mape)
        history["val_rmse"].append(mean_val_rmse)

        print(
            f"[Teacher][Epoch {epoch:03d}] "
            f"train_loss={mean_train_loss:.4f}, val_loss={mean_val_loss:.4f}, "
            f"train_mape={mean_train_mape:.4f}, val_mape={mean_val_mape:.4f}, "
            f"time={time.time() - epoch_start:.2f}s"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    checkpoint_path = os.path.join(args.save_dir, f"{args.exp_name}_best.pt")
    torch.save(
        {
            "model_type": "teacher",
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_nodes": num_nodes,
            "in_dim": in_dim,
            "seq_length": seq_length,
            "nhid": args.nhid,
            "dropout": args.dropout,
            "adjtype": args.adjtype,
            "gcn_bool": args.gcn_bool,
            "addaptadj": args.addaptadj,
            "aptonly": args.aptonly,
            "randomadj": args.randomadj,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
        },
        checkpoint_path,
    )

    report_path = os.path.join("outputs", "reports", f"{args.exp_name}_teacher_history.json")
    figure_path = os.path.join("outputs", "figures", f"{args.exp_name}_teacher_curve.png")
    save_history(history, report_path)
    plot_training_curves(history, figure_path)

    print(f"教师模型训练结束，最优 epoch={best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"模型已保存到: {checkpoint_path}")
    print(f"训练曲线已保存到: {figure_path}")


if __name__ == "__main__":
    main()
