import argparse
import copy
import os
import time

import numpy as np
import torch

import util
from engine import DistillationTrainer, count_parameters, prepare_batch
from model import GWNetTeacher, SimpleGCNStudent
from utils.plotting import plot_training_curves, save_history


def parse_args():
    parser = argparse.ArgumentParser(description="训练带创新点的蒸馏学生模型。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--student_hidden_dim", type=int, default=32)
    parser.add_argument("--student_layers", type=int, default=2)
    parser.add_argument("--student_order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hard_weight", type=float, default=0.6)
    parser.add_argument("--soft_weight", type=float, default=0.2)
    parser.add_argument("--feature_weight", type=float, default=0.1)
    parser.add_argument("--relation_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--disable_reliability", action="store_true")
    parser.add_argument("--disable_curriculum", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints/student")
    parser.add_argument("--exp_name", type=str, default="metr_student_rmgd")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_teacher_from_checkpoint(ckpt, device, supports):
    teacher_supports = None if ckpt.get("aptonly", False) else supports
    teacher = GWNetTeacher(
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
    teacher.load_state_dict(ckpt["model_state_dict"])
    return teacher


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    teacher_ckpt = util.load_checkpoint(args.teacher_checkpoint, map_location=device)
    teacher = build_teacher_from_checkpoint(teacher_ckpt, device, supports)

    num_nodes = dataloader["x_train"].shape[2]
    in_dim = dataloader["x_train"].shape[3]
    seq_length = dataloader["y_train"].shape[1]
    input_seq_len = dataloader["x_train"].shape[1]

    student = SimpleGCNStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=args.student_hidden_dim,
        out_dim=seq_length,
        dropout=args.dropout,
        support_len=len(supports),
        gcn_order=args.student_order,
        graph_layers=args.student_layers,
        input_seq_len=input_seq_len,
    ).to(device)

    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        supports=supports,
        scaler=scaler,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hard_weight=args.hard_weight,
        soft_weight=args.soft_weight,
        feature_weight=args.feature_weight,
        relation_weight=args.relation_weight,
        temperature=args.temperature,
        enable_reliability=not args.disable_reliability,
        enable_curriculum=not args.disable_curriculum,
    )

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student) + count_parameters(trainer.feature_adapter)
    compression_ratio = teacher_params / max(student_params, 1)

    ensure_dir(args.save_dir)
    history = {
        "train_loss": [],
        "train_mae": [],
        "train_mape": [],
        "train_rmse": [],
        "val_loss": [],
        "val_mae": [],
        "val_mape": [],
        "val_rmse": [],
        "hard_loss": [],
        "soft_loss": [],
        "feature_loss": [],
        "relation_loss": [],
        "visible_horizon": [],
        "val_latency_ms": [],
    }
    best_val_mae = float("inf")
    best_val_loss_at_best_mae = float("inf")
    best_state = None
    best_epoch = -1

    print(
        f"蒸馏训练开始: num_nodes={num_nodes}, in_dim={in_dim}, horizon={seq_length}, "
        f"student_hidden={args.student_hidden_dim}, device={device}"
    )
    print(
        f"教师参数量={teacher_params:,}, 学生参数量={student_params:,}, "
        f"压缩率={compression_ratio:.2f}x"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        trainer.set_epoch(epoch, args.epochs)
        dataloader["train_loader"].shuffle()

        train_losses, train_maes, train_mapes, train_rmses = [], [], [], []
        hard_losses, soft_losses, feature_losses, relation_losses = [], [], [], []

        for batch_idx, (x, y) in enumerate(dataloader["train_loader"].get_iterator(), start=1):
            inputs, targets = prepare_batch(x, y, device)
            metrics = trainer.train_batch(inputs, targets)
            train_losses.append(metrics["loss"])
            train_maes.append(metrics["mae"])
            train_mapes.append(metrics["mape"])
            train_rmses.append(metrics["rmse"])
            hard_losses.append(metrics["hard_loss"])
            soft_losses.append(metrics["soft_loss"])
            feature_losses.append(metrics["feature_loss"])
            relation_losses.append(metrics["relation_loss"])

            if batch_idx % args.print_every == 0 or batch_idx == 1:
                print(
                    f"[RMGD-KD][Epoch {epoch:03d}][Iter {batch_idx:03d}] "
                    f"total={metrics['loss']:.4f}, hard={metrics['hard_loss']:.4f}, "
                    f"soft={metrics['soft_loss']:.4f}, feat={metrics['feature_loss']:.4f}, "
                    f"rel={metrics['relation_loss']:.4f}, visible_h={metrics['visible_horizon']}"
                )

        val_losses, val_maes, val_mapes, val_rmses, val_latencies = [], [], [], [], []
        last_visible_horizon = seq_length
        for x, y in dataloader["val_loader"].get_iterator():
            inputs, targets = prepare_batch(x, y, device)
            metrics = trainer.eval_batch(inputs, targets)
            val_losses.append(metrics["loss"])
            val_maes.append(metrics["mae"])
            val_mapes.append(metrics["mape"])
            val_rmses.append(metrics["rmse"])
            val_latencies.append(metrics["latency_ms"])
            last_visible_horizon = metrics["visible_horizon"]

        mean_train_loss = float(np.mean(train_losses))
        mean_train_mae = float(np.mean(train_maes))
        mean_train_mape = float(np.mean(train_mapes))
        mean_train_rmse = float(np.mean(train_rmses))
        mean_val_loss = float(np.mean(val_losses))
        mean_val_mae = float(np.mean(val_maes))
        mean_val_mape = float(np.mean(val_mapes))
        mean_val_rmse = float(np.mean(val_rmses))
        mean_val_latency = float(np.mean(val_latencies))

        history["train_loss"].append(mean_train_loss)
        history["train_mae"].append(mean_train_mae)
        history["train_mape"].append(mean_train_mape)
        history["train_rmse"].append(mean_train_rmse)
        history["val_loss"].append(mean_val_loss)
        history["val_mae"].append(mean_val_mae)
        history["val_mape"].append(mean_val_mape)
        history["val_rmse"].append(mean_val_rmse)
        history["hard_loss"].append(float(np.mean(hard_losses)))
        history["soft_loss"].append(float(np.mean(soft_losses)))
        history["feature_loss"].append(float(np.mean(feature_losses)))
        history["relation_loss"].append(float(np.mean(relation_losses)))
        history["visible_horizon"].append(int(last_visible_horizon))
        history["val_latency_ms"].append(mean_val_latency)

        print(
            f"[RMGD-KD][Epoch {epoch:03d}] "
            f"train_total={mean_train_loss:.4f}, train_mae={mean_train_mae:.4f}, "
            f"val_total={mean_val_loss:.4f}, val_mae={mean_val_mae:.4f}, "
            f"soft={history['soft_loss'][-1]:.4f}, feat={history['feature_loss'][-1]:.4f}, "
            f"rel={history['relation_loss'][-1]:.4f}, visible_h={last_visible_horizon}, "
            f"val_latency={mean_val_latency:.2f}ms, time={time.time() - epoch_start:.2f}s"
        )

        if mean_val_mae < best_val_mae:
            best_val_mae = mean_val_mae
            best_val_loss_at_best_mae = mean_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(student.state_dict())

    checkpoint_path = os.path.join(args.save_dir, f"{args.exp_name}_best.pt")
    torch.save(
        {
            "model_type": "student",
            "method_name": "RMGD-KD",
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "best_val_loss_at_best_mae": best_val_loss_at_best_mae,
            "num_nodes": num_nodes,
            "in_dim": in_dim,
            "seq_length": seq_length,
            "input_seq_len": input_seq_len,
            "student_hidden_dim": args.student_hidden_dim,
            "student_layers": args.student_layers,
            "student_order": args.student_order,
            "dropout": args.dropout,
            "adjtype": args.adjtype,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "hard_weight": args.hard_weight,
            "soft_weight": args.soft_weight,
            "feature_weight": args.feature_weight,
            "relation_weight": args.relation_weight,
            "temperature": args.temperature,
            "disable_reliability": args.disable_reliability,
            "disable_curriculum": args.disable_curriculum,
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": compression_ratio,
        },
        checkpoint_path,
    )

    report_path = os.path.join("outputs", "reports", f"{args.exp_name}_student_history.json")
    figure_path = os.path.join("outputs", "figures", f"{args.exp_name}_student_curve.png")
    save_history(history, report_path)
    plot_training_curves(history, figure_path)
    print(f"[RMGD-KD] best checkpoint selected by val_mae: epoch={best_epoch}, val_mae={best_val_mae:.4f}, val_total={best_val_loss_at_best_mae:.4f}")
    best_val_loss = best_val_loss_at_best_mae

    print(f"学生模型训练结束，最优 epoch={best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"模型已保存到: {checkpoint_path}")
    print(f"训练曲线已保存到: {figure_path}")


if __name__ == "__main__":
    main()
