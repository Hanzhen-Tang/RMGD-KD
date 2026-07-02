from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def project_path(path: str | Path) -> str:
    path = resolve_path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_parent(path: str | Path) -> None:
    resolve_path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    resolve_path(path).mkdir(parents=True, exist_ok=True)


def float_token(value: float | str) -> str:
    text = f"{float(value):g}" if isinstance(value, (int, float)) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_").replace(":", "_")


def command_to_string(command: list[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def parse_loss_pair(text: str) -> tuple[float, float]:
    normalized = text.replace("/", ":").replace(",", ":")
    parts = [part.strip() for part in normalized.split(":") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"loss weight pair must look like 0.7/0.3, got: {text}")
    hard_weight, soft_weight = float(parts[0]), float(parts[1])
    return hard_weight, soft_weight


def add_student_training_args(parser) -> None:
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--teacher_checkpoint", type=str, default="checkpoints/teacher/metr_teacher_best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--student_hidden_dim", type=int, default=32)
    parser.add_argument("--student_layers", type=int, default=2)
    parser.add_argument("--student_order", type=int, default=2)
    parser.add_argument("--student_model", type=str, default="gcn", choices=["gcn", "tcn", "gru", "stid", "dlinear"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hard_weight", type=float, default=0.7)
    parser.add_argument("--soft_weight", type=float, default=0.3)
    parser.add_argument("--trend_weight", type=float, default=0.5)
    parser.add_argument("--feature_weight", type=float, default=0.0)
    parser.add_argument("--relation_weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument(
        "--curriculum_mode",
        type=str,
        default="dynamic_soft",
        choices=["standard", "short", "wide", "soft", "dynamic_soft"],
    )
    parser.add_argument("--dynamic_curriculum_alpha", type=float, default=0.45)
    parser.add_argument("--dynamic_curriculum_beta", type=float, default=0.35)
    parser.add_argument("--dynamic_curriculum_gamma", type=float, default=0.20)
    parser.add_argument("--dynamic_curriculum_eta", type=float, default=0.70)
    parser.add_argument("--dynamic_curriculum_ema", type=float, default=0.90)
    parser.add_argument("--dynamic_curriculum_warmup", type=int, default=5)
    parser.add_argument("--disable_confidence_filter", action="store_true")
    parser.add_argument("--disable_curriculum", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints/student")


def build_student_train_command(args, exp_name: str, seed: int, overrides: dict[str, Any] | None = None) -> list[str]:
    values = {
        "device": args.device,
        "data": args.data,
        "adjdata": args.adjdata,
        "adjtype": args.adjtype,
        "teacher_checkpoint": args.teacher_checkpoint,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "print_every": args.print_every,
        "student_hidden_dim": args.student_hidden_dim,
        "student_layers": args.student_layers,
        "student_order": args.student_order,
        "student_model": args.student_model,
        "dropout": args.dropout,
        "hard_weight": args.hard_weight,
        "soft_weight": args.soft_weight,
        "trend_weight": args.trend_weight,
        "feature_weight": args.feature_weight,
        "relation_weight": args.relation_weight,
        "temperature": args.temperature,
        "confidence_power": args.confidence_power,
        "curriculum_mode": args.curriculum_mode,
        "dynamic_curriculum_alpha": args.dynamic_curriculum_alpha,
        "dynamic_curriculum_beta": args.dynamic_curriculum_beta,
        "dynamic_curriculum_gamma": args.dynamic_curriculum_gamma,
        "dynamic_curriculum_eta": args.dynamic_curriculum_eta,
        "dynamic_curriculum_ema": args.dynamic_curriculum_ema,
        "dynamic_curriculum_warmup": args.dynamic_curriculum_warmup,
        "save_dir": args.save_dir,
        "exp_name": exp_name,
        "seed": seed,
    }
    if overrides:
        values.update(overrides)

    command = [args.python, "train_student_kd.py"]
    for key, value in values.items():
        command.extend([f"--{key}", str(value)])

    if getattr(args, "disable_confidence_filter", False):
        command.append("--disable_confidence_filter")
    if getattr(args, "disable_curriculum", False):
        command.append("--disable_curriculum")

    return command


def run_logged_command(command: list[str], log_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    command_text = command_to_string(command)
    if dry_run:
        print(f"[dry-run] {command_text}")
        return {"command": command_text, "returncode": 0, "elapsed_sec": 0.0}

    ensure_parent(log_path)
    start = time.time()
    print(f"[run] {command_text}")
    with resolve_path(log_path).open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_text}\n\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = process.wait()

    elapsed = time.time() - start
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return {"command": command_text, "returncode": returncode, "elapsed_sec": elapsed}


class CheckpointEvaluator:
    def __init__(self, device: str, data: str, adjdata: str, adjtype: str, batch_size: int):
        import torch

        import util
        from engine import count_parameters, prepare_batch

        self.torch = torch
        self.util = util
        self.count_parameters = count_parameters
        self.prepare_batch = prepare_batch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dataloader = util.load_dataset(str(resolve_path(data)), batch_size, batch_size, batch_size)
        self.scaler = self.dataloader["scaler"]
        _, _, adj_mx = util.load_adj(str(resolve_path(adjdata)), adjtype)
        self.supports = [torch.tensor(adj, dtype=torch.float32, device=self.device) for adj in adj_mx]

    def _build_model(self, ckpt: dict[str, Any], model_type: str):
        from model import build_student_from_checkpoint, build_teacher_from_checkpoint

        if model_type == "teacher":
            model = build_teacher_from_checkpoint(ckpt, self.supports, self.device)
        else:
            model = build_student_from_checkpoint(ckpt, self.supports, self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def evaluate(self, checkpoint_path: str | Path, model_type: str = "student") -> dict[str, Any]:
        torch = self.torch
        util = self.util
        ckpt = util.load_checkpoint(str(resolve_path(checkpoint_path)), map_location=self.device)
        model = self._build_model(ckpt, model_type)

        outputs = []
        reals = []
        latencies = []
        with torch.no_grad():
            for x, y in self.dataloader["test_loader"].get_iterator():
                inputs, targets = self.prepare_batch(x, y, self.device)
                reals.append(targets.unsqueeze(1))
                start = time.perf_counter()
                if model_type == "teacher":
                    pred = model(torch.nn.functional.pad(inputs, (1, 0, 0, 0))).transpose(1, 3)
                else:
                    pred = model(inputs, self.supports).transpose(1, 3)
                latencies.append((time.perf_counter() - start) * 1000.0)
                outputs.append(pred)

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(reals, dim=0)
        yhat = yhat[: realy.size(0), ...]
        yhat_denorm = self.scaler.inverse_transform(yhat)

        horizon_rows = []
        amae, amape, armse = [], [], []
        for horizon_idx in range(yhat_denorm.size(-1)):
            pred = yhat_denorm[:, :, :, horizon_idx]
            real = realy[:, :, :, horizon_idx]
            mae, mape, rmse = util.metric(pred, real)
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)
            horizon_rows.append(
                {
                    "Horizon": horizon_idx + 1,
                    "MAE": float(mae),
                    "MAPE": float(mape),
                    "RMSE": float(rmse),
                }
            )

        return {
            "MAE": float(sum(amae) / len(amae)),
            "MAPE": float(sum(amape) / len(amape)),
            "RMSE": float(sum(armse) / len(armse)),
            "LatencyMS": float(sum(latencies) / len(latencies)),
            "Params": int(self.count_parameters(model)),
            "CompressionRatio": ckpt.get("compression_ratio", ""),
            "BestEpoch": ckpt.get("best_epoch", ""),
            "BestValMAE": ckpt.get("best_val_mae", ckpt.get("best_val_loss", "")),
            "BestValLoss": ckpt.get("best_val_loss_at_best_mae", ckpt.get("best_val_loss", "")),
            "HorizonMetrics": horizon_rows,
        }


def rounded_metric(value: Any, digits: int = 4) -> Any:
    if value == "" or value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def mean_std(values: list[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        return float("nan"), float("nan")
    avg = sum(values) / len(values)
    denominator = len(values) - 1 if len(values) > 1 else len(values)
    variance = sum((value - avg) ** 2 for value in values) / denominator
    return avg, math.sqrt(variance)


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    ensure_parent(output_path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with resolve_path(output_path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Any, output_path: str | Path) -> None:
    ensure_parent(output_path)
    with resolve_path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_markdown_table(rows: list[dict[str, Any]], columns: list[str], output_path: str | Path, title: str) -> None:
    ensure_parent(output_path)
    with resolve_path(output_path).open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if not rows:
            handle.write("暂无结果。\n")
            return
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")
