from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiment_common import (
    CheckpointEvaluator,
    add_student_training_args,
    build_student_train_command,
    command_to_string,
    ensure_dir,
    mean_std,
    project_path,
    resolve_path,
    rounded_metric,
    run_logged_command,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CCKD student training with multiple random seeds and summarize test metrics."
    )
    add_student_training_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
    parser.add_argument("--exp_prefix", type=str, default="metr_cckd_seed")
    parser.add_argument("--report_dir", type=str, default="outputs/reports/seed_experiments")
    parser.add_argument("--log_dir", type=str, default="logs/seed_experiments")
    parser.add_argument("--skip_existing", action="store_true", help="Skip training if the checkpoint already exists.")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--dry_run", action="store_true", help="Print training commands without running them.")
    return parser.parse_args()


def checkpoint_for(args: argparse.Namespace, exp_name: str) -> str:
    return project_path(Path(args.save_dir) / f"{exp_name}_best.pt")


def build_row(
    *,
    seed: int,
    exp_name: str,
    checkpoint_path: str,
    command: list[str],
    run_info: dict,
    metrics: dict,
) -> dict:
    return {
        "Experiment": "random_seed",
        "Seed": seed,
        "ExpName": exp_name,
        "StudentModel": metrics.get("StudentModel", ""),
        "CurriculumMode": metrics.get("CurriculumMode", ""),
        "HardWeight": metrics.get("HardWeight", ""),
        "SoftWeight": metrics.get("SoftWeight", ""),
        "TrendWeight": metrics.get("TrendWeight", ""),
        "DynamicEta": metrics.get("DynamicEta", ""),
        "BestEpoch": metrics.get("BestEpoch", ""),
        "BestValMAE": rounded_metric(metrics.get("BestValMAE")),
        "TestMAE": rounded_metric(metrics.get("MAE")),
        "TestMAPE": rounded_metric(metrics.get("MAPE")),
        "TestRMSE": rounded_metric(metrics.get("RMSE")),
        "LatencyMS": rounded_metric(metrics.get("LatencyMS")),
        "Params": metrics.get("Params", ""),
        "CompressionRatio": rounded_metric(metrics.get("CompressionRatio")),
        "Checkpoint": checkpoint_path,
        "ElapsedSec": rounded_metric(run_info.get("elapsed_sec", "")),
        "Command": command_to_string(command),
    }


def enrich_metrics(metrics: dict, args: argparse.Namespace) -> dict:
    metrics = dict(metrics)
    metrics.update(
        {
            "StudentModel": args.student_model,
            "CurriculumMode": args.curriculum_mode,
            "HardWeight": args.hard_weight,
            "SoftWeight": args.soft_weight,
            "TrendWeight": args.trend_weight,
            "DynamicEta": args.dynamic_curriculum_eta,
        }
    )
    return metrics


def write_seed_markdown(rows: list[dict], output_path: str) -> None:
    metrics = ["TestMAE", "TestMAPE", "TestRMSE", "LatencyMS"]
    summary = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) != ""]
        if values:
            avg, std = mean_std(values)
            summary[metric] = f"{avg:.4f} ± {std:.4f}"

    columns = [
        "Seed",
        "ExpName",
        "BestEpoch",
        "BestValMAE",
        "TestMAE",
        "TestMAPE",
        "TestRMSE",
        "LatencyMS",
        "Checkpoint",
    ]

    with resolve_path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("# 三随机种子稳定性实验结果\n\n")
        handle.write(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write("- 统计口径：训练后加载 best checkpoint，在测试集上计算 12 步平均 MAE / MAPE / RMSE。\n")
        handle.write("- 标准差：当 seed 数量大于 1 时使用样本标准差。\n\n")
        if summary:
            handle.write("## 汇总\n\n")
            handle.write("| 指标 | mean ± std |\n")
            handle.write("| --- | --- |\n")
            for metric, value in summary.items():
                handle.write(f"| {metric} | {value} |\n")
            handle.write("\n")

        handle.write("## 每个 seed 的结果\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")
        handle.write("\n")

        handle.write("## 训练命令\n\n")
        for row in rows:
            handle.write(f"### seed={row['Seed']}\n\n")
            handle.write("```powershell\n")
            handle.write(row["Command"])
            handle.write("\n```\n\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.report_dir)
    ensure_dir(args.log_dir)

    if args.dry_run:
        for seed in args.seeds:
            exp_name = f"{args.exp_prefix}_seed{seed}"
            command = build_student_train_command(args, exp_name=exp_name, seed=seed)
            run_logged_command(command, Path(args.log_dir) / f"{exp_name}.log", dry_run=True)
        return

    evaluator = CheckpointEvaluator(
        device=args.device,
        data=args.data,
        adjdata=args.adjdata,
        adjtype=args.adjtype,
        batch_size=args.batch_size,
    )

    rows = []
    details = []
    for seed in args.seeds:
        exp_name = f"{args.exp_prefix}_seed{seed}"
        checkpoint_path = checkpoint_for(args, exp_name)
        checkpoint_abs = resolve_path(checkpoint_path)
        command = build_student_train_command(args, exp_name=exp_name, seed=seed)

        if args.eval_only or (args.skip_existing and checkpoint_abs.exists()):
            print(f"[skip] use existing checkpoint: {checkpoint_path}")
            run_info = {"elapsed_sec": 0.0, "command": command_to_string(command)}
        else:
            run_info = run_logged_command(command, Path(args.log_dir) / f"{exp_name}.log")

        metrics = evaluator.evaluate(checkpoint_path, model_type="student")
        metrics = enrich_metrics(metrics, args)
        rows.append(
            build_row(
                seed=seed,
                exp_name=exp_name,
                checkpoint_path=checkpoint_path,
                command=command,
                run_info=run_info,
                metrics=metrics,
            )
        )
        details.append({"seed": seed, "checkpoint": checkpoint_path, "metrics": metrics})

    csv_path = Path(args.report_dir) / "cckd_seed_runs.csv"
    json_path = Path(args.report_dir) / "cckd_seed_runs.json"
    md_path = Path(args.report_dir) / "cckd_seed_runs.md"
    write_csv(rows, csv_path)
    write_json({"rows": rows, "details": details}, json_path)
    write_seed_markdown(rows, str(md_path))

    print(f"csv_saved={project_path(csv_path)}")
    print(f"json_saved={project_path(json_path)}")
    print(f"md_saved={project_path(md_path)}")


if __name__ == "__main__":
    main()
