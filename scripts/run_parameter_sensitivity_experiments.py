from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from experiment_common import (
    CheckpointEvaluator,
    add_student_training_args,
    build_student_train_command,
    command_to_string,
    ensure_dir,
    float_token,
    mean_std,
    parse_loss_pair,
    project_path,
    resolve_path,
    rounded_metric,
    run_logged_command,
    write_csv,
    write_json,
)


PARAMETER_LABELS = {
    "loss_weight": "真实标签监督 / 蒸馏监督权重",
    "trend_weight": "趋势蒸馏权重系数",
    "dynamic_eta": "动态课程增强系数",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CCKD parameter sensitivity training and summarize test metrics."
    )
    add_student_training_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=["all"],
        choices=["all", "loss_weight", "trend_weight", "dynamic_eta"],
    )
    parser.add_argument(
        "--loss_weight_pairs",
        nargs="+",
        default=["0.9/0.1", "0.8/0.2", "0.7/0.3", "0.6/0.4", "0.5/0.5"],
        help="Candidate hard/soft pairs. Both 0.7/0.3 and 0.7:0.3 are accepted.",
    )
    parser.add_argument("--trend_weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--dynamic_etas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--exp_prefix", type=str, default="metr_cckd_sensitivity")
    parser.add_argument("--report_dir", type=str, default="outputs/reports/parameter_sensitivity")
    parser.add_argument("--log_dir", type=str, default="logs/parameter_sensitivity")
    parser.add_argument(
        "--plot_source_csv",
        type=str,
        default="outputs/reports/figure_3_7_metr_parameter_sensitivity_source.csv",
        help="CSV in the same schema as the paper figure source data.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip training if the checkpoint already exists.")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--dry_run", action="store_true", help="Print training commands without running them.")
    return parser.parse_args()


def selected_sweeps(args: argparse.Namespace) -> list[str]:
    if "all" in args.sweeps:
        return ["loss_weight", "trend_weight", "dynamic_eta"]
    return args.sweeps


def fmt_value(value: float) -> str:
    return f"{value:g}"


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = []
    sweeps = selected_sweeps(args)

    if "loss_weight" in sweeps:
        default_value = f"{fmt_value(args.hard_weight)}/{fmt_value(args.soft_weight)}"
        for pair in args.loss_weight_pairs:
            hard_weight, soft_weight = parse_loss_pair(pair)
            value = f"{fmt_value(hard_weight)}/{fmt_value(soft_weight)}"
            cases.append(
                {
                    "Sweep": "loss_weight",
                    "ParameterLabel": PARAMETER_LABELS["loss_weight"],
                    "Value": value,
                    "IsDefault": int(value == default_value),
                    "Token": f"loss_h{float_token(hard_weight)}_s{float_token(soft_weight)}",
                    "Overrides": {"hard_weight": hard_weight, "soft_weight": soft_weight},
                }
            )

    if "trend_weight" in sweeps:
        for trend_weight in args.trend_weights:
            cases.append(
                {
                    "Sweep": "trend_weight",
                    "ParameterLabel": PARAMETER_LABELS["trend_weight"],
                    "Value": fmt_value(trend_weight),
                    "IsDefault": int(abs(trend_weight - args.trend_weight) < 1e-12),
                    "Token": f"trend{float_token(trend_weight)}",
                    "Overrides": {"trend_weight": trend_weight},
                }
            )

    if "dynamic_eta" in sweeps:
        for eta in args.dynamic_etas:
            cases.append(
                {
                    "Sweep": "dynamic_eta",
                    "ParameterLabel": PARAMETER_LABELS["dynamic_eta"],
                    "Value": fmt_value(eta),
                    "IsDefault": int(abs(eta - args.dynamic_curriculum_eta) < 1e-12),
                    "Token": f"eta{float_token(eta)}",
                    "Overrides": {"dynamic_curriculum_eta": eta, "curriculum_mode": "dynamic_soft"},
                }
            )

    return cases


def checkpoint_for(args: argparse.Namespace, exp_name: str) -> str:
    return project_path(Path(args.save_dir) / f"{exp_name}_best.pt")


def row_from_metrics(
    *,
    case: dict[str, Any],
    exp_name: str,
    checkpoint_path: str,
    command: list[str],
    run_info: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    overrides = case["Overrides"]
    return {
        "Experiment": "parameter_sensitivity",
        "Sweep": case["Sweep"],
        "ParameterLabel": case["ParameterLabel"],
        "Value": case["Value"],
        "IsDefault": case["IsDefault"],
        "Seed": args.seed,
        "ExpName": exp_name,
        "StudentModel": args.student_model,
        "CurriculumMode": overrides.get("curriculum_mode", args.curriculum_mode),
        "HardWeight": overrides.get("hard_weight", args.hard_weight),
        "SoftWeight": overrides.get("soft_weight", args.soft_weight),
        "TrendWeight": overrides.get("trend_weight", args.trend_weight),
        "DynamicEta": overrides.get("dynamic_curriculum_eta", args.dynamic_curriculum_eta),
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


def write_sensitivity_markdown(rows: list[dict[str, Any]], output_path: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["Sweep"]].append(row)

    columns = [
        "Value",
        "IsDefault",
        "BestEpoch",
        "BestValMAE",
        "TestMAE",
        "TestMAPE",
        "TestRMSE",
        "LatencyMS",
        "Checkpoint",
    ]

    with resolve_path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("# 参数敏感性实验结果\n\n")
        handle.write(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write("- 统计口径：每个设置训练一个学生模型，加载 best checkpoint 后在测试集上计算 12 步平均指标。\n")
        handle.write("- 控制变量：每组实验只改变当前扫描参数，其余训练参数沿用主实验配置。\n")
        handle.write("- 注意：`dynamic_eta` 只有在 `curriculum_mode=dynamic_soft` 下生效，本脚本会对该扫描项自动启用 `dynamic_soft`。\n\n")

        for sweep in ["loss_weight", "trend_weight", "dynamic_eta"]:
            if sweep not in grouped:
                continue
            handle.write(f"## {PARAMETER_LABELS[sweep]}\n\n")
            handle.write("| " + " | ".join(columns) + " |\n")
            handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
            for row in grouped[sweep]:
                handle.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")
            handle.write("\n")

        handle.write("## 训练命令\n\n")
        for row in rows:
            handle.write(f"### {row['Sweep']}={row['Value']}\n\n")
            handle.write("```powershell\n")
            handle.write(row["Command"])
            handle.write("\n```\n\n")


def write_plot_source(rows: list[dict[str, Any]], output_path: str, dataset_name: str) -> None:
    source_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Sweep"], row["Value"])].append(row)

    for (sweep, value), group_rows in grouped.items():
        mae_values = [float(row["TestMAE"]) for row in group_rows if row.get("TestMAE") != ""]
        rmse_values = [float(row["TestRMSE"]) for row in group_rows if row.get("TestRMSE") != ""]
        mae_mean, mae_std = mean_std(mae_values)
        rmse_mean, rmse_std = mean_std(rmse_values)
        source_rows.append(
            {
                "dataset": dataset_name,
                "parameter": sweep,
                "parameter_label": PARAMETER_LABELS[sweep],
                "value": value,
                "is_default": group_rows[0].get("IsDefault", 0),
                "MAE": f"{mae_mean:.4f}",
                "MAE_std": f"{mae_std:.4f}",
                "RMSE": f"{rmse_mean:.4f}",
                "RMSE_std": f"{rmse_std:.4f}",
                "n": len(group_rows),
                "data_role": "real_experiment",
            }
        )

    write_csv(source_rows, output_path)


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    ensure_dir(args.report_dir)
    ensure_dir(args.log_dir)

    if args.dry_run:
        for case in cases:
            exp_name = f"{args.exp_prefix}_{case['Token']}_seed{args.seed}"
            command = build_student_train_command(args, exp_name=exp_name, seed=args.seed, overrides=case["Overrides"])
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
    for case in cases:
        exp_name = f"{args.exp_prefix}_{case['Token']}_seed{args.seed}"
        checkpoint_path = checkpoint_for(args, exp_name)
        checkpoint_abs = resolve_path(checkpoint_path)
        command = build_student_train_command(args, exp_name=exp_name, seed=args.seed, overrides=case["Overrides"])

        if args.eval_only or (args.skip_existing and checkpoint_abs.exists()):
            print(f"[skip] use existing checkpoint: {checkpoint_path}")
            run_info = {"elapsed_sec": 0.0, "command": command_to_string(command)}
        else:
            run_info = run_logged_command(command, Path(args.log_dir) / f"{exp_name}.log")

        metrics = evaluator.evaluate(checkpoint_path, model_type="student")
        row = row_from_metrics(
            case=case,
            exp_name=exp_name,
            checkpoint_path=checkpoint_path,
            command=command,
            run_info=run_info,
            metrics=metrics,
            args=args,
        )
        rows.append(row)
        details.append({"case": case, "checkpoint": checkpoint_path, "metrics": metrics})

    csv_path = Path(args.report_dir) / "cckd_parameter_sensitivity_runs.csv"
    json_path = Path(args.report_dir) / "cckd_parameter_sensitivity_runs.json"
    md_path = Path(args.report_dir) / "cckd_parameter_sensitivity_runs.md"
    write_csv(rows, csv_path)
    write_json({"rows": rows, "details": details}, json_path)
    write_sensitivity_markdown(rows, str(md_path))
    write_plot_source(rows, args.plot_source_csv, dataset_name=Path(args.data).name)

    print(f"csv_saved={project_path(csv_path)}")
    print(f"json_saved={project_path(json_path)}")
    print(f"md_saved={project_path(md_path)}")
    print(f"plot_source_saved={project_path(args.plot_source_csv)}")


if __name__ == "__main__":
    main()
