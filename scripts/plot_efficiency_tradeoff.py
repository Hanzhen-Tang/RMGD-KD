import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据结果汇总表绘制教师-学生时间与性能折中图。"
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="outputs/reports/result_summary.csv",
        help="由 scripts/collect_results.py 生成的结果汇总 CSV。",
    )
    parser.add_argument(
        "--teacher_name",
        type=str,
        default="Teacher",
        help="汇总表中教师模型对应的 Name 名称。",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="MAE",
        choices=["MAE", "RMSE", "MAPE"],
        help="用于定义性能达成度的误差指标，数值越小越好。",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/figures/efficiency_tradeoff.png",
        help="输出图像路径。",
    )
    parser.add_argument(
        "--derived_csv",
        type=str,
        default="outputs/reports/efficiency_tradeoff.csv",
        help="保存派生后的时间-性能结果表。",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def load_rows(summary_csv: str) -> List[Dict[str, str]]:
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"CSV 中缺少 {key} 的有效数值。")
    return float(value)


def build_derived_rows(rows: List[Dict[str, str]], teacher_name: str, metric: str):
    teacher_row = None
    for row in rows:
        if row.get("Name") == teacher_name:
            teacher_row = row
            break

    if teacher_row is None:
        raise ValueError(f"未在汇总表中找到教师模型: {teacher_name}")

    teacher_metric = to_float(teacher_row, metric)
    teacher_latency = to_float(teacher_row, "LatencyMS")

    if teacher_metric <= 0 or teacher_latency <= 0:
        raise ValueError("教师模型的误差值和推理时间必须大于 0。")

    derived_rows = []
    for row in rows:
        model_metric = to_float(row, metric)
        model_latency = to_float(row, "LatencyMS")

        # 误差越小越好，因此用教师误差 / 当前误差来表示达到教师性能的百分比。
        performance_pct = (teacher_metric / model_metric) * 100.0
        time_pct = (model_latency / teacher_latency) * 100.0
        speedup = teacher_latency / model_latency

        derived_rows.append(
            {
                "Name": row.get("Name", ""),
                "ModelType": row.get("ModelType", ""),
                "Metric": metric,
                "MetricValue": round(model_metric, 4),
                "LatencyMS": round(model_latency, 4),
                "TimePctOfTeacher": round(time_pct, 2),
                "PerformancePctOfTeacher": round(performance_pct, 2),
                "SpeedupVsTeacher": round(speedup, 4),
            }
        )

    return derived_rows


def save_derived_csv(rows: List[Dict[str, object]], save_path: str):
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Name",
                "ModelType",
                "Metric",
                "MetricValue",
                "LatencyMS",
                "TimePctOfTeacher",
                "PerformancePctOfTeacher",
                "SpeedupVsTeacher",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(rows: List[Dict[str, object]], teacher_name: str, metric: str, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(10, 6))

    teacher_x = None
    teacher_y = None
    for row in rows:
        x = row["TimePctOfTeacher"]
        y = row["PerformancePctOfTeacher"]
        name = row["Name"]
        is_teacher = name == teacher_name
        color = "#d62728" if is_teacher else "#1f77b4"
        size = 150 if is_teacher else 100

        plt.scatter(x, y, s=size, color=color, alpha=0.85)
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=10)

        if is_teacher:
            teacher_x = x
            teacher_y = y

    if teacher_x is not None and teacher_y is not None:
        plt.axvline(teacher_x, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
        plt.axhline(teacher_y, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)

    plt.xlabel("Inference Time (% of Teacher)")
    plt.ylabel(f"Prediction Performance Reached (% of Teacher, based on {metric})")
    plt.title("Teacher-Student Efficiency Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    rows = load_rows(args.summary_csv)
    derived_rows = build_derived_rows(rows, args.teacher_name, args.metric)
    save_derived_csv(derived_rows, args.derived_csv)
    plot_tradeoff(derived_rows, args.teacher_name, args.metric, args.save_path)
    print(f"figure_saved={args.save_path}")
    print(f"derived_csv_saved={args.derived_csv}")


if __name__ == "__main__":
    main()
