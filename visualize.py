import argparse
import json
import os

from utils.plotting import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="根据训练历史文件绘制损失曲线。")
    parser.add_argument("--history", type=str, required=True, help="训练历史 JSON 文件路径。")
    parser.add_argument("--save_path", type=str, required=True, help="图像输出路径。")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.history, "r", encoding="utf-8") as f:
        history = json.load(f)

    output_dir = os.path.dirname(args.save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plot_training_curves(history, args.save_path)
    print(f"训练曲线已保存到: {args.save_path}")


if __name__ == "__main__":
    main()
