import argparse
import time

import torch

import util
from engine import count_parameters, prepare_batch
from model import GWNetTeacher, SimpleGCNStudent


def parse_args():
    parser = argparse.ArgumentParser(description="统计模型参数量与推理速度。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


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
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]
    ckpt = util.load_checkpoint(args.checkpoint, map_location=device)
    model = build_model(args, ckpt, device, supports)
    params = count_parameters(model)

    iterator = dataloader["test_loader"].get_iterator()
    x, y = next(iterator)
    inputs, _ = prepare_batch(x, y, device)

    for _ in range(args.warmup):
        with torch.no_grad():
            if args.model_type == "teacher":
                _ = model(torch.nn.functional.pad(inputs, (1, 0, 0, 0)))
            else:
                _ = model(inputs, supports)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        with torch.no_grad():
            if args.model_type == "teacher":
                _ = model(torch.nn.functional.pad(inputs, (1, 0, 0, 0)))
            else:
                _ = model(inputs, supports)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

    print(f"model_type={args.model_type}")
    print(f"params={params:,}")
    print(f"avg_latency_ms={sum(latencies) / len(latencies):.4f}")
    print(f"batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
