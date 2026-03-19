import torch

from model import GWNetTeacher, SimpleGCNStudent


def main():
    device = torch.device("cpu")
    batch_size = 4
    num_nodes = 8
    in_dim = 2
    input_len = 12
    horizon = 12

    supports = [torch.eye(num_nodes), torch.eye(num_nodes)]
    teacher = GWNetTeacher(
        device=device,
        num_nodes=num_nodes,
        supports=supports,
        gcn_bool=True,
        addaptadj=True,
        in_dim=in_dim,
        out_dim=horizon,
    )
    student = SimpleGCNStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=horizon,
        support_len=len(supports),
        input_seq_len=input_len,
    )

    x = torch.randn(batch_size, in_dim, num_nodes, input_len)
    teacher_out = teacher(torch.nn.functional.pad(x, (1, 0, 0, 0)), return_features=True)
    student_out = student(x, supports, return_features=True)

    assert teacher_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert student_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert teacher_out["hidden_state"].shape[0] == batch_size
    assert student_out["hidden_state"].shape[0] == batch_size

    print("Sanity check passed.")
    print(f"teacher prediction shape: {teacher_out['prediction'].shape}")
    print(f"student prediction shape: {student_out['prediction'].shape}")


if __name__ == "__main__":
    main()
