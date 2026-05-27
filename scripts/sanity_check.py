import torch

from model import (
    GWNetTeacher,
    SimpleDLinearStudent,
    SimpleGCNStudent,
    SimpleGRUStudent,
    SimpleSTIDStudent,
    SimpleTCNStudent,
    STAEformerTeacher,
)


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
    stae_teacher = STAEformerTeacher(
        num_nodes=num_nodes,
        in_dim=in_dim,
        input_seq_len=input_len,
        out_dim=horizon,
        input_embedding_dim=8,
        tod_embedding_dim=8,
        adaptive_embedding_dim=16,
        feed_forward_dim=32,
        num_heads=4,
        num_layers=1,
    )
    gcn_student = SimpleGCNStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=horizon,
        support_len=len(supports),
        input_seq_len=input_len,
    )
    tcn_student = SimpleTCNStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=horizon,
        temporal_layers=2,
        input_seq_len=input_len,
    )
    gru_student = SimpleGRUStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=horizon,
        recurrent_layers=2,
        input_seq_len=input_len,
    )
    stid_student = SimpleSTIDStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=horizon,
        mlp_layers=2,
        input_seq_len=input_len,
    )
    dlinear_student = SimpleDLinearStudent(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=horizon,
        input_seq_len=input_len,
    )

    x = torch.randn(batch_size, in_dim, num_nodes, input_len)
    x[:, 1, :, :] = torch.rand(batch_size, num_nodes, input_len)
    teacher_out = teacher(torch.nn.functional.pad(x, (1, 0, 0, 0)), return_features=True)
    stae_teacher_out = stae_teacher(torch.nn.functional.pad(x, (1, 0, 0, 0)), return_features=True)
    gcn_out = gcn_student(x, supports, return_features=True)
    tcn_out = tcn_student(x, supports, return_features=True)
    gru_out = gru_student(x, supports, return_features=True)
    stid_out = stid_student(x, supports, return_features=True)
    dlinear_out = dlinear_student(x, supports, return_features=True)

    assert teacher_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert stae_teacher_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert gcn_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert tcn_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert gru_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert stid_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert dlinear_out["prediction"].shape == (batch_size, horizon, num_nodes, 1)
    assert teacher_out["hidden_state"].shape[0] == batch_size
    assert stae_teacher_out["hidden_state"].shape[0] == batch_size
    assert gcn_out["hidden_state"].shape[0] == batch_size
    assert tcn_out["hidden_state"].shape[0] == batch_size
    assert gru_out["hidden_state"].shape[0] == batch_size
    assert stid_out["hidden_state"].shape[0] == batch_size
    assert dlinear_out["hidden_state"].shape[0] == batch_size

    print("Sanity check passed.")
    print(f"teacher prediction shape: {teacher_out['prediction'].shape}")
    print(f"staeformer teacher prediction shape: {stae_teacher_out['prediction'].shape}")
    print(f"gcn student prediction shape: {gcn_out['prediction'].shape}")
    print(f"tcn student prediction shape: {tcn_out['prediction'].shape}")
    print(f"gru student prediction shape: {gru_out['prediction'].shape}")
    print(f"stid student prediction shape: {stid_out['prediction'].shape}")
    print(f"dlinear student prediction shape: {dlinear_out['prediction'].shape}")


if __name__ == "__main__":
    main()
