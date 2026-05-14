from models.student_gcn import SimpleGCNStudent
from models.student_gru import SimpleGRUStudent
from models.student_dlinear import SimpleDLinearStudent
from models.student_stid import SimpleSTIDStudent
from models.student_tcn import SimpleTCNStudent
from models.teacher_gwnet import GWNetTeacher


STUDENT_MODEL_CHOICES = ("gcn", "tcn", "gru", "stid", "dlinear")


def build_student_model(
    student_model,
    num_nodes,
    in_dim,
    hidden_dim,
    out_dim,
    dropout,
    support_len,
    gcn_order,
    graph_layers,
    input_seq_len,
):
    """Build a supported lightweight student architecture."""
    if student_model == "gcn":
        return SimpleGCNStudent(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            support_len=support_len,
            gcn_order=gcn_order,
            graph_layers=graph_layers,
            input_seq_len=input_seq_len,
        )
    if student_model == "tcn":
        return SimpleTCNStudent(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            temporal_layers=graph_layers,
            input_seq_len=input_seq_len,
        )
    if student_model == "gru":
        return SimpleGRUStudent(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            recurrent_layers=graph_layers,
            input_seq_len=input_seq_len,
        )
    if student_model == "stid":
        return SimpleSTIDStudent(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            mlp_layers=graph_layers,
            input_seq_len=input_seq_len,
        )
    if student_model == "dlinear":
        return SimpleDLinearStudent(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            input_seq_len=input_seq_len,
        )
    raise ValueError(f"Unsupported student_model: {student_model}")


def build_student_from_checkpoint(ckpt, supports, device):
    """Rebuild a student checkpoint, defaulting old checkpoints to the GCN student."""
    return build_student_model(
        student_model=ckpt.get("student_model", "gcn"),
        num_nodes=ckpt["num_nodes"],
        in_dim=ckpt["in_dim"],
        hidden_dim=ckpt["student_hidden_dim"],
        out_dim=ckpt["seq_length"],
        dropout=ckpt["dropout"],
        support_len=len(supports),
        gcn_order=ckpt.get("student_order", 2),
        graph_layers=ckpt["student_layers"],
        input_seq_len=ckpt["input_seq_len"],
    ).to(device)


__all__ = [
    "GWNetTeacher",
    "SimpleDLinearStudent",
    "SimpleGCNStudent",
    "SimpleGRUStudent",
    "SimpleSTIDStudent",
    "SimpleTCNStudent",
    "STUDENT_MODEL_CHOICES",
    "build_student_model",
    "build_student_from_checkpoint",
]
