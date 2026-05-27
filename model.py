from models.student_gcn import SimpleGCNStudent
from models.student_gru import SimpleGRUStudent
from models.student_dlinear import SimpleDLinearStudent
from models.student_stid import SimpleSTIDStudent
from models.student_tcn import SimpleTCNStudent
from models.teacher_gwnet import GWNetTeacher
from models.teacher_staeformer import STAEformerTeacher


TEACHER_MODEL_CHOICES = ("gwnet", "staeformer")
STUDENT_MODEL_CHOICES = ("gcn", "tcn", "gru", "stid", "dlinear")


def build_teacher_model(
    teacher_model,
    device,
    num_nodes,
    in_dim,
    out_dim,
    input_seq_len,
    dropout,
    supports=None,
    adjtype="doubletransition",
    gcn_bool=True,
    addaptadj=True,
    aptonly=False,
    randomadj=False,
    nhid=32,
    stae_steps_per_day=288,
    stae_input_embedding_dim=16,
    stae_tod_embedding_dim=16,
    stae_dow_embedding_dim=0,
    stae_spatial_embedding_dim=0,
    stae_adaptive_embedding_dim=32,
    stae_feed_forward_dim=128,
    stae_num_heads=4,
    stae_num_layers=2,
    stae_use_mixed_proj=True,
):
    """Build a supported teacher architecture."""
    if teacher_model == "gwnet":
        teacher_supports = None if aptonly else supports
        aptinit = None if randomadj or teacher_supports is None else teacher_supports[0]
        return GWNetTeacher(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=teacher_supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=out_dim,
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16,
        ).to(device)
    if teacher_model == "staeformer":
        return STAEformerTeacher(
            num_nodes=num_nodes,
            in_dim=in_dim,
            input_seq_len=input_seq_len,
            out_dim=out_dim,
            steps_per_day=stae_steps_per_day,
            input_embedding_dim=stae_input_embedding_dim,
            tod_embedding_dim=stae_tod_embedding_dim,
            dow_embedding_dim=stae_dow_embedding_dim,
            spatial_embedding_dim=stae_spatial_embedding_dim,
            adaptive_embedding_dim=stae_adaptive_embedding_dim,
            feed_forward_dim=stae_feed_forward_dim,
            num_heads=stae_num_heads,
            num_layers=stae_num_layers,
            dropout=dropout,
            use_mixed_proj=stae_use_mixed_proj,
        ).to(device)
    raise ValueError(f"Unsupported teacher_model: {teacher_model}")


def build_teacher_from_checkpoint(ckpt, supports, device):
    """Rebuild a teacher checkpoint, defaulting old checkpoints to GWNet."""
    teacher_model = ckpt.get("teacher_model", "gwnet")
    return build_teacher_model(
        teacher_model=teacher_model,
        device=device,
        num_nodes=ckpt["num_nodes"],
        in_dim=ckpt["in_dim"],
        out_dim=ckpt["seq_length"],
        input_seq_len=ckpt.get("input_seq_len", ckpt.get("seq_length", 12)),
        dropout=ckpt["dropout"],
        supports=supports,
        adjtype=ckpt.get("adjtype", "doubletransition"),
        gcn_bool=ckpt.get("gcn_bool", True),
        addaptadj=ckpt.get("addaptadj", True),
        aptonly=ckpt.get("aptonly", False),
        randomadj=ckpt.get("randomadj", False),
        nhid=ckpt.get("nhid", 32),
        stae_steps_per_day=ckpt.get("stae_steps_per_day", 288),
        stae_input_embedding_dim=ckpt.get("stae_input_embedding_dim", 16),
        stae_tod_embedding_dim=ckpt.get("stae_tod_embedding_dim", 16),
        stae_dow_embedding_dim=ckpt.get("stae_dow_embedding_dim", 0),
        stae_spatial_embedding_dim=ckpt.get("stae_spatial_embedding_dim", 0),
        stae_adaptive_embedding_dim=ckpt.get("stae_adaptive_embedding_dim", 32),
        stae_feed_forward_dim=ckpt.get("stae_feed_forward_dim", 128),
        stae_num_heads=ckpt.get("stae_num_heads", 4),
        stae_num_layers=ckpt.get("stae_num_layers", 2),
        stae_use_mixed_proj=ckpt.get("stae_use_mixed_proj", True),
    )


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
    "STAEformerTeacher",
    "SimpleDLinearStudent",
    "SimpleGCNStudent",
    "SimpleGRUStudent",
    "SimpleSTIDStudent",
    "SimpleTCNStudent",
    "TEACHER_MODEL_CHOICES",
    "STUDENT_MODEL_CHOICES",
    "build_teacher_model",
    "build_teacher_from_checkpoint",
    "build_student_model",
    "build_student_from_checkpoint",
]
