# STAEformer Teacher 接入说明与完整训练流程

更新时间：2026-05-25

本文档记录本次接入 `STAEformer Teacher` 的代码变化、实验命名规则，以及不覆盖现有实验产物的训练/测试命令。教师命名保持 `*_teacher_staeformer_wf`，学生实验统一使用 `v8_` 前缀，避免覆盖已经存在的 `*_v6`、`*_v7` 或历史 `*_best.pt` checkpoint。

## 1. 本次修改说明

本次修改的目标是把 `STAEformer Teacher` 作为第二种教师模型接入现有 CCKD 训练与评估流程，用于研究“更换教师模型后，CCKD 对主线 GCN 学生是否仍然有效”。

已完成的代码支持：

- 新增教师模型文件：`models/teacher_staeformer.py`。
- `model.py` 新增 `TEACHER_MODEL_CHOICES = ("gwnet", "staeformer")`。
- `model.py` 新增统一教师构建函数：`build_teacher_model()` 和 `build_teacher_from_checkpoint()`。
- `train.py` 支持 `--teacher_model gwnet|staeformer`。
- 教师 checkpoint 会保存 `teacher_model` 和 STAEformer 结构参数，旧 GWNet checkpoint 默认按 `gwnet` 兼容读取。
- `train_student_kd.py` 会根据教师 checkpoint 自动重建 GWNet 或 STAEformer Teacher。
- `test.py`、`scripts/benchmark_model.py`、`scripts/collect_results.py`、`compare_teacher_student.py`、`scripts/generate_distillation_heatmap.py` 均改为通过 checkpoint 元信息自动重建教师。
- `scripts/sanity_check.py` 已加入 STAEformer Teacher 前向形状检查。

未做的事情：

- 没有重跑任何训练。
- 没有覆盖任何已有 checkpoint。
- 没有改变 CCKD 损失逻辑。
- 没有改变已有学生模型结构。

## 2. 当前支持的教师和学生

教师模型：

```text
gwnet
staeformer
```

学生模型：

```text
gcn
tcn
gru
stid
dlinear
```

推荐论文主线仍然保持：

```text
GWNet Teacher -> Lightweight GCN Student
```

`STAEformer Teacher -> GCN Student` 建议作为“更换教师模型的泛化/消融实验”，不要直接替代主线故事。

## 3. 命名规则

为避免扰乱当前实验，教师命名保持不变；除教师外，学生实验命名统一使用 `v8_` 前缀：

```text
metr_teacher_staeformer_wf_best.pt
bay_teacher_staeformer_wf_best.pt
v8_metr_student_tcn_vanilla_stae_best.pt
v8_metr_student_tcn_cckd_stae_best.pt
v8_metr_student_gcn_stae_cckd_best.pt
```

如果你希望覆盖旧实验，请明确手动修改 `--exp_name`。默认不建议覆盖。

## 4. 训练 STAEformer Teacher
```bash
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_model staeformer --dropout 0.1 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --save_dir checkpoints/teacher --seed 42 --exp_name metr_teacher_staeformer_wf
python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_model staeformer --dropout 0.1 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --save_dir checkpoints/teacher --seed 42 --exp_name bay_teacher_staeformer_wf
```
### 4.1 METR-LA / STAEformer Teacher 训练

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_model staeformer --dropout 0.1 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --save_dir checkpoints/teacher --seed 42 --exp_name metr_teacher_staeformer_wf
```

### 4.2 METR-LA / STAEformer Teacher 测试

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --model_type teacher --exp_name metr_teacher_staeformer_wf_eval
```

### 4.3 PEMS-BAY / STAEformer Teacher 训练

```powershell
python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_model staeformer --dropout 0.1 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --save_dir checkpoints/teacher --seed 42 --exp_name bay_teacher_staeformer_wf
```

### 4.4 PEMS-BAY / STAEformer Teacher 测试

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --checkpoint checkpoints/teacher/bay_teacher_staeformer_wf_best.pt --model_type teacher --exp_name bay_teacher_staeformer_wf_eval
```

## 5. 泛化学生 Vanilla KD 与 CCKD

本节使用第 4 节训练得到的 STAEformer Teacher，考察不同轻量学生上的 Vanilla KD 与 CCKD。学生包括：

```text
gcn
tcn
gru
stid
dlinear
```

### 5.1 METR-LA / GCN Vanilla KD
```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_vanilla
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 80 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_cckd

```
```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_vanilla
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gcn_stae_vanilla_best.pt --model_type student --exp_name v8_metr_student_gcn_stae_vanilla_eval
```

### 5.2 METR-LA / GCN CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_cckd
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gcn_stae_cckd_best.pt --model_type student --exp_name v8_metr_student_gcn_stae_cckd_eval
```

### 5.3 METR-LA / TCN Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model tcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_tcn_vanilla_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_tcn_vanilla_stae_best.pt --model_type student --exp_name v8_metr_student_tcn_vanilla_stae_eval
```

### 5.4 METR-LA / TCN CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model tcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_tcn_cckd_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_tcn_cckd_stae_best.pt --model_type student --exp_name v8_metr_student_tcn_cckd_stae_eval
```

### 5.5 METR-LA / GRU Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gru --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gru_vanilla_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gru_vanilla_stae_best.pt --model_type student --exp_name v8_metr_student_gru_vanilla_stae_eval
```

### 5.6 METR-LA / GRU CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gru --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gru_cckd_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gru_cckd_stae_best.pt --model_type student --exp_name v8_metr_student_gru_cckd_stae_eval
```

### 5.7 METR-LA / STID Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model stid --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_stid_vanilla_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_stid_vanilla_stae_best.pt --model_type student --exp_name v8_metr_student_stid_vanilla_stae_eval
```

### 5.8 METR-LA / STID CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model stid --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_stid_cckd_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_stid_cckd_stae_best.pt --model_type student --exp_name v8_metr_student_stid_cckd_stae_eval
```

### 5.9 METR-LA / DLinear Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model dlinear --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_dlinear_vanilla_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_dlinear_vanilla_stae_best.pt --model_type student --exp_name v8_metr_student_dlinear_vanilla_stae_eval
```

### 5.10 METR-LA / DLinear CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model dlinear --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_dlinear_cckd_stae
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_dlinear_cckd_stae_best.pt --model_type student --exp_name v8_metr_student_dlinear_cckd_stae_eval
```

### 5.11 PEMS-BAY 泛化学生命令规则

PEMS-BAY 的命令与 METR-LA 完全一致，只替换下面四处：

```text
--data data/PEMS-BAY
--adjdata data/sensor_graph/adj_mx_bay.pkl
--teacher_checkpoint checkpoints/teacher/bay_teacher_staeformer_wf_best.pt
--exp_name v8_bay_student_<student>_<method>_stae
```

例如 PEMS-BAY / GCN Vanilla KD：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_bay_student_gcn_stae_vanilla
```

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --checkpoint checkpoints/student/v8_bay_student_gcn_stae_vanilla_best.pt --model_type student --exp_name v8_bay_student_gcn_stae_vanilla_eval
```

例如 PEMS-BAY / GCN CCKD：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_bay_student_gcn_stae_cckd
```

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --checkpoint checkpoints/student/v8_bay_student_gcn_stae_cckd_best.pt --model_type student --exp_name v8_bay_student_gcn_stae_cckd_eval
```

其余 `tcn`、`gru`、`stid`、`dlinear` 只需替换 `--student_model` 和 `--exp_name`，教师 checkpoint 仍使用对应数据集的 STAEformer Teacher。

## 6. 新教师 STAEformer -> GCN 学生消融

本节用于验证“更换教师后，CCKD 对主线 GCN 学生是否仍然有效”。这里的教师 checkpoint 使用第 4 节训练得到的 STAEformer Teacher。

### 6.1 METR-LA / STAEformer Teacher + GCN Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_vanilla
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gcn_stae_vanilla_best.pt --model_type student --exp_name v8_metr_student_gcn_stae_vanilla_eval
```

### 6.2 METR-LA / STAEformer Teacher + GCN w/o Confidence

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_wo_conf
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gcn_stae_wo_conf_best.pt --model_type student --exp_name v8_metr_student_gcn_stae_wo_conf_eval
```

### 6.3 METR-LA / STAEformer Teacher + GCN w/o Curriculum

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_curriculum --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_wo_curr
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gcn_stae_wo_curr_best.pt --model_type student --exp_name v8_metr_student_gcn_stae_wo_curr_eval
```

### 6.4 METR-LA / STAEformer Teacher + GCN Full CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_metr_student_gcn_stae_cckd
```

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/student/v8_metr_student_gcn_stae_cckd_best.pt --model_type student --exp_name v8_metr_student_gcn_stae_cckd_eval
```

### 6.5 PEMS-BAY / STAEformer Teacher + GCN 消融命令规则

PEMS-BAY 版本只替换数据、邻接矩阵、教师 checkpoint 和实验名前缀：

```text
--data data/PEMS-BAY
--adjdata data/sensor_graph/adj_mx_bay.pkl
--teacher_checkpoint checkpoints/teacher/bay_teacher_staeformer_wf_best.pt
--exp_name v8_bay_student_gcn_stae_<ablation>
```

例如 PEMS-BAY / STAEformer Teacher + GCN Full CCKD：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_staeformer_wf_best.pt --student_model gcn --student_hidden_dim 32 --student_layers 2 --dropout 0.3 --student_order 2 --epochs 50 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --save_dir checkpoints/student --seed 42 --exp_name v8_bay_student_gcn_stae_cckd
```

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --checkpoint checkpoints/student/v8_bay_student_gcn_stae_cckd_best.pt --model_type student --exp_name v8_bay_student_gcn_stae_cckd_eval
```

## 7. 结果汇总命令

METR-LA / STAEformer Teacher + GCN 消融汇总：

```powershell
python -m scripts.collect_results --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --output_csv outputs/reports/v8_metr_stae_gcn_ablation.csv --output_md outputs/reports/v8_metr_stae_gcn_ablation.md --run "STAEformer Teacher,teacher,checkpoints/teacher/metr_teacher_staeformer_wf_best.pt" --run "GCN Vanilla KD,student,checkpoints/student/v8_metr_student_gcn_stae_vanilla_best.pt" --run "GCN w/o Confidence,student,checkpoints/student/v8_metr_student_gcn_stae_wo_conf_best.pt" --run "GCN w/o Curriculum,student,checkpoints/student/v8_metr_student_gcn_stae_wo_curr_best.pt" --run "GCN CCKD,student,checkpoints/student/v8_metr_student_gcn_stae_cckd_best.pt"
```

PEMS-BAY / STAEformer Teacher + GCN 消融汇总：

```powershell
python -m scripts.collect_results --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --output_csv outputs/reports/v8_bay_stae_gcn_ablation.csv --output_md outputs/reports/v8_bay_stae_gcn_ablation.md --run "STAEformer Teacher,teacher,checkpoints/teacher/bay_teacher_staeformer_wf_best.pt" --run "GCN Vanilla KD,student,checkpoints/student/v8_bay_student_gcn_stae_vanilla_best.pt" --run "GCN w/o Confidence,student,checkpoints/student/v8_bay_student_gcn_stae_wo_conf_best.pt" --run "GCN w/o Curriculum,student,checkpoints/student/v8_bay_student_gcn_stae_wo_curr_best.pt" --run "GCN CCKD,student,checkpoints/student/v8_bay_student_gcn_stae_cckd_best.pt"
```

## 8. 注意事项

- `Vanilla KD` 建议显式设置 `--trend_weight 0.0 --disable_confidence_filter --disable_curriculum`，避免趋势项权重影响普通 KD 的软损失尺度。
- `CCKD` 建议显式设置 `--curriculum_mode soft`，这样论文里“所有 horizon 始终参与”的表述与代码一致。
- STAEformer Teacher 的 checkpoint 训练完成后，学生蒸馏入口会自动根据 checkpoint 中的 `teacher_model=staeformer` 重建教师，无需额外参数。
- 如果使用 `python train.py --help` 在 Windows 上遇到编码显示问题，先确认当前终端已经进入你自己选择的环境；这不是训练代码错误。
- 本文档命令仅给出流程，不代表已经跑完对应实验。填论文结果前必须用 `test.py` 或 `scripts.collect_results.py` 重新确认真实指标。





