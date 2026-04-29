# v6 泛化实验说明文档

更新时间：2026-04-29

## 1. 本次更新的原因

老师提出了一个很重要的问题：如果只在 `GWNet Teacher + Lightweight GCN Student` 这一组教师学生结构上验证 CCKD，审稿人可能会追问：

> 这个蒸馏策略是不是只对当前这个学生模型有效？

所以 v6 的目的不是继续追求单个模型的最高精度，而是补充一个“方法泛化性”实验：在不改变教师模型和蒸馏框架的情况下，更换不同类型的轻量学生模型，观察 CCKD 是否仍然能稳定优于未蒸馏学生和普通 KD。

v6 从原来只增加 `Lightweight TCN Student`，进一步扩展为两个额外学生：

```text
Lightweight GCN Student：原主学生，偏空间图结构建模
Lightweight TCN Student：新增学生，偏时间卷积建模
Lightweight GRU Student：新增学生，偏循环序列建模
```

这样论文里可以更稳妥地表达：

```text
本文方法不仅在主学生 Lightweight GCN 上有效，在时间卷积型学生和循环序列型学生上也能带来提升，说明 CCKD 对不同轻量学生结构具有一定适配能力。
```

注意：不要写成“充分证明模型无关泛化能力”，更稳妥的说法是“补充验证了对不同轻量学生结构的适配性”。

## 2. 本次代码改了什么

### 2.1 已有的 TCN 学生

文件：

```text
models/student_tcn.py
```

模型：

```text
SimpleTCNStudent
```

作用：作为时间卷积型轻量学生，用来验证 CCKD 对时间卷积学生是否仍然有效。

### 2.2 新增的 GRU 学生

文件：

```text
models/student_gru.py
```

模型：

```text
SimpleGRUStudent
```

作用：作为循环序列型轻量学生，用来验证 CCKD 对 recurrent sequence student 是否仍然有效。

该模型输入格式与原来的 GCN/TCN 学生一致：

```text
[B, C, N, T]
```

输出格式也一致：

```text
prediction:   [B, H, N, 1]
hidden_state: [B, hidden_dim, N, 1]
```

因此，GRU 学生可以直接复用现有的置信度双路径蒸馏和 soft curriculum 损失，不需要改动蒸馏模块。

### 2.3 学生模型选择参数

现在训练脚本支持：

```text
--student_model gcn|tcn|gru
```

含义：

```text
--student_model gcn：使用原主学生 Lightweight GCN Student
--student_model tcn：使用新增 Lightweight TCN Student
--student_model gru：使用新增 Lightweight GRU Student
```

旧 checkpoint 如果没有保存 `student_model` 字段，会默认按 `gcn` 读取，保证旧实验还能正常测试。

### 2.4 soft curriculum 方向修正

v6 已经修正 `curriculum_mode=soft` 的方向：

```text
训练早期：短期预测步权重更高
长期预测步：仍然参与训练，但初始权重较低
训练推进：长期预测步权重逐渐增大
训练后期：所有预测步权重趋于一致
```

如果论文中使用“Soft Curriculum Across 12-Step Forecasting Horizons”这张图，就建议最终 CCKD 实验使用：

```text
--curriculum_mode soft
```

## 3. 推荐实验顺序

建议按下面顺序跑，逻辑最清楚。

```text
第 0 步：确认两个教师 checkpoint 已存在
第 1 步：可选但建议，补跑 GCN CCKD soft 主实验
第 2 步：跑 TCN 泛化实验，2 个数据集 × 3 种策略 = 6 次训练
第 3 步：跑 GRU 泛化实验，2 个数据集 × 3 种策略 = 6 次训练
第 4 步：测试所有新 checkpoint
第 5 步：汇总结果表格
```

其中，第 1 步是为了让论文主实验和 soft curriculum 图文叙述完全对齐；第 2、3 步是为了证明方法对额外轻量学生结构也有效。

## 4. 运行前提

下面命令默认你已经有两个教师模型：

```text
checkpoints/teacher/metr_teacher_best.pt
checkpoints/teacher/bay_teacher_best.pt
```

如果没有，需要先训练教师模型。

## 5. 第 1 步：可选但建议，补跑 GCN CCKD soft 主实验

这两组不是泛化实验，而是“最终主实验候选”。如果论文保留 soft curriculum 图和叙述，建议跑。

### 5.1 METR-LA：GCN CCKD soft

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name metr_student_gcn_cckd_v6_soft
```

### 5.2 PEMS-BAY：GCN CCKD soft

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_gcn_cckd_v6_soft
```

### 5.3 测试 GCN CCKD soft

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_gcn_cckd_v6_soft_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_gcn_cckd_v6_soft_eval
```

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_gcn_cckd_v6_soft_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_gcn_cckd_v6_soft_eval
```

## 6. 第 2 步：TCN 泛化实验

TCN 泛化实验用于验证：当学生模型换成时间卷积型轻量结构时，CCKD 是否仍然优于未蒸馏学生和普通 KD。

### 6.1 METR-LA：TCN Student only

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_tcn_baseline_v6
```

### 6.2 METR-LA：TCN Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_tcn_vanilla_kd_v6
```

### 6.3 METR-LA：TCN CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name metr_student_tcn_cckd_v6_soft
```

### 6.4 PEMS-BAY：TCN Student only

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_tcn_baseline_v6
```

### 6.5 PEMS-BAY：TCN Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_tcn_vanilla_kd_v6
```

### 6.6 PEMS-BAY：TCN CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_tcn_cckd_v6_soft
```

## 7. 第 3 步：GRU 泛化实验

GRU 泛化实验用于验证：当学生模型换成循环序列型轻量结构时，CCKD 是否仍然有效。它和 TCN 互补，一个代表时间卷积，一个代表循环序列建模。

### 7.1 METR-LA：GRU Student only

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gru --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_gru_baseline_v6
```

### 7.2 METR-LA：GRU Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gru --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_gru_vanilla_kd_v6
```

### 7.3 METR-LA：GRU CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gru --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name metr_student_gru_cckd_v6_soft
```

### 7.4 PEMS-BAY：GRU Student only

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gru --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_gru_baseline_v6
```

### 7.5 PEMS-BAY：GRU Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gru --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_gru_vanilla_kd_v6
```

### 7.6 PEMS-BAY：GRU CCKD

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gru --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_gru_cckd_v6_soft
```

## 8. 第 4 步：测试 TCN 和 GRU 新 checkpoint

### 8.1 测试 METR-LA 上的 TCN

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_tcn_baseline_v6_best.pt --model_type student --exp_name metr_student_tcn_baseline_v6_eval
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_tcn_vanilla_kd_v6_best.pt --model_type student --exp_name metr_student_tcn_vanilla_kd_v6_eval
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_tcn_cckd_v6_soft_best.pt --model_type student --exp_name metr_student_tcn_cckd_v6_soft_eval
```

### 8.2 测试 PEMS-BAY 上的 TCN

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_tcn_baseline_v6_best.pt --model_type student --exp_name bay_student_tcn_baseline_v6_eval
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_tcn_vanilla_kd_v6_best.pt --model_type student --exp_name bay_student_tcn_vanilla_kd_v6_eval
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_tcn_cckd_v6_soft_best.pt --model_type student --exp_name bay_student_tcn_cckd_v6_soft_eval
```

### 8.3 测试 METR-LA 上的 GRU

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_gru_baseline_v6_best.pt --model_type student --exp_name metr_student_gru_baseline_v6_eval
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_gru_vanilla_kd_v6_best.pt --model_type student --exp_name metr_student_gru_vanilla_kd_v6_eval
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_gru_cckd_v6_soft_best.pt --model_type student --exp_name metr_student_gru_cckd_v6_soft_eval
```

### 8.4 测试 PEMS-BAY 上的 GRU

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_gru_baseline_v6_best.pt --model_type student --exp_name bay_student_gru_baseline_v6_eval
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_gru_vanilla_kd_v6_best.pt --model_type student --exp_name bay_student_gru_vanilla_kd_v6_eval
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_gru_cckd_v6_soft_best.pt --model_type student --exp_name bay_student_gru_cckd_v6_soft_eval
```

## 9. 第 5 步：汇总泛化实验结果表

### 9.1 METR-LA 泛化汇总表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v6_generalization.csv --output_md outputs/reports/metr_v6_generalization.md --run "TCN Student only,student,checkpoints/student/metr_student_tcn_baseline_v6_best.pt" --run "TCN Vanilla KD,student,checkpoints/student/metr_student_tcn_vanilla_kd_v6_best.pt" --run "TCN CCKD,student,checkpoints/student/metr_student_tcn_cckd_v6_soft_best.pt" --run "GRU Student only,student,checkpoints/student/metr_student_gru_baseline_v6_best.pt" --run "GRU Vanilla KD,student,checkpoints/student/metr_student_gru_vanilla_kd_v6_best.pt" --run "GRU CCKD,student,checkpoints/student/metr_student_gru_cckd_v6_soft_best.pt"
```

### 9.2 PEMS-BAY 泛化汇总表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --output_csv outputs/reports/bay_v6_generalization.csv --output_md outputs/reports/bay_v6_generalization.md --run "TCN Student only,student,checkpoints/student/bay_student_tcn_baseline_v6_best.pt" --run "TCN Vanilla KD,student,checkpoints/student/bay_student_tcn_vanilla_kd_v6_best.pt" --run "TCN CCKD,student,checkpoints/student/bay_student_tcn_cckd_v6_soft_best.pt" --run "GRU Student only,student,checkpoints/student/bay_student_gru_baseline_v6_best.pt" --run "GRU Vanilla KD,student,checkpoints/student/bay_student_gru_vanilla_kd_v6_best.pt" --run "GRU CCKD,student,checkpoints/student/bay_student_gru_cckd_v6_soft_best.pt"
```

### 9.3 可选：汇总 GCN soft 主实验对比

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v6_gcn_soft_summary.csv --output_md outputs/reports/metr_v6_gcn_soft_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "GCN Student only,student,checkpoints/student/metr_student_baseline_best.pt" --run "GCN Vanilla KD,student,checkpoints/student/metr_student_vanilla_kd_best.pt" --run "GCN CCKD standard,student,checkpoints/student/metr_student_cckd_v4_best.pt" --run "GCN CCKD soft,student,checkpoints/student/metr_student_gcn_cckd_v6_soft_best.pt"
```

```powershell
python scripts/collect_results.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --output_csv outputs/reports/bay_v6_gcn_soft_summary.csv --output_md outputs/reports/bay_v6_gcn_soft_summary.md --run "Teacher,teacher,checkpoints/teacher/bay_teacher_best.pt" --run "GCN Student only,student,checkpoints/student/bay_student_baseline_v4_best.pt" --run "GCN Vanilla KD,student,checkpoints/student/bay_student_vanilla_kd_v4_best.pt" --run "GCN CCKD standard,student,checkpoints/student/bay_student_cckd_v4_best.pt" --run "GCN CCKD soft,student,checkpoints/student/bay_student_gcn_cckd_v6_soft_best.pt"
```

## 10. 论文中建议怎么放这组结果

建议把这组实验放在主结果和消融实验之后，作为“泛化性分析”或“不同学生结构适配性分析”。

表格可以这样设计：

```text
Dataset | Student | Strategy | MAE | MAPE | RMSE | Gain over Student | Gain over Vanilla KD
METR-LA | LightTCN | Student only | 待填 | 待填 | 待填 | - | -
METR-LA | LightTCN | Vanilla KD | 待填 | 待填 | 待填 | 待填 | -
METR-LA | LightTCN | CCKD | 待填 | 待填 | 待填 | 待填 | 待填
METR-LA | LightGRU | Student only | 待填 | 待填 | 待填 | - | -
METR-LA | LightGRU | Vanilla KD | 待填 | 待填 | 待填 | 待填 | -
METR-LA | LightGRU | CCKD | 待填 | 待填 | 待填 | 待填 | 待填
PEMS-BAY | LightTCN | Student only | 待填 | 待填 | 待填 | - | -
PEMS-BAY | LightTCN | Vanilla KD | 待填 | 待填 | 待填 | 待填 | -
PEMS-BAY | LightTCN | CCKD | 待填 | 待填 | 待填 | 待填 | 待填
PEMS-BAY | LightGRU | Student only | 待填 | 待填 | 待填 | - | -
PEMS-BAY | LightGRU | Vanilla KD | 待填 | 待填 | 待填 | 待填 | -
PEMS-BAY | LightGRU | CCKD | 待填 | 待填 | 待填 | 待填 | 待填
```

建议论文表述：

```text
为验证所提出蒸馏策略对不同轻量学生结构的适配性，本文进一步选取 Lightweight TCN 与 Lightweight GRU 作为额外学生模型。二者分别代表时间卷积建模与循环序列建模范式。实验结果表明，在两个数据集上，CCKD 均能够相较 Student only 和 Vanilla KD 获得更低预测误差，说明所提出的置信度自适应双路径蒸馏与预测步课程加权机制并不依赖于单一学生结构。
```

如果某一组结果没有明显提升，也不要强行写“均显著提升”。可以改成：

```text
整体上，CCKD 在多数设置下优于对应的 Student only 与 Vanilla KD，表明该方法具有一定的跨学生结构适配潜力。
```

