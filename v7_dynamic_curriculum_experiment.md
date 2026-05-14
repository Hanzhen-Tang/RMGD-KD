# v7-DDASC 动态软课程实验流程

更新时间：2026-05-14

本文档是 v7 动态难度感知软课程（DDASC）的实验流程说明。v7 只改学生蒸馏阶段，教师 checkpoint 继续复用，不需要重新训练教师。

## 1. 方法说明

目标：

- 保留 v6 已有的 `soft` 课程权重作为稳定下限。
- 根据每轮验证集的 horizon 级信号，动态更新下一轮蒸馏权重。
- 保持旧模式 `standard`、`short`、`wide`、`soft` 完全可用。
- 只有显式设置 `--curriculum_mode dynamic_soft` 时才启用 v7。

每轮验证后统计三个 horizon 级信号：

- `E_h`：学生在第 `h` 个预测步上的验证 MAE。
- `G_h`：教师预测和学生预测在第 `h` 个预测步上的差距。
- `C_h`：教师在第 `h` 个预测步上的置信度。

DDASC 公式：

```text
D_h = 0.45 * E_h + 0.35 * G_h + 0.20 * (1 - C_h)
m_h = b_h + 0.70 * (1 - b_h) * (1 - normalize(D_h))
```

其中：

- `b_h` 是原有 fixed `soft` curriculum 的基础权重。
- `m_h` 是下一轮训练实际使用的动态权重。
- EMA 平滑默认使用 `ema=0.90`。
- 前 `5` 个 epoch 是 warmup，只使用基础 `soft` 权重。
- 强制约束：`m_h >= b_h`、`m_h <= 1`，并且短期权重不低于长期权重。

直观理解：

- 如果某个长期 horizon 当前还很难，动态权重就更接近基础 `soft` 下限。
- 如果某个 horizon 的学生误差小、师生差距小、教师置信度高，说明学生更适合接收该 horizon 的蒸馏压力，权重会被提高。
- 短期 horizon 始终保持高权重，不会被长期 horizon 反超。

## 2. 当前实现状态

相关代码：

- `utils/curriculum.py`：DDASC 调度器、基础 soft 权重、验证 horizon 信号统计。
- `losses/distillation.py`：支持外部传入 `curriculum_override`；不传时旧模式保持原行为。
- `engine.py`：把当前 epoch 的动态 horizon 权重传给蒸馏损失。
- `train_student_kd.py`：新增 `--curriculum_mode dynamic_soft` 和动态课程超参数。
- `models/student_stid.py`：新增 STID-style MLP 学生。
- `models/student_dlinear.py`：新增 DLinear 学生。
- `model.py`：`--student_model` 现在支持 `gcn|tcn|gru|stid|dlinear`。
- `project_handover.md`：记录 v7 当前状态和恢复上下文提示词。

v7 history/checkpoint 新增记录字段：

- `dynamic_curriculum_weight`
- `dynamic_curriculum_base_weight`
- `dynamic_curriculum_next_weight`
- `dynamic_curriculum_difficulty`
- `dynamic_curriculum_readiness`
- `dynamic_horizon_mae`
- `dynamic_teacher_student_gap`
- `dynamic_teacher_confidence`
- `dynamic_valid_count`
- `dynamic_curriculum_config`
- `dynamic_curriculum_final_state`

这些字段主要用于后面画动态权重变化图、分析每个 horizon 的难度变化，并保证实验可复现。

## 3. 主实验训练命令

主试验CCKD GCN
```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 100 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name v7_gcn_metr
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 100 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name v7_gcn_pems
```
```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/v7_gcn_metr_best.pt --model_type student --batch_size 64 --exp_name v7_gcn_metr
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/v7_gcn_pems_best.pt --model_type student --batch_size 64 --exp_name v7_gcn_pems
```
---------------TCN-----------------
```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 100 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name v7_tcn_metr
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 100 --batch_size 64 --student_model tcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name v7_tcn_pems
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/v7_tcn_metr_best.pt --model_type student --batch_size 64 --exp_name v7_tcn_metr
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/v7_tcn_pems_best.pt --model_type student --batch_size 64 --exp_name v7_tcn_pems
```
```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/v7_tcn_metr_best.pt --model_type student --batch_size 64 --exp_name v7_tcn_metr
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/v7_tcn_pems_best.pt --model_type student --batch_size 64 --exp_name v7_tcn_pems
```

## 4. fixed soft 直接对照

如果 v6 fixed soft 结果已经可靠，可以直接复用。若需要补跑，实验名必须和 v7 分开，不要混用。

METR-LA fixed soft：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name metr_student_gcn_cckd_v6_soft
```

PEMS-BAY fixed soft：

```bash
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_gcn_cckd_v6_soft
```

## 5. 检查命令

静态编译检查：

```bash
python -m py_compile train_student_kd.py engine.py losses/distillation.py utils/curriculum.py
```

调度器检查：

```bash
python -c "import numpy as np; from utils.curriculum import DynamicCurriculumScheduler; s=DynamicCurriculumScheduler(12,50); w=s.state_for_epoch(1)['weights']; assert w.shape==(12,); assert np.allclose(w, s.base_weights(1)); st=s.update(np.linspace(2,5,12), np.linspace(1,4,12), np.linspace(0.9,0.2,12)); w=s.state_for_epoch(6)['weights']; b=s.base_weights(6); assert w.shape==(12,); assert np.all(w>=b-1e-9); assert np.all(w<=1+1e-9); assert np.all(w[:-1]>=w[1:]-1e-9); print('scheduler_check_ok')"
```

可选 1 epoch smoke test：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 1 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name smoke_metr_v7_ddasc
```

旧 fixed soft 路径回归检查：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 1 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name smoke_metr_v6_soft_regression
```

## 6. 测试命令

METR-LA v7 学生测试：

```bash
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_gcn_cckd_v7_ddasc_best.pt --model_type student --batch_size 64 --exp_name metr_student_gcn_cckd_v7_ddasc
```

PEMS-BAY v7 学生测试：

```bash
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_gcn_cckd_v7_ddasc_best.pt --model_type student --batch_size 64 --exp_name bay_student_gcn_cckd_v7_ddasc
```

汇总结果示例：

```bash
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v7_ddasc_summary.csv --output_md outputs/reports/metr_v7_ddasc_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "GCN fixed soft,student,checkpoints/student/metr_student_gcn_cckd_v6_soft_best.pt" --run "GCN v7 DDASC,student,checkpoints/student/metr_student_gcn_cckd_v7_ddasc_best.pt"
```

## 7. 消融表模板

| 数据集 | 实验 | 学生模型 | 课程策略 | 置信度 | 趋势蒸馏 | MAE | MAPE | RMSE | 参数量 | 延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| METR-LA | Student only | GCN | 无 | 否 | 否 | 待填 | 待填 | 待填 | 待填 | 待填 |
| METR-LA | Vanilla KD | GCN | 无 | 否 | 否 | 待填 | 待填 | 待填 | 待填 | 待填 |
| METR-LA | CCKD fixed soft | GCN | soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| METR-LA | CCKD v7 DDASC | GCN | dynamic_soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| PEMS-BAY | Student only | GCN | 无 | 否 | 否 | 待填 | 待填 | 待填 | 待填 | 待填 |
| PEMS-BAY | Vanilla KD | GCN | 无 | 否 | 否 | 待填 | 待填 | 待填 | 待填 | 待填 |
| PEMS-BAY | CCKD fixed soft | GCN | soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| PEMS-BAY | CCKD v7 DDASC | GCN | dynamic_soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| METR-LA | CCKD v7 DDASC | STID | dynamic_soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| METR-LA | CCKD v7 DDASC | DLinear | dynamic_soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| PEMS-BAY | CCKD v7 DDASC | STID | dynamic_soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |
| PEMS-BAY | CCKD v7 DDASC | DLinear | dynamic_soft | 是 | 是 | 待填 | 待填 | 待填 | 待填 | 待填 |

可选 DDASC 敏感性实验：

| 数据集 | eta | ema | warmup | MAE | MAPE | RMSE | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| METR-LA | 0.70 | 0.90 | 5 | 待填 | 待填 | 待填 | 默认设置 |
| METR-LA | 0.50 | 0.90 | 5 | 待填 | 待填 | 待填 | 降低动态提升强度 |
| METR-LA | 0.70 | 0.80 | 5 | 待填 | 待填 | 待填 | 更快响应验证信号 |
| METR-LA | 0.70 | 0.90 | 3 | 待填 | 待填 | 待填 | 缩短 warmup |

## 8. STID + DLinear 泛化实验

新增两个轻量学生：

- `stid`：STID-style MLP 学生，用节点身份 embedding 和时间位置 embedding 表示时空身份信息。
- `dlinear`：DLinear 学生，用趋势/季节分解后的线性映射做极简预测。

建议实验优先级：

1. 先跑 GCN v7 主方法，这是论文主线。
2. 再跑 STID 和 DLinear 的 CCKD v7 完整方法，验证 DDASC/CCKD 能否迁移到不同结构的轻量学生。
3. 如果要把泛化实验写进主表或附录，最好再为 STID 和 DLinear 补跑 `Student only` 与 `Vanilla KD` 对照。

METR-LA STID v7：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model stid --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name metr_student_stid_cckd_v7_ddasc
```

PEMS-BAY STID v7：

```bash
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model stid --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name bay_student_stid_cckd_v7_ddasc
```

METR-LA DLinear v7：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model dlinear --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name metr_student_dlinear_cckd_v7_ddasc
```

PEMS-BAY DLinear v7：

```bash
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model dlinear --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --exp_name bay_student_dlinear_cckd_v7_ddasc
```

Student only 对照模板：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model stid --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_stid_baseline_v7
```

Vanilla KD 对照模板：

```bash
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model stid --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_stid_vanilla_kd_v7
```

上面两个对照模板中的 `--student_model`、数据集路径、teacher checkpoint 和 `exp_name` 按 `stid/dlinear`、`METR-LA/PEMS-BAY` 替换即可。

## 9. 记录规范

- v7 实验名和 fixed soft 实验名必须分开。
- 每次实验记录完整命令、checkpoint 路径、设备、时间和随机种子。
- 保留训练 history JSON，因为里面有动态权重变化。
- 画图时至少展示若干 epoch 下的 base weight、当前 dynamic weight、下一轮 dynamic weight。
- 主要对照是 `dynamic_soft` vs fixed `soft`，不要和 v6 TCN/GRU 泛化实验混在同一结论里。
- 如果 v7 只在一个数据集上提升，也要如实写，表述为有针对性的扩展，不要写成绝对普适提升。
