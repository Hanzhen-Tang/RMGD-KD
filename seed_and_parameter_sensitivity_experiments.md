# 三随机种子与参数敏感性实验执行说明

本文档对应论文中新增的两类补充实验：

1. 三随机种子稳定性实验：同一组 CCKD 主实验参数，使用 3 个不同随机种子重复训练，并报告测试集均值和标准差。
2. 参数敏感性实验：固定其它训练条件，只改变一个关键超参数，观察 12 步平均 MAE / RMSE 的变化。

新增脚本：

- `scripts/run_seed_experiments.py`
- `scripts/run_parameter_sensitivity_experiments.py`
- `scripts/experiment_common.py`

## 运行前检查

默认使用 METR-LA、GWNet 教师和 GCN 学生：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python scripts/sanity_check.py
```

默认教师权重：

```text
checkpoints/teacher/metr_teacher_best.pt
```

如果教师权重不存在，先训练教师：

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher
```

## 一、三随机种子实验

### 推荐正式命令

该命令会训练 3 个学生模型，默认 seed 为 `42 2024 3407`。

```powershell
python scripts/run_seed_experiments.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --dynamic_curriculum_eta 0.70 --seeds 42 2024 3407 --exp_prefix metr_cckd_3seed
```

### 输出文件

```text
checkpoints/student/metr_cckd_3seed_seed42_best.pt
checkpoints/student/metr_cckd_3seed_seed2024_best.pt
checkpoints/student/metr_cckd_3seed_seed3407_best.pt

logs/seed_experiments/*.log
outputs/reports/seed_experiments/cckd_seed_runs.csv
outputs/reports/seed_experiments/cckd_seed_runs.json
outputs/reports/seed_experiments/cckd_seed_runs.md
```

`cckd_seed_runs.md` 中会给出每个 seed 的测试集指标，以及 `mean ± std`，可直接作为论文稳定性实验统计来源。

### 只检查命令不训练

```powershell
python scripts/run_seed_experiments.py --dry_run --seeds 42 2024 3407 --exp_prefix metr_cckd_3seed
```

### 已经训练过，只重新评估

```powershell
python scripts/run_seed_experiments.py --eval_only --seeds 42 2024 3407 --exp_prefix metr_cckd_3seed
```

## 二、参数敏感性实验

### 默认扫描范围

| 扫描参数 | 候选取值 | 默认主实验值 | 对应训练参数 |
| --- | --- | --- | --- |
| 真实标签监督 / 蒸馏监督权重 | 0.9/0.1, 0.8/0.2, 0.7/0.3, 0.6/0.4, 0.5/0.5 | 0.7/0.3 | `--hard_weight`, `--soft_weight` |
| 趋势蒸馏权重系数 | 0, 0.25, 0.50, 0.75, 1.00 | 0.50 | `--trend_weight` |
| 动态课程增强系数 | 0, 0.30, 0.50, 0.70, 0.90 | 0.70 | `--dynamic_curriculum_eta` |

注意：`dynamic_curriculum_eta` 只有在 `--curriculum_mode dynamic_soft` 下生效。脚本扫描 `dynamic_eta` 时会自动启用 `dynamic_soft`。

### 推荐正式命令

该命令默认训练 15 个模型：3 组参数，每组 5 个取值。

```powershell
python scripts/run_parameter_sensitivity_experiments.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode dynamic_soft --dynamic_curriculum_eta 0.70 --seed 42 --exp_prefix metr_cckd_sensitivity
```

### 只跑某一类参数

只扫描损失权重：

```powershell
python scripts/run_parameter_sensitivity_experiments.py --sweeps loss_weight --epochs 50 --seed 42 --exp_prefix metr_cckd_sensitivity
```

只扫描趋势蒸馏权重：

```powershell
python scripts/run_parameter_sensitivity_experiments.py --sweeps trend_weight --epochs 50 --seed 42 --exp_prefix metr_cckd_sensitivity
```

只扫描动态课程增强系数：

```powershell
python scripts/run_parameter_sensitivity_experiments.py --sweeps dynamic_eta --epochs 50 --seed 42 --exp_prefix metr_cckd_sensitivity
```

### 自定义候选值

```powershell
python scripts/run_parameter_sensitivity_experiments.py --sweeps trend_weight --trend_weights 0.25 0.5 0.75 --epochs 50 --seed 42 --exp_prefix metr_cckd_sensitivity_custom
```

### 输出文件

```text
checkpoints/student/metr_cckd_sensitivity_*_best.pt
logs/parameter_sensitivity/*.log
outputs/reports/parameter_sensitivity/cckd_parameter_sensitivity_runs.csv
outputs/reports/parameter_sensitivity/cckd_parameter_sensitivity_runs.json
outputs/reports/parameter_sensitivity/cckd_parameter_sensitivity_runs.md
outputs/reports/figure_3_7_metr_parameter_sensitivity_source.csv
```

`cckd_parameter_sensitivity_runs.md` 是人读版结果；`cckd_parameter_sensitivity_runs.csv` 是完整机器可读结果；`figure_3_7_metr_parameter_sensitivity_source.csv` 采用论文参数敏感性图的数据源格式，后续可用于替换手工填入的占位结果。

## 三、快速 smoke test

如果只是确认代码能跑通，不想训练完整实验，可以先把 epoch 改成 1：

```powershell
python scripts/run_seed_experiments.py --epochs 1 --seeds 42 --exp_prefix smoke_seed
python scripts/run_parameter_sensitivity_experiments.py --epochs 1 --sweeps trend_weight --trend_weights 0.5 --seed 42 --exp_prefix smoke_sensitivity
```

确认没有问题后，再使用正式 `--epochs 50` 或论文最终采用的 epoch 数补跑完整实验。

## 四、PEMS-BAY 运行方式

如果要在 PEMS-BAY 上补同样实验，只需要替换数据、邻接矩阵、教师权重和实验名前缀：

```powershell
python scripts/run_seed_experiments.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --curriculum_mode dynamic_soft --seeds 42 2024 3407 --exp_prefix bay_cckd_3seed
```

```powershell
python scripts/run_parameter_sensitivity_experiments.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_model gcn --curriculum_mode dynamic_soft --seed 42 --exp_prefix bay_cckd_sensitivity --plot_source_csv outputs/reports/figure_3_7_bay_parameter_sensitivity_source.csv
```

## 五、论文记录建议

三随机种子实验建议报告：

```text
MAE = mean ± std
MAPE = mean ± std
RMSE = mean ± std
```

参数敏感性实验建议报告每个取值的测试集平均 MAE 和 RMSE，并说明实验遵循“单变量改变，其余设置保持主实验一致”的控制变量原则。
