# 可信度有效性分析实验说明

本实验用于验证 CCKD 中构建的可信度是否能够反映教师模型监督信号的可靠性。

实验描述可写为：

> 为避免使用同一批误差同时构造可信度和验证可信度，本文首先在训练集上统计教师模型在不同节点—预测步位置上的平均预测误差，并据此构造节点—预测步可信度。随后，按照训练集可信度得分的分位数，将节点—预测步位置划分为低可信、中可信和高可信三个区间，并在测试集上统计各区间对应位置的教师预测 MAE。

## 实验口径

- 分析对象：主教师模型。
- 默认数据集：`METR-LA` 和 `PEMS-BAY`。
- 默认可信度构造划分：`train`。
- 默认验证划分：`test`。
- 统计单元：节点-预测步。
- 可信度构造：在训练集上统计教师在节点维度和预测步维度上的误差，再按当前 CCKD 代码逻辑进行反向归一化，得到节点可信度和预测步可信度，两者相乘得到节点-预测步可信度。
- 分组方式：按训练集可信度从低到高排序，划分为 `0%-33%`、`33%-66%`、`66%-100%` 三个区间。
- 指标：在测试集上统计每个固定区间内教师预测的平均绝对误差，即测试集教师 MAE。

## 正式运行命令

默认会同时分析 METR-LA 与 PEMS-BAY：

```powershell
python scripts/analyze_confidence_effectiveness.py --device cuda:0 --batch_size 64 --confidence_split train --eval_split test
```

等价的显式写法如下：

```powershell
python scripts/analyze_confidence_effectiveness.py --device cuda:0 --batch_size 64 --confidence_split train --eval_split test --run METR-LA,data/METR-LA,data/sensor_graph/adj_mx.pkl,checkpoints/teacher/metr_teacher_best.pt --run PEMS-BAY,data/PEMS-BAY,data/sensor_graph/adj_mx_bay.pkl,checkpoints/teacher/bay_teacher_best.pt
```

## 输出文件

```text
outputs/reports/confidence_effectiveness/confidence_effectiveness_summary.csv
outputs/reports/confidence_effectiveness/confidence_effectiveness_summary.json
outputs/reports/confidence_effectiveness/confidence_effectiveness_summary.md
outputs/reports/confidence_effectiveness/confidence_effectiveness_node_horizon.csv
```

其中：

- `confidence_effectiveness_summary.md` 是论文表格的人读版。
- `confidence_effectiveness_summary.csv` 是按数据集和可信度区间汇总的机器可读结果。
- `confidence_effectiveness_node_horizon.csv` 保存每个节点-预测步位置的可信度、教师 MAE 和所属区间，便于后续画分箱曲线或散点图。

## 快速 smoke test

如果只想检查脚本能否跑通，可以限制只跑 1 个 batch：

```powershell
python scripts/analyze_confidence_effectiveness.py --device cuda:0 --batch_size 64 --confidence_split train --eval_split test --max_batches 1
```

如果只想检查默认路径和命令解析，不加载数据：

```powershell
python scripts/analyze_confidence_effectiveness.py --dry_run
```

## 论文表格格式

脚本生成的 Markdown 会直接给出如下格式：

| 可信度区间 | 区间划分依据 | METR-LA 测试集教师 MAE | PEMS-BAY 测试集教师 MAE |
| --- | --- | --- | --- |
| 低可信 | 训练集可信度 0%-33% | 由脚本计算 | 由脚本计算 |
| 中可信 | 训练集可信度 33%-66% | 由脚本计算 | 由脚本计算 |
| 高可信 | 训练集可信度 66%-100% | 由脚本计算 | 由脚本计算 |

如果可信度有效，通常应看到从低可信到高可信，测试集教师 MAE 呈下降趋势。这一结果说明训练集误差构造得到的可信度分组能够迁移到未参与可信度构造的测试样本上，从而作为“可信度能够刻画教师监督信号可靠性”的直接证据。
