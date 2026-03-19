# 调参优先级指南

这份文档的目标是帮你用尽量少的时间，把实验效果调到一个更像论文结果的状态。

核心原则：

- 先保证能稳定跑通
- 再提高学生精度
- 最后再优化压缩率和速度

## 1. 最推荐的调参顺序

不要一上来同时改很多参数。

推荐顺序：

1. `batch_size`
2. 教师模型 `nhid`
3. 学生模型 `student_hidden_dim`
4. 学生模型 `student_layers`
5. 蒸馏权重 `hard_weight / soft_weight / feature_weight / relation_weight`
6. 蒸馏温度 `temperature`
7. 学习率 `learning_rate`

## 2. 第一优先级：先保证能跑

### 2.1 batch_size

如果你显存不够，先调这个。

推荐尝试：

- `64`
- `32`
- `16`

现象：

- 太大：显存爆掉
- 太小：训练慢，波动可能更大

建议：

- GPU 较大先用 `64`
- 不稳定就降到 `32`

### 2.2 learning_rate

默认：

- `0.001`

推荐尝试：

- `0.001`
- `0.0005`
- `0.0002`

现象：

- 太大：loss 抖动很大
- 太小：收敛很慢

建议：

- 如果训练 loss 波动很大，先减小学习率

## 3. 第二优先级：先把教师训好

教师不好，学生蒸馏通常也不会太好。

### 3.1 教师隐藏维度 nhid

对应：

- [train.py](/C:/Users/86151/Documents/New%20project/train.py)

推荐尝试：

- `32`
- `40`
- `64`

现象：

- 大一些通常精度更高
- 但训练更慢，参数更多

建议：

- 如果你只是先做论文，教师可以稍微大一点
- 比如 `nhid=64`

示例：

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --nhid 64 --epochs 50 --batch_size 64 --exp_name metr_teacher_h64
```

## 4. 第三优先级：调学生模型容量

### 4.1 student_hidden_dim

对应：

- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

推荐尝试：

- `16`
- `32`
- `48`
- `64`

规律：

- 越大通常效果越好
- 但参数量和推理时间也会上升

论文建议：

- 至少选 2 到 3 个不同 hidden_dim 做对比

### 4.2 student_layers

推荐尝试：

- `1`
- `2`
- `3`

规律：

- 层数太浅，表达能力不足
- 层数太深，轻量模型优势会下降

建议：

- 首选 `2`
- 如果效果差，再试 `3`

## 5. 第四优先级：调蒸馏损失权重

这是最关键的一组参数。

### 5.1 hard_weight

作用：

- 控制学生直接学习真实值的强度

推荐范围：

- `0.5 ~ 0.8`

### 5.2 soft_weight

作用：

- 控制学生学习教师输出的强度

推荐范围：

- `0.1 ~ 0.4`

### 5.3 feature_weight

作用：

- 控制学生学习教师隐藏特征的强度

推荐范围：

- `0.05 ~ 0.2`

### 5.4 relation_weight

作用：

- 控制学生学习教师空间关系的强度

推荐范围：

- `0.05 ~ 0.2`

## 6. 一套推荐起点参数

这是我建议你最先跑的一组：

```text
hard_weight = 0.6
soft_weight = 0.2
feature_weight = 0.1
relation_weight = 0.1
temperature = 3.0
student_hidden_dim = 32
student_layers = 2
learning_rate = 0.001
batch_size = 64
```

## 7. 第二套建议参数

如果你觉得学生效果不够好，可以尝试：

```text
hard_weight = 0.5
soft_weight = 0.25
feature_weight = 0.1
relation_weight = 0.15
temperature = 3.0
student_hidden_dim = 48
student_layers = 2
learning_rate = 0.0005
batch_size = 32
```

## 8. 第三套建议参数

如果你想更追求压缩率，可以尝试：

```text
hard_weight = 0.65
soft_weight = 0.2
feature_weight = 0.1
relation_weight = 0.05
temperature = 2.0
student_hidden_dim = 16
student_layers = 2
learning_rate = 0.001
batch_size = 64
```

## 9. 如何判断该往哪个方向调

### 情况 1：学生精度远低于教师

优先尝试：

- 增大 `student_hidden_dim`
- 增大 `soft_weight`
- 增大 `relation_weight`
- 试试更好的教师模型

### 情况 2：学生 loss 很不稳定

优先尝试：

- 减小 `learning_rate`
- 减小 `soft_weight`
- 暂时关闭 `reliability` 或 `curriculum` 做对照

### 情况 3：学生速度不够快

优先尝试：

- 减小 `student_hidden_dim`
- 减少 `student_layers`
- 降低 batch_size 做测试

### 情况 4：消融结果不明显

优先尝试：

- 先把完整方法效果调好
- 再固定同一组超参数做消融
- 不要每个消融都换一套参数

## 10. 最值得做的调参实验

如果你时间有限，建议至少做这 4 类：

### 实验 1：学生容量对比

- `hidden_dim = 16`
- `hidden_dim = 32`
- `hidden_dim = 48`

### 实验 2：蒸馏权重对比

- `soft_weight = 0.1`
- `soft_weight = 0.2`
- `soft_weight = 0.3`

### 实验 3：关系蒸馏强度对比

- `relation_weight = 0.0`
- `relation_weight = 0.05`
- `relation_weight = 0.1`
- `relation_weight = 0.2`

### 实验 4：温度对比

- `temperature = 1.0`
- `temperature = 2.0`
- `temperature = 3.0`
- `temperature = 4.0`

## 11. 最推荐的调参流程

```text
1. 先固定教师模型
2. 先固定数据集，只在 METR-LA 上调
3. 先调学生 hidden_dim
4. 再调 soft / feature / relation 权重
5. 再调 temperature
6. 最后把最佳配置迁移到 PEMS-BAY
```

## 12. 和当前脚本配合使用的方法

训练完一个模型后，建议立刻做三件事：

1. 用 [test.py](/C:/Users/86151/Documents/New%20project/test.py) 看指标
2. 用 [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py) 看教师/学生对比图
3. 用 [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py) 汇总到结果表

