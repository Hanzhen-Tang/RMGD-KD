# 代码详细解释文档

这份文档的目的不是教你怎么运行，而是帮助你真正看懂这套代码。

你可以把它理解成：

- 这份工程每个模块分别负责什么
- 教师模型在哪一块
- 学生模型在哪一块
- 蒸馏发生在哪一块
- 创新点落在哪一块
- 训练和测试时数据是怎么流动的

---

## 1. 项目整体目标

这套工程是一个“交通预测 + 知识蒸馏”的实验框架。

整体目标是：

- 用 `GWNet` 作为教师模型
- 用轻量 `GCN` 作为学生模型
- 让学生在保持较低推理开销的同时，尽量接近教师的预测效果
- 在 `METR-LA` 和 `PEMS-BAY` 上完成实验

当前方法名可以写成：

`RMGD-KD`

全称可写为：

`Reliability-aware Multi-Granularity Graph Distillation for Lightweight Traffic Forecasting`

---

## 2. 先从宏观上看：哪些文件是核心

最核心的文件分成 6 类。

### 2.1 模型定义

- [models/teacher_gwnet.py](/C:/Users/86151/Documents/New%20project/models/teacher_gwnet.py)
  作用：定义教师模型 `GWNetTeacher`

- [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)
  作用：定义学生模型 `SimpleGCNStudent`

- [model.py](/C:/Users/86151/Documents/New%20project/model.py)
  作用：统一导出教师和学生模型，方便训练脚本直接导入

### 2.2 蒸馏损失

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
  作用：定义整个蒸馏损失，是创新点最集中的地方

### 2.3 训练控制

- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
  作用：把“模型 + 数据 + 损失 + 优化器”真正串起来

### 2.4 训练入口

- [train.py](/C:/Users/86151/Documents/New%20project/train.py)
  作用：训练教师模型

- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
  作用：训练学生模型并执行蒸馏

### 2.5 测试与可视化

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)
  作用：测试模型、输出指标、画预测图和关系图

- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)
  作用：把教师、学生和真实值画到同一张图上

- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
  作用：统计参数量和推理速度

- [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)
  作用：把多组实验结果汇总成表

- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)
  作用：生成“用了教师多少时间，达到了教师多少性能”的效率图

### 2.6 数据处理

- [generate_training_data.py](/C:/Users/86151/Documents/New%20project/generate_training_data.py)
  作用：把原始 `h5` 交通数据切成 `train/val/test`

- [util.py](/C:/Users/86151/Documents/New%20project/util.py)
  作用：加载数据、加载邻接矩阵、计算指标、归一化/反归一化

---

## 3. 教师模型在哪一块

教师模型在这些文件中：

- 模型本体：
  [models/teacher_gwnet.py](/C:/Users/86151/Documents/New%20project/models/teacher_gwnet.py)

- 训练入口：
  [train.py](/C:/Users/86151/Documents/New%20project/train.py)

- 训练器支持：
  [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
  其中是 `TeacherTrainer`

- 测试入口：
  [test.py](/C:/Users/86151/Documents/New%20project/test.py)

### 教师模型的职责

教师模型负责：

- 提供高精度预测结果
- 提供隐藏特征
- 提供空间关系知识
- 在蒸馏时作为“知识来源”

### 教师模型输出什么

教师模型在蒸馏时并不只是输出最终预测。

它至少提供：

- `prediction`
  用于软蒸馏

- `hidden_state`
  用于特征蒸馏和关系蒸馏

- `adaptive_adj`
  主要用于可视化和解释教师学习到的自适应图结构

---

## 4. 学生模型在哪一块

学生模型在这些文件中：

- 模型本体：
  [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)

- 训练入口：
  [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

- 训练器支持：
  [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
  其中是 `DistillationTrainer`

- 测试入口：
  [test.py](/C:/Users/86151/Documents/New%20project/test.py)

### 学生模型的职责

学生模型负责：

- 接收和教师相同的输入
- 产生自己的预测结果
- 产生自己的隐藏特征
- 在训练中同时学习真实标签和教师知识

### 为什么学生模型更轻量

学生模型设计得更轻量，主要体现在：

- 图卷积层更少
- 隐藏维度更小
- 结构更简单

所以它更适合做：

- 参数量压缩
- 推理速度提升
- 轻量部署

---

## 5. 蒸馏到底发生在哪一块

如果你只问一句“蒸馏在哪”，答案是：

- 核心定义在 [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- 真正调用发生在 [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- 参数开关入口在 [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

### 5.1 蒸馏损失在哪定义

蒸馏总损失类是：

- `RegressionDistillationLoss`

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

### 5.2 蒸馏在哪被真正调用

在 [engine.py](/C:/Users/86151/Documents/New%20project/engine.py) 的 `DistillationTrainer` 中：

- 教师先前向
- 学生再前向
- 学生和教师的预测、特征、关系被送入蒸馏损失
- 返回总损失后再反向传播更新学生

### 5.3 蒸馏训练入口在哪

在 [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py) 中：

- 构建教师模型
- 加载教师权重
- 构建学生模型
- 构建 `DistillationTrainer`
- 循环训练

---

## 6. 创新点在哪一块

这套代码的创新点主要集中在：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

### 6.1 创新点 1：可靠性加权蒸馏

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

关键函数：

- `compute_reliability_map`

它做了什么：

- 先看教师在每个节点、每个预测步上的误差
- 教师误差越小，说明该位置越可靠
- 就给这个位置更高的蒸馏权重

你可以把它理解成：

- 不是所有教师知识都一样值得学
- 要优先学教师更可靠的部分

### 6.2 创新点 2：图关系蒸馏

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

关键函数：

- `compute_relation_matrix`

它做了什么：

- 把教师隐藏特征变成一个节点关系矩阵
- 把学生隐藏特征也变成一个节点关系矩阵
- 让学生关系矩阵尽量逼近教师关系矩阵

你可以把它理解成：

- 不只学预测结果
- 还学教师“怎么看节点之间关系”

### 6.3 创新点 3：多步特征蒸馏

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

关键函数：

- `compute_curriculum_map`

它做了什么：

- 训练前期只蒸馏较短 horizon
- 训练中期扩大到更多 horizon
- 训练后期蒸馏全部 horizon

你可以把它理解成：

- 先学容易的
- 再学难的

### 6.4 创新点是怎么被真正接进训练的

只在损失函数里写还不够，训练器也要配合。

在 [engine.py](/C:/Users/86151/Documents/New%20project/engine.py) 中：

- `DistillationTrainer.set_epoch`
  负责告诉蒸馏损失当前训练到第几个 epoch

- `feature_adapter`
  负责把学生隐藏特征映射到教师特征维度，保证特征蒸馏能对齐

在 [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py) 中：

- `--disable_reliability`
- `--disable_curriculum`
- `--relation_weight`
- `--feature_weight`

这些开关负责让你直接做消融实验

---

## 7. 蒸馏损失由哪几部分组成

总损失形式是：

```text
L_total = a * L_hard + b * L_soft + c * L_feature + d * L_relation
```

### 7.1 `L_hard`

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

含义：

- 学生预测和真实值之间的监督损失

作用：

- 保证学生不是只会模仿教师
- 而是真正对真实交通标签负责

### 7.2 `L_soft`

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

含义：

- 学生预测和教师预测之间的软蒸馏损失

作用：

- 让学生学习教师输出层知识

### 7.3 `L_feature`

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)

含义：

- 学生隐藏特征和教师隐藏特征之间的差异

作用：

- 让学生学到更高层的表示

注意：

- 因为教师和学生通道数可能不同
- 所以在 [engine.py](/C:/Users/86151/Documents/New%20project/engine.py) 中增加了 `feature_adapter`

### 7.4 `L_relation`

位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

含义：

- 学生关系矩阵和教师关系矩阵之间的差异

作用：

- 让学生模仿教师的图结构认知

---

## 8. 训练时数据是怎么流动的

这里写最核心的主链。

### 8.1 数据准备阶段

原始数据先经过：

- [generate_training_data.py](/C:/Users/86151/Documents/New%20project/generate_training_data.py)

生成：

- `train.npz`
- `val.npz`
- `test.npz`

### 8.2 训练时读入数据

训练脚本通过：

- [util.py](/C:/Users/86151/Documents/New%20project/util.py)

完成：

- 数据加载
- 归一化
- dataloader 构建

### 8.3 batch 进入训练器前

在 [engine.py](/C:/Users/86151/Documents/New%20project/engine.py) 中：

- `prepare_batch`

它负责把输入整理成模型需要的维度。

关键维度：

- 原始输入 `x`：`[B, T_in, N, C_in]`
- 转换后输入：`[B, C_in, N, T_in]`

标签：

- 原始标签 `y` 转换后取第 0 通道
- 最终标签：`[B, N, H]`

### 8.4 教师前向

在 [engine.py](/C:/Users/86151/Documents/New%20project/engine.py) 的 `DistillationTrainer` 中：

- 教师输入先 `pad`
- 教师输出 `prediction`
- 教师输出 `hidden_state`

教师预测转成蒸馏使用格式后为：

```text
[B, 1, N, H]
```

### 8.5 学生前向

学生同样在 `DistillationTrainer` 中完成前向。

学生预测整理后也是：

```text
[B, 1, N, H]
```

### 8.6 进入蒸馏损失

送入 [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py) 的主要张量有：

- `student_pred`
- `teacher_pred`
- `real_value`
- `student_feature`
- `teacher_feature`

这一步完成：

- hard loss
- soft loss
- feature loss
- relation loss

### 8.7 反向传播

总损失返回给训练器后：

- 只更新学生模型
- 教师模型被冻结

这也是蒸馏最核心的训练逻辑。

---

## 9. 哪一块负责教师冻结，哪一块负责学生更新

位置：

- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)

在 `DistillationTrainer` 中：

- 教师模型被设为 `eval`
- 教师参数 `requires_grad = False`

这意味着：

- 教师只提供知识
- 不参与更新

而学生和 `feature_adapter` 会被送进优化器：

- 学生参数更新
- 特征适配层也更新

---

## 10. 哪一块负责做消融实验

位置：

- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

关键开关：

- `--disable_reliability`
- `--disable_curriculum`
- `--relation_weight 0.0`
- `--feature_weight 0.0`
- `--soft_weight 0.0`

也就是说，消融实验主要不是改代码，而是改命令行参数。

---

## 11. 哪一块负责论文图

### 11.1 预测曲线图

位置：

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)

### 11.2 教师和学生对比图

位置：

- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)

### 11.3 教师自适应邻接矩阵图

位置：

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)

### 11.4 节点关系热力图

位置：

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)

### 11.5 参数量和时延分析

位置：

- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)

### 11.6 时间-性能达成度图

位置：

- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)

这张图表达的是：

- 学生用了教师多少推理时间
- 学生达到了教师多少百分比的预测性能

---

## 12. 如果你以后只想快速理解代码，建议按什么顺序读

建议顺序如下：

1. [docs/paper_framework.md](/C:/Users/86151/Documents/New%20project/docs/paper_framework.md)
2. [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
3. [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
4. [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
5. [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)
6. [models/teacher_gwnet.py](/C:/Users/86151/Documents/New%20project/models/teacher_gwnet.py)
7. [test.py](/C:/Users/86151/Documents/New%20project/test.py)

这样读的好处是：

- 先知道整体目标
- 再知道训练入口
- 再知道训练如何串起来
- 再深入到蒸馏细节

---

## 13. 一句话总结

如果用一句话概括这份代码：

这是一套以 `GWNet` 为教师、以轻量 `GCN` 为学生、并通过“可靠性加权 + 图关系蒸馏 + 多步课程蒸馏”完成交通预测知识蒸馏的完整实验工程。
