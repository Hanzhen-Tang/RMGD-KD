# RMGD-KD 论文框架建议

## 1. 论文建议题目

你可以从下面几个题目里选一个风格接近的版本：

- `RMGD-KD: Reliability-aware Multi-Granularity Distillation for Lightweight Traffic Forecasting`
- `A Reliability-guided Graph Distillation Framework for Efficient Traffic Prediction`
- `Lightweight Traffic Forecasting via Reliability-aware and Relation-enhanced Knowledge Distillation`

如果你打算投中文期刊，也可以写成：

- `面向轻量化交通预测的可靠性引导多粒度知识蒸馏方法`

## 2. 论文核心定位

不要把文章写成“提出一个全新大模型”，而要写成：

- 教师模型精度高，但部署代价大。
- 学生模型轻量，但性能不足。
- 现有蒸馏方法对交通预测的时空结构利用不充分。
- 因此提出一种面向轻量部署的多粒度蒸馏框架。

这个定位更适合 B2 级别论文，也更容易讲清楚工程价值。

## 3. 可直接写进论文的创新点

### 创新点 1：可靠性加权蒸馏

问题：

- 教师模型并不是在所有节点、所有预测步上都同样可靠。
- 如果学生无差别学习教师，容易把教师在困难区域的噪声也学进去。

方法：

- 根据教师对真实值的预测误差，动态构造节点权重和 horizon 权重。
- 教师误差越小的位置，蒸馏权重越大。

可以写成一句话：

`提出一种可靠性引导蒸馏策略，根据教师在不同节点和不同预测步上的误差自适应分配蒸馏强度。`

### 创新点 2：图关系蒸馏

问题：

- 常规蒸馏主要约束输出值或隐藏特征，缺少对空间拓扑关系知识的显式传递。

方法：

- 使用教师隐藏特征构造节点关系矩阵。
- 让学生隐藏特征对应的节点关系矩阵逼近教师。

可以写成一句话：

`设计图关系蒸馏项，将教师模型学习到的节点间相关性结构显式迁移给轻量学生模型。`

### 创新点 3：多步课程蒸馏

问题：

- 交通预测中短时预测较稳定，长时预测更困难。
- 若一开始就对全部 horizon 强蒸馏，训练会不稳定。

方法：

- 前期只蒸馏前 1/3 预测步。
- 中期蒸馏前 2/3。
- 后期蒸馏全部 horizon。

可以写成一句话：

`提出面向多步预测的课程蒸馏机制，按照由易到难的方式逐步扩展蒸馏范围。`

## 4. 方法总框架

### 教师模型

- 使用 GWNet 作为教师模型
- 输入：历史交通序列
- 输出：未来 12 步交通状态

### 学生模型

- 使用轻量 GCN 作为学生模型
- 保留基本图结构建模能力
- 降低模型参数量和推理开销

### 蒸馏损失

总损失：

```text
L = λ1 * L_hard + λ2 * L_soft + λ3 * L_feat + λ4 * L_rel
```

其中：

- `L_hard`：学生预测与真实值之间的监督损失
- `L_soft`：学生预测与教师预测之间的可靠性加权软蒸馏损失
- `L_feat`：学生和教师隐藏特征之间的特征对齐损失
- `L_rel`：学生与教师节点关系矩阵之间的关系蒸馏损失

### 可靠性权重

可以定义：

```text
W_rel = W_node ⊗ W_horizon
```

其中：

- `W_node` 由教师在各节点上的平均误差生成
- `W_horizon` 由教师在各预测步上的平均误差生成
- 误差越小，权重越大

### 课程蒸馏

训练前期：

- 只蒸馏较短期 horizon

训练后期：

- 逐渐扩展到全部 horizon

## 5. 论文结构建议

### 5.1 引言

写法建议：

- 先写交通预测的重要性
- 再写高精度模型部署成本高
- 再写知识蒸馏适合做轻量化
- 最后指出现有交通蒸馏方法的不足

可以概括为三点不足：

- 忽略教师知识质量差异
- 缺少空间关系知识传递
- 忽略多步预测难度差异

### 5.2 相关工作

建议分 3 节：

- 交通预测模型
- 模型压缩与知识蒸馏
- 图神经网络蒸馏

### 5.3 方法

建议分 4 节：

- 问题定义
- 教师与学生结构
- 可靠性加权蒸馏
- 图关系蒸馏与课程蒸馏

### 5.4 实验

建议分：

- 数据集与评价指标
- 实现细节
- 与基线对比
- 消融实验
- 可视化分析
- 效率分析

## 6. 实验必须做的表格

### 表 1：主结果表

列建议：

- Model
- Params
- Latency
- MAE
- RMSE
- MAPE

行建议：

- GWNet Teacher
- Student GCN
- Student + Vanilla KD
- Student + Reliability KD
- Student + Reliability + Relation KD
- Student + Full RMGD-KD

### 表 2：消融实验

列建议：

- Method Variant
- MAE
- RMSE
- MAPE

行建议：

- Full model
- w/o reliability
- w/o relation
- w/o curriculum
- w/o feature distillation

### 表 3：效率分析

列建议：

- Model
- Params
- Compression Ratio
- Inference Time
- MAE

## 7. 推荐可视化图

至少做下面几张：

- 教师与学生的预测曲线对比图
- 教师自适应邻接矩阵热力图
- 教师与学生的节点关系热力图
- 不同 epoch 的课程蒸馏可见 horizon 变化图
- 精度-参数量 trade-off 图

当前代码中已经能直接生成：

- 教师与学生预测对比图：
  [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)

- 教师自适应邻接矩阵热力图：
  [test.py](/C:/Users/86151/Documents/New%20project/test.py)

- 节点关系热力图：
  [test.py](/C:/Users/86151/Documents/New%20project/test.py)

- 自动结果汇总表：
  [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)

## 8. 结论怎么写

结论不要写太大，建议写成：

- 提出了一种轻量化交通预测蒸馏框架
- 通过可靠性加权、图关系蒸馏和课程蒸馏提升了学生性能
- 在接近教师精度的同时显著降低了参数量和推理开销

## 9. 当前代码中对应的创新实现位置

- 可靠性加权蒸馏：
  [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

- 图关系蒸馏：
  [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

- 课程蒸馏：
  [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

- 训练流程接入：
  [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)

- 实验入口：
  [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

## 10. 最推荐的投稿策略

如果你目标是 B2，不要把战线拉太长。

建议：

1. 只做 METR-LA 和 PEMS-BAY 两个公开数据集
2. 教师固定为 GWNet
3. 学生先固定为轻量 GCN
4. 把消融实验和效率分析做扎实
5. 图表做清楚，叙事聚焦“轻量化部署”

这样成功率会比“方法做得很大但实验不扎实”更高。
