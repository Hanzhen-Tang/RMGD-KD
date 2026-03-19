# 代码数据流详细说明

这份文档专门解释项目里的数据流。

如果你想真正看懂这套代码是怎么从原始数据一路变成教师预测、学生预测、蒸馏损失的，优先看这份文档。

## 1. 整体数据流一句话概括

整个项目的数据流是：

```text
原始 h5 数据
-> 切成 train/val/test 的 npz
-> DataLoader 按 batch 读取
-> 转换成模型输入格式
-> 教师模型 / 学生模型前向传播
-> 输出预测结果和隐藏特征
-> 计算监督损失 + 蒸馏损失
-> 反向传播更新学生模型
```

## 2. 原始数据阶段

### 输入文件

原始公开数据一般是：

- `metr-la.h5`
- `pems-bay.h5`

它们本质上是一个时间序列表：

```text
行: 时间步
列: 传感器节点
值: 速度或流量
```

### 预处理脚本

对应脚本：

- [generate_training_data.py](/C:/Users/86151/Documents/New%20project/generate_training_data.py)

这个脚本会做两件事：

1. 把连续时间序列切成监督学习样本
2. 保存成 `train.npz / val.npz / test.npz`

## 3. 生成样本后的数据维度

在 `generate_training_data.py` 中，得到的样本维度是：

### 输入 `x`

```text
[样本数, 输入长度, 节点数, 输入特征数]
```

也就是：

```text
[B, T_in, N, C_in]
```

默认情况下通常是：

- `T_in = 12`
- `N = 节点数量`
- `C_in = 2`

这里的 2 个特征通常是：

- 第 0 维：交通值本身
- 第 1 维：时间特征

### 标签 `y`

```text
[样本数, 预测长度, 节点数, 输出特征数]
```

也就是：

```text
[B, H, N, C_out]
```

默认：

- `H = 12`

## 4. DataLoader 读取后的数据流

对应代码：

- [utils/data_utils.py](/C:/Users/86151/Documents/New%20project/utils/data_utils.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)

训练时，batch 会先从 `npz` 中读出来。

此时：

- `x.shape = [B, T_in, N, C_in]`
- `y.shape = [B, H, N, C_out]`

然后进入：

- `prepare_batch(...)`

这个函数会做最关键的维度整理。

## 5. prepare_batch 里的维度变化

对应代码：

- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)

### 第一步：输入转置

```python
inputs = torch.as_tensor(x).transpose(1, 3)
```

转置前：

```text
[B, T_in, N, C_in]
```

转置后：

```text
[B, C_in, N, T_in]
```

这是因为 `Conv2d` 需要通道维放在前面。

### 第二步：标签转置

```python
targets = torch.as_tensor(y).transpose(1, 3)
```

转置前：

```text
[B, H, N, C_out]
```

转置后：

```text
[B, C_out, N, H]
```

### 第三步：只取第 0 个输出通道

```python
targets = targets[:, 0, :, :]
```

这样就变成：

```text
[B, N, H]
```

所以后面训练里真正监督的是：

- 所有节点
- 未来 12 步
- 第 0 个交通值通道

## 6. 教师模型的数据流

对应代码：

- [models/teacher_gwnet.py](/C:/Users/86151/Documents/New%20project/models/teacher_gwnet.py)

### 教师输入

教师收到的输入是：

```text
[B, C_in, N, T_in]
```

在训练器里会先补一位：

```python
padded_inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
```

所以教师实际输入变成：

```text
[B, C_in, N, T_in + 1]
```

默认就是：

```text
[B, 2, N, 13]
```

### 为什么要 pad

因为当前 GWNet 的默认感受野是 13。

原始输入只有 12 个时间步，所以先在最左侧补 1 个时间步，保证时序卷积链路长度够用。

### 教师输出

教师最终输出：

```text
[B, H, N, 1]
```

其中：

- `H = 12`

这和原始 GWNet 的做法一致，本质上是把“未来 12 步”放在输出通道维上。

### 教师隐藏特征

教师还会返回中间特征：

```text
[B, C_teacher, N, 1]
```

默认：

- `C_teacher = end_channels = nhid * 16`

如果 `nhid = 32`，那么：

```text
C_teacher = 512
```

## 7. 学生模型的数据流

对应代码：

- [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)

### 学生输入

学生直接接收：

```text
[B, C_in, N, T_in]
```

默认：

```text
[B, 2, N, 12]
```

### 学生内部处理

大致顺序是：

```text
输入投影
-> 时间卷积
-> 多层图卷积
-> 时间压缩
-> 预测头
```

### 学生隐藏特征

学生时间压缩后的隐藏特征：

```text
[B, C_student, N, 1]
```

默认：

- `C_student = hidden_dim = 32`

### 学生预测输出

学生输出同样整理成：

```text
[B, H, N, 1]
```

也就是和教师原始输出保持一致。

## 8. 蒸馏前的统一维度整理

在训练器里，教师和学生输出都会做：

```python
prediction.transpose(1, 3)
```

所以：

### 教师预测

原始：

```text
[B, H, N, 1]
```

转置后：

```text
[B, 1, N, H]
```

### 学生预测

原始：

```text
[B, H, N, 1]
```

转置后：

```text
[B, 1, N, H]
```

### 标签

原始：

```text
[B, N, H]
```

扩一维后：

```text
[B, 1, N, H]
```

所以三者在损失计算时完全一致：

- 教师预测：`[B, 1, N, H]`
- 学生预测：`[B, 1, N, H]`
- 真实值：`[B, 1, N, H]`

## 9. 特征蒸馏的数据流

教师隐藏特征：

```text
[B, 512, N, 1]
```

学生隐藏特征：

```text
[B, 32, N, 1]
```

维度不同，不能直接做 MSE。

所以在训练器中加了一个：

```text
1x1 Conv 特征适配层
```

它会把学生隐藏特征从：

```text
[B, 32, N, 1]
```

映射到：

```text
[B, 512, N, 1]
```

这样就可以和教师隐藏特征对齐。

## 10. 图关系蒸馏的数据流

对应代码：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

在 `compute_relation_matrix(...)` 中：

### 输入

输入隐藏特征：

```text
[B, C, N, 1]
```

### 处理

会先变成：

```text
[B, N, C]
```

然后计算节点两两相似度，得到：

```text
[B, N, N]
```

这个矩阵表示：

- 每个节点与其他节点的关系强度

然后学生和教师的关系矩阵做 MSE。

## 11. 可靠性加权蒸馏的数据流

对应代码：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

### 输入

- 教师预测：`[B, 1, N, H]`
- 真实值：`[B, 1, N, H]`

### 先算误差

```text
teacher_error = |teacher_pred - real|
```

维度：

```text
[B, 1, N, H]
```

### 节点权重

沿 batch、通道、horizon 求平均：

```text
[1, 1, N, 1]
```

### horizon 权重

沿 batch、通道、节点求平均：

```text
[1, 1, 1, H]
```

### 最终可靠性图

把二者组合，得到：

```text
[1, 1, N, H]
```

然后乘到 soft distillation 的逐元素损失上。

## 12. 课程蒸馏的数据流

课程蒸馏不会改变张量值本身，它会生成一个 mask。

mask 维度：

```text
[1, 1, 1, H]
```

前期只开放前面几个 horizon，中期开放更多，后期开放全部 horizon。

这个 mask 会和可靠性图相乘。

所以最终加权图维度仍然是：

```text
[1, 1, N, H]
```

## 13. 总损失的数据流

当前总损失：

```text
L_total = a * L_hard + b * L_soft + c * L_feature + d * L_relation
```

### Hard Loss

输入：

- 学生预测 `[B, 1, N, H]`
- 真实值 `[B, 1, N, H]`

### Soft Loss

输入：

- 学生预测 `[B, 1, N, H]`
- 教师预测 `[B, 1, N, H]`
- 可靠性图 `[1, 1, N, H]`
- 课程 mask `[1, 1, 1, H]`

### Feature Loss

输入：

- 适配后的学生特征 `[B, 512, N, 1]`
- 教师特征 `[B, 512, N, 1]`

### Relation Loss

输入：

- 学生关系矩阵 `[B, N, N]`
- 教师关系矩阵 `[B, N, N]`

## 14. 测试阶段的数据流

对应代码：

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)

测试阶段会做：

1. 读取测试 batch
2. 生成教师或学生预测
3. 拼接所有 batch 的输出
4. 反标准化
5. 逐个 horizon 计算指标
6. 导出曲线图和 csv

如果开启：

- `--plot_adaptive_adj`
  - 会导出教师自适应邻接矩阵热力图

- `--plot_relation`
  - 会导出节点关系热力图

## 15. 为什么这份数据流说明对调参有帮助

因为你后面调参时最容易影响的就是这些地方：

### 1. 输入长度

如果你改：

```text
seq_length_x
```

那么学生模型里的：

- `input_seq_len`

也要对应一致。

### 2. 学生隐藏维度

如果你改：

```text
--student_hidden_dim
```

会影响：

- 学生特征维度
- 参数量
- 推理速度
- 适配层大小

### 3. 教师隐藏维度

如果你改教师的：

```text
nhid
```

就会影响教师隐藏特征通道数，也会影响适配层目标维度。

### 4. 蒸馏权重

如果你改：

- `hard_weight`
- `soft_weight`
- `feature_weight`
- `relation_weight`

本质上是在改不同信息流对学生更新的贡献强度。

## 16. 最推荐你重点看的代码位置

如果你就是想真正看懂数据流，建议按这个顺序看：

1. [generate_training_data.py](/C:/Users/86151/Documents/New%20project/generate_training_data.py)
2. [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
3. [models/teacher_gwnet.py](/C:/Users/86151/Documents/New%20project/models/teacher_gwnet.py)
4. [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)
5. [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
6. [test.py](/C:/Users/86151/Documents/New%20project/test.py)

