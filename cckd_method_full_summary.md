# CCKD 方法完整说明与换设备交接文档

更新时间：2026-05-25

本文档用于在新设备、新账号或空白项目上下文中快速理解本项目方法。新的模型或协作者应先读本文档，再读代码。本文档描述的是当前主线方法：**CCKD**，即面向轻量交通预测学生模型的置信度自适应知识蒸馏方法；v7 版本进一步加入 **DDASC 动态软课程**。

## 1. 方法一句话

CCKD 的核心不是提出一个新的重型交通预测 backbone，而是提出一种更可靠的教师-学生知识迁移方式：

> 在多步交通预测中，根据教师预测可信度决定学生应学习教师的具体数值还是趋势信息，并根据不同预测步的学习状态动态调整蒸馏权重，从而提升轻量学生模型的预测精度和部署效率。

最终部署时只保留学生模型，教师、置信度估计、双路径蒸馏和课程权重都只在训练阶段使用。

## 2. 任务设定

输入是历史交通序列，输出是未来多个预测步。当前代码通常使用：

- 输入长度：12 个历史时间片。
- 输出长度：12 个未来预测步。
- 数据间隔通常为 5 分钟。

因此未来预测步可以理解为：

```text
H1  = 未来 5 分钟
H2  = 未来 10 分钟
...
H12 = 未来 60 分钟
```

在代码中，蒸馏相关张量通常整理为：

```text
[B, 1, N, H]
```

其中：

- `B`：batch size。
- `N`：交通传感器节点数。
- `H`：预测步数量，通常为 12。

每个样本上都有三类预测/标签：

```text
Teacher prediction: T_h
Student prediction: S_h
Ground truth:       Y_h
```

教师和学生都会输出未来 12 个预测步，所以每个 horizon 上都可以做知识蒸馏。

## 3. 教师与学生

教师模型：

- `GWNet Teacher`
- 参数量较大，预测能力更强。
- 只在训练阶段使用，用于提供软目标和隐含预测知识。
- 推理部署时不使用教师。

新增可选教师：

- `STAEformer Teacher`
- 用于教师侧泛化和新教师消融实验。
- 代码入口为 `--teacher_model staeformer`。
- 不建议在论文主叙事中直接替代 GWNet，除非后续实验结果明确支持。

主学生模型：

- `GCN Student`
- 轻量图卷积学生，是论文主方法的核心部署模型。

泛化学生模型：

- `TCN Student`：轻量时间卷积学生。
- `GRU Student`：轻量循环学生。
- `STID-style MLP Student`：MLP/时空身份学生。
- `DLinear Student`：极简分解线性学生。

这些泛化学生用于证明 CCKD 不只适用于 GCN，也能迁移到不同结构的轻量学生。

## 4. Baseline 与方法命名

论文和实验中建议固定以下命名：

| 名称 | 含义 |
| --- | --- |
| `Teacher / GWNet` | 教师模型，只用于训练阶段提供知识 |
| `STAEformer Teacher` | 新增可选教师，用于教师侧泛化/消融 |
| `Student only / Baseline Student` | 只用真实标签训练学生，不用蒸馏 |
| `Vanilla KD` | 普通知识蒸馏：学生学习真实标签和教师预测 |
| `CCKD` | 本文方法：置信度自适应双路径蒸馏 + 课程权重 |
| `CCKD-v7 / CCKD-DDASC` | 加入动态软课程 DDASC 的 CCKD 版本 |

不要把 Vanilla KD 写成本文方法。Vanilla KD 只是普通蒸馏对照组。

## 5. 总体训练流程

每一轮训练的大流程如下：

```text
历史交通序列 X
   ↓
同时输入教师 GWNet 和学生模型
   ↓
教师输出 T，学生输出 S
   ↓
真实标签 Y 计算 hard loss
   ↓
教师预测 T 与学生预测 S 计算 distillation loss
   ↓
置信度模块决定教师知识是否可信
   ↓
双路径蒸馏模块决定学数值还是学趋势
   ↓
课程模块决定每个预测步蒸馏强度
   ↓
总损失反向传播，只更新学生参数
```

注意：

- 教师参数冻结，不参与更新。
- 学生参数更新。
- best checkpoint 按验证集 `val_mae` 选择。

## 6. 模块一：教师置信度估计

CCKD 不把教师知识简单分成“可用/不可用”，而是计算连续置信度。

首先计算教师预测误差：

```text
e = |T - Y|
```

其中无效真实值会被 mask。当前代码中默认把真实值为 0 的位置视为无效标签：

```text
valid_mask = |Y - 0| > eps
```

然后分别计算：

- 节点级教师误差：不同节点上的平均教师误差。
- 预测步级教师误差：不同 horizon 上的平均教师误差。

再用 inverse min-max normalization 转成置信度：

```text
confidence = 1 - normalize(error)
```

最终置信度为：

```text
c_{i,h} = node_confidence_i * horizon_confidence_h
```

其中：

- `i` 表示节点。
- `h` 表示预测步。

如果使用 `confidence_power`，则：

```text
c_{i,h} = c_{i,h}^{confidence_power}
```

当前默认：

```text
confidence_power = 1.0
```

重要表述：

- 这是连续软置信度，不是硬阈值。
- 不要写成 `c > 0.5` 走数值蒸馏、`c < 0.5` 走趋势蒸馏。
- 实际上两个路径都是软加权参与。

对应代码：

- `losses/distillation.py`
- `compute_confidence_score()`

## 7. 模块二：置信度自适应双路径蒸馏

CCKD 的核心蒸馏模块包含两条路径：

1. 绝对值蒸馏路径。
2. 趋势蒸馏路径。

### 7.1 绝对值蒸馏路径

当教师在某个节点和预测步上更可信时，学生更应该学习教师的具体预测值。

数值蒸馏使用 Smooth L1：

```text
L_abs = SmoothL1(S / tau, T / tau) * tau^2
```

其中：

- `S` 是学生预测。
- `T` 是教师预测。
- `tau` 是蒸馏温度，默认 `temperature=3.0`。

数值蒸馏的权重为：

```text
absolute_mask = confidence_score * curriculum_map
```

也就是：

```text
数值蒸馏权重 = 教师置信度 × 课程权重
```

教师越可信，数值蒸馏越强。

### 7.2 趋势蒸馏路径

当教师具体数值不够可靠时，不直接让学生照抄教师数值，而是让学生学习教师预测变化趋势。

趋势定义为相邻预测步差分：

```text
Student trend: S_h - S_{h-1}
Teacher trend: T_h - T_{h-1}
```

趋势蒸馏同样使用 Smooth L1：

```text
L_trend = SmoothL1(ΔS / tau, ΔT / tau) * tau^2
```

趋势路径使用低置信度加权。代码中对相邻两个 horizon 的置信度取平均：

```text
trend_confidence_h = 0.5 * (c_h + c_{h-1})
```

趋势蒸馏权重为：

```text
trend_mask = (1 - trend_confidence) * valid_mask * trend_curriculum
```

其中：

```text
trend_curriculum = curriculum_h * curriculum_{h-1}
```

直观理解：

- 高置信度：更相信教师具体预测值，强化数值蒸馏。
- 低置信度：不强迫学生照抄数值，而是学习变化趋势。
- 低置信度教师知识没有被丢弃，而是以更稳健的趋势形式迁移。

### 7.3 双路径融合

当前代码中，软蒸馏损失为：

```text
L_soft = (L_abs + λ_trend * L_trend) / (1 + λ_trend)
```

默认：

```text
λ_trend = 0.5
```

如果关闭置信度过滤：

```bash
--disable_confidence_filter
```

则趋势路径被关闭，退化为普通数值蒸馏，用于 Vanilla KD 对照。

对应代码：

- `RegressionDistillationLoss.forward()`

## 8. 模块三：软课程与动态软课程

多步交通预测中，不同预测步难度不同：

- 短期 horizon 更稳定、更容易。
- 长期 horizon 更难、不确定性更高。

因此蒸馏损失不应对所有 horizon 一视同仁。

## 9. fixed soft curriculum

v6 的 fixed soft curriculum 保证所有 horizon 始终参与训练，但长期 horizon 早期权重更低。

基础权重记为：

```text
b_h(t)
```

代码中：

```text
b_h = linspace(1.0, min_weight, H)
```

训练前半段：

```text
min_weight 从 0.4 逐渐增加到 1.0
```

训练后半段：

```text
min_weight = 1.0
```

这表示：

- 早期：H1 权重大，H12 权重较小。
- 后期：所有 horizon 权重逐渐接近 1。

与 hard curriculum 不同，soft curriculum 不关闭任何 horizon。

对应参数：

```bash
--curriculum_mode soft
```

## 10. v7 DDASC 动态软课程

v7 的 DDASC 是在 fixed soft curriculum 上增加验证集反馈机制。

完整名称：

```text
Dynamic Difficulty-Aware Soft Curriculum
```

简称：

```text
DDASC
```

它不是替代 soft curriculum，而是以 fixed soft 权重作为下限，然后根据验证集信号动态提高部分 horizon 的权重。

### 10.1 每轮验证后统计三个 horizon 级信号

每个 epoch 验证后，统计每个预测步：

| 符号 | 含义 |
| --- | --- |
| `E_h` | 学生在 horizon h 上的验证 MAE |
| `G_h` | 教师预测和学生预测在 horizon h 上的差距 |
| `C_h` | 教师在 horizon h 上的置信度 |

其中：

```text
E_h = mean(|S_h - Y_h|)
G_h = mean(|S_h - T_h|)
C_h = inverse_minmax(mean(|T_h - Y_h|))
```

无效标签位置会被 mask。

### 10.2 难度分数

DDASC 难度分数为：

```text
D_h = α E_h + β G_h + γ (1 - C_h)
```

默认参数：

```text
α = 0.45
β = 0.35
γ = 0.20
```

难度高意味着：

- 学生误差大。
- 师生差距大。
- 教师置信度低。

### 10.3 readiness

难度越高，并不是越应该立即加大蒸馏权重。我们的逻辑是：

```text
readiness_h = 1 - normalize(D_h)
```

也就是说：

- 难度低，readiness 高，说明该 horizon 已经更适合接受更强蒸馏。
- 难度高，readiness 低，说明该 horizon 仍需保持基础课程权重，不宜过早加压。

### 10.4 动态权重

最终动态权重为：

```text
m_h = b_h + η (1 - b_h) readiness_h
```

默认：

```text
η = 0.70
```

其中：

- `b_h` 是 fixed soft curriculum 的基础权重。
- `m_h` 是当前 epoch 使用的动态课程权重。
- `(1 - b_h)` 是权重还能向 1 提升的空间。
- `readiness_h` 决定是否值得提升。

因此：

```text
m_h >= b_h
m_h <= 1
```

DDASC 只会在 fixed soft 的基础上提高权重，不会削弱原来的 soft curriculum。

### 10.5 EMA 平滑

验证信号可能波动，所以 DDASC 使用 EMA 平滑：

```text
EMA_new = ema * EMA_old + (1 - ema) * current
```

默认：

```text
ema = 0.90
```

这样动态权重变化更稳定。

### 10.6 warmup

前 5 个 epoch 只使用基础 soft curriculum：

```text
m_h = b_h
```

默认：

```text
warmup = 5
```

原因是训练初期学生预测不稳定，验证信号容易误导调度器。

### 10.7 单调约束

DDASC 强制短期权重不低于长期权重：

```text
H1 >= H2 >= ... >= H12
```

这样可以保证课程学习逻辑不被破坏：

- 短期预测始终作为稳定基础。
- 长期预测逐步获得更强蒸馏，但不会反超短期预测。

对应参数：

```bash
--curriculum_mode dynamic_soft
--dynamic_curriculum_alpha 0.45
--dynamic_curriculum_beta 0.35
--dynamic_curriculum_gamma 0.20
--dynamic_curriculum_eta 0.70
--dynamic_curriculum_ema 0.90
--dynamic_curriculum_warmup 5
```

对应代码：

- `utils/curriculum.py`
- `DynamicCurriculumScheduler`
- `compute_horizon_signal_sums()`
- `reduce_horizon_signal_sums()`

## 11. 两个模块是串行还是并行

置信度双路径蒸馏和动态软课程不是严格串行关系，而是两个并行的加权模块，最后在损失中相乘融合。

可以理解为：

```text
教师预测 / 学生预测 / 真实值
        ↓
  置信度模块产生 c_{i,h}

验证集 horizon 统计信号
        ↓
  动态软课程产生 m_h

最终蒸馏损失
        ↓
  数值蒸馏：c_{i,h} * m_h
  趋势蒸馏：(1 - c_{i,h}) * m_h
```

二者控制的维度不同：

| 模块 | 解决问题 | 粒度 |
| --- | --- | --- |
| 置信度双路径蒸馏 | 教师知识是否可信，应学数值还是趋势 | 节点-预测步级 |
| 动态软课程 | 当前阶段哪个预测步应被强调 | 预测步级 |
| 最终蒸馏损失 | 共同决定蒸馏强度 | 节点-预测步级 |

## 12. 总损失

当前代码中的总损失为：

```text
L_total =
λ_hard L_hard
+ λ_soft L_soft
+ λ_feature L_feature
+ λ_relation L_relation
```

其中主实验通常使用：

```text
λ_hard = 0.7
λ_soft = 0.3
λ_trend = 0.5
λ_feature = 0.0
λ_relation = 0.0
temperature = 3.0
confidence_power = 1.0
```

当前主线实验实际上主要使用：

- hard label supervision。
- confidence-adaptive absolute distillation。
- low-confidence trend distillation。
- soft / dynamic soft curriculum。

feature alignment 和 relation alignment 代码保留，但主实验默认关闭：

```text
feature_weight = 0.0
relation_weight = 0.0
```

## 13. 如何退化到不同对照组

### 13.1 Student-only baseline

只训练学生，不使用蒸馏：

```bash
--hard_weight 1.0
--soft_weight 0.0
--trend_weight 0.0
--disable_confidence_filter
--disable_curriculum
```

### 13.2 Vanilla KD

普通蒸馏：学生学习真实标签和教师预测，但不使用置信度、趋势和课程：

```bash
--hard_weight 0.7
--soft_weight 0.3
--trend_weight 0.0
--disable_confidence_filter
--disable_curriculum
```

### 13.3 CCKD fixed soft

使用完整 CCKD，但课程为 fixed soft：

```bash
--hard_weight 0.7
--soft_weight 0.3
--trend_weight 0.5
--curriculum_mode soft
```

### 13.4 CCKD-v7 / DDASC

使用完整 CCKD，并启用动态软课程：

```bash
--hard_weight 0.7
--soft_weight 0.3
--trend_weight 0.5
--curriculum_mode dynamic_soft
```

## 14. 代码文件对应关系

核心文件：

| 文件 | 作用 |
| --- | --- |
| `train_student_kd.py` | 学生蒸馏训练入口 |
| `train.py` | 教师训练入口，支持 `gwnet` 和 `staeformer` |
| `engine.py` | 训练器，负责 teacher/student forward 和 batch 训练 |
| `losses/distillation.py` | CCKD 损失，包括置信度、双路径蒸馏、课程权重 |
| `utils/curriculum.py` | v7 DDASC 动态软课程调度器 |
| `model.py` | 统一构建教师/学生模型 |
| `models/teacher_gwnet.py` | GWNet 教师 |
| `models/teacher_staeformer.py` | STAEformer 教师 |
| `models/student_gcn.py` | GCN 学生 |
| `models/student_tcn.py` | TCN 学生 |
| `models/student_gru.py` | GRU 学生 |
| `models/student_stid.py` | STID-style MLP 学生 |
| `models/student_dlinear.py` | DLinear 学生 |
| `test.py` | 测试 checkpoint，输出 horizon 级和平均指标 |
| `scripts/collect_results.py` | 汇总多个模型测试结果 |

## 15. 当前支持的学生模型

当前训练入口支持：

```bash
--student_model gcn
--student_model tcn
--student_model gru
--student_model stid
--student_model dlinear
```

模型定位：

| 学生模型 | 结构类型 | 用途 |
| --- | --- | --- |
| GCN | 轻量图卷积 | 主方法学生 |
| TCN | 轻量时间卷积 | 泛化实验 |
| GRU | 轻量循环模型 | 泛化实验 |
| STID | MLP/时空身份学生 | 泛化实验 |
| DLinear | 极简线性学生 | 泛化边界实验 |

## 16. 当前支持的教师模型

当前教师训练入口支持：

```bash
--teacher_model gwnet
--teacher_model staeformer
```

教师模型定位：

| 教师模型 | 结构类型 | 用途 |
| --- | --- | --- |
| GWNet | 图时空卷积教师 | 论文主线教师 |
| STAEformer | 时空自注意力教师 | 教师侧泛化/新教师消融 |

旧 GWNet checkpoint 中通常没有 `teacher_model` 字段，当前代码会默认按 `gwnet` 读取，因此旧实验不受影响。STAEformer checkpoint 会保存 `teacher_model=staeformer` 以及输入嵌入、时间嵌入、自适应嵌入、attention 层数等结构参数，学生蒸馏和测试脚本会自动根据 checkpoint 重建教师。

STAEformer 新教师实验建议只作为补充表：

```text
STAEformer Teacher
GCN Vanilla KD
GCN w/o confidence-adaptive distillation
GCN w/o soft curriculum
GCN CCKD
```

完整训练与测试命令见：

```text
staeformer_teacher_workflow.md
```

## 17. 论文表述建议

可以写：

> 本文提出一种面向轻量交通预测模型的置信度自适应课程蒸馏方法 CCKD。该方法首先根据教师预测误差构造连续置信度，用于自适应调节数值蒸馏与趋势蒸馏的贡献；随后结合多步预测难度差异，通过动态软课程机制根据学生误差、师生预测差距和教师置信度调整不同预测步的蒸馏权重，从而实现更稳定、更细粒度的教师知识迁移。

不要写：

- 不要写教师知识被硬阈值分成高可信和低可信两类。
- 不要写 `c > 0.5` 走数值路径、`c < 0.5` 走趋势路径。
- 不要写部署时需要教师。
- 不要把 CCKD 写成新的交通预测 backbone。

推荐强调：

- 教师知识可靠性是异质的。
- 不同预测步的学习难度是异质的。
- CCKD 同时建模教师可靠性和 horizon 难度。
- 推理阶段只保留轻量学生，具备部署优势。

## 18. 实验设计建议

主实验：

- Teacher：GWNet。
- Main student：GCN。
- 对比：Teacher、GCN Baseline、GCN Vanilla KD、GCN CCKD。

泛化实验：

- TCN、GRU、STID、DLinear。
- 每个学生最好都有 Baseline、Vanilla KD、CCKD。

教师侧泛化实验：

- 新增 `STAEformer Teacher -> GCN Student`。
- 对比 Vanilla KD、w/o confidence、w/o curriculum、full CCKD。
- 不要和 GWNet Teacher 的结果混在同一张表里，除非表头明确标出 teacher。
- 使用 `staeformer_teacher_workflow.md` 中的 `_wf` 实验名可以避免覆盖现有实验。

训练设置要统一：

- 相同 epoch。
- 相同 batch size。
- 相同 learning rate。
- 相同 seed。
- 相同 teacher checkpoint。
- 相同 best checkpoint 选择规则。

建议主表中不要混用不同 epoch 的结果。否则提升无法明确归因于方法本身。

## 19. 推荐图表

实验部分最建议放：

1. Horizon-wise MAE 曲线：展示 H1-H12 上 CCKD 是否改善长期预测。
2. DDASC 动态权重演化图：展示不同 epoch 下每个 horizon 的权重变化。
3. 预测曲线对比图：真实值 vs Baseline vs Vanilla KD vs CCKD。
4. 精度-效率权衡图：横轴参数量或推理时间，纵轴 MAE。

训练 loss 曲线不建议放正文，可放附录或不放。

## 20. 新模型接手时的最短提示词

如果换设备或换模型后上下文丢失，可以直接给新模型这段话：

```text
请先阅读 cckd_method_full_summary.md。这个项目是交通预测轻量学生蒸馏论文，方法名为 CCKD。主线教师是 GWNet，新增可选教师是 STAEformer；教师只在训练阶段使用。学生包括 GCN、TCN、GRU、STID、DLinear，部署时只保留学生。CCKD 的核心是置信度自适应双路径蒸馏：教师预测可信时加强数值蒸馏，教师预测低可信时加强趋势蒸馏；二者是连续软加权，不是硬阈值分支。v7 版本加入 DDASC 动态软课程，以 fixed soft curriculum 为下限，根据每轮验证后的学生 horizon MAE、师生预测差距和教师 horizon 置信度更新下一轮 horizon 蒸馏权重。STAEformer Teacher 的训练和新教师 GCN 消融命令在 staeformer_teacher_workflow.md。实现主要在 model.py、models/teacher_staeformer.py、losses/distillation.py、utils/curriculum.py、engine.py 和 train_student_kd.py。
```
