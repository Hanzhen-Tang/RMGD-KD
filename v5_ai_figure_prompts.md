# v4 图示生成提示词（中文解释 + 英文生成词）

## 使用说明

这份文档用于生成 **CCKD-v5** 论文插图。  
要求如下：

- 图的说明、用途、摆放建议全部使用中文
- 真正喂给 AI 绘图工具的提示词使用英文
- 图中出现的文字也尽量使用英文
- 风格建议统一为：
  - 学术论文风格
  - 扁平化矢量示意图
  - 白底
  - 蓝、橙、灰为主色
  - 模块边界清晰
  - 线条简洁
  - 适合论文排版

---

## 图 1：整体方法框架图

### 中文解释

这张图用于展示整篇方法的总框架。  
应包含：

- 输入历史交通序列
- 教师模型 `GWNet`
- 学生模型轻量 `GCN`
- 教师输出与学生输出
- 可信度估计模块
- 高可信路径：绝对值蒸馏
- 低可信路径：趋势蒸馏
- soft curriculum 权重调度
- 真实标签监督
- 最终学生预测输出

这张图适合放在方法章节开头，作为“本文方法总览图”。

### 英文提示词

```text
Academic vector diagram, white background, clean conference-paper style, overall framework of a traffic forecasting knowledge distillation method. Show historical traffic sequence input on the left, a GWNet teacher branch on the top, a lightweight GCN student branch on the bottom. From teacher outputs to student outputs, add a confidence estimation module. Split distillation into two paths: high-confidence path with absolute-value distillation, low-confidence path with trend distillation. Add a soft curriculum weighting module over forecasting horizons. Also show hard supervision from ground-truth labels to the student. Final output is multi-step traffic forecasting. Use simple arrows, blue and orange accents, English labels only, neat and minimal layout.
更新
Academic vector infographic, white background, clean journal-paper style, publication-ready architecture diagram for a traffic forecasting distillation framework. Show historical traffic sequence input on the left feeding both a GWNet teacher branch and a lightweight GCN student branch. The teacher branch produces teacher predictions, and the student branch produces student predictions. Add a confidence estimation module that takes teacher predictions and ground-truth labels as inputs. Use this confidence module to route knowledge into two branches: high-confidence branch for absolute-value distillation and low-confidence branch for trend distillation. Add a soft curriculum weighting module over forecasting horizons, explicitly showing that short-term horizons have larger weights early in training while long-term horizons gradually receive larger weights later, without fully removing any horizon. Ground-truth labels should also provide hard supervision directly to the student branch. Combine hard supervision, absolute distillation, and trend distillation into a total loss block. The final output on the right is final student multi-step traffic forecasting. Use simple arrows, modular blocks, blue-gray-orange academic color palette, English labels only, balanced spacing, minimal and professional layout.
```

---

## 图 2：可信度感知双路径蒸馏图

### 中文解释

这张图专门解释 `CCKD-v5` 的核心机制。  
应突出下面这条逻辑链：

- 先根据教师误差构建连续可信度分数
- 高可信位置采用绝对值蒸馏
- 低可信位置采用趋势蒸馏

图中应明确区分：

- confidence score
- absolute-value distillation
- trend distillation

这张图最好放在方法章节中“蒸馏机制”部分。

### 英文提示词

```text
Academic vector figure, white background, clean and elegant. Illustrate confidence-adaptive dual-path distillation. Start with teacher prediction error, then generate a continuous confidence score. Split into two branches: high-confidence branch performs absolute-value distillation, low-confidence branch performs trend distillation. Show student prediction receiving guidance from both branches. Add concise English labels such as Teacher Error, Confidence Score, High Confidence, Low Confidence, Absolute Distillation, Trend Distillation. Flat infographic style, blue-orange-gray palette, highly readable for a journal paper.
```

---

## 图 3：soft curriculum 机制图

### 中文解释

这张图用来说明为什么本文采用的是 **soft curriculum**，而不是硬课程屏蔽。

图中应体现：

- 横轴为训练阶段或 epoch progression
- 纵向表示不同预测步 horizon
- 所有 horizon 始终存在
- 短期 horizon 前期权重大
- 长期 horizon 前期权重小
- 随着训练推进，长期 horizon 权重逐渐增大

这张图很关键，因为它能让审稿人看明白：
- 本文不是简单关掉后面 horizon
- 而是平滑地调整蒸馏强度

### 英文提示词

```text
Academic schematic figure, white background, soft curriculum mechanism for multi-horizon distillation. Horizontal axis is training progress or epochs, vertical axis is forecasting horizons from short-term to long-term. Show that all horizons remain active, but short-term horizons start with higher weights while long-term horizons start with lower weights and gradually increase over time. Use a smooth color gradient or weighted bars. Add English labels such as Training Progress, Horizon Weight, Short-term, Long-term, Soft Curriculum. Minimal, clean, publication-ready infographic style.
```

---

## 图 4：教师模型结构图

### 中文解释

这张图用于介绍教师模型 `GWNet`。  
应简洁说明：

- 输入序列
- 时序卷积模块
- 图卷积/自适应邻接模块
- skip connection
- 输出预测

不需要把每个卷积层都画得过细，重点是让读者理解教师模型为什么强。

### 英文提示词

```text
Academic vector architecture diagram of the GWNet teacher model for traffic forecasting. Show input sequence, temporal convolution blocks, graph convolution or adaptive adjacency module, skip connections, and final forecasting head. Keep it simplified, structured, and publication-ready. Use English labels only, white background, blue-gray-orange palette, clean neural network diagram style.
```

---

## 图 5：学生模型结构图

### 中文解释

这张图用于说明轻量学生模型 `GCN`。  
应体现：

- input projection
- temporal convolution
- multiple graph convolution layers
- temporal readout
- forecasting head

要突出“轻量”特点，说明它比教师更简单。

### 英文提示词

```text
Academic vector architecture diagram of a lightweight GCN student model for traffic forecasting. Show input projection, temporal convolution, stacked graph convolution layers, temporal readout, and forecasting head. Emphasize compact and lightweight structure compared with the teacher. English labels only, white background, minimal and clean journal-style design.
```

---

## 图 6：教师与学生预测对比图

### 中文解释

这张图不一定需要 AI 生成，通常建议用真实数据画。  
如果需要 AI 帮你做排版示意，可以生成一个“论文图风格模板”。

图里应包含：

- Real
- Teacher Prediction
- Student Prediction

适合用来说明学生是否接近教师，以及是否跟住真实趋势。

### 英文提示词

```text
Publication-style line chart template for traffic forecasting comparison. White background, subtle grid, elegant academic style. Three lines only: Real, Teacher Prediction, Student Prediction. Real line in dark gray, teacher in blue, student in orange. Include a legend, x-axis as Time Step, y-axis as Traffic Speed or Flow. Clean and professional look suitable for a journal paper.
```

---

## 图 7：精度-效率权衡图

### 中文解释

这张图非常重要，用来说明：

- 教师模型更准但更重
- 学生模型更快更轻
- `CCKD-v4` 在学生方法里取得更好的平衡

建议：

- 横轴用参数量或延迟
- 纵轴用 `MAE`
- 图中标出：
  - Teacher
  - Baseline
  - Vanilla KD
  - CCKD-v4

### 英文提示词

```text
Academic scatter plot for accuracy-efficiency tradeoff in traffic forecasting. White background, clean journal style. X-axis is model complexity or latency, Y-axis is MAE. Plot four labeled points: Teacher, Baseline Student, Vanilla KD, CCKD-v4. Teacher should appear as high accuracy but high cost, student methods as lower cost, and CCKD-v4 as the best tradeoff among student models. Use blue, orange, gray palette, minimal and elegant design.
```

---

## 图 8：消融实验结果图

### 中文解释

这张图用于直观展示：

- Full CCKD-v4
- w/o confidence
- w/o curriculum
- Vanilla KD
- Baseline

可以做成柱状图或横向条形图。  
适合放在实验章节“消融分析”部分。

### 英文提示词

```text
Academic bar chart template for ablation study of a traffic forecasting method. White background, clean journal style. Compare Full CCKD-v4, w/o confidence, w/o curriculum, Vanilla KD, and Baseline Student. Y-axis is MAE or RMSE, x-axis lists the methods. Use a clear highlight color for Full CCKD-v4 and muted colors for ablations. Minimal, elegant, publication-ready figure.
```

---

## 图 9：章节排版建议

### 中文解释

如果版面有限，建议至少保留下面 5 张：

1. 整体方法框架图
2. 可信度感知双路径蒸馏图
3. soft curriculum 机制图
4. 教师学生预测对比图
5. 精度-效率权衡图

如果版面更充足，再增加：

6. 教师结构图
7. 学生结构图
8. 消融实验图

---

## 最终提醒

这份图示文档已经按 **v4 + soft curriculum** 主版本重写。  
因此：

- 不要再按旧版 `v3` 的二值筛选方式画图
- 不要再画“低可信位置不参与蒸馏”的旧逻辑
- 现在的核心表达必须是：
  - 连续可信度
  - 双路径蒸馏
  - soft curriculum
