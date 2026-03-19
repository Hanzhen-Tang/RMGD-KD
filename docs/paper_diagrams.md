# 论文用结构图

这份文档专门给论文写作和制图使用。

建议你在论文中直接参考这里的图，再按期刊模板转成 Visio、ProcessOn、PowerPoint 或 draw.io 版本。

如果你后面需要，我也可以继续把这些图改成更偏论文风格的英文标注版。

## 图 1. 整体方法框架图

```mermaid
flowchart LR
    A["Historical Traffic Sequence<br/>X: [B, Tin, N, Cin]"] --> B["Data Loader + Transpose"]
    B --> C["Teacher: GWNet"]
    B --> D["Student: Lightweight GCN"]

    C --> E["Teacher Prediction<br/>Yt: [B, 1, N, H]"]
    C --> F["Teacher Hidden Feature<br/>Ft: [B, Ct, N, 1]"]
    C --> G["Adaptive Graph"]

    D --> H["Student Prediction<br/>Ys: [B, 1, N, H]"]
    D --> I["Student Hidden Feature<br/>Fs: [B, Cs, N, 1]"]

    I --> J["1x1 Feature Adapter"]
    J --> K["Adapted Student Feature"]

    E --> L["Reliability-aware Soft Distillation"]
    H --> L

    F --> M["Relation Distillation"]
    K --> M

    F --> N["Feature Distillation"]
    K --> N

    H --> O["Hard Supervision"]
    P["Ground Truth<br/>Y: [B, 1, N, H]"] --> O

    O --> Q["Total Loss"]
    L --> Q
    M --> Q
    N --> Q

    G --> L
```

### 图注建议

可写成：

`图1  所提 RMGD-KD 方法整体框架。模型以 GWNet 作为教师，以轻量 GCN 作为学生，并通过可靠性加权软蒸馏、特征蒸馏和图关系蒸馏实现知识迁移。`

## 图 2. 教师模型 GWNet 结构图

```mermaid
flowchart TD
    A["Input<br/>[B, Cin, N, Tin]"] --> B["Start 1x1 Conv"]
    B --> C["Dilated Temporal Conv"]
    C --> D["Gate Unit<br/>tanh x sigmoid"]
    D --> E["Graph Convolution"]
    E --> F["Residual Connection"]
    F --> G["BatchNorm"]
    G --> C

    D --> H["Skip Connection"]
    H --> I["End Conv 1"]
    I --> J["End Conv 2"]
    J --> K["Teacher Prediction<br/>[B, H, N, 1]"]
    I --> L["Teacher Hidden Feature<br/>[B, Ct, N, 1]"]
```

### 图注建议

`图2  教师模型 GWNet 结构。教师模型通过扩张时序卷积和图卷积联合建模交通数据中的时空依赖关系。`

## 图 3. 学生模型 Light GCN 结构图

```mermaid
flowchart TD
    A["Input<br/>[B, Cin, N, Tin]"] --> B["Input 1x1 Conv"]
    B --> C["Temporal Conv"]
    C --> D["GraphConv Layer 1"]
    D --> E["GraphConv Layer 2"]
    E --> F["Temporal Readout Conv"]
    F --> G["Dropout"]
    G --> H["Forecast Head"]
    H --> I["Student Prediction<br/>[B, H, N, 1]"]
    F --> J["Student Hidden Feature<br/>[B, Cs, N, 1]"]
```

### 图注建议

`图3  轻量学生模型结构。学生模型采用浅层时空 GCN 设计，以减少参数量和推理开销。`

## 图 4. 蒸馏损失设计图

```mermaid
flowchart TD
    A["Teacher Prediction"] --> B["Reliability Estimation"]
    G["Ground Truth"] --> B
    B --> C["Node Reliability Weight"]
    B --> D["Horizon Reliability Weight"]
    C --> E["Reliability Map"]
    D --> E

    F["Current Epoch"] --> H["Curriculum Mask"]
    E --> I["Weighted Soft Distillation"]
    H --> I

    J["Student Prediction"] --> I
    A --> I

    K["Teacher Hidden Feature"] --> L["Feature Distillation"]
    M["Adapted Student Feature"] --> L

    K --> N["Relation Matrix"]
    M --> O["Relation Matrix"]
    N --> P["Relation Distillation"]
    O --> P

    J --> Q["Hard Loss"]
    G --> Q

    I --> R["Total Loss"]
    L --> R
    P --> R
    Q --> R
```

### 图注建议

`图4  多粒度蒸馏损失设计。所提方法联合考虑硬标签监督、可靠性加权软蒸馏、特征蒸馏和图关系蒸馏，并通过课程机制逐步扩展蒸馏范围。`

## 图 5. 可靠性加权与课程蒸馏示意图

```mermaid
flowchart LR
    A["Teacher Error on Nodes"] --> B["Node Weight"]
    C["Teacher Error on Horizons"] --> D["Horizon Weight"]
    B --> E["Reliability Map"]
    D --> E

    F["Epoch 1 ~ T/3"] --> G["Use Short Horizons"]
    H["Epoch T/3 ~ 2T/3"] --> I["Use Medium Horizons"]
    J["Epoch 2T/3 ~ T"] --> K["Use Full Horizons"]

    G --> L["Curriculum Mask"]
    I --> L
    K --> L

    E --> M["Final Distillation Weight"]
    L --> M
```

### 图注建议

`图5  可靠性加权与课程蒸馏示意图。教师误差越小的位置被赋予更大的蒸馏权重，训练过程中蒸馏范围由短期预测逐步扩展到全 horizon。`

## 图 6. 实验流程图

```mermaid
flowchart LR
    A["Raw Data<br/>METR-LA / PEMS-BAY"] --> B["Generate train/val/test"]
    B --> C["Train Teacher GWNet"]
    C --> D["Evaluate Teacher"]
    C --> E["Train Student with RMGD-KD"]
    E --> F["Evaluate Student"]
    F --> G["Ablation Study"]
    D --> H["Performance Table"]
    F --> H
    G --> I["Ablation Table"]
    D --> J["Visualization"]
    F --> J
    C --> K["Efficiency Analysis"]
    E --> K
```

### 图注建议

`图6  实验流程图。整个实验包含数据处理、教师训练、学生蒸馏训练、模型评估、消融实验、效率分析与可视化。`

## 图 7. 维度流图

```mermaid
flowchart TD
    A["Raw Input<br/>[B, Tin, N, Cin]"] --> B["Transpose"]
    B --> C["Model Input<br/>[B, Cin, N, Tin]"]
    C --> D["Teacher Output<br/>[B, H, N, 1]"]
    C --> E["Student Output<br/>[B, H, N, 1]"]
    D --> F["Transpose"]
    E --> G["Transpose"]
    F --> H["Teacher Prediction for KD<br/>[B, 1, N, H]"]
    G --> I["Student Prediction for KD<br/>[B, 1, N, H]"]
    J["Ground Truth<br/>[B, N, H]"] --> K["Unsqueeze"]
    K --> L["Ground Truth for KD<br/>[B, 1, N, H]"]
```

### 图注建议

`图7  模型训练过程中的关键张量维度流。教师预测、学生预测和真实标签在蒸馏损失计算前统一整理到 [B, 1, N, H] 空间。`

## 推荐放进论文的图

如果篇幅有限，最建议保留这 4 张：

- 图1 整体方法框架图
- 图3 学生模型结构图
- 图4 蒸馏损失设计图
- 图7 维度流图

如果篇幅足够，可以再加：

- 图2 教师模型结构图
- 图5 可靠性与课程蒸馏示意图
- 图6 实验流程图

