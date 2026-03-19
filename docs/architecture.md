# GWNet-KD 模块结构图

## 1. 整体模块结构

```mermaid
flowchart LR
    A["原始数据<br/>METR-LA / PEMS-BAY"] --> B["generate_training_data.py<br/>切分 train/val/test"]
    B --> C["data/METR-LA 或 data/PEMS-BAY<br/>train.npz / val.npz / test.npz"]
    C --> D["train.py<br/>训练教师 GWNet"]
    C --> E["train_student_kd.py<br/>训练学生 GCN + 蒸馏"]
    F["data/sensor_graph/adj_mx.pkl"] --> D
    F --> E
    D --> G["教师检查点<br/>checkpoints/teacher/*.pt"]
    G --> E
    G --> H["test.py<br/>教师测试/可视化"]
    E --> I["学生检查点<br/>checkpoints/student/*.pt"]
    I --> J["test.py<br/>学生测试/可视化"]
    D --> K["outputs/reports + outputs/figures"]
    E --> K
    H --> K
    J --> K
```

## 2. 教师模型内部结构

```mermaid
flowchart TD
    A["输入 X<br/>[B, C, N, T]"] --> B["起始 1x1 卷积"]
    B --> C["扩张时序卷积块"]
    C --> D["门控机制 tanh x sigmoid"]
    D --> E["图卷积 / 自适应邻接矩阵"]
    E --> F["残差连接 + BN"]
    F --> C
    F --> G["Skip 累积"]
    G --> H["end_conv_1"]
    H --> I["end_conv_2"]
    I --> J["预测输出<br/>[B, H, N, 1]"]
```

## 3. 学生模型与蒸馏数据流

```mermaid
flowchart TD
    A["输入 X<br/>[B, C, N, T]"] --> B["学生 Temporal Conv"]
    B --> C["学生多层 GCN"]
    C --> D["时间压缩 Conv"]
    D --> E["学生预测<br/>[B, H, N, 1]"]

    A --> F["教师 GWNet"]
    F --> G["教师预测<br/>[B, H, N, 1]"]
    F --> H["教师隐藏特征"]
    D --> I["学生隐藏特征"]
    I --> J["1x1 特征适配层"]

    E --> K["硬标签损失 Hard Loss<br/>与真实值比较"]
    G --> L["软目标损失 Soft Loss<br/>与学生预测比较"]
    J --> M["特征蒸馏损失 Feature Loss<br/>与教师特征比较"]
    H --> M
    K --> N["总蒸馏损失"]
    L --> N
    M --> N
```

## 4. 关键张量维度

- 输入 `x` 原始维度：`[B, T_in, N, C_in]`
- 送入模型前转置后：`[B, C_in, N, T_in]`
- 教师输出：`[B, H_out, N, 1]`
- 教师输出转置后：`[B, 1, N, H_out]`
- 真实标签：`[B, N, H_out]`，扩维后为 `[B, 1, N, H_out]`
- 学生输出：`[B, H_out, N, 1]`
- 蒸馏比较时统一在 `[B, 1, N, H_out]` 空间下进行

