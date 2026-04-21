# CCKD-v5：面向轻量交通预测的可信度感知双路径蒸馏框架

## 1. 项目简介

本项目研究的是**交通预测中的轻量化知识蒸馏**问题。

整体思路是：

- 使用高性能时空图模型 **GWNet** 作为教师模型
- 使用轻量 **GCN** 作为学生模型
- 通过**可信度感知双路径蒸馏（Confidence-Adaptive Dual-Path Distillation）**和**软课程蒸馏（Soft Curriculum）**指导学生学习
- 在尽量保持预测性能的同时，显著降低参数量和推理开销

本文当前最终主线可概括为：

**CCKD-v5 = Confidence-Adaptive Distillation + Soft Curriculum + Lightweight Student**

其中：

- **高可信位置**：学生学习教师的**绝对预测值**
- **低可信位置**：学生学习教师的**变化趋势**
- **Soft Curriculum**：所有 horizon 都参与蒸馏，但前期短期步权重更高，长期步权重随后逐步增强

---

## 2. 当前方法的核心贡献

### 2.1 可信度感知双路径蒸馏

教师知识并不是在所有节点和所有预测步上都同样可靠，因此学生不应使用统一蒸馏方式学习教师。

本项目根据教师预测误差估计连续可信度分数：

- 高可信区域：执行 **Absolute-Value Distillation**
- 低可信区域：执行 **Trend Distillation**

这样做的目的不是“丢弃低可信位置”，而是：

**根据教师知识质量，为不同位置分配不同形式的知识迁移方式。**

### 2.2 软课程蒸馏

多步交通预测中，短期预测通常比长期预测更容易。

因此，本项目引入 **Soft Curriculum Weighting Over Horizons**：

- 所有 horizon 始终参与蒸馏
- 训练前期短期步蒸馏权重较高
- 长期步权重随着训练逐步增加

这比硬屏蔽式 curriculum 更平滑，也更容易适配不同数据集。

---

## 3. 目录结构

```text
New project/
├─ checkpoints/                  # 模型权重
│  ├─ teacher/
│  └─ student/
├─ data/                         # 数据集与图结构
│  ├─ METR-LA/
│  ├─ PEMS-BAY/
│  └─ sensor_graph/
├─ docs/                         # 长期记忆与说明文档
├─ losses/
│  └─ distillation.py            # 蒸馏损失定义
├─ models/
│  ├─ teacher_gwnet.py           # 教师模型
│  ├─ student_gcn.py             # 学生模型
│  └─ layers.py                  # 图卷积等基础层
├─ outputs/
│  ├─ figures/                   # 图像输出
│  ├─ predictions/               # 预测 CSV
│  └─ reports/                   # 结果汇总 / history
├─ scripts/
│  ├─ benchmark_model.py         # 参数量 / 推理速度统计
│  ├─ collect_results.py         # 批量汇总结果
│  ├─ plot_efficiency_tradeoff.py# 精度-效率图
│  └─ sanity_check.py            # 项目自检
├─ utils/
│  └─ plotting.py                # 训练曲线 / 可视化辅助
├─ compare_teacher_student.py    # 教师学生预测对比图
├─ engine.py                     # 训练与验证核心逻辑
├─ generate_training_data.py     # 生成 train/val/test
├─ test.py                       # 测试与画图
├─ train.py                      # 教师训练
├─ train_student_kd.py           # 学生训练（当前主入口）
├─ util.py                       # 数据与工具函数
├─ visualize.py                  # 历史曲线可视化
├─ v5_ai_figure_prompts.md       # 论文图生成提示词
└─ v5_paper_intro_for_ai.md      # 论文初稿介绍文本
```

---

## 4. 环境准备

### 4.1 建议环境

- Windows + PowerShell
- Python 3.9 / 3.10
- PyTorch（建议 CUDA 版本）

### 4.2 安装依赖

```powershell
cd "C:\Users\86151\Documents\New project"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果 `requirements.txt` 中未包含合适版本的 PyTorch，请额外安装：

```powershell
pip install torch
```

### 4.3 简单检查

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python scripts/sanity_check.py
```

---

## 5. 数据准备

### 5.1 原始文件位置

```text
data/sensor_graph/metr-la.h5
data/sensor_graph/pems-bay.h5
data/sensor_graph/adj_mx.pkl
data/sensor_graph/adj_mx_bay.pkl
```

### 5.2 生成 METR-LA 训练数据

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/metr-la.h5 --output_dir data/METR-LA --seq_length_x 12 --seq_length_y 12
```

### 5.3 生成 PEMS-BAY 训练数据

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/pems-bay.h5 --output_dir data/PEMS-BAY --seq_length_x 12 --seq_length_y 12
```

生成后将得到：

- `train.npz`
- `val.npz`
- `test.npz`

---

## 6. 版本说明

本项目经历过多个版本演化：

- `RMGD-KD`：早期多模块蒸馏框架
- `v2 / v3 / v4`：定位问题、收缩方法、引入双路径蒸馏
- **当前建议论文版本：`CCKD-v5`**

说明：

- 论文中建议统一使用 **CCKD-v5**
- 代码内部部分默认字符串或旧实验文件名仍可能带有 `v4`
- 这不影响运行，只要命令中的 `--exp_name` 与 checkpoint 路径保持一致即可

---

## 7. 当前推荐实验流程

当前推荐的总实验主线分为两部分：

1. **METR-LA**
   - 保留原有较稳定设置
   - curriculum 推荐使用 `standard`
2. **PEMS-BAY**
   - 推荐使用更平滑的课程策略
   - curriculum 推荐使用 `soft`

---

## 8. METR-LA 实验流程

### 8.1 训练教师模型

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher
```

### 8.2 测试教师模型

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name metr_teacher_eval
```

### 8.3 训练 Baseline Student

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_baseline_v5
```

### 8.4 测试 Baseline Student

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_baseline_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_baseline_v5_eval
```

### 8.5 训练 Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_vanilla_kd_v5
```

### 8.6 测试 Vanilla KD

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_vanilla_kd_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_vanilla_kd_v5_eval
```

### 8.7 训练 CCKD-v5（METR 推荐 `standard`）

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode standard --exp_name metr_student_cckd_v5
```

### 8.8 测试 CCKD-v5

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_cckd_v5_eval
```

### 8.9 消融：w/o confidence

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode standard --disable_confidence_filter --exp_name metr_student_wo_confidence_v5
```

测试：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_confidence_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_confidence_v5_eval
```

### 8.10 消融：w/o curriculum

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_curriculum --exp_name metr_student_wo_curriculum_v5
```

测试：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_curriculum_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_curriculum_v5_eval
```

---

## 9. PEMS-BAY 实验流程

### 9.1 训练教师模型

```powershell
python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 70 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name bay_teacher
```

### 9.2 测试教师模型

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name bay_teacher_eval
```

### 9.3 训练 Baseline Student

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_baseline_v5
```

### 9.4 测试 Baseline Student

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_baseline_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_baseline_v5_eval
```

### 9.5 训练 Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_vanilla_kd_v5
```

### 9.6 测试 Vanilla KD

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_vanilla_kd_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_vanilla_kd_v5_eval
```

### 9.7 训练 CCKD-v5（PEMS-BAY 推荐 `soft`）

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_cckd_v5_soft
```

### 9.8 测试 CCKD-v5

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v5_soft_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v5_soft_eval
```

### 9.9 消融：w/o confidence（保留 `soft`）

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --disable_confidence_filter --exp_name bay_student_wo_confidence_v5_soft
```

测试：

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_confidence_v5_soft_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_confidence_v5_soft_eval
```

### 9.10 消融：w/o curriculum

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_curriculum --exp_name bay_student_wo_curriculum_v5
```

测试：

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_curriculum_v5_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_curriculum_v5_eval
```

---

## 10. 课程模式说明

当前代码支持：

```powershell
--curriculum_mode standard|short|wide|soft
```

### 模式含义

- `standard`
  - 原始标准课程蒸馏
  - 更适合 `METR-LA`

- `short`
  - 前期只短暂使用 curriculum，随后很快全开
  - 适合更平稳的数据集做轻量尝试

- `wide`
  - 前期一开始就开放更多 horizon
  - 适合测试宽起点 curriculum

- `soft`
  - 所有 horizon 始终参与
  - 长期 horizon 权重前期较低，后期逐步提高
  - **当前推荐用于 `PEMS-BAY`**

---

## 11. 教师学生对比图、效率图和结果汇总

### 11.1 教师学生预测对比图

METR-LA：

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_cckd_v5_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_compare_cckd_v5
```

PEMS-BAY：

```powershell
python compare_teacher_student.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --student_checkpoint checkpoints/student/bay_student_cckd_v5_soft_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name bay_compare_cckd_v5_soft
```

### 11.2 参数量与速度统计

教师：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

学生：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v5_best.pt --model_type student --batch_size 64
```

### 11.3 汇总结果表

METR-LA：

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v5_summary.csv --output_md outputs/reports/metr_v5_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_v5_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_v5_best.pt" --run "CCKD-v5,student,checkpoints/student/metr_student_cckd_v5_best.pt" --run "w/o confidence,student,checkpoints/student/metr_student_wo_confidence_v5_best.pt" --run "w/o curriculum,student,checkpoints/student/metr_student_wo_curriculum_v5_best.pt"
```

PEMS-BAY：

```powershell
python scripts/collect_results.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --output_csv outputs/reports/bay_v5_summary.csv --output_md outputs/reports/bay_v5_summary.md --run "Teacher,teacher,checkpoints/teacher/bay_teacher_best.pt" --run "Baseline,student,checkpoints/student/bay_student_baseline_v5_best.pt" --run "VanillaKD,student,checkpoints/student/bay_student_vanilla_kd_v5_best.pt" --run "CCKD-v5,student,checkpoints/student/bay_student_cckd_v5_soft_best.pt" --run "w/o confidence,student,checkpoints/student/bay_student_wo_confidence_v5_soft_best.pt" --run "w/o curriculum,student,checkpoints/student/bay_student_wo_curriculum_v5_best.pt"
```

### 11.4 精度-效率权衡图

METR-LA：

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_v5_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_v5_efficiency_tradeoff.png --derived_csv outputs/reports/metr_v5_efficiency_tradeoff.csv
```

PEMS-BAY：

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/bay_v5_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/bay_v5_efficiency_tradeoff.png --derived_csv outputs/reports/bay_v5_efficiency_tradeoff.csv
```

---

## 12. 论文写作建议

当前最推荐的论文结构：

1. 引言  
2. 相关工作  
   - 交通预测  
   - 知识蒸馏  
3. 方法  
   - 问题定义  
   - 教师模型 GWNet  
   - 学生模型 GCN  
   - Confidence-Adaptive Distillation  
   - Soft Curriculum  
   - 总损失函数  
4. 实验  
   - 数据集与实验设置  
   - 主结果对比  
   - 消融实验  
   - 精度-效率分析  
   - 可视化分析  
5. 结论  

---

## 13. 图和论文辅助文档

根目录中已经准备好：

- [v5_ai_figure_prompts.md](/C:/Users/86151/Documents/New%20project/v5_ai_figure_prompts.md)
- [v5_paper_intro_for_ai.md](/C:/Users/86151/Documents/New%20project/v5_paper_intro_for_ai.md)

用途：

- `v5_ai_figure_prompts.md`
  - 用于生成总框架图、双路径机制图、soft curriculum 图等
- `v5_paper_intro_for_ai.md`
  - 用于喂给大模型生成论文初稿

长期记忆文档：

- [project_full_memory.md](/C:/Users/86151/Documents/New%20project/docs/project_full_memory.md)

---

## 14. 当前定位总结

这篇工作不是“重新发明一个最强交通预测 backbone”，而是：

**在轻量学生模型场景下，通过更合理的蒸馏机制提升学生性能，并实现更优的精度-效率平衡。**

最重要的两个关键词是：

- **教师知识质量异质性**
- **多步预测学习难度异质性**

对应到方法中分别是：

- `Confidence-Adaptive Distillation`
- `Soft Curriculum`

