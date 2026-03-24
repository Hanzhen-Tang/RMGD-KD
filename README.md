# RMGD-KD: 面向轻量化交通预测的可靠性引导多粒度知识蒸馏

本项目基于 `GWNet` 教师模型和轻量 `GCN` 学生模型，支持在 `METR-LA` 和 `PEMS-BAY` 数据集上完成交通预测知识蒸馏实验。

当前版本已经实现了下面这些可直接用于论文实验的内容：

- 教师模型训练：`GWNet`
- 学生模型训练：轻量 `GCN`
- 普通监督训练
- 可靠性加权蒸馏
- 图关系蒸馏
- 多步课程蒸馏
- 测试集指标评估
- 预测曲线可视化
- 教师自适应邻接矩阵热力图
- 学生/教师节点关系热力图
- 参数量统计
- 推理速度统计
- 消融实验开关

如果你只想快速开始，直接看“完整实验流程”这一节就行。

如果你想要最省心、一步一步照着做，请直接看：

- [docs/quickstart_foolproof.md](/C:/Users/86151/Documents/New%20project/docs/quickstart_foolproof.md)

如果你想系统调参，请看：

- [docs/tuning_guide.md](/C:/Users/86151/Documents/New%20project/docs/tuning_guide.md)

如果你跑完实验准备整理成论文表格，请看：

- [docs/result_table_template.md](/C:/Users/86151/Documents/New%20project/docs/result_table_template.md)

如果你以后换设备、重新接手项目、希望快速恢复上下文，请直接看：

- [docs/project_full_memory.md](/C:/Users/86151/Documents/New%20project/docs/project_full_memory.md)

## 项目结构

```text
C:\Users\86151\Documents\New project
├─ checkpoints/                    # 模型权重
├─ data/                           # 数据目录
├─ docs/
│  ├─ architecture.md              # 模块结构图和数据流图
│  └─ paper_framework.md           # 论文创新点与实验设计建议
├─ losses/
│  └─ distillation.py              # 蒸馏损失
├─ models/
│  ├─ teacher_gwnet.py             # 教师模型
│  └─ student_gcn.py               # 学生模型
├─ outputs/
│  ├─ figures/                     # 曲线图/热力图
│  ├─ predictions/                 # 预测 CSV
│  └─ reports/                     # 训练历史 JSON
├─ scripts/
│  ├─ sanity_check.py              # 维度检查
│  └─ benchmark_model.py           # 参数量和速度统计
├─ engine.py                       # 训练器
├─ generate_training_data.py       # 生成 train/val/test
├─ test.py                         # 测试与可视化
├─ train.py                        # 教师训练
├─ train_student_kd.py             # 学生蒸馏训练
├─ util.py                         # 数据和指标工具
└─ visualize.py                    # 根据历史文件画曲线
```

## 先看这几个文件

建议你在 PyCharm 里优先打开这几个文件：

- [train_student_kd.py](C:/Users/86151/Documents/New%20project/train_student_kd.py)
- [losses/distillation.py](C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [engine.py](C:/Users/86151/Documents/New%20project/engine.py)
- [test.py](C:/Users/86151/Documents/New%20project/test.py)
- [docs/paper_framework.md](C:/Users/86151/Documents/New%20project/docs/paper_framework.md)

## 已实现的论文实验能力

### 1. 可视化

已经实现：

- 预测曲线图
- 教师自适应邻接矩阵热力图
- 节点关系热力图
- 训练损失曲线

对应脚本：

- [test.py](C:/Users/86151/Documents/New%20project/test.py)
- [visualize.py](C:/Users/86151/Documents/New%20project/visualize.py)

### 2. 效率分析

已经实现：

- 参数量统计
- 平均推理时间统计
- 压缩率统计

对应脚本：

- [scripts/benchmark_model.py](C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
- [train_student_kd.py](C:/Users/86151/Documents/New%20project/train_student_kd.py)

### 3. 教师与学生对比图

已经实现：

- 同一张图中同时绘制真实值、教师预测、学生预测

对应脚本：

- [compare_teacher_student.py](C:/Users/86151/Documents/New%20project/compare_teacher_student.py)
- [scripts/plot_efficiency_tradeoff.py](C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)

### 4. 消融实验

已经支持直接通过命令开关进行：

- 去掉关系蒸馏：把 `--relation_weight 0.0`
- 去掉特征蒸馏：把 `--feature_weight 0.0`
- 去掉可靠性蒸馏：加 `--disable_reliability`
- 去掉课程蒸馏：加 `--disable_curriculum`

也就是说，你不用再改代码，直接换命令就能跑消融。

## 环境准备

### 1. 进入项目目录

```powershell
cd "C:\Users\86151\Documents\New project"
```

### 2. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. 安装基础依赖

```powershell
pip install -r requirements.txt
```

### 4. 安装 PyTorch

如果你只是想先跑通，可以先试：

```powershell
pip install torch
```

如果你有 GPU，建议按你自己的 CUDA 版本安装对应的 PyTorch。

### 5. 检查是否安装成功

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出最后是 `True`，说明可以用 GPU。

## 数据准备

### 1. 数据放置方式

把文件放成下面这样：

```text
data/
├─ METR-LA/
├─ PEMS-BAY/
└─ sensor_graph/
   ├─ metr-la.h5
   ├─ pems-bay.h5
   ├─ adj_mx.pkl
   └─ adj_mx_bay.pkl
```

如果你的文件名不同，只需要在命令里改成真实路径即可。

### 2. 生成 METR-LA 的训练集

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/metr-la.h5 --output_dir data/METR-LA --seq_length_x 12 --seq_length_y 12
```

### 3. 生成 PEMS-BAY 的训练集

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/pems-bay.h5 --output_dir data/PEMS-BAY --seq_length_x 12 --seq_length_y 12
```

生成成功后会得到：

- `data/METR-LA/train.npz`
- `data/METR-LA/val.npz`
- `data/METR-LA/test.npz`

或者：

- `data/PEMS-BAY/train.npz`
- `data/PEMS-BAY/val.npz`
- `data/PEMS-BAY/test.npz`

## 完整实验流程

下面给你一套从最开始到最后结果输出的完整流程。建议你先完整跑一遍 `METR-LA`，确认都正常后，再跑 `PEMS-BAY`。

### 第 0 步：维度自检

```powershell
python scripts/sanity_check.py
python -m scripts.sanity_check#这个
```

作用：

- 检查教师和学生输出维度是否一致
- 确保 `[B, 1, N, H]` 的蒸馏维度是对齐的

### 第 1 步：训练教师模型

METR-LA：

```powershell
#五组调参实验表
#基础base（未改参数）          #train_loss=2.8549, val_loss=2.7563, train_mape=0.0766, val_mape=0.0771, time=141.58s 目前效果最好
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name metr_teacher

#更大隐藏维度                 #train_loss=2.7607, val_loss=2.7390, train_mape=0.0734, val_mape=0.0750, time=638.91s
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --nhid 64 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher_h64

#更小 batch                  #train_loss=2.8537, val_loss=2.7655, train_mape=0.0766, val_mape=0.0777, time=143.85s
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 32 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher_bs32

#更长训练轮数
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 80 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher_e80

#更小学习率                  #train_loss=2.9168, val_loss=2.7686, train_mape=0.0789, val_mape=0.0748, time=148.85s
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --nhid 32 --learning_rate 0.0005 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher_lr5e4

```

如果你没有 GPU，请改成：
```powershell
python train.py --device cpu --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name metr_teacher
```

训练完成后会得到：

- `checkpoints/teacher/metr_teacher_best.pt`   #训练好的教师模型权重（各种参数和最优损失、epoch）用于测试test
- `outputs/reports/metr_teacher_teacher_history.json`  #日志（用于画图）
- `outputs/figures/metr_teacher_teacher_curve.png`    #训练曲线图

### 第 2 步：测试教师模型

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name metr_teacher_eval
```

这一步会输出：

- 每个 horizon 的 `MAE / RMSE / MAPE`
- 平均误差
- 预测曲线图
- 教师自适应邻接矩阵热力图
- 教师节点关系热力图

主要结果文件：

- `outputs/figures/metr_teacher_eval_teacher_sensor10_h12.png` #预测曲线图
- `outputs/figures/metr_teacher_eval_teacher_adaptive_adj.png` #教师自适应邻接矩阵热力图
- `outputs/figures/metr_teacher_eval_teacher_relation.png` #教师节点关系热力图
  - `outputs/predictions/metr_teacher_eval_teacher_sensor10.csv`#预测结果数值

### 第 3 步：训练完整方法 RMGD-KD 训练学生

```powershell
# 基础参数base         # train_loss=2.9186, val_loss=2.0322, soft=3.4932, feat=0.0041, rel=0.0201, visible_h=8, val_latency=143.47ms, time=447.74s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd
#选择最优标准从 val_loss 改成了 val_mae。 train_total=2.9652, train_mae=3.6756, val_total=2.0699, val_mae=3.1637, soft=3.7876, feat=0.0040, rel=0.0194, visible_h=12, val_latency=143.36ms, time=415.05s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd_mae_select
```

这一步已经包含：
 
- 真实标签监督
- 可靠性加权蒸馏
- 图关系蒸馏
- 多步课程蒸馏

输出文件：

- `checkpoints/student/metr_student_rmgd_best.pt` #训练好的学生模型权重（各种参数和最优损失、epoch）用于测试test
- `outputs/reports/metr_student_rmgd_student_history.json` #训练历史记录
- `outputs/figures/metr_student_rmgd_student_curve.png` #训练曲线图

### 第 4 步：测试完整方法 RMGD-KD

```powershell
# 基础参数base              #[student] average -> MAE=3.5549, MAPE=0.1034, RMSE=6.9349, params=27,404, latency=16.17ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_rmgd_eval
#选择最优标准从 val_loss 改成了 val_mae。[student] average -> MAE=3.5175, MAPE=0.1023, RMSE=6.7120, params=27,404, latency=14.02ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_mae_select_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_rmgd_mae_select_eval
```

输出文件：

- `outputs/figures/metr_student_rmgd_eval_student_sensor10_h12.png` #预测曲线图
- `outputs/figures/metr_student_rmgd_eval_student_relation.png` #学生节点关系热力图
- `outputs/predictions/metr_student_rmgd_eval_student_sensor10.csv` #预测结果数值

### 第 5 步：统计教师模型参数量和推理速度

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

### 第 6 步：统计学生模型参数量和推理速度

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --batch_size 64
```

终端会输出：

- `params`
- `avg_latency_ms`
- `batch_size`

这些就是你论文效率分析表格的核心数据。

### 第 7 步：如需手动重画训练曲线

教师：

```powershell
python visualize.py --history outputs/reports/metr_teacher_teacher_history.json --save_path outputs/figures/metr_teacher_teacher_curve_manual.png
```

学生：

```powershell
python visualize.py --history outputs/reports/metr_student_rmgd_student_history.json --save_path outputs/figures/metr_student_rmgd_student_curve_manual.png
```

### 第 8 步：生成教师与学生对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_rmgd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_student_compare
```

### 第 9 步：自动汇总结果表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_summary.csv --output_md outputs/reports/metr_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_best.pt" --run "RMGD-KD,student,checkpoints/student/metr_student_rmgd_best.pt"
```

### 第 10 步：生成时间-性能达成度图

这张图很适合加到论文里，因为它能直接说明：

- 学生用了教师多少推理时间
- 学生达到了教师多少百分比的预测性能

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_efficiency_tradeoff.png --derived_csv outputs/reports/metr_efficiency_tradeoff.csv
```

运行后会得到：

- `outputs/figures/metr_efficiency_tradeoff.png`
- `outputs/reports/metr_efficiency_tradeoff.csv`

默认定义说明：

- 横轴：`Inference Time (% of Teacher)`
- 纵轴：`Prediction Performance Reached (% of Teacher)`
- 因为 `MAE / RMSE / MAPE` 都是越小越好，所以性能达成度按 `teacher_metric / model_metric * 100%` 计算

## 消融实验完整命令

下面这些命令可以直接用于论文消融表。

### 1. Baseline Student

只保留硬标签监督，不做关系蒸馏、不做特征蒸馏、不做软蒸馏：

```powershell
# 训练 base参数              # train_loss=3.6102, val_loss=3.1212, soft=4.2633, feat=0.0170, rel=0.5845, visible_h=12, val_latency=142.72ms, time=447.97s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_reliability --disable_curriculum --exp_name metr_student_baseline
#测试                    # average -> MAE=3.4762, MAPE=0.1020, RMSE=6.6793, params=27,404, latency=13.03ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_baseline_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_baseline_eval
```
#预测曲线已保存到: outputs\figures\metr_student_baseline_eval_student_sensor10_h12.png
#预测数值已保存到: outputs\predictions\metr_student_baseline_eval_student_sensor10.csv
#节点关系热力图已保存到: outputs\figures\metr_student_baseline_eval_student_relation.png

### 2. Vanilla KD
有软蒸馏，但去掉可靠性和课程蒸馏，也去掉关系蒸馏：
```powershell
#base            train_loss=3.7261, val_loss=2.9924, soft=3.9597, feat=0.0170, rel=0.5845, visible_h=12, val_latency=142.68ms, time=374.18s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --disable_reliability --disable_curriculum --exp_name metr_student_vanilla_kd
#测试             average -> MAE=3.4645, MAPE=0.0999, RMSE=6.6605, params=27,404, latency=12.64ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_vanilla_kd_eval

```

### 3. 去掉可靠性蒸馏
```powershell
#base train_total=3.0030, train_mae=3.6496, val_total=2.4414, val_mae=3.1324, soft=4.0547, feat=0.0041, rel=0.0191, visible_h=12, val_latency=143.12ms,
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --disable_reliability --exp_name metr_student_wo_reliability
测试  [student] average -> MAE=3.4736, MAPE=0.0994, RMSE=6.6470, params=27,404, latency=14.95ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_reliability_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_reliability_eval
```

### 4. 去掉课程蒸馏

```powershell
train_total=2.9647, train_mae=3.6802, val_total=2.0488, val_mae=3.1610, soft=3.7710, feat=0.0040, rel=0.0196, visible_h=12, val_latency=145.07ms, time=542.
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --disable_curriculum --exp_name metr_student_wo_curriculum
#测试average -> MAE=3.5261, MAPE=0.1015, RMSE=6.8269, params=27,404, latency=14.37ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_curriculum_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_curriculum_eval

```

### 5. 去掉关系蒸馏

```powershell
#base  train_total=2.9644, train_mae=3.6705, val_total=2.0606, val_mae=3.1590, soft=3.8090, feat=0.0027, rel=0.0510, visible_h=12, val_latency=142.64ms, time=357.18s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.0 --exp_name metr_student_wo_relation
#测试  average -> MAE=3.5102, MAPE=0.0986, RMSE=6.7805, params=27,404, latency=15.04ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_relation_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_relation_eval
```
                                                            
### 6. 去掉特征蒸馏

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.0 --relation_weight 0.1 --exp_name metr_student_wo_feature
#测试[student] average -> MAE=3.5048, MAPE=0.1009, RMSE=6.7468, params=27,404, latency=15.92ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_feature_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_feature_eval
```

### 7. 完整方法

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd
```

## PEMS-BAY 完整流程

如果你跑 PEMS-BAY，只需要把 `data`、`adjdata`、`teacher_checkpoint`、`exp_name` 换成 PEMS-BAY 对应版本。

### 1. 教师训练

```powershell
python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name bay_teacher
```

### 2. 测试教师模型

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name bay_teacher_eval
```

### 3. 完整方法训练

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name bay_student_rmgd
```

### 4. 测试完整方法

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_rmgd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_rmgd_eval
```

### 5. 参数量和速度统计

教师：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --batch_size 64
```

学生：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_rmgd_best.pt --model_type student --batch_size 64
```

### 6. 教师和学生对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --student_checkpoint checkpoints/student/bay_student_rmgd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name bay_teacher_student_compare
```

### 7. 自动汇总结果表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --output_csv outputs/reports/bay_summary.csv --output_md outputs/reports/bay_summary.md --run "Teacher,teacher,checkpoints/teacher/bay_teacher_best.pt" --run "Baseline,student,checkpoints/student/bay_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/bay_student_vanilla_kd_best.pt" --run "RMGD-KD,student,checkpoints/student/bay_student_rmgd_best.pt"
```

### 8. 生成时间-性能达成度图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/bay_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/bay_efficiency_tradeoff.png --derived_csv outputs/reports/bay_efficiency_tradeoff.csv
```

### 9. PEMS-BAY 消融实验

Baseline Student：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_reliability --disable_curriculum --exp_name bay_student_baseline
```

Vanilla KD：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --disable_reliability --disable_curriculum --exp_name bay_student_vanilla_kd
```

w/o Reliability：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --disable_reliability --exp_name bay_student_wo_reliability
```

w/o Curriculum：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --disable_curriculum --exp_name bay_student_wo_curriculum
```

w/o Relation：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.0 --exp_name bay_student_wo_relation
```

w/o Feature：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.0 --relation_weight 0.1 --exp_name bay_student_wo_feature
```

Full RMGD-KD：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name bay_student_rmgd
```

## 论文建议怎么使用这些输出

### 主结果表

你需要收集：

- 教师测试指标
- Baseline Student 指标
- Vanilla KD 指标
- 各种消融指标
- 完整方法指标
- 参数量
- 推理时间

### 推荐图

你现在已经可以直接得到这些图：

- 预测曲线图
- 教师自适应邻接矩阵热力图
- 学生节点关系热力图
- 教师节点关系热力图
- 训练曲线图

### 需要你整理成表格的结果

终端输出和 `outputs/` 目录中的文件可以整理成：

- 主实验对比表
- 消融实验表
- 效率分析表

## 当前最推荐的操作顺序

```text
1. 安装环境
2. 生成数据
3. 运行 sanity_check.py
4. 训练教师 GWNet
5. 测试教师并保存图
6. 训练完整方法 RMGD-KD
7. 测试完整方法并保存图
8. 运行 benchmark_model.py 统计速度和参数
9. 依次运行消融命令
10. 把结果整理成论文表格和图
```

## 相关文档

- [docs/architecture.md](C:/Users/86151/Documents/New%20project/docs/architecture.md)
- [docs/dataflow_explained.md](C:/Users/86151/Documents/New%20project/docs/dataflow_explained.md)
- [docs/paper_diagrams.md](C:/Users/86151/Documents/New%20project/docs/paper_diagrams.md)
- [docs/paper_framework.md](C:/Users/86151/Documents/New%20project/docs/paper_framework.md)
- [docs/tuning_guide.md](C:/Users/86151/Documents/New%20project/docs/tuning_guide.md)
- [docs/result_table_template.md](C:/Users/86151/Documents/New%20project/docs/result_table_template.md)

## 你现在最常用的 4 条命令

### 1. 训练教师

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name metr_teacher
```

### 2. 训练完整学生方法

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd
```

### 3. 测试完整学生方法

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_rmgd_eval
```

### 4. 做效率分析

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --batch_size 64
```

### 5. 画教师和学生对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_rmgd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_student_compare
```

### 6. 画时间-性能达成度图
```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_efficiency_tradeoff.png --derived_csv outputs/reports/metr_efficiency_tradeoff.csv
```
