# RMGD-KD 傻瓜式实验手册

这份文档是给第一次跑项目的人准备的。

你不用先理解所有代码，只要按顺序一步一步做，就可以把完整实验跑出来。

如果你以后换电脑、换环境、换设备，优先先看这份文档。

## 0. 你最终要得到什么

你跑完整个流程后，应该拿到下面几类结果：

- 教师模型权重
- 学生模型权重
- 教师测试指标
- 学生测试指标
- 预测曲线图
- 节点关系热力图
- 教师自适应邻接矩阵热力图
- 参数量和推理时间
- 消融实验结果

这些结果足够支撑你做论文的：

- 主结果表
- 消融实验表
- 效率分析表
- 可视化图

## 1. 先做最重要的一件事

打开终端，进入项目目录：

```powershell
cd "C:\Users\86151\Documents\New project"
```

如果你能成功进入这个目录，说明你已经站在正确的位置了。

## 2. 创建虚拟环境

输入：

```powershell
python -m venv .venv
```

成功后，项目目录里会出现一个 `.venv` 文件夹。

然后激活它：

```powershell
.\.venv\Scripts\activate
```

成功后，终端前面通常会出现 `(.venv)`。

## 3. 安装依赖

输入：

```powershell
pip install -r requirements.txt
```

然后安装 PyTorch：

```powershell
pip install torch
```

如果你有 GPU，也可以安装适合自己 CUDA 版本的 PyTorch。

## 4. 检查 PyTorch 是否正常

输入：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

你会看到两行输出。

如果最后一行是：

- `True`：说明可以用 GPU
- `False`：说明只能先用 CPU

如果是 `False`，后面命令里的 `--device cuda:0` 全部改成 `--device cpu`

## 5. 放数据

你需要把数据放到这个目录里：

```text
data/sensor_graph/
```

建议放成这样：

```text
data/sensor_graph/
├─ metr-la.h5
├─ pems-bay.h5
├─ adj_mx.pkl
└─ adj_mx_bay.pkl
```

## 6. 生成训练数据

### 6.1 生成 METR-LA

输入：

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/metr-la.h5 --output_dir data/METR-LA --seq_length_x 12 --seq_length_y 12
python generate_training_data.py --traffic_df_filename data/sensor_graph/metr-la.h5 --output_dir data/METR-LA --seq_length_x 12 --seq_length_y 12
```

成功后，你应该能看到这些文件：

- `data/METR-LA/train.npz`
- `data/METR-LA/val.npz`
- `data/METR-LA/test.npz`

### 6.2 生成 PEMS-BAY

输入：

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/pems-bay.h5 --output_dir data/PEMS-BAY --seq_length_x 12 --seq_length_y 12
```

成功后，你应该能看到：

- `data/PEMS-BAY/train.npz`
- `data/PEMS-BAY/val.npz`
- `data/PEMS-BAY/test.npz`

## 7. 做维度检查

输入：

```powershell
python scripts/sanity_check.py
python -m scripts.sanity_check
```

如果成功，你应该看到：

```text
Sanity check passed.
```

这一步的作用是：

- 检查教师模型输出维度
- 检查学生模型输出维度
- 检查蒸馏用的张量维度是否一致

如果这一步都过不了，不要继续训练。

## 8. 先训练教师模型

先只跑 METR-LA。

### 8.1 如果你用 GPU

输入：

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name metr_teacher
```

### 8.2 如果你用 CPU

输入：

```powershell
python train.py --device cpu --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name metr_teacher
```

### 8.3 成功后你应该得到

- `checkpoints/teacher/metr_teacher_best.pt`
- `outputs/reports/metr_teacher_teacher_history.json`
- `outputs/figures/metr_teacher_teacher_curve.png`

如果这 3 个文件都出现了，说明教师训练成功。

## 9. 测试教师模型

输入：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name metr_teacher_eval
```

如果你用 CPU，把 `cuda:0` 改成 `cpu`。

### 9.1 成功后你应该得到

- `outputs/figures/metr_teacher_eval_teacher_sensor10_h12.png`
- `outputs/figures/metr_teacher_eval_teacher_adaptive_adj.png`
- `outputs/figures/metr_teacher_eval_teacher_relation.png`
- `outputs/predictions/metr_teacher_eval_teacher_sensor10.csv`

### 9.2 这一步的作用

- 输出教师模型测试指标
- 画教师预测曲线
- 画教师自适应图结构
- 画教师节点关系图

## 10. 训练完整学生模型 RMGD-KD

这一步是整篇方法最重要的一步。

输入：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd
```

### 10.1 这一步已经包含的创新点

- 可靠性加权蒸馏
- 图关系蒸馏
- 多步课程蒸馏
- 特征蒸馏

### 10.2 成功后你应该得到

- `checkpoints/student/metr_student_rmgd_best.pt`
- `outputs/reports/metr_student_rmgd_student_history.json`
- `outputs/figures/metr_student_rmgd_student_curve.png`

## 11. 测试完整学生模型

输入：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_rmgd_eval
```

### 11.1 成功后你应该得到

- `outputs/figures/metr_student_rmgd_eval_student_sensor10_h12.png`
- `outputs/figures/metr_student_rmgd_eval_student_relation.png`
- `outputs/predictions/metr_student_rmgd_eval_student_sensor10.csv`

### 11.2 这一步的作用

- 输出学生模型测试指标
- 输出学生预测曲线
- 输出学生节点关系图

## 12. 做效率分析

### 12.1 统计教师模型参数量和速度

输入：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

### 12.2 统计学生模型参数量和速度

输入：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --batch_size 64
```

### 12.3 你会得到什么

终端会显示：

- `params`
- `avg_latency_ms`

这两个值后面要放到论文效率分析表里。

## 12.5 画教师和学生对比图

输入：

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_rmgd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_student_compare
```

成功后你应该得到：

- `outputs/figures/metr_teacher_student_compare_sensor10_h12.png`
- `outputs/predictions/metr_teacher_student_compare_sensor10_h12.csv`

这张图里会同时有：

- Real
- Teacher
- Student

## 13. 做消融实验

这一部分是论文很重要的支撑。

### 13.1 跑 Baseline Student

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_reliability --disable_curriculum --exp_name metr_student_baseline
```

### 13.2 跑 Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --disable_reliability --disable_curriculum --exp_name metr_student_vanilla_kd
```

### 13.3 去掉可靠性蒸馏

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --disable_reliability --exp_name metr_student_wo_reliability
```

### 13.4 去掉课程蒸馏

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --disable_curriculum --exp_name metr_student_wo_curriculum
```

### 13.5 去掉关系蒸馏

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.0 --exp_name metr_student_wo_relation
```

### 13.6 去掉特征蒸馏

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.0 --relation_weight 0.1 --exp_name metr_student_wo_feature
```

### 13.7 完整方法

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd
```

## 13.8 自动汇总结果表

如果你已经跑完多个模型，可以直接汇总成表：

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_summary.csv --output_md outputs/reports/metr_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_best.pt" --run "RMGD-KD,student,checkpoints/student/metr_student_rmgd_best.pt"
```

成功后你应该得到：

- `outputs/reports/metr_summary.csv`
- `outputs/reports/metr_summary.md`

## 14. 如果你要跑 PEMS-BAY

你只要把命令中的：

- `data data/METR-LA`
- `adjdata data/sensor_graph/adj_mx.pkl`
- `teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt`

替换为：

- `data data/PEMS-BAY`
- `adjdata data/sensor_graph/adj_mx_bay.pkl`
- `teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt`

就可以了。

## 15. 跑完整个实验后，你手里应该至少有这些内容

### 模型文件

- 教师模型权重
- 学生模型权重
- 各种消融模型权重

### 图

- 教师预测曲线图
- 学生预测曲线图
- 教师自适应邻接热力图
- 教师节点关系图
- 学生节点关系图
- 训练曲线图

### 指标

- 教师测试指标
- 学生测试指标
- 消融实验指标
- 参数量
- 推理时间

## 16. 如果报错了先看哪里

### 报错 1：没有 torch

说明你没有安装 PyTorch：

```powershell
pip install torch
```

### 报错 2：找不到数据文件

说明：

- 你的 `h5` 文件路径不对
- 或者 `adj_mx.pkl` 路径不对

### 报错 3：CUDA 不可用

说明你的机器当前不能用 GPU，把所有命令里的：

```text
--device cuda:0
```

改成：

```text
--device cpu
```

### 报错 4：显存不足

把：

```text
--batch_size 64
```

改成：

```text
--batch_size 32
```

或者：

```text
--batch_size 16
```

## 17. 如果你现在只想跑最关键的 4 步

### 第一步

```powershell
python generate_training_data.py --traffic_df_filename data/sensor_graph/metr-la.h5 --output_dir data/METR-LA --seq_length_x 12 --seq_length_y 12
```

### 第二步

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --exp_name metr_teacher
```

### 第三步

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd
```

### 第四步

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_rmgd_eval
```

---

如果你以后重新打开这个项目，建议先看：

- [docs/project_full_memory.md](/C:/Users/86151/Documents/New%20project/docs/project_full_memory.md)
- [docs/tuning_guide.md](/C:/Users/86151/Documents/New%20project/docs/tuning_guide.md)
- [README.md](/C:/Users/86151/Documents/New%20project/README.md)

## 18. 生成时间-性能达成度图

先确保你已经运行过：

- `scripts/collect_results.py`

然后直接输入：

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_efficiency_tradeoff.png --derived_csv outputs/reports/metr_efficiency_tradeoff.csv
```

成功后你应该得到：

- `outputs/figures/metr_efficiency_tradeoff.png`
- `outputs/reports/metr_efficiency_tradeoff.csv`

这张图怎么理解：

- 横轴越小越好，说明用的推理时间比教师更少
- 纵轴越大越好，说明越接近教师性能
- 如果学生点落在左上区域，说明它是很好的轻量化结果
