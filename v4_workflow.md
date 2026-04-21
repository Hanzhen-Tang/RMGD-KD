# v4 运行流程

## 先说明

这份文档只对应 **v4 方法**。

相较于 `v3`：

- 不再采用二值可信度筛选
- 高可信位置做绝对值蒸馏
- 低可信位置做趋势蒸馏
- 课程蒸馏继续保留

所有命令都建议在项目根目录运行：

```powershell
cd "C:\Users\86151\Documents\New project"
```

---

## 一、METR-LA

### 1. 如果教师模型还没有，就先训练教师

```powershell
python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name metr_teacher
```

主要输出：

- `checkpoints/teacher/metr_teacher_best.pt`
- `outputs/reports/metr_teacher_teacher_history.json`
- `outputs/figures/metr_teacher_teacher_curve.png`

中文解释：

- `best.pt`：教师模型最终最优权重
- `history.json`：教师训练过程记录
- `curve.png`：教师训练曲线图

### 2. 测试教师模型

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name metr_teacher_eval
```

主要输出：

- `outputs/figures/metr_teacher_eval_teacher_sensor10_h12.png`
- `outputs/figures/metr_teacher_eval_teacher_adaptive_adj.png`
- `outputs/figures/metr_teacher_eval_teacher_relation.png`
- `outputs/predictions/metr_teacher_eval_teacher_sensor10.csv`

中文解释：

- `*_sensor10_h12.png`：教师在第 10 个传感器、第 12 个预测步上的预测曲线
- `*_adaptive_adj.png`：教师自适应邻接矩阵热力图
- `*_relation.png`：教师节点关系热力图
- `*.csv`：对应预测曲线的数值文件

### 3. 训练 Baseline Student

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_baseline_v4
```

### 4. 测试 Baseline Student

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_baseline_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_baseline_v4_eval
```

### 5. 训练 Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name metr_student_vanilla_kd_v4
```

### 6. 测试 Vanilla KD

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_vanilla_kd_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_vanilla_kd_v4_eval
```

### 7. 训练 CCKD-v4   

```powershell
 train_total=3.0723, train_mae=3.6238, val_total=2.4955, val_mae=3.1352, abs=2.5669, trend=0.2223, mean_conf=0.2880, trend_ratio=0.7025, visible_h=12, val_latency=141.91ms, time=517.00s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --exp_name metr_student_cckd_v4
```

### 8. 测试 CCKD-v4

```powershell
average -> MAE=3.4853, MAPE=0.0984, RMSE=6.5964, params=27,404, latency=15.89ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_cckd_v4_eval
```

### 9. 消融：去掉可信度模块 `w/o confidence`

训练：

```powershell
train_total=3.2662, train_mae=3.5980, val_total=2.6758, val_mae=3.1195, abs=3.7381, trend=0.0000, mean_conf=0.2880, trend_ratio=0.0000, visible_h=12, val_latency=142.54ms, time=519.14s
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --exp_name metr_student_wo_confidence_v4
```

测试：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_confidence_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_confidence_v4_eval
```

### 10. 消融：去掉课程蒸馏 `w/o curriculum`

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_curriculum --exp_name metr_student_wo_curriculum_v4
```

测试：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_curriculum_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_curriculum_v4_eval
```

### 11. 统计教师与学生的参数量和推理速度

教师：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

学生：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v4_best.pt --model_type student --batch_size 64
```

中文解释：

- 这一步主要拿到参数量和平均推理时间
- 后面可以直接写进效率表格

### 12. 生成教师-学生对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_cckd_v4_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_compare_cckd_v4
```

主要输出：

- `outputs/figures/metr_compare_cckd_v4_sensor10_h12.png`
- `outputs/predictions/metr_compare_cckd_v4_sensor10_h12.csv`

中文解释：

- 图中会同时显示真实值、教师预测、学生预测
- csv 方便你后续用 Origin、Excel、Python 再画图

### 13. 汇总结果表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v4_summary.csv --output_md outputs/reports/metr_v4_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_v4_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_v4_best.pt" --run "CCKD-v4,student,checkpoints/student/metr_student_cckd_v4_best.pt" --run "w/o confidence,student,checkpoints/student/metr_student_wo_confidence_v4_best.pt" --run "w/o curriculum,student,checkpoints/student/metr_student_wo_curriculum_v4_best.pt"
```

主要输出：

- `outputs/reports/metr_v4_summary.csv`
- `outputs/reports/metr_v4_summary.md`

### 14. 生成精度-效率折中图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_v4_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_v4_efficiency_tradeoff.png --derived_csv outputs/reports/metr_v4_efficiency_tradeoff.csv
```

主要输出：

- `outputs/figures/metr_v4_efficiency_tradeoff.png`
- `outputs/reports/metr_v4_efficiency_tradeoff.csv`

中文解释：

- 横轴：推理时间占教师模型的百分比
- 纵轴：预测性能达到教师模型的百分比
- 这张图适合放到论文里讨论“精度-效率平衡”

---

## 二、PEMS-BAY

### 1. 训练 PEMS-BAY 教师模型

```powershell
train_loss=1.4801, val_loss=1.6113, train_mape=0.0320, val_mape=0.0371, time=364.90s E70
python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 70 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name bay_teacher_e70
```

主要输出：

- `checkpoints/teacher/bay_teacher_best.pt`
- `outputs/reports/bay_teacher_teacher_history.json`
- `outputs/figures/bay_teacher_teacher_curve.png`

### 2. 测试 PEMS-BAY 教师模型

```powershell
 MAE=1.5905, MAPE=0.0355, RMSE=3.5096, params=311,760, latency=38.87ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name bay_teacher_eval
MAE=1.5853, MAPE=0.0360, RMSE=3.5478, params=311,760, latency=37.50ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_e70_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name bay_teacher_eval_e70
```

中文解释：

- `PEMS-BAY` 不能直接复用 `METR-LA` 的教师模型
- 因为两个数据集的节点数、图结构和分布不同
- 所以必须先单独训练本数据集的教师

### 3. 训练 Baseline Student

```powershell
train_total=1.9398, train_mae=1.9398, val_total=1.9058, val_mae=1.9058, abs=1.8783, trend=0.0000, mean_conf=0.2751, trend_ratio=0.0000, visible_h=12, val_latency=250.14ms, time=230.53s
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_baseline_v4
```

### 4. 测试 Baseline Student

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_baseline_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_baseline_v4_eval
```

### 5. 训练 Vanilla KD

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_vanilla_kd_v4
```

### 6. 测试 Vanilla KD

```powershell
average -> MAE=1.7643, MAPE=0.0408, RMSE=3.7756, params=27,404, latency=14.07ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_vanilla_kd_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_vanilla_kd_v4_eval
```

### 7. 训练 CCKD-v4


```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --exp_name bay_student_cckd_v4_e60
```

### 8. 测试 CCKD-v4

```powershell
MAE=1.7702, MAPE=0.0408, RMSE=3.8443, params=27,404, latency=14.14ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_eval
### e60
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_e60_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_eval_e60
```

### 9. 消融：去掉可信度模块 `w/o confidence`

训练：

```powershell
train_total=1.7278, train_mae=1.9456, val_total=1.6143, val_mae=1.8958, abs=1.8292, trend=0.0000, mean_conf=0.2751, trend_ratio=0.0000, visible_h=12, val_latency=248.30ms, time=228.35s
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --exp_name bay_student_wo_confidence_v4
```

测试：

```powershell
MAE=1.7618, MAPE=0.0403, RMSE=3.8045, params=27,404, latency=14.34ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_confidence_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_confidence_v4_eval
```

### 10. 消融：去掉课程蒸馏 `w/o curriculum`

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_curriculum --exp_name bay_student_wo_curriculum_v4
```

测试：

```powershell
MAE=1.7588, MAPE=0.0405, RMSE=3.7794, params=27,404, latency=17.59ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_curriculum_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_curriculum_v4_eval
```

### 11. 统计教师与学生的参数量和推理速度

教师：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --batch_size 64
```

学生：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_best.pt --model_type student --batch_size 64
```

### 12. 生成教师-学生对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --student_checkpoint checkpoints/student/bay_student_cckd_v4_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name bay_compare_cckd_v4
```
comparison_figure=outputs\figures\bay_compare_cckd_v4_sensor10_h12.png
comparison_csv=outputs\predictions\bay_compare_cckd_v4_sensor10_h12.csv
### 13. 汇总结果表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --output_csv outputs/reports/bay_v4_summary.csv --output_md outputs/reports/bay_v4_summary.md --run "Teacher,teacher,checkpoints/teacher/bay_teacher_best.pt" --run "Baseline,student,checkpoints/student/bay_student_baseline_v4_best.pt" --run "VanillaKD,student,checkpoints/student/bay_student_vanilla_kd_v4_best.pt" --run "CCKD-v4,student,checkpoints/student/bay_student_cckd_v4_best.pt" --run "w/o confidence,student,checkpoints/student/bay_student_wo_confidence_v4_best.pt" --run "w/o curriculum,student,checkpoints/student/bay_student_wo_curriculum_v4_best.pt"
```

### 14. 生成精度-效率折中图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/bay_v4_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/bay_v4_efficiency_tradeoff.png --derived_csv outputs/reports/bay_v4_efficiency_tradeoff.csv
```

---

## 三、最推荐你最后放到论文里的图

建议优先生成这几类：

1. 教师预测曲线图
2. 学生预测曲线图
3. 教师-学生对比预测图
4. 教师自适应邻接矩阵热力图
5. 节点关系热力图
6. `CCKD-v4` 训练曲线图



---

## 课程模式切换说明

为了保证 **METR-LA** 继续使用当前已经验证有效的课程蒸馏策略，同时允许 **PEMS-BAY** 使用更温和的课程策略，`v4` 现在新增了可切换参数：

```powershell
--curriculum_mode standard|short|wide|soft
```

### 各模式含义

- `standard`
  - 默认模式
  - 保留原来的课程蒸馏逻辑
  - **METR-LA 建议继续使用这个模式**

- `short`
  - 前期只做一个较短的课程预热，随后快速开放全部 horizon
  - **优先推荐给 PEMS-BAY**

- `wide`
  - 前期一开始就开放更多 horizon
  - 适合希望减少前期限制的情况

- `soft`
  - 不再硬屏蔽后面 horizon，而是用更平滑的权重逐步提升长期 horizon 的蒸馏强度
  - 更适合后续扩展实验

### 推荐使用方式

#### METR-LA

保持默认即可，也可以显式写出：

```powershell
--curriculum_mode standard
```

#### PEMS-BAY

建议先尝试：

```powershell
--curriculum_mode short
```

如果还想进一步比较，可再试：

```powershell
--curriculum_mode wide
```

### PEMS-BAY 优化版 CCKD-v4 soft

```powershell
训练 soft 50轮 命名错误（命名成60实际为50）
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_cckd_v4_e60_soft
测试 MAE=1.7764, MAPE=0.0405, RMSE=3.8016, params=27,404, latency=13.95ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_e60_soft_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_e60_soft_eval

训练 soft 60轮
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 60 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode soft --exp_name bay_student_cckd_v4_ee60_soft
测试 MAE=1.7650, MAPE=0.0413, RMSE=3.8335, params=27,404, latency=13.64ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_ee60_soft_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_ee60_soft_eval
```

```powershell
训练 wide
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 60 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode wide --exp_name bay_student_cckd_v4_e60_wide
测试 
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_e60_wide_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_e60_soft_eval
```
训练：

```powershell 
训练 short e49
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --curriculum_mode short --exp_name bay_student_cckd_v4_short
```

测试：

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_short_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_short_eval
```

### 说明

- 这次新增的是“**可切换课程模式**”，不是直接替换原逻辑。
- 因此：
  - `METR-LA` 原来的 `standard` 逻辑不受影响
  - `PEMS-BAY` 可以单独尝试更弱的课程策略
7. 精度-效率折中图

这些图已经足够支撑：

- 方法结构说明
- 定性分析
- 效率讨论
- 论文写作与答辩展示

```powershell
METR-LA 教师误差热力图 可信度热力图
python scripts/generate_distillation_heatmap.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --mode both --node_limit 48 --node_select top_error --exp_name metr_distill
```
