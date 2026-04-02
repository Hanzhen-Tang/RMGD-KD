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
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --exp_name bay_student_cckd_v4
```

### 8. 测试 CCKD-v4

```powershell
MAE=1.7702, MAPE=0.0408, RMSE=3.8443, params=27,404, latency=14.14ms/batch
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v4_eval
```

### 9. 消融：去掉可信度模块 `w/o confidence`

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_confidence_filter --exp_name bay_student_wo_confidence_v4
```

测试：

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_confidence_v4_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_confidence_v4_eval
```

### 10. 消融：去掉课程蒸馏 `w/o curriculum`

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --trend_weight 0.5 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_power 1.0 --disable_curriculum --exp_name bay_student_wo_curriculum_v4
```

测试：

```powershell
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
7. 精度-效率折中图

这些图已经足够支撑：

- 方法结构说明
- 定性分析
- 效率讨论
- 论文写作与答辩展示
