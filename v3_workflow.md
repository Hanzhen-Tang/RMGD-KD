# v3 运行流程手册

本手册只对应当前 `v3` 版本方法：

- 方法名：`CCKD-v3`
- 教师模型：`GWNet`
- 学生模型：轻量 `GCN`
- 蒸馏主干：`Vanilla KD`
- 保留模块：
  - `可信度筛选蒸馏 (confidence filtering)`
  - `课程蒸馏 (curriculum distillation)`
- 删除模块：
  - `feature distillation`
  - `relation distillation`

请始终先进入项目根目录后再执行命令：

```powershell
cd "C:\Users\86151\Documents\New project"
```

---

## 一、当前 v3 论文主线建议

建议保留并重点汇报的结果包括：

- `Teacher`
- `Baseline Student`
- `Vanilla KD`

需要重跑：

- `CCKD-v3`
- `w/o confidence v3`
- `w/o curriculum v3`

其中：

- `Teacher / Baseline / Vanilla KD` 可以继续复用之前结果
- `CCKD-v3` 和两个 `v3` 消融是当前论文的重点

---

## 二、METR-LA 全流程

### 1. 教师模型

如果你已经训练并测试过教师，可以直接复用：

```text
checkpoints/teacher/metr_teacher_best.pt
```

如果还没有教师结果，可先按旧流程完成教师训练与测试。

---

### 2. 训练 CCKD-v3

```powershell
train_total=3.2368, train_mae=3.6028, val_total=2.5242, val_mae=3.0988, soft=2.3828,
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --exp_name metr_student_cckd_v3
```

训练完成后将生成：

- `checkpoints/student/metr_student_cckd_v3_best.pt`
  中文解释：训练好的 `CCKD-v3` 最优学生模型权重，后续测试、画图、统计速度时都要加载它。
- `outputs/reports/metr_student_cckd_v3_student_history.json`
  中文解释：学生训练过程记录文件，里面有每轮的 `train/val` 指标、损失和课程蒸馏信息。
- `outputs/figures/metr_student_cckd_v3_student_curve.png`
  中文解释：训练曲线图，用于观察学生模型是否稳定收敛。

---

### 3. 测试 CCKD-v3

```powershell
[student] average -> MAE=3.4438, MAPE=0.0989, RMSE=6.6185, params=27,404, latency=11.04ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_cckd_v3_eval
```

常见输出：

- `outputs/figures/metr_student_cckd_v3_eval_student_sensor10_h12.png`
  中文解释：第 10 个传感器、第 12 个预测步的真实值与学生预测值对比曲线图。
- `outputs/figures/metr_student_cckd_v3_eval_student_relation.png`
  中文解释：学生节点关系热力图，用于观察学生模型学到的空间结构模式。
- `outputs/predictions/metr_student_cckd_v3_eval_student_sensor10.csv`
  中文解释：用于绘图和论文分析的真实值与预测值数值表。

---

### 4. 消融：去掉可信度筛选

#### 训练

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --disable_confidence_filter --exp_name metr_student_wo_confidence_v3
```

#### 测试

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_confidence_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_confidence_v3_eval
```

---

### 5. 消融：去掉课程蒸馏

#### 训练

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --disable_curriculum --exp_name metr_student_wo_curriculum_v3
```
#### 测试

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_curriculum_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_curriculum_v3_eval
```

---
###  Baseline Student 对比
```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 1.0 --soft_weight 0.0 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_baseline
### 测试 
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_baseline_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_baseline_eval
```
###  Vanilla KD 只保留软蒸馏
```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --disable_confidence_filter --disable_curriculum --exp_name bay_student_vanilla_kd
### 测试
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_vanilla_kd_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_vanilla_kd_eval
```
- `Teacher`
- `Baseline Student`
- `Vanilla KD`
- `CCKD-v3`
- `w/o confidence v3`
- `w/o curriculum v3`

### 6. 教师与学生参数量、推理速度统计

#### 教师

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

#### CCKD-v3

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v3_best.pt --model_type student --batch_size 64
```

#### Vanilla KD

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --model_type student --batch_size 64
```

---

### 7. 教师与学生预测对比图

#### 教师 vs CCKD-v3

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_cckd_v3_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_cckd_v3
```

#### 教师 vs Vanilla KD

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_vanilla_kd_v3paper
```

#### 教师 vs Baseline Student

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_baseline_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_baseline_v3paper
```

---

### 8. 结果汇总表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v3_summary.csv --output_md outputs/reports/metr_v3_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_best.pt" --run "CCKD-v3,student,checkpoints/student/metr_student_cckd_v3_best.pt" --run "w/oConfidence-v3,student,checkpoints/student/metr_student_wo_confidence_v3_best.pt" --run "w/oCurriculum-v3,student,checkpoints/student/metr_student_wo_curriculum_v3_best.pt"
```

输出：

- `outputs/reports/metr_v3_summary.csv`
  中文解释：所有模型结果的表格数据，可直接用于论文主结果表与消融实验表。
- `outputs/reports/metr_v3_summary.md`
  中文解释：Markdown 格式的结果汇总，便于直接复制到文档中。

---

### 9. 精度-效率权衡图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_v3_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_v3_efficiency_tradeoff.png --derived_csv outputs/reports/metr_v3_efficiency_tradeoff.csv
```

输出：

- `outputs/figures/metr_v3_efficiency_tradeoff.png`
  中文解释：论文里常用的精度-效率权衡图，展示教师、基线学生、Vanilla KD 与 CCKD-v3 的对比。
- `outputs/reports/metr_v3_efficiency_tradeoff.csv`
  中文解释：用于作图的中间统计表，可进一步在 Excel 或 Origin 中美化。

---

## 三、PEMS-BAY 全流程

说明：`PEMS-BAY` 的 `v3` 流程和 `METR-LA` 完全一致，只是把：

- `--data data/METR-LA` 改成 `--data data/PEMS-BAY`
- `--adjdata data/sensor_graph/adj_mx.pkl` 改成 `--adjdata data/sensor_graph/adj_mx_bay.pkl`
- 教师权重改成 `bay_teacher_best.pt`
- 学生实验名改成 `bay_...`

另外要特别注意：

`PEMS-BAY` 不能直接复用 `METR-LA` 的教师模型，因为两个数据集的节点数、图结构和数据分布不同，所以必须先单独训练 `PEMS-BAY` 的教师模型。

---

### 1. 训练 PEMS-BAY 教师模型

```powershell
python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --gcn_bool --addaptadj --epochs 50 --batch_size 64 --nhid 32 --learning_rate 0.001 --dropout 0.3 --weight_decay 0.0001 --exp_name bay_teacher
```

训练完成后会生成：

- `checkpoints/teacher/bay_teacher_best.pt`
  中文解释：`PEMS-BAY` 专用教师模型权重，后续学生蒸馏训练必须加载它。
- `outputs/reports/bay_teacher_teacher_history.json`
  中文解释：教师训练过程记录文件。
- `outputs/figures/bay_teacher_teacher_curve.png`
  中文解释：教师模型训练曲线图。

---

### 2. 测试 PEMS-BAY 教师模型

```powershell
[teacher] average -> MAE=1.5851, MAPE=0.0351, RMSE=3.4995, params=311,760
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --plot_sensor 10 --plot_horizon 11 --plot_adaptive_adj --plot_relation --exp_name bay_teacher_eval
```

常见输出：

- `outputs/figures/bay_teacher_eval_teacher_sensor10_h12.png`
  中文解释：教师模型在某个节点和某个预测步上的预测曲线图。
- `outputs/figures/bay_teacher_eval_teacher_adaptive_adj.png`
  中文解释：教师模型自适应邻接矩阵热力图。
- `outputs/figures/bay_teacher_eval_teacher_relation.png`
  中文解释：教师模型节点关系热力图。
- `outputs/predictions/bay_teacher_eval_teacher_sensor10.csv`
  中文解释：教师模型真实值和预测值的数值输出表。

---

### 3. 训练 CCKD-v3

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --exp_name bay_student_cckd_v3
```

训练完成后将生成：

- `checkpoints/student/bay_student_cckd_v3_best.pt`
  中文解释：`PEMS-BAY` 上训练好的 `CCKD-v3` 最优学生模型权重。
- `outputs/reports/bay_student_cckd_v3_student_history.json`
  中文解释：学生训练过程记录文件。
- `outputs/figures/bay_student_cckd_v3_student_curve.png`
  中文解释：学生训练曲线图。

---

### 4. 测试 CCKD-v3

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_cckd_v3_eval
```

常见输出：

- `outputs/figures/bay_student_cckd_v3_eval_student_sensor10_h12.png`
  中文解释：学生模型在某个节点和某个预测步上的预测曲线图。
- `outputs/figures/bay_student_cckd_v3_eval_student_relation.png`
  中文解释：学生模型节点关系热力图。
- `outputs/predictions/bay_student_cckd_v3_eval_student_sensor10.csv`
  中文解释：学生模型真实值和预测值的数值输出表。

---

### 5. 消融：去掉可信度筛选

#### 训练

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --disable_confidence_filter --exp_name bay_student_wo_confidence_v3
```

#### 测试

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_confidence_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_confidence_v3_eval
```

---

### 6. 消融：去掉课程蒸馏

#### 训练

```powershell
python train_student_kd.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --disable_curriculum --exp_name bay_student_wo_curriculum_v3
```

#### 测试

```powershell
python test.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_wo_curriculum_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name bay_student_wo_curriculum_v3_eval
```

---

### 7. 教师与学生参数量、推理速度统计

#### 教师

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/bay_teacher_best.pt --model_type teacher --batch_size 64
```

#### CCKD-v3

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_cckd_v3_best.pt --model_type student --batch_size 64
```

#### Vanilla KD

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --checkpoint checkpoints/student/bay_student_vanilla_kd_best.pt --model_type student --batch_size 64
```

---

### 8. 教师与学生预测对比图

#### 教师 vs CCKD-v3

```powershell
python compare_teacher_student.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --student_checkpoint checkpoints/student/bay_student_cckd_v3_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name bay_teacher_vs_cckd_v3
```

#### 教师 vs Vanilla KD

```powershell
python compare_teacher_student.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --student_checkpoint checkpoints/student/bay_student_vanilla_kd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name bay_teacher_vs_vanilla_kd_v3paper
```

#### 教师 vs Baseline Student

```powershell
python compare_teacher_student.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/bay_teacher_best.pt --student_checkpoint checkpoints/student/bay_student_baseline_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name bay_teacher_vs_baseline_v3paper
```

---

### 9. 结果汇总表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --output_csv outputs/reports/bay_v3_summary.csv --output_md outputs/reports/bay_v3_summary.md --run "Teacher,teacher,checkpoints/teacher/bay_teacher_best.pt" --run "Baseline,student,checkpoints/student/bay_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/bay_student_vanilla_kd_best.pt" --run "CCKD-v3,student,checkpoints/student/bay_student_cckd_v3_best.pt" --run "w/oConfidence-v3,student,checkpoints/student/bay_student_wo_confidence_v3_best.pt" --run "w/oCurriculum-v3,student,checkpoints/student/bay_student_wo_curriculum_v3_best.pt"
```

输出：

- `outputs/reports/bay_v3_summary.csv`
  中文解释：`PEMS-BAY` 数据集上的所有模型结果总表。
- `outputs/reports/bay_v3_summary.md`
  中文解释：适合直接复制到文档中的 Markdown 结果汇总。

---

### 10. 精度-效率权衡图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/bay_v3_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/bay_v3_efficiency_tradeoff.png --derived_csv outputs/reports/bay_v3_efficiency_tradeoff.csv
```

输出：

- `outputs/figures/bay_v3_efficiency_tradeoff.png`
  中文解释：`PEMS-BAY` 上的精度-效率权衡图。
- `outputs/reports/bay_v3_efficiency_tradeoff.csv`
  中文解释：对应的中间统计表。

---

## 四、推荐执行顺序

### METR-LA

1. 复用教师结果
2. 训练 `CCKD-v3`
3. 测试 `CCKD-v3`
4. 训练 `w/o confidence v3`
5. 测试 `w/o confidence v3`
6. 训练 `w/o curriculum v3`
7. 测试 `w/o curriculum v3`
8. 统计参数量和速度
9. 生成教师学生对比图
10. 汇总结果表和精度-效率图

### PEMS-BAY

1. 训练教师模型
2. 测试教师模型
3. 训练 `CCKD-v3`
4. 测试 `CCKD-v3`
5. 训练 `w/o confidence v3`
6. 测试 `w/o confidence v3`
7. 训练 `w/o curriculum v3`
8. 测试 `w/o curriculum v3`
9. 统计参数量和速度
10. 生成教师学生对比图
11. 汇总结果表和精度-效率图

---

## 五、最关键的文件

- [v3_branch_notes.md](/C:/Users/86151/Documents/New%20project/v3_branch_notes.md)
- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [test.py](/C:/Users/86151/Documents/New%20project/test.py)
- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)
- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
- [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)
- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)
