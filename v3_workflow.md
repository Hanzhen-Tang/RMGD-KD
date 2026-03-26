# v3 运行手册

本手册对应 `v3` 分支。

最上层目标见：

- [v3_branch_notes.md](/C:/Users/86151/Documents/New%20project/v3_branch_notes.md)

本手册只写 `v3` 真实执行顺序，不和旧版 README 混用。

---

## 1. 先进入项目根目录

所有命令都必须在项目根目录执行：

```powershell
cd "C:\Users\86151\Documents\New project"
```

---

## 2. 本轮哪些结果可以直接复用

不需要重跑：

- 教师模型 `Teacher`
- `Baseline Student`
- `Vanilla KD`

需要重跑：

- `CCKD-v3`
- `w/o confidence v3`
- `w/o curriculum v3`

---

## 3. 复用教师模型

本轮继续直接使用已有教师：

```text
checkpoints/teacher/metr_teacher_best.pt
```

不需要重训教师。

---

## 4. 训练 v3 主方法

### 第 1 步：训练 `CCKD-v3`

```powershell
train_total=3.2368, train_mae=3.6028, val_total=2.5242, val_mae=3.0988, soft=2.3828,
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --exp_name metr_student_cckd_v3
```

训练输出重点文件：

- `checkpoints/student/metr_student_cckd_v3_best.pt`
- `outputs/reports/metr_student_cckd_v3_student_history.json`
- `outputs/figures/metr_student_cckd_v3_student_curve.png`

### 第 2 步：测试 `CCKD-v3`

```powershell
[student] average -> MAE=3.4438, MAPE=0.0989, RMSE=6.6185, params=27,404, latency=11.04ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_cckd_v3_eval
```

测试输出重点文件：

- `outputs/figures/metr_student_cckd_v3_eval_student_sensor10_h12.png`
- `outputs/figures/metr_student_cckd_v3_eval_student_relation.png`
- `outputs/predictions/metr_student_cckd_v3_eval_student_sensor10.csv`

---

## 5. v3 消融实验

### 5.1 去掉可信度筛选 `w/o confidence v3`

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --disable_confidence_filter --exp_name metr_student_wo_confidence_v3
```

测试：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_confidence_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_confidence_v3_eval
```

### 5.2 去掉课程蒸馏 `w/o curriculum v3`

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.7 --soft_weight 0.3 --feature_weight 0.0 --relation_weight 0.0 --temperature 3.0 --confidence_keep_ratio 0.7 --disable_curriculum --exp_name metr_student_wo_curriculum_v3
```

测试：

```powershell
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_curriculum_v3_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_curriculum_v3_eval
```

---

## 6. 本轮主结果表建议保留哪些方法

建议最终至少比较：

- `Teacher`
- `Baseline Student`
- `Vanilla KD`
- `CCKD-v3`
- `w/o confidence v3`
- `w/o curriculum v3`

---

## 7. 统计教师与学生参数量、推理速度

### 教师

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

### `CCKD-v3`

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_cckd_v3_best.pt --model_type student --batch_size 64
```

### `Vanilla KD`

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --model_type student --batch_size 64
```

---

## 8. 生成论文可用对比图

### 教师 vs `CCKD-v3`

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_cckd_v3_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_cckd_v3
```

### 教师 vs `Vanilla KD`

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_vanilla_kd_v3paper
```

### 教师 vs `Baseline Student`

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_baseline_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_baseline_v3paper
```

---

## 9. 自动汇总结果表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v3_summary.csv --output_md outputs/reports/metr_v3_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_best.pt" --run "CCKD-v3,student,checkpoints/student/metr_student_cckd_v3_best.pt" --run "w/oConfidence-v3,student,checkpoints/student/metr_student_wo_confidence_v3_best.pt" --run "w/oCurriculum-v3,student,checkpoints/student/metr_student_wo_curriculum_v3_best.pt"
```

输出重点文件：

- `outputs/reports/metr_v3_summary.csv`
- `outputs/reports/metr_v3_summary.md`

---

## 10. 生成时间-性能达成度图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_v3_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_v3_efficiency_tradeoff.png --derived_csv outputs/reports/metr_v3_efficiency_tradeoff.csv
```

---

## 11. 建议执行顺序

按下面顺序最稳：

1. 复用教师结果
2. 训练 `CCKD-v3`
3. 测试 `CCKD-v3`
4. 训练并测试 `w/o confidence v3`
5. 训练并测试 `w/o curriculum v3`
6. 统计教师与学生参数量、速度
7. 生成教师学生对比图
8. 汇总主结果表
9. 生成时间-性能达成度图

---

## 12. 本轮判断标准

如果 `CCKD-v3` 满足下面任一情况，就说明 v3 路线值得继续：

- 超过 `Vanilla KD`
- 接近 `Vanilla KD`，但消融显示创新模块有清晰正贡献
- 在保持较低参数量和较高速度下，结果明显优于 `Baseline Student`

如果 `CCKD-v3` 仍明显不如 `Vanilla KD`，则保底路线仍然成立：

- 以 `Vanilla KD` 为最稳主方法
- 将 v3 实验作为“可信度筛选与课程蒸馏适配性分析”

---

## 13. 本轮最需要看的文件

- [v3_branch_notes.md](/C:/Users/86151/Documents/New%20project/v3_branch_notes.md)
- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [test.py](/C:/Users/86151/Documents/New%20project/test.py)
- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)
- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
- [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)
- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)
