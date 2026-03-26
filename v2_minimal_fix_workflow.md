# V2 最小修正版运行手册

本手册对应第二轮最小修正版方法。

本版本只修了两处核心细节：

1. `reliability map` 计算时加入有效值掩码，避免 `real == 0` 的无效位置污染可靠性权重。
2. `curriculum distillation` 在未开放的 horizon 上真正不参与软蒸馏，而不是仅给极小权重。

适用场景：

- 你已经完成第一轮实验；
- 教师模型已经训练好；
- `Baseline Student`、`Vanilla KD`、旧版 `RMGD-KD` 已有结果；
- 现在要做第二轮最小修正版验证。

---

## 1. 本版本与旧 README 的关系

- 旧版 [README.md](/C:/Users/86151/Documents/New%20project/README.md) 主要记录第一轮版本流程。
- 本文档只服务于第二轮最小修正版，不与旧版结果混用。
- 本轮建议统一用新的实验名后缀：
  - `*_v2`

---

## 2. 本轮建议保留不变的部分

本轮不需要重训教师，也不需要重跑第一轮这些基础结果：

- 教师模型 `Teacher`
- 学生基线 `Baseline Student`
- 普通蒸馏 `Vanilla KD`

本轮需要重跑的是：

- 新版 `RMGD-KD`
- 如有需要，再跑新版消融：
  - `w/o reliability`
  - `w/o curriculum`

---

## 3. 教师模型结果复用

如果你已经有教师模型：

- `checkpoints/teacher/metr_teacher_best.pt`

则本轮继续直接复用，不需要重新训练教师。

如果你还没有教师结果，可以先按旧 README 的教师流程训练和测试教师。

---

## 4. 第二轮核心训练流程

### 第 1 步：训练新版完整方法 RMGD-KD

```powershell
train_total=2.6695, train_mae=3.6439, val_total=2.1233, val_mae=3.1292, soft=2.4030, feat=0.0046, rel=0.0208, visible_h=12, val_latency=146.07ms
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --exp_name metr_student_rmgd_v2
```

训练完成后，重点文件：

- `checkpoints/student/metr_student_rmgd_v2_best.pt`
- `outputs/reports/metr_student_rmgd_v2_student_history.json`
- `outputs/figures/metr_student_rmgd_v2_student_curve.png`

### 第 2 步：测试新版完整方法 RMGD-KD

```powershell
average -> MAE=3.4851, MAPE=0.1013, RMSE=6.7346, params=27,404, latency=15.19ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_v2_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_rmgd_v2_eval
```

测试后重点输出：

- `outputs/figures/metr_student_rmgd_v2_eval_student_sensor10_h12.png`
- `outputs/figures/metr_student_rmgd_v2_eval_student_relation.png`
- `outputs/predictions/metr_student_rmgd_v2_eval_student_sensor10.csv`

---

## 5. 第二轮推荐消融

### 5.1 新版 w/o reliability

训练：

```powershell
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --disable_reliability --exp_name metr_student_wo_reliability_v2
```

测试：

```powershell 
average -> MAE=3.4802, MAPE=0.0998, RMSE=6.6157, params=27,404, latency=17.45ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_reliability_v2_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_reliability_v2_eval
```

### 5.2 新版 w/o curriculum

训练：

```powershell
train_total=2.6380, train_mae=3.6151, val_total=2.1009, val_mae=3.1314, soft=2.3318, feat=0.0047, rel=0.0205, visible_h=12
python train_student_kd.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --epochs 50 --batch_size 64 --student_hidden_dim 32 --student_layers 2 --hard_weight 0.6 --soft_weight 0.2 --feature_weight 0.1 --relation_weight 0.1 --temperature 3.0 --disable_curriculum --exp_name metr_student_wo_curriculum_v2
```

测试：

```powershell
average -> MAE=3.4717, MAPE=0.1000, RMSE=6.6716, params=27,404, latency=13.63ms/batch
python test.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_wo_curriculum_v2_best.pt --model_type student --plot_sensor 10 --plot_horizon 11 --plot_relation --exp_name metr_student_wo_curriculum_v2_eval
```

---

## 6. 本轮对照表建议

第二轮建议至少整理下面 5 组：

- `Teacher`
- `Baseline Student`
- `Vanilla KD`
- `RMGD-KD v2`
- `w/o reliability v2`
- `w/o curriculum v2`

如果你还想和第一轮做对照，也可以额外保留：

- 旧版 `RMGD-KD`
- 旧版 `w/o relation`
- 旧版 `w/o feature`

但正式论文主结果不要把第一轮和第二轮方法版本混在同一表里，需要注明版本。

---

## 7. 统计教师与学生参数量、推理速度

### 7.1 统计教师

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/teacher/metr_teacher_best.pt --model_type teacher --batch_size 64
```

### 7.2 统计新版学生

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_rmgd_v2_best.pt --model_type student --batch_size 64
```

如果你还要统计 `Vanilla KD`，命令类似：

```powershell
python scripts/benchmark_model.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --model_type student --batch_size 64
```

---

## 8. 生成论文可用对比图

### 8.1 教师 vs 新版学生对比曲线图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_rmgd_v2_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_rmgd_v2
```

输出重点文件：

- `outputs/figures/metr_teacher_vs_rmgd_v2_sensor10_h12.png`
- `outputs/predictions/metr_teacher_vs_rmgd_v2_sensor10_h12.csv`

### 8.2 教师 vs Vanilla KD 对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_vanilla_kd_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_vanilla_kd
```

### 8.3 教师 vs Baseline Student 对比图

```powershell
python compare_teacher_student.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --teacher_checkpoint checkpoints/teacher/metr_teacher_best.pt --student_checkpoint checkpoints/student/metr_student_baseline_best.pt --plot_sensor 10 --plot_horizon 11 --exp_name metr_teacher_vs_baseline
```

---

## 9. 自动汇总结果表

```powershell
python scripts/collect_results.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --output_csv outputs/reports/metr_v2_summary.csv --output_md outputs/reports/metr_v2_summary.md --run "Teacher,teacher,checkpoints/teacher/metr_teacher_best.pt" --run "Baseline,student,checkpoints/student/metr_student_baseline_best.pt" --run "VanillaKD,student,checkpoints/student/metr_student_vanilla_kd_best.pt" --run "RMGD-KD-v2,student,checkpoints/student/metr_student_rmgd_v2_best.pt" --run "w/oReliability-v2,student,checkpoints/student/metr_student_wo_reliability_v2_best.pt" --run "w/oCurriculum-v2,student,checkpoints/student/metr_student_wo_curriculum_v2_best.pt"
```

输出重点文件：

- `outputs/reports/metr_v2_summary.csv`
- `outputs/reports/metr_v2_summary.md`

---

## 10. 生成时间-性能达成度图

```powershell
python scripts/plot_efficiency_tradeoff.py --summary_csv outputs/reports/metr_v2_summary.csv --teacher_name Teacher --metric MAE --save_path outputs/figures/metr_v2_efficiency_tradeoff.png --derived_csv outputs/reports/metr_v2_efficiency_tradeoff.csv
```

输出重点文件：

- `outputs/figures/metr_v2_efficiency_tradeoff.png`
- `outputs/reports/metr_v2_efficiency_tradeoff.csv`

---

## 11. 本轮推荐执行顺序

建议严格按下面顺序走：

1. 复用现有教师，不重训教师
2. 训练 `RMGD-KD v2`
3. 测试 `RMGD-KD v2`
4. 训练并测试 `w/o reliability v2`
5. 训练并测试 `w/o curriculum v2`
6. 统计教师与学生参数量、推理速度
7. 生成教师与学生对比图
8. 自动汇总结果表
9. 生成时间-性能达成度图

---

## 12. 本轮结果怎么判断

如果第二轮满足下面任意一种情况，就说明最小修正是有价值的：

- `RMGD-KD v2` 明显优于第一轮 `RMGD-KD`
- `RMGD-KD v2` 接近或超过 `Vanilla KD`
- `w/o reliability v2` 和 `w/o curriculum v2` 的结果更符合方法预期

如果第二轮仍明显不如 `Vanilla KD`，则建议转入保底路线：

- 以 `Vanilla KD` 为学生主方法
- 将第二轮实验作为“模块有效性与失效原因分析”

---

## 13. 需要优先查看的文件

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [test.py](/C:/Users/86151/Documents/New%20project/test.py)
- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)
- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
- [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)
- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)
