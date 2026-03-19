# 论文结果表模板

这份文档是给你后面整理论文结果用的。

你跑完实验之后，可以直接把结果往这里填。

建议使用方式：

1. 先用 [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py) 自动生成结果汇总表
2. 再把关键数字整理到这里
3. 最后再根据论文模板美化

## 1. 主结果表模板

建议用于对比：

- 教师模型
- Baseline Student
- Vanilla KD
- 完整方法

```markdown
| Model | Params | Compression Ratio | Latency (ms) | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| GWNet Teacher |  | 1.00x |  |  |  |  |
| Student Baseline |  |  |  |  |  |  |
| Student + Vanilla KD |  |  |  |  |  |  |
| Student + RMGD-KD |  |  |  |  |  |  |
```

## 2. 消融实验表模板

建议用于对比：

- 完整模型
- 去掉可靠性蒸馏
- 去掉课程蒸馏
- 去掉关系蒸馏
- 去掉特征蒸馏

```markdown
| Method Variant | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| Full RMGD-KD |  |  |  |
| w/o Reliability |  |  |  |
| w/o Curriculum |  |  |  |
| w/o Relation Distillation |  |  |  |
| w/o Feature Distillation |  |  |  |
```

## 3. 学生容量对比表模板

如果你后面做学生大小对比，可以用这个表。

```markdown
| Student Setting | Params | Latency (ms) | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- | --- |
| hidden=16, layers=2 |  |  |  |  |  |
| hidden=32, layers=2 |  |  |  |  |  |
| hidden=48, layers=2 |  |  |  |  |  |
| hidden=64, layers=2 |  |  |  |  |  |
```

## 4. 温度与蒸馏权重对比表模板

如果你做调参实验，可以用：

```markdown
| Setting | hard | soft | feature | relation | temp | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp-1 |  |  |  |  |  |  |  |  |
| Exp-2 |  |  |  |  |  |  |  |  |
| Exp-3 |  |  |  |  |  |  |  |  |
```

## 5. 效率分析表模板

```markdown
| Model | Params | Compression Ratio | Latency (ms) | MAE |
| --- | --- | --- | --- | --- |
| GWNet Teacher |  | 1.00x |  |  |
| Student Baseline |  |  |  |  |
| Student + Vanilla KD |  |  |  |  |
| Student + RMGD-KD |  |  |  |  |
```

## 6. 图编号登记表模板

这个表是为了防止你后面画了很多图，自己也乱掉。

```markdown
| Figure ID | File Path | Description | Whether Used in Paper |
| --- | --- | --- | --- |
| Fig-1 | outputs/figures/... | Overall framework | Yes/No |
| Fig-2 | outputs/figures/... | Teacher prediction curve | Yes/No |
| Fig-3 | outputs/figures/... | Student prediction curve | Yes/No |
| Fig-4 | outputs/figures/... | Teacher vs Student comparison | Yes/No |
| Fig-5 | outputs/figures/... | Adaptive adjacency heatmap | Yes/No |
| Fig-6 | outputs/figures/... | Node relation heatmap | Yes/No |
```

## 7. 结果填写建议

### 第一步

先把这些模型都跑出来：

- Teacher
- Baseline Student
- Vanilla KD
- w/o Reliability
- w/o Curriculum
- w/o Relation
- w/o Feature
- Full RMGD-KD

### 第二步

运行：

- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
- [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)
- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)

### 第三步

把自动输出的表复制到这里。

### 第四步

再把你想放论文的那部分结果单独精简出来。

## 8. 推荐最终保留在论文里的图

如果篇幅有限，建议至少保留：

- 教师与学生对比图
- 教师自适应邻接矩阵热力图
- 节点关系热力图
- 训练曲线图

## 9. 对应脚本

- 对比图：
  [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)

- 测试图：
  [test.py](/C:/Users/86151/Documents/New%20project/test.py)

- 结果汇总：
  [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)

- 效率统计：
  [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)

- 时间-性能达成度图：
  [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)

## 10. 时间-性能达成度表模板

```markdown
| Model | Latency (ms) | Time (% of Teacher) | MAE | Performance Reached (% of Teacher) | Speedup |
| --- | --- | --- | --- | --- | --- |
| GWNet Teacher |  | 100.00 |  | 100.00 | 1.00x |
| Student Baseline |  |  |  |  |  |
| Student + Vanilla KD |  |  |  |  |  |
| Student + RMGD-KD |  |  |  |  |  |
```
