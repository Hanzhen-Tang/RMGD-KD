# 项目完整说明与记忆文档

这份文档的目的不是给第一次使用的人看，而是给“以后重新接手这个项目的人”看。

如果你换了电脑、换了环境、重新打开项目，或者以后希望让我快速恢复上下文，请优先把这份文档给我。

## 1. 项目目标

本项目服务于一个交通预测与知识蒸馏结合的实验。

目标是：

- 使用 `GWNet` 作为教师模型
- 设计一个轻量学生模型 `GCN`
- 通过知识蒸馏把教师知识迁移给学生
- 在公开交通数据集 `METR-LA` 和 `PEMS-BAY` 上开展实验
- 结果要可以做可视化
- 整个工程要适合写论文和做消融

## 2. 当前项目的论文定位

当前项目不再只是一个“普通 KD baseline”，而是一个轻量论文原型方法。

现在的论文方法名可以暂时叫：

`RMGD-KD`

全称可写为：

`Reliability-aware Multi-Granularity Graph Distillation for Lightweight Traffic Forecasting`

中文可以写成：

`面向轻量化交通预测的可靠性引导多粒度图蒸馏方法`

## 3. 当前已经实现的创新点

### 创新点 1：可靠性加权蒸馏

实现位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

核心思想：

- 教师在不同节点、不同预测步上的可靠性不同
- 教师误差越小的位置，蒸馏权重越大
- 通过教师误差构造节点权重和 horizon 权重

对应函数：

- `compute_reliability_map`

### 创新点 2：图关系蒸馏

实现位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

核心思想：

- 不仅蒸馏预测值，还蒸馏空间关系知识
- 用隐藏特征构造节点关系矩阵
- 让学生关系矩阵逼近教师关系矩阵

对应函数：

- `compute_relation_matrix`

### 创新点 3：多步课程蒸馏

实现位置：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

核心思想：

- 交通预测短期步容易，长期步更难
- 训练前期只蒸馏短期 horizon
- 后期逐渐扩展到全部 horizon

对应函数：

- `compute_curriculum_map`

## 4. 模型结构

### 教师模型

教师使用：

- `GWNet`

代码位置：

- [models/teacher_gwnet.py](/C:/Users/86151/Documents/New%20project/models/teacher_gwnet.py)

训练入口：

- [train.py](/C:/Users/86151/Documents/New%20project/train.py)

### 学生模型

学生使用：

- 轻量 `GCN`

代码位置：

- [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)

训练入口：

- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

## 5. 损失函数组成

当前总损失由四部分组成：

```text
L_total = a * L_hard + b * L_soft + c * L_feature + d * L_relation
```

其中：

- `L_hard`：学生预测与真实值之间的监督损失
- `L_soft`：学生与教师预测之间的软蒸馏损失
- `L_feature`：学生与教师隐藏特征之间的特征蒸馏损失
- `L_relation`：学生与教师节点关系矩阵之间的关系蒸馏损失

默认权重大致为：

- `a = 0.6`
- `b = 0.2`
- `c = 0.1`
- `d = 0.1`

## 6. 张量维度约定

这是项目最关键的基础约定。

### 原始输入

数据集中的输入 `x`：

```text
[B, T_in, N, C_in]
```

### 送入模型前

经过转置：

```text
[B, C_in, N, T_in]
```

### 教师输出

教师原始输出：

```text
[B, H_out, N, 1]
```

转置后用于蒸馏：

```text
[B, 1, N, H_out]
```

### 真实标签

真实标签：

```text
[B, N, H_out]
```

扩维后：

```text
[B, 1, N, H_out]
```

### 学生输出

学生输出也整理到：

```text
[B, 1, N, H_out]
```

所以教师、学生、真实值在计算损失时是维度对齐的。

## 7. 数据集

支持两个公开数据集：

- `METR-LA`
- `PEMS-BAY`

原始数据处理脚本：

- [generate_training_data.py](/C:/Users/86151/Documents/New%20project/generate_training_data.py)

## 8. 当前已实现的脚本功能

### 数据预处理

- [generate_training_data.py](/C:/Users/86151/Documents/New%20project/generate_training_data.py)

作用：

- 从 `h5` 文件生成 `train.npz / val.npz / test.npz`

### 教师训练

- [train.py](/C:/Users/86151/Documents/New%20project/train.py)

作用：

- 训练 GWNet 教师模型

### 学生蒸馏训练

- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

作用：

- 训练学生模型
- 接入可靠性蒸馏
- 接入关系蒸馏
- 接入课程蒸馏
- 支持消融开关

### 测试与可视化

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)

作用：

- 输出测试指标
- 画预测曲线
- 画教师自适应邻接矩阵
- 画节点关系热力图

### 参数量与速度分析

- [scripts/benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)

作用：

- 统计模型参数量
- 统计平均推理时间

### 教师与学生对比图

- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)

作用：

- 在同一张图上画出真实值、教师预测、学生预测
- 用于直观比较蒸馏前后效果

### 自动汇总结果表

- [scripts/collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)

作用：

- 自动评估多个教师/学生模型
- 输出 `CSV`
- 输出 `Markdown 表格`
- 方便直接整理论文结果表

### 调参指南

- [docs/tuning_guide.md](/C:/Users/86151/Documents/New%20project/docs/tuning_guide.md)

作用：

- 给出调参优先级
- 给出推荐范围
- 给出不同现象下的调参方向

### 论文结果表模板

- [docs/result_table_template.md](/C:/Users/86151/Documents/New%20project/docs/result_table_template.md)

作用：

- 整理主结果表
- 整理消融实验表
- 整理效率分析表
- 登记图编号和文件路径

### 维度检查

- [scripts/sanity_check.py](/C:/Users/86151/Documents/New%20project/scripts/sanity_check.py)

作用：

- 检查教师和学生输出维度是否对齐

## 9. 当前支持的消融开关

在 [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py) 中支持：

- `--disable_reliability`
- `--disable_curriculum`
- `--relation_weight 0.0`
- `--feature_weight 0.0`
- `--soft_weight 0.0`

因此可以直接构建：

- Baseline Student
- Vanilla KD
- w/o reliability
- w/o curriculum
- w/o relation
- w/o feature
- Full RMGD-KD

## 10. 当前已经写好的文档

### 项目结构图和数据流图

- [docs/architecture.md](/C:/Users/86151/Documents/New%20project/docs/architecture.md)
- [docs/paper_diagrams.md](/C:/Users/86151/Documents/New%20project/docs/paper_diagrams.md)

### 论文框架建议

- [docs/paper_framework.md](/C:/Users/86151/Documents/New%20project/docs/paper_framework.md)

### 傻瓜版实验流程

- [docs/quickstart_foolproof.md](/C:/Users/86151/Documents/New%20project/docs/quickstart_foolproof.md)

### 数据流说明

- [docs/dataflow_explained.md](/C:/Users/86151/Documents/New%20project/docs/dataflow_explained.md)

### 调参指南

- [docs/tuning_guide.md](/C:/Users/86151/Documents/New%20project/docs/tuning_guide.md)

### 论文结果表模板

- [docs/result_table_template.md](/C:/Users/86151/Documents/New%20project/docs/result_table_template.md)

### 主 README

- [README.md](/C:/Users/86151/Documents/New%20project/README.md)

## 11. 当前最推荐的实验顺序

1. 先用 `generate_training_data.py` 生成 `METR-LA`
2. 跑 `scripts/sanity_check.py`
3. 跑 `train.py` 训练教师
4. 跑 `test.py` 测试教师并保存图
5. 跑 `train_student_kd.py` 训练完整方法
6. 跑 `test.py` 测试完整方法
7. 跑 `scripts/benchmark_model.py` 做效率分析
8. 跑各种消融命令
9. 整理成论文表格和图

## 12. 如果以后继续扩展，最优先改哪里

### 如果要改学生模型

改：

- [models/student_gcn.py](/C:/Users/86151/Documents/New%20project/models/student_gcn.py)

### 如果要改蒸馏损失

改：

- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

### 如果要改训练流程

改：

- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)

### 如果要加更多图

改：

- [test.py](/C:/Users/86151/Documents/New%20project/test.py)
- [utils/plotting.py](/C:/Users/86151/Documents/New%20project/utils/plotting.py)

## 13. 当前环境验证状态

目前已经做过：

- `python -m compileall .`

说明：

- 语法层面通过

但还没有在当前对话环境里实际跑训练，因为当前环境里没有现成的 `torch` 运行条件。

所以如果你本机准备跑真实实验，第一步依然建议：

```powershell
pip install torch
python scripts/sanity_check.py
```

## 14. 这份文档怎么用

以后如果你换设备、换环境、重新打开这个项目，最简单的做法是：

1. 把这个项目目录重新放好
2. 打开这份文档
3. 再把这份文档发给我
4. 我就能非常快地恢复当前项目上下文

---

一句话总结：

这不是一个普通的 GWNet 复现工程，而是一个已经加入“可靠性加权蒸馏 + 图关系蒸馏 + 多步课程蒸馏”的轻量交通预测论文原型工程。

## 15. 新增的时间-性能达成度图脚本

脚本位置：
- [scripts/plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)

作用：
- 从 `scripts/collect_results.py` 生成的汇总 CSV 直接画图
- 用来表达“学生用了教师多少时间，达到了教师多少预测性能”

默认公式：
- `TimePctOfTeacher = model_latency / teacher_latency * 100`
- `PerformancePctOfTeacher = teacher_metric / model_metric * 100`

说明：
- 当前默认使用 `MAE`，也可以切换成 `RMSE` 或 `MAPE`
- 因为这些都是误差指标，所以越小越好
- 当学生与教师表现相同时，性能达成度为 `100%`
- 当学生更差时，低于 `100%`
- 当学生更好时，高于 `100%`

论文定位：
- 适合作为“效率分析图”或“精度-效率折中图”
- 很适合放在主结果表之后

## 16. 新增的代码详细解释文档

文档位置：
- [docs/code_detailed_explanation.md](/C:/Users/86151/Documents/New%20project/docs/code_detailed_explanation.md)

这份文档的作用：
- 解释每个核心模块是干什么的
- 明确教师模型在哪一块
- 明确学生模型在哪一块
- 明确蒸馏逻辑在哪一块
- 明确创新点落在哪一块
- 解释训练时的数据流和损失组成

以后如果重新接手项目，除了看本记忆文档，还建议优先看：
- [docs/code_detailed_explanation.md](/C:/Users/86151/Documents/New%20project/docs/code_detailed_explanation.md)

## 17. 新增的论文写作版代码说明

文档位置：
- [docs/paper_method_writing_guide.md](/C:/Users/86151/Documents/New%20project/docs/paper_method_writing_guide.md)

这份文档的作用：
- 把当前工程整理成更适合论文方法章节的语言
- 说明教师模型、学生模型、蒸馏框架和创新点该怎么写
- 给出损失函数、训练策略、章节结构和贡献点写法

使用场景：
- 写论文方法章节
- 写开题或中期答辩材料
- 后续需要快速把代码翻译成论文语言时直接参考

## 18. README 已补全 PEMS-BAY 后半流程

当前 [README.md](/C:/Users/86151/Documents/New%20project/README.md) 中，`PEMS-BAY` 部分已经不再只有训练主链，现已补全：

- 顺序已调整为与 `METR-LA` 一致，即先教师训练、再教师测试、再学生训练与学生测试
- 教师测试
- 学生测试
- 参数量和速度统计
- 教师学生对比图
- 结果汇总
- 时间-性能达成度图
- 全套消融实验命令

以后如果用户问“PEMS-BAY 是否和 METR-LA 一样只需替换路径”，答案是：

- 流程相同
- 但现在 README 已经给出完整的 PEMS-BAY 版本命令，不必再手动自行推导

## 19. 新增简洁中文版框架图文档

文档位置：
- [docs/simple_paper_diagrams_cn.md](/C:/Users/86151/Documents/New%20project/docs/simple_paper_diagrams_cn.md)

这份文档的作用：
- 提供比原来更简洁的中文版论文结构图
- 采用“大模块相连 + 每个模块再细化”的方式
- 适合直接在 draw.io、Visio、ProcessOn、PPT 中重画

与原有图文档的区别：
- [docs/paper_diagrams.md](/C:/Users/86151/Documents/New%20project/docs/paper_diagrams.md)
  更详细，线条更多，适合完整表达
- [docs/simple_paper_diagrams_cn.md](/C:/Users/86151/Documents/New%20project/docs/simple_paper_diagrams_cn.md)
  更简洁，适合论文排版和人工重画
## 2026-03-22 RMGD-KD Current Status

- Teacher on METR-LA:
  - MAE = 3.0417
  - MAPE = 0.0830
  - RMSE = 6.0484
  - params = 309,400
  - latency = 54.40 ms/batch
- Student comparison on the same fixed student structure:
  - Baseline Student: MAE = 3.4762, MAPE = 0.1020, RMSE = 6.6793
  - Vanilla KD: MAE = 3.4645, MAPE = 0.0999, RMSE = 6.6605
  - RMGD-KD (old checkpoint selection): MAE = 3.5549, MAPE = 0.1034, RMSE = 6.9349
  - RMGD-KD (selected by val_mae after fix): MAE = 3.5175, MAPE = 0.1023, RMSE = 6.7120
  - w/o relation: MAE = 3.5102, MAPE = 0.0986, RMSE = 6.7805
  - w/o feature: MAE = 3.5048, MAPE = 0.1009, RMSE = 6.7468

## 2026-03-22 Key Conclusion

- Current RMGD-KD is still worse than both Baseline Student and Vanilla KD on METR-LA.
- Therefore the full method is not yet validated.
- The issue is no longer only the checkpoint-selection bug. That bug was fixed, but the full method still underperforms.
- Initial ablation indicates both relation distillation and feature distillation are currently negative contributors.
- Removing feature distillation improves the full method slightly more than removing relation distillation.

## 2026-03-22 Code Fix Already Applied

- `train_student_kd.py` now selects the best student checkpoint by `val_mae` instead of total distillation loss.
- `engine.py` now returns student prediction `mae` during train/eval.
- `utils/plotting.py` now prefers plotting `train_mae` / `val_mae` when available.

## 2026-03-22 Next Recommended Experiments

- Priority 1:
  - run `w/o curriculum`
  - run `w/o reliability`
- Reason:
  - `Vanilla KD` is better than current `RMGD-KD`
  - relation distillation and feature distillation have already shown negative contribution under the current setting
  - reliability weighting and curriculum distillation still need to be isolated
- After that:
  - compare `Teacher`, `Baseline Student`, `Vanilla KD`, `w/o relation`, `w/o feature`, `w/o curriculum`, `w/o reliability`, and `RMGD-KD`
- Only after the core method becomes reasonable should the user continue with later paper-packaging steps such as benchmark summaries, efficiency tradeoff figure, and full result table polishing.

## 2026-03-23 Review Before Further Code Changes

- The user decided not to modify the method code immediately.
- Current plan:
  - first finish the remaining two ablations:
    - `w/o curriculum`
    - `w/o reliability`
  - then decide whether to enter a second-round method revision
- Reason:
  - changing the method definition now would invalidate most current full-method ablation results as final-paper evidence
  - it is better to finish the current version's diagnostic ablations first

## 2026-03-23 Likely Method-Level Weak Points Identified (Not Yet Modified)

- Reliability map currently does not use masked invalid values (`real == 0`) when computing teacher error.
- Curriculum distillation currently does not fully shut off future horizons because the weighting path adds a tiny epsilon before normalization.
- Feature distillation may be semantically misaligned:
  - teacher feature comes from a much deeper high-level representation
  - student feature comes from a much shallower lightweight readout
- Relation distillation may be too late-stage and over-smoothed because it is computed on already time-collapsed features.

## 2026-03-23 Decision Rule

- If `w/o curriculum` and `w/o reliability` are also poor:
  - do not rush into weight tuning only
  - first consider a second-round method cleanup
  - top priority fixes to consider next:
    - masked reliability map
    - true curriculum masking
- If one of those ablations becomes clearly better than current `RMGD-KD`:
  - isolate that module as a likely problem source
  - decide whether to simplify the final method rather than keep all modules
