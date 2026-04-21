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
  - w/o reliability: MAE = 3.4736, MAPE = 0.0994, RMSE = 6.6470
  - w/o curriculum: MAE = 3.5261, MAPE = 0.1015, RMSE = 6.8269

## 2026-03-22 Key Conclusion

- Current RMGD-KD is still worse than both Baseline Student and Vanilla KD on METR-LA.
- Therefore the full method is not yet validated.
- The issue is no longer only the checkpoint-selection bug. That bug was fixed, but the full method still underperforms.
- Initial ablation indicates both relation distillation and feature distillation are currently negative contributors.
- Removing feature distillation improves the full method slightly more than removing relation distillation.
- Reliability weighting appears to be the strongest negative contributor among the four added modules.
- Curriculum distillation appears to be the only clearly positive contributor among the four added modules under the current implementation.

## 2026-03-22 Code Fix Already Applied

- `train_student_kd.py` now selects the best student checkpoint by `val_mae` instead of total distillation loss.
- `engine.py` now returns student prediction `mae` during train/eval.
- `utils/plotting.py` now prefers plotting `train_mae` / `val_mae` when available.

## 2026-03-22 Next Recommended Experiments

- Priority 1:
  - current ablations are now complete for:
    - `w/o relation`
    - `w/o feature`
    - `w/o reliability`
    - `w/o curriculum`
- Reason:
  - `Vanilla KD` is still the best student baseline
  - reliability is negative
  - feature is negative
  - relation is negative
  - curriculum is positive, but not enough to offset the negative modules
- After that:
  - compare `Teacher`, `Baseline Student`, `Vanilla KD`, `w/o relation`, `w/o feature`, `w/o curriculum`, `w/o reliability`, and `RMGD-KD`
  - then decide whether to:
    - enter a second-round method cleanup, or
    - switch to a conservative fallback route centered on `Vanilla KD`
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

## 2026-03-24 Second-Round Minimal Fix Applied

- Modified code file:
  - `losses/distillation.py`
- Two targeted fixes were applied:
  - masked invalid targets when computing reliability weights
  - true curriculum masking for soft distillation horizons
- No teacher-architecture change was made.
- No student-architecture change was made.
- Therefore:
  - teacher checkpoint can be reused
  - student-side full-method experiments must be retrained as a new version

## 2026-03-25 RMGD-KD v2 Result

- New `RMGD-KD v2` test result on METR-LA:
  - MAE = 3.4851
  - MAPE = 0.1013
  - RMSE = 6.7346
- Interpretation:
  - clearly better than first-round `RMGD-KD` (3.5175)
  - still slightly worse than `Baseline Student` (3.4762)
  - still worse than `Vanilla KD` (3.4645)
- Conclusion:
  - the minimal fixes helped
  - the suspected implementation details were indeed part of the problem
  - but the full method is still not yet stronger than `Vanilla KD`

## 2026-03-26 Second-Round V2 Ablation Results

- `RMGD-KD v2`:
  - MAE = 3.4851
  - MAPE = 0.1013
  - RMSE = 6.7346
- `w/o reliability v2`:
  - MAE = 3.4802
  - MAPE = 0.0998
  - RMSE = 6.6157
- `w/o curriculum v2`:
  - MAE = 3.4717
  - MAPE = 0.1000
  - RMSE = 6.6716

## 2026-03-26 Updated Interpretation

- After the minimal fixes, the full method improved compared with first-round `RMGD-KD`, so the implementation fixes were meaningful.
- However, `RMGD-KD v2` is still not the best student model.
- `w/o reliability v2` is better than `RMGD-KD v2`, which means reliability weighting still behaves as a negative contributor under the current design.
- `w/o curriculum v2` is also better than `RMGD-KD v2`, which means curriculum is no longer providing a net positive gain in the current second-round setting.
- Among the currently observed student variants:
  - `Vanilla KD` remains the best MAE result at 3.4645
  - `w/o curriculum v2` is the closest among the revised-method variants

## 2026-03-26 Practical Decision Hint

- At this stage, the full four-part method is still not validated.
- A safer paper direction is likely:
  - use `Vanilla KD` as the strongest stable baseline
  - optionally build a simplified method around only the most defensible extra component(s), if any
  - otherwise switch to the conservative fallback narrative

## 2026-03-26 v3 Branch Direction

- The user created a new `v3` branch for a cleaner method redesign.
- v3 method target:
  - keep `Vanilla KD` as the stable backbone
  - keep `curriculum distillation`
  - replace continuous reliability weighting with simpler `confidence filtering`
  - remove `feature distillation`
  - remove `relation distillation`

## 2026-03-26 v3 Code Changes

- Updated:
  - [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
  - [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
  - [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- v3 defaults now represent:
  - hard supervision
  - confidence-filtered soft distillation
  - curriculum distillation
- `feature_weight` and `relation_weight` are now zero by default in v3.
- `feature_adapter` is no longer counted or optimized unless feature/relation alignment is explicitly enabled.

## 2026-03-26 v3 Root Documents

- Branch explanation:
  - [v3_branch_notes.md](/C:/Users/86151/Documents/New%20project/v3_branch_notes.md)
- v3 run workflow:
  - [v3_workflow.md](/C:/Users/86151/Documents/New%20project/v3_workflow.md)

## 2026-03-26 v3 Naming Rule

- All new v3 experiments should explicitly include the `v3` suffix, for example:
  - `metr_student_cckd_v3`
  - `metr_student_wo_confidence_v3`
  - `metr_student_wo_curriculum_v3`

## 2026-03-28 v3 Core Results

- `CCKD-v3`:
  - MAE = 3.4438
  - MAPE = 0.0989
  - RMSE = 6.6185
- `w/o confidence v3`:
  - MAE = 3.4645
  - MAPE = 0.0976
  - RMSE = 6.6382
- `w/o curriculum v3`:
  - MAE = 3.4681
  - MAPE = 0.0988
  - RMSE = 6.6553

## 2026-03-28 v3 Interpretation

- `CCKD-v3` is now better than:
  - `Baseline Student`
  - `Vanilla KD`
  - `w/o confidence v3`
  - `w/o curriculum v3`
- This means the v3 method is validated under the current METR-LA setting.
- Both retained modules show positive contribution:
  - removing confidence filtering hurts MAE
  - removing curriculum also hurts MAE
- By MAE, curriculum contributes slightly more than confidence filtering in the current setting, but both are useful.

## 2026-03-30 v4 Branch Direction

- The user created a new `v4` branch after advisor feedback on `CCKD-v3`.
- Advisor concern:
  - binary confidence filtering looked too much like an engineering trick
  - low-confidence positions should not simply stop receiving teacher guidance
- New v4 method direction:
  - keep hard supervision
  - keep curriculum distillation
  - replace binary confidence filtering with a more academic dual-path mechanism
  - high-confidence positions use absolute-value distillation
  - low-confidence positions use trend distillation instead of being discarded from teacher guidance

## 2026-03-30 v4 Core Method Definition

- Method name:
  - `CCKD-v4`
- Core idea:
  - `L_hard`: supervision from real labels
  - `L_abs`: absolute prediction distillation on high-confidence positions
  - `L_trend`: trend distillation on low-confidence positions
  - `M_curr`: curriculum mask over horizons
- Conceptual form:
  - `L = alpha * L_hard + beta * M_curr * [ c * L_abs + lambda * (1 - c) * L_trend ]`
- This is no longer a simple "distill or drop" strategy.
- This is now a confidence-adaptive dual-path curriculum distillation framework.

## 2026-03-30 v4 Code Changes Applied

- Updated:
  - `losses/distillation.py`
  - `engine.py`
  - `train_student_kd.py`
- Main implementation changes:
  - binary confidence filtering replaced by continuous confidence scoring
  - high-confidence branch uses absolute-value distillation
  - low-confidence branch uses trend distillation
  - curriculum remains active
  - feature and relation distillation remain disabled by default
- Student best checkpoint selection still uses `val_mae`

## 2026-03-30 v4 Root Documents Added

- Branch notes:
  - `v4_branch_notes.md`
- Workflow:
  - `v4_workflow.md`
- AI figure prompts:
  - `v4_ai_figure_prompts.md`
- Paper-intro draft helper:
  - `v4_paper_intro_for_ai.md`
- All four v4 root documents were later rewritten into Chinese so the user can directly read, run, and reuse them for writing.

## 2026-03-30 v4 Experiment Naming Rule

- Recommended METR-LA names:
  - `metr_student_baseline_v4`
  - `metr_student_vanilla_kd_v4`
  - `metr_student_cckd_v4`
  - `metr_student_wo_confidence_v4`
  - `metr_student_wo_curriculum_v4`
- Recommended PEMS-BAY names:
  - `bay_student_baseline_v4`
  - `bay_student_vanilla_kd_v4`
  - `bay_student_cckd_v4`
  - `bay_student_wo_confidence_v4`
  - `bay_student_wo_curriculum_v4`

## 2026-03-30 v4 Execution Scope

- Teacher models do not need retraining for v4 because v4 only changes the student-side distillation mechanism.
- Baseline and Vanilla KD are still valid comparison baselines under the same teacher and student architecture.
- New student-side v4 experiments should be rerun.
- The recommended order is recorded in `v4_workflow.md`.

## 2026-03-24 New Workflow Document

- Added a dedicated second-round workflow document:
  - [v2_minimal_fix_workflow.md](/C:/Users/86151/Documents/New%20project/v2_minimal_fix_workflow.md)
- Purpose:
  - avoid mixing first-round README instructions with second-round revised-method commands
  - provide a clean command order for:
    - `RMGD-KD v2`
    - `w/o reliability v2`
    - `w/o curriculum v2`
    - benchmark
    - teacher-student comparison figures
    - result collection
    - efficiency tradeoff figure
## 2026-03-28 新增记录：相近文献与 v3 图示提示

- 当前 `CCKD-v3` 最新结果：
  - `CCKD-v3`: `MAE=3.4438`, `MAPE=0.0989`, `RMSE=6.6185`
  - `w/o confidence`: `MAE=3.4645`, `MAPE=0.0976`, `RMSE=6.6382`
  - `w/o curriculum`: `MAE=3.4681`, `MAPE=0.0988`, `RMSE=6.6553`
- 当前结论：
  - `CCKD-v3` 已优于 `Baseline Student` 与 `Vanilla KD`
  - `confidence filtering` 与 `curriculum` 两个模块均有正贡献
  - `v3` 已可作为当前论文主方法版本
- 新增根目录图示提示文档：
  - [v3_ai_figure_prompts.md](C:/Users/86151/Documents/New%20project/v3_ai_figure_prompts.md)
  - 该文档专门用于 `CCKD-v3` 的论文图示生成，避免误用旧版四模块方法图
- 根目录运行手册已重写并补齐 `PEMS-BAY`：
  - [v3_workflow.md](C:/Users/86151/Documents/New%20project/v3_workflow.md)
  - 当前该文档已包含：
    - `METR-LA` 全流程
    - `PEMS-BAY` 全流程
    - `PEMS-BAY` 前置教师训练与教师测试
    - 关键输出文件的中文解释
    - `CCKD-v3`
    - `w/o confidence v3`
    - `w/o curriculum v3`
    - benchmark / compare / collect_results / efficiency tradeoff
- 对外文献参考建议：
  - 最接近当前主题的参考写作模板优先考虑“Traffic prediction + Knowledge Distillation”方向文献
  - 重点不是照搬对方方法，而是参考其“摘要-引言-相关工作-方法-实验-消融-结论”的组织方式

## 2026-04-07 Switchable Curriculum Update

- The user explicitly requested that `METR-LA` keep the original curriculum behavior while `PEMS-BAY` can try a weaker optimized curriculum.
- This has now been implemented in `v4` as a switchable curriculum design.

### New CLI option

- `--curriculum_mode standard|short|wide|soft`

### Meaning of each mode

- `standard`
  - preserves the original METR-friendly schedule
- `short`
  - short warm-up, then fully open all horizons
- `wide`
  - opens more horizons earlier
- `soft`
  - keeps all horizons active with softer long-horizon weights

### Files updated

- [distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
- [v4_workflow.md](/C:/Users/86151/Documents/New%20project/v4_workflow.md)
- [v4_branch_notes.md](/C:/Users/86151/Documents/New%20project/v4_branch_notes.md)

### Recommended usage after the update

- `METR-LA`
  - keep using `--curriculum_mode standard`
- `PEMS-BAY`
  - try `--curriculum_mode short` first
  - then compare `wide` if needed

### Scope note

- This is a switchable extension, not a hard replacement.
- The goal is to optimize `PEMS-BAY` without invalidating the existing METR-oriented setup.

## 2026-04-15 README rewrite

The project `README.md` has been fully rewritten to match the current stabilized paper line.

### Current README positioning

- The README now uses **CCKD-v5** as the paper-facing main method name.
- It no longer follows the original RMGD-KD workflow.
- It presents the project as:
  - `Confidence-Adaptive Distillation`
  - `Soft Curriculum`
  - `GWNet teacher + lightweight GCN student`

### What the new README contains

- project overview and contributions
- environment setup
- data preparation
- version notes
- full METR-LA workflow
- full PEMS-BAY workflow
- curriculum mode explanation
- plotting / benchmarking / summary commands
- paper writing pointers and linked v5 docs

### Important note

- The README now serves as the main operational guide.
- Older v2/v3/v4 workflow documents may still exist for history, but the new README is the recommended entry point.
