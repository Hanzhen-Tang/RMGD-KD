# v4 版本说明

## 本次改了什么

本分支是在 `v3` 的基础上继续升级得到的 `v4` 版本。

`v3` 的核心思想是：

- 高可信位置参与蒸馏
- 低可信位置不参与蒸馏

老师提出的问题是：

- 这种做法更像一种工程筛选手段
- 低可信位置虽然没有删除训练，但在软蒸馏中等于没有继续接受教师指导
- 如果低可信位置能够采用另一种更合适的“教学方式”，会更有学术性

因此，`v4` 的核心修改是：

- 不再使用“二值筛选后直接不蒸馏”的策略
- 改为“**可信度感知双路径蒸馏**”

具体来说：

- 高可信位置：蒸馏教师的**绝对预测值**
- 低可信位置：蒸馏教师的**变化趋势**
- 课程蒸馏继续保留
- 真实标签监督继续保留

也就是说，`v4` 不再是：

- 学或者不学

而是：

- **根据可信度不同，学习不同类型的教师知识**

---

## v4 方法概括

`v4` 的最终方法可以概括为：

**可信度感知双路径课程蒸馏**

它包含 4 个核心组成部分：

1. `L_hard`
   学生直接学习真实标签

2. `L_abs`
   高可信位置学习教师绝对输出

3. `L_trend`
   低可信位置学习教师预测趋势

4. `M_curr`
   课程蒸馏掩码，控制 horizon 由短到长逐步开放

整体可以写成：

```text
L = alpha * L_hard + beta * M_curr * [ c * L_abs + lambda * (1 - c) * L_trend ]
```

其中：

- `c` 为连续可信度分数，范围在 `[0,1]`
- `alpha`、`beta` 为监督与蒸馏权重
- `lambda` 为趋势蒸馏权重

---

## 本次主要修改的代码文件

- `losses/distillation.py`
  - 改成连续可信度 + 双路径蒸馏

- `engine.py`
  - 让训练器与新的 `v4` 损失项保持一致

- `train_student_kd.py`
  - 更新参数入口、日志字段、checkpoint 保存信息

---

## 没有大改的文件

下面这些文件的主逻辑没有因为 `v4` 而重写：

- `models/teacher_gwnet.py`
- `models/student_gcn.py`
- `test.py`
- `compare_teacher_student.py`
- `scripts/benchmark_model.py`
- `scripts/collect_results.py`
- `scripts/plot_efficiency_tradeoff.py`

---

## v4 命名建议

建议所有新的实验结果都显式带上 `v4` 后缀，避免和 `v3` 混淆。

例如：

- `metr_student_cckd_v4`
- `metr_student_wo_confidence_v4`
- `metr_student_wo_curriculum_v4`
- `bay_student_cckd_v4`
- `bay_student_wo_confidence_v4`
- `bay_student_wo_curriculum_v4`

---

## v4 的实验逻辑

每个数据集建议按下面的顺序组织实验：

1. Teacher
2. Baseline Student
3. Vanilla KD
4. CCKD-v4
5. w/o confidence
6. w/o curriculum

这样就能完整支撑：

- 主结果对比
- 消融实验
- 精度-效率分析
- 教师学生可视化对比

---

## 备注

- `v4` 只修改了学生端的蒸馏机制，所以**教师模型不需要重训**
- 但 `v4` 的学生方法属于新方法，所以**学生侧方法需要重新训练**

## 2026-04-07 课程蒸馏可切换升级

为避免 `PEMS-BAY` 上较强课程蒸馏限制前期学习，同时不影响 `METR-LA` 已验证有效的默认策略，`v4` 继续升级为“可切换课程模式”版本：

- 新增参数：
  - `--curriculum_mode standard|short|wide|soft`
- 默认值：
  - `standard`
- 设计原则：
  - `METR-LA` 保持原来的 `standard`
  - `PEMS-BAY` 可以尝试 `short` / `wide` / `soft`

当前最推荐的使用方式：

- `METR-LA`
  - `--curriculum_mode standard`
- `PEMS-BAY`
  - 优先试 `--curriculum_mode short`
