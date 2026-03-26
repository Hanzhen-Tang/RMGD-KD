# v3 Branch Notes

## 分支更改目标

本分支的目标是将原来的四模块 `RMGD-KD` 收缩成一个更稳定、也更容易讲清楚的版本：

- 以 `Vanilla KD` 作为稳定主干
- 保留 `curriculum distillation`
- 将原来的连续 reliability 加权改成更简单的 **可信度筛选蒸馏**
- 移除 `feature distillation`
- 移除 `relation distillation`

简化后的 v3 方法关键词：

- `Confidence Filtering`
- `Curriculum Distillation`
- `Lightweight Student`

建议工作名：

- `CCKD-v3`
- 全称可写为：`Confidence-Filtered Curriculum Knowledge Distillation for Traffic Forecasting`

---

## v3 与旧版本的关系

### 相比第一轮四模块版本

v3 删除了：

- feature distillation
- relation distillation

### 相比第二轮 v2 修正版

v3 不再把 reliability 作为连续加权图来用，而是改成：

- 仅蒸馏教师更可信的位置
- 通过 top-k 过滤降低低质量教师知识的负迁移

---

## v3 默认训练目标

v3 默认损失为：

```text
L = alpha * L_hard + beta * L_conf_soft
```

其中：

- `L_hard`：学生对真实标签的监督损失
- `L_conf_soft`：经过可信度筛选与课程蒸馏后的软蒸馏损失

默认不再启用：

- feature loss
- relation loss

---

## v3 主要代码变动文件

- [train_student_kd.py](/C:/Users/86151/Documents/New%20project/train_student_kd.py)
- [engine.py](/C:/Users/86151/Documents/New%20project/engine.py)
- [losses/distillation.py](/C:/Users/86151/Documents/New%20project/losses/distillation.py)

---

## v3 结果命名原则

本分支中所有新实验都建议显式带上 `v3` 后缀，例如：

- `metr_student_cckd_v3`
- `metr_student_wo_confidence_v3`
- `metr_student_wo_curriculum_v3`

这样方便和旧版本区分。
