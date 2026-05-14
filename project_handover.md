# CCKD Paper Handover - Current State

Last updated: 2026-05-14

This document is the quick handover file for a future model or a new account with no chat memory. Read this file first. The longer file `docs/project_full_memory.md` contains historical development notes, including abandoned versions, so its older sections must not override this current-state handover.

## 1. Project Goal

The project is a Chinese academic paper on lightweight traffic forecasting with knowledge distillation. The current paper-facing method name is `CCKD`.

Recommended paper title:

`CCKD：可信度自适应双路径蒸馏与软课程机制的轻量化交通预测方法`

Do not use version names such as `v3`, `v4`, or `v5` in the paper.

Core positioning:

- This is not a new strongest traffic forecasting backbone.
- The contribution is a better distillation strategy for a lightweight student model.
- The paper should emphasize the accuracy-efficiency trade-off, not absolute SOTA over all heavy models.
- The key story is that teacher knowledge reliability and forecasting-horizon difficulty are both heterogeneous in traffic forecasting.

## 2. Current Model Structure

Teacher:

- `GWNet Teacher`
- Provides stronger spatiotemporal forecasting knowledge during training.

Student:

- `Lightweight GCN Student`
- Final deployable model.
- Only the student is used during inference.

Additional v6 generalization students:

- `Lightweight TCN Student`
- `Lightweight GRU Student`
- Added only for teacher-student generalization experiments.
- They are not meant to replace the main GCN student in the core paper story.

Additional v7 generalization students:

- `STID-style MLP Student`
- `DLinear Student`
- Added to broaden the lightweight student family beyond graph, temporal-convolution, and recurrent students.
- Use them as generalization/transferability evidence, not as replacements for the main GCN student.

Training framework:

- Historical traffic sequence is fed to both teacher and student.
- Teacher and student forecasts are used by the confidence-adaptive dual-path distillation module.
- Distillation losses are further adjusted by the soft curriculum weighting module over forecasting horizons.
- v7 可选启用 DDASC 动态难度感知软课程，对应参数为 `--curriculum_mode dynamic_soft`。
- Ground-truth labels provide hard supervision.
- Total loss trains the student.

Inference framework:

- Teacher, confidence estimation, dual-path distillation, and curriculum weighting are training-time mechanisms.
- Inference efficiency should be measured on the final student model only.

## 3. Core Contribution 1: Confidence-Adaptive Dual-Path Distillation

This is the main innovation.

Correct interpretation:

- Teacher prediction error against ground-truth labels is converted into a continuous confidence score `c in [0, 1]`.
- This is soft routing, not hard filtering.
- Do not write that `c > 0.5` goes to absolute-value distillation and `c < 0.5` goes to trend distillation.
- For each node-horizon position, both distillation paths can contribute.
- Absolute-value distillation is weighted by confidence `c`.
- Trend distillation is weighted by the complementary low-confidence term.
- High confidence emphasizes direct value matching.
- Low confidence emphasizes trend consistency.
- Low-confidence teacher knowledge is not discarded; it is transferred in a more robust trend form.

Current code behavior:

- Implemented in `losses/distillation.py`.
- Teacher error is computed as `abs(teacher_pred - real_value)` with invalid targets masked.
- Node-level error and horizon-level error are averaged separately.
- Each error map is inverse min-max normalized into confidence.
- Final confidence is `node_confidence * horizon_confidence`, then masked by valid labels.
- This means the draft formula `c = 1 / (1 + e / tau)` is not faithful to the current code and should not be used unless the code is changed.

Current loss behavior:

- Absolute path uses a mask like `confidence_score * curriculum_map`.
- Trend path computes differences between adjacent forecast horizons.
- Trend confidence uses the average confidence of two adjacent horizons.
- Trend mask uses the complementary term of that averaged confidence.
- Both absolute and trend losses currently use Smooth L1 on temperature-scaled predictions.

Paper wording to preserve:

`本文并未采用硬阈值将教师知识划分为高可信和低可信两类，而是将教师预测误差映射为连续可信度分数，并以该分数作为软权重调节两类蒸馏损失。其中，绝对值蒸馏由可信度分数加权，趋势蒸馏由其互补项加权，从而实现平滑的双路径知识迁移。`

## 4. Core Contribution 2: Soft Curriculum Over Forecasting Horizons

Paper-facing idea:

- Multi-step traffic forecasting has horizon difficulty heterogeneity.
- Short-term horizons are generally easier and more stable.
- Long-term horizons are harder and more uncertain.
- The student should receive a smooth horizon curriculum rather than uniform distillation pressure.

Important distinction:

- The intended paper story is soft curriculum, not hard curriculum.
- All 12 horizons should remain active if the paper claims soft curriculum.
- Early training should emphasize short-term horizons.
- Long-term horizon weights should gradually increase as training progresses.

Implementation warning:

- Current code supports `curriculum_mode` values: `standard`, `short`, `wide`, `soft`, and `dynamic_soft`.
- Current training script default is `standard`.
- `standard`, `short`, and `wide` contain hard horizon opening behavior.
- If the final paper claims that all horizons are always active, final experiments should use `--curriculum_mode soft` or the code should be adjusted to match the paper.
- v6 fixed the `soft` weight direction so short-term horizons are emphasized early and long-term weights gradually increase.
- v7 保持所有旧模式不变，只新增显式启用的 `dynamic_soft` 模式。
- 在 `dynamic_soft` 中，现有 `soft` curriculum 是权重下限；每轮验证后的 horizon 级信号会用于更新下一轮蒸馏权重。
- DDASC 难度分数为 `D_h = 0.45 * E_h + 0.35 * G_h + 0.20 * (1 - C_h)`，其中 `E_h` 是学生验证 MAE，`G_h` 是师生预测差距，`C_h` 是教师 horizon 置信度。
- DDASC 权重为 `m_h = b_h + 0.70 * (1 - b_h) * (1 - normalize(D_h))`，并使用 EMA 平滑、5 epoch warmup、`m_h >= b_h`、`m_h <= 1`、短期到长期单调不增等约束。

## 5. Formula Notes

Target venue constraints mentioned by the user:

- Final equations should be entered with MathType.
- Equation numbers should use Chinese style such as `（1）`.
- Do not use teacher/student superscripts except real powers.
- Prefer teacher/student explanatory subscripts such as `te` and `st`.
- Every symbol, including subscripts, must be explained.
- Variables should be single-letter italic where possible.
- If `log` appears, specify the base. The current method does not need `log`.

Known draft-paper formula issues:

- The current draft used `c = 1 / (1 + e / tau)`, which does not match the code.
- The current draft used trend weight `1 - c_{i,h}`, but the code uses the complement of adjacent-horizon averaged confidence for trend differences.
- The current draft soft curriculum formula may not match current code.
- Before finalizing the paper, align formulas, code, and figures.

## 6. Current Figures

The user currently has four core paper figures:

1. Overall framework figure.
2. Teacher-student architecture figure.
3. Confidence-Adaptive Dual-Path Distillation module figure.
4. Soft Curriculum Weighting module figure.

Overall framework figure:

- It is a training framework figure.
- `Total Loss` can be the terminal node.
- A dashed optimization/backpropagation arrow from total loss to the student is acceptable.
- Stacked visual effects are appropriate for historical input, teacher module, and student module.
- Stacked effects are not recommended for teacher forecasts, student forecasts, total loss, or ground-truth labels.

Dual-path module figure:

- Avoid drawing hard branch selection.
- Label the absolute path as `weighted by c`.
- Label the trend path as `weighted by 1-c` or complementary confidence.
- Include wording such as `soft routing: both paths are active`.

Soft curriculum figure:

- Recommended representative horizons: `H1`, `H3`, `H5`, `H7`, `H10`, `H12`.
- It should communicate that all horizons are active only if final experiments use true soft curriculum.

Teacher-student structure figure:

- Use two subfigures if needed: `(a) GWNet Teacher`, `(b) Lightweight GCN Student`.
- Teacher can show gated spatiotemporal block, graph convolution, residual/skip connection, BatchNorm, prediction head.
- Student should look simpler: input projection, temporal conv, lightweight graph block, temporal readout, prediction head.

## 7. Experiment and Table Plan

Use `[待填]` placeholders until real results are available. Do not invent numbers.

Recommended tables:

- Table 1: dataset statistics for `METR-LA` and `PEMS-BAY`.
- Table 2: main results on both datasets, including teacher, student-only, vanilla KD, and CCKD.
- Table 3: accuracy-efficiency comparison with MAE/MAPE/RMSE, parameters, inference time, and deploy model.
- Table 4: ablation study with student-only, vanilla KD, w/o confidence-adaptive distillation, w/o soft curriculum, and full CCKD.
- Optional Table 5: generalization across another lightweight student or teacher-student combination.
- Optional Table 6: curriculum or hyperparameter sensitivity.

Main comparison interpretation:

- Do not claim CCKD beats every heavy classical model.
- The main claim is that CCKD improves the lightweight student compared with student-only and vanilla KD.
- Also show whether CCKD narrows the teacher-student performance gap.
- Efficiency should be reported for the deployed student.

Classical model comparison:

- Possible baselines: `STGCN`, `DCRNN`, `Graph WaveNet`, `AGCRN`, `GWNet Teacher`, `Student only`, `Vanilla KD`, `CCKD`.
- If `Graph WaveNet` and `GWNet Teacher` are the same implementation, avoid duplicate rows or explain the distinction clearly.

Generalization experiment suggested by the advisor:

- Purpose: show that CCKD is not only effective for one teacher-student pair.
- v6 added `Lightweight TCN Student` and `Lightweight GRU Student` for this experiment.
- Medium-cost design: keep `GWNet Teacher`, use two additional lightweight students, and compare `Student only`, `Vanilla KD`, and `CCKD` on both `METR-LA` and `PEMS-BAY`.
- This gives 12 generalization runs: 2 datasets x 2 extra students x 3 training strategies.
- This requires 6 new training runs.
- Do not expand to unrelated domains such as image classification unless the project scope changes.

## 8. Paper Draft Status

The GPT-generated draft `CCKD_中文学术论文初稿_带引用占位符.doc/.docx` was reviewed.

Good aspects:

- Overall story is correct.
- It says CCKD is not a new strongest backbone.
- It says inference keeps only the student.
- It says confidence is soft routing rather than hard thresholding.
- It says low-confidence teacher knowledge is not discarded.
- It includes tables with `[待填]` rather than fabricated results.

Required fixes before using the draft as the paper base:

- Replace the confidence formula to match code or change code to match the formula.
- Replace the trend weighting formula to reflect adjacent-horizon averaged confidence.
- Ensure curriculum formula, figure, and final experiments all match.
- Convert final equations to MathType.
- Replace placeholder references with real references.
- Save the file as real `.docx`; the old file had `.doc` extension but docx-like contents.

## 9. Key Project Files

Important files:

- `README.md`: project usage overview, but verify it does not still expose paper-facing version names.
- `losses/distillation.py`: current confidence, dual-path, and curriculum loss implementation.
- `train_student_kd.py`: student distillation training entry.
- `engine.py`: training loops and teacher/student prediction handling.
- `utils/curriculum.py`: v7 DDASC 调度器和验证 horizon 信号统计工具。
- `compare_teacher_student.py`: teacher/student prediction visualization.
- `scripts/generate_distillation_heatmap.py`: teacher error and confidence heatmaps.
- `scripts/benchmark_model.py`: parameter and inference speed benchmarking.
- `scripts/plot_efficiency_tradeoff.py`: accuracy-efficiency figure support.
- `models/student_tcn.py`: lightweight TCN student added for v6 generalization experiments.
- `models/student_gru.py`: lightweight GRU student added for v6 generalization experiments.
- `models/student_stid.py`: STID-style MLP student added for v7 generalization experiments.
- `models/student_dlinear.py`: DLinear student added for v7 generalization experiments.
- `v6_generalization_experiment.md`: root-level v6 change notes and runnable experiment commands.
- `v7_dynamic_curriculum_experiment.md`: 根目录下的 v7 DDASC 中文实验流程和记录规范。
- `docs/project_full_memory.md`: long historical memory; older sections may be outdated.
- `docs/project_handover.md`: current quick handover; this file should be trusted first.

## 10. Immediate Next Steps

Recommended next steps:

1. 按 `v7_dynamic_curriculum_experiment.md` 跑 v7 DDASC 学生实验；教师 checkpoint 复用，不重跑教师。
2. 将 v7 `dynamic_soft` 和已有或补跑的 v6/fixed `soft` 学生结果做直接对照。
3. v6 TCN/GRU 泛化实验和 v7 DDASC 实验先分开管理，除非后续论文同时需要两部分。
4. 按 `losses/distillation.py` 和 `utils/curriculum.py` 对齐论文公式。
5. 根据置信度、趋势蒸馏和课程权重的最新实现更新论文初稿。
6. 把真实实验结果填入表格，不要编造数值。
7. 完成四张核心图；如果 v7 结果有效，可以额外加入 DDASC 权重演化图。
8. 最终公式转成 MathType，并替换参考文献占位符。

## 11. v6 Latest Experiment Status

This section records the latest state after the advisor suggested testing whether the proposed distillation strategy can transfer to other lightweight students.

Purpose:

- Show that CCKD is not only effective for the original `Lightweight GCN Student`.
- Keep the same `GWNet Teacher`.
- Add alternative lightweight students and compare `Student only`, `Vanilla KD`, and `CCKD`.
- Use this as a generalization experiment, not as a replacement of the main GCN-based paper story.

Implemented code support:

- `models/student_tcn.py` adds `Lightweight TCN Student`.
- `models/student_gru.py` adds `Lightweight GRU Student`.
- `models/student_stid.py` adds `STID-style MLP Student`.
- `models/student_dlinear.py` adds `DLinear Student`.
- `train_student_kd.py` supports `--student_model gcn|tcn|gru|stid|dlinear`.
- `test.py`, `scripts/benchmark_model.py`, `scripts/collect_results.py`, and related utilities can rebuild the correct student architecture from checkpoint metadata.
- Root document `v6_generalization_experiment.md` records the runnable training and testing commands.

Current partial results recorded in `docs/结果.md`:

- `METR-LA` / `Lightweight TCN Student only`: `MAE=3.6748`, `MAPE=0.1044`, `RMSE=7.2083`, `params=27,468`, `latency=13.35ms/batch`.
- `METR-LA` / `TCN Vanilla KD`: `MAE=3.6682`, `MAPE=0.1056`, `RMSE=7.1411`, `params=27,468`, `latency=20.32ms/batch`.
- `METR-LA` / `TCN CCKD soft`: `MAE=3.6839`, `MAPE=0.1051`, `RMSE=7.2074`, `params=27,468`, `latency=21.08ms/batch`.
- `METR-LA` / `TCN CCKD standard`: `MAE=3.7008`, `MAPE=0.1068`, `RMSE=7.2146`, `params=27,468`, `latency=14.79ms/batch`.
- For comparison, `METR-LA` / `GCN CCKD soft`: `MAE=3.4699`, `MAPE=0.0981`, `RMSE=6.5075`, `params=27,404`, `latency=12.84ms/batch`.

Current interpretation:

- The v6 TCN results do not yet support a strong claim that CCKD improves every lightweight student.
- `TCN Vanilla KD` is currently better than TCN CCKD by MAE/RMSE on the recorded METR-LA runs.
- The result is still useful diagnostically, but should not be written as a positive generalization conclusion unless later GRU/PEMS-BAY results support it.
- If the paper includes a v6 generalization table, be honest and phrase it as an analysis of transferability across lightweight students, not as universal improvement.

Still pending:

- Complete the GRU runs on `METR-LA`.
- Complete TCN/GRU runs on `PEMS-BAY`.
- Decide whether the generalization table should enter the main paper, appendix, or be omitted depending on results.

## 12. v7 动态课程当前状态

目的：

- v7 为主线 GCN 学生蒸馏阶段新增 DDASC 动态难度感知软课程。
- 它针对不同预测步的难度差异做动态调整：短期 horizon 通常更稳定，长期 horizon 更不确定，但所有 horizon 都保持参与训练。
- 当前 `soft` curriculum 作为固定基线和权重下限；`dynamic_soft` 只会在这个下限之上，根据验证信号提高下一轮蒸馏权重。

已实现代码支持：

- `utils/curriculum.py` 包含 `DynamicCurriculumScheduler`、fixed soft 基础权重、horizon 级验证信号统计和 DDASC 状态保存。
- `losses/distillation.py` 支持可选的 `curriculum_override`；不传 override 时，所有旧课程模式保持原行为。
- `engine.py` 会把当前 epoch 的动态权重传给蒸馏损失。
- `train_student_kd.py` 支持 `--curriculum_mode dynamic_soft`，并新增 `alpha`、`beta`、`gamma`、`eta`、`ema`、`warmup` 等动态课程参数。
- 训练 history 会记录当前权重、基础权重、下一轮权重、horizon MAE、师生差距、教师置信度、difficulty、readiness 和有效样本数。
- checkpoint 会保存 DDASC 配置和最终调度器状态，便于复现实验。

实验规则：

- 教师 checkpoint 继续复用：`checkpoints/teacher/metr_teacher_best.pt` 和 `checkpoints/teacher/bay_teacher_best.pt`。
- 除非教师结构或教师训练流程发生变化，否则 v7 不需要重新训练教师。
- v7 主实验命名建议使用 `metr_student_gcn_cckd_v7_ddasc` 和 `bay_student_gcn_cckd_v7_ddasc`。
- fixed soft 直接对照命名建议使用 `*_v6_soft` 或 `*_fixed_soft`，不要和 v7 混名。
- 如果只能先跑一个新实验，优先跑完整学生方法的 `--curriculum_mode dynamic_soft`。

## 13. Resume Prompt for a New Model

Use this prompt if context is lost:

`This is a Chinese academic paper project on CCKD for lightweight traffic forecasting. The current method uses a GWNet teacher and a lightweight GCN student. The main contributions are confidence-adaptive dual-path distillation and soft curriculum weighting over forecasting horizons. v7 adds optional DDASC dynamic soft curriculum with --curriculum_mode dynamic_soft; it reuses teacher checkpoints and only reruns student distillation experiments. Confidence is continuous soft routing, not a hard 0.5 threshold: absolute-value distillation is weighted by confidence and trend distillation by complementary low confidence. Low-confidence teacher knowledge is not discarded. The final deployed model is only the lightweight student. Read docs/project_handover.md first, then v7_dynamic_curriculum_experiment.md for v7 commands; use docs/project_full_memory.md only as historical context because older sections contain abandoned RMGD-KD/v3/v4 notes. Before writing or editing the paper, align formulas with losses/distillation.py and utils/curriculum.py, especially confidence calculation, adjacent-horizon trend weighting, and curriculum mode.`
