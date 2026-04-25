# CCKD Paper Project Handover

## 1. Project Purpose

This project is for a traffic forecasting paper centered on a lightweight knowledge distillation framework named **CCKD**.

The paper focus is:

- lightweight traffic forecasting
- teacher-student knowledge distillation
- confidence-adaptive dual-path distillation
- soft curriculum over forecasting horizons
- performance-efficiency trade-off rather than absolute state-of-the-art accuracy

This handover document is intended to help quickly resume work if the conversation context is lost or the user logs in with a different account.

---

## 2. Paper Title and Positioning

Current Chinese title:

`可信度感知双路径蒸馏与软课程机制的轻量交通预测方法`

Positioning:

- This paper is **not** framed as a new strongest forecasting backbone.
- The contribution is a **better distillation strategy for a lightweight student model**.
- The core story is that teacher knowledge quality and forecasting difficulty are both heterogeneous in traffic forecasting, so the student should not mimic the teacher uniformly.

---

## 3. Core Method Definition

### 3.1 Teacher and Student

- Teacher model: `GWNet`
- Student model: lightweight `GCN`

The teacher is used to provide stronger spatiotemporal forecasting knowledge, while the student is the lightweight deployable model.

### 3.2 Confidence-Adaptive Dual-Path Distillation

This is the main innovation.

Key logic:

- Estimate teacher knowledge confidence from teacher prediction error against ground-truth labels.
- Map teacher error into a continuous confidence score `c in [0, 1]`.
- Use **soft routing**, not a hard threshold.
- Absolute-value distillation is weighted by `c`.
- Trend distillation is weighted by `1 - c`.
- High-confidence regions therefore emphasize absolute-value distillation.
- Low-confidence regions therefore emphasize trend distillation.
- Low-confidence regions are **not discarded**; instead, they use a more robust trend-transfer path.

Correct interpretation:

- This is **not** simple filtering.
- This is **not** a binary rule such as `c > 0.5` for absolute distillation and `c < 0.5` for trend distillation.
- For each node-horizon position, both paths can contribute; confidence only changes their relative weights.
- This is **confidence-guided differentiated distillation**.

### 3.3 Soft Curriculum Over Horizons

This is the second main innovation.

Key logic:

- All forecasting horizons are active from the beginning.
- Short-term horizons have larger distillation weights in early training.
- Long-term horizon weights increase gradually as training proceeds.
- This is a **soft curriculum**, not a hard curriculum.

Important clarification:

- `standard / short / wide / soft` are curriculum strategy variants from experimentation and implementation tuning.
- In the paper, the **main final method should emphasize `soft curriculum`**.
- Other variants should be discussed in experiments or ablations, not presented as equally central method components.

---

## 4. Main Storyline for the Paper

The paper should consistently tell this story:

1. Heavy traffic forecasting models can achieve strong accuracy but are costly to deploy.
2. Lightweight student models need effective distillation to approach teacher performance.
3. Vanilla KD assumes that teacher knowledge is equally reliable at all nodes and horizons, which is not true in traffic forecasting.
4. Teacher knowledge quality is heterogeneous.
5. Forecasting difficulty across horizons is also heterogeneous.
6. Therefore, the student should learn **different knowledge types in different regions**, and should absorb teacher knowledge with a **progressive, smooth horizon curriculum**.

This is the main narrative thread that should remain stable across abstract, introduction, method, and experiments.

---

## 5. Datasets and Evaluation

Main datasets:

- `METR-LA`
- `PEMS-BAY`

Metrics:

- `MAE`
- `MAPE`
- `RMSE`

Interpretation note:

- `METR-LA` is treated as the more complex dataset.
- `PEMS-BAY` is treated as more stable/easier.
- This difference is one reason curriculum strategy sensitivity became an important design point.

---

## 6. Current Figure Strategy

### Figures to keep in the paper

Recommended figure set:

1. Overall framework figure
2. Teacher/student structure figure
3. Soft curriculum mechanism figure
4. Teacher-student prediction comparison figure
5. Performance-efficiency trade-off figure
6. Optional heatmap figure (`teacher error` or `confidence`)

### Important figure decision

The dual-path distillation mechanism **does not necessarily need a separate figure** if the overall framework already contains enough detail.

Current consensus:

- If the overall framework figure includes a relatively detailed dual-path module, a second dual-path figure may feel repetitive.
- In that case, it is acceptable to **keep only the overall framework figure** and explain the dual-path mechanism in the text.
- If the dual-path module inside the overall framework becomes too tiny or unreadable, either simplify it or bring back a separate mechanism figure.

### Overall framework figure guidance

- It is a **training framework figure**, not only an inference graph.
- `Total Loss` can be the terminal node; it does not require an additional output box.
- If desired, a dashed feedback arrow from `Total Loss` back to the student can indicate optimization/backpropagation.
- Stacked overlap effects are appropriate for:
  - historical input
  - teacher module
  - student module
- Stacked overlap effects are **not recommended** for:
  - teacher forecasts
  - student forecasts
  - total loss
  - ground-truth labels

### Overall framework wording choices

Teacher module:

- `GWNet Teacher`
- `ST Block × 4`
- `Prediction Head`

Student module:

- `Lightweight GCN Student`
- `GCN Block × 3`
- `Prediction Head`

Forecast outputs:

- `Teacher Forecasts`
- `Student Forecasts`

If formulas are shown in the figure but cannot be rendered in LaTeX, a hand-written style like `Ŷ_te` and `Ŷ_st` is acceptable.

### Soft curriculum figure guidance

The final curriculum figure should focus on **soft curriculum only**, not all curriculum strategy variants.

Recommended representative horizons:

- `H1`
- `H3`
- `H5`
- `H7`
- `H10`
- `H12`

This figure should communicate:

- all horizons are active
- short-term first
- long-term weights gradually increase

---

## 7. Teacher and Student Structure Figures

These are separate from the overall framework and can contain more internal detail.

### Teacher structure figure

May show:

- `1×1 Conv`
- gated temporal branches
- `Tanh`
- `Sigmoid`
- `Graph Conv`
- residual connection
- `BatchNorm`
- `Prediction Head`

### Student structure figure

May show:

- `1×1 Conv`
- temporal conv
- `ReLU`
- `Graph Conv`
- residual add
- `BatchNorm`
- `Temporal Readout`
- `Dropout`
- `Prediction Head`

These figures should be more detailed than the overall framework, but still not overly dense.

---

## 8. Formula and Writing Conventions

### Naming conventions

- Paper method name: `CCKD`
- Avoid version names like `v3`, `v4`, `v5` in the paper body.

### Formula constraints from the target venue

The target venue requires:

- equations entered with **MathType**
- equation numbering as `（1）`, `（2）`, ...
- no teacher/student superscripts except real powers
- single-letter italic variables
- explanatory subscripts in upright style
- every symbol explained clearly

Important consequence:

- Avoid `Ŷ^T`, `Ŷ^S`
- Prefer teacher/student subscripts such as `te`, `st`

### Draft formula status

Multiple draft DOCX files were reviewed.

Key issue:

- The revised formula drafts improved the symbol system, but the equations are still **not actual MathType objects** yet.

Known specific formula issue:

- There was a typo like `au_{c}` that should be corrected to `\tau_{c}` before final formatting.

If formula work resumes, the main remaining tasks are:

1. unify symbols
2. rewrite formula text cleanly
3. manually enter all final equations using MathType

---

## 9. Experiment and Table Strategy

### Main results table

Should compare:

- `Teacher`
- `Baseline Student`
- `Vanilla KD`
- `CCKD`

On both:

- `METR-LA`
- `PEMS-BAY`

### Classical model comparison table

Purpose:

- show competitiveness and efficiency
- not claim superiority over all heavy models

Recommended columns:

- `Model`
- `Type`
- `METR-LA MAE / MAPE / RMSE`
- `PEMS-BAY MAE / MAPE / RMSE`
- `Params`
- `Latency`

Recommended models:

- `STGCN`
- `DCRNN`
- `Graph WaveNet`
- `AGCRN`
- `CCKD`

### Ablation table

Should include at least:

- full `CCKD`
- `w/o confidence`
- `w/o curriculum`

---

## 10. Heatmap Analysis Support

A heatmap script was added to the project to support paper visualizations.

Script:

- [generate_distillation_heatmap.py](/C:/Users/86151/Documents/New%20project/scripts/generate_distillation_heatmap.py)

Supported outputs:

- `teacher_error` heatmap
- `confidence` heatmap
- or both

Outputs go to:

- `outputs/figures/`
- `outputs/reports/`

Recommended paper use:

- If only one heatmap is used, prefer the **teacher error heatmap**
- If two are used, show both teacher error and confidence

---

## 11. Key Generated / Updated Project Files

Important project files and their purposes:

- [README.md](/C:/Users/86151/Documents/New%20project/README.md)
  - rewritten around the current CCKD pipeline

- [project_full_memory.md](/C:/Users/86151/Documents/New%20project/docs/project_full_memory.md)
  - long-form evolving project memory

- [project_handover.md](/C:/Users/86151/Documents/New%20project/docs/project_handover.md)
  - current-state handover document

- [generate_distillation_heatmap.py](/C:/Users/86151/Documents/New%20project/scripts/generate_distillation_heatmap.py)
  - teacher error / confidence heatmap generation

- [compare_teacher_student.py](/C:/Users/86151/Documents/New%20project/compare_teacher_student.py)
  - teacher/student prediction comparison plotting

- [collect_results.py](/C:/Users/86151/Documents/New%20project/scripts/collect_results.py)
  - summary table generation

- [plot_efficiency_tradeoff.py](/C:/Users/86151/Documents/New%20project/scripts/plot_efficiency_tradeoff.py)
  - performance-efficiency figure generation

- [benchmark_model.py](/C:/Users/86151/Documents/New%20project/scripts/benchmark_model.py)
  - parameters / speed benchmarking

---

## 12. Immediate Next Steps

Recommended next priorities:

1. Finalize the overall framework figure
2. Finalize the teacher/student structure figure(s)
3. Finalize the soft curriculum figure
4. Decide whether the dual-path module remains detailed in the overall framework or is separated again
5. Continue rewriting and polishing the paper draft
6. Convert final equations into MathType
7. Fill the remaining experimental results and tables

---

## 13. Resume Prompt for a Future Session

If context is lost, the following short instruction can be used to resume efficiently:

`This project is a Chinese paper on CCKD for lightweight traffic forecasting. The teacher is GWNet, the student is a lightweight GCN. The core contributions are confidence-adaptive dual-path distillation and soft curriculum over horizons. Please read docs/project_handover.md first and continue from the current paper-writing and figure-refinement stage.`
