# Reproducibility Summary From Conversation Log

This document summarizes the implementation decisions and execution steps that were fixed during the chat session, so another reader can reproduce the same axis-construction stage without relying on chat context.

## 1. Scope fixed in this work
- Objective: complete only `start -> axis construction` for Attribution OoD.
- Principle: `Step Separation` (axis design and detector design are strictly separated).
- This package does **not** include detector training/threshold optimization as part of axis selection.

## 2. Data design fixed in the session
Initial idea changed during execution:
- Rejected: very small pilot scale (`N=360`).
- Revised target: large enough scale for stable axis selection.
- Operational dataset plan finalized on Colab/TFDS:
  - ID: `food101`, `train[:10000]`
  - OoD: `cifar100`, `train[:5000]` + `imagenet_r`, `test[:5000]`
  - Total: 20,000 samples

Important incident and resolution:
- `sun397/tfds` failed due upstream URL `404`.
- The pipeline was updated to use `cifar100` as OoD replacement while keeping `imagenet_r`.

## 3. Axis selection policy fixed in the session
- Removed heuristic weighted objective `J(k)`.
- Adopted statistical selection:
  - SparsePCA grid search over `(k, alpha)`
  - Cross-validation reconstruction error (MSE)
  - One-Standard-Error (1SE) rule for conservative model choice
- Added variance-threshold preprocessing to remove near-constant features.
- Updated feature design to **4 variables** by dropping `msp`-based delta
  because `msp` and `conf` are identical in this implementation.

## 4. Implementation files and roles
- `colab/colab_tfds_axis_builder.py`
  - TFDS loading, OpenCLIP inference, feature deltas, variance filtering,
    SparsePCA CV + 1SE selection, axis fixation.
- `colab/colab_make_axis_figures.py`
  - Generates 4 key figures from axis outputs.
- `report/N-Axis_Attribution_OoD_2026-02-16.pdf`
  - Final report for sharing/review.
- `report/figs/*.png`
  - Figure artifacts used in the report.

## 5. Colab execution runbook (exact order)
### 5.1 Install
```bash
pip install -U open-clip-torch==2.26.1 ftfy tensorflow tensorflow-datasets scikit-learn pandas matplotlib scipy pillow tqdm
```

### 5.2 Build axes
```bash
python colab_tfds_axis_builder.py \
  --n-id 10000 \
  --n-ood-cifar 5000 \
  --n-ood-imagenetr 5000 \
  --batch-size 64 \
  --prompt-template "a photo of {name}" \
  --k-max 4 \
  --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
  --output-dir /content/outputs/axis_build_4d \
  --quiet
```

### 5.3 Multi-seed stability
```bash
for s in 42 43 44; do
  python colab_tfds_axis_builder.py \
    --seed $s \
    --n-id 10000 --n-ood-cifar 5000 --n-ood-imagenetr 5000 \
    --batch-size 64 --k-max 4 --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
    --output-dir /content/outputs/axis_build_4d_s${s} \
    --quiet
done
```

### 5.4 Make figures
```bash
python colab_make_axis_figures.py \
  --axis-scores /content/outputs/axis_build_4d/axis_scores.csv \
  --axis-loadings /content/outputs/axis_build_4d/axis_loadings.csv \
  --cv-table /content/outputs/axis_build_4d/sparsepca_cv_table.csv \
  --summary-json /content/outputs/axis_build_4d/summary_axis_only.json \
  --out-dir /content/outputs/axis_build_4d/figs
```

### 5.5 Detector comparison (A)
```bash
python colab_eval_detectors.py \
  --axis-scores /content/outputs/axis_build_4d/axis_scores.csv \
  --sample-metrics /content/outputs/axis_build_4d/sample_metrics.csv \
  --out-dir /content/outputs/eval_detectors
```

### 5.6 Quadrant case extraction (B)
```bash
python colab_quadrant_cases.py \
  --axis-scores /content/outputs/axis_build_4d/axis_scores.csv \
  --out-dir /content/outputs/quadrant_cases \
  --top-n 8
```

### 5.7 Robustness runner (C)
```bash
python colab_run_robustness.py \
  --project-dir /content \
  --output-root /content/outputs/robustness \
  --prompt-templates "a photo of {name}||a close-up photo of {name}" \
  --seeds "42,43,44" \
  --models "ViT-B-32:laion2b_s34b_b79k||ViT-B-16:laion2b_s34b_b88k"
```

### 5.8 Statistical tables + 10 figures
```bash
python colab_make_statistics_figures.py \
  --axis-dir /content/outputs/axis_build_4d \
  --out-dir /content/outputs/stats_figs_2026-02-16_v3 \
  --test-size 0.30 \
  --seed 42 \
  --bootstrap 800
```

Main outputs:
- `table_model_performance.csv`
- `table_feature_significance.csv`
- `table_quadrant_errors.csv`
- `fig01_perf_ci_bar.png` ... `fig10_effect_size_forest.png`

## 6. Key expected results (reference values)
Latest finalized run (4-variable setting):
- axis selection:
  - `k_selected = 3`
  - `alpha_selected = 1.0`
  - `axis_u_index = 2`, `axis_c_index = 0`
- fixed-axis formulas:
  - `z_u = +1.0000*d_entropy_gain`
  - `z_c = +0.7071*d_conf_drop +0.7071*d_oodscore_gain`

A) Detector comparison with 95% bootstrap CI:
- `energy_single`: AUROC `0.996254` [`0.994955`, `0.997424`], TNR@95 `0.996000` [`0.993021`, `0.998002`]
- `logistic_2d`: AUROC `0.941065` [`0.934870`, `0.946460`], TNR@95 `0.687000` [`0.649121`, `0.726301`]
- `linear_svm`: AUROC `0.941045` [`0.934851`, `0.946450`], TNR@95 `0.687333` [`0.647829`, `0.726287`]
- `zsum_1d`: AUROC `0.907249` [`0.900101`, `0.914356`], TNR@95 `0.598667` [`0.566543`, `0.633346`]
- `msp_single`: AUROC `0.886241` [`0.878169`, `0.894477`], TNR@95 `0.578000` [`0.540523`, `0.607400`]

B) Quadrant observations:
- `Q1(+u,+c)`: `total=8930`, `ood_rate=0.861478`, `fp=778`, `fn=294`
- `Q2(-u,+c)`: `total=982`, `ood_rate=0.191446`, `fp=7`, `fn=177`
- `Q3(-u,-c)`: `total=9597`, `ood_rate=0.176305`, `fp=211`, `fn=1062`
- `Q4(+u,-c)`: `total=491`, `ood_rate=0.869654`, `fp=64`, `fn=0`

C) Robustness (`ViT-B-32`/`ViT-B-16`, seed `42/43/44`, 2 prompts):
- axis mapping fixed at `axis_u_index=2`, `axis_c_index=0` for all runs.
- cosine similarities were `1.0` for all reported comparisons.

## 7. Troubleshooting captured in the session
- `ModuleNotFoundError: open_clip`
  - install `open-clip-torch` and rerun.
- Excessive Colab output freezes UI
  - run with `--quiet` and redirect logs to file.
- LaTeX compile issue with `k^*`
  - use `k^{\ast}`.
- If figures are not rendered in TeX
  - place png files under `report/figs/` with exact names used in TeX.

## 8. Reproducibility checklist for this stage
- [x] Dataset identities and sample counts fixed.
- [x] Axis selection criterion fixed (CV MSE + 1SE).
- [x] Variance-threshold preprocessing included.
- [x] Random-seed stability evaluated.
- [x] Figure generation scripts included.
- [x] Report and compiled PDF included.
