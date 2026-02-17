# VLM Attribution OoD Update (2026-02-17)

このディレクトリは、以下の会話起点から進めた再現実装の新規まとめです。

> 「A) 判定性能比較 / B) 四象限事例分析 / C) ロバスト性検証」

## Scope

- Axis construction (4-feature setting)
- A) Detector performance comparison:
  - `(z_u, z_c)` based models: `zsum_1d`, `linear_svm`, `logistic_2d`
  - baselines: `msp_single`, `energy_single`
- B) Quadrant error analysis on `z_u-z_c` plane
- C) Robustness across prompt / seed / model
- Statistical report package:
  - 3 tables + 10 figures

## Directory Layout

- `colab/`
  - `colab_tfds_axis_builder.py`
  - `colab_eval_detectors.py`
  - `colab_quadrant_cases.py`
  - `colab_run_robustness.py`
  - `colab_make_axis_figures.py`
  - `colab_make_statistics_figures.py`
- `docs/`
  - `end_to_end_flow_from_abc_request.md` (A/B/C提案から現在までの時系列)
  - `experiment_update_2026-02-17.md` (latest summary)
  - `experiment_update_2026-02-16.md` (previous summary)
  - `reproducibility_from_chat.md`
  - `reproducibility_one_page_ja.md`
- `analysis_outputs/stats_figs_2026-02-16_v3/`
  - `table_model_performance.csv`
  - `table_feature_significance.csv`
  - `table_quadrant_errors.csv`
  - `fig01_perf_ci_bar.png` ... `fig10_effect_size_forest.png`
- `report/`
  - `250217-2.pdf` (new report)
  - `N-Axis_Attribution_OoD_2026-02-16.pdf` (previous report)

## Key Results (A/B/C)

- A) Binary OoD detection:
  - `energy_single` is best (`AUROC≈0.996`, `TNR@95≈0.996`).
  - `(z_u, z_c)` models (`logistic_2d`, `linear_svm`) are clearly better than `msp_single` and keep interpretability.
- B) Quadrant analysis:
  - FP集中: `Q1(+u,+c)`, `Q4(+u,-c)`
  - FN集中: `Q3(-u,-c)`
- C) Robustness:
  - Across seeds (`42/43/44`), prompts (`a photo`, `a close-up photo`), models (`ViT-B-32`, `ViT-B-16`), axis cosine similarities were reported as `1.0`.

See `docs/experiment_update_2026-02-17.md` for exact values.
For full chronology, see `docs/end_to_end_flow_from_abc_request.md`.
