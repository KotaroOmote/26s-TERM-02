# VLM Attribution OoD 更新パッケージ（2026-02-17）

このディレクトリは、次の要求から進めた再現実装をまとめたものです。

> 「A) 判定性能比較 / B) 四象限事例分析 / C) ロバスト性検証」

## 実施順（この順で進行）

1. 軸の定め方（4変数 + SparsePCA + 1SE）
2. A) 判定性能比較
3. B) 四象限事例分析
4. C) ロバスト性検証

## 対象範囲

- 軸構築（4変数設定）
- A) 判定性能比較
  - `(z_u, z_c)` を入力にした `zsum_1d` / `linear_svm` / `logistic_2d`
  - 単独ベースライン `msp_single` / `energy_single`
- B) `z_u-z_c` 平面での四象限誤り分析
- C) prompt / seed / model を跨ぐロバスト性評価
- 統計レポート出力
  - 統計表 3種 + 図 10種

## ディレクトリ構成

- `colab/`
  - `colab_tfds_axis_builder.py`
  - `colab_eval_detectors.py`
  - `colab_quadrant_cases.py`
  - `colab_run_robustness.py`
  - `colab_make_axis_figures.py`
  - `colab_make_statistics_figures.py`
- `docs/`
  - `experiment_update_2026-02-17.md`（最新結果）
  - `experiment_update_2026-02-16.md`（前段結果）
  - `end_to_end_flow_from_abc_request.md`（A/B/C提案から現在までの時系列）
  - `reproducibility_from_chat.md`
  - `reproducibility_one_page_ja.md`
- `analysis_outputs/stats_figs_2026-02-16_v3/`
  - `table_model_performance.csv`
  - `table_feature_significance.csv`
  - `table_quadrant_errors.csv`
  - `fig01_perf_ci_bar.png` 〜 `fig10_effect_size_forest.png`
- `report/`
  - `250217-2.pdf`（最新版）
  - `N-Axis_Attribution_OoD_2026-02-16.pdf`（前版）

## 主要結果（A/B/C）

- A) 判定性能:
  - 二値OoD性能は `energy_single` が最良（`AUROC≈0.996`, `TNR@95≈0.996`）
  - `(z_u, z_c)` 系モデル（`logistic_2d`, `linear_svm`）は `msp_single` より明確に高性能
- B) 四象限分析:
  - FP集中: `Q1(+u,+c)`, `Q4(+u,-c)`
  - FN集中: `Q3(-u,-c)`
- C) ロバスト性:
  - `seed(42/43/44)`, `prompt(2種)`, `model(ViT-B-32/16)` で軸のコサイン類似度は報告値 `1.0`

## 参照先

- 数値の詳細: `docs/experiment_update_2026-02-17.md`
- 実装の流れ: `docs/end_to_end_flow_from_abc_request.md`
