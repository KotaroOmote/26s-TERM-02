# 実験更新メモ（2026-02-17）

対象:  
`A) 判定性能` / `B) 四象限事例` / `C) ロバスト性` の追試結果を、再現スクリプトと一致する形で固定。

## 1. A) 判定性能（test split, 95% CI）

参照: `analysis_outputs/stats_figs_2026-02-16_v3/table_model_performance.csv`

| model | AUROC | AUROC 95% CI | TNR@95TPR | TNR@95TPR 95% CI | AUPR |
|---|---:|---|---:|---|---:|
| energy_single | 0.996254 | [0.994952, 0.997425] | 0.996000 | [0.992931, 0.998003] | 0.996895 |
| logistic_2d | 0.941065 | [0.934870, 0.946372] | 0.687000 | [0.651050, 0.725988] | 0.945177 |
| linear_svm | 0.941045 | [0.934851, 0.946372] | 0.687333 | [0.651237, 0.725593] | 0.945179 |
| zsum_1d | 0.907249 | [0.900193, 0.914486] | 0.598667 | [0.567388, 0.630475] | 0.909341 |
| msp_single | 0.886241 | [0.878105, 0.894234] | 0.578000 | [0.541767, 0.605631] | 0.882821 |

要点:
- 二値検知の最適性能は `energy_single`。
- 一方で `logistic_2d/linear_svm` は `msp_single` より大きく改善し、`z_u-z_c` 空間の解釈性と両立している。

## 2. B) `z_u-z_c` 四象限誤り傾向

参照: `analysis_outputs/stats_figs_2026-02-16_v3/table_quadrant_errors.csv`

| quadrant | total | ood_rate | fp | fn |
|---|---:|---:|---:|---:|
| Q1(+u,+c) | 2664 | 0.867117 | 216 | 106 |
| Q2(-u,+c) | 287 | 0.167247 | 2 | 45 |
| Q3(-u,-c) | 2904 | 0.176997 | 59 | 315 |
| Q4(+u,-c) | 145 | 0.882759 | 17 | 0 |

要点:
- FPは主に `Q1(+u,+c)` に集中。
- FNは主に `Q3(-u,-c)` に集中。
- 単一スコアでは把握しにくい失敗モードの局在が確認できる。

## 3. C) ロバスト性（prompt / seed / model）

設定:
- prompts: `"a photo of {name}"`, `"a close-up photo of {name}"`
- seeds: `42, 43, 44`
- models: `ViT-B-32 (laion2b_s34b_b79k)`, `ViT-B-16 (laion2b_s34b_b88k)`

観測:
- `k=3`, `axis_u_index=2`, `axis_c_index=0` が全runで一致。
- cosine similarity は報告値で全比較 `1.0`。

## 4. 主要図（説得用）

`analysis_outputs/stats_figs_2026-02-16_v3/` に以下を保存。

- `fig01_perf_ci_bar.png`: AUROC/TNR@95 (95%CI)
- `fig02_roc_curves.png`: ROC曲線
- `fig03_pr_curves.png`: PR曲線
- `fig04_score_distributions.png`: energy / p(OoD) 分布
- `fig05_zu_zc_scatter_boundary.png`: `z_u-z_c` 散布 + logistic境界
- `fig06_zu_zc_density_contours.png`: ID/OoD 密度等高線
- `fig07_quadrant_errors.png`: 四象限のFP/FN
- `fig08_axis_loadings_heatmap.png`: SparsePCA荷重
- `fig09_cv_heatmap_1se.png`: CV + 1SE選択
- `fig10_effect_size_forest.png`: 効果量と95%CI

## 5. 報告書

- 新版: `report/250217-2.pdf`
- 旧版: `report/N-Axis_Attribution_OoD_2026-02-16.pdf`
