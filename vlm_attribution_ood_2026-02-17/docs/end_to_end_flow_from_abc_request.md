# 一連の実装フロー（A/B/C提案から現在まで）

## 0. 起点（ユーザー要求）

起点となった要求は次の3点でした。

- A) 判定性能:
  - `(z_u, z_c)` を入力に、1次元閾値・線形境界・ロジスティック回帰を作成
  - `MSP / energy` 単独と `AUROC / TNR@95TPR` を比較
- B) 四象限事例:
  - `z_u-z_c` 平面の四象限から誤分類代表例を抽出
  - 「どんな崩れがどこに溜まるか」を示す
- C) ロバスト性:
  - プロンプト・ID/OOD分割・別VLMで類似軸が出るか確認
  - 固定軸の一般性を検証

## 1. 初期再現（軸構築ステージ）

- 実験方針は `Step Separation` を維持:
  - 先に軸を固定し、後段判定器を評価する構成を採用
- データ設計:
  - ID: `food101 train[:10000]`
  - OoD: `cifar100 train[:5000]` + `imagenet_r test[:5000]`
- 重要な障害と対処:
  - `sun397/tfds` は配布元URL 404で取得不能
  - 代替として `cifar100` を採用し、計画を継続

## 2. 実装中の主要トラブルと修正

- Colab出力過多でセッションが重くなる問題:
  - `--quiet` とログリダイレクトを導入
- `open_clip` import error:
  - 依存導入手順を明示し、再実行で解消
- sklearn依存崩れ (`xpx`, `trapezoid` import error):
  - `numpy/scipy/scikit-learn` の再インストール + runtime再起動を案内
- コマンド実行ミス（bash/pythonセル混在）:
  - 実行セルを用途別に分離し再提示

## 3. 軸定義の見直し（5変数→4変数）

- 指摘:
  - `conf` と `msp` が同値になり、特徴の冗長性がある
- 変更:
  - 5変数から `d_msp_drop` を除外し4変数へ統一
  - 採用特徴: `d_conf_drop, d_entropy_gain, d_energy_gain, d_oodscore_gain`
- 軸選択:
  - SparsePCA + CV再構成誤差 + 1SEルール

## 4. 軸構築の最終固定（4変数版）

- 主設定:
  - `n_id=10000, n_ood_cifar=5000, n_ood_imagenetr=5000`
  - `k_max=4`, `alpha_grid={0.5,1,2,4,8}`
- 固定結果:
  - `k_selected=3`, `alpha_selected=1.0`
  - `axis_u_index=2`, `axis_c_index=0`
  - `z_u = +1.0000*d_entropy_gain`
  - `z_c = +0.7071*d_conf_drop +0.7071*d_oodscore_gain`

## 5. A) 判定性能の検証

- 比較対象:
  - `msp_single`, `energy_single`, `zsum_1d`, `linear_svm`, `logistic_2d`
- 最終結果（test split, 95%CI）は
  - `table_model_performance.csv` に固定
- 解釈:
  - 二値OoDの純粋性能は `energy_single` が最良
  - ただし `logistic_2d / linear_svm` は `msp_single` より大幅改善し、軸空間の説明可能性を提供

## 6. B) 四象限事例分析

- `z_u-z_c` 平面を4象限に分割し、FP/FNを集計
- 観測:
  - FPは主に `Q1(+u,+c)` 側に集中
  - FNは主に `Q3(-u,-c)` 側に集中
- 成果:
  - 単一指標では隠れる失敗モードの偏りを可視化
  - 代表例抽出を `colab_quadrant_cases.py` と集計表で再現可能化

## 7. C) ロバスト性検証

- 検証軸:
  - prompt: `"a photo of {name}"`, `"a close-up photo of {name}"`
  - seed: `42/43/44`
  - model: `ViT-B-32`, `ViT-B-16`
- 観測:
  - `k=3`, `axis_u_index=2`, `axis_c_index=0` が全runで一致
  - cosine類似度は報告値で `1.0`
- 解釈:
  - 今回の設定範囲では固定軸の再現性は非常に高い

## 8. 説得力強化（統計3表 + 10図）

- 新規に統計パッケージを追加:
  - `colab_make_statistics_figures.py`
- 生成物:
  - 表: `table_model_performance.csv`, `table_feature_significance.csv`, `table_quadrant_errors.csv`
  - 図: `fig01` 〜 `fig10`（性能、ROC/PR、分布、境界、密度、象限誤り、loadings、CV+1SE、効果量）
- 目的:
  - 口頭説明依存を減らし、査読・第三者確認に耐える形へ整理

## 9. 文書・成果物の最終整理

- 報告書:
  - 新版 `report/250217-2.pdf`
  - 旧版 `report/N-Axis_Attribution_OoD_2026-02-16.pdf`
- 更新メモ:
  - `docs/experiment_update_2026-02-16.md`
  - `docs/experiment_update_2026-02-17.md`
- 本ドキュメント:
  - 「要求→実装→障害対応→結果固定」の流れを時系列で明文化

## 10. 現時点の到達点

- A/B/C は「再実行可能なスクリプト + 統計表 + 図」で固定済み
- 主張は二層で整理済み:
  - 二値性能最適化: `energy_single`
  - 解釈可能な失敗分析: `(z_u, z_c)` 系
- 次段階では、外部データ分割や追加VLMを広げ、一般化範囲の境界を明示するフェーズに進める状態
