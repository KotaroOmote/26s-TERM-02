# 再現実装 1ページ要約（N-Axis Attribution OoD, 2026-02-16）

## 1. この実装で再現する範囲
- 対象: `最初 -> 軸作成` フェーズのみ
- 方針: **Step Separation**（軸設計と判定器設計を分離）
- 非対象: 判定器学習・しきい値最適化・最終性能比較

## 2. データ設計（会話で確定した設定）
- ID: `food101 train[:10000]`
- OoD: `cifar100 train[:5000]` + `imagenet_r test[:5000]`
- 合計: 20,000 サンプル
- 備考: `sun397/tfds` は配布元 404 のため代替として `cifar100` を採用

## 3. 軸選択の中核仕様
- 特徴: `conf / entropy / energy / ood_score` から差分特徴を作成（4変数）
- 前処理: Variance Threshold（定数特徴を除外）
- 軸抽出: SparsePCA
- モデル選択: CV再構成誤差（MSE） + **1SEルール**
- 探索例: `k in [1..4]`, `alpha in {0.5,1.0,2.0,4.0,8.0}`

## 4. Colab再現手順（最短）
### 4.1 依存導入
```bash
pip install -U open-clip-torch==2.26.1 ftfy tensorflow tensorflow-datasets scikit-learn pandas matplotlib scipy pillow tqdm
```

### 4.2 軸作成
```bash
python colab/colab_tfds_axis_builder.py \
  --n-id 10000 \
  --n-ood-cifar 5000 \
  --n-ood-imagenetr 5000 \
  --batch-size 64 \
  --k-max 4 \
  --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
  --output-dir /content/outputs/axis_build_4d \
  --quiet
```

### 4.3 seed安定性（3本）
```bash
for s in 42 43 44; do
  python colab/colab_tfds_axis_builder.py \
    --seed $s \
    --n-id 10000 --n-ood-cifar 5000 --n-ood-imagenetr 5000 \
    --batch-size 64 --k-max 4 --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
    --output-dir /content/outputs/axis_build_4d_s${s} \
    --quiet
done
```

### 4.4 図生成（4種）
```bash
python colab/colab_make_axis_figures.py \
  --axis-scores /content/outputs/axis_build_4d/axis_scores.csv \
  --axis-loadings /content/outputs/axis_build_4d/axis_loadings.csv \
  --cv-table /content/outputs/axis_build_4d/sparsepca_cv_table.csv \
  --summary-json /content/outputs/axis_build_4d/summary_axis_only.json \
  --out-dir /content/outputs/axis_build_4d/figs
```

## 5. 期待される基準結果（2026-02-16 最終実行）
- 軸選択結果:
  - `k_selected = 3`
  - `alpha_selected = 1.0`
  - 軸対応: `axis_u_index = 2`, `axis_c_index = 0`
- 軸式:
  - `z_u = +1.0000*d_entropy_gain`
  - `z_c = +0.7071*d_conf_drop +0.7071*d_oodscore_gain`
- A) 判定性能（AUROC / TNR@95TPR）:
  - `energy_single`: `0.996254 / 0.996000`
  - `logistic_2d`: `0.941065 / 0.687000`
  - `linear_svm`: `0.941045 / 0.687333`
  - `zsum_1d`: `0.907249 / 0.598667`
  - `msp_single`: `0.886241 / 0.578000`
- B) 四象限傾向:
  - `Q1(+u,+c)` に FP が多い（`778`）
  - `Q3(-u,-c)` に FN が多い（`1062`）
- C) ロバスト性:
  - `ViT-B-16` / `ViT-B-32`, seed `42/43/44`, 2種prompt で
    `|cos(z_u)| = 1.0`, `|cos(z_c)| = 1.0`（全run）

## 6. 成果物（このリポジトリ）
- 軸作成コード: `colab/colab_tfds_axis_builder.py`
- 図生成コード: `colab/colab_make_axis_figures.py`
- 詳細手順: `docs/reproducibility_from_chat.md`
- レポート: `report/N-Axis_Attribution_OoD_2026-02-16.pdf`
- 図: `report/figs/fig1~fig4_*.png`

## 7. よくあるエラー
- `ModuleNotFoundError: open_clip`:
  - `open-clip-torch` をインストールして再実行
- Colabが出力過多で固まる:
  - `--quiet` とログリダイレクトを併用
- TeX式エラー（`k^*`）:
  - `k^{\ast}` を使用

## 8. 次フェーズ検証（A/B/C）
### A) 判定性能比較
```bash
python colab/colab_eval_detectors.py \
  --axis-scores /content/outputs/axis_build_4d/axis_scores.csv \
  --sample-metrics /content/outputs/axis_build_4d/sample_metrics.csv \
  --out-dir /content/outputs/eval_detectors
```

### B) 四象限代表誤分類の抽出
```bash
python colab/colab_quadrant_cases.py \
  --axis-scores /content/outputs/axis_build_4d/axis_scores.csv \
  --out-dir /content/outputs/quadrant_cases \
  --top-n 8
```

### C) ロバスト性（プロンプト・分割・別VLM）
```bash
python colab/colab_run_robustness.py \
  --project-dir /content \
  --output-root /content/outputs/robustness \
  --prompt-templates "a photo of {name}||a close-up photo of {name}" \
  --seeds "42,43,44" \
  --models "ViT-B-32:laion2b_s34b_b79k||ViT-B-16:laion2b_s34b_b88k"
```

### D) 統計表 + 10グラフ生成
```bash
python colab/colab_make_statistics_figures.py \
  --axis-dir /content/outputs/axis_build_4d \
  --out-dir /content/outputs/stats_figs_2026-02-16_v3 \
  --test-size 0.30 \
  --seed 42 \
  --bootstrap 800
```
