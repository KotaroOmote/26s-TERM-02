#!/usr/bin/env python3
"""Build statistical tables and 10 publication figures for Attribution OoD.

Inputs (under --axis-dir):
- axis_scores.csv
- axis_loadings.csv
- sparsepca_cv_table.csv
- summary_axis_only.json

Outputs (under --out-dir):
- table_model_performance.csv
- table_feature_significance.csv
- table_quadrant_errors.csv
- fig01_perf_ci_bar.png
- fig02_roc_curves.png
- fig03_pr_curves.png
- fig04_score_distributions.png
- fig05_zu_zc_scatter_boundary.png
- fig06_zu_zc_density_contours.png
- fig07_quadrant_errors.png
- fig08_axis_loadings_heatmap.png
- fig09_cv_heatmap_1se.png
- fig10_effect_size_forest.png
- summary_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--axis-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=800)
    p.add_argument("--plot-max-samples", type=int, default=12000)
    return p.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _standardize_ood_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_ood" not in out.columns:
        out["is_ood"] = (out["source"] != "id").astype(int)
    out["is_ood"] = out["is_ood"].astype(int)
    return out


def _mwu_effect_delta(u_stat: float, n_pos: int, n_neg: int) -> float:
    return float((2.0 * u_stat / (n_pos * n_neg)) - 1.0)


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


def _bootstrap_ci_metric(
    y_true: np.ndarray,
    score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: List[float] = []
    for _ in range(n_bootstrap):
        b = rng.integers(0, n, n)
        vals.append(float(metric_fn(y_true[b], score[b])))
    arr = np.asarray(vals, dtype=float)
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def _bootstrap_ci_effect_delta(
    x_ood: np.ndarray,
    x_id: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(x_ood), len(x_id)
    vals = []
    for _ in range(n_bootstrap):
        xo = x_ood[rng.integers(0, n_pos, n_pos)]
        xi = x_id[rng.integers(0, n_neg, n_neg)]
        u = mannwhitneyu(xo, xi, alternative="two-sided").statistic
        vals.append(_mwu_effect_delta(float(u), n_pos, n_neg))
    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def tnr_at_target_tpr(y_true: np.ndarray, score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return float("nan")
    return float(1.0 - fpr[idx[0]])


def quadrant_of(zu: float, zc: float) -> str:
    if zu >= 0 and zc >= 0:
        return "Q1(+u,+c)"
    if zu < 0 and zc >= 0:
        return "Q2(-u,+c)"
    if zu < 0 and zc < 0:
        return "Q3(-u,-c)"
    return "Q4(+u,-c)"


def _resolve_cv_columns(cv: pd.DataFrame) -> Tuple[str, str]:
    # Support both historical names and current names.
    if {"mean_mse", "se_mse"}.issubset(cv.columns):
        return "mean_mse", "se_mse"
    if {"cv_mse_mean", "cv_mse_se"}.issubset(cv.columns):
        return "cv_mse_mean", "cv_mse_se"
    raise ValueError("CV table must include either (mean_mse,se_mse) or (cv_mse_mean,cv_mse_se).")


def build_scores(df: pd.DataFrame, test_size: float, seed: int) -> Dict[str, object]:
    y = df["is_ood"].to_numpy(dtype=int)
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    zu = df["z_u"].to_numpy(dtype=float)
    zc = df["z_c"].to_numpy(dtype=float)
    conf = df["conf"].to_numpy(dtype=float)
    energy = df["energy"].to_numpy(dtype=float)

    scores: Dict[str, np.ndarray] = {}
    scores["msp_single"] = 1.0 - conf
    scores["energy_single"] = energy

    zu_n = (zu - zu[tr_idx].mean()) / (zu[tr_idx].std() + 1e-12)
    zc_n = (zc - zc[tr_idx].mean()) / (zc[tr_idx].std() + 1e-12)
    scores["zsum_1d"] = zu_n + zc_n

    X = np.c_[zu, zc]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    lr.fit(X[tr_idx], y[tr_idx])
    scores["logistic_2d"] = lr.predict_proba(X)[:, 1]

    svm = make_pipeline(StandardScaler(), LinearSVC(random_state=seed))
    svm.fit(X[tr_idx], y[tr_idx])
    scores["linear_svm"] = svm.decision_function(X)

    return {
        "tr_idx": tr_idx,
        "te_idx": te_idx,
        "y": y,
        "scores": scores,
        "logistic_model": lr,
    }


def make_performance_table(
    y: np.ndarray,
    te_idx: np.ndarray,
    scores: Dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    yt = y[te_idx]
    rows: List[Dict[str, object]] = []
    ordered = ["energy_single", "logistic_2d", "linear_svm", "zsum_1d", "msp_single"]
    for name in ordered:
        s = scores[name][te_idx]
        auroc = float(roc_auc_score(yt, s))
        aupr = float(average_precision_score(yt, s))
        tnr95 = float(tnr_at_target_tpr(yt, s, target_tpr=0.95))
        au_lo, au_hi = _bootstrap_ci_metric(yt, s, lambda a, b: roc_auc_score(a, b), n_bootstrap, seed)
        tn_lo, tn_hi = _bootstrap_ci_metric(
            yt,
            s,
            lambda a, b: tnr_at_target_tpr(a, b, target_tpr=0.95),
            n_bootstrap,
            seed + 1,
        )
        rows.append(
            {
                "model": name,
                "AUROC": auroc,
                "AUROC_95CI_low": au_lo,
                "AUROC_95CI_high": au_hi,
                "AUPR": aupr,
                "TNR@95TPR": tnr95,
                "TNR95_95CI_low": tn_lo,
                "TNR95_95CI_high": tn_hi,
            }
        )
    return pd.DataFrame(rows)


def make_significance_table(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    candidates = [
        "z_u",
        "z_c",
        "d_conf_drop",
        "d_entropy_gain",
        "d_energy_gain",
        "d_oodscore_gain",
        "conf",
        "entropy_norm",
        "energy",
        "ood_score",
    ]
    features = [c for c in candidates if c in df.columns]

    id_mask = df["is_ood"] == 0
    ood_mask = df["is_ood"] == 1

    rows = []
    for f in features:
        x_ood = df.loc[ood_mask, f].to_numpy(dtype=float)
        x_id = df.loc[id_mask, f].to_numpy(dtype=float)
        u_stat, p_u = mannwhitneyu(x_ood, x_id, alternative="two-sided")
        delta = _mwu_effect_delta(float(u_stat), len(x_ood), len(x_id))
        d_lo, d_hi = _bootstrap_ci_effect_delta(x_ood, x_id, n_bootstrap=max(300, n_bootstrap // 2), seed=seed)
        rows.append(
            {
                "feature": f,
                "ood_mean": float(np.mean(x_ood)),
                "id_mean": float(np.mean(x_id)),
                "mean_diff(ood-id)": float(np.mean(x_ood) - np.mean(x_id)),
                "mannwhitney_p": float(p_u),
                "effect_delta": delta,
                "effect_delta_ci_low": d_lo,
                "effect_delta_ci_high": d_hi,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_q"] = _bh_fdr(out["mannwhitney_p"].to_numpy(dtype=float))
    out = out.sort_values("fdr_q").reset_index(drop=True)
    return out


def make_quadrant_table(df: pd.DataFrame, te_idx: np.ndarray, score_logit: np.ndarray) -> pd.DataFrame:
    out = df.iloc[te_idx].copy()
    out["p_ood"] = score_logit[te_idx]
    out["pred_ood"] = (out["p_ood"] >= 0.5).astype(int)
    out["is_fp"] = ((out["is_ood"] == 0) & (out["pred_ood"] == 1)).astype(int)
    out["is_fn"] = ((out["is_ood"] == 1) & (out["pred_ood"] == 0)).astype(int)
    out["quadrant"] = [quadrant_of(float(a), float(b)) for a, b in zip(out["z_u"], out["z_c"])]

    q = (
        out.groupby("quadrant", as_index=False)
        .agg(total=("is_ood", "count"), ood_rate=("is_ood", "mean"), fp=("is_fp", "sum"), fn=("is_fn", "sum"))
        .set_index("quadrant")
        .reindex(["Q1(+u,+c)", "Q2(-u,+c)", "Q3(-u,-c)", "Q4(+u,-c)"])
        .reset_index()
    )
    return q


def plot_all(
    df: pd.DataFrame,
    loadings: pd.DataFrame,
    cv: pd.DataFrame,
    summary: Dict[str, object],
    perf: pd.DataFrame,
    sig: pd.DataFrame,
    quad: pd.DataFrame,
    y: np.ndarray,
    te_idx: np.ndarray,
    scores: Dict[str, np.ndarray],
    lr_model,
    out_dir: Path,
    max_samples: int,
) -> None:
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["font.size"] = 11

    # fig01
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    order = perf["model"].tolist()
    p = perf.set_index("model").loc[order]
    x = np.arange(len(order))
    ax[0].bar(x, p["AUROC"], color="#4C78A8")
    ax[0].errorbar(
        x,
        p["AUROC"],
        yerr=[p["AUROC"] - p["AUROC_95CI_low"], p["AUROC_95CI_high"] - p["AUROC"]],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(order, rotation=24, ha="right")
    ax[0].set_ylim(0.5, 1.01)
    ax[0].set_title("AUROC (95% CI)")

    ax[1].bar(x, p["TNR@95TPR"], color="#F58518")
    ax[1].errorbar(
        x,
        p["TNR@95TPR"],
        yerr=[p["TNR@95TPR"] - p["TNR95_95CI_low"], p["TNR95_95CI_high"] - p["TNR@95TPR"]],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(order, rotation=24, ha="right")
    ax[1].set_ylim(0.0, 1.01)
    ax[1].set_title("TNR@95TPR (95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_perf_ci_bar.png", bbox_inches="tight")
    plt.close(fig)

    # fig02
    fig, ax = plt.subplots(figsize=(7, 6))
    yt = y[te_idx]
    ordered = ["energy_single", "logistic_2d", "linear_svm", "zsum_1d", "msp_single"]
    for name in ordered:
        s = scores[name][te_idx]
        fpr, tpr, _ = roc_curve(yt, s)
        auc = roc_auc_score(yt, s)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves (test split)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_roc_curves.png", bbox_inches="tight")
    plt.close(fig)

    # fig03
    fig, ax = plt.subplots(figsize=(7, 6))
    base = float(np.mean(yt))
    for name in ordered:
        s = scores[name][te_idx]
        prec, rec, _ = precision_recall_curve(yt, s)
        ap = average_precision_score(yt, s)
        ax.plot(rec, prec, linewidth=2, label=f"{name} (AP={ap:.3f})")
    ax.hlines(base, 0, 1, colors="k", linestyles="--", label=f"baseline={base:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # fig04
    te_df = df.iloc[te_idx].copy()
    te_df["p_ood"] = scores["logistic_2d"][te_idx]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(te_df.loc[te_df["is_ood"] == 0, "energy"], bins=60, density=True, alpha=0.55, label="ID", color="#4C78A8")
    ax[0].hist(te_df.loc[te_df["is_ood"] == 1, "energy"], bins=60, density=True, alpha=0.55, label="OoD", color="#E45756")
    ax[0].set_title("Energy distribution (test)")
    ax[0].legend()
    ax[1].hist(te_df.loc[te_df["is_ood"] == 0, "p_ood"], bins=60, density=True, alpha=0.55, label="ID", color="#4C78A8")
    ax[1].hist(te_df.loc[te_df["is_ood"] == 1, "p_ood"], bins=60, density=True, alpha=0.55, label="OoD", color="#E45756")
    ax[1].set_title("Logistic p(OoD) distribution (test)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_score_distributions.png", bbox_inches="tight")
    plt.close(fig)

    # fig05
    p_df = te_df.sample(min(max_samples, len(te_df)), random_state=42).copy()
    x_min, x_max = np.percentile(te_df["z_u"], [1, 99])
    y_min, y_max = np.percentile(te_df["z_c"], [1, 99])
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.c_[gx.ravel(), gy.ravel()]
    gp = lr_model.predict_proba(grid)[:, 1].reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(gx, gy, gp, levels=np.linspace(0, 1, 15), cmap="RdBu_r", alpha=0.35)
    ax.contour(gx, gy, gp, levels=[0.5], colors="black", linewidths=1.8)
    id_plot = p_df[p_df["is_ood"] == 0]
    ood_plot = p_df[p_df["is_ood"] == 1]
    ax.scatter(id_plot["z_u"], id_plot["z_c"], s=7, alpha=0.25, label="ID", color="#4C78A8")
    ax.scatter(ood_plot["z_u"], ood_plot["z_c"], s=7, alpha=0.25, label="OoD", color="#E45756")
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("z_u")
    ax.set_ylabel("z_c")
    ax.set_title("z_u-z_c scatter with logistic boundary")
    ax.legend(loc="upper left")
    fig.colorbar(cf, ax=ax, label="p(OoD)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_zu_zc_scatter_boundary.png", bbox_inches="tight")
    plt.close(fig)

    # fig06
    fig, ax = plt.subplots(figsize=(8, 7))
    for mask, color, label in [
        (df["is_ood"] == 0, "#4C78A8", "ID"),
        (df["is_ood"] == 1, "#E45756", "OoD"),
    ]:
        x = df.loc[mask, "z_u"].to_numpy(dtype=float)
        yv = df.loc[mask, "z_c"].to_numpy(dtype=float)
        h, xe, ye = np.histogram2d(x, yv, bins=80, density=True)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        xx, yy = np.meshgrid(xc, yc, indexing="xy")
        hz = h.T
        valid = hz[hz > 0]
        if len(valid) > 0:
            levels = np.quantile(valid, [0.55, 0.70, 0.82, 0.90, 0.96])
            ax.contour(xx, yy, hz, levels=levels, colors=[color], linewidths=1.4)
        ax.plot([], [], color=color, label=label)
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("z_u")
    ax.set_ylabel("z_c")
    ax.set_title("z_u-z_c density contours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig06_zu_zc_density_contours.png", bbox_inches="tight")
    plt.close(fig)

    # fig07
    fig, ax = plt.subplots(figsize=(8, 5))
    xi = np.arange(len(quad))
    ax.bar(xi, quad["fp"], label="FP", color="#F58518")
    ax.bar(xi, quad["fn"], bottom=quad["fp"], label="FN", color="#E45756")
    ax.set_xticks(xi)
    ax.set_xticklabels(quad["quadrant"])
    ax.set_ylabel("Count (test split)")
    ax.set_title("Error concentration by quadrant")
    for i, r in quad.iterrows():
        ax.text(i, float(r["fp"] + r["fn"]) + 4, f"{int(r['total'])}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig07_quadrant_errors.png", bbox_inches="tight")
    plt.close(fig)

    # fig08
    axis_cols = [c for c in loadings.columns if c.startswith("axis_")]
    mat = loadings[axis_cols].to_numpy(dtype=float).T
    vmax = float(np.max(np.abs(mat))) if mat.size else 1.0
    fig, ax = plt.subplots(figsize=(max(7.5, 1.8 * len(loadings)), max(4.0, 1.2 * len(axis_cols))))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(loadings)))
    ax.set_xticklabels(loadings["feature"].tolist(), rotation=28, ha="right")
    ax.set_yticks(np.arange(len(axis_cols)))
    ax.set_yticklabels(axis_cols)
    ax.set_title("SparsePCA loadings heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig08_axis_loadings_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # fig09
    mse_col, se_col = _resolve_cv_columns(cv)
    one_se_thr = None
    k_sel = None
    a_sel = None
    if "one_se_selection" in summary:
        sel = summary["one_se_selection"]
        one_se_thr = sel.get("one_se_threshold")
        k_sel = sel.get("k_selected")
        a_sel = sel.get("alpha_selected")

    pv = cv.pivot_table(index="alpha", columns="k", values=mse_col, aggfunc="mean")
    alphas = sorted(pv.index.tolist())
    ks = sorted(pv.columns.tolist())
    arr = pv.loc[alphas, ks].to_numpy(dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    im = ax[0].imshow(arr, cmap="viridis", aspect="auto")
    ax[0].set_xticks(np.arange(len(ks)))
    ax[0].set_xticklabels(ks)
    ax[0].set_yticks(np.arange(len(alphas)))
    ax[0].set_yticklabels([f"{a:.2g}" for a in alphas])
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("alpha")
    ax[0].set_title("CV mean MSE heatmap")
    if a_sel in alphas and k_sel in ks:
        ai = alphas.index(float(a_sel))
        ki = ks.index(int(k_sel))
        ax[0].plot(ki, ai, marker="*", markersize=14, markeredgecolor="white", markerfacecolor="red")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    best_per_k = cv.sort_values(mse_col).groupby("k", as_index=False).first().sort_values("k")
    ax[1].errorbar(
        best_per_k["k"],
        best_per_k[mse_col],
        yerr=best_per_k[se_col] if se_col in best_per_k.columns else None,
        marker="o",
        capsize=4,
    )
    if one_se_thr is not None:
        ax[1].axhline(float(one_se_thr), color="red", linestyle="--", label="1SE threshold")
    if k_sel is not None:
        ax[1].axvline(int(k_sel), color="green", linestyle="--", label=f"selected k={int(k_sel)}")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("CV MSE (best alpha per k)")
    ax[1].set_title("1SE model selection")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig09_cv_heatmap_1se.png", bbox_inches="tight")
    plt.close(fig)

    # fig10
    s = sig.sort_values("effect_delta").reset_index(drop=True)
    yy = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 2))
    ax.hlines(yy, s["effect_delta_ci_low"], s["effect_delta_ci_high"], color="gray", linewidth=2)
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in s["effect_delta"]]
    ax.scatter(s["effect_delta"], yy, c=colors, s=45, zorder=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(s["feature"])
    ax.set_xlabel("Effect size delta (OoD - ID, Mann-Whitney based)")
    ax.set_title("Feature effect sizes with 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(out_dir / "fig10_effect_size_forest.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    axis_dir = Path(args.axis_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    axis_scores = pd.read_csv(axis_dir / "axis_scores.csv")
    axis_scores = _standardize_ood_label(axis_scores)
    axis_loadings = pd.read_csv(axis_dir / "axis_loadings.csv")
    cv_table = pd.read_csv(axis_dir / "sparsepca_cv_table.csv")
    summary = _read_summary(axis_dir / "summary_axis_only.json")

    needed = ["z_u", "z_c", "conf", "energy", "is_ood"]
    missing = [c for c in needed if c not in axis_scores.columns]
    if missing:
        raise ValueError(f"axis_scores.csv missing required columns: {missing}")

    scores_bundle = build_scores(axis_scores, test_size=args.test_size, seed=args.seed)
    perf = make_performance_table(
        y=scores_bundle["y"],
        te_idx=scores_bundle["te_idx"],
        scores=scores_bundle["scores"],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    sig = make_significance_table(axis_scores, n_bootstrap=args.bootstrap, seed=args.seed)
    quad = make_quadrant_table(
        axis_scores,
        te_idx=scores_bundle["te_idx"],
        score_logit=scores_bundle["scores"]["logistic_2d"],
    )

    perf.to_csv(out_dir / "table_model_performance.csv", index=False)
    sig.to_csv(out_dir / "table_feature_significance.csv", index=False)
    quad.to_csv(out_dir / "table_quadrant_errors.csv", index=False)

    plot_all(
        df=axis_scores,
        loadings=axis_loadings,
        cv=cv_table,
        summary=summary,
        perf=perf,
        sig=sig,
        quad=quad,
        y=scores_bundle["y"],
        te_idx=scores_bundle["te_idx"],
        scores=scores_bundle["scores"],
        lr_model=scores_bundle["logistic_model"],
        out_dir=out_dir,
        max_samples=args.plot_max_samples,
    )

    result = {
        "axis_dir": str(axis_dir),
        "out_dir": str(out_dir),
        "n_total": int(len(axis_scores)),
        "n_test": int(len(scores_bundle["te_idx"])),
        "top_model_by_auroc": str(perf.sort_values("AUROC", ascending=False).iloc[0]["model"]),
        "files": {
            "table_model_performance": str(out_dir / "table_model_performance.csv"),
            "table_feature_significance": str(out_dir / "table_feature_significance.csv"),
            "table_quadrant_errors": str(out_dir / "table_quadrant_errors.csv"),
            "figures": [f"fig{str(i).zfill(2)}" for i in range(1, 11)],
        },
    }
    with (out_dir / "summary_stats.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

