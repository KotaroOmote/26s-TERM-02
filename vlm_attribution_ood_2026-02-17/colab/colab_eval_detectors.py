#!/usr/bin/env python3
"""Evaluate OoD detectors on (z_u, z_c) and compare with MSP/Energy baselines.

Evaluated methods:
- msp_single:     score = 1 - msp
- energy_single:  score = oriented(energy)
- zsum_1d:        1D score from standardized (z_u + z_c)
- linear_svm:     Linear SVM decision function on (z_u, z_c)
- logistic_2d:    Logistic regression probability on (z_u, z_c)

Metrics:
- AUROC
- TNR@95TPR (positive class = OoD)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def tnr_at_95_tpr(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return float("nan")
    return float(1.0 - fpr[idx[0]])


def orient_score_by_train(y_train: np.ndarray, s_train: np.ndarray, s_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(np.unique(y_train)) < 2:
        return s_train, s_test
    au = roc_auc_score(y_train, s_train)
    if au < 0.5:
        return -s_train, -s_test
    return s_train, s_test


def compute_metrics(y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) >= 2 else float("nan"),
        "tnr_at_95tpr": tnr_at_95_tpr(y_true, score),
    }


def merge_inputs(axis_scores: pd.DataFrame, sample_metrics: pd.DataFrame) -> pd.DataFrame:
    # Prefer explicit keys; fallback to row-order merge for backward compatibility.
    if {"sample_id", "source"}.issubset(axis_scores.columns) and {"sample_id", "source"}.issubset(sample_metrics.columns):
        merged = axis_scores.merge(
            sample_metrics[["sample_id", "source", "msp", "energy", "ood_score", "conf", "entropy", "correct"]],
            on=["sample_id", "source"],
            how="left",
            suffixes=("", "_m"),
        )
        # keep axis_scores values if they already exist
        for c in ["msp", "energy", "ood_score", "conf", "entropy", "correct"]:
            alt = f"{c}_m"
            if c not in merged.columns and alt in merged.columns:
                merged[c] = merged[alt]
            elif c in merged.columns and alt in merged.columns:
                merged[c] = merged[c].fillna(merged[alt])
        merged = merged[[c for c in merged.columns if not c.endswith("_m")]]
        return merged

    n = min(len(axis_scores), len(sample_metrics))
    out = axis_scores.iloc[:n].copy()
    for c in ["msp", "energy", "ood_score", "conf", "entropy", "correct"]:
        if c not in out.columns and c in sample_metrics.columns:
            out[c] = sample_metrics.iloc[:n][c].to_numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis-scores", type=str, required=True)
    p.add_argument("--sample-metrics", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_scores = pd.read_csv(args.axis_scores)
    sample_metrics = pd.read_csv(args.sample_metrics)
    df = merge_inputs(axis_scores, sample_metrics)

    need_cols = ["z_u", "z_c", "is_ood", "msp", "energy"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")

    y = df["is_ood"].to_numpy(dtype=int)
    X = df[["z_u", "z_c"]].to_numpy(dtype=float)
    idx_all = np.arange(len(df))

    tr_idx, te_idx = train_test_split(
        idx_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    rows: List[Dict[str, object]] = []
    test_scores = pd.DataFrame({
        "row_id": te_idx,
        "is_ood": yte,
    })

    # Baseline: MSP single
    s_tr = 1.0 - df.iloc[tr_idx]["msp"].to_numpy(dtype=float)
    s_te = 1.0 - df.iloc[te_idx]["msp"].to_numpy(dtype=float)
    s_tr, s_te = orient_score_by_train(ytr, s_tr, s_te)
    m = compute_metrics(yte, s_te)
    rows.append({"model": "msp_single", **m})
    test_scores["score_msp_single"] = s_te

    # Baseline: Energy single
    e_tr = df.iloc[tr_idx]["energy"].to_numpy(dtype=float)
    e_te = df.iloc[te_idx]["energy"].to_numpy(dtype=float)
    e_tr, e_te = orient_score_by_train(ytr, e_tr, e_te)
    m = compute_metrics(yte, e_te)
    rows.append({"model": "energy_single", **m})
    test_scores["score_energy_single"] = e_te

    # 1D threshold model from (z_u, z_c): standardized z_u + z_c
    sc = StandardScaler().fit(Xtr)
    Xtrz = sc.transform(Xtr)
    Xtez = sc.transform(Xte)
    zsum_tr = (Xtrz[:, 0] + Xtrz[:, 1]) / np.sqrt(2.0)
    zsum_te = (Xtez[:, 0] + Xtez[:, 1]) / np.sqrt(2.0)
    zsum_tr, zsum_te = orient_score_by_train(ytr, zsum_tr, zsum_te)
    m = compute_metrics(yte, zsum_te)
    rows.append({"model": "zsum_1d", **m})
    test_scores["score_zsum_1d"] = zsum_te

    # Linear boundary (SVM)
    svm = LinearSVC(random_state=args.seed, class_weight="balanced")
    svm.fit(Xtrz, ytr)
    svm_tr = svm.decision_function(Xtrz)
    svm_te = svm.decision_function(Xtez)
    svm_tr, svm_te = orient_score_by_train(ytr, svm_tr, svm_te)
    m = compute_metrics(yte, svm_te)
    rows.append({"model": "linear_svm", **m})
    test_scores["score_linear_svm"] = svm_te

    # Logistic regression boundary
    lr = LogisticRegression(max_iter=3000, random_state=args.seed)
    lr.fit(Xtrz, ytr)
    lr_tr = lr.predict_proba(Xtrz)[:, 1]
    lr_te = lr.predict_proba(Xtez)[:, 1]
    lr_tr, lr_te = orient_score_by_train(ytr, lr_tr, lr_te)
    m = compute_metrics(yte, lr_te)
    rows.append({"model": "logistic_2d", **m})
    test_scores["score_logistic_2d"] = lr_te

    metric_df = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    metric_df.to_csv(out_dir / "detector_metrics.csv", index=False)
    test_scores.to_csv(out_dir / "detector_scores_test.csv", index=False)

    summary = {
        "n_total": int(len(df)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "files": {
            "detector_metrics": str(out_dir / "detector_metrics.csv"),
            "detector_scores_test": str(out_dir / "detector_scores_test.csv"),
        },
    }
    with (out_dir / "summary_eval.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

