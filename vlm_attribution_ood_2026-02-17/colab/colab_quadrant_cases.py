#!/usr/bin/env python3
"""Pick representative error cases from z_u-z_c quadrants.

Procedure:
1) Fit logistic detector on (z_u, z_c)
2) Choose threshold from train set at target TPR (default 95%)
3) Evaluate on test set and collect FP/FN
4) Summarize quadrant statistics and representative errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def threshold_at_target_tpr(y_true: np.ndarray, score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, thr = roc_curve(y_true, score)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return float(np.quantile(score, 0.95))
    return float(thr[idx[0]])


def assign_quadrants(z_u: np.ndarray, z_c: np.ndarray, u_center: float, c_center: float) -> np.ndarray:
    q = np.empty(len(z_u), dtype=object)
    q[(z_u >= u_center) & (z_c >= c_center)] = "Q1_high_u_high_c"
    q[(z_u < u_center) & (z_c >= c_center)] = "Q2_low_u_high_c"
    q[(z_u < u_center) & (z_c < c_center)] = "Q3_low_u_low_c"
    q[(z_u >= u_center) & (z_c < c_center)] = "Q4_high_u_low_c"
    return q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis-scores", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-tpr", type=float, default=0.95)
    p.add_argument("--top-n", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.axis_scores).copy()
    required = ["z_u", "z_c", "is_ood"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df))
    if "source" not in df.columns:
        df["source"] = "unknown"
    for c in ["conf", "entropy", "energy", "ood_score", "correct"]:
        if c not in df.columns:
            df[c] = np.nan

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

    scaler = StandardScaler().fit(Xtr)
    Xtrz = scaler.transform(Xtr)
    Xtez = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=3000, random_state=args.seed)
    clf.fit(Xtrz, ytr)
    s_tr = clf.predict_proba(Xtrz)[:, 1]
    s_te = clf.predict_proba(Xtez)[:, 1]

    tau = threshold_at_target_tpr(ytr, s_tr, target_tpr=args.target_tpr)
    pred_te = (s_te >= tau).astype(int)

    te = df.iloc[te_idx].copy().reset_index(drop=True)
    te["score_logistic"] = s_te
    te["pred_is_ood"] = pred_te
    te["error"] = (te["pred_is_ood"].to_numpy(dtype=int) != te["is_ood"].to_numpy(dtype=int)).astype(int)
    te["error_type"] = "correct"
    te.loc[(te["is_ood"] == 0) & (te["pred_is_ood"] == 1), "error_type"] = "FP"
    te.loc[(te["is_ood"] == 1) & (te["pred_is_ood"] == 0), "error_type"] = "FN"
    te["id_cls_error"] = ((te["is_ood"] == 0) & (te["correct"] == 0)).astype(int)

    u_center = float(np.median(df["z_u"].to_numpy(dtype=float)))
    c_center = float(np.median(df["z_c"].to_numpy(dtype=float)))
    te["quadrant"] = assign_quadrants(
        te["z_u"].to_numpy(dtype=float),
        te["z_c"].to_numpy(dtype=float),
        u_center=u_center,
        c_center=c_center,
    )

    q_rows: List[Dict[str, object]] = []
    for q, grp in te.groupby("quadrant"):
        q_rows.append(
            {
                "quadrant": q,
                "n_total": int(len(grp)),
                "n_error": int(grp["error"].sum()),
                "error_rate": float(grp["error"].mean()),
                "n_fp": int((grp["error_type"] == "FP").sum()),
                "n_fn": int((grp["error_type"] == "FN").sum()),
                "ood_ratio": float(grp["is_ood"].mean()),
                "id_cls_error_rate": float(grp.loc[grp["is_ood"] == 0, "id_cls_error"].mean()) if np.any(grp["is_ood"] == 0) else np.nan,
                "mean_conf": float(grp["conf"].mean()),
                "mean_entropy": float(grp["entropy"].mean()),
                "mean_energy": float(grp["energy"].mean()),
                "mean_ood_score": float(grp["ood_score"].mean()),
            }
        )
    q_summary = pd.DataFrame(q_rows).sort_values("quadrant").reset_index(drop=True)
    q_summary.to_csv(out_dir / "quadrant_summary.csv", index=False)

    err = te[te["error"] == 1].copy()
    rep_rows: List[pd.DataFrame] = []
    for q in sorted(err["quadrant"].unique().tolist()):
        gq = err[err["quadrant"] == q]
        # Strong FP: high score but actually ID
        fp = gq[gq["error_type"] == "FP"].sort_values("score_logistic", ascending=False).head(args.top_n)
        if len(fp) > 0:
            rep_rows.append(fp)
        # Strong FN: low score but actually OoD
        fn = gq[gq["error_type"] == "FN"].sort_values("score_logistic", ascending=True).head(args.top_n)
        if len(fn) > 0:
            rep_rows.append(fn)

    if len(rep_rows) > 0:
        rep = pd.concat(rep_rows, axis=0, ignore_index=True)
    else:
        rep = err.copy()

    rep_cols = [
        "quadrant",
        "error_type",
        "sample_id",
        "source",
        "is_ood",
        "pred_is_ood",
        "score_logistic",
        "z_u",
        "z_c",
        "conf",
        "entropy",
        "energy",
        "ood_score",
        "correct",
        "id_cls_error",
    ]
    rep = rep[rep_cols]
    rep.to_csv(out_dir / "quadrant_error_examples.csv", index=False)

    summary = {
        "n_total": int(len(df)),
        "n_test": int(len(te)),
        "target_tpr": float(args.target_tpr),
        "threshold": float(tau),
        "quadrant_center": {"z_u_median": u_center, "z_c_median": c_center},
        "files": {
            "quadrant_summary": str(out_dir / "quadrant_summary.csv"),
            "quadrant_error_examples": str(out_dir / "quadrant_error_examples.csv"),
        },
    }
    with (out_dir / "summary_quadrant_cases.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

