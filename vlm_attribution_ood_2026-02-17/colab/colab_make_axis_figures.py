#!/usr/bin/env python3
"""Create publication-ready figures for N-Axis Attribution OoD (Colab).

Inputs (CSV):
- axis_scores.csv
- axis_loadings.csv
- sparsepca_cv_table.csv
- optional: summary_axis_only.json

Outputs (PNG):
- fig1_zu_zc_scatter.png
- fig2_zu_zc_density.png
- fig3_axis_loadings.png
- fig4_cv_1se.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_zu_zc_scatter(axis_scores: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    id_mask = axis_scores["is_ood"].to_numpy(dtype=int) == 0
    ood_mask = ~id_mask

    ax.scatter(
        axis_scores.loc[id_mask, "z_u"],
        axis_scores.loc[id_mask, "z_c"],
        s=8,
        alpha=0.35,
        c="#2A6F97",
        label="ID",
        edgecolors="none",
    )
    ax.scatter(
        axis_scores.loc[ood_mask, "z_u"],
        axis_scores.loc[ood_mask, "z_c"],
        s=8,
        alpha=0.35,
        c="#C44536",
        label="OoD",
        edgecolors="none",
    )
    ax.set_xlabel(r"$z_u$ (uncertainty-related axis)")
    ax.set_ylabel(r"$z_c$ (confidence-related axis)")
    ax.set_title(r"Axis Space Separation: $z_u$ vs $z_c$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _kde_contour(ax, x: np.ndarray, y: np.ndarray, color: str, label: str) -> None:
    # Use scipy when available; fallback to 2D hist contour.
    try:
        from scipy.stats import gaussian_kde

        values = np.vstack([x, y])
        kde = gaussian_kde(values)
        x_min, x_max = np.percentile(x, [1, 99])
        y_min, y_max = np.percentile(y, [1, 99])
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 120),
            np.linspace(y_min, y_max, 120),
        )
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        levels = np.quantile(zz, [0.65, 0.8, 0.9, 0.97])
        ax.contour(xx, yy, zz, levels=levels, colors=color, linewidths=1.2)
        ax.plot([], [], color=color, linewidth=1.8, label=label)
    except Exception:
        h, xedges, yedges = np.histogram2d(x, y, bins=60, density=True)
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        xx, yy = np.meshgrid(xc, yc, indexing="xy")
        h2 = h.T
        levels = np.quantile(h2[h2 > 0], [0.6, 0.8, 0.9]) if np.any(h2 > 0) else [0.1, 0.2, 0.3]
        ax.contour(xx, yy, h2, levels=levels, colors=color, linewidths=1.2)
        ax.plot([], [], color=color, linewidth=1.8, label=label)


def plot_zu_zc_density(axis_scores: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    id_mask = axis_scores["is_ood"].to_numpy(dtype=int) == 0
    ood_mask = ~id_mask

    x_id = axis_scores.loc[id_mask, "z_u"].to_numpy(dtype=float)
    y_id = axis_scores.loc[id_mask, "z_c"].to_numpy(dtype=float)
    x_ood = axis_scores.loc[ood_mask, "z_u"].to_numpy(dtype=float)
    y_ood = axis_scores.loc[ood_mask, "z_c"].to_numpy(dtype=float)

    _kde_contour(ax, x_id, y_id, "#2A6F97", "ID density")
    _kde_contour(ax, x_ood, y_ood, "#C44536", "OoD density")

    ax.set_xlabel(r"$z_u$ (uncertainty-related axis)")
    ax.set_ylabel(r"$z_c$ (confidence-related axis)")
    ax.set_title(r"2D Density in Axis Space")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_axis_loadings(axis_loadings: pd.DataFrame, out_path: Path) -> None:
    plot_cols = [c for c in axis_loadings.columns if c not in {"axis", "nnz_abs_gt_1e-6"}]
    n_axes = axis_loadings.shape[0]

    fig, axes = plt.subplots(n_axes, 1, figsize=(9.2, 2.2 * n_axes), dpi=150, sharex=True)
    if n_axes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        row = axis_loadings.iloc[i]
        vals = row[plot_cols].to_numpy(dtype=float)
        colors = ["#2A9D8F" if v >= 0 else "#E76F51" for v in vals]
        ax.bar(plot_cols, vals, color=colors, alpha=0.9)
        ax.axhline(0.0, color="black", linewidth=0.8)
        axis_name = str(row["axis"])
        nnz = int(row["nnz_abs_gt_1e-6"]) if "nnz_abs_gt_1e-6" in row else np.nan
        ax.set_ylabel(axis_name)
        ax.set_title(f"{axis_name} loadings (nnz={nnz})", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.2)

    axes[-1].tick_params(axis="x", rotation=25)
    fig.suptitle("SparsePCA Loadings per Axis", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cv_1se(cv_table: pd.DataFrame, out_path: Path, summary: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=150)
    k_values = sorted(cv_table["k"].unique().tolist())

    # Per-alpha lines across k
    for alpha, grp in cv_table.groupby("alpha"):
        g = grp.sort_values("k")
        ax.plot(
            g["k"].to_numpy(dtype=int),
            g["cv_mse_mean"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.2,
            alpha=0.7,
            label=f"alpha={alpha:g}",
        )

    # Highlight global best
    best_idx = int(cv_table["cv_mse_mean"].idxmin())
    best_row = cv_table.loc[best_idx]
    ax.scatter(
        [best_row["k"]],
        [best_row["cv_mse_mean"]],
        c="black",
        s=50,
        marker="*",
        zorder=5,
        label="global min",
    )

    # Draw 1SE threshold when present
    if summary is not None and "one_se_selection" in summary:
        one_se = summary["one_se_selection"].get("one_se_threshold", None)
        k_sel = summary["one_se_selection"].get("k_selected", None)
        a_sel = summary["one_se_selection"].get("alpha_selected", None)
        if one_se is not None:
            ax.axhline(float(one_se), color="#C44536", linestyle="--", linewidth=1.5, label="1SE threshold")
        if k_sel is not None:
            # mark selected point
            sel_rows = cv_table[(cv_table["k"] == int(k_sel)) & (cv_table["alpha"] == float(a_sel))]
            if len(sel_rows) > 0:
                y_sel = float(sel_rows.iloc[0]["cv_mse_mean"])
                ax.scatter([k_sel], [y_sel], c="#264653", s=55, marker="D", zorder=6, label="selected")

    ax.set_xticks(k_values)
    ax.set_xlabel("Number of axes (k)")
    ax.set_ylabel("CV reconstruction MSE")
    ax.set_title("Model Selection Curve (CV MSE + 1SE)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis-scores", type=str, required=True)
    p.add_argument("--axis-loadings", type=str, required=True)
    p.add_argument("--cv-table", type=str, required=True)
    p.add_argument("--summary-json", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    axis_scores = pd.read_csv(args.axis_scores)
    axis_loadings = pd.read_csv(args.axis_loadings)
    cv_table = pd.read_csv(args.cv_table)

    summary = None
    if args.summary_json:
        s_path = Path(args.summary_json)
        if s_path.exists():
            with s_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)

    plot_zu_zc_scatter(axis_scores, out_dir / "fig1_zu_zc_scatter.png")
    plot_zu_zc_density(axis_scores, out_dir / "fig2_zu_zc_density.png")
    plot_axis_loadings(axis_loadings, out_dir / "fig3_axis_loadings.png")
    plot_cv_1se(cv_table, out_dir / "fig4_cv_1se.png", summary=summary)

    manifest = {
        "figures": [
            "fig1_zu_zc_scatter.png",
            "fig2_zu_zc_density.png",
            "fig3_axis_loadings.png",
            "fig4_cv_1se.png",
        ]
    }
    with (out_dir / "figure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

