#!/usr/bin/env python3
"""Colab-friendly axis builder for VLM Attribution OoD (step: data -> axis fixed).

What this script does:
1) Load TFDS datasets (ID/OoD)
2) Run OpenCLIP inference and build per-sample behavior metrics
3) Build axis feature matrix with variance-threshold filtering
4) Select SparsePCA (k, alpha) by CV reconstruction error + One-Standard-Error rule
5) Fit final SparsePCA and export fixed axes

No detector training is included here (step separation).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


@dataclass
class RunConfig:
    n_id: int = 10000
    n_ood_cifar: int = 5000
    n_ood_imagenetr: int = 5000
    batch_size: int = 64
    seed: int = 42
    device: str = "cuda"
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    prompt_template: str = "a photo of {name}"
    output_dir: str = "/content/outputs/axis_build"
    abstain_threshold: float = 0.45
    var_threshold: float = 1e-10
    k_min: int = 1
    k_max: int = 8
    alpha_grid: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0, 2.0)
    cv_splits: int = 5
    quiet: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_tfds_splits(cfg: RunConfig):
    import tensorflow_datasets as tfds

    if cfg.quiet:
        tfds.disable_progress_bar()

    id_ds, id_info = tfds.load(
        "food101",
        split=f"train[:{cfg.n_id}]",
        as_supervised=True,
        shuffle_files=False,
        with_info=True,
    )
    ood_cifar = tfds.load(
        "cifar100",
        split=f"train[:{cfg.n_ood_cifar}]",
        as_supervised=True,
        shuffle_files=False,
    )
    ood_imr = tfds.load(
        "imagenet_r",
        split=f"test[:{cfg.n_ood_imagenetr}]",
        as_supervised=True,
        shuffle_files=False,
    )

    label_names = list(id_info.features["label"].names)
    return id_ds, ood_cifar, ood_imr, label_names


def init_openclip(model_name: str, pretrained: str, device: str):
    import open_clip
    import torch

    use_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=use_device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return torch, model, preprocess, tokenizer, use_device


def iter_tfds_numpy(ds) -> Iterable[Tuple[np.ndarray, int]]:
    import tensorflow_datasets as tfds

    for image, label in tfds.as_numpy(ds):
        yield image, int(label)


def _flush_batch(
    rows: List[Dict[str, object]],
    batch_images: List[Image.Image],
    batch_labels: List[int],
    batch_indices: List[int],
    source: str,
    is_ood: int,
    label_names: Sequence[str],
    torch,
    model,
    preprocess,
    text_feat,
    device: str,
) -> None:
    if not batch_images:
        return

    with torch.no_grad():
        x = torch.stack([preprocess(img) for img in batch_images]).to(device)
        image_feat = model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        logits = 100.0 * (image_feat @ text_feat.T)
        probs = torch.softmax(logits, dim=-1)

        conf, pred_idx = torch.max(probs, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        energy = -logsumexp

    conf_np = conf.detach().cpu().numpy()
    pred_np = pred_idx.detach().cpu().numpy().astype(int)
    entropy_np = entropy.detach().cpu().numpy()
    energy_np = energy.detach().cpu().numpy()

    ent_denom = max(math.log(max(2, len(label_names))), 1e-12)
    entropy_norm_np = entropy_np / ent_denom
    ood_score_np = 0.5 * (1.0 - conf_np) + 0.5 * entropy_norm_np

    for i in range(len(batch_images)):
        gt_label = int(batch_labels[i]) if is_ood == 0 else -1
        pred_label = int(pred_np[i])
        correct = int(pred_label == gt_label) if is_ood == 0 else np.nan
        rows.append(
            {
                "sample_id": int(batch_indices[i]),
                "source": source,
                "is_ood": int(is_ood),
                "gt_label_idx": gt_label,
                "pred_label_idx": pred_label,
                "gt_label_name": label_names[gt_label] if gt_label >= 0 else "N/A",
                "pred_label_name": label_names[pred_label],
                "correct": correct,
                "conf": float(conf_np[i]),
                "msp": float(conf_np[i]),
                "entropy": float(entropy_np[i]),
                "entropy_norm": float(entropy_norm_np[i]),
                "energy": float(energy_np[i]),
                "ood_score": float(ood_score_np[i]),
            }
        )


def run_openclip_metrics(
    ds,
    n_samples: int,
    source: str,
    is_ood: int,
    label_names: Sequence[str],
    batch_size: int,
    torch,
    model,
    preprocess,
    tokenizer,
    device: str,
    prompt_template: str = "a photo of {name}",
    quiet: bool = False,
) -> pd.DataFrame:
    prompts: List[str] = []
    for name in label_names:
        if "{name}" in prompt_template:
            prompts.append(prompt_template.format(name=name))
        else:
            prompts.append(prompt_template.format(name))
    with torch.no_grad():
        tok = tokenizer(prompts).to(device)
        text_feat = model.encode_text(tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    rows: List[Dict[str, object]] = []
    batch_images: List[Image.Image] = []
    batch_labels: List[int] = []
    batch_indices: List[int] = []

    iterator = tqdm(
        iter_tfds_numpy(ds),
        total=n_samples,
        desc=f"Inference {source}",
        disable=quiet,
    )
    for idx, (img_np, label) in enumerate(iterator):
        img = Image.fromarray(np.asarray(img_np)).convert("RGB")
        batch_images.append(img)
        batch_labels.append(int(label))
        batch_indices.append(idx)
        if len(batch_images) >= batch_size:
            _flush_batch(
                rows=rows,
                batch_images=batch_images,
                batch_labels=batch_labels,
                batch_indices=batch_indices,
                source=source,
                is_ood=is_ood,
                label_names=label_names,
                torch=torch,
                model=model,
                preprocess=preprocess,
                text_feat=text_feat,
                device=device,
            )
            batch_images, batch_labels, batch_indices = [], [], []

    _flush_batch(
        rows=rows,
        batch_images=batch_images,
        batch_labels=batch_labels,
        batch_indices=batch_indices,
        source=source,
        is_ood=is_ood,
        label_names=label_names,
        torch=torch,
        model=model,
        preprocess=preprocess,
        text_feat=text_feat,
        device=device,
    )
    return pd.DataFrame(rows)


def build_axis_features(df: pd.DataFrame, abstain_threshold: float) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    out = df.copy()
    out["abstain"] = (out["conf"] < abstain_threshold).astype(float)

    id_base = out[out["is_ood"] == 0]
    if len(id_base) == 0:
        raise ValueError("ID samples are required to define baseline deltas.")

    base = {
        "mean_conf": float(id_base["conf"].mean()),
        "mean_entropy_norm": float(id_base["entropy_norm"].mean()),
        "mean_energy": float(id_base["energy"].mean()),
        "mean_ood_score": float(id_base["ood_score"].mean()),
    }

    out["d_conf_drop"] = base["mean_conf"] - out["conf"].to_numpy(dtype=float)
    out["d_entropy_gain"] = out["entropy_norm"].to_numpy(dtype=float) - base["mean_entropy_norm"]
    out["d_energy_gain"] = out["energy"].to_numpy(dtype=float) - base["mean_energy"]
    out["d_oodscore_gain"] = out["ood_score"].to_numpy(dtype=float) - base["mean_ood_score"]

    feature_cols = [
        "d_conf_drop",
        "d_entropy_gain",
        "d_energy_gain",
        "d_oodscore_gain",
    ]
    return out, feature_cols, base


def apply_variance_threshold(
    X: np.ndarray,
    feature_cols: Sequence[str],
    threshold: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    vt = VarianceThreshold(threshold=threshold)
    Xf = vt.fit_transform(X)
    kept_mask = vt.get_support()
    kept = [c for c, keep in zip(feature_cols, kept_mask) if keep]
    dropped = [c for c, keep in zip(feature_cols, kept_mask) if not keep]
    if Xf.shape[1] == 0:
        raise ValueError("All features were removed by variance threshold.")
    return Xf, kept, dropped


def sparsepca_cv_table(
    X: np.ndarray,
    y: np.ndarray,
    k_values: Sequence[int],
    alpha_values: Sequence[float],
    cv_splits: int,
    seed: int,
) -> pd.DataFrame:
    if len(np.unique(y)) >= 2:
        splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X)

    # Materialize splits once for fair comparison across all (k, alpha).
    splits = list(split_iter)

    rows: List[Dict[str, float]] = []
    for k in k_values:
        for alpha in alpha_values:
            fold_errors: List[float] = []
            for tr_idx, va_idx in splits:
                Xtr, Xva = X[tr_idx], X[va_idx]
                scaler = StandardScaler().fit(Xtr)
                Xtrz = scaler.transform(Xtr)
                Xvaz = scaler.transform(Xva)

                model = SparsePCA(
                    n_components=k,
                    alpha=alpha,
                    random_state=seed,
                    max_iter=2000,
                )
                model.fit(Xtrz)
                Zva = model.transform(Xvaz)
                Xhat = Zva @ model.components_
                mse = float(np.mean((Xvaz - Xhat) ** 2))
                fold_errors.append(mse)

            mean_mse = float(np.mean(fold_errors))
            se_mse = float(np.std(fold_errors, ddof=1) / math.sqrt(len(fold_errors)))
            rows.append(
                {
                    "k": int(k),
                    "alpha": float(alpha),
                    "cv_mse_mean": mean_mse,
                    "cv_mse_se": se_mse,
                }
            )
    return pd.DataFrame(rows).sort_values(["k", "alpha"]).reset_index(drop=True)


def select_by_one_se(cv_table: pd.DataFrame) -> Dict[str, float]:
    best_idx = int(cv_table["cv_mse_mean"].idxmin())
    best_row = cv_table.loc[best_idx]
    threshold = float(best_row["cv_mse_mean"] + best_row["cv_mse_se"])

    cand = cv_table[cv_table["cv_mse_mean"] <= threshold].copy()
    # Simpler model preference: lower k first, then higher alpha (sparser).
    cand = cand.sort_values(["k", "alpha"], ascending=[True, False]).reset_index(drop=True)
    picked = cand.iloc[0]

    return {
        "k_selected": int(picked["k"]),
        "alpha_selected": float(picked["alpha"]),
        "cv_mse_min": float(best_row["cv_mse_mean"]),
        "cv_mse_se_at_min": float(best_row["cv_mse_se"]),
        "one_se_threshold": threshold,
    }


def fit_final_sparsepca(
    X: np.ndarray,
    k: int,
    alpha: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    model = SparsePCA(
        n_components=k,
        alpha=alpha,
        random_state=seed,
        max_iter=3000,
    )
    model.fit(Xz)
    Z = model.transform(Xz)
    return model.components_, Z, scaler


def infer_zu_zc_axes(loadings: np.ndarray, feature_cols: Sequence[str]) -> Dict[str, int]:
    feat_to_idx = {f: i for i, f in enumerate(feature_cols)}
    u_feats = [f for f in ["d_entropy_gain", "d_oodscore_gain", "d_energy_gain"] if f in feat_to_idx]
    c_feats = [f for f in ["d_conf_drop"] if f in feat_to_idx]

    abs_load = np.abs(loadings)
    u_scores = np.zeros(loadings.shape[0], dtype=float)
    c_scores = np.zeros(loadings.shape[0], dtype=float)

    for f in u_feats:
        u_scores += abs_load[:, feat_to_idx[f]]
    for f in c_feats:
        c_scores += abs_load[:, feat_to_idx[f]]

    axis_u = int(np.argmax(u_scores))
    axis_c = int(np.argmax(c_scores))
    if axis_c == axis_u and loadings.shape[0] > 1:
        order = np.argsort(-c_scores)
        axis_c = int(order[1])

    return {
        "axis_u_index": axis_u,
        "axis_c_index": axis_c,
    }


def run(cfg: RunConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    if cfg.quiet:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        try:
            from absl import logging as absl_logging

            absl_logging.set_verbosity("error")
        except Exception:
            pass

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_ds, ood_cifar, ood_imr, label_names = load_tfds_splits(cfg)
    torch, model, preprocess, tokenizer, device = init_openclip(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        device=cfg.device,
    )

    df_id = run_openclip_metrics(
        ds=id_ds,
        n_samples=cfg.n_id,
        source="food101_id",
        is_ood=0,
        label_names=label_names,
        batch_size=cfg.batch_size,
        torch=torch,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        prompt_template=cfg.prompt_template,
        quiet=cfg.quiet,
    )
    df_ood_cifar = run_openclip_metrics(
        ds=ood_cifar,
        n_samples=cfg.n_ood_cifar,
        source="cifar100_ood",
        is_ood=1,
        label_names=label_names,
        batch_size=cfg.batch_size,
        torch=torch,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        prompt_template=cfg.prompt_template,
        quiet=cfg.quiet,
    )
    df_ood_imr = run_openclip_metrics(
        ds=ood_imr,
        n_samples=cfg.n_ood_imagenetr,
        source="imagenet_r_ood",
        is_ood=1,
        label_names=label_names,
        batch_size=cfg.batch_size,
        torch=torch,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        prompt_template=cfg.prompt_template,
        quiet=cfg.quiet,
    )

    metrics_df = pd.concat([df_id, df_ood_cifar, df_ood_imr], axis=0, ignore_index=True)
    metrics_df.to_csv(out_dir / "sample_metrics.csv", index=False)

    feat_df, feat_cols, baseline = build_axis_features(metrics_df, abstain_threshold=cfg.abstain_threshold)
    X = feat_df[feat_cols].to_numpy(dtype=float)
    y = feat_df["is_ood"].to_numpy(dtype=int)

    Xf, kept_cols, dropped_cols = apply_variance_threshold(X, feat_cols, threshold=cfg.var_threshold)
    feat_df.to_csv(out_dir / "axis_features_raw.csv", index=False)

    k_upper = min(cfg.k_max, Xf.shape[1])
    k_values = list(range(cfg.k_min, k_upper + 1))
    cv_tbl = sparsepca_cv_table(
        X=Xf,
        y=y,
        k_values=k_values,
        alpha_values=cfg.alpha_grid,
        cv_splits=cfg.cv_splits,
        seed=cfg.seed,
    )
    cv_tbl.to_csv(out_dir / "sparsepca_cv_table.csv", index=False)

    sel = select_by_one_se(cv_tbl)
    loadings, Z, scaler = fit_final_sparsepca(
        X=Xf,
        k=sel["k_selected"],
        alpha=sel["alpha_selected"],
        seed=cfg.seed,
    )

    mapping = infer_zu_zc_axes(loadings, kept_cols)

    load_df = pd.DataFrame(loadings, columns=kept_cols)
    load_df.insert(0, "axis", [f"axis_{i}" for i in range(loadings.shape[0])])
    load_df["nnz_abs_gt_1e-6"] = (np.abs(loadings) > 1e-6).sum(axis=1)
    load_df.to_csv(out_dir / "axis_loadings.csv", index=False)

    score_df = pd.DataFrame(Z, columns=[f"axis_{i}" for i in range(Z.shape[1])])
    score_df["sample_id"] = feat_df["sample_id"].to_numpy(dtype=int)
    score_df["is_ood"] = y
    score_df["source"] = feat_df["source"].to_numpy()
    score_df["conf"] = feat_df["conf"].to_numpy(dtype=float)
    score_df["msp"] = feat_df["msp"].to_numpy(dtype=float)
    score_df["entropy"] = feat_df["entropy"].to_numpy(dtype=float)
    score_df["energy"] = feat_df["energy"].to_numpy(dtype=float)
    score_df["ood_score"] = feat_df["ood_score"].to_numpy(dtype=float)
    score_df["correct"] = feat_df["correct"].to_numpy()
    score_df["z_u"] = score_df[f"axis_{mapping['axis_u_index']}"]
    score_df["z_c"] = score_df[f"axis_{mapping['axis_c_index']}"]
    score_df.to_csv(out_dir / "axis_scores.csv", index=False)

    var_info = {
        "variance_threshold": float(cfg.var_threshold),
        "features_before": feat_cols,
        "features_kept": kept_cols,
        "features_dropped": dropped_cols,
    }
    with (out_dir / "variance_filter.json").open("w", encoding="utf-8") as f:
        json.dump(var_info, f, ensure_ascii=False, indent=2)

    summary = {
        "config": {
            "n_id": cfg.n_id,
            "n_ood_cifar": cfg.n_ood_cifar,
            "n_ood_imagenetr": cfg.n_ood_imagenetr,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "device": device,
            "model_name": cfg.model_name,
            "pretrained": cfg.pretrained,
            "prompt_template": cfg.prompt_template,
            "quiet": cfg.quiet,
        },
        "baseline_id_means": baseline,
        "one_se_selection": sel,
        "axis_mapping": mapping,
        "files": {
            "sample_metrics": str(out_dir / "sample_metrics.csv"),
            "axis_features_raw": str(out_dir / "axis_features_raw.csv"),
            "variance_filter": str(out_dir / "variance_filter.json"),
            "sparsepca_cv_table": str(out_dir / "sparsepca_cv_table.csv"),
            "axis_loadings": str(out_dir / "axis_loadings.csv"),
            "axis_scores": str(out_dir / "axis_scores.csv"),
        },
    }

    with (out_dir / "summary_axis_only.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TFDS -> SparsePCA axis builder (axis-only stage).")
    p.add_argument("--n-id", type=int, default=10000)
    p.add_argument("--n-ood-cifar", type=int, default=5000)
    p.add_argument("--n-ood-imagenetr", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-name", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--prompt-template", type=str, default="a photo of {name}")
    p.add_argument("--output-dir", type=str, default="/content/outputs/axis_build")
    p.add_argument("--abstain-threshold", type=float, default=0.45)
    p.add_argument("--var-threshold", type=float, default=1e-10)
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--alpha-grid", type=str, default="0.1,0.3,0.5,1.0,2.0")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alpha_vals = tuple(float(x.strip()) for x in args.alpha_grid.split(",") if x.strip())
    cfg = RunConfig(
        n_id=args.n_id,
        n_ood_cifar=args.n_ood_cifar,
        n_ood_imagenetr=args.n_ood_imagenetr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        model_name=args.model_name,
        pretrained=args.pretrained,
        prompt_template=args.prompt_template,
        output_dir=args.output_dir,
        abstain_threshold=args.abstain_threshold,
        var_threshold=args.var_threshold,
        k_min=args.k_min,
        k_max=args.k_max,
        alpha_grid=alpha_vals,
        cv_splits=args.cv_splits,
        quiet=args.quiet,
    )
    summary = run(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
