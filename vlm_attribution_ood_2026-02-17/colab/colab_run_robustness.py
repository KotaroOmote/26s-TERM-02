#!/usr/bin/env python3
"""Run robustness study for axis generality across prompts/splits/VLMs.

This script repeatedly calls `colab_tfds_axis_builder.py` with different:
- prompt templates
- random seeds (split/randomness proxy)
- model/pretrained pairs

Then it compares axis similarity (|cos|) against the first run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_prompt_templates(text: str) -> List[str]:
    # Delimiter for multi-template input in shell-safe form
    # e.g. "a photo of {name}||an image of {name}"
    return [x.strip() for x in text.split("||") if x.strip()]


def parse_model_specs(text: str) -> List[Tuple[str, str]]:
    # Format: model:pretrained,model:pretrained
    out: List[Tuple[str, str]] = []
    for item in parse_csv_list(text):
        if ":" not in item:
            raise ValueError(f"Invalid model spec: {item}. Expected model:pretrained")
        m, p = item.split(":", 1)
        out.append((m.strip(), p.strip()))
    return out


def load_axis_vector(run_dir: Path, which: str) -> pd.Series:
    summary = json.loads((run_dir / "summary_axis_only.json").read_text(encoding="utf-8"))
    load = pd.read_csv(run_dir / "axis_loadings.csv")
    axis_idx = summary["axis_mapping"][f"axis_{which}_index"]
    axis_name = f"axis_{axis_idx}"
    row = load[load["axis"] == axis_name].iloc[0]
    feat_cols = [c for c in load.columns if c not in {"axis", "nnz_abs_gt_1e-6"}]
    return row[feat_cols].astype(float)


def abs_cosine(a: pd.Series, b: pd.Series) -> float:
    common = sorted(set(a.index).intersection(set(b.index)))
    if len(common) == 0:
        return float("nan")
    va = a[common].to_numpy(dtype=float)
    vb = b[common].to_numpy(dtype=float)
    den = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
    return float(abs(np.dot(va, vb) / den))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=str, default="/content")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--prompt-templates", type=str, default="a photo of {name}")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--models", type=str, default="ViT-B-32:laion2b_s34b_b79k")
    p.add_argument("--n-id", type=int, default=10000)
    p.add_argument("--n-ood-cifar", type=int, default=5000)
    p.add_argument("--n-ood-imagenetr", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--alpha-grid", type=str, default="0.5,1.0,2.0,4.0,8.0")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    project_dir = Path(args.project_dir)
    script = project_dir / "colab_tfds_axis_builder.py"
    if not script.exists():
        raise FileNotFoundError(f"Builder script not found: {script}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prompt_templates = parse_prompt_templates(args.prompt_templates)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    model_specs = parse_model_specs(args.models)

    run_rows: List[Dict[str, object]] = []
    run_dirs: List[Path] = []
    run_id = 0
    for model_name, pretrained in model_specs:
        for prompt in prompt_templates:
            for seed in seeds:
                rdir = out_root / f"run_{run_id:03d}"
                cmd = [
                    sys.executable,
                    str(script),
                    "--seed",
                    str(seed),
                    "--n-id",
                    str(args.n_id),
                    "--n-ood-cifar",
                    str(args.n_ood_cifar),
                    "--n-ood-imagenetr",
                    str(args.n_ood_imagenetr),
                    "--batch-size",
                    str(args.batch_size),
                    "--k-max",
                    str(args.k_max),
                    "--alpha-grid",
                    args.alpha_grid,
                    "--model-name",
                    model_name,
                    "--pretrained",
                    pretrained,
                    "--prompt-template",
                    prompt,
                    "--output-dir",
                    str(rdir),
                ]
                if args.quiet:
                    cmd.append("--quiet")
                subprocess.run(cmd, check=True)

                summary = json.loads((rdir / "summary_axis_only.json").read_text(encoding="utf-8"))
                run_rows.append(
                    {
                        "run_id": run_id,
                        "output_dir": str(rdir),
                        "seed": seed,
                        "model_name": model_name,
                        "pretrained": pretrained,
                        "prompt_template": prompt,
                        "k_selected": summary["one_se_selection"]["k_selected"],
                        "alpha_selected": summary["one_se_selection"]["alpha_selected"],
                        "axis_u_index": summary["axis_mapping"]["axis_u_index"],
                        "axis_c_index": summary["axis_mapping"]["axis_c_index"],
                    }
                )
                run_dirs.append(rdir)
                run_id += 1

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(out_root / "robustness_runs.csv", index=False)

    if len(run_dirs) == 0:
        raise RuntimeError("No runs executed.")

    ref_dir = run_dirs[0]
    ref_u = load_axis_vector(ref_dir, "u")
    ref_c = load_axis_vector(ref_dir, "c")

    sim_rows: List[Dict[str, object]] = []
    for ridx, rdir in enumerate(run_dirs):
        cur_u = load_axis_vector(rdir, "u")
        cur_c = load_axis_vector(rdir, "c")
        sim_rows.append(
            {
                "run_id": ridx,
                "ref_run_id": 0,
                "abs_cos_zu": abs_cosine(ref_u, cur_u),
                "abs_cos_zc": abs_cosine(ref_c, cur_c),
            }
        )
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(out_root / "robustness_axis_similarity.csv", index=False)

    summary = {
        "n_runs": int(len(run_dirs)),
        "reference_run_id": 0,
        "files": {
            "runs": str(out_root / "robustness_runs.csv"),
            "axis_similarity": str(out_root / "robustness_axis_similarity.csv"),
        },
    }
    with (out_root / "summary_robustness.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

