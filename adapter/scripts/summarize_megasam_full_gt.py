#!/usr/bin/env python3
"""Summarize full-frame MegaSaM-vs-GT evaluations into CSV and Markdown.

Mirrors summarize_vipe_full_gt.py but reads the "megasam" key out of each
gt_eval.json. Run this after scripts/run_megasam_full_gt_batch.sh finishes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD = "megasam"


def load_summary(clip_out: Path) -> dict | None:
    eval_json = clip_out / "gt_eval.json"
    if not eval_json.exists():
        return None
    data = json.loads(eval_json.read_text())
    method_data = data.get("methods", {}).get(METHOD)
    if not method_data:
        return None
    sim3 = method_data["alignments"]["sim3"]
    intr = method_data.get("intrinsics", {})

    axis_json = clip_out / "gt_eval_axis_search.json"
    best_axis = {}
    if axis_json.exists():
        axis_data = json.loads(axis_json.read_text())
        best_axis = (
            axis_data.get("best_by_method", {})
            .get(METHOD, {})
            .get("best_by_ate_rmse", {})
        )

    return {
        "clip_id": clip_out.name,
        "num_frames": sim3.get("num_frames"),
        "ate_rmse_sim3": sim3.get("ate_rmse"),
        "ate_mean_sim3": sim3.get("ate_mean"),
        "rpe_rot_deg_mean": sim3.get("rpe_rot_deg_mean"),
        "rpe_trans_mean": sim3.get("rpe_trans_mean"),
        "scale_sim3": sim3.get("scale"),
        "fx_rel_mean": intr.get("fx_rel_mean"),
        "fy_rel_mean": intr.get("fy_rel_mean"),
        "axis_transform": best_axis.get("axis_transform"),
        "axis_ate_rmse": best_axis.get("ate_rmse"),
        "axis_scale": best_axis.get("scale"),
        "axis_rpe_rot_deg_mean": best_axis.get("rpe_rot_deg_mean"),
    }


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        f"# {METHOD.capitalize()} Full-Frame GT Summary",
        "",
        "| clip_id | frames | ATE Sim3 | RPE rot deg | scale | fx rel | best fixed axis | axis ATE | axis scale |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["clip_id"]),
                    fmt(r["num_frames"], 0),
                    fmt(r["ate_rmse_sim3"]),
                    fmt(r["rpe_rot_deg_mean"]),
                    fmt(r["scale_sim3"]),
                    fmt(r["fx_rel_mean"]),
                    str(r.get("axis_transform") or ""),
                    fmt(r.get("axis_ate_rmse")),
                    fmt(r.get("axis_scale")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--pattern", default="gt_2026-05-14_*_full")
    parser.add_argument("--csv-out", type=Path, default=Path(f"outputs/{METHOD}_full_gt_summary.csv"))
    parser.add_argument("--md-out", type=Path, default=Path(f"outputs/{METHOD}_full_gt_summary.md"))
    args = parser.parse_args()

    rows = []
    for clip_out in sorted(args.outputs_root.glob(args.pattern)):
        row = load_summary(clip_out)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"no completed {METHOD} evals found under {args.outputs_root}/{args.pattern}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.md_out, rows)

    print(f"[summarize] wrote {args.csv_out}")
    print(f"[summarize] wrote {args.md_out}")
    for r in rows:
        print(
            f"{r['clip_id']}: "
            f"ATE={fmt(r['ate_rmse_sim3'])} "
            f"RPE_rot={fmt(r['rpe_rot_deg_mean'])} "
            f"scale={fmt(r['scale_sim3'])} "
            f"axis={r.get('axis_transform')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
