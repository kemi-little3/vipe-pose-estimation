#!/usr/bin/env python3
"""Plot aggregate GT, full-clip ViPE, and per-window ViPE trajectories.

This reads completed compare_vipe_window_modes.py outputs and creates one figure
where each video is drawn as one separated segment with three overlaid lines:
GT, clip_once/full_clip, and per_window.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POSE_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = POSE_ROOT / "adapter"
sys.path.insert(0, str(ADAPTER_ROOT))

from pose_adapter.geometry import invert_se3_batch, relative_to_first  # noqa: E402
from pose_adapter.schema import load  # noqa: E402

EVAL_SCRIPT_DIR = POSE_ROOT / "adapter" / "scripts"
sys.path.insert(0, str(EVAL_SCRIPT_DIR))
from evaluate_with_gt import evaluate_pair, read_gt_csv  # noqa: E402

COLORS = {
    "gt": "#111111",
    "clip_once": "#1f77b4",
    "per_window": "#ff7f0e",
}
LABELS = {
    "gt": "GT",
    "clip_once": "full_clip / clip_once ViPE",
    "per_window": "per_window ViPE",
}


def read_completed_experiments(input_root: Path) -> list[dict[str, Any]]:
    aggregate_csv = input_root / "_aggregate" / "per_video_summary.csv"
    if aggregate_csv.exists():
        with aggregate_csv.open() as f:
            rows = list(csv.DictReader(f))
        return rows

    rows = []
    for summary_csv in sorted(input_root.glob("*/summary.csv")):
        exp = summary_csv.parent.name
        stem = exp.split("_ws", 1)[0]
        rows.append({"experiment": exp, "stem": stem})
    return rows


def centers(poses: np.ndarray) -> np.ndarray:
    return poses[:, :3, 3].astype(np.float64)


def finite_points(points: np.ndarray) -> np.ndarray:
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def normalize_segment(points: np.ndarray) -> np.ndarray:
    points = finite_points(points)
    if len(points) == 0:
        return points
    return points - points[0]


def load_gt_for_frames(gt_csv: Path, frame_indices: np.ndarray) -> np.ndarray:
    gt_all, _ = read_gt_csv(gt_csv)
    if int(frame_indices.max()) >= len(gt_all):
        raise ValueError(f"frame index {int(frame_indices.max())} exceeds GT rows {len(gt_all)} for {gt_csv}")
    return gt_all[frame_indices]


def align_pred_to_gt(pred_poses: np.ndarray, gt_poses: np.ndarray) -> np.ndarray:
    pred_rel = relative_to_first(pred_poses.astype(np.float64))
    gt_rel = relative_to_first(gt_poses.astype(np.float64))
    _, aligned = evaluate_pair(pred_rel, gt_rel, "sim3")
    return aligned


def load_full_clip_aligned(exp_dir: Path, stem: str, gt_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    method_dir = exp_dir / "outputs" / "full_clip" / f"{stem}_full" / "vipe"
    full = load(method_dir)
    frame_indices = full.frame_indices.astype(np.int64)
    gt = load_gt_for_frames(gt_csv, frame_indices)
    aligned = align_pred_to_gt(full.poses_c2w, gt)
    return gt, aligned


def load_per_window_aligned(exp_dir: Path, gt_csv: Path) -> tuple[np.ndarray, np.ndarray, int]:
    chunks_gt: list[np.ndarray] = []
    chunks_pred: list[np.ndarray] = []
    window_dirs = sorted((exp_dir / "outputs" / "per_window").glob("window_*"))
    for window_dir in window_dirs:
        method_dir = window_dir / "vipe"
        if not (method_dir / "poses_c2w.npy").exists():
            continue
        out = load(method_dir)
        frame_indices = out.frame_indices.astype(np.int64)
        gt = load_gt_for_frames(gt_csv, frame_indices)
        aligned = align_pred_to_gt(out.poses_c2w, gt)
        chunks_gt.append(gt)
        chunks_pred.append(aligned)
    if not chunks_gt:
        raise ValueError(f"no per_window outputs under {exp_dir}")
    return np.concatenate(chunks_gt, axis=0), np.concatenate(chunks_pred, axis=0), len(chunks_gt)


def make_segments(input_root: Path, gt_root: Path, limit: int) -> list[dict[str, Any]]:
    rows = read_completed_experiments(input_root)
    if limit > 0:
        rows = rows[:limit]
    segments: list[dict[str, Any]] = []
    for row in rows:
        stem = row["stem"]
        exp = row.get("experiment") or next(input_root.glob(f"{stem}_ws*"), None).name
        exp_dir = input_root / exp
        gt_csv = gt_root / f"telemetry_vehicle_nav_{stem}_aligned_processed.csv"
        if not exp_dir.exists():
            raise FileNotFoundError(exp_dir)
        if not gt_csv.exists():
            raise FileNotFoundError(gt_csv)
        gt_full, clip_aligned = load_full_clip_aligned(exp_dir, stem, gt_csv)
        gt_window, per_aligned, window_count = load_per_window_aligned(exp_dir, gt_csv)
        segments.append(
            {
                "stem": stem,
                "experiment": exp,
                "gt": normalize_segment(centers(relative_to_first(gt_full))),
                "clip_once": normalize_segment(centers(clip_aligned)),
                "per_window": normalize_segment(centers(per_aligned)),
                "gt_window": normalize_segment(centers(relative_to_first(gt_window))),
                "frames": int(len(gt_full)),
                "windows": int(window_count),
            }
        )
    return segments


def extent_xy(segments: list[dict[str, Any]]) -> tuple[float, float]:
    xs = []
    ys = []
    for seg in segments:
        for key in ("gt", "clip_once", "per_window"):
            pts = seg[key]
            if len(pts):
                xs.extend(pts[:, 0].tolist())
                ys.extend(pts[:, 1].tolist())
    if not xs or not ys:
        return 1.0, 1.0
    return max(xs) - min(xs), max(ys) - min(ys)


def add_offsets(segments: list[dict[str, Any]], gap_scale: float) -> list[dict[str, Any]]:
    width, _ = extent_xy(segments)
    gap = max(width * gap_scale, 5.0)
    shifted = []
    offset_x = 0.0
    for seg in segments:
        seg_out = dict(seg)
        local_width = 0.0
        for key in ("gt", "clip_once", "per_window"):
            pts = seg[key]
            if len(pts):
                local_width = max(local_width, float(pts[:, 0].max() - pts[:, 0].min()))
                seg_out[key] = pts + np.array([offset_x, 0.0, 0.0])
        shifted.append(seg_out)
        offset_x += max(local_width, 1.0) + gap
    return shifted


def draw_trajectory(ax, pts: np.ndarray, *, color: str, label: str | None, linewidth: float, alpha: float, style: str = "-") -> None:
    pts = finite_points(pts)
    if len(pts) == 0:
        return
    ax.plot(pts[:, 0], pts[:, 1], style, color=color, linewidth=linewidth, alpha=alpha, label=label)
    ax.scatter([pts[0, 0]], [pts[0, 1]], color=color, marker="s", s=24, alpha=alpha)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], color=color, marker="o", s=20, alpha=alpha)


def plot(segments: list[dict[str, Any]], out: Path, title: str) -> None:
    shifted = add_offsets(segments, gap_scale=0.35)
    fig, ax = plt.subplots(figsize=(18, 8))
    used_labels = set()
    for i, seg in enumerate(shifted, start=1):
        for key, lw, alpha, style in (
            ("gt", 2.4, 0.95, "-"),
            ("clip_once", 1.8, 0.88, "-"),
            ("per_window", 1.4, 0.88, "--"),
        ):
            label = LABELS[key] if key not in used_labels else None
            draw_trajectory(ax, seg[key], color=COLORS[key], label=label, linewidth=lw, alpha=alpha, style=style)
            used_labels.add(key)
        gt_pts = finite_points(seg["gt"])
        if len(gt_pts):
            ax.text(gt_pts[0, 0], gt_pts[0, 1], f"{i}. {seg['stem']}\n{seg['frames']}f / {seg['windows']}w", fontsize=8, va="bottom")
    ax.set_title(title)
    ax.set_xlabel("x position, each video relative to its first frame; videos separated horizontally")
    ax.set_ylabel("y position")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(path: Path, segments: list[dict[str, Any]], out: Path) -> None:
    data = {
        "output_png": str(out),
        "output_pdf": str(out.with_suffix(".pdf")),
        "segments": [
            {
                "stem": seg["stem"],
                "experiment": seg["experiment"],
                "frames": seg["frames"],
                "windows": seg["windows"],
                "gt_points": int(len(seg["gt"])),
                "clip_once_points": int(len(seg["clip_once"])),
                "per_window_points": int(len(seg["per_window"])),
            }
            for seg in segments
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=POSE_ROOT / "outputs" / "cmp_original_all")
    parser.add_argument("--gt-root", type=Path, default=POSE_ROOT / "pose_demo" / "data")
    parser.add_argument("--out", type=Path, default=POSE_ROOT / "outputs" / "cmp_original_all" / "_aggregate" / "gt_full_clip_per_window_trajectory_5videos.png")
    parser.add_argument("--limit", type=int, default=5, help="Number of completed videos to plot; 0 means all.")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    gt_root = args.gt_root.resolve()
    out = args.out.resolve()
    segments = make_segments(input_root, gt_root, args.limit)
    if not segments:
        raise SystemExit(f"no completed summaries found under {input_root}")
    plot(segments, out, f"GT vs full_clip/clip_once vs per_window trajectories ({len(segments)} videos, Sim3 aligned)")
    write_summary(out.with_suffix(".json"), segments, out)
    print(f"[trajectory] wrote {out}")
    print(f"[trajectory] wrote {out.with_suffix('.pdf')}")
    print(f"[trajectory] wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
