#!/usr/bin/env python3
"""Evaluate unified pose outputs against a per-frame GT CSV.

The GT CSV is expected to contain one c2w pose per original video frame:
frame_idx,r00,r01,r02,tx,r10,...,tz,fx,fy,cx,cy

Usage:
    python adapter/scripts/evaluate_with_gt.py \
        --clip-out outputs/<clip_id> \
        --clip-in inputs/<clip_id> \
        --gt-csv test_with_gt/<gt>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ADAPTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ADAPTER_ROOT))

from pose_adapter.geometry import invert_se3_batch, relative_to_first
from pose_adapter.schema import load


METHODS = ("vipe", "vggsfm", "dust3r", "vggt", "megasam")
COLORS = {
    "gt": "#111111",
    "vipe": "#1f77b4",
    "vggsfm": "#ff7f0e",
    "dust3r": "#2ca02c",
    "vggt": "#d62728",
    "megasam": "#9467bd",
}
ALIGN_TYPES = ("none", "se3", "sim3")


def read_gt_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"empty GT CSV: {path}")

    max_idx = max(int(r["frame_idx"]) for r in rows)
    poses = np.full((max_idx + 1, 4, 4), np.nan, dtype=np.float64)
    intr = np.full((max_idx + 1, 4), np.nan, dtype=np.float64)
    for r in rows:
        i = int(r["frame_idx"])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = [
            [float(r["r00"]), float(r["r01"]), float(r["r02"])],
            [float(r["r10"]), float(r["r11"]), float(r["r12"])],
            [float(r["r20"]), float(r["r21"]), float(r["r22"])],
        ]
        T[:3, 3] = [float(r["tx"]), float(r["ty"]), float(r["tz"])]
        poses[i] = T
        intr[i] = [float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"])]
    return poses, intr


def project_to_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def estimate_alignment(src: np.ndarray, dst: np.ndarray, allow_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping src points to dst points."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching [N,3] arrays, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError("need at least 3 points for trajectory alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    H = X.T @ Y / len(src)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    if allow_scale:
        var_src = float(np.mean(np.sum(X * X, axis=1)))
        scale = float(S.sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def apply_alignment(poses: np.ndarray, scale: float, R_align: np.ndarray, t_align: np.ndarray) -> np.ndarray:
    out = poses.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", R_align, poses[:, :3, :3])
    out[:, :3, 3] = scale * (poses[:, :3, 3] @ R_align.T) + t_align
    return out


def signed_permutation_matrices() -> list[tuple[str, np.ndarray]]:
    """Return all right-handed signed axis permutations as 3x3 rotations."""
    out: list[tuple[str, np.ndarray]] = [("identity", np.eye(3, dtype=np.float64))]
    axes = ("x", "y", "z")
    seen = {"identity"}
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)):
            R = np.zeros((3, 3), dtype=np.float64)
            parts = []
            for dst_axis, src_axis in enumerate(perm):
                R[dst_axis, src_axis] = signs[dst_axis]
                sign = "" if signs[dst_axis] > 0 else "-"
                parts.append(f"{axes[dst_axis]}={sign}{axes[src_axis]}")
            if np.linalg.det(R) < 0:
                continue
            name = ",".join(parts)
            if name not in seen:
                out.append((name, R))
                seen.add(name)
    return out


def apply_axis_transform(poses: np.ndarray, R_axis: np.ndarray) -> np.ndarray:
    """Map a relative c2w trajectory into another world-axis basis.

    This changes the coordinates of camera centers and orientations by a fixed
    left-multiplied world-basis rotation, without estimating any trajectory fit.
    """
    out = poses.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", R_axis, poses[:, :3, :3])
    out[:, :3, 3] = poses[:, :3, 3] @ R_axis.T
    return out


def evaluate_scale_only(pred: np.ndarray, gt: np.ndarray) -> tuple[dict, np.ndarray]:
    finite = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
    if finite.sum() < 3:
        raise ValueError(f"only {int(finite.sum())} finite paired poses")
    pred_eval = pred[finite]
    gt_eval = gt[finite]
    p = pred_eval[:, :3, 3]
    g = gt_eval[:, :3, 3]
    denom = float(np.sum(p * p))
    raw_scale = float(np.sum(p * g) / denom) if denom > 0 else 1.0
    scale = raw_scale if raw_scale > 0 else 0.0
    pred_scaled_all = pred.copy()
    pred_scaled_all[:, :3, 3] *= scale
    pred_scaled = pred_scaled_all[finite]

    center_err = np.linalg.norm(pred_scaled[:, :3, 3] - gt_eval[:, :3, 3], axis=1)
    R_err = np.einsum("nij,njk->nik", np.transpose(pred_scaled[:, :3, :3], (0, 2, 1)), gt_eval[:, :3, :3])
    rot_err = rotation_angle_deg(R_err)
    pred_delta = np.einsum("nij,njk->nik", invert_se3_batch(pred_scaled[:-1]), pred_scaled[1:])
    gt_delta = np.einsum("nij,njk->nik", invert_se3_batch(gt_eval[:-1]), gt_eval[1:])
    rpe_trans = np.linalg.norm(pred_delta[:, :3, 3] - gt_delta[:, :3, 3], axis=1)
    rpe_R_err = np.einsum("nij,njk->nik", np.transpose(pred_delta[:, :3, :3], (0, 2, 1)), gt_delta[:, :3, :3])
    rpe_rot = rotation_angle_deg(rpe_R_err)
    return {
        "align_type": "axis_scale",
        "num_frames": int(len(pred)),
        "num_valid": int(finite.sum()),
        "scale": scale,
        "raw_scale": raw_scale,
        "ate_rmse": float(np.sqrt(np.mean(center_err ** 2))),
        "ate_mean": float(center_err.mean()),
        "ate_median": float(np.median(center_err)),
        "ate_max": float(center_err.max()),
        "rot_err_deg_mean": float(rot_err.mean()),
        "rot_err_deg_median": float(np.median(rot_err)),
        "rot_err_deg_max": float(rot_err.max()),
        "rpe_trans_mean": float(rpe_trans.mean()) if len(rpe_trans) else 0.0,
        "rpe_trans_max": float(rpe_trans.max()) if len(rpe_trans) else 0.0,
        "rpe_rot_deg_mean": float(rpe_rot.mean()) if len(rpe_rot) else 0.0,
        "rpe_rot_deg_max": float(rpe_rot.max()) if len(rpe_rot) else 0.0,
        "final_trans_err": float(center_err[-1]),
        "final_rot_err_deg": float(rot_err[-1]),
    }, pred_scaled_all


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def make_relative_unscaled(poses_c2w: np.ndarray) -> np.ndarray:
    mask = np.isfinite(poses_c2w).all(axis=(1, 2))
    if not mask.any():
        raise ValueError("no finite poses")
    first = int(np.argmax(mask))
    rel = np.full_like(poses_c2w, np.nan)
    finite = poses_c2w[mask]
    w2c0 = invert_se3_batch(poses_c2w[first:first + 1])[0]
    rel[mask] = np.einsum("ij,njk->nik", w2c0, finite)
    return rel


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, align_type: str) -> tuple[dict, np.ndarray]:
    finite = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
    if finite.sum() < 3:
        raise ValueError(f"only {int(finite.sum())} finite paired poses")

    pred_eval = pred[finite]
    gt_eval = gt[finite]
    if align_type == "none":
        scale = 1.0
        R_align = np.eye(3)
        t_align = np.zeros(3)
    elif align_type == "se3":
        scale, R_align, t_align = estimate_alignment(pred_eval[:, :3, 3], gt_eval[:, :3, 3], allow_scale=False)
    elif align_type == "sim3":
        scale, R_align, t_align = estimate_alignment(pred_eval[:, :3, 3], gt_eval[:, :3, 3], allow_scale=True)
    else:
        raise ValueError(align_type)

    pred_aligned_all = apply_alignment(pred, scale, R_align, t_align)
    pred_aligned = pred_aligned_all[finite]

    center_err = np.linalg.norm(pred_aligned[:, :3, 3] - gt_eval[:, :3, 3], axis=1)
    R_err = np.einsum("nij,njk->nik", np.transpose(pred_aligned[:, :3, :3], (0, 2, 1)), gt_eval[:, :3, :3])
    rot_err = rotation_angle_deg(R_err)

    pred_delta = np.einsum("nij,njk->nik", invert_se3_batch(pred_aligned[:-1]), pred_aligned[1:])
    gt_delta = np.einsum("nij,njk->nik", invert_se3_batch(gt_eval[:-1]), gt_eval[1:])
    rpe_trans = np.linalg.norm(pred_delta[:, :3, 3] - gt_delta[:, :3, 3], axis=1)
    rpe_R_err = np.einsum("nij,njk->nik", np.transpose(pred_delta[:, :3, :3], (0, 2, 1)), gt_delta[:, :3, :3])
    rpe_rot = rotation_angle_deg(rpe_R_err)

    metrics = {
        "align_type": align_type,
        "num_frames": int(len(pred)),
        "num_valid": int(finite.sum()),
        "scale": float(scale),
        "ate_rmse": float(np.sqrt(np.mean(center_err ** 2))),
        "ate_mean": float(center_err.mean()),
        "ate_median": float(np.median(center_err)),
        "ate_max": float(center_err.max()),
        "rot_err_deg_mean": float(rot_err.mean()),
        "rot_err_deg_median": float(np.median(rot_err)),
        "rot_err_deg_max": float(rot_err.max()),
        "rpe_trans_mean": float(rpe_trans.mean()) if len(rpe_trans) else 0.0,
        "rpe_trans_max": float(rpe_trans.max()) if len(rpe_trans) else 0.0,
        "rpe_rot_deg_mean": float(rpe_rot.mean()) if len(rpe_rot) else 0.0,
        "rpe_rot_deg_max": float(rpe_rot.max()) if len(rpe_rot) else 0.0,
        "final_trans_err": float(center_err[-1]),
        "final_rot_err_deg": float(rot_err[-1]),
    }
    return metrics, pred_aligned_all


def intrinsics_metrics(pred_intr: np.ndarray, gt_intr: np.ndarray) -> dict:
    mask = np.isfinite(pred_intr).all(axis=1) & np.isfinite(gt_intr).all(axis=1)
    if not mask.any():
        return {"num_valid": 0}
    pred = pred_intr[mask].astype(np.float64)
    gt = gt_intr[mask].astype(np.float64)
    diff = pred - gt
    rel_fx = np.abs(diff[:, 0]) / np.maximum(np.abs(gt[:, 0]), 1e-12)
    rel_fy = np.abs(diff[:, 1]) / np.maximum(np.abs(gt[:, 1]), 1e-12)
    return {
        "num_valid": int(mask.sum()),
        "fx_rel_mean": float(rel_fx.mean()),
        "fy_rel_mean": float(rel_fy.mean()),
        "fx_abs_mean": float(np.abs(diff[:, 0]).mean()),
        "fy_abs_mean": float(np.abs(diff[:, 1]).mean()),
        "cx_abs_mean": float(np.abs(diff[:, 2]).mean()),
        "cy_abs_mean": float(np.abs(diff[:, 3]).mean()),
    }


def write_metrics_csv(path: Path, rows: list[dict], extra_keys: tuple[str, ...] = ()) -> None:
    keys = [
        *extra_keys,
        "method", "align_type", "num_frames", "num_valid", "scale",
        "ate_rmse", "ate_mean", "ate_median", "ate_max",
        "rot_err_deg_mean", "rot_err_deg_median", "rot_err_deg_max",
        "rpe_trans_mean", "rpe_trans_max", "rpe_rot_deg_mean", "rpe_rot_deg_max",
        "final_trans_err", "final_rot_err_deg",
        "fx_rel_mean", "fy_rel_mean", "fx_abs_mean", "fy_abs_mean", "cx_abs_mean", "cy_abs_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def plot_gt_comparison(out_png: Path, gt: np.ndarray, aligned: dict[str, np.ndarray], title: str) -> None:
    fig = plt.figure(figsize=(16, 5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)

    def draw(name: str, poses: np.ndarray, style: str = "-") -> None:
        t = poses[:, :3, 3]
        mask = np.isfinite(t).all(axis=1)
        t = t[mask]
        if len(t) == 0:
            return
        color = COLORS.get(name, None)
        ax3d.plot(t[:, 0], t[:, 1], t[:, 2], style, color=color, linewidth=1.4, markersize=2, label=name)
        ax3d.scatter([t[0, 0]], [t[0, 1]], [t[0, 2]], color=color, marker="s", s=35)
        ax_xy.plot(t[:, 0], t[:, 1], style, color=color, linewidth=1.4, markersize=2, label=name)
        ax_xy.scatter([t[0, 0]], [t[0, 1]], color=color, marker="s", s=35)
        ax_xz.plot(t[:, 0], t[:, 2], style, color=color, linewidth=1.4, markersize=2, label=name)
        ax_xz.scatter([t[0, 0]], [t[0, 2]], color=color, marker="s", s=35)

    draw("gt", gt, "-")
    for method, poses in aligned.items():
        draw(method, poses, "-o")

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("3D")
    ax3d.legend()
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_title("top-down candidate (x,y)")
    ax_xy.axis("equal"); ax_xy.grid(True, alpha=0.3); ax_xy.legend()
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.set_title("side/top candidate (x,z)")
    ax_xz.axis("equal"); ax_xz.grid(True, alpha=0.3); ax_xz.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-out", type=Path, required=True, help="outputs/<clip_id>/")
    parser.add_argument("--clip-in", type=Path, required=True, help="inputs/<clip_id>/")
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="Output prefix. Defaults to <clip-out>/gt_eval")
    parser.add_argument("--plot-align", choices=ALIGN_TYPES, default="sim3")
    parser.add_argument("--axis-search", action="store_true",
                        help="Also enumerate fixed right-handed axis transforms and report scale-only metrics.")
    parser.add_argument("--axis-search-method", choices=METHODS, default="vipe",
                        help="Method used to choose the best axis transform for the axis-search plot.")
    args = parser.parse_args()

    clip_out = args.clip_out
    out_prefix = args.out_prefix or (clip_out / "gt_eval")
    frame_indices = np.load(args.clip_in / "frame_indices.npy").astype(np.int64)

    gt_all, gt_intr_all = read_gt_csv(args.gt_csv)
    if frame_indices.max() >= len(gt_all):
        raise ValueError(f"frame index {frame_indices.max()} exceeds GT rows {len(gt_all)}")
    gt = gt_all[frame_indices]
    gt_intr = gt_intr_all[frame_indices]

    # The provided GT is already relative-to-first c2w, but this keeps the script robust.
    if np.max(np.abs(gt[0] - np.eye(4))) > 1e-6:
        gt = relative_to_first(gt)

    all_rows: list[dict] = []
    axis_rows: list[dict] = []
    json_out: dict = {
        "clip_out": str(clip_out),
        "clip_in": str(args.clip_in),
        "gt_csv": str(args.gt_csv),
        "frame_indices": frame_indices.tolist(),
        "gt_num_frames": int(len(gt)),
        "methods": {},
        "axis_search": {},
    }
    aligned_for_plot: dict[str, np.ndarray] = {}

    for method in METHODS:
        method_dir = clip_out / method
        if not method_dir.exists():
            print(f"SKIP {method}: {method_dir} missing", file=sys.stderr)
            continue
        u = load(method_dir)
        pred = make_relative_unscaled(u.poses_c2w)
        n = min(len(pred), len(gt))
        pred = pred[:n]
        gt_n = gt[:n]
        gt_intr_n = gt_intr[:n]
        pred_intr = u.intrinsics[:n]

        method_json = {"alignments": {}, "intrinsics": intrinsics_metrics(pred_intr, gt_intr_n)}
        for align_type in ALIGN_TYPES:
            metrics, pred_aligned = evaluate_pair(pred, gt_n, align_type)
            row = {"method": method, **metrics, **method_json["intrinsics"]}
            all_rows.append(row)
            method_json["alignments"][align_type] = metrics
            if align_type == args.plot_align:
                aligned_for_plot[method] = pred_aligned
        json_out["methods"][method] = method_json

        if args.axis_search:
            method_axis_rows = []
            for axis_name, R_axis in signed_permutation_matrices():
                pred_axis = apply_axis_transform(pred, R_axis)
                metrics, _ = evaluate_scale_only(pred_axis, gt_n)
                row = {
                    "axis_transform": axis_name,
                    "axis_matrix": R_axis.astype(int).tolist(),
                    "method": method,
                    **metrics,
                    **method_json["intrinsics"],
                }
                axis_rows.append(row)
                method_axis_rows.append(row)
            best = min(method_axis_rows, key=lambda r: (r["ate_rmse"], r["rpe_rot_deg_mean"]))
            json_out["axis_search"][method] = {
                "best_by_ate_rmse": best,
                "all_count": len(method_axis_rows),
            }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    with json_path.open("w") as f:
        json.dump(json_out, f, indent=2)
    write_metrics_csv(csv_path, all_rows)
    plot_gt_comparison(png_path, gt, aligned_for_plot, f"{clip_out.name} GT comparison ({args.plot_align})")

    if args.axis_search:
        axis_json_path = out_prefix.parent / f"{out_prefix.name}_axis_search.json"
        axis_csv_path = out_prefix.parent / f"{out_prefix.name}_axis_search.csv"
        axis_png_path = out_prefix.parent / f"{out_prefix.name}_axis_search.png"
        axis_json = {
            "clip_out": str(clip_out),
            "clip_in": str(args.clip_in),
            "gt_csv": str(args.gt_csv),
            "best_by_method": json_out["axis_search"],
        }
        with axis_json_path.open("w") as f:
            json.dump(axis_json, f, indent=2)
        write_metrics_csv(axis_csv_path, axis_rows, extra_keys=("axis_transform", "axis_matrix"))

        best_rows = {
            m: json_out["axis_search"][m]["best_by_ate_rmse"]
            for m in json_out["axis_search"]
        }
        aligned_axis_for_plot: dict[str, np.ndarray] = {}
        for method, best in best_rows.items():
            method_dir = clip_out / method
            if not method_dir.exists():
                continue
            u = load(method_dir)
            pred = make_relative_unscaled(u.poses_c2w)[:len(gt)]
            R_axis = np.array(best["axis_matrix"], dtype=np.float64)
            pred_axis = apply_axis_transform(pred, R_axis)
            _, pred_axis_scaled = evaluate_scale_only(pred_axis, gt[:len(pred_axis)])
            aligned_axis_for_plot[method] = pred_axis_scaled
        plot_gt_comparison(
            axis_png_path,
            gt,
            aligned_axis_for_plot,
            f"{clip_out.name} GT comparison (best fixed axis + scale)",
        )

    print(f"[evaluate_with_gt] wrote {json_path}")
    print(f"[evaluate_with_gt] wrote {csv_path}")
    print(f"[evaluate_with_gt] wrote {png_path}")
    if args.axis_search:
        print(f"[evaluate_with_gt] wrote {axis_json_path}")
        print(f"[evaluate_with_gt] wrote {axis_csv_path}")
        print(f"[evaluate_with_gt] wrote {axis_png_path}")
    print("\nSummary (Sim3):")
    for row in all_rows:
        if row["align_type"] == "sim3":
            print(
                f"  {row['method']:6s} ATE_RMSE={row['ate_rmse']:.4f} "
                f"RPE_rot={row['rpe_rot_deg_mean']:.3f}deg scale={row['scale']:.4f} "
                f"fx_rel={row.get('fx_rel_mean', math.nan):.3f}"
            )
    if args.axis_search:
        print("\nBest fixed-axis + scale:")
        for method, result in json_out["axis_search"].items():
            best = result["best_by_ate_rmse"]
            print(
                f"  {method:6s} axis={best['axis_transform']} "
                f"ATE_RMSE={best['ate_rmse']:.4f} RPE_rot={best['rpe_rot_deg_mean']:.3f}deg "
                f"scale={best['scale']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
