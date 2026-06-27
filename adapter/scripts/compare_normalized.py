#!/usr/bin/env python3
"""Compare ViPE and GT trajectories using relative_to_first + normalize_max_translation.

Pipeline applied to ViPE before comparison:
  1. relative_to_first:        P_rel[t] = inv(P[0]) @ P[t]
  2. axis transform x=x,y=z,z=-y: map ViPE axes to GT axes
  3. normalize_max_translation: divide by max ||t_i||

GT is only relative_to_first + normalize_max_translation (no axis transform needed).
No scale alignment, no global rotation fitted.

Usage:
    python adapter/scripts/compare_normalized.py \
        --out-dir outputs/normalized_comparison
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_ADAPTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ADAPTER_ROOT))

from pose_adapter.geometry import invert_se3_batch, relative_to_first, normalize_max_translation

GT_CSV_ROOT = Path("/data1/hexuming/pose_estimation/pose_demo/pose_demo")
OUTPUTS_ROOT = Path("/data1/hexuming/pose_estimation/outputs")
INPUTS_ROOT = Path("/data1/hexuming/pose_estimation/inputs")

CLIPS = [
    "gt_2026-05-14_10-57-13_full",
    "gt_2026-05-14_10-57-50_full",
    "gt_2026-05-14_10-58-32_full",
    "gt_2026-05-14_11-00-05_full",
    "gt_2026-05-14_11-01-24_full",
    "gt_2026-05-14_11-02-37_full",
    "gt_2026-05-14_11-03-27_full",
    "gt_2026-05-14_11-04-53_full",
    "gt_2026-05-14_11-05-53_full",
    "gt_2026-05-14_11-06-27_full",
]

# ViPE → GT fixed axis transform:  x=x, y=z, z=-y
# GT_y = ViPE_z,  GT_z = -ViPE_y
VIPE_TO_GT_AXIS = np.array([[1,  0,  0],
                             [0,  0,  1],
                             [0, -1,  0]], dtype=np.float64)

GT_COLOR = "#222222"
VIPE_COLOR = "#1f77b4"


def read_gt_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
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


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def apply_axis_transform(poses: np.ndarray, R_axis: np.ndarray) -> np.ndarray:
    """Apply a fixed world-basis rotation to a batch of c2w poses."""
    out = poses.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", R_axis, poses[:, :3, :3])
    out[:, :3, 3] = poses[:, :3, 3] @ R_axis.T
    return out


def normalize_trajectory(poses_c2w: np.ndarray) -> tuple[np.ndarray, float]:
    """relative_to_first then normalize_max_translation. Returns (normalized, scale_factor)."""
    rel = relative_to_first(poses_c2w)
    normed, scale, _ = normalize_max_translation(rel)
    return normed, scale


def compute_metrics(pred_norm: np.ndarray, gt_norm: np.ndarray) -> dict:
    finite = np.isfinite(pred_norm).all(axis=(1, 2)) & np.isfinite(gt_norm).all(axis=(1, 2))
    pred = pred_norm[finite]
    gt = gt_norm[finite]

    trans_err = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    R_err = np.einsum("nij,njk->nik", np.transpose(pred[:, :3, :3], (0, 2, 1)), gt[:, :3, :3])
    rot_err = rotation_angle_deg(R_err)

    pred_delta = np.einsum("nij,njk->nik", invert_se3_batch(pred[:-1]), pred[1:])
    gt_delta = np.einsum("nij,njk->nik", invert_se3_batch(gt[:-1]), gt[1:])
    rpe_t = np.linalg.norm(pred_delta[:, :3, 3] - gt_delta[:, :3, 3], axis=1)
    rpe_R = np.einsum("nij,njk->nik", np.transpose(pred_delta[:, :3, :3], (0, 2, 1)), gt_delta[:, :3, :3])
    rpe_rot = rotation_angle_deg(rpe_R)

    return {
        "n_frames": int(finite.sum()),
        "ate_rmse": float(np.sqrt(np.mean(trans_err ** 2))),
        "ate_mean": float(trans_err.mean()),
        "ate_max": float(trans_err.max()),
        "rot_err_mean_deg": float(rot_err.mean()),
        "rot_err_max_deg": float(rot_err.max()),
        "rpe_trans_mean": float(rpe_t.mean()) if len(rpe_t) else 0.0,
        "rpe_rot_mean_deg": float(rpe_rot.mean()) if len(rpe_rot) else 0.0,
    }


def _timestamp_from_clip_id(clip_id: str) -> str:
    """Extract timestamp like '2026-05-14_10-57-13' from clip_id string."""
    parts = clip_id.replace("gt_", "").replace("_full", "")
    return parts


def load_clip(clip_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load matched (vipe_c2w, gt_c2w) for a clip. Returns None if missing."""
    vipe_dir = OUTPUTS_ROOT / clip_id / "vipe"
    input_dir = INPUTS_ROOT / clip_id

    if not (vipe_dir / "poses_c2w.npy").exists():
        print(f"  [SKIP] {clip_id}: no ViPE poses", file=sys.stderr)
        return None
    if not (input_dir / "frame_indices.npy").exists():
        print(f"  [SKIP] {clip_id}: no frame_indices.npy", file=sys.stderr)
        return None

    ts = _timestamp_from_clip_id(clip_id)
    gt_csv = GT_CSV_ROOT / f"telemetry_vehicle_nav_{ts}_aligned_processed.csv"
    if not gt_csv.exists():
        print(f"  [SKIP] {clip_id}: no GT CSV at {gt_csv}", file=sys.stderr)
        return None

    frame_indices = np.load(input_dir / "frame_indices.npy").astype(np.int64)
    vipe_c2w = np.load(vipe_dir / "poses_c2w.npy").astype(np.float64)  # [T, 4, 4]

    gt_all, _ = read_gt_csv(gt_csv)
    gt_c2w = gt_all[frame_indices]  # [T, 4, 4]

    n = min(len(vipe_c2w), len(gt_c2w))
    return vipe_c2w[:n], gt_c2w[:n]


def plot_summary(results: list[dict], out_png: Path) -> None:
    """2-row × 5-col grid, each cell shows the XY trajectory of one clip."""
    n = len(results)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).reshape(nrows, ncols)

    for i, res in enumerate(results):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        gt_t = res["gt_norm"][:, :3, 3]
        vipe_t = res["vipe_norm"][:, :3, 3]
        gm = np.isfinite(gt_t).all(axis=1)
        vm = np.isfinite(vipe_t).all(axis=1)

        ax.plot(gt_t[gm, 0], gt_t[gm, 1], "-", color=GT_COLOR, lw=1.5, label="GT")
        ax.scatter(gt_t[gm, 0][:1], gt_t[gm, 1][:1], color=GT_COLOR, marker="s", s=40, zorder=5)
        ax.plot(vipe_t[vm, 0], vipe_t[vm, 1], "-", color=VIPE_COLOR, lw=1.5, label="ViPE")
        ax.scatter(vipe_t[vm, 0][:1], vipe_t[vm, 1][:1], color=VIPE_COLOR, marker="s", s=40, zorder=5)

        m = res["metrics"]
        label = res["clip_id"].replace("gt_", "").replace("_full", "")
        ax.set_title(f"{label}\nATE={m['ate_rmse']:.3f}  rot={m['rot_err_mean_deg']:.1f}°", fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.axis("equal"); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("ViPE vs GT  (relative_to_first + normalize_max_translation, no alignment)\nXY projection", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_per_clip(res: dict, out_png: Path) -> None:
    """3D + XY + XZ + YZ views for a single clip."""
    fig = plt.figure(figsize=(18, 5))
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 4, 2)
    ax_xz = fig.add_subplot(1, 4, 3)
    ax_yz = fig.add_subplot(1, 4, 4)

    gt_t = res["gt_norm"][:, :3, 3]
    vipe_t = res["vipe_norm"][:, :3, 3]
    gm = np.isfinite(gt_t).all(axis=1)
    vm = np.isfinite(vipe_t).all(axis=1)

    for ax3d_or_2d, xi, yi, xlabel, ylabel, title in [
        (ax_xy, 0, 1, "x", "y", "XY (top-down)"),
        (ax_xz, 0, 2, "x", "z", "XZ (side)"),
        (ax_yz, 1, 2, "y", "z", "YZ (front)"),
    ]:
        ax3d_or_2d.plot(gt_t[gm, xi], gt_t[gm, yi], "-", color=GT_COLOR, lw=1.5, label="GT")
        ax3d_or_2d.scatter(gt_t[gm, xi][:1], gt_t[gm, yi][:1], color=GT_COLOR, marker="s", s=50, zorder=5)
        ax3d_or_2d.plot(vipe_t[vm, xi], vipe_t[vm, yi], "-", color=VIPE_COLOR, lw=1.5, label="ViPE")
        ax3d_or_2d.scatter(vipe_t[vm, xi][:1], vipe_t[vm, yi][:1], color=VIPE_COLOR, marker="s", s=50, zorder=5)
        ax3d_or_2d.set_xlabel(xlabel); ax3d_or_2d.set_ylabel(ylabel)
        ax3d_or_2d.set_title(title); ax3d_or_2d.axis("equal")
        ax3d_or_2d.grid(True, alpha=0.3); ax3d_or_2d.legend(fontsize=8)

    ax3d.plot(gt_t[gm, 0], gt_t[gm, 1], gt_t[gm, 2], "-", color=GT_COLOR, lw=1.5, label="GT")
    ax3d.scatter([gt_t[gm, 0][0]], [gt_t[gm, 1][0]], [gt_t[gm, 2][0]], color=GT_COLOR, marker="s", s=50)
    ax3d.plot(vipe_t[vm, 0], vipe_t[vm, 1], vipe_t[vm, 2], "-", color=VIPE_COLOR, lw=1.5, label="ViPE")
    ax3d.scatter([vipe_t[vm, 0][0]], [vipe_t[vm, 1][0]], [vipe_t[vm, 2][0]], color=VIPE_COLOR, marker="s", s=50)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("3D"); ax3d.legend(fontsize=8)

    m = res["metrics"]
    clip_label = res["clip_id"].replace("gt_", "").replace("_full", "")
    fig.suptitle(
        f"{clip_label}  |  frames={m['n_frames']}  "
        f"ATE_RMSE={m['ate_rmse']:.4f}  rot_mean={m['rot_err_mean_deg']:.2f}°  "
        f"RPE_t={m['rpe_trans_mean']:.4f}  RPE_rot={m['rpe_rot_mean_deg']:.2f}°\n"
        f"(ViPE scale_factor={res['vipe_scale']:.4f}, GT scale_factor={res['gt_scale']:.4f})",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/data1/hexuming/pose_estimation/outputs/normalized_comparison"))
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Subset of clip IDs. Default: all 10 full GT clips.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = args.clips or CLIPS

    results = []
    for clip_id in clips:
        print(f"Loading {clip_id} ...")
        pair = load_clip(clip_id)
        if pair is None:
            continue
        vipe_c2w, gt_c2w = pair

        vipe_rel = relative_to_first(vipe_c2w)
        vipe_axis = apply_axis_transform(vipe_rel, VIPE_TO_GT_AXIS)
        vipe_norm, vipe_scale = normalize_max_translation(vipe_axis)[:2]

        gt_norm, gt_scale = normalize_trajectory(gt_c2w)

        metrics = compute_metrics(vipe_norm, gt_norm)
        results.append({
            "clip_id": clip_id,
            "vipe_norm": vipe_norm,
            "gt_norm": gt_norm,
            "vipe_scale": vipe_scale,
            "gt_scale": gt_scale,
            "metrics": metrics,
        })
        print(f"  frames={metrics['n_frames']}  ATE_RMSE={metrics['ate_rmse']:.4f}"
              f"  rot_mean={metrics['rot_err_mean_deg']:.2f}°"
              f"  RPE_t={metrics['rpe_trans_mean']:.4f}  RPE_rot={metrics['rpe_rot_mean_deg']:.2f}°")

    if not results:
        print("No clips loaded.", file=sys.stderr)
        return

    # per-clip detailed figures
    for res in results:
        clip_label = res["clip_id"].replace("gt_", "").replace("_full", "")
        per_png = args.out_dir / f"{clip_label}.png"
        plot_per_clip(res, per_png)

    # summary overview
    summary_png = args.out_dir / "summary.png"
    plot_summary(results, summary_png)

    # metrics table to stdout
    print("\n=== Summary (normalized, no alignment) ===")
    print(f"{'clip':30s}  {'frames':>6}  {'ATE↓':>8}  {'rot_mean↓':>10}  {'RPE_t↓':>8}  {'RPE_rot↓':>9}  "
          f"{'ViPE_scale':>10}  {'GT_scale':>9}")
    print("-" * 110)
    for res in results:
        m = res["metrics"]
        label = res["clip_id"].replace("gt_", "").replace("_full", "")
        print(f"{label:30s}  {m['n_frames']:>6}  {m['ate_rmse']:>8.4f}  "
              f"{m['rot_err_mean_deg']:>10.2f}  {m['rpe_trans_mean']:>8.4f}  "
              f"{m['rpe_rot_mean_deg']:>9.2f}  {res['vipe_scale']:>10.4f}  {res['gt_scale']:>9.4f}")


if __name__ == "__main__":
    main()
