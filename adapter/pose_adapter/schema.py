"""Unified output schema shared by all three adapters.

See plan file for field definitions. The UnifiedOutput dataclass is the in-memory
representation; save()/load() serialize it to the standard per-method directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import invert_se3_batch


# Required keys in quality.json. Method-specific extras are allowed.
REQUIRED_QUALITY_KEYS = {
    "method",
    "status",
    "pose_convention",
    "camera_coordinate",
    "camera_model",
    "scale_status",
    "scale_rule",
    "scale_factor",
    "total_path_length_before_norm",
    "resolution_reference",
    "intrinsics_shared_per_frame",
    "num_frames",
    "frame_source_fps",
    "original_fps",
    "t_meaning",
    "elapsed_s",
    "warnings",
}


@dataclass
class UnifiedOutput:
    intrinsics: np.ndarray         # [T, 4] float32  fx, fy, cx, cy at original resolution
    poses_c2w: np.ndarray          # [T, 4, 4] float64  OpenCV c2w, world = first frame
    poses_w2c: np.ndarray          # [T, 4, 4] float64
    relative_poses: np.ndarray     # [T, 4, 4] float64  first frame = I4, translations normalized
    scale_factor: float            # divisor used in normalization
    frame_indices: np.ndarray      # [T] int32  which mp4 frame each sample corresponds to
    timestamps: np.ndarray         # [T] float64  seconds
    quality: dict = field(default_factory=dict)


def save(out_dir: Path, u: UnifiedOutput) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "intrinsics.npy", u.intrinsics.astype(np.float32))
    np.save(out_dir / "poses_c2w.npy", u.poses_c2w.astype(np.float64))
    np.save(out_dir / "poses_w2c.npy", u.poses_w2c.astype(np.float64))
    np.save(out_dir / "relative_poses.npy", u.relative_poses.astype(np.float64))
    np.save(out_dir / "scale_factor.npy", np.float64(u.scale_factor))
    np.save(out_dir / "frame_indices.npy", u.frame_indices.astype(np.int32))
    np.save(out_dir / "timestamps.npy", u.timestamps.astype(np.float64))

    missing = REQUIRED_QUALITY_KEYS - set(u.quality.keys())
    if missing:
        raise ValueError(f"quality.json missing required keys: {sorted(missing)}")
    with (out_dir / "quality.json").open("w") as f:
        json.dump(u.quality, f, indent=2)


def load(out_dir: Path) -> UnifiedOutput:
    out_dir = Path(out_dir)
    with (out_dir / "quality.json").open() as f:
        quality = json.load(f)
    return UnifiedOutput(
        intrinsics=np.load(out_dir / "intrinsics.npy"),
        poses_c2w=np.load(out_dir / "poses_c2w.npy"),
        poses_w2c=np.load(out_dir / "poses_w2c.npy"),
        relative_poses=np.load(out_dir / "relative_poses.npy"),
        scale_factor=float(np.load(out_dir / "scale_factor.npy")),
        frame_indices=np.load(out_dir / "frame_indices.npy"),
        timestamps=np.load(out_dir / "timestamps.npy"),
        quality=quality,
    )


def validate(u: UnifiedOutput, orig_wh: tuple[int, int]) -> list[str]:
    """Return a list of warning strings; empty list means fully consistent."""
    warnings: list[str] = []
    T = u.intrinsics.shape[0]
    W, H = orig_wh

    if u.intrinsics.shape != (T, 4):
        warnings.append(f"intrinsics shape {u.intrinsics.shape} != ({T}, 4)")
    for name, arr in [
        ("poses_c2w", u.poses_c2w),
        ("poses_w2c", u.poses_w2c),
        ("relative_poses", u.relative_poses),
    ]:
        if arr.shape != (T, 4, 4):
            warnings.append(f"{name} shape {arr.shape} != ({T}, 4, 4)")

    if u.frame_indices.shape != (T,):
        warnings.append(f"frame_indices shape {u.frame_indices.shape} != ({T},)")
    if u.timestamps.shape != (T,):
        warnings.append(f"timestamps shape {u.timestamps.shape} != ({T},)")

    # c2w <-> w2c consistency
    if u.poses_c2w.shape == u.poses_w2c.shape == (T, 4, 4):
        recon = invert_se3_batch(u.poses_c2w)
        err = np.nanmax(np.abs(recon - u.poses_w2c))
        if err > 1e-4:
            warnings.append(f"poses_w2c != inv(poses_c2w): max abs err {err:.2e}")

    # first frame identity in relative
    if u.relative_poses.shape == (T, 4, 4) and T > 0:
        err = np.max(np.abs(u.relative_poses[0] - np.eye(4)))
        if err > 1e-4:
            warnings.append(f"relative_poses[0] != I4: max abs err {err:.2e}")

    # max-norm == 1
    if u.relative_poses.shape == (T, 4, 4) and T > 0:
        max_norm = float(np.linalg.norm(u.relative_poses[:, :3, 3], axis=1).max())
        if abs(max_norm - 1.0) > 1e-3 and max_norm > 0:
            warnings.append(f"max relative translation norm {max_norm:.4f} != 1.0")

    # intrinsics sanity
    fx = u.intrinsics[:, 0]
    fy = u.intrinsics[:, 1]
    cx = u.intrinsics[:, 2]
    cy = u.intrinsics[:, 3]
    if np.any((fx < 0.3 * W) | (fx > 3 * W)):
        warnings.append(f"fx out of [0.3W, 3W]: min {fx.min():.1f} max {fx.max():.1f} (W={W})")
    if np.any((fy < 0.3 * W) | (fy > 3 * W)):
        warnings.append(f"fy out of [0.3W, 3W]: min {fy.min():.1f} max {fy.max():.1f}")
    if np.any(np.abs(cx - W / 2) > 0.15 * W):
        warnings.append(f"cx far from W/2: min {cx.min():.1f} max {cx.max():.1f}")
    if np.any(np.abs(cy - H / 2) > 0.15 * H):
        warnings.append(f"cy far from H/2: min {cy.min():.1f} max {cy.max():.1f}")

    return warnings
