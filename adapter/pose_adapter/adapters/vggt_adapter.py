"""VGGT adapter: run VGGT on extracted frames and convert cameras to schema.

VGGT returns OpenCV camera-from-world extrinsics ([T, 3, 4] w2c) and per-frame
intrinsics in the resized model image coordinates. This adapter pads w2c to
4x4, inverts to c2w, and rescales intrinsics back to the source resolution.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..geometry import invert_se3_batch, normalize_max_translation, relative_to_first
from ..schema import UnifiedOutput, save


VGGT_REPO = Path(os.environ.get("VGGT_REPO", "/data1/linyuxi/projects/vggt"))
VGGT_CKPT = Path(os.environ.get("VGGT_CKPT", "/data1/linyuxi/torch_cache/hub/checkpoints/model.pt"))
VGGT_MAX_FRAMES = int(os.environ.get("VGGT_MAX_FRAMES", "120"))


def _pad_w2c(extrinsic: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    out[:, :3, :4] = extrinsic.astype(np.float64)
    return out


def _rescale_intrinsics_from_crop_mode(
    K_model: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> tuple[np.ndarray, list[str]]:
    """Undo VGGT load_and_preprocess_images(..., mode='crop') coordinates."""
    target_w = 518.0
    resized_h = round(orig_h * (target_w / orig_w) / 14.0) * 14.0
    crop_y = max((resized_h - 518.0) / 2.0, 0.0)
    sx = orig_w / target_w
    sy = orig_h / resized_h

    K = K_model.astype(np.float64)
    intr = np.empty((K.shape[0], 4), dtype=np.float32)
    intr[:, 0] = K[:, 0, 0] * sx
    intr[:, 1] = K[:, 1, 1] * sy
    intr[:, 2] = K[:, 0, 2] * sx
    intr[:, 3] = (K[:, 1, 2] + crop_y) * sy

    warnings = ["vggt_intrinsics_rescaled_from_model_crop_coordinates"]
    if resized_h > 518.0:
        warnings.append(f"vggt_input_center_cropped_height_from_{resized_h:.0f}_to_518")
    return intr, warnings


def run_vggt(clip_dir: Path, output_dir: Path) -> None:
    clip_dir = Path(clip_dir).resolve()
    output_dir = Path(output_dir).resolve()

    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    wanted_all = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps_all = np.load(clip_dir / "timestamps.npy")
    orig_w, orig_h = meta["original_resolution"]

    frame_files = sorted((clip_dir / "frames").glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"VGGT requires extracted frames under {clip_dir / 'frames'}")
    if len(frame_files) != len(wanted_all):
        raise RuntimeError(f"frames count {len(frame_files)} != frame_indices length {len(wanted_all)}")

    warnings: list[str] = []
    status = "ok"
    if len(frame_files) > VGGT_MAX_FRAMES:
        warnings.append(f"vggt truncated {len(frame_files)} frames to {VGGT_MAX_FRAMES}")
        status = "degraded"
        frame_files = frame_files[:VGGT_MAX_FRAMES]
        wanted = wanted_all[:VGGT_MAX_FRAMES]
        timestamps = timestamps_all[:VGGT_MAX_FRAMES]
    else:
        wanted = wanted_all
        timestamps = timestamps_all

    if not VGGT_REPO.exists():
        raise FileNotFoundError(f"VGGT repo not found: {VGGT_REPO}")
    if not VGGT_CKPT.exists():
        raise FileNotFoundError(f"VGGT checkpoint not found: {VGGT_CKPT}")
    if str(VGGT_REPO) not in sys.path:
        sys.path.insert(0, str(VGGT_REPO))

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT adapter requires CUDA")

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    started = time.time()

    model = VGGT()
    model.load_state_dict(torch.load(str(VGGT_CKPT), map_location="cpu"))
    model.eval().to(device)

    images = load_and_preprocess_images([str(p) for p in frame_files]).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    extrinsic_np = extrinsic.detach().cpu().numpy().squeeze(0)
    intrinsic_np = intrinsic.detach().cpu().numpy().squeeze(0)
    poses_w2c = _pad_w2c(extrinsic_np)
    poses_c2w = invert_se3_batch(poses_w2c)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)
    intrinsics, intr_warnings = _rescale_intrinsics_from_crop_mode(
        intrinsic_np,
        int(orig_w),
        int(orig_h),
    )
    warnings.extend(intr_warnings)

    quality = {
        "method": "vggt",
        "status": status,
        "pose_convention": "c2w",
        "camera_coordinate": "opencv",
        "camera_model": "pinhole",
        "scale_status": "normalized",
        "scale_rule": "max_translation_norm",
        "scale_factor": scale_factor,
        "total_path_length_before_norm": total_path,
        "resolution_reference": [orig_w, orig_h],
        "intrinsics_shared_per_frame": False,
        "num_frames": int(poses_c2w.shape[0]),
        "frame_source_fps": meta["sampled_fps"],
        "original_fps": meta["original_fps"],
        "t_meaning": "sampled_frames",
        "elapsed_s": time.time() - started,
        "warnings": warnings,
        "vggt_repo": str(VGGT_REPO),
        "vggt_ckpt": str(VGGT_CKPT),
    }

    save(output_dir, UnifiedOutput(
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        poses_w2c=poses_w2c,
        relative_poses=rel_norm,
        scale_factor=scale_factor,
        frame_indices=wanted,
        timestamps=timestamps,
        quality=quality,
    ))
