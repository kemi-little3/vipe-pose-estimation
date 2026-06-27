"""DUSt3R adapter: fully programmatic — load images, run pairwise inference,
global-align, extract scene poses/focals, convert to the unified schema.

Must be run from within the `dust3r` conda env with the checkpoint available
(path via $DUST3R_CKPT env var, or the default below).
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


# DUSt3R repo to import from. Override with DUST3R_REPO env var.
# Default points to colleague linyuxi's copy because that's where the working env expects it.
DUST3R_REPO = Path(
    os.environ.get("DUST3R_REPO", "/data1/linyuxi/projects/dust3r")
)

# Default checkpoint path. Override with DUST3R_CKPT env var.
DEFAULT_CKPT = "/data1/linyuxi/projects/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# A100-80GB sanity cap with scene_graph='swin-3'. Above this, we truncate + warn.
MAX_SAFE_FRAMES = int(os.environ.get("DUST3R_MAX_FRAMES", "120"))

DUST3R_LONG_SIDE = 512


def run_dust3r(
    clip_dir: Path,
    output_dir: Path,
    ckpt_path: str | None = None,
    scene_graph: str = "swin-3",
    niter: int = 300,
    lr: float = 0.01,
    schedule: str = "linear",
) -> None:
    clip_dir = Path(clip_dir).resolve()
    output_dir = Path(output_dir).resolve()

    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    wanted_all = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps_all = np.load(clip_dir / "timestamps.npy")
    orig_w, orig_h = meta["original_resolution"]

    frame_files = sorted((clip_dir / "frames").glob("*.png"))
    T_total = len(frame_files)

    warnings: list[str] = ["pp_simplified_to_image_center"]
    if T_total > MAX_SAFE_FRAMES:
        warnings.append(
            f"truncated {T_total} frames to {MAX_SAFE_FRAMES} (A100-80GB cap with swin-3)"
        )
        frame_files = frame_files[:MAX_SAFE_FRAMES]
        wanted = wanted_all[:MAX_SAFE_FRAMES]
        timestamps = timestamps_all[:MAX_SAFE_FRAMES]
        status = "degraded"
    else:
        wanted = wanted_all
        timestamps = timestamps_all
        status = "ok"

    T = len(frame_files)

    if str(DUST3R_REPO) not in sys.path:
        sys.path.insert(0, str(DUST3R_REPO))

    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import load_model
    from dust3r.utils.image import load_images

    ckpt = ckpt_path or os.environ.get("DUST3R_CKPT", DEFAULT_CKPT)
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"DUSt3R checkpoint not found at {ckpt}. Set DUST3R_CKPT env var or pass ckpt_path."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    started = time.time()

    model = load_model(ckpt, device=device, verbose=False)
    imgs = load_images(
        [str(p) for p in frame_files], size=DUST3R_LONG_SIDE, verbose=False
    )
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=1, verbose=False)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if T >= 3
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode, verbose=False)

    alignment_loss = None
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )
        alignment_loss = float(loss)

    poses_c2w = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
    focals = scene.get_focals().detach().cpu().numpy().reshape(-1).astype(np.float64)

    assert poses_c2w.shape == (T, 4, 4), f"get_im_poses shape {poses_c2w.shape} != ({T},4,4)"
    assert focals.shape == (T,), f"get_focals shape {focals.shape} != ({T},)"

    # DUSt3R focal lives in the resized-image coordinate system (long side = 512).
    # Rescale back to the original resolution.
    long_orig = float(max(orig_w, orig_h))
    f_orig = focals * (long_orig / DUST3R_LONG_SIDE)
    cx = orig_w / 2.0
    cy = orig_h / 2.0

    intrinsics = np.stack(
        [f_orig, f_orig, np.full(T, cx), np.full(T, cy)], axis=1
    ).astype(np.float32)

    poses_w2c = invert_se3_batch(poses_c2w)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)

    quality = {
        "method": "dust3r",
        "status": status,
        "pose_convention": "c2w",
        "camera_coordinate": "opencv",
        "camera_model": "pinhole",
        "scale_status": "normalized",
        "scale_rule": "max_translation_norm",
        "scale_factor": scale_factor,
        "total_path_length_before_norm": total_path,
        "resolution_reference": [orig_w, orig_h],
        "intrinsics_shared_per_frame": False,  # focals are per-frame even though pp is shared
        "num_frames": int(T),
        "frame_source_fps": meta["sampled_fps"],
        "original_fps": meta["original_fps"],
        "t_meaning": "sampled_frames",
        "elapsed_s": time.time() - started,
        "warnings": warnings,
        "dust3r_scene_graph": scene_graph,
        "dust3r_niter": niter,
        "dust3r_lr": lr,
        "dust3r_schedule": schedule,
        "dust3r_alignment_loss": alignment_loss,
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
