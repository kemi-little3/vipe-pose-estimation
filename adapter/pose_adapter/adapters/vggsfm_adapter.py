"""VGGSfM adapter: runs VideoRunner in-process, reads back sparse/ via pycolmap.

Must be executed from within the `vggsfm` conda env (pycolmap + vggsfm installed).
The VGGSfM runner writes K already rescaled to the original image resolution
(runner.py:1009-1040, back_to_original_resolution=True default), so this adapter
does not need to rescale intrinsics itself.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..geometry import invert_se3_batch, normalize_max_translation, relative_to_first
from ..schema import UnifiedOutput, save


# VGGSfM repo to import from. Override with VGGSFM_REPO env var.
# Default points to colleague duanmengxuan's repo because that's where the matching
# conda env was set up (ckpt + deps already in place).
VGGSFM_REPO = Path(
    os.environ.get("VGGSFM_REPO", "/data1/duanmengxuan/projects/kemi/VGGsfm/vggsfm")
)

# Single-window cap: override with VGGSFM_MAX_FRAMES env var.
VGGSFM_SINGLE_WINDOW_CAP = int(os.environ.get("VGGSFM_MAX_FRAMES", "64"))


def _pad_w2c(w2c_34: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = w2c_34
    return out


def _stage_scene_dir(frame_files: list[Path], scene_dir: Path) -> None:
    """Create `scene_dir/images/` symlinking the given frame files."""
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    for src in frame_files:
        dst = scene_dir / "images" / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def run_vggsfm(clip_dir: Path, output_dir: Path) -> None:
    clip_dir = Path(clip_dir).resolve()
    output_dir = Path(output_dir).resolve()

    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    wanted = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps = np.load(clip_dir / "timestamps.npy")
    orig_w, orig_h = meta["original_resolution"]

    frame_files = sorted((clip_dir / "frames").glob("*.png"))
    T_total = len(frame_files)
    assert T_total == len(wanted), f"frames/ count {T_total} != frame_indices length {len(wanted)}"

    # Truncate overly long clips so everything fits in a single init window;
    # this sidesteps the boundary bug where T slightly exceeds init_window_size.
    truncation_warning: str | None = None
    if T_total > VGGSFM_SINGLE_WINDOW_CAP:
        truncation_warning = (
            f"vggsfm truncated {T_total} frames to {VGGSFM_SINGLE_WINDOW_CAP} "
            "(single-window cap to avoid sliding-window BA failure)"
        )
        frame_files = frame_files[:VGGSFM_SINGLE_WINDOW_CAP]
        wanted = wanted[:VGGSFM_SINGLE_WINDOW_CAP]
        timestamps = timestamps[:VGGSFM_SINGLE_WINDOW_CAP]
    T = len(frame_files)

    # Make vggsfm importable from its repo layout.
    if str(VGGSFM_REPO) not in sys.path:
        sys.path.insert(0, str(VGGSFM_REPO))

    # visdom is a top-level `import visdom` in vggsfm modules, so a missing
    # package would crash at import time even though we disable all viz features.
    # Stub it only when the real package is not installed in the active env.
    try:
        import visdom  # noqa: F401
    except ImportError:
        _visdom_stub = type(sys)("visdom")
        _visdom_stub.Visdom = type("Visdom", (), {})
        sys.modules["visdom"] = _visdom_stub

    # poselib is faster but optional. If not available, force pycolmap fallback.
    try:
        import poselib  # noqa: F401
        _use_poselib = True
    except ImportError:
        _use_poselib = False

    import hydra
    import torch
    from omegaconf import OmegaConf

    from vggsfm.datasets.demo_loader import DemoLoader
    from vggsfm.runners.video_runner import VideoRunner
    from vggsfm.utils.utils import seed_all_random_engines
    import pycolmap

    started = time.time()

    with tempfile.TemporaryDirectory(prefix="vggsfm_scene_") as tmpdir:
        scene_dir = Path(tmpdir)
        _stage_scene_dir(frame_files, scene_dir)

        # Compose the video_demo config with our overrides.
        with hydra.initialize_config_dir(
            config_dir=str(VGGSFM_REPO / "cfgs"), version_base=None
        ):
            cfg = hydra.compose(
                config_name="video_demo",
                overrides=[
                    f"SCENE_DIR={scene_dir}",
                    "shared_camera=True",
                    "camera_type=SIMPLE_PINHOLE",
                    "save_to_disk=True",
                    "load_gt=False",
                    # auto_download_ckpt=True would call hf_hub_download, which needs a writable HF cache.
                    # The repo ships ckpt/vggsfm_v2_0_0.bin already; load it directly instead.
                    "auto_download_ckpt=False",
                    # Single-window init: covers the whole clip (T <= VGGSFM_SINGLE_WINDOW_CAP),
                    # avoiding the sliding-window BA failure on tail windows.
                    f"init_window_size={T}",
                    f"use_poselib={_use_poselib}",
                    "viz_visualize=False",
                    "gr_visualize=False",
                    "make_reproj_video=False",
                    "visual_tracks=False",
                    "visual_query_points=False",
                    "visual_dense_point_cloud=False",
                ],
            )
        OmegaConf.set_struct(cfg, False)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        seed_all_random_engines(cfg.seed)

        # Must chdir into the repo so relative auto-download ckpt paths resolve.
        prev_cwd = os.getcwd()
        os.chdir(VGGSFM_REPO)
        try:
            runner = VideoRunner(cfg)
            dataset = DemoLoader(
                SCENE_DIR=str(scene_dir),
                img_size=cfg.img_size,
                normalize_cameras=False,
                load_gt=False,
            )
            seq_name = dataset.sequence_list[0]
            batch, image_paths = dataset.get_data(
                sequence_name=seq_name, return_path=True
            )
            runner.run(
                batch["image"],
                masks=batch["masks"] if batch["masks"] is not None else None,
                original_images=batch["original_images"],
                image_paths=image_paths,
                crop_params=batch["crop_params"] if batch["crop_params"] is not None else None,
                seq_name=seq_name,
                output_dir=str(scene_dir),
                init_window_size=cfg.init_window_size,
                window_size=cfg.window_size,
                joint_BA_interval=cfg.joint_BA_interval,
            )
        finally:
            os.chdir(prev_cwd)

        # Read back the COLMAP reconstruction.
        sparse_dir = scene_dir / "sparse"
        if not (sparse_dir / "cameras.bin").exists():
            alt = sparse_dir / "0"
            if (alt / "cameras.bin").exists():
                sparse_dir = alt
            else:
                raise RuntimeError(f"vggsfm produced no cameras.bin under {scene_dir}/sparse")
        rec = pycolmap.Reconstruction(str(sparse_dir))

        # shared_camera=True -> one camera record, broadcast to T rows.
        if len(rec.cameras) != 1:
            raise RuntimeError(
                f"expected 1 shared camera with shared_camera=True, got {len(rec.cameras)}"
            )
        cam = next(iter(rec.cameras.values()))
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        # Map image name -> w2c. Our filenames are 0000.png, 0001.png, ... matching `wanted` order.
        name_to_w2c: dict[str, np.ndarray] = {}
        for im in rec.images.values():
            name_to_w2c[im.name] = np.asarray(im.cam_from_world.matrix(), dtype=np.float64)

    warnings: list[str] = []
    if truncation_warning is not None:
        warnings.append(truncation_warning)
    poses_w2c = np.zeros((T, 4, 4), dtype=np.float64)
    registered_mask = np.zeros(T, dtype=bool)
    for i, path in enumerate(frame_files):
        w2c_34 = name_to_w2c.get(path.name)
        if w2c_34 is None:
            poses_w2c[i] = np.eye(4)  # placeholder; will override via NaN below
            registered_mask[i] = False
            continue
        poses_w2c[i] = _pad_w2c(w2c_34)
        registered_mask[i] = True

    n_missing = int((~registered_mask).sum())
    if n_missing == T:
        raise RuntimeError("vggsfm failed to register any frame")
    if n_missing > 0:
        warnings.append(f"vggsfm dropped {n_missing} frames; filled with NaN")

    poses_c2w = invert_se3_batch(poses_w2c)
    # Poison rows for unregistered frames so downstream can detect them.
    if n_missing > 0:
        poses_c2w[~registered_mask] = np.nan
        poses_w2c[~registered_mask] = np.nan

    # Relative + normalize must be computed from only the registered frames to avoid NaN propagation.
    # Strategy: compute on registered subset, then scatter back. First frame must be registered;
    # if not, skip to the first registered one and record a warning.
    first_reg = int(np.argmax(registered_mask)) if registered_mask[0] else int(np.argmax(registered_mask))
    if not registered_mask[0]:
        warnings.append(f"frame 0 not registered by vggsfm; using frame {first_reg} as world origin")

    rel = np.full_like(poses_c2w, np.nan)
    reg_c2w = poses_c2w[registered_mask]
    rel_reg = relative_to_first(reg_c2w) if first_reg == 0 else None
    if first_reg != 0:
        # Rebase against first registered frame explicitly.
        w2c0 = invert_se3_batch(reg_c2w[:1])[0]
        rel_reg = np.einsum("ij,njk->nik", w2c0, reg_c2w)
    rel_norm_reg, scale_factor, total_path = normalize_max_translation(rel_reg)
    rel[registered_mask] = rel_norm_reg

    intrinsics = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (T, 1))
    if n_missing > 0:
        intrinsics[~registered_mask] = np.nan

    quality = {
        "method": "vggsfm",
        "status": "ok" if (n_missing == 0 and truncation_warning is None) else "degraded",
        "pose_convention": "c2w",
        "camera_coordinate": "opencv",
        "camera_model": "pinhole",
        "scale_status": "normalized",
        "scale_rule": "max_translation_norm",
        "scale_factor": scale_factor,
        "total_path_length_before_norm": total_path,
        "resolution_reference": [orig_w, orig_h],
        "intrinsics_shared_per_frame": True,
        "num_frames": int(T),
        "num_registered_images": int(registered_mask.sum()),
        "frame_source_fps": meta["sampled_fps"],
        "original_fps": meta["original_fps"],
        "t_meaning": "sampled_frames",
        "elapsed_s": time.time() - started,
        "warnings": warnings,
    }

    save(output_dir, UnifiedOutput(
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        poses_w2c=poses_w2c,
        relative_poses=rel,
        scale_factor=scale_factor,
        frame_indices=wanted,
        timestamps=timestamps,
        quality=quality,
    ))
