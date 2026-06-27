"""Mega-SaM adapter: run the colleague wrapper and convert *_droid.npz output.

The Mega-SaM wrapper writes outputs/<scene>_droid.npz containing cam_c2w and a
single intrinsic matrix in the resized tracking image coordinates.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..geometry import invert_se3_batch, normalize_max_translation, relative_to_first
from ..schema import UnifiedOutput, save


MEGASAM_REPO = Path(os.environ.get("MEGASAM_REPO", "/data1/duanmengxuan/projects/kemi/MegaSam/mega-sam"))
MEGASAM_PYTHON = os.environ.get("MEGASAM_PYTHON", "/data1/duanmengxuan/envs/conda_envs/mega_sam/bin/python")
MEGASAM_GPU_ID = os.environ.get("MEGASAM_GPU_ID", "1")
MEGASAM_WORK_ROOT = Path(os.environ.get("MEGASAM_WORK_ROOT", "/tmp/megasam_adapter_runs"))


def _scene_name(clip_dir: Path) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in clip_dir.name)
    return f"adapter_{safe}"


def _intrinsics_from_megasam(K: np.ndarray, result_wh: tuple[int, int], orig_wh: tuple[int, int]) -> np.ndarray:
    result_w, result_h = result_wh
    orig_w, orig_h = orig_wh
    sx = float(orig_w) / float(result_w)
    sy = float(orig_h) / float(result_h)
    return np.array(
        [K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy],
        dtype=np.float32,
    )


def _symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def _prepare_workdir(scene: str) -> Path:
    """Create a writable Mega-SaM root with code/checkpoint symlinks."""
    root = MEGASAM_WORK_ROOT / scene
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for name in [
        "base",
        "camera_tracking_scripts",
        "checkpoints",
        "cvd_opt",
        "mono_depth_scripts",
        "tools",
        "colmap_read_model.py",
        "megasam",
    ]:
        _symlink_force(MEGASAM_REPO / name, root / name)

    depth_root = root / "Depth-Anything"
    depth_root.mkdir(parents=True, exist_ok=True)
    for name in ["checkpoints", "depth_anything", "run_videos.py"]:
        _symlink_force(MEGASAM_REPO / "Depth-Anything" / name, depth_root / name)
    (depth_root / "video_visualization").mkdir(parents=True, exist_ok=True)

    uni_root = root / "UniDepth"
    uni_root.mkdir(parents=True, exist_ok=True)
    for name in ["assets", "configs", "hubconf.py", "pyproject.toml", "scripts", "unidepth"]:
        _symlink_force(MEGASAM_REPO / "UniDepth" / name, uni_root / name)
    (uni_root / "outputs").mkdir(parents=True, exist_ok=True)

    for name in ["data", "outputs", "reconstructions"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    _symlink_force(Path("/data1/duanmengxuan/cache/torch/hub"), root / "torchhub")

    return root


def run_megasam(clip_dir: Path, output_dir: Path) -> None:
    clip_dir = Path(clip_dir).resolve()
    output_dir = Path(output_dir).resolve()

    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    wanted_all = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps_all = np.load(clip_dir / "timestamps.npy")
    orig_w, orig_h = meta["original_resolution"]
    num_raw = meta.get("num_raw_frames")
    is_full_frame = (
        num_raw is not None
        and len(wanted_all) == int(num_raw)
        and np.array_equal(wanted_all, np.arange(int(num_raw), dtype=np.int32))
    )
    frame_mode = os.environ.get("MEGASAM_FRAME_MODE", "all" if is_full_frame else "fps")
    fps = str(os.environ.get("MEGASAM_FPS", meta.get("sampled_fps", 10)))

    scene = os.environ.get("MEGASAM_SCENE_NAME", _scene_name(clip_dir))

    if not MEGASAM_REPO.exists():
        raise FileNotFoundError(f"Mega-SaM repo not found: {MEGASAM_REPO}")

    started = time.time()
    workdir = _prepare_workdir(scene)
    npz_path = workdir / "outputs" / f"{scene}_droid.npz"
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": MEGASAM_PYTHON,
        "GPU_ID": MEGASAM_GPU_ID,
        "SCENE_NAME": scene,
        "VIDEO_PATH": str(clip_dir / "clip.mp4"),
        "FRAME_MODE": frame_mode,
        "FPS": fps,
        "TORCH_HOME": env.get("TORCH_HOME", "/data1/duanmengxuan/cache/torch"),
        "HF_HOME": env.get("HF_HOME", "/data1/duanmengxuan/cache/huggingface"),
        "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
    })
    subprocess.run(
        [
            "bash",
            "-lc",
            """
set -euo pipefail
echo "[MegaSaM] root: ${PWD}"
echo "[MegaSaM] scene: ${SCENE_NAME}"
echo "[MegaSaM] video: ${VIDEO_PATH}"
echo "[MegaSaM] gpu: ${GPU_ID}"
echo "[MegaSaM] frame mode: ${FRAME_MODE}"

FRAME_DIR="data/${SCENE_NAME}"
DA_OUT="Depth-Anything/video_visualization/${SCENE_NAME}"
UNI_OUT="UniDepth/outputs"
VIS_DIR="outputs/${SCENE_NAME}_vis"
mkdir -p "${FRAME_DIR}"

if [ "${FRAME_MODE}" = "all" ]; then
  ffmpeg -y -i "${VIDEO_PATH}" "${FRAME_DIR}/%05d.jpg"
elif [ "${FRAME_MODE}" = "fps" ]; then
  ffmpeg -y -i "${VIDEO_PATH}" -vf "fps=${FPS}" "${FRAME_DIR}/%05d.jpg"
else
  echo "FRAME_MODE must be 'all' or 'fps'" >&2
  exit 2
fi

echo "[MegaSaM] frames: $(find "${FRAME_DIR}" -maxdepth 1 -type f \\( -name '*.jpg' -o -name '*.png' \\) | wc -l)"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  Depth-Anything/run_videos.py \
  --encoder vitl \
  --localhub \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path "${FRAME_DIR}" \
  --outdir "${DA_OUT}"

PYTHONPATH="${PWD}/UniDepth" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  UniDepth/scripts/demo_mega-sam.py \
  --scene-name "${SCENE_NAME}" \
  --img-path "${FRAME_DIR}" \
  --outdir "${UNI_OUT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  camera_tracking_scripts/test_demo.py \
  --datapath="${FRAME_DIR}" \
  --weights=checkpoints/megasam_final.pth \
  --scene_name "${SCENE_NAME}" \
  --mono_depth_path "${PWD}/Depth-Anything/video_visualization" \
  --metric_depth_path "${PWD}/UniDepth/outputs" \
  --disable_vis
""",
        ],
        cwd=str(workdir),
        env=env,
        check=True,
    )

    if not npz_path.exists():
        raise FileNotFoundError(f"Mega-SaM did not produce {npz_path}")

    data = np.load(npz_path)
    poses_c2w_all = data["cam_c2w"].astype(np.float64)
    images = data["images"]
    K = data["intrinsic"].astype(np.float64)

    n = min(len(poses_c2w_all), len(wanted_all), len(timestamps_all))
    poses_c2w = poses_c2w_all[:n]
    wanted = wanted_all[:n]
    timestamps = timestamps_all[:n]

    warnings: list[str] = [f"megasam_ran_frame_mode_{frame_mode}"]
    if frame_mode == "fps":
        warnings.append(f"megasam_fps_{fps}")
    status = "ok"
    if n < len(wanted_all):
        status = "degraded"
        warnings.append(f"megasam returned {n} frames; truncated adapter metadata to match")
    if n == 0:
        raise RuntimeError(f"Mega-SaM output contains no frames: {npz_path}")

    result_h, result_w = int(images.shape[1]), int(images.shape[2])
    intr = _intrinsics_from_megasam(K, (result_w, result_h), (int(orig_w), int(orig_h)))
    intrinsics = np.tile(intr, (n, 1)).astype(np.float32)
    poses_w2c = invert_se3_batch(poses_c2w)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = output_dir / f"{scene}_droid.npz"
    if not raw_copy.exists():
        shutil.copy2(npz_path, raw_copy)

    quality = {
        "method": "megasam",
        "status": status,
        "pose_convention": "c2w",
        "camera_coordinate": "opencv",
        "camera_model": "pinhole",
        "scale_status": "normalized",
        "scale_rule": "max_translation_norm",
        "scale_factor": scale_factor,
        "total_path_length_before_norm": total_path,
        "resolution_reference": [orig_w, orig_h],
        "intrinsics_shared_per_frame": True,
        "num_frames": int(n),
        "frame_source_fps": meta["sampled_fps"],
        "original_fps": meta["original_fps"],
        "t_meaning": "sampled_frames",
        "elapsed_s": time.time() - started,
        "warnings": warnings,
        "megasam_repo": str(MEGASAM_REPO),
        "megasam_workdir": str(workdir),
        "megasam_scene_name": scene,
        "megasam_raw_npz": str(npz_path),
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
