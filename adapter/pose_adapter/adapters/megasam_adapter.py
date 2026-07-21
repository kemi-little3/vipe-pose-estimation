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


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MEGASAM_REPO = PROJECT_ROOT / "mega-sam" / "mega-sam"
DEFAULT_MEGASAM_PYTHON = PROJECT_ROOT / "conda_envs" / "SKfP-megasam" / "bin" / "python"
DEFAULT_TORCHHUB_DIR = PROJECT_ROOT / "models" / "torchhub"
DEFAULT_HF_HOME = PROJECT_ROOT / "models" / "huggingface"

MEGASAM_REPO = Path(os.environ.get("MEGASAM_REPO", str(DEFAULT_MEGASAM_REPO)))
MEGASAM_PYTHON = os.environ.get("MEGASAM_PYTHON", str(DEFAULT_MEGASAM_PYTHON))
MEGASAM_GPU_ID = os.environ.get("MEGASAM_GPU_ID", "0")
MEGASAM_WORK_ROOT = Path(os.environ.get("MEGASAM_WORK_ROOT", "/tmp/megasam_adapter_runs"))


def _render_keyframe_select_expr(frame_indices: list[int] | np.ndarray) -> str:
    indices = [int(i) for i in frame_indices]
    if not indices:
        raise ValueError("MegaSaM keyframe_indices must not be empty")
    if any(i < 0 for i in indices):
        raise ValueError(f"MegaSaM keyframe_indices must be non-negative: {indices}")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"MegaSaM keyframe_indices must be strictly increasing: {indices}")
    return "+".join(f"eq(n\\,{i})" for i in indices)


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
    _symlink_force(DEFAULT_TORCHHUB_DIR, root / "torchhub")

    return root


def _build_run_script() -> str:
    return """
set -euo pipefail
echo "[MegaSaM] root: ${PWD}"
echo "[MegaSaM] scene: ${SCENE_NAME}"
echo "[MegaSaM] video: ${VIDEO_PATH}"
echo "[MegaSaM] gpu: ${GPU_ID}"
echo "[MegaSaM] keyframe list: megasam_keyframe_indices.txt"

FRAME_DIR="data/${SCENE_NAME}"
DA_OUT="Depth-Anything/video_visualization/${SCENE_NAME}"
UNI_OUT="UniDepth/outputs"
mkdir -p "${FRAME_DIR}"

ffmpeg -y -i "${VIDEO_PATH}" -vf "select=${KEYFRAME_EXPR}" -vsync 0 "${FRAME_DIR}/%05d.jpg"

echo "[MegaSaM] requested keyframes: $(wc -l < megasam_keyframe_indices.txt)"
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
"""


def run_megasam(clip_dir: Path, output_dir: Path) -> None:
    clip_dir = Path(clip_dir).resolve()
    output_dir = Path(output_dir).resolve()

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    wanted_all = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps_all = np.load(clip_dir / "timestamps.npy")
    orig_w, orig_h = meta["original_resolution"]
    keyframe_expr = _render_keyframe_select_expr(wanted_all)
    timings["read_input_s"] = time.perf_counter() - started

    scene = os.environ.get("MEGASAM_SCENE_NAME", _scene_name(clip_dir))

    if not MEGASAM_REPO.exists():
        raise FileNotFoundError(f"Mega-SaM repo not found: {MEGASAM_REPO}")

    started = time.perf_counter()
    workdir = _prepare_workdir(scene)
    (workdir / "megasam_keyframe_indices.txt").write_text(
        "\n".join(str(int(i)) for i in wanted_all) + "\n"
    )
    timings["prepare_workdir_s"] = time.perf_counter() - started
    npz_path = workdir / "outputs" / f"{scene}_droid.npz"
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": MEGASAM_PYTHON,
        "GPU_ID": MEGASAM_GPU_ID,
        "SCENE_NAME": scene,
        "VIDEO_PATH": str(clip_dir / "clip.mp4"),
        "KEYFRAME_EXPR": keyframe_expr,
        "TORCH_HOME": str(DEFAULT_TORCHHUB_DIR.parent),
        "HF_HOME": str(DEFAULT_HF_HOME),
        "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
    })
    started = time.perf_counter()
    subprocess.run(
        [
            "bash",
            "-lc",
            _build_run_script(),
        ],
        cwd=str(workdir),
        env=env,
        check=True,
    )
    timings["external_run_s"] = time.perf_counter() - started

    started = time.perf_counter()
    if not npz_path.exists():
        raise FileNotFoundError(f"Mega-SaM did not produce {npz_path}")

    data = np.load(npz_path)
    poses_c2w_all = data["cam_c2w"].astype(np.float64)
    images = data["images"]
    K = data["intrinsic"].astype(np.float64)
    timings["load_artifacts_s"] = time.perf_counter() - started

    started = time.perf_counter()
    n = min(len(poses_c2w_all), len(wanted_all), len(timestamps_all))
    poses_c2w = poses_c2w_all[:n]
    wanted = wanted_all[:n]
    timestamps = timestamps_all[:n]

    warnings: list[str] = ["megasam_keyframe_only"]
    status = "ok"
    if len(poses_c2w_all) != len(wanted_all):
        status = "degraded"
        warnings.append(
            f"megasam returned {len(poses_c2w_all)} frames for {len(wanted_all)} requested keyframes; truncated to {n}"
        )
    if n == 0:
        raise RuntimeError(f"Mega-SaM output contains no frames: {npz_path}")

    result_h, result_w = int(images.shape[1]), int(images.shape[2])
    intr = _intrinsics_from_megasam(K, (result_w, result_h), (int(orig_w), int(orig_h)))
    intrinsics = np.tile(intr, (n, 1)).astype(np.float32)
    poses_w2c = invert_se3_batch(poses_c2w)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)
    timings["postprocess_s"] = time.perf_counter() - started

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = output_dir / f"{scene}_droid.npz"
    if not raw_copy.exists():
        shutil.copy2(npz_path, raw_copy)
    timings["copy_raw_s"] = time.perf_counter() - started

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
        "elapsed_s": 0.0,
        "timings": timings,
        "warnings": warnings,
        "megasam_repo": str(MEGASAM_REPO),
        "megasam_workdir": str(workdir),
        "megasam_scene_name": scene,
        "megasam_raw_npz": str(npz_path),
    }

    timings["total_before_save_s"] = time.perf_counter() - total_started
    quality["elapsed_s"] = timings["total_before_save_s"]

    started = time.perf_counter()
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
    timings["save_s"] = time.perf_counter() - started
    timings["total_s"] = time.perf_counter() - total_started
    quality["elapsed_s"] = timings["total_s"]
    quality["timings"] = timings
    (output_dir / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
