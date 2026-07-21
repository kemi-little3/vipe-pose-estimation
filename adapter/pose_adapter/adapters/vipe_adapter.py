"""ViPE adapter: runs `vipe infer` via subprocess and converts its .npz output.

ViPE already emits c2w in OpenCV convention at the video's original resolution, so
this adapter is mostly glue: trigger the run, locate the artifact files, pick the
rows matching our sampled frame_indices, and write the unified schema.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..geometry import invert_se3_batch, normalize_max_translation, relative_to_first
from ..schema import UnifiedOutput, save


def _default_vipe_cmd() -> str:
    """Prefer the `vipe` console script that sits next to the currently-running
    python interpreter (e.g. /home/hexuming/miniconda3/envs/vipe/bin/vipe).

    Falls back to bare "vipe" if that file is missing, which lets a user-set
    PATH take over. Override entirely with the VIPE_CMD env var.
    """
    sibling = Path(sys.executable).parent / "vipe"
    if sibling.exists():
        return str(sibling)
    return "vipe"


DEFAULT_VIPE_CMD = os.environ.get("VIPE_CMD", _default_vipe_cmd())
def _default_vipe_repo() -> str:
    # adapter/pose_adapter/adapters/vipe_adapter.py -> pose_estimation/repo/vipe
    pose_estimation_root = Path(__file__).resolve().parents[3]
    return str(pose_estimation_root / "repo" / "vipe")


DEFAULT_VIPE_REPO = os.environ.get("VIPE_REPO", _default_vipe_repo())
DEFAULT_VIPE_MODEL_ROOT = Path(
    os.environ.get(
        "VIPE_MODEL_ROOT",
        "/mlp-01/duanmengxuan/data_pipeline/vipe/models",
    )
)
# When 1 (default) skip the dense depth alignment post-step, which is unused
# downstream — saves ~30-50% wall time per window. Set to 0 to fall back to the
# original `vipe infer` invocation.
USE_RUN_PY = os.environ.get("VIPE_DISABLE_DEPTH_ALIGN", "1") == "1"


def _find_artifact(base: Path, subdir: str, suffix: str) -> Path:
    matches = sorted((base / subdir).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"no {suffix} artifact under {base / subdir}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple {suffix} artifacts under {base / subdir}: {matches}")
    return matches[0]


def _select_rows_by_indices(
    data: np.ndarray,
    inds: np.ndarray,
    wanted: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """For each `wanted` mp4 frame index, pick the row in `data` whose `inds` matches.

    Falls back to the nearest `inds` when an exact match is missing. Returns the
    selected rows and the list of wanted positions where a fallback was used.
    """
    inds_sorted_pos = np.argsort(inds)
    sorted_inds = inds[inds_sorted_pos]

    out_rows = np.empty((len(wanted), *data.shape[1:]), dtype=data.dtype)
    fallback_positions: list[int] = []

    for i, w in enumerate(wanted):
        pos = np.searchsorted(sorted_inds, w)
        # Consider pos and pos-1 to find the true nearest.
        candidates = []
        if pos < len(sorted_inds):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best = min(candidates, key=lambda p: abs(int(sorted_inds[p]) - int(w)))
        matched_ind = int(sorted_inds[best])
        if matched_ind != int(w):
            fallback_positions.append(i)
        out_rows[i] = data[inds_sorted_pos[best]]

    return out_rows, fallback_positions


def run_vipe(clip_dir: Path, output_dir: Path, vipe_cmd: str = DEFAULT_VIPE_CMD) -> None:
    clip_dir = Path(clip_dir).resolve()
    output_dir = Path(output_dir).resolve()

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    wanted = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps = np.load(clip_dir / "timestamps.npy")
    orig_w, orig_h = meta["original_resolution"]
    timings["read_input_s"] = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="vipe_out_") as tmpdir:
        tmp = Path(tmpdir)
        started = time.perf_counter()
        env = os.environ.copy()
        env.setdefault("VIPE_MODEL_ROOT", str(DEFAULT_VIPE_MODEL_ROOT))
        env.setdefault("TORCH_HOME", str(DEFAULT_VIPE_MODEL_ROOT / "torch"))
        env.setdefault("HF_HOME", str(DEFAULT_VIPE_MODEL_ROOT / "huggingface"))
        if USE_RUN_PY:
            # Drive ViPE via run.py with a hydra override that disables the dense
            # depth post-alignment step (we only consume pose+intrinsics, never
            # the depth artefacts). Saves ~30-50% per-window wall time.
            python_bin = str(Path(vipe_cmd).parent / "python") if Path(vipe_cmd).exists() else sys.executable
            subprocess.run(
                [
                    python_bin,
                    "run.py",
                    "pipeline=default",
                    "streams=raw_mp4_stream",
                    f"streams.base_path={clip_dir / 'clip.mp4'}",
                    f"pipeline.output.path={tmp}",
                    "pipeline.output.save_artifacts=true",
                    "pipeline.output.save_viz=false",
                    "pipeline.post.depth_align_model=null",
                ],
                check=True,
                cwd=DEFAULT_VIPE_REPO,
                env=env,
            )
        else:
            subprocess.run(
                [vipe_cmd, "infer", str(clip_dir / "clip.mp4"), "--output", str(tmp)],
                check=True,
                env=env,
            )
        timings["external_run_s"] = time.perf_counter() - started

        started = time.perf_counter()
        pose_npz = _find_artifact(tmp, "pose", ".npz")
        intr_npz = _find_artifact(tmp, "intrinsics", ".npz")

        pose_data = np.load(pose_npz)
        intr_data = np.load(intr_npz)
        pose_all = pose_data["data"]          # [N, 4, 4] c2w
        pose_inds = pose_data["inds"]         # [N] int
        intr_all = intr_data["data"]          # [N, 4] fx, fy, cx, cy
        intr_inds = intr_data["inds"]
        timings["load_artifacts_s"] = time.perf_counter() - started

    started = time.perf_counter()
    pose_sel, pose_fallback = _select_rows_by_indices(pose_all, pose_inds, wanted)
    intr_sel, intr_fallback = _select_rows_by_indices(intr_all, intr_inds, wanted)

    poses_c2w = pose_sel.astype(np.float64)
    poses_w2c = invert_se3_batch(poses_c2w)

    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)

    warnings: list[str] = []
    if pose_fallback:
        warnings.append(f"vipe pose nearest-neighbour fallback at {len(pose_fallback)} positions")
    if intr_fallback:
        warnings.append(f"vipe intrinsics nearest-neighbour fallback at {len(intr_fallback)} positions")
    timings["postprocess_s"] = time.perf_counter() - started

    quality = {
        "method": "vipe",
        "status": "ok",
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
        "elapsed_s": 0.0,
        "timings": timings,
        "warnings": warnings,
    }

    timings["total_before_save_s"] = time.perf_counter() - total_started
    quality["elapsed_s"] = timings["total_before_save_s"]

    started = time.perf_counter()
    save(output_dir, UnifiedOutput(
        intrinsics=intr_sel.astype(np.float32),
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
