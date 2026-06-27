"""Frame extraction: mp4 -> inputs/<clip_uuid>/ layout.

Produces:
  clip.mp4                           (copy / symlink of source)
  frames/0000.png, 0001.png, ...     sampled at target_fps
  timestamps.npy                     [T] seconds
  frame_indices.npy                  [T] int32, which original mp4 frame each sample is
  meta.json                          {original_resolution, original_fps, duration_s, num_raw_frames, sampled_fps, num_sampled}
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _ffprobe_stream(mp4: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "json",
            str(mp4),
        ],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]

    # r_frame_rate is a rational string like "30000/1001" or "24/1"
    num, den = stream.get("r_frame_rate", "0/1").split("/")
    r_fps = float(num) / float(den) if float(den) != 0 else 0.0

    num, den = stream.get("avg_frame_rate", "0/1").split("/")
    avg_fps = float(num) / float(den) if float(den) != 0 else 0.0

    # Prefer avg_frame_rate when it's sane (r_frame_rate can be wrong for VFR).
    fps = avg_fps if avg_fps > 0 else r_fps

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(stream["nb_frames"]) if "nb_frames" in stream and stream["nb_frames"].isdigit() else None,
        "duration": float(stream["duration"]) if "duration" in stream else None,
    }


def extract_clip(
    mp4: Path,
    out_dir: Path,
    fps: int = 4,
    copy_source: bool = False,
) -> dict:
    """Extract frames at `fps` using ffmpeg, write the standard input layout.

    If copy_source=False (default), the source mp4 is symlinked into out_dir/clip.mp4
    to save disk space. Set True to copy.
    """
    mp4 = Path(mp4).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    # Stage the mp4
    clip_link = out_dir / "clip.mp4"
    if clip_link.exists() or clip_link.is_symlink():
        clip_link.unlink()
    if copy_source:
        shutil.copy2(mp4, clip_link)
    else:
        clip_link.symlink_to(mp4)

    info = _ffprobe_stream(mp4)
    orig_fps = info["fps"]
    if orig_fps <= 0:
        raise RuntimeError(f"could not determine fps for {mp4}")

    # Extract. -vsync vfr + -vf fps= avoids duplicate frames on VFR input.
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp4),
            "-vf", f"fps={fps}",
            "-vsync", "vfr",
            "-start_number", "0",
            str(out_dir / "frames" / "%04d.png"),
        ],
        check=True,
    )

    frame_files = sorted((out_dir / "frames").glob("*.png"))
    T = len(frame_files)
    if T == 0:
        raise RuntimeError(f"ffmpeg produced no frames for {mp4}")

    # Timestamps at evenly spaced target_fps starts: 0/fps, 1/fps, ...
    timestamps = np.arange(T, dtype=np.float64) / float(fps)
    # Map each sampled timestamp back to the nearest original mp4 frame index.
    frame_indices = np.round(timestamps * orig_fps).astype(np.int32)
    if info["nb_frames"] is not None:
        frame_indices = np.clip(frame_indices, 0, info["nb_frames"] - 1)

    np.save(out_dir / "timestamps.npy", timestamps)
    np.save(out_dir / "frame_indices.npy", frame_indices)

    meta = {
        "source_mp4": str(mp4),
        "original_resolution": [info["width"], info["height"]],
        "original_fps": orig_fps,
        "duration_s": info["duration"],
        "num_raw_frames": info["nb_frames"],
        "sampled_fps": int(fps),
        "num_sampled": int(T),
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return meta
