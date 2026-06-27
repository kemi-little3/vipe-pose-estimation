#!/usr/bin/env python3
"""Prepare an inputs/<clip_id>/ directory for full-frame GT evaluation.

Unlike extract_clip.py, this does not export image frames. It only stages the
mp4 and writes frame_indices.npy/timestamps.npy/meta.json from the GT CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
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
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]

    def parse_rate(value: str) -> float:
        num, den = value.split("/")
        return float(num) / float(den) if float(den) != 0 else 0.0

    avg_fps = parse_rate(stream.get("avg_frame_rate", "0/1"))
    r_fps = parse_rate(stream.get("r_frame_rate", "0/1"))
    fps = avg_fps if avg_fps > 0 else r_fps
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
        "duration": float(stream["duration"]) if "duration" in stream else None,
    }


def read_gt_frame_indices(gt_csv: Path) -> np.ndarray:
    indices: list[int] = []
    with gt_csv.open() as f:
        for row in csv.DictReader(f):
            indices.append(int(row["frame_idx"]))
    if not indices:
        raise ValueError(f"empty GT CSV: {gt_csv}")
    return np.asarray(indices, dtype=np.int32)


def prepare(mp4: Path, gt_csv: Path, out_dir: Path, copy_source: bool = False) -> dict:
    mp4 = mp4.resolve()
    gt_csv = gt_csv.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_link = out_dir / "clip.mp4"
    if clip_link.exists() or clip_link.is_symlink():
        clip_link.unlink()
    if copy_source:
        shutil.copy2(mp4, clip_link)
    else:
        clip_link.symlink_to(mp4)

    info = _ffprobe_stream(mp4)
    if info["fps"] <= 0:
        raise RuntimeError(f"could not determine fps for {mp4}")

    frame_indices = read_gt_frame_indices(gt_csv)
    timestamps = frame_indices.astype(np.float64) / float(info["fps"])
    np.save(out_dir / "frame_indices.npy", frame_indices)
    np.save(out_dir / "timestamps.npy", timestamps)

    meta = {
        "source_mp4": str(mp4),
        "gt_csv": str(gt_csv),
        "original_resolution": [info["width"], info["height"]],
        "original_fps": info["fps"],
        "duration_s": info["duration"],
        "num_raw_frames": info["nb_frames"],
        "sampled_fps": info["fps"],
        "sample_policy": "all_gt_csv_frame_idx",
        "num_sampled": int(len(frame_indices)),
        "first_frame_idx": int(frame_indices[0]),
        "last_frame_idx": int(frame_indices[-1]),
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mp4", type=Path)
    parser.add_argument("gt_csv", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    meta = prepare(args.mp4, args.gt_csv, args.out_dir, copy_source=args.copy)
    print(
        f"[prepare_gt_full_input] wrote {meta['num_sampled']} GT frames "
        f"({meta['first_frame_idx']}..{meta['last_frame_idx']}) to {args.out_dir}"
    )
    print(
        f"  original: {meta['original_resolution'][0]}x{meta['original_resolution'][1]}"
        f" @ {meta['original_fps']:.3f} fps, duration {meta['duration_s']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
