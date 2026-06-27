#!/usr/bin/env python3
"""CLI: extract frames from an mp4 into the unified inputs/<clip_uuid>/ layout.

Usage:
    python scripts/extract_clip.py <mp4> <out_dir> [--fps 4] [--copy]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without pip install.
_ADAPTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ADAPTER_ROOT))

from pose_adapter.frames import extract_clip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mp4", type=Path, help="Input video file")
    parser.add_argument("out_dir", type=Path, help="Output directory (inputs/<clip_uuid>/)")
    parser.add_argument("--fps", type=int, default=4, help="Sampling rate (default: 4)")
    parser.add_argument("--copy", action="store_true", help="Copy mp4 instead of symlinking")
    args = parser.parse_args()

    meta = extract_clip(args.mp4, args.out_dir, fps=args.fps, copy_source=args.copy)
    print(f"[extract_clip] wrote {meta['num_sampled']} frames at {args.fps} fps to {args.out_dir}")
    print(f"  original: {meta['original_resolution'][0]}x{meta['original_resolution'][1]}"
          f" @ {meta['original_fps']:.3f} fps, duration {meta['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
