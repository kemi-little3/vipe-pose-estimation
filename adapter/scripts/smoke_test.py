#!/usr/bin/env python3
"""Smoke test: validate per-method unified outputs for a given clip.

Usage:
    python scripts/smoke_test.py outputs/<clip_uuid>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ADAPTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ADAPTER_ROOT))

from pose_adapter.schema import load, validate


def _check_one(method_dir: Path, orig_wh: tuple[int, int]) -> tuple[int, int]:
    """Return (num_errors, num_warnings)."""
    print(f"\n=== {method_dir} ===")
    if not method_dir.exists():
        print("  SKIP: directory missing")
        return (0, 0)

    try:
        u = load(method_dir)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return (1, 0)

    T = u.intrinsics.shape[0]
    print(f"  T = {T}, status = {u.quality.get('status')}")
    print(f"  scale_factor = {u.scale_factor:.4f}")
    print(f"  total_path_before_norm = {u.quality.get('total_path_length_before_norm'):.4f}")
    print(f"  fx, fy mean = {float(np.nanmean(u.intrinsics[:, 0])):.1f}, {float(np.nanmean(u.intrinsics[:, 1])):.1f}")
    print(f"  cx, cy mean = {float(np.nanmean(u.intrinsics[:, 2])):.1f}, {float(np.nanmean(u.intrinsics[:, 3])):.1f}")
    print(f"  elapsed_s = {u.quality.get('elapsed_s'):.2f}")
    print(f"  method warnings = {u.quality.get('warnings', [])}")

    issues = validate(u, orig_wh)
    for w in issues:
        print(f"  VALIDATION WARNING: {w}")
    return (0, len(issues))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_out_dir", type=Path, help="outputs/<clip_uuid>/")
    parser.add_argument("--inputs-root", type=Path, default=None,
                        help="inputs/ root, to read original resolution. If omitted, uses resolution_reference from quality.json.")
    args = parser.parse_args()

    clip_out = args.clip_out_dir
    if not clip_out.exists():
        print(f"missing {clip_out}", file=sys.stderr)
        return 2

    # Find the original resolution.
    orig_wh = None
    if args.inputs_root is not None:
        clip_uuid = clip_out.name
        meta_path = args.inputs_root / clip_uuid / "meta.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            orig_wh = tuple(meta["original_resolution"])
    if orig_wh is None:
        # Fallback: read from any method's quality.json.
        for q in clip_out.rglob("quality.json"):
            with q.open() as f:
                orig_wh = tuple(json.load(f)["resolution_reference"])
            break
    if orig_wh is None:
        print("could not determine original resolution", file=sys.stderr)
        return 2

    print(f"clip = {clip_out.name}  original resolution = {orig_wh}")

    total_err = 0
    total_warn = 0
    for method in ("vipe", "vggsfm", "dust3r", "vggt", "megasam"):
        e, w = _check_one(clip_out / method, orig_wh)
        total_err += e
        total_warn += w

    print(f"\nSummary: {total_err} errors, {total_warn} validation warnings")
    return 1 if total_err > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
