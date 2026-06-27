#!/usr/bin/env python3
"""Dispatch a single method on a single clip.

Usage:
    python scripts/run_adapter.py \
        --method {vipe|vggsfm|dust3r|vggt|megasam} \
        --clip-dir inputs/<clip_uuid> \
        --out-root outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ADAPTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ADAPTER_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["vipe", "vggsfm", "dust3r", "vggt", "megasam"])
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="outputs/ root; per-method subdirs created under <out_root>/<clip_uuid>/<method>/")
    parser.add_argument("--clip-uuid", type=str, default=None,
                        help="Override clip uuid (default: clip-dir basename)")
    args = parser.parse_args()

    clip_uuid = args.clip_uuid or args.clip_dir.name
    out_dir = args.out_root / clip_uuid / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "vipe":
        from pose_adapter.adapters.vipe_adapter import run_vipe
        run_vipe(args.clip_dir, out_dir)
    elif args.method == "vggsfm":
        from pose_adapter.adapters.vggsfm_adapter import run_vggsfm
        run_vggsfm(args.clip_dir, out_dir)
    elif args.method == "dust3r":
        from pose_adapter.adapters.dust3r_adapter import run_dust3r
        run_dust3r(args.clip_dir, out_dir)
    elif args.method == "vggt":
        from pose_adapter.adapters.vggt_adapter import run_vggt
        run_vggt(args.clip_dir, out_dir)
    elif args.method == "megasam":
        from pose_adapter.adapters.megasam_adapter import run_megasam
        run_megasam(args.clip_dir, out_dir)
    else:
        raise ValueError(args.method)

    print(f"[run_adapter] {args.method} done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
