#!/usr/bin/env python3
"""Run ViPE A/B comparison for all original pose_demo videos."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

POSE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIPE_PYTHON = POSE_ROOT / ".." / "envs" / "vipe" / "bin" / "python"


def original_stems(gt_root: Path) -> list[str]:
    stems: list[str] = []
    for mp4 in sorted(gt_root.glob("*_aligned.mp4")):
        stem = mp4.name.removesuffix("_aligned.mp4")
        if "_16fps_" in stem or "_640x352" in stem:
            continue
        csv_path = gt_root / f"telemetry_vehicle_nav_{stem}_aligned_processed.csv"
        if csv_path.exists():
            stems.append(stem)
        else:
            print(f"[skip] missing GT CSV for {stem}: {csv_path}", flush=True)
    return stems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=POSE_ROOT / "pose_demo" / "data")
    parser.add_argument("--output-root", type=Path, default=POSE_ROOT / "outputs" / "cmp_original_all")
    parser.add_argument("--window-size", type=int, default=81)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--vipe-python", type=Path, default=DEFAULT_VIPE_PYTHON)
    parser.add_argument("--runner", choices=["subprocess", "resident"], default="resident")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--limit-videos", type=int, default=0)
    args = parser.parse_args(argv)

    gt_root = args.gt_root.resolve()
    output_root = args.output_root.resolve()
    stride = args.stride or args.window_size
    stems = original_stems(gt_root)
    if args.limit_videos > 0:
        stems = stems[: args.limit_videos]
    if not stems:
        raise SystemExit(f"no original *_aligned.mp4 + GT CSV pairs found in {gt_root}")

    print(f"[run-all] videos={len(stems)} output_root={output_root}", flush=True)
    compare_script = POSE_ROOT / "scripts" / "compare_vipe_window_modes.py"
    for index, stem in enumerate(stems, start=1):
        exp_dir = output_root / f"{stem}_ws{args.window_size}_stride{stride}_start0"
        summary = exp_dir / "summary.csv"
        if args.skip_existing and summary.exists() and not args.force:
            print(f"[run-all] skip existing {index}/{len(stems)} {stem}: {summary}", flush=True)
            continue
        print(f"[run-all] run {index}/{len(stems)} {stem}", flush=True)
        cmd = [
            str(args.vipe_python.resolve()),
            str(compare_script),
            "--runner", args.runner,
            "--stem", stem,
            "--gt-root", str(gt_root),
            "--window-size", str(args.window_size),
            "--limit-windows", "0",
            "--cuda-visible-devices", args.cuda_visible_devices,
            "--output-root", str(output_root),
        ]
        if args.stride is not None:
            cmd.extend(["--stride", str(args.stride)])
        if args.force:
            cmd.append("--force")
        subprocess.run(cmd, check=True)

    aggregate_dir = output_root / "_aggregate"
    summarize_script = POSE_ROOT / "scripts" / "summarize_vipe_window_mode_compare.py"
    subprocess.run(
        [
            str(args.vipe_python.resolve()),
            str(summarize_script),
            "--input-root", str(output_root),
            "--out-dir", str(aggregate_dir),
            "--gt-root", str(gt_root),
        ],
        check=True,
    )
    print(f"[run-all] aggregate summary: {aggregate_dir / 'aggregate_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
