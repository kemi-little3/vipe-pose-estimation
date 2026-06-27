#!/usr/bin/env python3
"""Extract all frames from each _full GT clip, run VGGT (MAX_FRAMES=500), evaluate vs GT.

Clips are processed sequentially shortest-first so an OOM on large clips is visible
without wasting time. Frames are extracted at original resolution into inputs/<clip>/frames/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE        = Path('/data1/hexuming/pose_estimation')
ADAPTER_ROOT = BASE / 'adapter'
VIPE_PYTHON  = '/home/hexuming/miniconda3/envs/vipe/bin/python'
VGGT_PYTHON  = '/data1/linyuxi/miniconda3/envs/vggt/bin/python'
GT_CSV_ROOT  = BASE / 'pose_demo' / 'pose_demo'

CLIPS_FULL = [
    'gt_2026-05-14_11-05-53_full',   # 425 f  (shortest — verify memory first)
    'gt_2026-05-14_10-57-13_full',   # 772 f
    'gt_2026-05-14_10-57-50_full',   # 874 f
    'gt_2026-05-14_11-00-05_full',   # 909 f
    'gt_2026-05-14_11-02-37_full',   # 990 f
    'gt_2026-05-14_11-04-53_full',   # 1280 f
    'gt_2026-05-14_11-03-27_full',   # 1645 f
    'gt_2026-05-14_11-01-24_full',   # 1750 f
    'gt_2026-05-14_11-06-27_full',   # 1750 f
    'gt_2026-05-14_10-58-32_full',   # 1751 f
]


def gt_csv(clip_id: str) -> Path:
    ts = clip_id.replace('gt_', '').replace('_full', '')
    return GT_CSV_ROOT / f'telemetry_vehicle_nav_{ts}_aligned_processed.csv'


def extract_frames(clip_dir: Path) -> int:
    """Extract all frames from clip.mp4 into frames/. Skip if already done."""
    frames_dir = clip_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    existing = sorted(frames_dir.glob('*.png'))
    meta = json.load((clip_dir / 'meta.json').open())
    expected = int(meta['num_raw_frames'])
    if len(existing) == expected:
        print(f'  frames/ already complete ({expected} PNGs), skipping extraction')
        return expected
    print(f'  extracting {expected} frames (have {len(existing)})...')
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(clip_dir / 'clip.mp4'),
        '-vsync', '0',
        str(frames_dir / '%04d.png'),
    ], check=True)
    actual = len(sorted(frames_dir.glob('*.png')))
    print(f'  extracted {actual} PNGs')
    return actual


def run_vggt(clip_id: str) -> bool:
    print(f'\n[VGGT] {clip_id}')
    clip_dir  = BASE / 'inputs' / clip_id
    out_dir   = BASE / 'outputs' / clip_id / 'vggt'

    if out_dir.exists() and (out_dir / 'quality.json').exists():
        import json as _j
        q = _j.load((out_dir / 'quality.json').open())
        print(f'  already done ({q["num_frames"]} frames, {q["elapsed_s"]:.1f}s) — skipping')
        return True

    extract_frames(clip_dir)

    result = subprocess.run([
        VGGT_PYTHON,
        str(ADAPTER_ROOT / 'scripts' / 'run_adapter.py'),
        '--method', 'vggt',
        '--clip-dir', str(clip_dir),
        '--out-root', str(BASE / 'outputs'),
        '--clip-uuid', clip_id,
    ], env={**__import__('os').environ, 'VGGT_MAX_FRAMES': '500'},
       capture_output=True, text=True)

    if result.returncode != 0:
        print(f'  VGGT FAILED:\n{result.stderr[-3000:]}')
        return False

    import json as _j
    q = _j.load((out_dir / 'quality.json').open())
    print(f'  OK: {q["num_frames"]} frames  elapsed={q["elapsed_s"]:.1f}s  '
          f'status={q["status"]}  warnings={q["warnings"]}')
    return True


def run_eval(clip_id: str) -> None:
    csv = gt_csv(clip_id)
    if not csv.exists():
        print(f'  GT CSV missing: {csv}')
        return
    out_prefix = BASE / 'outputs' / clip_id / 'gt_eval'
    result = subprocess.run([
        VIPE_PYTHON,
        str(ADAPTER_ROOT / 'scripts' / 'evaluate_with_gt.py'),
        '--clip-out', str(BASE / 'outputs' / clip_id),
        '--clip-in',  str(BASE / 'inputs'  / clip_id),
        '--gt-csv',   str(csv),
        '--out-prefix', str(out_prefix),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  eval FAILED:\n{result.stderr[-1000:]}')
    else:
        print(result.stdout.strip())


if __name__ == '__main__':
    for clip_id in CLIPS_FULL:
        ok = run_vggt(clip_id)
        if ok:
            run_eval(clip_id)
        else:
            print(f'  skipping eval due to VGGT failure')

    print('\n=== all done ===')
