#!/usr/bin/env python3
"""Run vggsfm / dust3r / vggt on gt_2026-05-14_10-57-13_full, then evaluate vs GT."""
from __future__ import annotations

import subprocess, sys, json
from pathlib import Path

BASE         = Path('/data1/hexuming/pose_estimation')
ADAPTER_ROOT = BASE / 'adapter'
CLIP         = 'gt_2026-05-14_11-05-53_full'
CLIP_DIR     = BASE / 'inputs'  / CLIP
OUT_ROOT     = BASE / 'outputs'
GT_CSV       = BASE / 'pose_demo' / 'pose_demo' / 'telemetry_vehicle_nav_2026-05-14_11-05-53_aligned_processed.csv'

VIPE_PYTHON  = '/home/hexuming/miniconda3/envs/vipe/bin/python'
VGGSFM_PY   = '/home/hexuming/miniconda3/envs/vggsfm/bin/python'
DUST3R_PY   = '/data1/linyuxi/miniconda3/envs/dust3r/bin/python'
VGGT_PY     = '/data1/linyuxi/miniconda3/envs/vggt/bin/python'

JOBS = [
    ('vggsfm', VGGSFM_PY, {'VGGSFM_MAX_FRAMES': '425'}),
    ('dust3r',  DUST3R_PY,  {'DUST3R_MAX_FRAMES': '425'}),
    ('vggt',    VGGT_PY,    {'CUDA_VISIBLE_DEVICES': '7', 'VGGT_MAX_FRAMES': '425'}),
]


def run_method(method, pybin, extra_env):
    out_dir = OUT_ROOT / CLIP / method
    if out_dir.exists() and (out_dir / 'quality.json').exists():
        q = json.load((out_dir / 'quality.json').open())
        print(f'[{method}] already done ({q["num_frames"]}f, {q["elapsed_s"]:.0f}s) — skip')
        return True

    print(f'\n[{method}] running...')
    env = {**__import__('os').environ, **extra_env}
    r = subprocess.run(
        [pybin, str(ADAPTER_ROOT / 'scripts' / 'run_adapter.py'),
         '--method', method, '--clip-dir', str(CLIP_DIR),
         '--out-root', str(OUT_ROOT), '--clip-uuid', CLIP],
        env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f'[{method}] FAILED:\n{r.stderr[-2000:]}')
        return False
    q = json.load((out_dir / 'quality.json').open())
    print(f'[{method}] OK: {q["num_frames"]}f  {q["elapsed_s"]:.0f}s  status={q["status"]}  warnings={q["warnings"]}')
    return True


def run_eval():
    print('\n[eval] evaluate_with_gt...')
    r = subprocess.run(
        [VIPE_PYTHON, str(ADAPTER_ROOT / 'scripts' / 'evaluate_with_gt.py'),
         '--clip-out', str(OUT_ROOT / CLIP),
         '--clip-in',  str(CLIP_DIR),
         '--gt-csv',   str(GT_CSV),
         '--out-prefix', str(OUT_ROOT / CLIP / 'gt_eval'),
         '--axis-search'],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f'[eval] FAILED:\n{r.stderr[-1000:]}')
    else:
        print(r.stdout.strip())


if __name__ == '__main__':
    for method, pybin, extra_env in JOBS:
        run_method(method, pybin, extra_env)
    run_eval()
    print('\n=== done ===')
