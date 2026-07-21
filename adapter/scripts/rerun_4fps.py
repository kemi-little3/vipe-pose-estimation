#!/usr/bin/env python3
"""Re-run ViPE at 4fps on three 1b9902f6 clips; run VGGT + MegaSaM on the first clip.

ViPE previously processed the full 29.97fps mp4 (~256 frames) while other methods
only saw 34 subsampled frames. This script makes the comparison fair by giving ViPE
a 4fps mp4 built from the already-extracted frames/ PNGs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BASE = Path('/data1/hexuming/pose_estimation')
ADAPTER_ROOT = BASE / 'adapter'
VIPE_REPO = Path('/data1/hexuming/pose_estimation/repo/vipe')
VIPE_PYTHON = '/home/hexuming/miniconda3/envs/vipe/bin/python'
VGGT_PYTHON = '/data1/linyuxi/miniconda3/envs/vggt/bin/python'

sys.path.insert(0, str(ADAPTER_ROOT))
from pose_adapter.adapters.vipe_adapter import _find_artifact, _select_rows_by_indices
from pose_adapter.geometry import invert_se3_batch, normalize_max_translation, relative_to_first
from pose_adapter.schema import UnifiedOutput, save

CLIPS = ['1b9902f6', '1b9902f6_256_511', '1b9902f6_512_767']


def make_4fps_mp4(frames_dir: Path, out_mp4: Path) -> None:
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', '4',
        '-i', str(frames_dir / '%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '4',
        str(out_mp4),
    ], check=True)
    print(f'  created 4fps mp4: {out_mp4} ({out_mp4.stat().st_size // 1024}KB)')


def run_vipe_on_4fps_mp4(clip_dir: Path, output_dir: Path, tmp_mp4: Path) -> None:
    with (clip_dir / 'meta.json').open() as f:
        meta = json.load(f)
    timestamps = np.load(clip_dir / 'timestamps.npy')
    orig_w, orig_h = meta['original_resolution']
    frame_files = sorted((clip_dir / 'frames').glob('*.png'))
    T = len(frame_files)
    # Sequential indices: the 4fps mp4 has frames [0..T-1]
    wanted = np.arange(T, dtype=np.int32)

    env = os.environ.copy()
    vipe_model_root = Path(os.environ.get('VIPE_MODEL_ROOT', '/mlp-01/duanmengxuan/data_pipeline/vipe/models'))
    env.update({
        'VIPE_MODEL_ROOT': str(vipe_model_root),
        'TORCH_HOME': str(vipe_model_root / 'torch'),
        'HF_HOME': str(vipe_model_root / 'huggingface'),
    })
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='vipe_out_') as vipe_tmp:
        tmp = Path(vipe_tmp)
        subprocess.run([
            VIPE_PYTHON, 'run.py',
            'pipeline=default',
            'streams=raw_mp4_stream',
            f'streams.base_path={tmp_mp4}',
            f'pipeline.output.path={tmp}',
            'pipeline.output.save_artifacts=true',
            'pipeline.output.save_viz=false',
            'pipeline.post.depth_align_model=null',
        ], check=True, cwd=str(VIPE_REPO), env=env)

        pose_npz = _find_artifact(tmp, 'pose', '.npz')
        intr_npz = _find_artifact(tmp, 'intrinsics', '.npz')
        pose_data = np.load(pose_npz)
        intr_data = np.load(intr_npz)
        pose_all = pose_data['data']
        pose_inds = pose_data['inds']
        intr_all = intr_data['data']
        intr_inds = intr_data['inds']

    pose_sel, pose_fallback = _select_rows_by_indices(pose_all, pose_inds, wanted)
    intr_sel, intr_fallback = _select_rows_by_indices(intr_all, intr_inds, wanted)

    poses_c2w = pose_sel.astype(np.float64)
    poses_w2c = invert_se3_batch(poses_c2w)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)

    warnings: list[str] = ['vipe_ran_on_4fps_mp4_from_extracted_frames']
    if pose_fallback:
        warnings.append(f'vipe_pose_nn_fallback_{len(pose_fallback)}_positions')
    if intr_fallback:
        warnings.append(f'vipe_intrinsics_nn_fallback_{len(intr_fallback)}_positions')

    quality = {
        'method': 'vipe', 'status': 'ok',
        'pose_convention': 'c2w', 'camera_coordinate': 'opencv',
        'camera_model': 'pinhole', 'scale_status': 'normalized',
        'scale_rule': 'max_translation_norm',
        'scale_factor': scale_factor,
        'total_path_length_before_norm': total_path,
        'resolution_reference': [orig_w, orig_h],
        'intrinsics_shared_per_frame': False,
        'num_frames': int(poses_c2w.shape[0]),
        'frame_source_fps': 4,
        'original_fps': meta['original_fps'],
        't_meaning': 'sampled_frames',
        'elapsed_s': time.time() - started,
        'warnings': warnings,
    }

    save(output_dir, UnifiedOutput(
        intrinsics=intr_sel.astype(np.float32),
        poses_c2w=poses_c2w, poses_w2c=poses_w2c,
        relative_poses=rel_norm, scale_factor=scale_factor,
        frame_indices=wanted, timestamps=timestamps,
        quality=quality,
    ))
    print(f'  vipe done: {T} frames, {quality["elapsed_s"]:.1f}s, '
          f'scale={scale_factor:.4f}, fallbacks={len(pose_fallback)+len(intr_fallback)}')


def run_vggt_subprocess(clip_dir: Path, output_dir: Path) -> None:
    """Run VGGT via linyuxi's python (the only env with vggt installed)."""
    subprocess.run([
        VGGT_PYTHON,
        str(ADAPTER_ROOT / 'scripts' / 'run_adapter.py'),
        '--method', 'vggt',
        '--clip-dir', str(clip_dir),
        '--out-root', str(output_dir.parent.parent),
        '--clip-uuid', output_dir.parent.name,
    ], check=True)
    print(f'  vggt done -> {output_dir}')


def make_temp_clip_dir(clip_dir: Path, tmp_mp4: Path) -> Path:
    """Create a temp input dir with a working clip.mp4 and sequential frame_indices.

    MegaSaM reads clip.mp4 and re-extracts frames at fps internally. Using the
    4fps mp4 here means MegaSaM sees the same 34 frames as all other methods.
    The sequential frame_indices [0..T-1] match the 4fps mp4's frame numbering.
    """
    tmp_input = Path(tempfile.mkdtemp(prefix='megasam_input_'))
    with (clip_dir / 'meta.json').open() as f:
        meta = json.load(f)
    timestamps = np.load(clip_dir / 'timestamps.npy')
    T = len(sorted((clip_dir / 'frames').glob('*.png')))

    frame_indices = np.arange(T, dtype=np.int32)
    np.save(tmp_input / 'frame_indices.npy', frame_indices)
    np.save(tmp_input / 'timestamps.npy', timestamps)

    new_meta = {
        **meta,
        'num_raw_frames': T,
        'sampled_fps': 4,
        'num_sampled': T,
    }
    with (tmp_input / 'meta.json').open('w') as f:
        json.dump(new_meta, f, indent=2)

    # Copy (not symlink) so MegaSaM can always access it
    shutil.copy2(tmp_mp4, tmp_input / 'clip.mp4')
    return tmp_input


def run_megasam(clip_dir: Path, output_dir: Path, tmp_mp4: Path) -> None:
    tmp_input = make_temp_clip_dir(clip_dir, tmp_mp4)
    try:
        from pose_adapter.adapters.megasam_adapter import run_megasam as _run
        _run(tmp_input, output_dir)
        print(f'  megasam done -> {output_dir}')
    finally:
        shutil.rmtree(tmp_input, ignore_errors=True)


if __name__ == '__main__':
    # ------------------------------------------------------------------ ViPE
    for clip in CLIPS:
        print(f'\n[ViPE 4fps] {clip}')
        clip_dir = BASE / 'inputs' / clip
        output_dir = BASE / 'outputs' / clip / 'vipe'
        with tempfile.TemporaryDirectory(prefix=f'mp4_{clip}_') as mp4_dir:
            tmp_mp4 = Path(mp4_dir) / 'input_4fps.mp4'
            make_4fps_mp4(clip_dir / 'frames', tmp_mp4)
            run_vipe_on_4fps_mp4(clip_dir, output_dir, tmp_mp4)

    # ------------------------------------------------------------------ VGGT
    print(f'\n[VGGT] 1b9902f6')
    run_vggt_subprocess(
        BASE / 'inputs' / '1b9902f6',
        BASE / 'outputs' / '1b9902f6' / 'vggt',
    )

    # ------------------------------------------------------------------ MegaSaM
    print(f'\n[MegaSaM] 1b9902f6')
    clip_dir = BASE / 'inputs' / '1b9902f6'
    with tempfile.TemporaryDirectory(prefix='megasam_mp4_') as mp4_dir:
        tmp_mp4 = Path(mp4_dir) / 'input_4fps.mp4'
        make_4fps_mp4(clip_dir / 'frames', tmp_mp4)
        run_megasam(clip_dir, BASE / 'outputs' / '1b9902f6' / 'megasam', tmp_mp4)

    # ------------------------------------------------------------------ plots
    print(f'\nRegenerating trajectory comparison plots...')
    for clip in CLIPS:
        clip_out = BASE / 'outputs' / clip
        subprocess.run([
            VIPE_PYTHON,
            str(ADAPTER_ROOT / 'scripts' / 'compare_trajectories.py'),
            str(clip_out),
        ], check=True)
        print(f'  wrote {clip_out}/trajectory_compare.png')

    print('\n=== all done ===')
