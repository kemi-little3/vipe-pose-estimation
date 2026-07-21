#!/usr/bin/env python3
"""Compare two ViPE strategies on GT pose_demo clips.

Modes compared on one GPU by default:
A. resident_per_window: start one internal stage-style ViPE worker, keep it alive,
   and send each window to that worker as an independent ViPE job.
B. resident_clip_once_then_slice: after A exits, start a second internal stage-style
   ViPE worker, run the full clip once, and slice its pose output by window.

Each window is evaluated against the same GT frame indices so accuracy numbers are
comparable. The resident runner mirrors the internal vLLM caption stage lifecycle:
stage_setup() once, process_job() many times, destroy() once.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

POSE_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = POSE_ROOT / "adapter"
sys.path.insert(0, str(ADAPTER_ROOT))

from pose_adapter.geometry import invert_se3_batch, normalize_max_translation, relative_to_first  # noqa: E402
from pose_adapter.schema import UnifiedOutput, load, save  # noqa: E402


@dataclass(frozen=True)
class WindowSpec:
    index: int
    start_frame: int
    end_frame: int
    frame_indices: np.ndarray

    @property
    def window_id(self) -> str:
        return f"window_{self.index:04d}_{self.start_frame}_{self.end_frame}"

    @property
    def num_frames(self) -> int:
        return int(len(self.frame_indices))


def log_cmd(cmd: list[str | Path]) -> None:
    print("+ " + shlex.join(str(x) for x in cmd), flush=True)


def run(cmd: list[str | Path], *, env: dict[str, str] | None = None, dry_run: bool = False) -> float:
    log_cmd(cmd)
    if dry_run:
        return 0.0
    started = time.perf_counter()
    subprocess.run([str(x) for x in cmd], check=True, env=env)
    return time.perf_counter() - started


def first_gpu_index(cuda_visible_devices: str | None) -> str | None:
    if not cuda_visible_devices:
        return None
    first = cuda_visible_devices.split(",")[0].strip()
    return first or None


def check_cuda_available(vipe_python: Path, env: dict[str, str]) -> None:
    code = """
import json
import os
import torch
out = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "available": torch.cuda.is_available(),
    "count": torch.cuda.device_count(),
    "names": [],
}
if out["available"]:
    out["names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps(out, ensure_ascii=False))
"""
    result = subprocess.run(
        [str(vipe_python), "-c", code],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    print(
        "[cuda-check] "
        f"CUDA_VISIBLE_DEVICES={info.get('cuda_visible_devices')} "
        f"available={info.get('available')} count={info.get('count')} "
        f"torch={info.get('torch')} torch_cuda={info.get('torch_cuda')} "
        f"names={info.get('names')}",
        flush=True,
    )
    if not info.get("available") or int(info.get("count") or 0) <= 0:
        raise SystemExit(
            "ViPE requires a visible CUDA GPU, but torch in the ViPE env cannot see one. "
            "Check nvidia-smi, CUDA_VISIBLE_DEVICES, and whether the current shell/container exposes /dev/nvidia*."
        )


def parse_rate(value: str) -> float:
    num, den = value.split("/")
    return float(num) / float(den) if float(den) else 0.0


def ffprobe_stream(mp4: Path) -> dict[str, Any]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "json",
            str(mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    fps = parse_rate(stream.get("avg_frame_rate", "0/1")) or parse_rate(stream.get("r_frame_rate", "0/1"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
        "duration": float(stream["duration"]) if stream.get("duration") else None,
    }


def count_frames_exact(mp4: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip())


def read_gt_frame_indices(gt_csv: Path) -> np.ndarray:
    rows: list[int] = []
    with gt_csv.open() as f:
        for row in csv.DictReader(f):
            rows.append(int(row["frame_idx"]))
    if not rows:
        raise ValueError(f"empty GT CSV: {gt_csv}")
    return np.asarray(rows, dtype=np.int32)


def build_windows(
    gt_indices: np.ndarray,
    *,
    window_size: int,
    stride: int,
    start_index: int,
    limit_windows: int,
    include_partial: bool,
) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    i = start_index
    while i < len(gt_indices):
        frame_indices = gt_indices[i:i + window_size]
        if len(frame_indices) < window_size and not include_partial:
            break
        if len(frame_indices) == 0:
            break
        windows.append(
            WindowSpec(
                index=len(windows),
                start_frame=int(frame_indices[0]),
                end_frame=int(frame_indices[-1]),
                frame_indices=frame_indices.copy(),
            )
        )
        if limit_windows > 0 and len(windows) >= limit_windows:
            break
        i += stride
    return windows


def prepare_full_input(mp4: Path, gt_csv: Path, clip_dir: Path, info: dict[str, Any], gt_indices: np.ndarray) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_link = clip_dir / "clip.mp4"
    if clip_link.exists() or clip_link.is_symlink():
        clip_link.unlink()
    clip_link.symlink_to(mp4.resolve())
    timestamps = gt_indices.astype(np.float64) / float(info["fps"])
    np.save(clip_dir / "frame_indices.npy", gt_indices.astype(np.int32))
    np.save(clip_dir / "timestamps.npy", timestamps)
    meta = {
        "source_mp4": str(mp4.resolve()),
        "gt_csv": str(gt_csv.resolve()),
        "original_resolution": [info["width"], info["height"]],
        "original_fps": info["fps"],
        "duration_s": info["duration"],
        "num_raw_frames": info["nb_frames"],
        "sampled_fps": info["fps"],
        "sample_policy": "all_gt_csv_frame_idx",
        "num_sampled": int(len(gt_indices)),
        "first_frame_idx": int(gt_indices[0]),
        "last_frame_idx": int(gt_indices[-1]),
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def cut_window_mp4(src_mp4: Path, out_mp4: Path, start_frame: int, end_frame: int, *, dry_run: bool) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_mp4,
        "-vf", f"trim=start_frame={start_frame}:end_frame={end_frame + 1},setpts=PTS-STARTPTS",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    run(cmd, dry_run=dry_run)


def prepare_window_adapter_input(window_mp4: Path, clip_dir: Path, spec: WindowSpec, info: dict[str, Any]) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_link = clip_dir / "clip.mp4"
    if clip_link.exists() or clip_link.is_symlink():
        clip_link.unlink()
    clip_link.symlink_to(window_mp4.resolve())
    local_indices = np.arange(spec.num_frames, dtype=np.int32)
    np.save(clip_dir / "frame_indices.npy", local_indices)
    np.save(clip_dir / "timestamps.npy", local_indices.astype(np.float64) / float(info["fps"]))
    meta = {
        "source_mp4": str(window_mp4.resolve()),
        "original_resolution": [info["width"], info["height"]],
        "original_fps": info["fps"],
        "duration_s": spec.num_frames / float(info["fps"]),
        "num_raw_frames": spec.num_frames,
        "sampled_fps": info["fps"],
        "sample_policy": "local_window_all_frames",
        "num_sampled": spec.num_frames,
        "first_frame_idx": 0,
        "last_frame_idx": spec.num_frames - 1,
        "original_video_start_frame": spec.start_frame,
        "original_video_end_frame": spec.end_frame,
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def prepare_eval_input(clip_dir: Path, spec: WindowSpec, info: dict[str, Any], mp4: Path, gt_csv: Path) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    np.save(clip_dir / "frame_indices.npy", spec.frame_indices.astype(np.int32))
    np.save(clip_dir / "timestamps.npy", spec.frame_indices.astype(np.float64) / float(info["fps"]))
    meta = {
        "source_mp4": str(mp4.resolve()),
        "gt_csv": str(gt_csv.resolve()),
        "original_resolution": [info["width"], info["height"]],
        "original_fps": info["fps"],
        "sampled_fps": info["fps"],
        "sample_policy": "original_gt_frame_idx_for_window_eval",
        "num_sampled": spec.num_frames,
        "first_frame_idx": spec.start_frame,
        "last_frame_idx": spec.end_frame,
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def run_adapter(
    *,
    vipe_python: Path,
    adapter_script: Path,
    clip_dir: Path,
    out_root: Path,
    clip_uuid: str,
    env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> float:
    vipe_out = out_root / clip_uuid / "vipe"
    if not force and (vipe_out / "poses_c2w.npy").exists() and (vipe_out / "intrinsics.npy").exists():
        print(f"[skip vipe] {vipe_out} exists", flush=True)
        return read_quality_elapsed(vipe_out)
    return run(
        [
            vipe_python,
            adapter_script,
            "--method", "vipe",
            "--clip-dir", clip_dir,
            "--out-root", out_root,
            "--clip-uuid", clip_uuid,
        ],
        env=env,
        dry_run=dry_run,
    )


def read_quality_elapsed(method_dir: Path) -> float:
    q = method_dir / "quality.json"
    if not q.exists():
        return 0.0
    return float(json.loads(q.read_text()).get("elapsed_s", 0.0))




def write_jobs_jsonl(path: Path, jobs: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")


def run_resident_stage_worker_jobs(
    *,
    label: str,
    jobs: list[dict[str, str]],
    exp_dir: Path,
    vipe_python: Path,
    env: dict[str, str],
    cuda_visible_devices: str | None,
    gpu_sample_interval: float,
) -> dict[str, Any]:
    worker_script = POSE_ROOT / "scripts" / "vipe_resident_worker.py"
    jobs_path = exp_dir / "worker_jobs" / f"{label}.jsonl"
    report_path = exp_dir / "worker_reports" / f"{label}_report.json"
    gpu_csv = exp_dir / "gpu" / f"{label}_gpu.csv"
    log_path = exp_dir / "worker_logs" / f"{label}_stage_worker.log"
    write_jobs_jsonl(jobs_path, jobs)
    for out_path in (report_path.parent, gpu_csv.parent, log_path.parent):
        out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        vipe_python,
        worker_script,
        "stage-worker",
        "--jobs-jsonl", jobs_path,
        "--report-json", report_path,
        "--gpu-samples-csv", gpu_csv,
        "--vipe-repo", POSE_ROOT / "repo" / "vipe",
        "--gpu-sample-interval", str(gpu_sample_interval),
    ]
    gpu_index = first_gpu_index(cuda_visible_devices or env.get("CUDA_VISIBLE_DEVICES"))
    if gpu_index is not None:
        cmd.extend(["--gpu-index", gpu_index])

    print(f"[resident] starting {label} internal stage worker with {len(jobs)} job(s) on one visible GPU", flush=True)
    log_cmd(cmd)
    started = time.perf_counter()
    with log_path.open("w") as log_f:
        proc = subprocess.run([str(x) for x in cmd], env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    wall_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"resident stage worker failed with code {proc.returncode}; log={log_path}")
    report = json.loads(report_path.read_text())
    report["subprocess_wall_s"] = wall_s
    report["stage_worker_log"] = str(log_path)
    report["jobs_path"] = str(jobs_path)
    report["gpu_samples_csv"] = str(gpu_csv)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[resident] {label} internal stage worker exited after {wall_s:.2f}s; report={report_path}", flush=True)
    return report


def job_elapsed_with_worker_start(report: dict[str, Any], job_index: int) -> float:
    jobs = report.get("jobs", [])
    job_elapsed = float(jobs[job_index].get("job_total_s", 0.0))
    if job_index == 0:
        job_elapsed += float(report.get("worker_startup_s", 0.0))
    return job_elapsed


def comparison_mode_name(base_mode: str, runner: str) -> str:
    if runner == "resident":
        if base_mode == "per_window":
            return "resident_per_window"
        if base_mode == "clip_once":
            return "resident_clip_once_then_slice"
    return base_mode


def write_clip_once_slice(full_method_dir: Path, out_method_dir: Path, spec: WindowSpec) -> None:
    full = load(full_method_dir)
    positions_by_frame = {int(frame): i for i, frame in enumerate(full.frame_indices.tolist())}
    missing = [int(frame) for frame in spec.frame_indices if int(frame) not in positions_by_frame]
    if missing:
        raise ValueError(f"full clip ViPE output missing frame indices for {spec.window_id}: {missing[:10]}")
    pos = np.asarray([positions_by_frame[int(frame)] for frame in spec.frame_indices], dtype=np.int64)
    poses_c2w = full.poses_c2w[pos].astype(np.float64)
    poses_w2c = invert_se3_batch(poses_c2w)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)
    quality = dict(full.quality)
    quality.update(
        {
            "method": "vipe",
            "num_frames": int(len(pos)),
            "t_meaning": "sliced_from_full_clip_vipe",
            "source_mode": "clip_once_slice",
            "source_full_num_frames": int(len(full.frame_indices)),
            "source_full_elapsed_s": float(full.quality.get("elapsed_s", 0.0)),
            "window_start_frame": spec.start_frame,
            "window_end_frame": spec.end_frame,
            "scale_factor": scale_factor,
            "total_path_length_before_norm": total_path,
        }
    )
    save(
        out_method_dir,
        UnifiedOutput(
            intrinsics=full.intrinsics[pos].astype(np.float32),
            poses_c2w=poses_c2w,
            poses_w2c=poses_w2c,
            relative_poses=rel_norm,
            scale_factor=scale_factor,
            frame_indices=spec.frame_indices.astype(np.int32),
            timestamps=full.timestamps[pos].astype(np.float64),
            quality=quality,
        ),
    )


def run_eval(
    *,
    vipe_python: Path,
    eval_script: Path,
    clip_out: Path,
    clip_in: Path,
    gt_csv: Path,
    force: bool,
    dry_run: bool,
) -> None:
    if not force and (clip_out / "gt_eval.json").exists() and (clip_out / "gt_eval_axis_search.json").exists():
        print(f"[skip eval] {clip_out}/gt_eval.json exists", flush=True)
        return
    run(
        [
            vipe_python,
            eval_script,
            "--clip-out", clip_out,
            "--clip-in", clip_in,
            "--gt-csv", gt_csv,
            "--axis-search",
        ],
        dry_run=dry_run,
    )


def read_eval_row(mode: str, spec: WindowSpec, clip_out: Path, vipe_elapsed_s: float, wall_s: float) -> dict[str, Any]:
    eval_json = json.loads((clip_out / "gt_eval.json").read_text())
    sim3 = eval_json["methods"]["vipe"]["alignments"]["sim3"]
    intr = eval_json["methods"]["vipe"].get("intrinsics", {})
    axis_path = clip_out / "gt_eval_axis_search.json"
    axis = {}
    if axis_path.exists():
        axis_data = json.loads(axis_path.read_text())
        axis = axis_data.get("best_by_method", {}).get("vipe", {}).get("best_by_ate_rmse", {})
    return {
        "mode": mode,
        "window_id": spec.window_id,
        "start_frame": spec.start_frame,
        "end_frame": spec.end_frame,
        "num_frames": spec.num_frames,
        "vipe_elapsed_s": vipe_elapsed_s,
        "wall_s": wall_s,
        "ate_rmse_sim3": sim3.get("ate_rmse"),
        "ate_mean_sim3": sim3.get("ate_mean"),
        "rot_err_deg_mean_sim3": sim3.get("rot_err_deg_mean"),
        "rpe_rot_deg_mean_sim3": sim3.get("rpe_rot_deg_mean"),
        "rpe_trans_mean_sim3": sim3.get("rpe_trans_mean"),
        "scale_sim3": sim3.get("scale"),
        "fx_rel_mean": intr.get("fx_rel_mean"),
        "axis_transform": axis.get("axis_transform"),
        "axis_ate_rmse": axis.get("ate_rmse"),
        "axis_rpe_rot_deg_mean": axis.get("rpe_rot_deg_mean"),
        "axis_scale": axis.get("scale"),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)



def run_resident_comparison(
    *,
    args: argparse.Namespace,
    exp_dir: Path,
    mp4: Path,
    gt_csv: Path,
    stem: str,
    info: dict[str, Any],
    gt_indices: np.ndarray,
    windows: list[WindowSpec],
    env: dict[str, str],
) -> None:
    full_clip_id = f"{stem}_full"
    full_input = exp_dir / "inputs" / "full_clip" / full_clip_id
    full_output_root = exp_dir / "outputs" / "full_clip"
    full_method_dir = full_output_root / full_clip_id / "vipe"
    exp_dir.mkdir(parents=True, exist_ok=True)
    prepare_full_input(mp4, gt_csv, full_input, info, gt_indices)

    rows: list[dict[str, Any]] = []
    mode_a = comparison_mode_name("per_window", "resident")
    mode_b = comparison_mode_name("clip_once", "resident")
    timings: dict[str, Any] = {"runner": "resident", "comparison": {"A": mode_a, "B": mode_b}, "windows": {}, "worker_reports": {}}

    per_jobs: list[dict[str, str]] = []
    per_specs: list[WindowSpec] = []
    for spec in windows:
        print("=" * 60)
        print(f"[resident prepare window] {spec.window_id}")
        eval_input = exp_dir / "eval_inputs" / spec.window_id
        prepare_eval_input(eval_input, spec, info, mp4, gt_csv)

        window_mp4 = exp_dir / "window_mp4" / f"{spec.window_id}.mp4"
        adapter_input = exp_dir / "inputs" / "per_window" / spec.window_id
        if args.force or not window_mp4.exists():
            cut_window_mp4(mp4, window_mp4, spec.start_frame, spec.end_frame, dry_run=False)
        actual_frames = count_frames_exact(window_mp4)
        if actual_frames != spec.num_frames:
            raise RuntimeError(f"{window_mp4} has {actual_frames} frames, expected {spec.num_frames}")
        prepare_window_adapter_input(window_mp4, adapter_input, spec, info)
        per_jobs.append(
            {
                "clip_dir": str(adapter_input),
                "out_root": str(exp_dir / "outputs" / "per_window"),
                "clip_uuid": spec.window_id,
            }
        )
        per_specs.append(spec)

    print(f"[resident] phase 1/2: A={mode_a}. This worker must exit before B starts.", flush=True)
    per_report = run_resident_stage_worker_jobs(
        label="per_window",
        jobs=per_jobs,
        exp_dir=exp_dir,
        vipe_python=args.vipe_python.resolve(),
        env=env,
        cuda_visible_devices=args.cuda_visible_devices,
        gpu_sample_interval=args.gpu_sample_interval,
    )
    timings["worker_reports"]["per_window"] = str(exp_dir / "worker_reports" / "per_window_report.json")

    print(f"[resident] A worker closed; phase 2/2: B={mode_b}.", flush=True)
    clip_report = run_resident_stage_worker_jobs(
        label="clip_once",
        jobs=[{"clip_dir": str(full_input), "out_root": str(full_output_root), "clip_uuid": full_clip_id}],
        exp_dir=exp_dir,
        vipe_python=args.vipe_python.resolve(),
        env=env,
        cuda_visible_devices=args.cuda_visible_devices,
        gpu_sample_interval=args.gpu_sample_interval,
    )
    timings["worker_reports"]["clip_once"] = str(exp_dir / "worker_reports" / "clip_once_report.json")
    full_elapsed_s = job_elapsed_with_worker_start(clip_report, 0)
    timings["full_clip_elapsed_s"] = full_elapsed_s

    per_jobs_by_uuid = {job["clip_uuid"]: i for i, job in enumerate(per_report.get("jobs", []))}
    for spec in per_specs:
        eval_input = exp_dir / "eval_inputs" / spec.window_id
        per_window_output_root = exp_dir / "outputs" / "per_window"
        per_window_clip_out = per_window_output_root / spec.window_id
        per_idx = per_jobs_by_uuid[spec.window_id]
        per_elapsed_s = job_elapsed_with_worker_start(per_report, per_idx)
        per_wall_s = float(per_report["jobs"][per_idx].get("job_total_s", 0.0))
        run_eval(
            vipe_python=args.vipe_python.resolve(),
            eval_script=args.eval_script.resolve(),
            clip_out=per_window_clip_out,
            clip_in=eval_input,
            gt_csv=gt_csv,
            force=args.force,
            dry_run=False,
        )
        rows.append(read_eval_row(mode_a, spec, per_window_clip_out, per_elapsed_s, per_wall_s))

        clip_once_clip_out = exp_dir / "outputs" / "clip_once" / spec.window_id
        print(f"[clip_once] slicing full-clip resident output for {spec.window_id}", flush=True)
        if args.force or not (clip_once_clip_out / "vipe" / "poses_c2w.npy").exists():
            write_clip_once_slice(full_method_dir, clip_once_clip_out / "vipe", spec)
        run_eval(
            vipe_python=args.vipe_python.resolve(),
            eval_script=args.eval_script.resolve(),
            clip_out=clip_once_clip_out,
            clip_in=eval_input,
            gt_csv=gt_csv,
            force=args.force,
            dry_run=False,
        )
        rows.append(read_eval_row(mode_b, spec, clip_once_clip_out, full_elapsed_s, 0.0))
        timings["windows"][spec.window_id] = {
            "per_window_elapsed_s": per_elapsed_s,
            "per_window_job_s": per_wall_s,
            "clip_once_elapsed_s": full_elapsed_s,
        }

    (exp_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    write_summary(exp_dir, rows, full_elapsed_s)

def write_summary(exp_dir: Path, rows: list[dict[str, Any]], full_elapsed_s: float) -> None:
    summary_csv = exp_dir / "summary.csv"
    summary_md = exp_dir / "summary.md"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    modes = sorted({r["mode"] for r in rows})
    lines = [
        "# ViPE Window Mode Comparison",
        "",
        "## Aggregate",
        "",
        "| mode | windows | ViPE elapsed total | mean ATE Sim3 | mean RPE rot deg | mean axis ATE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        elapsed_total = full_elapsed_s if mode in {"clip_once", "resident_clip_once_then_slice"} else sum(float(r["vipe_elapsed_s"] or 0.0) for r in rs)
        mean_ate = float(np.mean([r["ate_rmse_sim3"] for r in rs if r["ate_rmse_sim3"] is not None]))
        mean_rpe_rot = float(np.mean([r["rpe_rot_deg_mean_sim3"] for r in rs if r["rpe_rot_deg_mean_sim3"] is not None]))
        axis_values = [r["axis_ate_rmse"] for r in rs if r["axis_ate_rmse"] is not None]
        mean_axis = float(np.mean(axis_values)) if axis_values else None
        lines.append(
            f"| {mode} | {len(rs)} | {fmt(elapsed_total, 2)} | {fmt(mean_ate)} | {fmt(mean_rpe_rot)} | {fmt(mean_axis)} |"
        )

    lines.extend([
        "",
        "## Per Window",
        "",
        "| window | mode | frames | ViPE elapsed | ATE Sim3 | RPE rot deg | axis ATE | axis |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for r in rows:
        lines.append(
            f"| {r['window_id']} | {r['mode']} | {r['num_frames']} | {fmt(r['vipe_elapsed_s'], 2)} | "
            f"{fmt(r['ate_rmse_sim3'])} | {fmt(r['rpe_rot_deg_mean_sim3'])} | "
            f"{fmt(r['axis_ate_rmse'])} | {r.get('axis_transform') or ''} |"
        )
    summary_md.write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {summary_csv}", flush=True)
    print(f"[summary] wrote {summary_md}", flush=True)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.mp4:
        mp4 = Path(args.mp4).resolve()
        stem = args.stem or mp4.name.removesuffix("_aligned.mp4").removesuffix(".mp4")
    else:
        if not args.stem:
            raise SystemExit("Either --stem or --mp4 is required")
        stem = args.stem
        mp4 = args.gt_root / f"{stem}_aligned.mp4"
    if args.gt_csv:
        gt_csv = Path(args.gt_csv).resolve()
    else:
        gt_csv = args.gt_root / f"telemetry_vehicle_nav_{stem}_aligned_processed.csv"
    if not mp4.exists():
        raise SystemExit(f"mp4 not found: {mp4}")
    if not gt_csv.exists():
        raise SystemExit(f"gt csv not found: {gt_csv}")
    return mp4, gt_csv, stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", help="Example: 2026-05-14_10-57-13")
    parser.add_argument("--mp4", type=Path, help="Optional explicit *_aligned.mp4 path")
    parser.add_argument("--gt-csv", type=Path, help="Optional explicit *_aligned_processed.csv path")
    parser.add_argument("--gt-root", type=Path, default=POSE_ROOT / "pose_demo" / "data")
    parser.add_argument("--output-root", type=Path, default=POSE_ROOT / "outputs" / "window_mode_compare")
    parser.add_argument("--window-size", type=int, default=81)
    parser.add_argument("--stride", type=int, default=None, help="Default: same as --window-size")
    parser.add_argument("--start-index", type=int, default=0, help="Start position in GT frame_indices")
    parser.add_argument("--limit-windows", type=int, default=3, help="0 means all windows")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--vipe-python", type=Path, default=POSE_ROOT / ".." / "envs" / "vipe" / "bin" / "python")
    parser.add_argument("--adapter-script", type=Path, default=POSE_ROOT / "adapter" / "scripts" / "run_adapter.py")
    parser.add_argument("--eval-script", type=Path, default=POSE_ROOT / "adapter" / "scripts" / "evaluate_with_gt.py")
    parser.add_argument("--vipe-model-root", type=Path, default=POSE_ROOT.parent / "models")
    parser.add_argument("--cuda-visible-devices", help="Optional one-GPU mask for local A/B testing, e.g. 0. Equivalent to CUDA_VISIBLE_DEVICES=0")
    parser.add_argument("--runner", choices=["subprocess", "resident"], default="subprocess")
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0, help="Resident worker GPU sampling interval in seconds; <=0 disables sampling")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.gt_root = args.gt_root.resolve()
    args.output_root = args.output_root.resolve()
    stride = args.stride or args.window_size
    if args.window_size <= 0 or stride <= 0:
        raise SystemExit("--window-size and --stride must be positive")

    mp4, gt_csv, stem = resolve_inputs(args)
    info = ffprobe_stream(mp4)
    if info["fps"] <= 0:
        raise SystemExit(f"could not determine fps for {mp4}")
    if info["nb_frames"] is None:
        info["nb_frames"] = count_frames_exact(mp4)
    gt_indices = read_gt_frame_indices(gt_csv)
    if len(gt_indices) > 1 and not np.all(np.diff(gt_indices) == 1):
        print("[warn] GT frame_idx is not contiguous; ffmpeg frame trimming assumes original frame numbers.", flush=True)

    windows = build_windows(
        gt_indices,
        window_size=args.window_size,
        stride=stride,
        start_index=args.start_index,
        limit_windows=args.limit_windows,
        include_partial=args.include_partial,
    )
    if not windows:
        raise SystemExit("no windows selected")

    exp_dir = args.output_root / f"{stem}_ws{args.window_size}_stride{stride}_start{args.start_index}"
    print(f"POSE_ROOT={POSE_ROOT}")
    print(f"mp4={mp4}")
    print(f"gt_csv={gt_csv}")
    print(f"video={info['width']}x{info['height']} fps={info['fps']:.3f} frames={info['nb_frames']}")
    print(f"experiment={exp_dir}")
    print(f"windows={len(windows)} window_size={args.window_size} stride={stride}")
    print(f"vipe_python={args.vipe_python.resolve()}")
    print(f"adapter_script={args.adapter_script.resolve()}")
    print(f"eval_script={args.eval_script.resolve()}")
    print(f"VIPE_MODEL_ROOT={args.vipe_model_root.resolve()}")
    print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices or os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} (resident A/B is serial; use one GPU)")
    print(f"runner={args.runner} gpu_sample_interval={args.gpu_sample_interval}")
    print(f"dry_run={args.dry_run} force={args.force}")
    print()

    env = os.environ.copy()
    env["TORCH_HOME"] = str(args.vipe_model_root.resolve() / "torch")
    env["HF_HOME"] = str(args.vipe_model_root.resolve() / "huggingface")
    env.setdefault("VIPE_REPO", str(POSE_ROOT / "repo" / "vipe"))
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if not args.dry_run:
        check_cuda_available(args.vipe_python.resolve(), env)

    if args.runner == "resident" and not args.dry_run:
        run_resident_comparison(
            args=args,
            exp_dir=exp_dir,
            mp4=mp4,
            gt_csv=gt_csv,
            stem=stem,
            info=info,
            gt_indices=gt_indices,
            windows=windows,
            env=env,
        )
        print("Done.")
        return 0

    full_clip_id = f"{stem}_full"
    full_input = exp_dir / "inputs" / "full_clip" / full_clip_id
    full_output_root = exp_dir / "outputs" / "full_clip"
    full_method_dir = full_output_root / full_clip_id / "vipe"

    if args.dry_run:
        print("[dry-run] selected windows:")
        for spec in windows:
            print(f"  {spec.window_id}: {spec.start_frame}..{spec.end_frame} ({spec.num_frames}f)")
        print("[dry-run] no files will be created and ViPE will not run")
        return 0

    exp_dir.mkdir(parents=True, exist_ok=True)
    prepare_full_input(mp4, gt_csv, full_input, info, gt_indices)
    print("[clip_once] running full-clip ViPE")
    full_wall_s = run_adapter(
        vipe_python=args.vipe_python.resolve(),
        adapter_script=args.adapter_script.resolve(),
        clip_dir=full_input,
        out_root=full_output_root,
        clip_uuid=full_clip_id,
        env=env,
        force=args.force,
        dry_run=False,
    )
    full_elapsed_s = read_quality_elapsed(full_method_dir) or full_wall_s

    rows: list[dict[str, Any]] = []
    timings: dict[str, Any] = {"full_clip_wall_s": full_wall_s, "full_clip_elapsed_s": full_elapsed_s, "windows": {}}

    for spec in windows:
        print("=" * 60)
        print(f"[window] {spec.window_id}")
        eval_input = exp_dir / "eval_inputs" / spec.window_id
        prepare_eval_input(eval_input, spec, info, mp4, gt_csv)

        window_mp4 = exp_dir / "window_mp4" / f"{spec.window_id}.mp4"
        adapter_input = exp_dir / "inputs" / "per_window" / spec.window_id
        if args.force or not window_mp4.exists():
            cut_window_mp4(mp4, window_mp4, spec.start_frame, spec.end_frame, dry_run=False)
        actual_frames = count_frames_exact(window_mp4)
        if actual_frames != spec.num_frames:
            raise RuntimeError(f"{window_mp4} has {actual_frames} frames, expected {spec.num_frames}")
        prepare_window_adapter_input(window_mp4, adapter_input, spec, info)

        per_window_output_root = exp_dir / "outputs" / "per_window"
        per_window_clip_out = per_window_output_root / spec.window_id
        print("[per_window] running window ViPE")
        per_wall_s = run_adapter(
            vipe_python=args.vipe_python.resolve(),
            adapter_script=args.adapter_script.resolve(),
            clip_dir=adapter_input,
            out_root=per_window_output_root,
            clip_uuid=spec.window_id,
            env=env,
            force=args.force,
            dry_run=False,
        )
        per_elapsed_s = read_quality_elapsed(per_window_clip_out / "vipe") or per_wall_s
        run_eval(
            vipe_python=args.vipe_python.resolve(),
            eval_script=args.eval_script.resolve(),
            clip_out=per_window_clip_out,
            clip_in=eval_input,
            gt_csv=gt_csv,
            force=args.force,
            dry_run=False,
        )
        rows.append(read_eval_row("per_window", spec, per_window_clip_out, per_elapsed_s, per_wall_s))

        clip_once_clip_out = exp_dir / "outputs" / "clip_once" / spec.window_id
        print("[clip_once] slicing full-clip ViPE output")
        if args.force or not (clip_once_clip_out / "vipe" / "poses_c2w.npy").exists():
            write_clip_once_slice(full_method_dir, clip_once_clip_out / "vipe", spec)
        run_eval(
            vipe_python=args.vipe_python.resolve(),
            eval_script=args.eval_script.resolve(),
            clip_out=clip_once_clip_out,
            clip_in=eval_input,
            gt_csv=gt_csv,
            force=args.force,
            dry_run=False,
        )
        rows.append(read_eval_row("clip_once", spec, clip_once_clip_out, full_elapsed_s, 0.0))
        timings["windows"][spec.window_id] = {
            "per_window_wall_s": per_wall_s,
            "per_window_elapsed_s": per_elapsed_s,
            "clip_once_elapsed_s": full_elapsed_s,
        }

    (exp_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    write_summary(exp_dir, rows, full_elapsed_s)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
