#!/usr/bin/env python3
"""Run ViPE jobs through an internal stage-style resident worker.

This mirrors the internal vLLM caption stage shape: stage_setup() runs once,
process_job() is called repeatedly in the same worker process, and destroy()
cleans up at the end. Each JSONL job has:
  {"clip_dir": "...", "out_root": "...", "clip_uuid": "..."}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

POSE_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = POSE_ROOT / "adapter"
sys.path.insert(0, str(ADAPTER_ROOT))

from pose_adapter.adapters.vipe_adapter import _find_artifact, _select_rows_by_indices  # noqa: E402
from pose_adapter.geometry import invert_se3_batch, normalize_max_translation, relative_to_first  # noqa: E402
from pose_adapter.schema import UnifiedOutput, save  # noqa: E402


CURRENT_JOB_TIMINGS: dict[str, Any] | None = None
CURRENT_BACKEND_CONTEXT: str | None = None


def now() -> float:
    return time.perf_counter()


def add_timing(name: str, elapsed: float) -> None:
    if CURRENT_JOB_TIMINGS is None:
        return
    CURRENT_JOB_TIMINGS[name] = float(CURRENT_JOB_TIMINGS.get(name, 0.0)) + float(elapsed)
    CURRENT_JOB_TIMINGS[f"{name}_count"] = int(CURRENT_JOB_TIMINGS.get(f"{name}_count", 0)) + 1


def set_timing(name: str, value: float) -> None:
    if CURRENT_JOB_TIMINGS is None:
        return
    CURRENT_JOB_TIMINGS[name] = float(value)


def residual(parent: float, *children: float) -> float:
    value = float(parent) - sum(float(child) for child in children)
    if abs(value) < 1e-6:
        return 0.0
    return value


def _pick_timing(timings: dict[str, Any], key: str) -> float | None:
    value = timings.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize_job_report(job_report: dict[str, Any]) -> dict[str, Any]:
    timings = job_report.get("timings") if isinstance(job_report.get("timings"), dict) else {}
    summary: dict[str, Any] = {
        "clip_uuid": job_report.get("clip_uuid"),
        "input_mode": job_report.get("input_mode"),
        "job_total_s": job_report.get("job_total_s"),
        "pipeline_run_s": _pick_timing(timings, "pipeline_run_s"),
        "artifact_convert_s": _pick_timing(timings, "artifact_convert_s"),
        "status": job_report.get("status") or ("error" if job_report.get("error") else "ok"),
    }
    if job_report.get("error"):
        summary["error"] = job_report.get("error")
    return {key: value for key, value in summary.items() if value is not None}


def summarize_stage_report(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "mode",
        "job_count",
        "planned_job_count",
        "worker_startup_s",
        "worker_total_s",
        "jobs_total_s",
        "stage_residual_s",
        "cuda_visible_devices",
        "gpu_index",
    ):
        if key in report:
            summary[key] = report[key]
    summary["jobs"] = [summarize_job_report(job) for job in report.get("jobs", []) if isinstance(job, dict)]
    return summary


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def read_jobs(path: Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            for key in ("clip_dir", "out_root", "clip_uuid"):
                if key not in job:
                    raise ValueError(f"job missing {key}: {job}")
            jobs.append({key: str(job[key]) for key in ("clip_dir", "out_root", "clip_uuid")})
    return jobs


def start_gpu_sampler(path: Path, gpu_index: str | None, interval_s: float) -> tuple[threading.Event, threading.Thread | None]:
    if not gpu_index or interval_s <= 0:
        return threading.Event(), None

    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def sample() -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_time_s", "timestamp", "gpu_index", "util_gpu_pct", "memory_used_mib", "power_w"])
            while not stop.is_set():
                sample_t = time.time()
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits",
                            "-i",
                            gpu_index,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    for raw in result.stdout.strip().splitlines():
                        cols = [x.strip() for x in raw.split(",")]
                        if len(cols) >= 5:
                            writer.writerow([f"{sample_t:.3f}", *cols[:5]])
                except Exception as exc:
                    writer.writerow([f"{sample_t:.3f}", "ERROR", gpu_index, "", "", str(exc).replace("\n", " ")])
                f.flush()
                stop.wait(interval_s)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return stop, thread


def install_timing_patches(report: dict[str, Any]) -> None:
    import pickle

    import torch
    import vipe.pipeline.default as default_pipeline
    import vipe.pipeline.processors as processors
    import vipe.slam.components.backend as backend_mod
    import vipe.slam.components.buffer as buffer_mod
    import vipe.slam.components.inner_filler as inner_filler_mod
    import vipe.slam.system as slam_system

    model_cache: dict[tuple[Any, ...], Any] = {}
    model_init = report.setdefault("model_init_s", {})

    orig_pipeline_run = default_pipeline.DefaultAnnotationPipeline.run
    orig_geocalib = processors.GeoCalib
    orig_droid = slam_system.DroidNet
    orig_depth = slam_system.make_depth_model
    orig_build_components = slam_system.SLAMSystem._build_components
    orig_backend_run = backend_mod.SLAMBackend.run
    orig_backend_run_if_needed = backend_mod.SLAMBackend.run_if_necessary
    orig_inner_compute = inner_filler_mod.InnerFiller.compute
    orig_ba = buffer_mod.GraphBuffer.bundle_adjustment

    def timed_model(key: tuple[Any, ...], label: str, factory):
        if key not in model_cache:
            sync_cuda()
            t0 = now()
            model_cache[key] = factory()
            sync_cuda()
            model_init[label] = float(model_init.get(label, 0.0)) + now() - t0
        return model_cache[key]

    def cached_geocalib(*args, **kwargs):
        label = f"geocalib:{kwargs.get('weights', args[0] if args else 'default')}"
        return timed_model(("geocalib", tuple(args), tuple(sorted(kwargs.items()))), label, lambda: orig_geocalib(*args, **kwargs))

    def cached_droid(*args, **kwargs):
        return timed_model(("droid", tuple(args), tuple(sorted(kwargs.items()))), "droidnet", lambda: orig_droid(*args, **kwargs))

    def cached_depth(*args, **kwargs):
        label = f"depth:{args[0] if args else kwargs.get('model_name', 'default')}"
        return timed_model(("depth", tuple(args), tuple(sorted(kwargs.items()))), label, lambda: orig_depth(*args, **kwargs))

    def timed_call(name: str, func, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return func(*args, **kwargs)
        finally:
            sync_cuda()
            add_timing(name, now() - t0)

    def timed_pipeline_run(self, video_data):
        if isinstance(video_data, default_pipeline.MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [default_pipeline.io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()
        else:
            assert isinstance(video_data, default_pipeline.VideoStream)
            video_streams = [video_data]
            artifact_paths = [default_pipeline.io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = default_pipeline.AnnotationPipelineOutput()
        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            default_pipeline.logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        t0 = now()
        slam_streams = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]
        add_timing("pipeline_init_cache_process_s", now() - t0)

        t0 = now()
        slam_pipeline = default_pipeline.SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        add_timing("pipeline_slam_system_init_s", now() - t0)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        t0 = now()
        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]
        add_timing("pipeline_post_cache_depth_s", now() - t0)

        t0 = now()
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                default_pipeline.logger.info(f"Saving artifacts to {artifact_path}")
                default_pipeline.io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                default_pipeline.save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                default_pipeline.logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)
        add_timing("pipeline_save_artifacts_s", now() - t0)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams
        return annotate_output

    def timed_build_components(self, *args, **kwargs):
        return timed_call("slam_build_components_s", orig_build_components, self, *args, **kwargs)

    @torch.no_grad()
    def timed_slam_run(self, video_streams, rig=None, camera_type=slam_system.CameraType.PINHOLE):
        global CURRENT_BACKEND_CONTEXT

        sync_cuda()
        slam_t0 = now()
        try:
            assert len(video_streams) > 0
            resizers = [slam_system.StandardResizeStreamProcessor() for _ in video_streams]
            video_streams = [
                slam_system.ProcessedVideoStream(video_stream, [resizer])
                for video_stream, resizer in zip(video_streams, resizers)
            ]

            frame_size = video_streams[0].frame_size()
            total_n_frames = len(video_streams[0])
            for vs in video_streams:
                assert vs.frame_size() == frame_size
                assert len(vs) == total_n_frames

            if rig is None:
                assert len(video_streams) == 1, "Need rig for multiple views"
                rig = slam_system.SE3.Identity(1)
            self.rig = rig

            self.config.update(
                {
                    "height": frame_size[0],
                    "width": frame_size[1],
                    "n_views": len(video_streams),
                    "has_init_pose": slam_system.FrameAttribute.POSE in video_streams[0].attributes(),
                    "camera_type": camera_type,
                }
            )

            self._build_components()

            if self.visualize:
                slam_system.rr.init("ViPE Visualization", spawn=True, recording_id=slam_system.uuid.uuid4())
                slam_system.rr.log("world", slam_system.rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

            pass_t0 = now()
            frame_data_list: list[slam_system.VideoFrame]
            frame_idx: int = 0
            for frame_idx, frame_data_list in slam_system.pbar(
                enumerate(zip(*video_streams)), desc="SLAM Pass (1/2)", total=total_n_frames
            ):
                images, buffer_masks = timed_call("slam_precompute_features_pass1_s", self._precompute_features, frame_data_list)
                timed_call("slam_sparse_tracks_track_image_s", self.sparse_tracks.track_image, frame_data_list)
                should_keyframe = timed_call("slam_motion_filter_check_s", self.motion_filter.check, images, buffer_masks)

                if should_keyframe or frame_idx == total_n_frames - 1:
                    is_keyframe = True
                    timed_call("slam_add_keyframe_phase1_s", self._add_keyframe, frame_idx, images, buffer_masks, frame_data_list, phase=1)
                else:
                    is_keyframe = False

                timed_call("slam_frontend_run_s", self.frontend.run)

                if self.visualize:
                    self.buffer.log(self.config.map_filter_thresh)
                    self.frontend.graph.log()

                if self.buffer.n_frames in self.config.frontend_backend_iters and is_keyframe:
                    old_context = CURRENT_BACKEND_CONTEXT
                    CURRENT_BACKEND_CONTEXT = "slam_backend_run_early"
                    try:
                        self.backend.run_if_necessary(5, log=self.visualize)
                    finally:
                        CURRENT_BACKEND_CONTEXT = old_context
            add_timing("slam_pass1_s", now() - pass_t0)

            if self.visualize:
                self.buffer.log_tracks()

            old_context = CURRENT_BACKEND_CONTEXT
            CURRENT_BACKEND_CONTEXT = "slam_backend_global_ba_1"
            try:
                self.backend.run(7, log=self.visualize)
            finally:
                CURRENT_BACKEND_CONTEXT = old_context

            old_context = CURRENT_BACKEND_CONTEXT
            CURRENT_BACKEND_CONTEXT = "slam_backend_global_ba_2"
            try:
                self.backend.run(self.config.backend_iters, update_depth=False, log=self.visualize)
            finally:
                CURRENT_BACKEND_CONTEXT = old_context

            self.inner_filler.set_start_idx(self.buffer.n_frames)
            pass_t0 = now()
            for frame_idx, frame_data_list in slam_system.pbar(
                enumerate(zip(*video_streams)), desc="SLAM Pass (2/2)", total=total_n_frames
            ):
                images, buffer_masks = timed_call("slam_precompute_features_pass2_s", self._precompute_features, frame_data_list)
                timed_call("slam_add_keyframe_phase2_s", self._add_keyframe, frame_idx, images, buffer_masks, frame_data_list, phase=2)
                if self.inner_filler.check() or frame_idx == total_n_frames - 1:
                    self.inner_filler.compute()
            add_timing("slam_pass2_s", now() - pass_t0)

            filled_return = self.inner_filler.get_result()
            if filled_return.poses.shape[0] != total_n_frames:
                raise ValueError("Your video might be malformed. Try using streams.cached=true in the config.")

            if self.visualize:
                self._log_final(video_streams, filled_return)

            t0 = now()
            slam_map = self.buffer.extract_slam_map(filter_thresh=self.config.map_filter_thresh)
            slam_map.backend_graph = self.backend.last_graph
            add_timing("slam_extract_slam_map_s", now() - t0)

            original_intrinsics = torch.stack(
                [resizer.recover_intrinsics(self.buffer.intrinsics[v]) for v, resizer in enumerate(resizers)]
            )

            return slam_system.SLAMOutput(
                trajectory=filled_return.poses.inv(),
                intrinsics=original_intrinsics,
                rig=slam_system.SE3(self.buffer.rig.clone()),
                slam_map=slam_map,
            )
        finally:
            sync_cuda()
            add_timing("slam_run_total_s", now() - slam_t0)

    def timed_backend_run(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_backend_run(self, *args, **kwargs)
        finally:
            sync_cuda()
            elapsed = now() - t0
            add_timing("backend_run_s", elapsed)
            if CURRENT_BACKEND_CONTEXT:
                add_timing(f"{CURRENT_BACKEND_CONTEXT}_s", elapsed)

    def timed_backend_run_if_needed(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_backend_run_if_needed(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("backend_run_if_necessary_s", now() - t0)

    def timed_inner_compute(self, *args, **kwargs):
        return timed_call("inner_filler_compute_s", orig_inner_compute, self, *args, **kwargs)

    def timed_ba(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_ba(self, *args, **kwargs)
        finally:
            sync_cuda()
            elapsed = now() - t0
            add_timing("bundle_adjustment_s", elapsed)
            if CURRENT_BACKEND_CONTEXT:
                add_timing(f"{CURRENT_BACKEND_CONTEXT}_bundle_adjustment_s", elapsed)

    processors.GeoCalib = cached_geocalib
    slam_system.DroidNet = cached_droid
    slam_system.make_depth_model = cached_depth
    default_pipeline.DefaultAnnotationPipeline.run = timed_pipeline_run
    slam_system.SLAMSystem._build_components = timed_build_components
    slam_system.SLAMSystem.run = timed_slam_run
    backend_mod.SLAMBackend.run = timed_backend_run
    backend_mod.SLAMBackend.run_if_necessary = timed_backend_run_if_needed
    inner_filler_mod.InnerFiller.compute = timed_inner_compute
    buffer_mod.GraphBuffer.bundle_adjustment = timed_ba


def detect_input_mode(clip_dir: Path) -> tuple[str, Path]:
    images_dir = clip_dir / "images"
    if images_dir.is_dir() and any(images_dir.glob("frame_*.png")):
        return "frame_dir", images_dir
    clip_mp4 = clip_dir / "clip.mp4"
    if clip_mp4.exists():
        return "clip_mp4", clip_mp4
    raise FileNotFoundError(f"expected either {images_dir}/frame_*.png or {clip_mp4}")


def compose_vipe_config(vipe_repo: Path, input_mode: str, input_path: Path, output_path: Path):
    import hydra

    if input_mode == "frame_dir":
        stream_override = "streams=frame_dir_stream"
    elif input_mode == "clip_mp4":
        stream_override = "streams=raw_mp4_stream"
    else:
        raise ValueError(f"unsupported ViPE input mode: {input_mode}")

    with hydra.initialize_config_dir(config_dir=str(vipe_repo / "configs"), version_base=None):
        return hydra.compose(
            config_name="default",
            overrides=[
                "pipeline=default",
                stream_override,
                f"streams.base_path={input_path}",
                f"pipeline.output.path={output_path}",
                "pipeline.output.save_artifacts=true",
                "pipeline.output.save_viz=false",
                "pipeline.post.depth_align_model=null",
            ],
        )


def convert_artifacts(tmp: Path, clip_dir: Path, output_dir: Path, elapsed_s: float, extra_quality: dict[str, Any]) -> None:
    with (clip_dir / "meta.json").open() as f:
        meta = json.load(f)
    sparse_wanted = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
    timestamps = np.load(clip_dir / "timestamps.npy")
    source_indices = np.asarray(meta.get("source_frame_indices", sparse_wanted.tolist()), dtype=np.int32)
    if len(source_indices) != len(sparse_wanted):
        raise ValueError(
            f"source_frame_indices length {len(source_indices)} does not match sparse frame_indices {len(sparse_wanted)}"
        )
    orig_w, orig_h = meta["original_resolution"]

    pose_npz = _find_artifact(tmp, "pose", ".npz")
    intr_npz = _find_artifact(tmp, "intrinsics", ".npz")

    pose_data = np.load(pose_npz)
    intr_data = np.load(intr_npz)
    pose_sel, pose_fallback = _select_rows_by_indices(pose_data["data"], pose_data["inds"], sparse_wanted)
    intr_sel, intr_fallback = _select_rows_by_indices(intr_data["data"], intr_data["inds"], sparse_wanted)

    poses_c2w = pose_sel.astype(np.float64)
    poses_w2c = invert_se3_batch(poses_c2w)
    rel = relative_to_first(poses_c2w)
    rel_norm, scale_factor, total_path = normalize_max_translation(rel)

    warnings: list[str] = []
    if pose_fallback:
        warnings.append(f"vipe pose nearest-neighbour fallback at {len(pose_fallback)} positions")
    if intr_fallback:
        warnings.append(f"vipe intrinsics nearest-neighbour fallback at {len(intr_fallback)} positions")

    quality = {
        "method": "vipe",
        "status": "ok",
        "pose_convention": "c2w",
        "camera_coordinate": "opencv",
        "camera_model": "pinhole",
        "scale_status": "normalized",
        "scale_rule": "max_translation_norm",
        "scale_factor": scale_factor,
        "total_path_length_before_norm": total_path,
        "resolution_reference": [orig_w, orig_h],
        "intrinsics_shared_per_frame": False,
        "num_frames": int(poses_c2w.shape[0]),
        "frame_source_fps": meta["sampled_fps"],
        "original_fps": meta["original_fps"],
        "t_meaning": "sampled_frames",
        "elapsed_s": float(elapsed_s),
        "warnings": warnings,
        "sparse_frame_indices": sparse_wanted.astype(int).tolist(),
        "source_frame_indices": source_indices.astype(int).tolist(),
        **extra_quality,
    }

    save(
        output_dir,
        UnifiedOutput(
            intrinsics=intr_sel.astype(np.float32),
            poses_c2w=poses_c2w,
            poses_w2c=poses_w2c,
            relative_poses=rel_norm,
            scale_factor=scale_factor,
            frame_indices=source_indices,
            timestamps=timestamps,
            quality=quality,
        ),
    )


def run_one_job(job: dict[str, str], vipe_repo: Path, *, report_level: str = "summary") -> dict[str, Any]:
    global CURRENT_JOB_TIMINGS

    clip_dir = Path(job["clip_dir"]).resolve()
    out_root = Path(job["out_root"]).resolve()
    clip_uuid = job["clip_uuid"]
    output_dir = out_root / clip_uuid / "vipe"
    output_dir.mkdir(parents=True, exist_ok=True)

    job_report: dict[str, Any] = {
        "clip_uuid": clip_uuid,
        "clip_dir": str(clip_dir),
        "output_dir": str(output_dir),
        "timings": {},
    }
    CURRENT_JOB_TIMINGS = job_report["timings"]
    job_t0 = now()
    try:
        with tempfile.TemporaryDirectory(prefix="vipe_resident_out_") as tmpdir:
            tmp = Path(tmpdir)

            t0 = now()
            input_mode, input_path = detect_input_mode(clip_dir)
            job_report["input_mode"] = input_mode
            job_report["input_path"] = str(input_path)
            cfg = compose_vipe_config(vipe_repo, input_mode, input_path, tmp)
            job_report["timings"]["compose_config_s"] = now() - t0

            t0 = now()
            from vipe.pipeline import make_pipeline
            from vipe.streams.frame_dir_stream import FrameDirStream
            from vipe.streams.raw_mp4_stream import RawMp4Stream

            pipeline = make_pipeline(cfg.pipeline)
            job_report["timings"]["make_pipeline_s"] = now() - t0

            t0 = now()
            if input_mode == "frame_dir":
                stream = FrameDirStream(input_path)
                job_report["timings"]["frame_dir_stream_init_s"] = now() - t0
            else:
                stream = RawMp4Stream(input_path)
                job_report["timings"]["raw_stream_init_s"] = now() - t0

            sync_cuda()
            t0 = now()
            pipeline.run(stream)
            sync_cuda()
            job_report["timings"]["pipeline_run_s"] = now() - t0

            elapsed = now() - job_t0
            t0 = now()
            convert_artifacts(
                tmp,
                clip_dir,
                output_dir,
                elapsed_s=elapsed,
                extra_quality=(
                    {
                        "runner": "resident_worker",
                        "worker_clip_uuid": clip_uuid,
                        "worker_input_mode": job_report.get("input_mode"),
                        "worker_input_path": job_report.get("input_path"),
                        "worker_timings": job_report["timings"],
                    }
                    if report_level == "debug"
                    else {
                        "runner": "resident_worker",
                        "worker_summary": summarize_job_report(job_report),
                    }
                ),
            )
            job_report["timings"]["artifact_convert_s"] = now() - t0
    finally:
        CURRENT_JOB_TIMINGS = None

    job_report["job_total_s"] = now() - job_t0
    timings = job_report["timings"]
    timings["pipeline_unattributed_s"] = residual(
        timings.get("pipeline_run_s", 0.0),
        timings.get("pipeline_init_cache_process_s", 0.0),
        timings.get("pipeline_slam_system_init_s", 0.0),
        timings.get("slam_run_total_s", 0.0),
        timings.get("pipeline_post_cache_depth_s", 0.0),
        timings.get("pipeline_save_artifacts_s", 0.0),
    )
    timings["slam_unattributed_s"] = residual(
        timings.get("slam_run_total_s", 0.0),
        timings.get("slam_build_components_s", 0.0),
        timings.get("slam_pass1_s", 0.0),
        timings.get("slam_backend_global_ba_1_s", 0.0),
        timings.get("slam_backend_global_ba_2_s", 0.0),
        timings.get("slam_pass2_s", 0.0),
        timings.get("slam_extract_slam_map_s", 0.0),
    )
    timings["slam_pass1_unattributed_s"] = residual(
        timings.get("slam_pass1_s", 0.0),
        timings.get("slam_precompute_features_pass1_s", 0.0),
        timings.get("slam_sparse_tracks_track_image_s", 0.0),
        timings.get("slam_motion_filter_check_s", 0.0),
        timings.get("slam_add_keyframe_phase1_s", 0.0),
        timings.get("slam_frontend_run_s", 0.0),
        timings.get("slam_backend_run_early_s", 0.0),
    )
    timings["slam_pass2_unattributed_s"] = residual(
        timings.get("slam_pass2_s", 0.0),
        timings.get("slam_precompute_features_pass2_s", 0.0),
        timings.get("slam_add_keyframe_phase2_s", 0.0),
        timings.get("inner_filler_compute_s", 0.0),
    )
    for prefix in ("slam_backend_run_early", "slam_backend_global_ba_1", "slam_backend_global_ba_2"):
        timings[f"{prefix}_backend_unattributed_s"] = residual(
            timings.get(f"{prefix}_s", 0.0),
            timings.get(f"{prefix}_bundle_adjustment_s", 0.0),
        )
    timings["job_unattributed_s"] = residual(
        job_report["job_total_s"],
        timings.get("compose_config_s", 0.0),
        timings.get("make_pipeline_s", 0.0),
        timings.get("raw_stream_init_s", 0.0),
        timings.get("frame_dir_stream_init_s", 0.0),
        timings.get("pipeline_run_s", 0.0),
        timings.get("artifact_convert_s", 0.0),
    )
    return job_report




class VipeResidentStageWorker:
    """Small stage-like worker: setup once, process many jobs, destroy once."""

    def __init__(
        self,
        *,
        report_json: Path,
        gpu_samples_csv: Path | None,
        vipe_repo: Path,
        gpu_index: str | None,
        gpu_sample_interval: float,
        report_level: str = "summary",
    ) -> None:
        self.report_json = report_json
        self.gpu_samples_csv = gpu_samples_csv
        self.vipe_repo = vipe_repo.resolve()
        self.gpu_index = gpu_index
        self.gpu_sample_interval = gpu_sample_interval
        self.report_level = report_level
        self.worker_t0 = now()
        self.stop_gpu: threading.Event | None = None
        self.gpu_thread: threading.Thread | None = None
        self.report: dict[str, Any] = {
            "mode": "internal_stage_worker",
            "report_json": str(report_json.resolve()),
            "vipe_repo": str(self.vipe_repo),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_index": gpu_index,
            "report_level": report_level,
            "jobs": [],
        }

    def stage_setup(self) -> None:
        setup_t0 = now()
        sys.path.insert(0, str(self.vipe_repo))
        os.chdir(self.vipe_repo)

        import_t0 = now()
        import torch  # noqa: F401
        import vipe.pipeline.processors  # noqa: F401
        import vipe.slam.system  # noqa: F401

        self.report["worker_import_s"] = now() - import_t0

        patch_t0 = now()
        install_timing_patches(self.report)
        self.report["worker_patch_s"] = now() - patch_t0

        self.stop_gpu, self.gpu_thread = start_gpu_sampler(
            self.gpu_samples_csv,
            self.gpu_index,
            self.gpu_sample_interval,
        )
        self.report["stage_setup_s"] = now() - setup_t0
        self.report["worker_startup_s"] = now() - self.worker_t0
        self.report["worker_started_wall_time_s"] = time.time()
        self._write_report()

    def process_job(self, job: dict[str, str]) -> dict[str, Any]:
        job_idx = len(self.report["jobs"]) + 1
        print(f"[resident-stage-worker] process_job {job_idx}: {job['clip_uuid']}", flush=True)
        job_report = run_one_job(job, self.vipe_repo, report_level=self.report_level)
        self.report["jobs"].append(job_report if self.report_level == "debug" else summarize_job_report(job_report))
        self.report["job_count"] = len(self.report["jobs"])
        self._write_report()
        return job_report

    def destroy(self) -> None:
        if self.stop_gpu is not None:
            self.stop_gpu.set()
        if self.gpu_thread is not None:
            self.gpu_thread.join(timeout=5)
        sync_cuda()
        self.report["worker_total_s"] = now() - self.worker_t0
        jobs_total = sum(float(job.get("job_total_s", 0.0)) for job in self.report.get("jobs", []))
        self.report["jobs_total_s"] = jobs_total
        self.report["stage_residual_s"] = residual(
            self.report["worker_total_s"],
            self.report.get("worker_startup_s", 0.0),
            jobs_total,
        )
        self.report["worker_finished_wall_time_s"] = time.time()
        self._write_report()
        print(f"[resident-stage-worker] wrote {self.report_json}", flush=True)

    def _write_report(self) -> None:
        self.report_json.parent.mkdir(parents=True, exist_ok=True)
        payload = self.report if self.report_level == "debug" else summarize_stage_report(self.report)
        self.report_json.write_text(json.dumps(payload, indent=2))


def run_stage_worker(args: argparse.Namespace) -> int:
    jobs = read_jobs(args.jobs_jsonl)
    worker = VipeResidentStageWorker(
        report_json=args.report_json,
        gpu_samples_csv=args.gpu_samples_csv,
        vipe_repo=args.vipe_repo,
        gpu_index=args.gpu_index,
        gpu_sample_interval=args.gpu_sample_interval,
        report_level=args.report_level,
    )
    worker.report["jobs_jsonl"] = str(args.jobs_jsonl.resolve())
    worker.report["planned_job_count"] = len(jobs)
    try:
        worker.stage_setup()
        for job in jobs:
            worker.process_job(job)
    finally:
        worker.destroy()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="stage-worker", help="Internal stage-style resident worker command")
    parser.add_argument("--jobs-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--gpu-samples-csv", type=Path)
    parser.add_argument("--vipe-repo", type=Path, default=POSE_ROOT / "repo" / "vipe")
    parser.add_argument("--gpu-index", help="Physical GPU index for nvidia-smi sampling, e.g. 0")
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--report-level", choices=("summary", "debug"), default=os.environ.get("SKFP_REPORT_LEVEL", "summary"))
    args = parser.parse_args()
    if args.command != "stage-worker":
        raise SystemExit(f"unknown command: {args.command}")
    return run_stage_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
