from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import summarize_vipe_timing_tree as timing_tree  # noqa: E402


def test_build_tree_computes_closed_residuals(tmp_path: Path) -> None:
    report = {
        "worker_startup_s": 2.0,
        "worker_total_s": 112.0,
        "jobs": [
            {
                "job_total_s": 100.0,
                "timings": {
                    "compose_config_s": 1.0,
                    "make_pipeline_s": 2.0,
                    "raw_stream_init_s": 3.0,
                    "pipeline_run_s": 90.0,
                    "artifact_convert_s": 2.0,
                    "pipeline_init_cache_process_s": 10.0,
                    "pipeline_slam_system_init_s": 1.0,
                    "slam_run_total_s": 60.0,
                    "pipeline_post_cache_depth_s": 5.0,
                    "pipeline_save_artifacts_s": 4.0,
                    "slam_build_components_s": 3.0,
                    "slam_pass1_s": 20.0,
                    "slam_backend_global_ba_1_s": 10.0,
                    "slam_backend_global_ba_2_s": 5.0,
                    "slam_pass2_s": 15.0,
                    "slam_extract_slam_map_s": 2.0,
                    "slam_precompute_features_pass1_s": 4.0,
                    "slam_sparse_tracks_track_image_s": 1.0,
                    "slam_motion_filter_check_s": 1.0,
                    "slam_add_keyframe_phase1_s": 2.0,
                    "slam_frontend_run_s": 3.0,
                    "slam_backend_run_early_s": 4.0,
                    "slam_backend_run_early_bundle_adjustment_s": 1.0,
                    "slam_backend_global_ba_1_bundle_adjustment_s": 7.0,
                    "slam_backend_global_ba_2_bundle_adjustment_s": 4.0,
                    "slam_precompute_features_pass2_s": 3.0,
                    "slam_add_keyframe_phase2_s": 2.0,
                    "inner_filler_compute_s": 6.0,
                },
            }
        ],
    }
    report_path = tmp_path / "worker_report.json"
    report_path.write_text(json.dumps(report))

    aggregate = timing_tree.aggregate_reports([report_path])
    root = timing_tree.build_tree(aggregate)
    rows = timing_tree.flatten_tree(root)
    by_path = {row["path"]: row for row in rows}

    assert by_path["vipe_stage_total/stage_residual"]["time_s"] == 10.0
    assert by_path["vipe_stage_total/jobs_total/job_total/job_unattributed"]["time_s"] == 2.0
    assert by_path["vipe_stage_total/jobs_total/job_total/pipeline_run/pipeline_unattributed"]["time_s"] == 10.0
    assert by_path[
        "vipe_stage_total/jobs_total/job_total/pipeline_run/slam_run_total/slam_unattributed"
    ]["time_s"] == 5.0
    assert by_path[
        "vipe_stage_total/jobs_total/job_total/pipeline_run/slam_run_total/backend_global_ba_1/backend_unattributed"
    ]["time_s"] == 3.0
    assert by_path["vipe_stage_total/jobs_total/job_total/pipeline_run"]["pct_of_parent"] == 90.0


def test_write_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "worker_report.json"
    report_path.write_text(json.dumps({"worker_startup_s": 1.0, "worker_total_s": 1.0, "jobs": []}))

    out_dir = tmp_path / "summary"
    timing_tree.write_summary([report_path], out_dir)

    assert (out_dir / "timing_tree.json").exists()
    assert (out_dir / "timing_tree.csv").exists()
    md = (out_dir / "timing_tree.md").read_text()
    assert "vipe_stage_total" in md
    assert "pct_of_parent" in md
