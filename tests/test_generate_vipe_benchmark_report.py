from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import generate_vipe_benchmark_report as report  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_report_rows_from_experiment_dir(tmp_path: Path) -> None:
    exp = tmp_path / "demo_ws81_stride81_start0"
    write_csv(
        exp / "summary.csv",
        [
            {
                "mode": "resident_per_window",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 10.0,
                "wall_s": 9.0,
                "ate_rmse_sim3": 0.1,
                "rpe_rot_deg_mean_sim3": 0.2,
                "axis_ate_rmse": 0.3,
            },
            {
                "mode": "resident_clip_once_then_slice",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 20.0,
                "wall_s": 0.0,
                "ate_rmse_sim3": 0.4,
                "rpe_rot_deg_mean_sim3": 0.5,
                "axis_ate_rmse": 0.6,
            },
        ],
    )
    (exp / "timings.json").write_text(
        json.dumps(
            {
                "runner": "resident",
                "worker_reports": {
                    "per_window": str(exp / "worker_reports" / "per_window_report.json"),
                    "clip_once": str(exp / "worker_reports" / "clip_once_report.json"),
                },
            }
        )
    )
    for name, pipeline_run in (("per_window", 100.0), ("clip_once", 200.0)):
        worker_report = {
            "worker_total_s": pipeline_run + 10.0,
            "subprocess_wall_s": pipeline_run + 11.0,
            "jobs_total_s": pipeline_run + 5.0,
            "jobs": [{"timings": {"pipeline_run_s": pipeline_run}}],
        }
        path = exp / "worker_reports" / f"{name}_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(worker_report))
    meta_dir = exp / "inputs" / "per_window" / "window_0000_0_80"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"duration_s": 2.7, "num_raw_frames": 81}))

    rows, context = report.build_report(exp)

    assert context["input_video_count"] == 1
    assert context["success_windows"] == 1
    assert rows[0]["scheme"] == "C_resident_clip_once_then_slice"
    assert rows[0]["workers"] == 1
    assert rows[0]["resident"] == "yes"
    assert rows[0]["success_windows"] == 1
    assert rows[0]["processed_success_duration_s"] == 2.7
    assert rows[0]["pipeline_run_min"] == 200.0 / 60.0


def test_scheme_names_map_to_final_abc_names() -> None:
    assert report.final_scheme_name("per_window") == "A_subprocess_per_window"
    assert report.final_scheme_name("resident_per_window") == "B_resident_per_window"
    assert report.final_scheme_name("resident_clip_once_then_slice") == "C_resident_clip_once_then_slice"


def test_subprocess_report_keeps_only_a_scheme(tmp_path: Path) -> None:
    exp = tmp_path / "demo_ws81_stride81_start0"
    write_csv(
        exp / "summary.csv",
        [
            {
                "mode": "per_window",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 10.0,
                "wall_s": 9.0,
                "ate_rmse_sim3": 0.1,
                "rpe_rot_deg_mean_sim3": 0.2,
                "axis_ate_rmse": 0.3,
            },
            {
                "mode": "clip_once",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 20.0,
                "wall_s": 0.0,
                "ate_rmse_sim3": 0.4,
                "rpe_rot_deg_mean_sim3": 0.5,
                "axis_ate_rmse": 0.6,
            },
        ],
    )
    (exp / "timings.json").write_text(json.dumps({"windows": {}}))
    meta_dir = exp / "inputs" / "per_window" / "window_0000_0_80"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"duration_s": 2.7, "num_raw_frames": 81}))

    rows, _ = report.build_report(exp)

    assert [row["scheme"] for row in rows] == ["A_subprocess_per_window"]
    assert rows[0]["workers"] == 1
    assert rows[0]["vipe_stage_cumulative_s"] == 10.0
    assert rows[0]["vipe_stage_wall_min"] == 9.0 / 60.0
    assert rows[0]["pipeline_run_min"] == 10.0 / 60.0


def test_write_report_creates_final_scheme_folders(tmp_path: Path) -> None:
    exp = tmp_path / "demo_ws81_stride81_start0"
    write_csv(
        exp / "summary.csv",
        [
            {
                "mode": "per_window",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 10.0,
                "wall_s": 9.0,
                "ate_rmse_sim3": 0.1,
                "rpe_rot_deg_mean_sim3": 0.2,
                "axis_ate_rmse": 0.3,
            },
            {
                "mode": "clip_once",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 20.0,
                "wall_s": 0.0,
                "ate_rmse_sim3": 0.4,
                "rpe_rot_deg_mean_sim3": 0.5,
                "axis_ate_rmse": 0.6,
            },
        ],
    )
    (exp / "timings.json").write_text(json.dumps({"windows": {}}))
    meta_dir = exp / "inputs" / "per_window" / "window_0000_0_80"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"duration_s": 2.7, "num_raw_frames": 81}))

    report.write_report(exp)

    scheme_dir = exp / "schemes" / "A_subprocess_per_window"
    assert (scheme_dir / "scheme_report.md").exists()
    assert (scheme_dir / "gpu_summary.csv").exists()
    assert (scheme_dir / "gpu_utilization.png").exists()
    assert (scheme_dir / "timing_tree.md").exists()
    md = (exp / "benchmark_report.md").read_text()
    assert "每个方案的具体性能文件路径" in md
    assert "`N/A`" in md


def make_subprocess_exp(root: Path, name: str, ate: float = 0.1) -> Path:
    exp = root / name
    write_csv(
        exp / "summary.csv",
        [
            {
                "mode": "per_window",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 10.0,
                "wall_s": 9.0,
                "ate_rmse_sim3": ate,
                "rpe_rot_deg_mean_sim3": 0.2,
                "axis_ate_rmse": 0.3,
            },
            {
                "mode": "clip_once",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 20.0,
                "wall_s": 0.0,
                "ate_rmse_sim3": 0.4,
                "rpe_rot_deg_mean_sim3": 0.5,
                "axis_ate_rmse": 0.6,
            },
        ],
    )
    (exp / "timings.json").write_text(json.dumps({"windows": {}}))
    meta_dir = exp / "inputs" / "per_window" / "window_0000_0_80"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"duration_s": 2.7, "num_raw_frames": 81}))
    full_dir = exp / "inputs" / "full_clip" / f"{name}_full"
    full_dir.mkdir(parents=True)
    (full_dir / "meta.json").write_text(json.dumps({"source_mp4": f"/data/{name}.mp4", "duration_s": 2.7, "num_raw_frames": 81}))
    return exp


def test_write_aggregate_report_creates_total_and_scheme_files(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_subprocess_exp(root, "a_ws81_stride81_start0", ate=0.1)
    make_subprocess_exp(root, "b_ws81_stride81_start0", ate=0.3)
    out_dir = root / "_aggregate"

    report.write_aggregate_report(root, out_dir=out_dir, limit_experiments=5)

    assert (out_dir / "benchmark_report.md").exists()
    scheme_dir = out_dir / "schemes" / "A_subprocess_per_window"
    assert (scheme_dir / "scheme_report.md").exists()
    assert (scheme_dir / "per_video_reports.csv").exists()
    rows, context = report.build_aggregate_report(root, limit_experiments=5)
    assert context["input_video_count"] == 2
    assert rows[0]["success_windows"] == 2
    assert rows[0]["workers"] == 1
    assert rows[0]["ate_sim3"] == 0.2


def make_resident_exp(root: Path, name: str, util_values: tuple[int, int]) -> Path:
    exp = root / name
    write_csv(
        exp / "summary.csv",
        [
            {
                "mode": "resident_per_window",
                "window_id": "window_0000_0_80",
                "num_frames": 81,
                "vipe_elapsed_s": 10.0,
                "wall_s": 9.0,
                "ate_rmse_sim3": 0.1,
                "rpe_rot_deg_mean_sim3": 0.2,
                "axis_ate_rmse": 0.3,
            }
        ],
    )
    report_path = exp / "worker_reports" / "per_window_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "worker_total_s": 12.0,
                "subprocess_wall_s": 11.0,
                "jobs_total_s": 10.0,
                "jobs": [{"timings": {"pipeline_run_s": 8.0}}],
            }
        )
    )
    (exp / "timings.json").write_text(
        json.dumps(
            {
                "runner": "resident",
                "worker_reports": {"per_window": str(report_path)},
            }
        )
    )
    meta_dir = exp / "inputs" / "per_window" / "window_0000_0_80"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({"duration_s": 2.7, "num_raw_frames": 81}))
    full_dir = exp / "inputs" / "full_clip" / f"{name}_full"
    full_dir.mkdir(parents=True)
    (full_dir / "meta.json").write_text(json.dumps({"source_mp4": f"/data/{name}.mp4", "duration_s": 2.7, "num_raw_frames": 81}))
    gpu_csv = exp / "gpu" / "per_window_gpu.csv"
    write_csv(
        gpu_csv,
        [
            {
                "sample_time_s": 1.0,
                "timestamp": "2026",
                "gpu_index": 0,
                "util_gpu_pct": util_values[0],
                "memory_used_mib": 1000,
                "power_w": 200,
            },
            {
                "sample_time_s": 2.0,
                "timestamp": "2026",
                "gpu_index": 0,
                "util_gpu_pct": util_values[1],
                "memory_used_mib": 2000,
                "power_w": 250,
            },
        ],
    )
    return exp


def test_write_aggregate_report_merges_scheme_gpu_csvs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_resident_exp(root, "a_ws81_stride81_start0", (10, 50))
    make_resident_exp(root, "b_ws81_stride81_start0", (20, 60))
    out_dir = root / "_aggregate"

    report.write_aggregate_report(root, out_dir=out_dir, limit_experiments=5)

    scheme_dir = out_dir / "schemes" / "B_resident_per_window"
    merged_csv = scheme_dir / "gpu_merged.csv"
    assert merged_csv.exists()
    merged = list(csv.DictReader(merged_csv.open()))
    assert len(merged) == 4
    assert {row["source_experiment"] for row in merged} == {"a_ws81_stride81_start0", "b_ws81_stride81_start0"}
    gpu_summary = list(csv.DictReader((scheme_dir / "gpu_summary.csv").open()))[0]
    assert gpu_summary["gpu_csv"] == str(merged_csv)
    assert float(gpu_summary["gpu_util_min_pct"]) == 10.0
    assert float(gpu_summary["gpu_util_mean_pct"]) == 35.0
    assert float(gpu_summary["gpu_util_max_pct"]) == 60.0
    assert (scheme_dir / "gpu_utilization.png").stat().st_size > 0


def test_gpu_summary_reads_sampler_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "gpu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sample_time_s,timestamp,gpu_index,util_gpu_pct,memory_used_mib,power_w",
                "1.0,2026,0,10,1000,200",
                "2.0,2026,0,50,2000,250",
                "",
            ]
        )
    )

    summary = report.summarize_gpu_csv(csv_path)

    assert summary["gpu_util_mean_pct"] == 30.0
    assert summary["gpu_util_min_pct"] == 10.0
    assert summary["gpu_util_max_pct"] == 50.0
    assert summary["memory_used_max_mib"] == 2000.0
