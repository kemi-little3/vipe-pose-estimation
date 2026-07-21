#!/usr/bin/env python3
"""Generate the formal ViPE benchmark report for one experiment directory."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_TO_WORKER = {
    "resident_per_window": "per_window",
    "per_window": "per_window",
    "resident_clip_once_then_slice": "clip_once",
    "clip_once": "clip_once",
}
FINAL_SCHEME_NAMES = {
    "per_window": "A_subprocess_per_window",
    "resident_per_window": "B_resident_per_window",
    "resident_clip_once_then_slice": "C_resident_clip_once_then_slice",
    "clip_once": "C_resident_clip_once_then_slice",
}


def final_scheme_name(mode: str) -> str:
    return FINAL_SCHEME_NAMES.get(mode, mode)
EXP_RE = re.compile(r"^(?P<stem>.+)_ws(?P<window_size>\d+)_stride(?P<stride>\d+)_start(?P<start>\d+)$")


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def f(value: Any) -> float:
    parsed = parse_float(value)
    return 0.0 if parsed is None else parsed


def mean(values: list[float | None]) -> float | None:
    xs = [value for value in values if value is not None]
    return sum(xs) / len(xs) if xs else None


def read_summary_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    for row in rows:
        for key in ("num_frames", "start_frame", "end_frame"):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
        for key in (
            "vipe_elapsed_s",
            "wall_s",
            "ate_rmse_sim3",
            "rpe_rot_deg_mean_sim3",
            "axis_ate_rmse",
        ):
            row[key] = parse_float(row.get(key))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        return json.load(fp)


def resolve_worker_report(exp_dir: Path, timings: dict[str, Any], worker_key: str) -> Path | None:
    raw = timings.get("worker_reports", {}).get(worker_key)
    candidates = []
    if raw:
        candidates.append(Path(raw))
    candidates.append(exp_dir / "worker_reports" / f"{worker_key}_report.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def worker_stats(report_path: Path | None) -> dict[str, Any]:
    if report_path is None:
        return {
            "workers": 0,
            "vipe_stage_cumulative_s": 0.0,
            "vipe_stage_wall_s": 0.0,
            "pipeline_run_s": 0.0,
            "worker_report": "",
        }
    report = load_json(report_path)
    pipeline_run_s = 0.0
    for job in report.get("jobs", []):
        pipeline_run_s += f(job.get("timings", {}).get("pipeline_run_s"))
    stage_cumulative = f(report.get("jobs_total_s")) or sum(f(job.get("job_total_s")) for job in report.get("jobs", []))
    stage_wall = f(report.get("subprocess_wall_s")) or f(report.get("worker_total_s"))
    return {
        "workers": 1,
        "vipe_stage_cumulative_s": stage_cumulative,
        "vipe_stage_wall_s": stage_wall,
        "pipeline_run_s": pipeline_run_s,
        "worker_report": str(report_path),
    }


def window_duration_map(exp_dir: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    for meta_path in (exp_dir / "inputs" / "per_window").glob("*/meta.json"):
        meta = load_json(meta_path)
        duration = parse_float(meta.get("duration_s"))
        if duration is None and meta.get("num_raw_frames") and meta.get("sampled_fps"):
            duration = f(meta.get("num_raw_frames")) / f(meta.get("sampled_fps"))
        durations[meta_path.parent.name] = 0.0 if duration is None else duration
    return durations


def input_context(exp_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    match = EXP_RE.match(exp_dir.name)
    full_meta_paths = sorted((exp_dir / "inputs" / "full_clip").glob("*/meta.json"))
    full_metas = [load_json(path) for path in full_meta_paths]
    duration_by_window = window_duration_map(exp_dir)
    success_window_ids = sorted({row["window_id"] for row in rows})
    input_paths = sorted({meta.get("source_mp4", "") for meta in full_metas if meta.get("source_mp4")})
    return {
        "experiment_dir": str(exp_dir),
        "experiment": exp_dir.name,
        "stem": match.group("stem") if match else exp_dir.name,
        "window_size": int(match.group("window_size")) if match else None,
        "stride": int(match.group("stride")) if match else None,
        "input_paths": input_paths,
        "input_video_count": len(input_paths) or len(full_metas) or int(bool(success_window_ids)),
        "input_duration_s": sum(f(meta.get("duration_s")) for meta in full_metas),
        "input_total_frames": sum(int(meta.get("num_raw_frames") or 0) for meta in full_metas),
        "success_windows": len(success_window_ids),
        "success_duration_s": sum(duration_by_window.get(window_id, 0.0) for window_id in success_window_ids),
        "success_total_frames": sum(
            int(row.get("num_frames") or 0)
            for window_id in success_window_ids
            for row in rows
            if row["window_id"] == window_id
        )
        // max(1, len({row["mode"] for row in rows})),
    }


def build_report(exp_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exp_dir = exp_dir.resolve()
    summary_rows = read_summary_rows(exp_dir / "summary.csv")
    timings = load_json(exp_dir / "timings.json")
    durations = window_duration_map(exp_dir)
    context = input_context(exp_dir, summary_rows)

    if timings.get("runner") != "resident":
        summary_rows = [row for row in summary_rows if row.get("mode") == "per_window"]

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_mode[row["mode"]].append(row)

    rows: list[dict[str, Any]] = []
    for mode in sorted(by_mode):
        mode_rows = by_mode[mode]
        worker_key = MODE_TO_WORKER.get(mode, mode)
        stats = worker_stats(resolve_worker_report(exp_dir, timings, worker_key))
        if stats["workers"] == 0 and mode == "per_window":
            stats["workers"] = 1
            stats["vipe_stage_cumulative_s"] = sum(f(row.get("vipe_elapsed_s")) for row in mode_rows)
            stats["vipe_stage_wall_s"] = sum(f(row.get("wall_s")) for row in mode_rows)
            stats["pipeline_run_s"] = stats["vipe_stage_cumulative_s"]
        success_windows = len(mode_rows)
        processed_duration = sum(durations.get(row["window_id"], 0.0) for row in mode_rows)
        rows.append(
            {
                "scheme": final_scheme_name(mode),
                "raw_mode": mode,
                "workers": stats["workers"],
                "resident": "yes" if "resident" in mode or timings.get("runner") == "resident" else "no",
                "success_windows": success_windows,
                "processed_success_duration_s": processed_duration,
                "vipe_stage_cumulative_s": stats["vipe_stage_cumulative_s"],
                "vipe_stage_wall_min": stats["vipe_stage_wall_s"] / 60.0,
                "pipeline_run_min": stats["pipeline_run_s"] / 60.0,
                "ate_sim3": mean([row.get("ate_rmse_sim3") for row in mode_rows]),
                "rpe_rot_deg": mean([row.get("rpe_rot_deg_mean_sim3") for row in mode_rows]),
                "axis_ate": mean([row.get("axis_ate_rmse") for row in mode_rows]),
                "worker_report": stats["worker_report"],
                "timing_tree": str(exp_dir / "timing_tree_summary" / "timing_tree.md"),
                "gpu_csv": str(exp_dir / "gpu" / f"{worker_key}_gpu.csv"),
            }
        )
    return rows, context


def summarize_gpu_csv(path: Path) -> dict[str, Any]:
    values: dict[str, list[float]] = {"util": [], "mem": [], "power": []}
    if not path.exists():
        return {
            "gpu_csv": str(path),
            "gpu_util_min_pct": None,
            "gpu_util_mean_pct": None,
            "gpu_util_max_pct": None,
            "memory_used_min_mib": None,
            "memory_used_mean_mib": None,
            "memory_used_max_mib": None,
            "power_min_w": None,
            "power_mean_w": None,
            "power_max_w": None,
        }
    with path.open() as fp:
        for row in csv.DictReader(fp):
            util = parse_float(row.get("util_gpu_pct"))
            mem = parse_float(row.get("memory_used_mib"))
            power = parse_float(row.get("power_w"))
            if util is not None:
                values["util"].append(util)
            if mem is not None:
                values["mem"].append(mem)
            if power is not None:
                values["power"].append(power)

    def stat(key: str, fn) -> float | None:
        xs = values[key]
        return float(fn(xs)) if xs else None

    return {
        "gpu_csv": str(path),
        "gpu_util_min_pct": stat("util", min),
        "gpu_util_mean_pct": stat("util", lambda xs: sum(xs) / len(xs)),
        "gpu_util_max_pct": stat("util", max),
        "memory_used_min_mib": stat("mem", min),
        "memory_used_mean_mib": stat("mem", lambda xs: sum(xs) / len(xs)),
        "memory_used_max_mib": stat("mem", max),
        "power_min_w": stat("power", min),
        "power_mean_w": stat("power", lambda xs: sum(xs) / len(xs)),
        "power_max_w": stat("power", max),
    }


def merge_gpu_csvs(per_video_rows: list[dict[str, Any]], out_path: Path) -> Path:
    fieldnames = [
        "source_experiment",
        "source_gpu_csv",
        "sample_time_s",
        "timestamp",
        "gpu_index",
        "util_gpu_pct",
        "memory_used_mib",
        "power_w",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for per_video in per_video_rows:
            gpu_csv = Path(str(per_video.get("gpu_csv") or ""))
            if not gpu_csv.exists():
                continue
            with gpu_csv.open() as src_fp:
                for src_row in csv.DictReader(src_fp):
                    writer.writerow(
                        {
                            "source_experiment": per_video.get("experiment", ""),
                            "source_gpu_csv": str(gpu_csv),
                            "sample_time_s": src_row.get("sample_time_s", ""),
                            "timestamp": src_row.get("timestamp", ""),
                            "gpu_index": src_row.get("gpu_index", ""),
                            "util_gpu_pct": src_row.get("util_gpu_pct", ""),
                            "memory_used_mib": src_row.get("memory_used_mib", ""),
                            "power_w": src_row.get("power_w", ""),
                        }
                    )
    return out_path


def plot_gpu_util(csv_path: Path, out_path: Path) -> None:
    times: list[float] = []
    utils: list[float] = []
    if csv_path.exists():
        with csv_path.open() as fp:
            rows = list(csv.DictReader(fp))
        t0 = parse_float(rows[0].get("sample_time_s")) if rows else None
        for row in rows:
            util = parse_float(row.get("util_gpu_pct"))
            t = parse_float(row.get("sample_time_s"))
            if util is not None and t is not None:
                times.append(t - (t0 or t))
                utils.append(util)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if times:
        ax.plot(times, utils, linewidth=1.5)
    ax.set_title(csv_path.stem)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("GPU util (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def display_path(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    text = str(value)
    if text.startswith("/") and not Path(text).exists():
        return "N/A"
    return text


def display_workers(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if count <= 0 else str(count)


def copy_text_or_placeholder(src: str, dst: Path, title: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src and Path(src).exists():
        dst.write_text(Path(src).read_text())
    else:
        dst.write_text(f"# {title}\n\nN/A\n")


def write_scheme_report(path: Path, row: dict[str, Any], gpu_row: dict[str, Any]) -> None:
    lines = [
        f"# {row['scheme']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| workers | {display_workers(row['workers'])} |",
        f"| 是否常驻 | {row['resident']} |",
        f"| 成功window | {row['success_windows']} |",
        f"| 处理成功的视频长度(s) | {fmt(row['processed_success_duration_s'], 3)} |",
        f"| vipe stage累计耗时(s) | {fmt(row['vipe_stage_cumulative_s'], 3)} |",
        f"| vipe stage墙钟耗时(min) | {fmt(row['vipe_stage_wall_min'], 3)} |",
        f"| pipeline run耗时(min) | {fmt(row['pipeline_run_min'], 3)} |",
        f"| ATE Sim3 | {fmt(row['ate_sim3'], 4)} |",
        f"| RPE rot deg | {fmt(row['rpe_rot_deg'], 4)} |",
        f"| axis ATE | {fmt(row['axis_ate'], 4)} |",
        "",
        "## Files",
        "",
        f"- worker report: `{display_path(row['worker_report'])}`",
        f"- GPU CSV: `{display_path(row['gpu_csv'])}`",
        f"- timing tree: `{display_path(row['timing_tree'])}`",
        f"- GPU util plot: `{display_path(gpu_row.get('gpu_util_plot'))}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], context: dict[str, Any], gpu_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ViPE Benchmark Report",
        "",
        "## Summary",
        "",
        f"- 处理的输入: {', '.join(f'`{p}`' for p in context['input_paths']) or '`unknown`'}",
        "- 执行的处理脚本: `vipe/pose_estimation/scripts/compare_vipe_window_modes.py`",
        f"- 输入 video 个数: {context['input_video_count']}",
        f"- 输入视频总时长: {fmt(context['input_duration_s'], 3)} s",
        f"- 输入总帧数: {context['input_total_frames']}",
        "- 测试使用的 GPU: 见 `gpu_summary.csv`",
        "- 每个方案的具体性能文件路径: 见下方 `Per-Scheme Files`",
        "",
        "| 方案 | workers | 是否常驻 | 成功window | 处理成功的视频长度(s) | vipe stage累计耗时(s) | vipe stage墙钟耗时(min) | pipeline run耗时(min) | ATE Sim3 | RPE rot deg | axis ATE | 性能文件 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scheme} | {workers} | {resident} | {success_windows} | {duration} | {stage} | {wall} | {pipeline} | {ate} | {rpe} | {axis} | `{perf}` |".format(
                scheme=row["scheme"],
                workers=display_workers(row["workers"]),
                resident=row["resident"],
                success_windows=row["success_windows"],
                duration=fmt(row["processed_success_duration_s"], 3),
                stage=fmt(row["vipe_stage_cumulative_s"], 3),
                wall=fmt(row["vipe_stage_wall_min"], 3),
                pipeline=fmt(row["pipeline_run_min"], 3),
                ate=fmt(row["ate_sim3"], 4),
                rpe=fmt(row["rpe_rot_deg"], 4),
                axis=fmt(row["axis_ate"], 4),
                perf=row.get("scheme_report", row["worker_report"]),
            )
        )
    lines.extend(
        [
            "",
            "## GPU Performance",
            "",
            "| 方案 | GPU util min | GPU util mean | GPU util max | memory max MiB | power mean W | util plot |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in gpu_rows:
        lines.append(
            f"| {row['scheme']} | {fmt(row['gpu_util_min_pct'], 2)} | {fmt(row['gpu_util_mean_pct'], 2)} | "
            f"{fmt(row['gpu_util_max_pct'], 2)} | {fmt(row['memory_used_max_mib'], 1)} | "
            f"{fmt(row['power_mean_w'], 2)} | `{row['gpu_util_plot']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-Scheme Files",
            "",
            "| 方案 | worker report | GPU CSV | timing tree |",
            "|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['scheme']} | `{display_path(row.get('worker_report'))}` | "
            f"`{display_path(row.get('gpu_csv'))}` | `{display_path(row.get('timing_tree'))}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_report(exp_dir: Path, out_dir: Path | None = None) -> None:
    rows, context = build_report(exp_dir)
    out_dir = (out_dir or exp_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_rows: list[dict[str, Any]] = []
    for row in rows:
        scheme_dir = out_dir / "schemes" / row["scheme"]
        scheme_dir.mkdir(parents=True, exist_ok=True)
        gpu_csv = Path(row["gpu_csv"])
        gpu_summary = summarize_gpu_csv(gpu_csv)
        gpu_plot = scheme_dir / "gpu_utilization.png"
        plot_gpu_util(gpu_csv, gpu_plot)
        gpu_row = {"scheme": row["scheme"], "gpu_util_plot": str(gpu_plot), **gpu_summary}
        gpu_rows.append(gpu_row)
        write_csv(scheme_dir / "gpu_summary.csv", [gpu_row])
        row["scheme_report"] = str(scheme_dir / "scheme_report.md")
        row["scheme_dir"] = str(scheme_dir)
        copy_text_or_placeholder(row.get("timing_tree", ""), scheme_dir / "timing_tree.md", "Timing Tree")
        write_scheme_report(scheme_dir / "scheme_report.md", row, gpu_row)

    write_csv(out_dir / "benchmark_report.csv", rows)
    write_csv(out_dir / "gpu_summary.csv", gpu_rows)
    write_markdown(out_dir / "benchmark_report.md", rows, context, gpu_rows)


def experiment_dirs(input_root: Path, limit_experiments: int = 0) -> list[Path]:
    dirs = sorted(path for path in input_root.glob("*_ws*_stride*_start*") if path.is_dir())
    if limit_experiments > 0:
        dirs = dirs[:limit_experiments]
    return dirs


def build_aggregate_report(input_root: Path, limit_experiments: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exps = experiment_dirs(input_root.resolve(), limit_experiments)
    per_exp_rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for exp in exps:
        rows, context = build_report(exp)
        contexts.append(context)
        for row in rows:
            item = dict(row)
            item["experiment"] = context["experiment"]
            item["experiment_dir"] = str(exp)
            item["benchmark_report"] = str(exp / "benchmark_report.md")
            per_exp_rows.append(item)

    by_scheme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_exp_rows:
        by_scheme[row["scheme"]].append(row)

    total_rows: list[dict[str, Any]] = []
    for scheme in sorted(by_scheme):
        rs = by_scheme[scheme]
        total_rows.append(
            {
                "scheme": scheme,
                "raw_mode": rs[0].get("raw_mode", ""),
                "workers": max((int(r["workers"]) for r in rs if isinstance(r.get("workers"), int) and int(r["workers"]) > 0), default=0),
                "resident": rs[0]["resident"],
                "success_windows": sum(int(r.get("success_windows") or 0) for r in rs),
                "processed_success_duration_s": sum(f(r.get("processed_success_duration_s")) for r in rs),
                "vipe_stage_cumulative_s": sum(f(r.get("vipe_stage_cumulative_s")) for r in rs),
                "vipe_stage_wall_min": sum(f(r.get("vipe_stage_wall_min")) for r in rs),
                "pipeline_run_min": sum(f(r.get("pipeline_run_min")) for r in rs),
                "ate_sim3": mean([r.get("ate_sim3") for r in rs]),
                "rpe_rot_deg": mean([r.get("rpe_rot_deg") for r in rs]),
                "axis_ate": mean([r.get("axis_ate") for r in rs]),
                "worker_report": "",
                "gpu_csv": "",
                "timing_tree": "",
                "per_video_rows": rs,
            }
        )

    input_paths: list[str] = []
    for context in contexts:
        input_paths.extend(context.get("input_paths", []))
    aggregate_context = {
        "experiment_dir": str(input_root.resolve()),
        "experiment": input_root.resolve().name,
        "input_paths": sorted(set(input_paths)),
        "input_video_count": sum(int(c.get("input_video_count") or 0) for c in contexts),
        "input_duration_s": sum(f(c.get("input_duration_s")) for c in contexts),
        "input_total_frames": sum(int(c.get("input_total_frames") or 0) for c in contexts),
        "success_windows": sum(int(c.get("success_windows") or 0) for c in contexts),
        "success_duration_s": sum(f(c.get("success_duration_s")) for c in contexts),
        "success_total_frames": sum(int(c.get("success_total_frames") or 0) for c in contexts),
        "experiment_count": len(exps),
    }
    return total_rows, aggregate_context


def write_aggregate_report(input_root: Path, out_dir: Path | None = None, limit_experiments: int = 0) -> None:
    rows, context = build_aggregate_report(input_root, limit_experiments)
    out_dir = (out_dir or (input_root / "_aggregate")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_rows: list[dict[str, Any]] = []
    for row in rows:
        scheme_dir = out_dir / "schemes" / row["scheme"]
        scheme_dir.mkdir(parents=True, exist_ok=True)
        per_video_rows = row.pop("per_video_rows")
        row["scheme_dir"] = str(scheme_dir)
        row["scheme_report"] = str(scheme_dir / "scheme_report.md")
        write_csv(scheme_dir / "per_video_reports.csv", per_video_rows)
        merged_gpu_csv = merge_gpu_csvs(per_video_rows, scheme_dir / "gpu_merged.csv")
        gpu_plot = scheme_dir / "gpu_utilization.png"
        plot_gpu_util(merged_gpu_csv, gpu_plot)
        gpu_row = {
            "scheme": row["scheme"],
            "gpu_util_plot": str(gpu_plot),
            **summarize_gpu_csv(merged_gpu_csv),
        }
        write_csv(scheme_dir / "gpu_summary.csv", [gpu_row])
        (scheme_dir / "timing_tree.md").write_text("# Timing Tree\n\nSee per-video reports.\n")
        write_scheme_report(scheme_dir / "scheme_report.md", row, gpu_row)
        gpu_rows.append(gpu_row)

    write_csv(out_dir / "benchmark_report.csv", rows)
    write_csv(out_dir / "gpu_summary.csv", gpu_rows)
    write_markdown(out_dir / "benchmark_report.md", rows, context, gpu_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit-experiments", type=int, default=0)
    args = parser.parse_args()
    if args.input_root:
        write_aggregate_report(args.input_root, args.out_dir, args.limit_experiments)
    elif args.exp_dir:
        write_report(args.exp_dir, args.out_dir)
    else:
        parser.error("one of --exp-dir or --input-root is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
