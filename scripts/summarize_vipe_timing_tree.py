#!/usr/bin/env python3
"""Summarize ViPE resident-worker timings into a closed timing tree."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp_residual(value: float) -> float:
    if abs(value) < 1e-6:
        return 0.0
    return value


def residual(parent: float, children: list[float]) -> float:
    return clamp_residual(parent - sum(children))


@dataclass
class Node:
    name: str
    time_s: float
    children: list["Node"] = field(default_factory=list)


def load_report(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return json.load(fp)


def aggregate_reports(paths: list[Path]) -> dict[str, Any]:
    reports = [load_report(path) for path in paths]
    metrics: dict[str, float] = {}

    def add(key: str, value: float) -> None:
        metrics[key] = metrics.get(key, 0.0) + float(value)

    add("worker_startup_s", sum(f(report.get("worker_startup_s")) for report in reports))
    add("worker_total_s", sum(f(report.get("worker_total_s")) for report in reports))

    for report in reports:
        for job in report.get("jobs", []):
            timings = job.get("timings", {})
            add("job_total_s", f(job.get("job_total_s")))
            for key, value in timings.items():
                if key.endswith("_count"):
                    continue
                add(key, f(value))

    jobs_total = metrics.get("jobs_total_s") or metrics.get("job_total_s", 0.0)
    metrics["jobs_total_s"] = jobs_total
    if metrics.get("worker_total_s", 0.0) <= 0.0:
        metrics["worker_total_s"] = metrics.get("worker_startup_s", 0.0) + jobs_total

    metrics["stage_residual_s"] = metrics.get(
        "stage_residual_s",
        residual(metrics["worker_total_s"], [metrics.get("worker_startup_s", 0.0), jobs_total]),
    )
    metrics["job_unattributed_s"] = metrics.get(
        "job_unattributed_s",
        residual(
            metrics.get("job_total_s", 0.0),
            [
                metrics.get("compose_config_s", 0.0),
                metrics.get("make_pipeline_s", 0.0),
                metrics.get("raw_stream_init_s", 0.0),
                metrics.get("pipeline_run_s", 0.0),
                metrics.get("artifact_convert_s", 0.0),
            ],
        ),
    )
    metrics["pipeline_unattributed_s"] = metrics.get(
        "pipeline_unattributed_s",
        residual(
            metrics.get("pipeline_run_s", 0.0),
            [
                metrics.get("pipeline_init_cache_process_s", 0.0),
                metrics.get("pipeline_slam_system_init_s", 0.0),
                metrics.get("slam_run_total_s", 0.0),
                metrics.get("pipeline_post_cache_depth_s", 0.0),
                metrics.get("pipeline_save_artifacts_s", 0.0),
            ],
        ),
    )
    metrics["slam_unattributed_s"] = metrics.get(
        "slam_unattributed_s",
        residual(
            metrics.get("slam_run_total_s", 0.0),
            [
                metrics.get("slam_build_components_s", 0.0),
                metrics.get("slam_pass1_s", 0.0),
                metrics.get("slam_backend_global_ba_1_s", 0.0),
                metrics.get("slam_backend_global_ba_2_s", 0.0),
                metrics.get("slam_pass2_s", 0.0),
                metrics.get("slam_extract_slam_map_s", 0.0),
            ],
        ),
    )
    metrics["slam_pass1_unattributed_s"] = metrics.get(
        "slam_pass1_unattributed_s",
        residual(
            metrics.get("slam_pass1_s", 0.0),
            [
                metrics.get("slam_precompute_features_pass1_s", 0.0),
                metrics.get("slam_sparse_tracks_track_image_s", 0.0),
                metrics.get("slam_motion_filter_check_s", 0.0),
                metrics.get("slam_add_keyframe_phase1_s", 0.0),
                metrics.get("slam_frontend_run_s", 0.0),
                metrics.get("slam_backend_run_early_s", 0.0),
            ],
        ),
    )
    metrics["slam_pass2_unattributed_s"] = metrics.get(
        "slam_pass2_unattributed_s",
        residual(
            metrics.get("slam_pass2_s", 0.0),
            [
                metrics.get("slam_precompute_features_pass2_s", 0.0),
                metrics.get("slam_add_keyframe_phase2_s", 0.0),
                metrics.get("inner_filler_compute_s", 0.0),
            ],
        ),
    )
    for prefix in ("slam_backend_run_early", "slam_backend_global_ba_1", "slam_backend_global_ba_2"):
        metrics[f"{prefix}_backend_unattributed_s"] = metrics.get(
            f"{prefix}_backend_unattributed_s",
            residual(metrics.get(f"{prefix}_s", 0.0), [metrics.get(f"{prefix}_bundle_adjustment_s", 0.0)]),
        )

    return {"reports": [str(path.resolve()) for path in paths], "metrics": metrics}


def build_tree(aggregate: dict[str, Any]) -> Node:
    m = aggregate["metrics"]
    g = lambda key: float(m.get(key, 0.0))

    backend_early = Node(
        "backend_run_early",
        g("slam_backend_run_early_s"),
        [
            Node("bundle_adjustment", g("slam_backend_run_early_bundle_adjustment_s")),
            Node("backend_unattributed", g("slam_backend_run_early_backend_unattributed_s")),
        ],
    )
    backend_ba_1 = Node(
        "backend_global_ba_1",
        g("slam_backend_global_ba_1_s"),
        [
            Node("bundle_adjustment", g("slam_backend_global_ba_1_bundle_adjustment_s")),
            Node("backend_unattributed", g("slam_backend_global_ba_1_backend_unattributed_s")),
        ],
    )
    backend_ba_2 = Node(
        "backend_global_ba_2",
        g("slam_backend_global_ba_2_s"),
        [
            Node("bundle_adjustment", g("slam_backend_global_ba_2_bundle_adjustment_s")),
            Node("backend_unattributed", g("slam_backend_global_ba_2_backend_unattributed_s")),
        ],
    )
    slam_pass1 = Node(
        "slam_pass1",
        g("slam_pass1_s"),
        [
            Node("precompute_features_pass1", g("slam_precompute_features_pass1_s")),
            Node("sparse_tracks_track_image", g("slam_sparse_tracks_track_image_s")),
            Node("motion_filter_check", g("slam_motion_filter_check_s")),
            Node("add_keyframe_phase1", g("slam_add_keyframe_phase1_s")),
            Node("frontend_run", g("slam_frontend_run_s")),
            backend_early,
            Node("slam_pass1_unattributed", g("slam_pass1_unattributed_s")),
        ],
    )
    slam_pass2 = Node(
        "slam_pass2",
        g("slam_pass2_s"),
        [
            Node("precompute_features_pass2", g("slam_precompute_features_pass2_s")),
            Node("add_keyframe_phase2", g("slam_add_keyframe_phase2_s")),
            Node("inner_filler_compute", g("inner_filler_compute_s")),
            Node("slam_pass2_unattributed", g("slam_pass2_unattributed_s")),
        ],
    )
    slam = Node(
        "slam_run_total",
        g("slam_run_total_s"),
        [
            Node("slam_build_components", g("slam_build_components_s")),
            slam_pass1,
            backend_ba_1,
            backend_ba_2,
            slam_pass2,
            Node("extract_slam_map", g("slam_extract_slam_map_s")),
            Node("slam_unattributed", g("slam_unattributed_s")),
        ],
    )
    pipeline = Node(
        "pipeline_run",
        g("pipeline_run_s"),
        [
            Node("init_cache_process", g("pipeline_init_cache_process_s")),
            Node("slam_system_init", g("pipeline_slam_system_init_s")),
            slam,
            Node("post_cache_depth", g("pipeline_post_cache_depth_s")),
            Node("save_artifacts", g("pipeline_save_artifacts_s")),
            Node("pipeline_unattributed", g("pipeline_unattributed_s")),
        ],
    )
    job = Node(
        "job_total",
        g("job_total_s"),
        [
            Node("compose_config", g("compose_config_s")),
            Node("make_pipeline", g("make_pipeline_s")),
            Node("raw_stream_init", g("raw_stream_init_s")),
            pipeline,
            Node("artifact_convert", g("artifact_convert_s")),
            Node("job_unattributed", g("job_unattributed_s")),
        ],
    )
    return Node(
        "vipe_stage_total",
        g("worker_total_s"),
        [
            Node("worker_startup", g("worker_startup_s")),
            Node("jobs_total", g("jobs_total_s"), [job]),
            Node("stage_residual", g("stage_residual_s")),
        ],
    )


def flatten_tree(root: Node) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root_total = root.time_s

    def visit(node: Node, parent_time: float | None, path: str, level: int, prefix: str, is_last: bool, is_root: bool) -> None:
        pct_parent = 100.0 if parent_time in (None, 0.0) else node.time_s / parent_time * 100.0
        pct_root = 0.0 if root_total == 0.0 else node.time_s / root_total * 100.0
        connector = "" if is_root else ("└── " if is_last else "├── ")
        rows.append(
            {
                "stage": f"{prefix}{connector}{node.name}",
                "name": node.name,
                "path": path,
                "level": level,
                "time_s": node.time_s,
                "pct_of_parent": pct_parent,
                "pct_of_vipe_stage_total": pct_root,
            }
        )
        child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")
        for idx, child in enumerate(node.children):
            visit(
                child,
                node.time_s,
                f"{path}/{child.name}",
                level + 1,
                child_prefix,
                idx == len(node.children) - 1,
                False,
            )

    visit(root, None, root.name, 0, "", True, True)
    return rows


def write_summary(report_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_reports(report_paths)
    root = build_tree(aggregate)
    rows = flatten_tree(root)

    (out_dir / "timing_tree.json").write_text(json.dumps({"aggregate": aggregate, "rows": rows}, indent=2))
    with (out_dir / "timing_tree.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["stage", "time_s", "pct_of_parent", "pct_of_vipe_stage_total", "level", "path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# ViPE timing tree",
        "",
        "## Inputs",
        "",
        "| worker_report |",
        "|---|",
    ]
    lines.extend(f"| `{path}` |" for path in aggregate["reports"])
    lines.extend(
        [
            "",
            "## Timing Tree",
            "",
            "| stage | time_s | pct_of_parent | pct_of_vipe_stage_total |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['stage']}` | {row['time_s']:.3f} | {row['pct_of_parent']:.2f}% | "
            f"{row['pct_of_vipe_stage_total']:.2f}% |"
        )
    (out_dir / "timing_tree.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True, help="resident worker report JSON files")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    write_summary(args.reports, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
