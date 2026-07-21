from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import run_all_original_vipe_ab_compare as run_all  # noqa: E402


def test_run_all_passes_requested_runner(tmp_path: Path, monkeypatch) -> None:
    gt_root = tmp_path / "data"
    gt_root.mkdir()
    (gt_root / "demo_aligned.mp4").touch()
    (gt_root / "telemetry_vehicle_nav_demo_aligned_processed.csv").write_text("frame_idx\n0\n")

    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append([str(x) for x in cmd])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    rc = run_all.main(
        [
            "--gt-root",
            str(gt_root),
            "--output-root",
            str(tmp_path / "out"),
            "--runner",
            "subprocess",
            "--limit-videos",
            "1",
            "--force",
        ]
    )

    assert rc == 0
    assert any("--runner" in call and call[call.index("--runner") + 1] == "subprocess" for call in calls)
