#!/usr/bin/env python3
"""Plot methods' normalized relative-pose translations on one 3D axes.

After max-norm, trajectories share the same scale (max ||t|| = 1), so
eyeballing them on the same axes tells you whether they agree on shape.

Usage:
    python scripts/compare_trajectories.py outputs/<clip_uuid> [--out path.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("vipe", "vggsfm", "dust3r", "vggt", "megasam")
COLORS = {
    "vipe": "#1f77b4",
    "vggsfm": "#ff7f0e",
    "dust3r": "#2ca02c",
    "vggt": "#d62728",
    "megasam": "#9467bd",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_out_dir", type=Path, help="outputs/<clip_uuid>/")
    parser.add_argument("--out", type=Path, default=None,
                        help="PNG path (default: <clip_out_dir>/trajectory_compare.png)")
    args = parser.parse_args()

    clip_out = args.clip_out_dir
    out_png = args.out or (clip_out / "trajectory_compare.png")

    fig = plt.figure(figsize=(14, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    for method in METHODS:
        rel_path = clip_out / method / "relative_poses.npy"
        if not rel_path.exists():
            print(f"  SKIP {method}: {rel_path} missing", file=sys.stderr)
            continue
        rel = np.load(rel_path)
        t = rel[:, :3, 3]
        mask = np.isfinite(t).all(axis=1)
        t = t[mask]
        color = COLORS[method]
        ax3d.plot(t[:, 0], t[:, 1], t[:, 2], "-o", color=color,
                  markersize=3, linewidth=1.2, label=method)
        ax3d.scatter([t[0, 0]], [t[0, 1]], [t[0, 2]], color=color, marker="s", s=50)
        ax_top.plot(t[:, 0], t[:, 2], "-o", color=color,
                    markersize=3, linewidth=1.2, label=method)
        ax_top.scatter([t[0, 0]], [t[0, 2]], color=color, marker="s", s=50)

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"{clip_out.name} — relative camera translation (max-norm)")
    ax3d.legend()
    ax_top.set_xlabel("x"); ax_top.set_ylabel("z")
    ax_top.set_title("top-down (x,z)")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[compare_trajectories] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
