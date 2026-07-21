import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import vipe_resident_worker as worker


def test_convert_artifacts_maps_sparse_rows_to_source_frame_indices(tmp_path, monkeypatch):
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    np.save(clip_dir / "frame_indices.npy", np.asarray([0, 1, 2], dtype=np.int32))
    np.save(clip_dir / "timestamps.npy", np.asarray([0.0, 0.75, 1.75], dtype=np.float64))
    (clip_dir / "meta.json").write_text(
        json.dumps(
            {
                "original_resolution": [64, 48],
                "sampled_fps": 4.0,
                "original_fps": 4.0,
                "source_frame_indices": [100, 103, 107],
                "sparse_frame_indices": [0, 1, 2],
            }
        )
    )

    tmp = tmp_path / "tmp_artifacts"
    tmp.mkdir()
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = [10.0, 20.0, 30.0]
    intr = np.repeat(np.eye(3, dtype=np.float64)[None], 3, axis=0)
    np.savez(tmp / "pose_artifact.npz", data=poses, inds=np.asarray([0, 1, 2], dtype=np.int32))
    np.savez(tmp / "intrinsics_artifact.npz", data=intr, inds=np.asarray([0, 1, 2], dtype=np.int32))

    monkeypatch.setattr(worker, "_find_artifact", lambda base, name, suffix: tmp / ("pose_artifact.npz" if name == "pose" else "intrinsics_artifact.npz"))

    out = tmp_path / "out"
    worker.convert_artifacts(tmp, clip_dir, out, elapsed_s=1.25, extra_quality={})

    assert np.load(out / "frame_indices.npy").tolist() == [100, 103, 107]
    assert np.load(out / "poses_c2w.npy")[:, 0, 3].tolist() == [10.0, 20.0, 30.0]
    quality = json.loads((out / "quality.json").read_text())
    assert quality["sparse_frame_indices"] == [0, 1, 2]
    assert quality["source_frame_indices"] == [100, 103, 107]
