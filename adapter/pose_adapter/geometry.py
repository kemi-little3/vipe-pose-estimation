"""SE(3) utilities shared by the three adapters.

Conventions:
- All pose matrices are 4x4 homogeneous transforms.
- c2w = camera-to-world. Its translation column is the camera center in world coords.
- w2c = world-to-camera = inv(c2w).
- OpenCV camera frame: x-right, y-down, z-forward.
"""
from __future__ import annotations

import numpy as np


def invert_se3_batch(T: np.ndarray) -> np.ndarray:
    """Invert a batch of SE(3) matrices using R.T and -R.T @ t.

    Faster and more numerically stable than np.linalg.inv for rigid transforms.
    """
    if T.ndim != 3 or T.shape[-2:] != (4, 4):
        raise ValueError(f"expected [N,4,4], got {T.shape}")
    R = T[:, :3, :3]
    t = T[:, :3, 3:4]
    R_t = np.transpose(R, (0, 2, 1))
    out = np.zeros_like(T)
    out[:, :3, :3] = R_t
    out[:, :3, 3:4] = -R_t @ t
    out[:, 3, 3] = 1.0
    return out


def relative_to_first(c2w: np.ndarray) -> np.ndarray:
    """Rebase a c2w trajectory so frame 0 is the world origin.

    rel[i] = inv(c2w[0]) @ c2w[i]
    rel[0] is exactly I4.
    """
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected [N,4,4], got {c2w.shape}")
    w2c_0 = invert_se3_batch(c2w[:1])[0]
    return np.einsum("ij,njk->nik", w2c_0, c2w)


def normalize_max_translation(c2w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Rescale translations so max ||t_i|| == 1.

    The rotation part is untouched. Returns the normalized trajectory, the divisor,
    and the pre-normalization total path length for downstream motion-intensity use.

    If max ||t_i|| == 0 (camera did not move), the trajectory is returned unchanged
    and scale_factor is set to 1.0.
    """
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected [N,4,4], got {c2w.shape}")

    translations = c2w[:, :3, 3]  # [N, 3]
    norms = np.linalg.norm(translations, axis=1)  # [N]
    scale = float(norms.max())

    segs = np.linalg.norm(np.diff(translations, axis=0), axis=1) if len(translations) > 1 else np.zeros(0)
    total_path = float(segs.sum())

    if scale <= 0.0:
        return c2w.copy(), 1.0, total_path

    out = c2w.copy()
    out[:, :3, 3] = translations / scale
    return out, scale, total_path
