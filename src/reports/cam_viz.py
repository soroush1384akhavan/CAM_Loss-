from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def signed_cam_map(
    cam: np.ndarray, scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return a diverging RGB map, signed values in [-1, 1], and scale."""
    cam = np.asarray(cam, dtype=np.float32)
    scale = max(float(np.abs(cam).max()), 1e-6) if scale is None else max(float(scale), 1e-6)
    signed = np.clip(cam / scale, -1.0, 1.0)
    values = (signed + 1.0) / 2.0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        heat = cm.coolwarm(values)[..., :3]
    except ImportError:
        heat = np.empty((*signed.shape, 3), dtype=np.float32)
        heat[..., 0] = np.where(signed >= 0, 1.0, 1.0 + signed)
        heat[..., 1] = 1.0 - np.abs(signed)
        heat[..., 2] = np.where(signed <= 0, 1.0, 1.0 - signed)
    return (heat * 255).clip(0, 255).astype(np.uint8), signed, scale


def signed_cam_overlay(
    image: np.ndarray,
    cam: np.ndarray,
    boundary: Optional[np.ndarray] = None,
    max_alpha: float = 0.42,
    gamma: float = 0.8,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Blend a signed CAM over an image while keeping the image readable."""
    heat, signed, scale = signed_cam_map(cam, scale=scale)
    strength = max_alpha * np.abs(signed) ** gamma
    overlay = (strength[..., None] * heat + (1.0 - strength[..., None]) * image).clip(0, 255)
    overlay = overlay.astype(np.uint8)
    if boundary is not None:
        overlay[np.asarray(boundary, dtype=bool)] = np.array([0, 255, 0], dtype=np.uint8)
    return heat, overlay, scale
