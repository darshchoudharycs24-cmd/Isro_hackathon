"""
preview.py - Generate visual previews for Streamlit/UI consumption.

Saves:
  - current.png           – normalised current image
  - aligned_reference.png – aligned historical reference
  - difference_map.png    – absolute difference heatmap
  - fusion_preview.png    – RGB view of fused image
  - overlay.png           – 50/50 alpha blend of current + aligned
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import to_uint8_png

logger = logging.getLogger(__name__)


def save_all_previews(
    current: np.ndarray,
    aligned: np.ndarray,
    fused: np.ndarray,
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Generate and save all pipeline preview images.

    Parameters
    ----------
    current : np.ndarray
        Current float32 image (H, W, C) in [0,1].
    aligned : np.ndarray
        Aligned historical float32 image (H, W, C) in [0,1].
    fused : np.ndarray
        Fused float32 image (H, W, C_fused) in [0,1].
    output_dir : str or Path
        Directory to write PNGs into.

    Returns
    -------
    dict
        Mapping of label → file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    paths["current"] = _save_rgb(current, out_dir / "current.png")
    paths["aligned_reference"] = _save_rgb(aligned, out_dir / "aligned_reference.png")
    paths["difference_map"] = _save_difference_map(current, aligned, out_dir / "difference_map.png")
    paths["fusion_preview"] = _save_rgb(fused, out_dir / "fusion_preview.png")
    paths["overlay"] = _save_overlay(current, aligned, out_dir / "overlay.png")

    logger.info("Preview images saved to %s", out_dir)
    return paths


# ─── Internal helpers ────────────────────────────────────────────────────────

def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """Extract first 3 channels and convert to uint8."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = img[:, :, :3]
    return to_uint8_png(rgb)


def _save_rgb(img: np.ndarray, path: Path) -> str:
    u8 = _to_rgb_u8(img)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    logger.debug("Saved preview: %s", path.name)
    return str(path)


def _save_difference_map(
    img_a: np.ndarray,
    img_b: np.ndarray,
    path: Path,
) -> str:
    """Save a heatmap of the absolute per-pixel difference."""
    # Use only shared channels
    n = min(img_a.shape[-1] if img_a.ndim == 3 else 1,
            img_b.shape[-1] if img_b.ndim == 3 else 1, 3)

    a = img_a[:, :, :n] if img_a.ndim == 3 else img_a[:, :, np.newaxis]
    b = img_b[:, :, :n] if img_b.ndim == 3 else img_b[:, :, np.newaxis]

    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)
    diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)

    # Apply colourmap (COLORMAP_JET highlights large differences in red)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), heatmap)
    logger.debug("Saved difference map: %s", path.name)
    return str(path)


def _save_overlay(
    img_a: np.ndarray,
    img_b: np.ndarray,
    path: Path,
    alpha: float = 0.5,
) -> str:
    """50/50 alpha blend of two images."""
    a = _to_rgb_u8(img_a).astype(np.float32)
    b = _to_rgb_u8(img_b).astype(np.float32)

    overlay = (alpha * a + (1 - alpha) * b).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.debug("Saved overlay: %s", path.name)
    return str(path)
