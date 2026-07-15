"""
sar_handler.py - Sentinel-1 SAR image loading, normalisation, and alignment.

SAR imagery is treated as an additional input channel stack.
This module handles:
  - Loading SAR GeoTIFF (single or dual polarisation)
  - Log-scale / linear normalisation
  - Geometric alignment to current image grid
  - Channel stacking (VV, VH or single-band)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import GeometricAligner
from .utils import load_image

logger = logging.getLogger(__name__)


class SARHandler:
    """
    Loads and preprocesses Sentinel-1 SAR data for fusion.

    Parameters
    ----------
    align_to_current : bool
        Perform geometric alignment against current image. Default True.
    log_scale : bool
        Convert backscatter to dB (10 * log10). Default False.
    alignment_cfg : dict, optional
        Keyword arguments forwarded to :class:`GeometricAligner`.
    """

    def __init__(
        self,
        align_to_current: bool = True,
        log_scale: bool = False,
        alignment_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.align_to_current = align_to_current
        self.log_scale = log_scale
        self._aligner = GeometricAligner(**(alignment_cfg or {}))

    def load_and_prepare(
        self,
        sar_path: str | Path,
        current_image: np.ndarray,
    ) -> np.ndarray:
        """
        Load SAR image, normalise, and optionally align.

        Parameters
        ----------
        sar_path : str or Path
            Path to the SAR GeoTIFF.
        current_image : np.ndarray
            Current optical image (H, W, C), float32 [0,1].
            Used as alignment target and for spatial shape matching.

        Returns
        -------
        sar_normalised : np.ndarray
            Float32 SAR array of shape (H, W, C_sar) in [0, 1].
        """
        sar_path = Path(sar_path)
        logger.info("Loading SAR image: %s", sar_path.name)

        sar_data, sar_meta = load_image(sar_path)
        sar_float = sar_data.astype(np.float32)

        # Log-scale conversion
        if self.log_scale:
            sar_float = self._to_db(sar_float)

        # Per-band [0,1] normalisation
        sar_norm = self._normalise(sar_float)

        # Ensure 3D (H, W, C)
        if sar_norm.ndim == 2:
            sar_norm = sar_norm[:, :, np.newaxis]

        # Resize to current if sizes differ (fast path)
        ch, cw = current_image.shape[:2]
        sh, sw = sar_norm.shape[:2]
        if sh != ch or sw != cw:
            import cv2  # noqa: PLC0415
            resized_channels = []
            for c in range(sar_norm.shape[-1]):
                resized_channels.append(
                    cv2.resize(sar_norm[:, :, c], (cw, ch), interpolation=cv2.INTER_LINEAR)
                )
            sar_norm = np.stack(resized_channels, axis=-1)

        # Geometric alignment
        if self.align_to_current:
            try:
                aligned, _ = self._aligner.align(sar_norm, current_image)
                sar_norm = aligned if aligned.ndim == 3 else aligned[:, :, np.newaxis]
                logger.info("SAR aligned to current image.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAR alignment failed (%s); using raw resize.", exc)

        logger.info(
            "SAR ready | shape=%s bands=%d", sar_norm.shape, sar_norm.shape[-1]
        )
        return sar_norm

    @staticmethod
    def _to_db(data: np.ndarray) -> np.ndarray:
        """Convert linear backscatter to dB: 10 * log10(x + eps)."""
        eps = 1e-10
        return (10.0 * np.log10(np.abs(data) + eps)).astype(np.float32)

    @staticmethod
    def _normalise(data: np.ndarray) -> np.ndarray:
        """Per-channel min-max normalisation to [0, 1]."""
        if data.ndim == 2:
            lo, hi = data.min(), data.max()
            if hi - lo > 1e-8:
                return ((data - lo) / (hi - lo)).astype(np.float32)
            return np.zeros_like(data, dtype=np.float32)

        out = np.empty_like(data, dtype=np.float32)
        for c in range(data.shape[-1]):
            band = data[:, :, c]
            lo, hi = band.min(), band.max()
            out[:, :, c] = (band - lo) / (hi - lo) if hi - lo > 1e-8 else 0.0
        return out
