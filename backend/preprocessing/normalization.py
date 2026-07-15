"""
normalization.py - Per-band image normalisation.

Supports:
  - min-max normalisation → [0, 1]
  - mean/std (z-score) normalisation
  - no-op pass-through
  - percentile-based clipping before normalisation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = ("minmax", "meanstd", "none")


class BandNormalizer:
    """
    Normalises each spectral band independently.

    Parameters
    ----------
    method : str
        ``'minmax'``, ``'meanstd'``, or ``'none'``.
    clip_percentile : tuple[float, float]
        (low, high) percentile bounds applied *before* normalisation
        to suppress outliers. Set to (0, 100) to disable.
    """

    def __init__(
        self,
        method: str = "minmax",
        clip_percentile: tuple[float, float] = (2.0, 98.0),
    ) -> None:
        method = method.lower()
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown normalisation method '{method}'. "
                f"Choose from: {_SUPPORTED_METHODS}"
            )
        self.method = method
        self.clip_percentile = clip_percentile

        # Per-band statistics populated after normalise() is called
        self.band_stats: list[dict[str, float]] = []

    # ─── Public API ────────────────────────────────────────────────────────

    def normalise(
        self,
        data: np.ndarray,
        stats: list[dict[str, float]] | None = None,
    ) -> np.ndarray:
        """
        Normalise a multi-band image.

        Parameters
        ----------
        data : np.ndarray
            Input array (H, W) or (H, W, C). Any numeric dtype.
        stats : list of dict, optional
            Pre-computed per-band stats to use instead of computing fresh.
            Each dict must have keys required by the chosen method:
            ``'min'`` / ``'max'`` for minmax;
            ``'mean'`` / ``'std'`` for meanstd.

        Returns
        -------
        np.ndarray
            float32 array, same spatial shape as input.
            - minmax  → [0, 1]
            - meanstd → zero-mean, unit-variance (unbounded)
            - none    → float32 cast of input
        """
        if self.method == "none":
            return data.astype(np.float32)

        img = data.astype(np.float32)
        single_band = img.ndim == 2
        if single_band:
            img = img[:, :, np.newaxis]

        n_bands = img.shape[-1]
        out = np.empty_like(img)
        self.band_stats = []

        for c in range(n_bands):
            band = img[:, :, c].copy()
            band_stats_c = self._compute_stats(band, stats[c] if stats else None)
            out[:, :, c] = self._apply_normalisation(band, band_stats_c)
            self.band_stats.append(band_stats_c)

        logger.debug(
            "Normalisation '%s' applied | bands=%d", self.method, n_bands
        )

        return out[:, :, 0] if single_band else out

    def denormalise(
        self,
        data: np.ndarray,
        stats: list[dict[str, float]] | None = None,
    ) -> np.ndarray:
        """
        Reverse normalisation using stored or provided stats.

        Parameters
        ----------
        data : np.ndarray
            Normalised array (H, W) or (H, W, C).
        stats : list of dict, optional
            Override stats. Defaults to ``self.band_stats``.

        Returns
        -------
        np.ndarray
            Denormalised float32 array.
        """
        _stats = stats or self.band_stats
        if not _stats:
            raise RuntimeError("No band stats available. Call normalise() first.")

        img = data.astype(np.float32)
        single_band = img.ndim == 2
        if single_band:
            img = img[:, :, np.newaxis]

        out = np.empty_like(img)
        for c, s in enumerate(_stats):
            if c >= img.shape[-1]:
                break
            if self.method == "minmax":
                out[:, :, c] = img[:, :, c] * (s["max"] - s["min"]) + s["min"]
            elif self.method == "meanstd":
                out[:, :, c] = img[:, :, c] * s["std"] + s["mean"]

        return out[:, :, 0] if single_band else out

    # ─── Internal ──────────────────────────────────────────────────────────

    def _compute_stats(
        self,
        band: np.ndarray,
        provided: dict[str, float] | None,
    ) -> dict[str, float]:
        """Compute or pass through per-band statistics."""
        if provided is not None:
            return provided

        lo, hi = self.clip_percentile
        p_lo = float(np.percentile(band, lo))
        p_hi = float(np.percentile(band, hi))

        return {
            "min": p_lo,
            "max": p_hi,
            "mean": float(np.mean(band)),
            "std": float(np.std(band) + 1e-8),
        }

    def _apply_normalisation(
        self,
        band: np.ndarray,
        stats: dict[str, float],
    ) -> np.ndarray:
        """Apply the configured normalisation to a single band."""
        # Percentile clip
        band = np.clip(band, stats["min"], stats["max"])

        if self.method == "minmax":
            denom = stats["max"] - stats["min"]
            if denom < 1e-8:
                return np.zeros_like(band)
            return (band - stats["min"]) / denom

        if self.method == "meanstd":
            return (band - stats["mean"]) / stats["std"]

        return band  # 'none'
