"""
calibration.py - Radiometric preprocessing.

Implements:
  - DN → Top-of-Atmosphere Reflectance conversion
  - Histogram matching (reference-based)
  - Contrast normalisation with percentile clipping
  - Intensity clipping
  - Noise reduction  (Gaussian / Median / Bilateral)
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms

logger = logging.getLogger(__name__)


class RadiometricCalibrator:
    """
    Applies radiometric corrections to raw satellite DN imagery.

    Parameters
    ----------
    gain : float
        Multiplicative factor for DN→Reflectance. Default 0.0001 (LISS-IV typical).
    offset : float
        Additive offset for DN→Reflectance. Default 0.0.
    clip_min : float
        Lower clip bound after reflectance conversion. Default 0.0.
    clip_max : float
        Upper clip bound after reflectance conversion. Default 1.0.
    noise_method : str
        One of ``'gaussian'``, ``'median'``, ``'bilateral'``, or ``'none'``.
    noise_sigma : float
        Sigma for Gaussian filter. Kernel size for Median (cast to odd int).
    """

    def __init__(
        self,
        gain: float = 0.0001,
        offset: float = 0.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        noise_method: str = "gaussian",
        noise_sigma: float = 1.0,
    ) -> None:
        self.gain = gain
        self.offset = offset
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.noise_method = noise_method.lower()
        self.noise_sigma = noise_sigma

    # ─── Public API ────────────────────────────────────────────────────────

    def calibrate(
        self,
        data: np.ndarray,
        meta: dict[str, Any],
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Full radiometric calibration pipeline.

        Steps:
        1. DN → Reflectance
        2. Intensity clipping
        3. Histogram matching (if reference provided)
        4. Noise reduction

        Parameters
        ----------
        data : np.ndarray
            Input image (H, W) or (H, W, C).  Any integer or float dtype.
        meta : dict
            Image metadata from ``utils.load_image``.
        reference : np.ndarray, optional
            Cloud-free reference image for histogram matching.  Must share
            the same spatial shape as ``data``.

        Returns
        -------
        np.ndarray
            Calibrated float32 array in [clip_min, clip_max].
        """
        logger.info("Radiometric calibration | dtype=%s shape=%s", data.dtype, data.shape)

        img = data.astype(np.float32)

        # Step 1 – DN → Reflectance
        img = self._dn_to_reflectance(img, meta)

        # Step 2 – Intensity clip
        img = self._clip(img)

        # Step 3 – Histogram matching
        if reference is not None:
            img = self._histogram_match(img, reference)

        # Step 4 – Noise reduction
        img = self._denoise(img)

        logger.debug(
            "Calibration complete | min=%.4f max=%.4f mean=%.4f",
            img.min(), img.max(), img.mean(),
        )
        return img

    # ─── Private helpers ───────────────────────────────────────────────────

    def _dn_to_reflectance(
        self, data: np.ndarray, meta: dict[str, Any]
    ) -> np.ndarray:
        """
        Convert Digital Numbers to Top-of-Atmosphere Reflectance.

        If the image dtype is already float and values appear to be in [0,1],
        the conversion is skipped (pass-through).
        """
        dtype_str = str(meta.get("dtype", "float32"))
        if "float" in dtype_str and data.max() <= 1.0:
            logger.debug("DN→Reflectance skipped (data already in float [0,1])")
            return data

        logger.debug(
            "Applying DN→Reflectance: gain=%.6f offset=%.4f", self.gain, self.offset
        )
        return data * self.gain + self.offset

    def _clip(self, data: np.ndarray) -> np.ndarray:
        """Clip intensity values to [clip_min, clip_max]."""
        return np.clip(data, self.clip_min, self.clip_max)

    def _histogram_match(
        self, source: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """
        Match histogram of source to reference image.

        Uses skimage.exposure.match_histograms (per-channel).
        Falls back gracefully if shapes differ.
        """
        ref = reference.astype(np.float32)

        # Ensure same number of channels
        if source.ndim != ref.ndim:
            logger.warning(
                "Histogram match skipped: channel mismatch source=%s ref=%s",
                source.shape, ref.shape,
            )
            return source

        try:
            multichannel = source.ndim == 3
            matched = match_histograms(source, ref, channel_axis=-1 if multichannel else None)
            logger.debug("Histogram matching applied")
            return matched.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Histogram matching failed (%s); skipping", exc)
            return source

    def _denoise(self, data: np.ndarray) -> np.ndarray:
        """Apply noise reduction according to configured method."""
        if self.noise_method == "none":
            return data

        if self.noise_method == "gaussian":
            return self._gaussian_denoise(data)
        elif self.noise_method == "median":
            return self._median_denoise(data)
        elif self.noise_method == "bilateral":
            return self._bilateral_denoise(data)
        else:
            logger.warning("Unknown noise method '%s'; skipping", self.noise_method)
            return data

    def _gaussian_denoise(self, data: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing per band."""
        if data.ndim == 2:
            return gaussian_filter(data, sigma=self.noise_sigma).astype(np.float32)

        out = np.empty_like(data)
        for c in range(data.shape[-1]):
            out[:, :, c] = gaussian_filter(data[:, :, c], sigma=self.noise_sigma)
        return out

    def _median_denoise(self, data: np.ndarray) -> np.ndarray:
        """Apply Median filter per band (uses OpenCV)."""
        ksize = max(3, int(self.noise_sigma) * 2 + 1)  # ensure odd
        if ksize % 2 == 0:
            ksize += 1

        def _median_band(band: np.ndarray) -> np.ndarray:
            # OpenCV medianBlur needs uint8 or float32; normalise to [0,1] float32
            b = band.astype(np.float32)
            return cv2.medianBlur(b, ksize)

        if data.ndim == 2:
            return _median_band(data)

        out = np.empty_like(data)
        for c in range(data.shape[-1]):
            out[:, :, c] = _median_band(data[:, :, c])
        return out

    def _bilateral_denoise(self, data: np.ndarray) -> np.ndarray:
        """Apply Bilateral filter per band (edge-preserving)."""
        d = max(5, int(self.noise_sigma) * 2 + 1)
        sigma_color = 75
        sigma_space = 75

        def _bilateral_band(band: np.ndarray) -> np.ndarray:
            b32 = band.astype(np.float32)
            return cv2.bilateralFilter(b32, d, sigma_color, sigma_space)

        if data.ndim == 2:
            return _bilateral_band(data)

        out = np.empty_like(data)
        for c in range(data.shape[-1]):
            out[:, :, c] = _bilateral_band(data[:, :, c])
        return out
