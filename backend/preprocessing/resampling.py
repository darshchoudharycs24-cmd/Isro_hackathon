"""
resampling.py - Spatial resampling to match target resolution and extent.

Wraps rasterio's reproject-to-match logic with a clean numpy interface,
supporting nearest-neighbour, bilinear, and cubic resampling.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling as RasterioResampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

logger = logging.getLogger(__name__)

# Map config string → rasterio enum
_RESAMPLE_METHODS: dict[str, RasterioResampling] = {
    "nearest": RasterioResampling.nearest,
    "bilinear": RasterioResampling.bilinear,
    "cubic": RasterioResampling.cubic,
    "lanczos": RasterioResampling.lanczos,
    "average": RasterioResampling.average,
}


class ImageResampler:
    """
    Resamples imagery to match a reference image's resolution and spatial extent.

    Parameters
    ----------
    method : str
        Resampling method: ``'nearest'``, ``'bilinear'``, ``'cubic'``.
        Defaults to ``'bilinear'``.
    """

    def __init__(self, method: str = "bilinear") -> None:
        method = method.lower()
        if method not in _RESAMPLE_METHODS:
            raise ValueError(
                f"Unknown resampling method '{method}'. "
                f"Choose from: {list(_RESAMPLE_METHODS)}"
            )
        self.method = method
        self._rio_method = _RESAMPLE_METHODS[method]

    # ─── Public API ────────────────────────────────────────────────────────

    def resample_to_match(
        self,
        source: np.ndarray,
        source_meta: dict[str, Any],
        target_meta: dict[str, Any],
    ) -> np.ndarray:
        """
        Resample source array to match the grid of the target image.

        If resolution and size already match, returns the source unchanged.

        Parameters
        ----------
        source : np.ndarray
            Source array (H, W) or (H, W, C). Any dtype.
        source_meta : dict
            Metadata dict from ``utils.load_image`` for the source.
        target_meta : dict
            Metadata dict from ``utils.load_image`` for the target.

        Returns
        -------
        np.ndarray
            Resampled array matching target spatial dimensions, same dtype
            as source.
        """
        src_h, src_w = source.shape[:2]
        tgt_h = target_meta["height"]
        tgt_w = target_meta["width"]

        # Check if already matching
        if src_h == tgt_h and src_w == tgt_w:
            res_match = (
                abs(source_meta["resolution"][0] - target_meta["resolution"][0]) < 1e-6
            )
            if res_match:
                logger.debug("Resampling skipped (already matches target grid)")
                return source

        logger.info(
            "Resampling %dx%d → %dx%d using '%s'",
            src_w, src_h, tgt_w, tgt_h, self.method,
        )

        return self._resample(source, source_meta, target_meta, tgt_w, tgt_h)

    def resample_to_shape(
        self,
        source: np.ndarray,
        source_meta: dict[str, Any],
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        """
        Resample source to an explicit (width, height) without a reference meta.

        Parameters
        ----------
        source : np.ndarray
            Input array.
        source_meta : dict
            Source image metadata.
        target_width : int
        target_height : int

        Returns
        -------
        np.ndarray
            Resampled array of shape (target_height, target_width[, C]).
        """
        logger.info(
            "Resampling to explicit shape %dx%d using '%s'",
            target_width, target_height, self.method,
        )
        # Build synthetic target meta that shares source bounds
        t = source_meta.get("transform")
        if t is not None:
            from rasterio.transform import Affine  # noqa: PLC0415
            left = t.c
            top = t.f
            right = left + source_meta["width"] * t.a
            bottom = top + source_meta["height"] * t.e
            target_transform = from_bounds(left, bottom, right, top, target_width, target_height)
        else:
            target_transform = None

        synthetic_meta: dict[str, Any] = {
            **source_meta,
            "width": target_width,
            "height": target_height,
            "transform": target_transform,
        }
        return self._resample(source, source_meta, synthetic_meta, target_width, target_height)

    # ─── Internal ──────────────────────────────────────────────────────────

    def _resample(
        self,
        source: np.ndarray,
        source_meta: dict[str, Any],
        target_meta: dict[str, Any],
        tgt_w: int,
        tgt_h: int,
    ) -> np.ndarray:
        """
        Core resampling using rasterio.warp.reproject.

        Works per-band to handle arbitrary channel counts.
        """
        original_dtype = source.dtype
        src_float = source.astype(np.float32)

        # Normalise to (C, H, W)
        if src_float.ndim == 2:
            src_chw = src_float[np.newaxis, ...]
        else:
            src_chw = np.moveaxis(src_float, -1, 0)

        n_bands = src_chw.shape[0]
        out_chw = np.zeros((n_bands, tgt_h, tgt_w), dtype=np.float32)

        src_crs = source_meta.get("crs_obj")
        tgt_crs = target_meta.get("crs_obj")
        src_transform = source_meta.get("transform")
        tgt_transform = target_meta.get("transform")

        for b in range(n_bands):
            reproject(
                source=src_chw[b],
                destination=out_chw[b],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=tgt_transform,
                dst_crs=tgt_crs if tgt_crs else src_crs,
                resampling=self._rio_method,
            )

        # Back to (H, W, C) or (H, W)
        out = np.moveaxis(out_chw, 0, -1)
        if out.shape[-1] == 1:
            out = out[:, :, 0]

        return out.astype(original_dtype)
