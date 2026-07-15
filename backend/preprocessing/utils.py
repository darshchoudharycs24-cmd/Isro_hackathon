"""
utils.py - Shared I/O and conversion utilities.

Provides low-level helpers for loading/saving raster images,
dtype conversions, CRS handling, and logging setup.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

# ─── Supported formats ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ─── Image I/O ──────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a raster image from disk.

    Supports GeoTIFF, TIFF, PNG, JPEG. Automatically extracts
    resolution, CRS, band count, dtype, nodata, and transform.

    Parameters
    ----------
    path : str or Path
        Path to input image.

    Returns
    -------
    data : np.ndarray
        Image array of shape (H, W) or (H, W, C).
    meta : dict
        Metadata dict containing crs, transform, resolution,
        n_bands, dtype, nodata, width, height, file_path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.
    RuntimeError
        If the file cannot be opened/read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    try:
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
            crs = src.crs
            transform = src.transform
            nodata = src.nodata

            # Pixel resolution in CRS units (usually metres or degrees)
            res_x = abs(transform.a)
            res_y = abs(transform.e)

            meta: dict[str, Any] = {
                "file_path": str(path),
                "crs": crs.to_wkt() if crs else None,
                "crs_obj": crs,
                "transform": transform,
                "resolution": (res_x, res_y),
                "n_bands": src.count,
                "dtype": src.dtypes[0],
                "nodata": nodata,
                "width": src.width,
                "height": src.height,
                "driver": src.driver,
            }

    except rasterio.errors.RasterioIOError as exc:
        raise RuntimeError(f"Failed to open image '{path}': {exc}") from exc

    # Convert (C, H, W) → (H, W, C), squeeze single-band to (H, W)
    data = np.moveaxis(data, 0, -1)
    if data.shape[-1] == 1:
        data = data[:, :, 0]

    logger.debug(
        "Loaded '%s' | shape=%s dtype=%s CRS=%s res=%s",
        path.name,
        data.shape,
        data.dtype,
        "present" if crs else "missing",
        meta["resolution"],
    )
    return data, meta


def save_image(
    data: np.ndarray,
    path: str | Path,
    meta: dict[str, Any] | None = None,
    dtype: str | None = None,
) -> None:
    """
    Save a numpy array as a GeoTIFF (or inferred format from extension).

    Parameters
    ----------
    data : np.ndarray
        (H, W) or (H, W, C) array.
    path : str or Path
        Output file path.
    meta : dict, optional
        Rasterio-compatible metadata. If None, writes a plain raster.
    dtype : str, optional
        Override output dtype. Defaults to data.dtype.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        arr = data[np.newaxis, ...]  # (1, H, W)
    else:
        arr = np.moveaxis(data, -1, 0)  # (C, H, W)

    out_dtype = dtype or str(data.dtype)
    count, height, width = arr.shape

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": out_dtype,
    }

    if meta:
        if meta.get("crs_obj"):
            profile["crs"] = meta["crs_obj"]
        if meta.get("transform"):
            profile["transform"] = meta["transform"]
        if meta.get("nodata") is not None:
            profile["nodata"] = meta["nodata"]

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)

    logger.debug("Saved '%s' | shape=%s dtype=%s", path.name, data.shape, out_dtype)


def to_uint8_png(data: np.ndarray) -> np.ndarray:
    """
    Convert a float32 [0,1] array to uint8 [0,255] suitable for PNG export.

    Parameters
    ----------
    data : np.ndarray
        Float32 array, values expected in [0, 1].

    Returns
    -------
    np.ndarray
        uint8 array in [0, 255].
    """
    clipped = np.clip(data, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def to_float32_tensor(data: np.ndarray) -> np.ndarray:
    """
    Convert any numeric array to float32, normalised to [0, 1].

    Parameters
    ----------
    data : np.ndarray
        Input array (any dtype).

    Returns
    -------
    np.ndarray
        float32 array in [0, 1], shape (H, W, C).
    """
    arr = data.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Per-channel min-max
    for c in range(arr.shape[-1]):
        lo, hi = arr[:, :, c].min(), arr[:, :, c].max()
        if hi - lo > 1e-8:
            arr[:, :, c] = (arr[:, :, c] - lo) / (hi - lo)
        else:
            arr[:, :, c] = 0.0
    return arr


def stack_bands(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Stack a list of (H, W) or (H, W, C) arrays along the channel axis.

    Parameters
    ----------
    arrays : list of np.ndarray
        All arrays must share (H, W).

    Returns
    -------
    np.ndarray
        (H, W, total_channels) array.
    """
    expanded = []
    for a in arrays:
        if a.ndim == 2:
            expanded.append(a[:, :, np.newaxis])
        else:
            expanded.append(a)
    return np.concatenate(expanded, axis=-1)


def ensure_float32(data: np.ndarray) -> np.ndarray:
    """Cast array to float32 without normalisation."""
    return data.astype(np.float32)


def check_spatial_consistency(
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
) -> dict[str, bool]:
    """
    Compare two image metadata dicts for spatial compatibility.

    Returns
    -------
    dict with keys: same_crs, same_resolution, same_size.
    """
    def _parse_crs(meta: dict) -> CRS | None:
        # Prefer the live object; fall back to string (WKT or authority)
        if meta.get("crs_obj"):
            return meta["crs_obj"]
        crs_str = meta.get("crs")
        if not crs_str:
            return None
        try:
            return CRS.from_user_input(crs_str)
        except Exception:  # noqa: BLE001
            return None

    crs_a = _parse_crs(meta_a)
    crs_b = _parse_crs(meta_b)

    same_crs = (crs_a is not None and crs_b is not None and crs_a == crs_b)
    same_res = (
        abs(meta_a["resolution"][0] - meta_b["resolution"][0]) < 1e-6
        and abs(meta_a["resolution"][1] - meta_b["resolution"][1]) < 1e-6
    )
    same_size = (
        meta_a["width"] == meta_b["width"]
        and meta_a["height"] == meta_b["height"]
    )
    return {"same_crs": same_crs, "same_resolution": same_res, "same_size": same_size}


# ─── Timing context manager ─────────────────────────────────────────────────

@contextmanager
def timer(label: str):
    """Context manager that logs elapsed time for a labelled pipeline step."""
    start = time.perf_counter()
    logger.info("▶ [%s] started", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("✔ [%s] completed in %.2f s", label, elapsed)
