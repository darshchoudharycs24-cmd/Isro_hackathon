"""
geotiff_export.py - Analysis-ready GeoTIFF export with full spatial metadata.

Exports any pipeline array as a GeoTIFF preserving:
  - CRS
  - Affine transform
  - dtype
  - nodata value
  - Band descriptions

This is separate from utils.save_image so exports remain audit-friendly
with full rasterio metadata and optional LZW compression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


def export_geotiff(
    data: np.ndarray,
    output_path: str | Path,
    reference_meta: dict[str, Any],
    dtype: str | None = None,
    compress: str = "lzw",
    band_descriptions: list[str] | None = None,
    nodata: float | None = None,
) -> str:
    """
    Export a numpy array to a fully geo-referenced GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        Image array (H, W) or (H, W, C). Any numeric dtype.
    output_path : str or Path
        Destination .tif file.
    reference_meta : dict
        Metadata dict from ``utils.load_image`` used to supply CRS, transform,
        nodata, width, height.
    dtype : str, optional
        Override output dtype (e.g. ``'float32'``, ``'uint8'``).
        Defaults to data.dtype.
    compress : str
        Rasterio compression: ``'lzw'`` (default), ``'deflate'``, or ``'none'``.
    band_descriptions : list[str], optional
        Human-readable band names written into the GeoTIFF metadata.
    nodata : float, optional
        Override nodata value. Falls back to reference_meta nodata.

    Returns
    -------
    str
        Absolute path to the written GeoTIFF.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise to (C, H, W)
    arr = data
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    else:
        arr = np.moveaxis(arr, -1, 0)  # (H,W,C) → (C,H,W)

    out_dtype = dtype or str(data.dtype)
    count, height, width = arr.shape

    # Cast to target dtype
    try:
        arr = arr.astype(out_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot cast to dtype '{out_dtype}': {exc}") from exc

    # Build rasterio profile
    crs: CRS | None = reference_meta.get("crs_obj")
    transform = reference_meta.get("transform")
    _nodata = nodata if nodata is not None else reference_meta.get("nodata")

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": out_dtype,
    }
    if crs:
        profile["crs"] = crs
    if transform:
        profile["transform"] = transform
    if _nodata is not None:
        profile["nodata"] = _nodata
    if compress.lower() != "none":
        profile["compress"] = compress.lower()
        profile["tiled"] = True
        profile["blockxsize"] = 256
        profile["blockysize"] = 256

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        if band_descriptions:
            for i, desc in enumerate(band_descriptions[:count], start=1):
                dst.update_tags(i, description=desc)

    logger.info(
        "GeoTIFF exported → %s | bands=%d dtype=%s crs=%s",
        out_path.name,
        count,
        out_dtype,
        "present" if crs else "missing",
    )
    return str(out_path)


def export_pipeline_geotiffs(
    current: np.ndarray,
    aligned_reference: np.ndarray,
    fused: np.ndarray,
    reference_meta: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Export all primary pipeline arrays as GeoTIFFs in one call.

    Parameters
    ----------
    current : np.ndarray
        Normalised current image (H, W, C) float32.
    aligned_reference : np.ndarray
        Aligned historical image (H, W, C) float32.
    fused : np.ndarray
        Fused multi-channel image (H, W, C_fused) float32.
    reference_meta : dict
        Spatial metadata from current image.
    output_dir : str or Path
        Directory to write TIFFs into.

    Returns
    -------
    dict
        Mapping of label → file path for:
        ``current_tif``, ``aligned_reference_tif``, ``fused_tif``.
    """
    out_dir = Path(output_dir)
    n_bands_cur = current.shape[-1] if current.ndim == 3 else 1
    n_bands_hist = aligned_reference.shape[-1] if aligned_reference.ndim == 3 else 1
    n_bands_fused = fused.shape[-1] if fused.ndim == 3 else 1

    cur_bands = [f"current_band_{i+1}" for i in range(n_bands_cur)]
    hist_bands = [f"historical_band_{i+1}" for i in range(n_bands_hist)]
    fused_bands = _fused_band_names(n_bands_fused, n_bands_cur)

    return {
        "current_tif": export_geotiff(
            current, out_dir / "current.tif", reference_meta,
            dtype="float32", band_descriptions=cur_bands,
        ),
        "aligned_reference_tif": export_geotiff(
            aligned_reference, out_dir / "aligned_reference.tif", reference_meta,
            dtype="float32", band_descriptions=hist_bands,
        ),
        "fused_tif": export_geotiff(
            fused, out_dir / "fused_image.tif", reference_meta,
            dtype="float32", band_descriptions=fused_bands,
        ),
    }


def _fused_band_names(total: int, n_optical: int) -> list[str]:
    """Generate human-readable fused band descriptions."""
    names = []
    for i in range(total):
        if i < n_optical:
            names.append(f"current_band_{i+1}")
        elif i < 2 * n_optical:
            names.append(f"historical_band_{i - n_optical + 1}")
        else:
            names.append(f"sar_band_{i - 2 * n_optical + 1}")
    return names
