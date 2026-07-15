"""
exporter.py - Reconstruction-ready package export.

Generates a structured output directory for each processed sample:

    outputs/
      {sample_id}/
          current.png
          historical.png
          aligned_reference.png
          fused_image.png
          fused_image.npy
          fused_image.tif
          cloud_mask.png          (optional)
          metadata.json
          preview/
              overlay.png
              difference.png
              fusion_preview.png

Called after the pipeline completes. Returns a ReconstructionOutput
dataclass so downstream teams have zero ambiguity about paths.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .interfaces import ReconstructionOutput
from .geotiff_export import export_geotiff
from .utils import to_uint8_png

logger = logging.getLogger(__name__)


def export_for_reconstruction(
    sample_id: str,
    current: np.ndarray,
    historical: np.ndarray,
    fused: np.ndarray,
    reference_meta: dict[str, Any],
    metadata_dict: dict[str, Any],
    output_root: str | Path,
    cloud_mask: np.ndarray | None = None,
    preview_sources: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> ReconstructionOutput:
    """
    Build a reconstruction-ready export package for one sample.

    Parameters
    ----------
    sample_id : str
        Unique identifier (e.g. filename stem or UUID).
    current : np.ndarray
        Normalised current image (H, W, C) float32 [0, 1].
    historical : np.ndarray
        Aligned, normalised historical image (H, W, C) float32 [0, 1].
    fused : np.ndarray
        Fused image (H, W, C_fused) float32 [0, 1].
    reference_meta : dict
        Spatial metadata (CRS, transform, etc.) from the current image.
    metadata_dict : dict
        Rich metadata dict from MetadataManager.to_dict().
    output_root : str or Path
        Root directory. Package will be written to output_root/sample_id/.
    cloud_mask : np.ndarray, optional
        Binary float32 mask (H, W), 1.0 = cloud.
    preview_sources : dict[str, str], optional
        Mapping of label → existing PNG paths (from preview.py).
        If provided, previews are copied into the preview/ sub-directory.
    warnings : list[str], optional
        Non-fatal warnings to embed in metadata.

    Returns
    -------
    ReconstructionOutput
        Typed dataclass with all output paths.
    """
    out_dir = Path(output_root) / sample_id
    preview_dir = out_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting reconstruction package → %s", out_dir)

    # ── PNG exports ────────────────────────────────────────────────────────
    current_png = _save_rgb_png(current, out_dir / "current.png")
    historical_png = _save_rgb_png(historical, out_dir / "historical.png")
    _save_rgb_png(fused, out_dir / "fused_image.png")

    # ── NumPy tensor ───────────────────────────────────────────────────────
    fused_npy = out_dir / "fused_image.npy"
    np.save(str(fused_npy), fused)

    # ── GeoTIFFs ───────────────────────────────────────────────────────────
    fused_tif = export_geotiff(
        fused, out_dir / "fused_image.tif", reference_meta, dtype="float32"
    )
    aligned_ref_tif = export_geotiff(
        historical, out_dir / "aligned_reference.tif", reference_meta, dtype="float32"
    )

    # ── Cloud mask ────────────────────────────────────────────────────────
    cloud_mask_png: Path | None = None
    if cloud_mask is not None:
        cloud_mask_png = out_dir / "cloud_mask.png"
        mask_u8 = (cloud_mask * 255).astype(np.uint8)
        cv2.imwrite(str(cloud_mask_png), mask_u8)
        logger.info("Cloud mask saved → %s", cloud_mask_png.name)

    # ── Metadata JSON ─────────────────────────────────────────────────────
    import json  # noqa: PLC0415
    _meta = dict(metadata_dict)
    _meta["warnings"] = warnings or []
    _meta["export_sample_id"] = sample_id
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(_meta, fh, indent=2, default=str)
    logger.info("Metadata JSON saved → %s", meta_path.name)

    # ── Copy / stub previews ──────────────────────────────────────────────
    _copy_previews(preview_sources or {}, preview_dir)

    result = ReconstructionOutput(
        output_directory=out_dir,
        fused_png=out_dir / "fused_image.png",
        fused_npy=fused_npy,
        fused_tif=Path(fused_tif),
        metadata_json=meta_path,
        current_png=Path(current_png),
        historical_png=Path(historical_png),
        aligned_reference_tif=Path(aligned_ref_tif),
        cloud_mask_png=cloud_mask_png,
        preview_dir=preview_dir,
        warnings=warnings or [],
    )

    logger.info("Reconstruction package complete | %s", out_dir)
    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_rgb_png(img: np.ndarray, path: Path) -> str:
    """Save float32 [0,1] array as RGB uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = to_uint8_png(img)
    if u8.ndim == 2:
        cv2.imwrite(str(path), u8)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(u8[:, :, :3], cv2.COLOR_RGB2BGR))
    return str(path)


def _copy_previews(preview_sources: dict[str, str], dest_dir: Path) -> None:
    """Copy existing preview PNGs into the preview sub-directory."""
    _name_map = {
        "overlay": "overlay.png",
        "difference_map": "difference.png",
        "fusion_preview": "fusion_preview.png",
        "current": "current_preview.png",
        "aligned_reference": "aligned_reference_preview.png",
    }
    for label, src_path in preview_sources.items():
        dest_name = _name_map.get(label, f"{label}.png")
        dest = dest_dir / dest_name
        try:
            shutil.copy2(src_path, dest)
        except (FileNotFoundError, shutil.Error) as exc:
            logger.warning("Could not copy preview '%s': %s", label, exc)
