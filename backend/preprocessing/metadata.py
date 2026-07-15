"""
metadata.py - Pipeline metadata collection and JSON export.

Collects processing parameters, image properties, and timing info
into a structured JSON document for downstream model consumption.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Accumulates metadata throughout the pipeline and serialises it to JSON.

    Usage::

        mgr = MetadataManager()
        mgr.set_image_info("image_id", current_meta)
        mgr.set_registration_error(1.23)
        mgr.set_normalization_stats(normalizer.band_stats)
        mgr.set_fusion_info(method="channel_stack", sar_used=True)
        mgr.save("output/metadata.json")
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            # ── Core image info ──────────────────────────────────────────
            "image_id": None,
            "resolution": None,
            "CRS": None,
            "bands": None,
            "width": None,
            "height": None,
            "dtype": None,
            # ── Registration ─────────────────────────────────────────────
            "historical_image_used": None,
            "registration_error": None,       # reprojection RMSE (pixels)
            "registration_rmse": None,        # alias kept for rich schema
            "feature_matches": None,          # number of RANSAC inlier matches
            # ── Quality scores ───────────────────────────────────────────
            "ssim_score": None,
            "histogram_score": None,
            # ── Normalisation ────────────────────────────────────────────
            "normalization": None,
            # ── Fusion & extras ──────────────────────────────────────────
            "fusion_method": None,
            "SAR_used": False,
            "cloud_mask_used": False,
            # ── Timing & version ─────────────────────────────────────────
            "processing_time": None,
            "pipeline_version": "1.0.0",
        }
        self._start_time: float = time.perf_counter()

    # ─── Setters ───────────────────────────────────────────────────────────

    def set_image_info(
        self,
        image_id: str,
        meta: dict[str, Any],
    ) -> None:
        """
        Populate core image properties from rasterio metadata dict.

        Parameters
        ----------
        image_id : str
            Human-readable identifier (e.g. filename stem).
        meta : dict
            Metadata returned by ``utils.load_image``.
        """
        self._data["image_id"] = image_id
        self._data["resolution"] = meta.get("resolution")
        self._data["CRS"] = meta.get("crs")
        self._data["bands"] = meta.get("n_bands")
        self._data["width"] = meta.get("width")
        self._data["height"] = meta.get("height")
        self._data["dtype"] = meta.get("dtype")

    def set_historical_image(self, path: str | Path) -> None:
        """Record which historical image was selected as the reference."""
        self._data["historical_image_used"] = str(path)

    def set_registration_error(self, error: float, feature_matches: int = 0) -> None:
        """Record mean reprojection error in pixels and inlier match count."""
        self._data["registration_error"] = round(error, 4)
        self._data["registration_rmse"] = round(error, 4)
        self._data["feature_matches"] = feature_matches

    def set_quality_scores(
        self,
        ssim_score: float | None = None,
        histogram_score: float | None = None,
    ) -> None:
        """Record patch-selection quality scores for the chosen historical image."""
        if ssim_score is not None:
            self._data["ssim_score"] = round(float(ssim_score), 4)
        if histogram_score is not None:
            self._data["histogram_score"] = round(float(histogram_score), 4)

    def set_cloud_mask_used(self, used: bool) -> None:
        """Record whether a cloud mask was applied."""
        self._data["cloud_mask_used"] = used

    def set_normalization_stats(
        self,
        band_stats: list[dict[str, float]],
        method: str = "minmax",
    ) -> None:
        """Record per-band normalisation statistics."""
        self._data["normalization"] = {
            "method": method,
            "band_stats": band_stats,
        }

    def set_fusion_info(
        self,
        method: str,
        sar_used: bool = False,
        sar_path: str | None = None,
    ) -> None:
        """Record fusion method and SAR usage."""
        self._data["fusion_method"] = method
        self._data["SAR_used"] = sar_used
        if sar_path:
            self._data["sar_path"] = str(sar_path)

    def set_processing_time(self, seconds: float | None = None) -> None:
        """
        Record total pipeline processing time.

        If ``seconds`` is None, computes elapsed time since object creation.
        """
        if seconds is None:
            seconds = time.perf_counter() - self._start_time
        self._data["processing_time"] = round(seconds, 3)

    def update(self, extra: dict[str, Any]) -> None:
        """Merge arbitrary key-value pairs into the metadata."""
        self._data.update(extra)

    # ─── Accessors ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the metadata dict."""
        return dict(self._data)

    def save(self, output_path: str | Path) -> None:
        """
        Write metadata to a JSON file.

        Parameters
        ----------
        output_path : str or Path
            Destination .json file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=str)

        logger.info("Metadata saved → %s", path)

    @classmethod
    def load(cls, json_path: str | Path) -> "MetadataManager":
        """
        Load a previously saved metadata JSON back into a MetadataManager.

        Parameters
        ----------
        json_path : str or Path
            Path to the JSON file.

        Returns
        -------
        MetadataManager
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        mgr = cls()
        with open(path, encoding="utf-8") as fh:
            mgr._data = json.load(fh)
        return mgr
