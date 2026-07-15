"""
interfaces.py - Strongly typed dataclasses for cross-module integration.

Defines the contract between the preprocessing pipeline and:
  - Cloud Detection module   (provides cloud_mask)
  - Reconstruction module    (consumes ReconstructionInput)
  - Streamlit Dashboard      (reads ReconstructionOutput paths)

Usage::

    from backend.preprocessing.interfaces import ReconstructionInput, ReconstructionOutput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ReconstructionInput:
    """
    Fully typed input bundle for the Generative AI reconstruction model.

    All image arrays are float32, normalised to [0, 1], shape (H, W, C).

    Parameters
    ----------
    current_image : np.ndarray
        Normalised current (cloudy) LISS-IV image.  Shape (H, W, C).
    historical_image : np.ndarray
        Aligned, normalised cloud-free historical reference.  Shape (H, W, C).
    fused_image : np.ndarray
        Multi-source fused tensor (current + historical [+ SAR]). Shape (H, W, C_fused).
    cloud_mask : np.ndarray, optional
        Binary mask from Cloud Detection module.
        1 = cloud, 0 = clear.  Shape (H, W) or (H, W, 1).
    sar_image : np.ndarray, optional
        Normalised SAR image if available.  Shape (H, W, C_sar).
    metadata : dict
        Full pipeline metadata dict (see MetadataManager.to_dict()).
    """

    current_image: np.ndarray
    historical_image: np.ndarray
    fused_image: np.ndarray
    cloud_mask: Optional[np.ndarray] = None
    sar_image: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes and dtypes on construction."""
        _check_array(self.current_image, "current_image")
        _check_array(self.historical_image, "historical_image")
        _check_array(self.fused_image, "fused_image")

        if self.cloud_mask is not None:
            _check_array(self.cloud_mask, "cloud_mask", allow_2d=True)

        if self.sar_image is not None:
            _check_array(self.sar_image, "sar_image")

        # Spatial consistency
        h, w = self.current_image.shape[:2]
        for name, arr in [
            ("historical_image", self.historical_image),
            ("fused_image", self.fused_image),
        ]:
            if arr.shape[:2] != (h, w):
                raise ValueError(
                    f"Shape mismatch: current_image is ({h},{w}) "
                    f"but {name} is {arr.shape[:2]}"
                )

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """(H, W) of all images in this bundle."""
        return self.current_image.shape[:2]

    @property
    def has_cloud_mask(self) -> bool:
        return self.cloud_mask is not None

    @property
    def has_sar(self) -> bool:
        return self.sar_image is not None


@dataclass
class ReconstructionOutput:
    """
    Paths to all files in one reconstruction-ready export package.

    Parameters
    ----------
    output_directory : Path
        Root directory for this sample's outputs.
    fused_png : Path
        RGB uint8 PNG preview of the fused image.
    fused_npy : Path
        Float32 NumPy tensor of fused image (H, W, C).
    fused_tif : Path
        GeoTIFF of fused image (preserving CRS + transform).
    metadata_json : Path
        Rich metadata JSON.
    current_png : Path
        Current image RGB preview.
    historical_png : Path
        Aligned historical image RGB preview.
    aligned_reference_tif : Path
        Aligned historical GeoTIFF.
    cloud_mask_png : Path, optional
        Cloud mask PNG (if provided).
    preview_dir : Path
        Directory with overlay / difference map / fusion preview.
    warnings : list[str]
        Any non-fatal warnings raised during processing.
    """

    output_directory: Path
    fused_png: Path
    fused_npy: Path
    fused_tif: Path
    metadata_json: Path
    current_png: Path
    historical_png: Path
    aligned_reference_tif: Path
    cloud_mask_png: Optional[Path] = None
    preview_dir: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise all paths to a JSON-friendly dict."""
        return {
            "output_directory": str(self.output_directory),
            "fused_png": str(self.fused_png),
            "fused_npy": str(self.fused_npy),
            "fused_tif": str(self.fused_tif),
            "metadata_json": str(self.metadata_json),
            "current_png": str(self.current_png),
            "historical_png": str(self.historical_png),
            "aligned_reference_tif": str(self.aligned_reference_tif),
            "cloud_mask_png": str(self.cloud_mask_png) if self.cloud_mask_png else None,
            "preview_dir": str(self.preview_dir) if self.preview_dir else None,
            "warnings": self.warnings,
        }


# ─── Internal helpers ────────────────────────────────────────────────────────

def _check_array(arr: np.ndarray, name: str, allow_2d: bool = False) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy ndarray, got {type(arr)}")
    if arr.ndim == 2 and not allow_2d:
        raise ValueError(
            f"'{name}' must be 3-D (H, W, C). "
            "Use allow_2d=True for masks."
        )
    if arr.ndim not in (2, 3):
        raise ValueError(f"'{name}' must be 2-D or 3-D, got {arr.ndim}-D")
