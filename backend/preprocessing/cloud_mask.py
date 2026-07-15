"""
cloud_mask.py - Cloud mask integration layer.

Accepts a mask file produced by the Cloud Detection teammate
(mask.png or mask.tif) and prepares it for pipeline use:
  - Load and validate binary mask
  - Validate spatial dimensions against current image
  - Auto-align (resize/warp) if shapes differ
  - Save aligned mask alongside outputs

This module contains NO cloud detection logic — it is purely
an adapter that connects the Cloud Detection module output
to the preprocessing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import SUPPORTED_EXTENSIONS, load_image, save_image

logger = logging.getLogger(__name__)


class CloudMaskHandler:
    """
    Loads, validates, and aligns a cloud mask to match the current image grid.

    The Cloud Detection module should produce one of:
    - ``mask.png``  – uint8 grayscale, 255 = cloud, 0 = clear
    - ``mask.tif``  – GeoTIFF, 1 = cloud, 0 = clear

    This handler normalises both to a float32 binary array (1.0 = cloud).
    """

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None
        self._mask_path: Path | None = None

    # ─── Public API ────────────────────────────────────────────────────────

    def load_and_prepare(
        self,
        mask_path: str | Path,
        reference_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Load a cloud mask and align it to a reference spatial shape.

        Parameters
        ----------
        mask_path : str or Path
            Path to mask.png or mask.tif from the Cloud Detection module.
        reference_shape : tuple[int, int]
            (H, W) of the current image — the mask will be resized to this.

        Returns
        -------
        np.ndarray
            Binary float32 mask, shape (H, W), values 0.0 (clear) or 1.0 (cloud).

        Raises
        ------
        FileNotFoundError
            If the mask file does not exist.
        ValueError
            If the file cannot be interpreted as a valid binary mask.
        """
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Cloud mask not found: {mask_path}")
        if mask_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported mask format: {mask_path.suffix}")

        logger.info("Loading cloud mask: %s", mask_path.name)

        raw, meta = load_image(mask_path)
        mask = self._to_binary_float32(raw)

        # Align to reference grid if shapes differ
        ref_h, ref_w = reference_shape
        if mask.shape[:2] != (ref_h, ref_w):
            logger.info(
                "Resizing cloud mask from %s to (%d, %d)",
                mask.shape[:2], ref_h, ref_w,
            )
            mask = cv2.resize(mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.float32)

        self._validate(mask)

        self._mask = mask
        self._mask_path = mask_path
        cloud_pct = float(mask.mean() * 100)
        logger.info(
            "Cloud mask ready | shape=%s cloud_coverage=%.1f%%",
            mask.shape, cloud_pct,
        )
        return mask

    def save(
        self,
        output_path: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """
        Save the prepared mask as PNG (uint8) alongside pipeline outputs.

        Parameters
        ----------
        output_path : str or Path
            Destination file (should end with .png or .tif).
        meta : dict, optional
            Rasterio metadata for GeoTIFF output.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        if self._mask is None:
            raise RuntimeError("No mask loaded. Call load_and_prepare() first.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() in {".tif", ".tiff"}:
            save_image(self._mask, out_path, meta, dtype="float32")
        else:
            # PNG: scale to uint8 (0 / 255)
            mask_u8 = (self._mask * 255).astype(np.uint8)
            cv2.imwrite(str(out_path), mask_u8)

        logger.info("Cloud mask saved → %s", out_path)
        return str(out_path)

    @property
    def cloud_coverage(self) -> float:
        """Fraction of pixels classified as cloud (0.0–1.0)."""
        if self._mask is None:
            return 0.0
        return float(self._mask.mean())

    # ─── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_binary_float32(raw: np.ndarray) -> np.ndarray:
        """
        Normalise any mask representation to float32 {0.0, 1.0}.

        Handles:
        - uint8 with values 0/255
        - uint8 with values 0/1
        - float with values 0.0/1.0
        - multi-channel (first channel used)
        """
        arr = raw
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        arr = arr.astype(np.float32)

        # Uint8 0–255 → binary
        if arr.max() > 1.5:
            arr = arr / 255.0

        # Threshold to strict binary
        return (arr > 0.5).astype(np.float32)

    @staticmethod
    def _validate(mask: np.ndarray) -> None:
        """Raise if mask contains values other than 0 and 1."""
        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                f"Mask must be binary (0/1). Found unique values: {unique}"
            )


def load_cloud_mask(
    mask_path: str | Path | None,
    reference_shape: tuple[int, int],
) -> np.ndarray | None:
    """
    Convenience function — load mask or return None gracefully.

    Parameters
    ----------
    mask_path : str, Path, or None
        Path to mask. If None or file missing, returns None.
    reference_shape : tuple[int, int]
        (H, W) to align the mask to.

    Returns
    -------
    np.ndarray or None
        Binary float32 mask or None.
    """
    if mask_path is None:
        return None
    try:
        handler = CloudMaskHandler()
        return handler.load_and_prepare(mask_path, reference_shape)
    except FileNotFoundError:
        logger.warning("Cloud mask path provided but file not found: %s", mask_path)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cloud mask loading failed (%s); continuing without mask.", exc)
        return None
