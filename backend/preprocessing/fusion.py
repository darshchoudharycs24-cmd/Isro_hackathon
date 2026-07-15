"""
fusion.py - Multi-source feature fusion.

Combines:
  - Current RGB image
  - Aligned historical RGB image
  - Optional Sentinel-1 SAR image

Supported fusion strategies:
  - channel_stack        : simple concatenation along channel axis
  - weighted_average     : weighted blend (current + historical only)
  - laplacian_pyramid    : multi-scale frequency blend

Outputs:
  - fused_image.npy  (float32, shape H×W×C)
  - fused_image.png  (uint8 RGB preview)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import stack_bands, to_uint8_png

logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Fuses current, historical, and optional SAR imagery.

    Parameters
    ----------
    method : str
        One of ``'channel_stack'``, ``'weighted_average'``,
        ``'laplacian_pyramid'``.
    weight_current : float
        Weight for current image (weighted_average only).
    weight_historical : float
        Weight for historical image (weighted_average only).
    """

    def __init__(
        self,
        method: str = "channel_stack",
        weight_current: float = 0.5,
        weight_historical: float = 0.5,
    ) -> None:
        method = method.lower()
        _valid = ("channel_stack", "weighted_average", "laplacian_pyramid")
        if method not in _valid:
            raise ValueError(f"Unknown fusion method '{method}'. Choose from {_valid}")

        self.method = method
        self.w_current = weight_current
        self.w_historical = weight_historical

    # ─── Public API ────────────────────────────────────────────────────────

    def fuse(
        self,
        current: np.ndarray,
        historical: np.ndarray,
        sar: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fuse current, historical (and optionally SAR) images.

        Parameters
        ----------
        current : np.ndarray
            Current image float32 (H, W, C).
        historical : np.ndarray
            Aligned historical image float32 (H, W, C).
            Must match ``current`` spatial dimensions.
        sar : np.ndarray, optional
            SAR image float32 (H, W) or (H, W, C).

        Returns
        -------
        fused : np.ndarray
            Fused float32 array of shape (H, W, C_fused).

        Raises
        ------
        ValueError
            If spatial dimensions do not match.
        """
        self._check_shapes(current, historical, sar)
        logger.info(
            "Fusing images | method=%s sar=%s",
            self.method, sar is not None,
        )

        if self.method == "channel_stack":
            fused = self._channel_stack(current, historical, sar)
        elif self.method == "weighted_average":
            fused = self._weighted_average(current, historical, sar)
        elif self.method == "laplacian_pyramid":
            fused = self._laplacian_pyramid(current, historical, sar)
        else:
            fused = self._channel_stack(current, historical, sar)

        logger.info(
            "Fusion complete | output shape=%s dtype=%s", fused.shape, fused.dtype
        )
        return fused

    def save_outputs(
        self,
        fused: np.ndarray,
        output_dir: str | Path,
        rgb_channels: tuple[int, int, int] = (0, 1, 2),
    ) -> dict[str, str]:
        """
        Save fused array as .npy and RGB .png.

        Parameters
        ----------
        fused : np.ndarray
            (H, W, C) float32 fused array.
        output_dir : str or Path
            Destination directory.
        rgb_channels : tuple
            Which channel indices to use for the PNG preview.

        Returns
        -------
        dict with keys ``'npy'`` and ``'png'`` pointing to saved file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / "fused_image.npy"
        png_path = out_dir / "fused_image.png"

        np.save(str(npy_path), fused)
        logger.info("Saved fused NumPy array → %s", npy_path)

        # RGB preview
        n_ch = fused.shape[-1] if fused.ndim == 3 else 1
        rgb_idx = [c for c in rgb_channels if c < n_ch]
        if len(rgb_idx) >= 3:
            rgb = fused[:, :, rgb_idx[:3]]
        elif n_ch >= 3:
            rgb = fused[:, :, :3]
        else:
            rgb = np.stack([fused[:, :, 0]] * 3, axis=-1)

        rgb_u8 = to_uint8_png(rgb)
        # OpenCV writes BGR; convert for correct colours
        cv2.imwrite(str(png_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        logger.info("Saved fused RGB PNG → %s", png_path)

        return {"npy": str(npy_path), "png": str(png_path)}

    # ─── Fusion strategies ─────────────────────────────────────────────────

    def _channel_stack(
        self,
        current: np.ndarray,
        historical: np.ndarray,
        sar: np.ndarray | None,
    ) -> np.ndarray:
        """Concatenate all available sources along the channel axis."""
        arrays = [current, historical]
        if sar is not None:
            arrays.append(sar if sar.ndim == 3 else sar[:, :, np.newaxis])
        return stack_bands(arrays)

    def _weighted_average(
        self,
        current: np.ndarray,
        historical: np.ndarray,
        sar: np.ndarray | None,
    ) -> np.ndarray:
        """
        Weighted blend of current and historical; SAR stacked on top.

        Result channels:
          - Blended RGB (same channel count as current)
          - SAR channels appended if present
        """
        total_w = self.w_current + self.w_historical
        w_c = self.w_current / total_w
        w_h = self.w_historical / total_w

        blended = w_c * current + w_h * historical

        if sar is not None:
            sar_ch = sar if sar.ndim == 3 else sar[:, :, np.newaxis]
            return stack_bands([blended, sar_ch])
        return blended

    def _laplacian_pyramid(
        self,
        current: np.ndarray,
        historical: np.ndarray,
        sar: np.ndarray | None,
        levels: int = 5,
    ) -> np.ndarray:
        """
        Laplacian pyramid blend.

        Low-frequency detail from historical; high-frequency from current.
        SAR stacked on top if present.
        """
        n_ch = current.shape[-1]
        blended_channels = []

        for c in range(n_ch):
            cur_band = current[:, :, c].astype(np.float32)
            hist_band = historical[:, :, c].astype(np.float32)
            blended_channels.append(self._blend_laplacian(cur_band, hist_band, levels))

        blended = np.stack(blended_channels, axis=-1)

        if sar is not None:
            sar_ch = sar if sar.ndim == 3 else sar[:, :, np.newaxis]
            return stack_bands([blended, sar_ch])
        return blended

    @staticmethod
    def _blend_laplacian(
        a: np.ndarray, b: np.ndarray, levels: int
    ) -> np.ndarray:
        """Single-band Laplacian pyramid blend (high-freq from a, low-freq from b)."""
        # Build Gaussian pyramids
        gp_a = [a]
        gp_b = [b]
        for _ in range(levels):
            gp_a.append(cv2.pyrDown(gp_a[-1]))
            gp_b.append(cv2.pyrDown(gp_b[-1]))

        # Build Laplacian pyramids
        lp_a, lp_b = [], []
        for i in range(levels):
            up_a = cv2.pyrUp(gp_a[i + 1], dstsize=(gp_a[i].shape[1], gp_a[i].shape[0]))
            up_b = cv2.pyrUp(gp_b[i + 1], dstsize=(gp_b[i].shape[1], gp_b[i].shape[0]))
            lp_a.append(gp_a[i] - up_a)
            lp_b.append(gp_b[i] - up_b)

        # Blend: take high-freq bands from a, low-freq from b
        blended_pyr = []
        for i in range(levels):
            # Blend weight shifts gradually from a at top (fine) to b at bottom (coarse)
            alpha = float(i) / levels
            blended_pyr.append((1 - alpha) * lp_a[i] + alpha * lp_b[i])

        # Reconstruct
        result = gp_b[levels]
        for i in range(levels - 1, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
            result = result + blended_pyr[i]

        return np.clip(result, 0.0, 1.0)

    # ─── Validation ────────────────────────────────────────────────────────

    @staticmethod
    def _check_shapes(
        current: np.ndarray,
        historical: np.ndarray,
        sar: np.ndarray | None,
    ) -> None:
        if current.shape[:2] != historical.shape[:2]:
            raise ValueError(
                f"Shape mismatch: current={current.shape[:2]} "
                f"vs historical={historical.shape[:2]}"
            )
        if sar is not None and sar.shape[:2] != current.shape[:2]:
            raise ValueError(
                f"SAR shape mismatch: sar={sar.shape[:2]} "
                f"vs current={current.shape[:2]}"
            )
