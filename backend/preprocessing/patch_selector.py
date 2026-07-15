"""
patch_selector.py - Automatic best-reference-image selector.

Scores every candidate historical image on four criteria:
  1. Date proximity     – prefer images closest in time
  2. Feature similarity – ORB-based descriptor distance
  3. SSIM               – structural similarity to current image
  4. Histogram correlation

Returns the path to the single best historical image.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .utils import load_image

logger = logging.getLogger(__name__)

_DATE_FMT_PATTERNS = [
    "%Y%m%d",
    "%Y-%m-%d",
    "%Y_%m_%d",
    "%d%m%Y",
    "%d-%m-%Y",
]


def _parse_date_from_filename(name: str) -> date | None:
    """Try to parse a date from a filename stem using common patterns."""
    stem = Path(name).stem
    # Try substrings of length 8 (YYYYMMDD) up to full stem
    for token in [stem] + [stem[i: i + 8] for i in range(len(stem))]:
        for fmt in _DATE_FMT_PATTERNS:
            try:
                return datetime.strptime(token, fmt).date()
            except ValueError:
                continue
    return None


class PatchSelector:
    """
    Selects the best cloud-free historical image from a folder.

    Parameters
    ----------
    date_weight : float
    ssim_weight : float
    histogram_weight : float
    feature_weight : float
        Importance weights (must sum to 1.0, automatically normalised if not).
    max_date_diff_days : int
        Historical images beyond this age receive a date score of 0.
    """

    def __init__(
        self,
        date_weight: float = 0.25,
        ssim_weight: float = 0.35,
        histogram_weight: float = 0.25,
        feature_weight: float = 0.15,
        max_date_diff_days: int = 365,
    ) -> None:
        total = date_weight + ssim_weight + histogram_weight + feature_weight
        self.w_date = date_weight / total
        self.w_ssim = ssim_weight / total
        self.w_hist = histogram_weight / total
        self.w_feat = feature_weight / total
        self.max_date_diff = max_date_diff_days

    # ─── Public API ────────────────────────────────────────────────────────

    def select_best(
        self,
        current_image: np.ndarray,
        historical_dir: str | Path,
        current_date: date | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """
        Score all candidate images and return the best one.

        Parameters
        ----------
        current_image : np.ndarray
            Current (possibly cloudy) image, float32.
        historical_dir : str or Path
            Directory containing candidate historical images.
        current_date : date, optional
            Acquisition date of current image. If None, date scoring is skipped.

        Returns
        -------
        best_path : Path
            Path to the best-scoring historical image.
        scores : dict
            Per-image score breakdown keyed by filename.

        Raises
        ------
        FileNotFoundError
            If no valid images are found in the directory.
        """
        hist_dir = Path(historical_dir)
        candidates = self._find_candidates(hist_dir)

        if not candidates:
            raise FileNotFoundError(
                f"No valid images found in historical directory: {hist_dir}"
            )

        if len(candidates) == 1:
            logger.info("Single historical image found; skipping scoring.")
            return candidates[0], {}

        logger.info("Scoring %d historical candidates.", len(candidates))

        cur_gray = self._to_gray_u8(current_image)
        scores: dict[str, dict[str, float]] = {}
        totals: dict[str, float] = {}

        for path in candidates:
            try:
                hist_data, _ = load_image(path)
                hist_gray = self._to_gray_u8(hist_data)

                # Resize to current for fair comparison
                if hist_gray.shape != cur_gray.shape:
                    hist_gray = cv2.resize(
                        hist_gray, (cur_gray.shape[1], cur_gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                s_date = self._score_date(path, current_date)
                s_ssim = self._score_ssim(cur_gray, hist_gray)
                s_hist = self._score_histogram(cur_gray, hist_gray)
                s_feat = self._score_features(cur_gray, hist_gray)

                total = (
                    self.w_date * s_date
                    + self.w_ssim * s_ssim
                    + self.w_hist * s_hist
                    + self.w_feat * s_feat
                )

                scores[path.name] = {
                    "date_score": s_date,
                    "ssim_score": s_ssim,
                    "histogram_score": s_hist,
                    "feature_score": s_feat,
                    "total": total,
                }
                totals[path.name] = total

                logger.debug(
                    "%s | date=%.3f ssim=%.3f hist=%.3f feat=%.3f → total=%.3f",
                    path.name, s_date, s_ssim, s_hist, s_feat, total,
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not score '%s': %s", path.name, exc)

        if not totals:
            raise RuntimeError("All historical images failed scoring.")

        best_name = max(totals, key=totals.__getitem__)
        best_path = hist_dir / best_name
        logger.info("Best historical image: '%s' (score=%.3f)", best_name, totals[best_name])
        return best_path, scores

    # ─── Scoring functions ─────────────────────────────────────────────────

    def _score_date(self, path: Path, current_date: date | None) -> float:
        """Score in [0,1]: 1.0 = same day, 0.0 = max_date_diff days apart."""
        if current_date is None:
            return 0.5  # neutral

        hist_date = _parse_date_from_filename(path.name)
        if hist_date is None:
            return 0.5  # neutral when date unavailable

        diff = abs((current_date - hist_date).days)
        score = max(0.0, 1.0 - diff / self.max_date_diff)
        return float(score)

    @staticmethod
    def _score_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """SSIM score in [0, 1]."""
        try:
            val = ssim(a, b, data_range=255)
            return float(np.clip(val, 0.0, 1.0))
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _score_histogram(a: np.ndarray, b: np.ndarray) -> float:
        """Normalised histogram correlation in [0, 1]."""
        hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] → [0,1]

    @staticmethod
    def _score_features(a: np.ndarray, b: np.ndarray) -> float:
        """
        ORB-based feature similarity in [0, 1].

        Score = good_matches / (max_features / 2) capped at 1.0.
        """
        try:
            orb = cv2.ORB_create(nfeatures=500)
            _, des_a = orb.detectAndCompute(a, None)
            _, des_b = orb.detectAndCompute(b, None)

            if des_a is None or des_b is None:
                return 0.0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_a, des_b)
            score = min(1.0, len(matches) / 250.0)
            return float(score)
        except Exception:  # noqa: BLE001
            return 0.0

    # ─── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_candidates(hist_dir: Path) -> list[Path]:
        """Return all supported image files in the directory (non-recursive)."""
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        return sorted(p for p in hist_dir.iterdir() if p.suffix.lower() in exts)

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        """Convert any image to uint8 grayscale for scoring."""
        arr = img.astype(np.float32)
        if arr.ndim == 3:
            n = min(arr.shape[-1], 3)
            w = np.array([0.2126, 0.7152, 0.0722][:n], dtype=np.float32)
            arr = (arr[:, :, :n] * w).sum(axis=-1)

        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-8:
            arr = (arr - lo) / (hi - lo)
        return (arr * 255).astype(np.uint8)
