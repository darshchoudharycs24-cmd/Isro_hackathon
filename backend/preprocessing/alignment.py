"""
alignment.py - Geometric alignment of historical image to current image.

Strategy:
  1. Primary  : Feature-based (ORB → FLANN homography)
  2. Secondary: SIFT if available (OpenCV contrib)
  3. Fallback : ECC (Enhanced Correlation Coefficient) alignment
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── SIFT availability check ────────────────────────────────────────────────
try:
    _sift = cv2.SIFT_create()
    SIFT_AVAILABLE = True
    del _sift
except (AttributeError, cv2.error):
    SIFT_AVAILABLE = False
    logger.debug("SIFT not available; will fall back to ORB.")


class AlignmentError(Exception):
    """Raised when all alignment strategies fail."""


class GeometricAligner:
    """
    Aligns a source (historical) image to a target (current) image.

    Parameters
    ----------
    detector : str
        Feature detector: ``'orb'`` or ``'sift'``. Falls back to ORB if
        SIFT is unavailable.
    max_features : int
        Maximum keypoints for ORB detector.
    match_ratio : float
        Lowe's ratio test threshold.
    min_match_count : int
        Minimum good matches required for homography estimation.
    ecc_fallback : bool
        Attempt ECC alignment when feature matching fails.
    ecc_iterations : int
        Maximum ECC iterations.
    ecc_epsilon : float
        ECC convergence threshold.
    warp_mode : str
        ``'homography'`` (perspective) or ``'affine'``.
    """

    def __init__(
        self,
        detector: str = "orb",
        max_features: int = 5000,
        match_ratio: float = 0.75,
        min_match_count: int = 10,
        ecc_fallback: bool = True,
        ecc_iterations: int = 200,
        ecc_epsilon: float = 1e-5,
        warp_mode: str = "homography",
    ) -> None:
        self.detector = detector.lower()
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.min_match_count = min_match_count
        self.ecc_fallback = ecc_fallback
        self.ecc_iterations = ecc_iterations
        self.ecc_epsilon = ecc_epsilon
        self.warp_mode = warp_mode.lower()

        # Registration error (reprojection, pixels) stored after each call
        self.registration_error: float = 0.0
        # Number of RANSAC inlier matches from the last alignment call
        self.feature_matches: int = 0

    # ─── Public API ────────────────────────────────────────────────────────

    def align(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Align source image to target image.

        Parameters
        ----------
        source : np.ndarray
            Historical image (H, W) or (H, W, C), float32 [0,1].
        target : np.ndarray
            Current image (H, W) or (H, W, C), float32 [0,1].

        Returns
        -------
        aligned : np.ndarray
            Warped source image with same shape as target.
        registration_error : float
            Mean reprojection error in pixels (0.0 when ECC used).

        Raises
        ------
        AlignmentError
            When all strategies fail.
        """
        logger.info(
            "Geometric alignment | source=%s target=%s detector=%s",
            source.shape, target.shape, self.detector,
        )

        target_gray = self._to_gray(target)
        source_gray = self._to_gray(source)

        # --- Strategy 1: Feature-based ---
        aligned, error = self._feature_align(source, source_gray, target, target_gray)
        if aligned is not None:
            self.registration_error = error
            logger.info("Feature alignment succeeded | reprojection_error=%.3f px | matches=%d", error, self.feature_matches)
            return aligned, error

        # --- Strategy 2: ECC fallback ---
        if self.ecc_fallback:
            logger.warning("Feature alignment failed; trying ECC fallback.")
            aligned = self._ecc_align(source, source_gray, target, target_gray)
            if aligned is not None:
                self.registration_error = 0.0
                logger.info("ECC alignment succeeded.")
                return aligned, 0.0

        raise AlignmentError(
            "All alignment strategies failed. "
            "Check that source and target share overlapping content."
        )

    # ─── Feature-based alignment ───────────────────────────────────────────

    def _feature_align(
        self,
        source: np.ndarray,
        source_gray: np.ndarray,
        target: np.ndarray,
        target_gray: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        """Detect features, match, estimate homography, warp."""

        try:
            kp_src, des_src = self._detect_and_describe(source_gray)
            kp_tgt, des_tgt = self._detect_and_describe(target_gray)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature detection failed: %s", exc)
            return None, 0.0

        if des_src is None or des_tgt is None:
            logger.warning("No descriptors found.")
            return None, 0.0

        good_matches = self._match(des_src, des_tgt)

        if len(good_matches) < self.min_match_count:
            logger.warning(
                "Insufficient matches: %d < %d", len(good_matches), self.min_match_count
            )
            return None, 0.0

        src_pts = np.float32(
            [kp_src[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_tgt[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            logger.warning("Homography estimation failed.")
            return None, 0.0

        # Compute reprojection error on inliers
        inlier_mask = mask.ravel().astype(bool)
        error = self._reprojection_error(src_pts, dst_pts, M, inlier_mask)
        self.feature_matches = int(inlier_mask.sum())

        # Warp
        h, w = target.shape[:2]
        aligned = self._warp(source, M, w, h)
        return aligned, error

    def _detect_and_describe(
        self, gray: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Run configured feature detector on a grayscale image."""
        use_sift = self.detector == "sift" and SIFT_AVAILABLE

        if use_sift:
            detector = cv2.SIFT_create()
        else:
            detector = cv2.ORB_create(nfeatures=self.max_features)

        # OpenCV expects uint8
        gray_u8 = self._to_uint8(gray)
        kp, des = detector.detectAndCompute(gray_u8, None)
        return kp, des

    def _match(
        self, des_src: np.ndarray, des_tgt: np.ndarray
    ) -> list[cv2.DMatch]:
        """Match descriptors using FLANN (SIFT) or BFMatcher (ORB)."""
        use_sift = self.detector == "sift" and SIFT_AVAILABLE

        if use_sift:
            index_params = {"algorithm": 1, "trees": 5}
            search_params = {"checks": 50}
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            des_src_f = des_src.astype(np.float32)
            des_tgt_f = des_tgt.astype(np.float32)
            raw_matches = matcher.knnMatch(des_src_f, des_tgt_f, k=2)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_matches = matcher.knnMatch(des_src, des_tgt, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)

        logger.debug("Matches: total=%d good=%d", len(raw_matches), len(good))
        return good

    def _warp(
        self, source: np.ndarray, M: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """Apply perspective or affine warp."""
        flags = cv2.INTER_LINEAR
        border = cv2.BORDER_REFLECT_101

        if self.warp_mode == "affine" and M.shape == (3, 3):
            M = M[:2, :]  # downgrade to affine

        if M.shape == (2, 3):
            return cv2.warpAffine(source, M, (w, h), flags=flags, borderMode=border)
        return cv2.warpPerspective(source, M, (w, h), flags=flags, borderMode=border)

    # ─── ECC alignment ─────────────────────────────────────────────────────

    def _ecc_align(
        self,
        source: np.ndarray,
        source_gray: np.ndarray,
        target: np.ndarray,
        target_gray: np.ndarray,
    ) -> np.ndarray | None:
        """ECC-based alignment (cv2.findTransformECC)."""
        warp_mode = cv2.MOTION_HOMOGRAPHY
        M = np.eye(3, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iterations,
            self.ecc_epsilon,
        )

        tgt_u8 = self._to_uint8(target_gray)
        src_u8 = self._to_uint8(source_gray)

        try:
            _, M = cv2.findTransformECC(tgt_u8, src_u8, M, warp_mode, criteria)
        except cv2.error as exc:
            logger.warning("ECC alignment failed: %s", exc)
            return None

        h, w = target.shape[:2]
        return self._warp(source, M, w, h)

    # ─── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert (H,W,C) float32 to (H,W) float32 grayscale."""
        if img.ndim == 2:
            return img.astype(np.float32)
        if img.shape[-1] == 1:
            return img[:, :, 0].astype(np.float32)
        # Luminosity weights for RGB
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        n = min(img.shape[-1], 3)
        return (img[:, :, :n] * weights[:n]).sum(axis=-1)

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Normalise to [0,255] uint8."""
        lo, hi = img.min(), img.max()
        if hi - lo < 1e-8:
            return np.zeros(img.shape, dtype=np.uint8)
        norm = (img - lo) / (hi - lo)
        return (norm * 255).astype(np.uint8)

    @staticmethod
    def _reprojection_error(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        M: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> float:
        """Mean reprojection error over RANSAC inliers (pixels)."""
        if not inlier_mask.any():
            return float("inf")
        src_in = src_pts[inlier_mask]
        dst_in = dst_pts[inlier_mask]
        projected = cv2.perspectiveTransform(src_in, M)
        err = np.linalg.norm(projected - dst_in, axis=2).mean()
        return float(err)
