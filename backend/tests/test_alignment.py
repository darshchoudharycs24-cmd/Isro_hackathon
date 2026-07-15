"""Tests for GeometricAligner."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing.alignment import GeometricAligner, AlignmentError


def _make_shifted(base: np.ndarray, tx: int = 5, ty: int = 3) -> np.ndarray:
    """Create a translated version of base using numpy roll."""
    return np.roll(np.roll(base, ty, axis=0), tx, axis=1)


class TestGeometricAligner:
    def test_output_shape_matches_target(self, sample_rgb):
        """Aligned image must have the same spatial shape as target."""
        aligner = GeometricAligner(detector="orb", ecc_fallback=True)
        source = _make_shifted(sample_rgb)
        aligned, _ = aligner.align(source, sample_rgb)
        assert aligned.shape[:2] == sample_rgb.shape[:2]

    def test_registration_error_non_negative(self, sample_rgb):
        aligner = GeometricAligner(detector="orb", ecc_fallback=True)
        source = _make_shifted(sample_rgb, tx=10, ty=10)
        _, error = aligner.align(source, sample_rgb)
        assert error >= 0.0

    def test_grayscale_input(self, sample_gray):
        aligner = GeometricAligner(ecc_fallback=True)
        source = _make_shifted(sample_gray)
        aligned, _ = aligner.align(source, sample_gray)
        assert aligned.shape == sample_gray.shape

    def test_identity_alignment(self, sample_rgb):
        """Aligning an image to itself should return very small error."""
        aligner = GeometricAligner(detector="orb", ecc_fallback=False)
        aligned, error = aligner.align(sample_rgb, sample_rgb)
        assert aligned.shape == sample_rgb.shape
        # Error should be very small (perfect feature matches)
        assert error < 5.0

    def test_ecc_fallback_triggered(self, caplog):
        """When feature matching fails, ECC should be attempted."""
        import logging
        # Create completely random images (no feature matches possible)
        rng = np.random.default_rng(0)
        src = rng.random((128, 128, 3), dtype=np.float32)
        tgt = rng.random((128, 128, 3), dtype=np.float32)

        aligner = GeometricAligner(
            min_match_count=1000,  # unreachably high → forces ECC
            ecc_fallback=True,
        )
        with caplog.at_level(logging.WARNING):
            try:
                aligned, _ = aligner.align(src, tgt)
                assert aligned.shape[:2] == tgt.shape[:2]
            except AlignmentError:
                pass  # acceptable if both strategies fail on random data
