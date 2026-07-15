"""Tests for BandNormalizer."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing.normalization import BandNormalizer


class TestMinMaxNormalisation:
    def test_output_range(self, sample_rgb):
        norm = BandNormalizer(method="minmax", clip_percentile=(0, 100))
        result = norm.normalise(sample_rgb)
        assert result.dtype == np.float32
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_single_band(self, sample_gray):
        norm = BandNormalizer(method="minmax")
        result = norm.normalise(sample_gray)
        assert result.shape == sample_gray.shape
        assert result.max() <= 1.0 + 1e-6

    def test_constant_band_returns_zeros(self):
        norm = BandNormalizer(method="minmax", clip_percentile=(0, 100))
        data = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = norm.normalise(data)
        # Constant band should be all zeros (no range)
        np.testing.assert_array_equal(result[:, :, 0], 0.0)

    def test_band_stats_populated(self, sample_rgb):
        norm = BandNormalizer(method="minmax")
        norm.normalise(sample_rgb)
        assert len(norm.band_stats) == 3
        for s in norm.band_stats:
            assert "min" in s and "max" in s


class TestMeanStdNormalisation:
    def test_output_dtype(self, sample_rgb):
        norm = BandNormalizer(method="meanstd")
        result = norm.normalise(sample_rgb)
        assert result.dtype == np.float32

    def test_approximately_zero_mean(self, sample_rgb):
        norm = BandNormalizer(method="meanstd", clip_percentile=(0, 100))
        result = norm.normalise(sample_rgb)
        # After z-score, mean should be very close to 0
        assert abs(result[:, :, 0].mean()) < 0.1


class TestDenormalisation:
    def test_roundtrip_minmax(self, sample_rgb):
        norm = BandNormalizer(method="minmax", clip_percentile=(0, 100))
        normalised = norm.normalise(sample_rgb)
        restored = norm.denormalise(normalised)
        np.testing.assert_allclose(restored, sample_rgb, atol=1e-4)

    def test_denormalise_without_stats_raises(self, sample_rgb):
        norm = BandNormalizer(method="minmax")
        with pytest.raises(RuntimeError, match="No band stats"):
            norm.denormalise(sample_rgb)


class TestInvalidMethod:
    def test_raises_on_bad_method(self):
        with pytest.raises(ValueError, match="Unknown normalisation method"):
            BandNormalizer(method="invalid")
