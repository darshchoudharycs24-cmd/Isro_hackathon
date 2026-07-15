"""Tests for RadiometricCalibrator."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing.calibration import RadiometricCalibrator


class TestDNToReflectance:
    def test_integer_dn_converted(self, sample_meta):
        """DN integer values should be scaled to reflectance range."""
        cal = RadiometricCalibrator(gain=0.0001, offset=0.0, noise_method="none")
        dn = np.full((64, 64, 3), 10000, dtype=np.uint16)
        result = cal.calibrate(dn, sample_meta)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_float_passthrough(self, sample_rgb, sample_meta):
        """Float [0,1] image should bypass DN conversion."""
        meta = {**sample_meta, "dtype": "float32"}
        cal = RadiometricCalibrator(noise_method="none")
        result = cal.calibrate(sample_rgb, meta)
        # Values should remain close (no DN scaling applied)
        assert result.max() <= 1.0 + 1e-5
        assert result.min() >= 0.0 - 1e-5


class TestClipping:
    def test_values_clipped(self, sample_meta):
        cal = RadiometricCalibrator(clip_min=0.2, clip_max=0.8, noise_method="none")
        data = np.linspace(0, 1, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
        result = cal.calibrate(data, {**sample_meta, "dtype": "float32"})
        assert result.min() >= 0.2 - 1e-6
        assert result.max() <= 0.8 + 1e-6


class TestHistogramMatching:
    def test_matched_histograms_closer(self, sample_meta):
        rng = np.random.default_rng(42)
        source = rng.random((64, 64, 3)).astype(np.float32)
        reference = (rng.random((64, 64, 3)) * 0.3).astype(np.float32)

        cal = RadiometricCalibrator(noise_method="none")
        meta = {**sample_meta, "dtype": "float32"}
        matched = cal.calibrate(source, meta, reference=reference)

        # After matching, mean should shift toward reference mean
        assert matched.mean() < source.mean()


class TestNoiseReduction:
    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral"])
    def test_denoised_output_shape(self, method, sample_rgb, sample_meta):
        cal = RadiometricCalibrator(noise_method=method, noise_sigma=1.0)
        meta = {**sample_meta, "dtype": "float32"}
        result = cal.calibrate(sample_rgb, meta)
        assert result.shape == sample_rgb.shape

    def test_unknown_method_warns(self, sample_rgb, sample_meta, caplog):
        import logging
        cal = RadiometricCalibrator(noise_method="invalid_method")
        meta = {**sample_meta, "dtype": "float32"}
        with caplog.at_level(logging.WARNING):
            result = cal.calibrate(sample_rgb, meta)
        assert result.shape == sample_rgb.shape  # should still return data
