"""Tests for FeatureFusion."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing.fusion import FeatureFusion


def _img(h=64, w=64, c=3, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, c)).astype(np.float32)


class TestChannelStack:
    def test_shape_no_sar(self):
        fuse = FeatureFusion(method="channel_stack")
        cur = _img(seed=0)
        hist = _img(seed=1)
        result = fuse.fuse(cur, hist)
        assert result.shape == (64, 64, 6)  # 3 + 3

    def test_shape_with_sar(self):
        fuse = FeatureFusion(method="channel_stack")
        cur = _img(seed=0)
        hist = _img(seed=1)
        sar = _img(c=2, seed=2)
        result = fuse.fuse(cur, hist, sar)
        assert result.shape == (64, 64, 8)  # 3 + 3 + 2

    def test_shape_sar_single_band(self):
        fuse = FeatureFusion(method="channel_stack")
        cur = _img(seed=0)
        hist = _img(seed=1)
        sar = np.random.default_rng(3).random((64, 64)).astype(np.float32)
        result = fuse.fuse(cur, hist, sar)
        assert result.shape == (64, 64, 7)


class TestWeightedAverage:
    def test_output_shape(self):
        fuse = FeatureFusion(method="weighted_average")
        cur = _img(seed=0)
        hist = _img(seed=1)
        result = fuse.fuse(cur, hist)
        assert result.shape == (64, 64, 3)

    def test_blended_values_bounded(self):
        fuse = FeatureFusion(method="weighted_average", weight_current=0.5, weight_historical=0.5)
        cur = np.ones((64, 64, 3), dtype=np.float32)
        hist = np.zeros((64, 64, 3), dtype=np.float32)
        result = fuse.fuse(cur, hist)
        np.testing.assert_allclose(result, 0.5, atol=1e-5)


class TestLaplacianPyramid:
    def test_output_shape(self):
        fuse = FeatureFusion(method="laplacian_pyramid")
        cur = _img(seed=0)
        hist = _img(seed=1)
        result = fuse.fuse(cur, hist)
        assert result.shape == (64, 64, 3)

    def test_values_in_range(self):
        fuse = FeatureFusion(method="laplacian_pyramid")
        cur = _img(seed=0)
        hist = _img(seed=1)
        result = fuse.fuse(cur, hist)
        assert result.min() >= 0.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5


class TestShapeMismatch:
    def test_current_historical_mismatch_raises(self):
        fuse = FeatureFusion(method="channel_stack")
        cur = _img(h=64, w=64)
        hist = _img(h=32, w=32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            fuse.fuse(cur, hist)

    def test_sar_mismatch_raises(self):
        fuse = FeatureFusion(method="channel_stack")
        cur = _img(h=64, w=64)
        hist = _img(h=64, w=64)
        sar = _img(h=32, w=32)
        with pytest.raises(ValueError, match="SAR shape mismatch"):
            fuse.fuse(cur, hist, sar)
