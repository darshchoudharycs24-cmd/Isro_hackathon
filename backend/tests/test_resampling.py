"""Tests for ImageResampler."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing.resampling import ImageResampler


def _make_meta(width: int, height: int, res: float = 0.001) -> dict:
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    crs = CRS.from_epsg(4326)
    return {
        "width": width,
        "height": height,
        "resolution": (res, res),
        "crs": crs.to_wkt(),
        "crs_obj": crs,
        "transform": from_bounds(77.0, 28.0, 77.0 + width * res, 28.0 + height * res, width, height),
        "n_bands": 3,
        "dtype": "float32",
        "nodata": None,
    }


class TestResampleToMatch:
    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
    def test_output_shape(self, method):
        resampler = ImageResampler(method=method)
        src = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
        src_meta = _make_meta(128, 128, res=0.002)
        tgt_meta = _make_meta(64, 64, res=0.001)

        result = resampler.resample_to_match(src, src_meta, tgt_meta)
        assert result.shape[:2] == (64, 64)
        assert result.shape[2] == 3

    def test_no_op_when_same_size(self):
        resampler = ImageResampler()
        src = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
        meta = _make_meta(64, 64)
        result = resampler.resample_to_match(src, meta, meta)
        assert result.shape == src.shape

    def test_single_band_resampling(self):
        resampler = ImageResampler(method="bilinear")
        src = np.random.default_rng(5).random((128, 128)).astype(np.float32)
        src_meta = _make_meta(128, 128, res=0.002)
        src_meta["n_bands"] = 1
        tgt_meta = _make_meta(64, 64)
        tgt_meta["n_bands"] = 1
        result = resampler.resample_to_match(src, src_meta, tgt_meta)
        assert result.shape == (64, 64)

    def test_dtype_preserved(self):
        resampler = ImageResampler(method="nearest")
        src = np.random.default_rng(1).random((64, 64, 3)).astype(np.float64)
        src_meta = _make_meta(64, 64, res=0.002)
        tgt_meta = _make_meta(32, 32)
        result = resampler.resample_to_match(src, src_meta, tgt_meta)
        assert result.dtype == np.float64

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown resampling method"):
            ImageResampler(method="lanczos5000")
