"""Tests for utils — I/O, dtype conversions, spatial checks."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from backend.preprocessing.utils import (
    load_image,
    save_image,
    to_uint8_png,
    to_float32_tensor,
    stack_bands,
    check_spatial_consistency,
)


class TestLoadImage:
    def test_loads_geotiff(self, geotiff_path):
        data, meta = load_image(geotiff_path)
        assert data.ndim == 3
        assert data.shape[2] == 3
        assert meta["n_bands"] == 3
        assert meta["width"] == 64
        assert meta["height"] == 64

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.tif")

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "image.xyz"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_image(bad)

    def test_single_band_squeezed(self, tmp_path):
        """Single-band GeoTIFF should return (H, W) not (H, W, 1)."""
        path = tmp_path / "single.tif"
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        data = np.random.default_rng(0).integers(0, 255, (1, 32, 32), dtype=np.uint8)
        with rasterio.open(
            path, "w", driver="GTiff",
            height=32, width=32, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)

        arr, meta = load_image(path)
        assert arr.ndim == 2
        assert meta["n_bands"] == 1


class TestSaveImage:
    def test_save_and_reload(self, tmp_path, sample_rgb, sample_meta):
        path = tmp_path / "out.tif"
        save_image(sample_rgb, path, sample_meta, dtype="float32")
        assert path.exists()

        reloaded, _ = load_image(path)
        assert reloaded.shape == sample_rgb.shape

    def test_creates_parent_dirs(self, tmp_path, sample_rgb):
        path = tmp_path / "a" / "b" / "out.tif"
        save_image(sample_rgb, path)
        assert path.exists()


class TestToUint8:
    def test_range(self, sample_rgb):
        result = to_uint8_png(sample_rgb)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_ones_map_to_255(self):
        arr = np.ones((10, 10, 3), dtype=np.float32)
        result = to_uint8_png(arr)
        assert result.max() == 255

    def test_zeros_map_to_0(self):
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        result = to_uint8_png(arr)
        assert result.max() == 0


class TestToFloat32Tensor:
    def test_dtype_and_range(self, sample_rgb):
        result = to_float32_tensor(sample_rgb)
        assert result.dtype == np.float32
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_output_3d(self, sample_gray):
        result = to_float32_tensor(sample_gray)
        assert result.ndim == 3
        assert result.shape[2] == 1


class TestStackBands:
    def test_stacks_2d_arrays(self):
        a = np.zeros((64, 64), dtype=np.float32)
        b = np.ones((64, 64), dtype=np.float32)
        result = stack_bands([a, b])
        assert result.shape == (64, 64, 2)

    def test_stacks_3d_arrays(self):
        a = np.zeros((64, 64, 3), dtype=np.float32)
        b = np.ones((64, 64, 2), dtype=np.float32)
        result = stack_bands([a, b])
        assert result.shape == (64, 64, 5)


class TestSpatialConsistency:
    def test_same_image_is_consistent(self, sample_meta):
        result = check_spatial_consistency(sample_meta, sample_meta)
        assert result["same_resolution"] is True
        assert result["same_size"] is True
