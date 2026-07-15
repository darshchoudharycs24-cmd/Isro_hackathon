"""
conftest.py - Shared pytest fixtures for all preprocessing tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


@pytest.fixture()
def sample_rgb() -> np.ndarray:
    """256×256 float32 RGB image with values in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((256, 256, 3), dtype=np.float32)


@pytest.fixture()
def sample_gray() -> np.ndarray:
    """256×256 float32 grayscale image."""
    rng = np.random.default_rng(0)
    return rng.random((256, 256), dtype=np.float32)


@pytest.fixture()
def sample_dn_image() -> np.ndarray:
    """256×256 uint16 DN image (typical raw satellite data)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 4096, (256, 256, 3), dtype=np.uint16)


@pytest.fixture()
def sample_meta() -> dict:
    """Minimal metadata dict for a 256×256 GeoTIFF-like image."""
    transform = from_bounds(77.0, 28.0, 77.1, 28.1, 256, 256)
    return {
        "file_path": "test_image.tif",
        "crs": "EPSG:4326",
        "crs_obj": None,
        "transform": transform,
        "resolution": (0.000390625, 0.000390625),
        "n_bands": 3,
        "dtype": "uint16",
        "nodata": None,
        "width": 256,
        "height": 256,
        "driver": "GTiff",
    }


@pytest.fixture()
def geotiff_path(tmp_path) -> str:
    """Write a tiny GeoTIFF to a temp file and return its path."""
    path = tmp_path / "test.tif"
    transform = from_bounds(77.0, 28.0, 77.1, 28.1, 64, 64)
    data = (np.random.default_rng(1).random((3, 64, 64)) * 255).astype(np.uint8)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=64, width=64,
        count=3, dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture()
def historical_dir(tmp_path) -> str:
    """Create a historical directory with two tiny GeoTIFFs."""
    hist = tmp_path / "historical"
    hist.mkdir()
    transform = from_bounds(77.0, 28.0, 77.1, 28.1, 64, 64)
    rng = np.random.default_rng(99)
    for name in ("20230101.tif", "20230601.tif"):
        data = (rng.random((3, 64, 64)) * 255).astype(np.uint8)
        with rasterio.open(
            hist / name, "w",
            driver="GTiff", height=64, width=64,
            count=3, dtype="uint8",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)
    return str(hist)


@pytest.fixture()
def binary_mask_path(tmp_path) -> str:
    """Write a 64×64 binary cloud mask (uint8 0/255) GeoTIFF."""
    from rasterio.transform import from_bounds
    path = tmp_path / "mask.tif"
    transform = from_bounds(77.0, 28.0, 77.1, 28.1, 64, 64)
    rng = np.random.default_rng(55)
    data = ((rng.random((1, 64, 64)) > 0.7) * 255).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=64, width=64, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture()
def reconstruction_input(sample_rgb) -> "ReconstructionInput":
    """Minimal valid ReconstructionInput for testing."""
    from backend.preprocessing.interfaces import ReconstructionInput
    return ReconstructionInput(
        current_image=sample_rgb,
        historical_image=sample_rgb,
        fused_image=np.concatenate([sample_rgb, sample_rgb], axis=-1),
        metadata={},
    )
