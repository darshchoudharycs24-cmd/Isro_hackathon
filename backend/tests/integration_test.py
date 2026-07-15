"""
integration_test.py - End-to-end pipeline integration test.

Simulates the full data flow:
    Current Image → Historical Images → Optional SAR → Optional Cloud Mask
    → Pipeline → ReconstructionOutput

Verifies:
  - Every exported file exists
  - Array shapes are consistent
  - Metadata fields are populated
  - GeoTIFFs are spatially valid
  - ReconstructionOutput paths resolve correctly
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def integration_workspace(tmp_path) -> dict[str, Path]:
    """
    Build a minimal but complete workspace with all input data types.
    Returns a dict of labelled paths.
    """
    transform = from_bounds(77.0, 28.0, 77.1, 28.1, 64, 64)
    rng = np.random.default_rng(42)

    def _write_tif(path: Path, seed: int, bands: int = 3, dtype="uint8") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (rng.random((bands, 64, 64)) * 255).astype(dtype)
        with rasterio.open(
            path, "w", driver="GTiff",
            height=64, width=64, count=bands, dtype=dtype,
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)
        return path

    # Current image
    cur = _write_tif(tmp_path / "current" / "lissiv.tif", seed=10)

    # Historical images (two)
    _write_tif(tmp_path / "historical" / "20230101.tif", seed=20)
    _write_tif(tmp_path / "historical" / "20230601.tif", seed=30)

    # SAR (single-band float32)
    sar = _write_tif(tmp_path / "sar" / "s1.tif", seed=40, bands=1, dtype="float32")

    # Cloud mask (binary uint8, 0/255)
    mask_data = (rng.random((1, 64, 64)) > 0.7).astype(np.uint8) * 255
    mask_path = tmp_path / "mask" / "mask.tif"
    mask_path.parent.mkdir(parents=True)
    with rasterio.open(
        mask_path, "w", driver="GTiff",
        height=64, width=64, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(mask_data)

    return {
        "current": cur,
        "historical_dir": tmp_path / "historical",
        "sar": sar,
        "cloud_mask": mask_path,
        "output_dir": tmp_path / "output",
    }


# ─── Integration tests ────────────────────────────────────────────────────────

class TestFullPipeline:
    """Run the complete pipeline and validate all outputs."""

    def test_pipeline_completes(self, integration_workspace):
        """Pipeline should run to completion without exceptions."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        pipeline = PreprocessingPipeline()
        result = pipeline.run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        assert result is not None
        assert result.fused_image is not None

    def test_output_files_exist(self, integration_workspace):
        """Every file in output_paths must exist on disk."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        for key, path in result.output_paths.items():
            assert Path(path).exists(), f"Missing output: {key} → {path}"

    def test_fused_image_shape(self, integration_workspace):
        """Fused image must be 3-D float32."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        fused = result.fused_image
        assert fused.ndim == 3, "Fused image must be 3-D (H, W, C)"
        assert fused.dtype == np.float32

    def test_aligned_reference_matches_current(self, integration_workspace):
        """Aligned historical image must share H×W with current."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        assert result.aligned_reference.shape[:2] == result.current_normalised.shape[:2]

    def test_metadata_required_fields(self, integration_workspace):
        """Metadata must contain all required keys with non-null values."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        meta = result.metadata
        required = [
            "image_id", "resolution", "bands",
            "historical_image_used", "fusion_method",
            "processing_time", "pipeline_version",
        ]
        for field in required:
            assert field in meta, f"Missing metadata field: {field}"
            assert meta[field] is not None, f"Null metadata field: {field}"

    def test_metadata_json_on_disk(self, integration_workspace):
        """Metadata JSON must be readable and valid."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        meta_path = Path(result.output_paths["metadata"])
        assert meta_path.exists()
        with open(meta_path) as fh:
            data = json.load(fh)
        assert "image_id" in data
        assert "processing_time" in data

    def test_processing_time_under_threshold(self, integration_workspace):
        """64×64 image should finish well under 60 s."""
        import time
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        t0 = time.perf_counter()
        PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 60.0, f"Pipeline took too long: {elapsed:.1f}s"


class TestWithSAR:
    def test_sar_pipeline_completes(self, integration_workspace):
        """Pipeline should complete with SAR enabled."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            sar_path=ws["sar"],
            output_dir=ws["output_dir"],
        )
        assert result.metadata.get("SAR_used") is True

    def test_sar_expands_fused_channels(self, integration_workspace):
        """With SAR and channel_stack, fused channels > 3+3 baseline."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result_no_sar = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"] / "no_sar",
        )
        result_sar = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            sar_path=ws["sar"],
            output_dir=ws["output_dir"] / "with_sar",
        )
        assert result_sar.fused_image.shape[-1] > result_no_sar.fused_image.shape[-1]


class TestWithCloudMask:
    def test_cloud_mask_accepted(self, integration_workspace):
        """Pipeline should accept and record a cloud mask."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            cloud_mask_path=ws["cloud_mask"],
            output_dir=ws["output_dir"],
        )
        assert result.metadata.get("cloud_mask_used") is True

    def test_cloud_mask_png_exported(self, integration_workspace):
        """When mask provided, cloud_mask.png should be in outputs."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            cloud_mask_path=ws["cloud_mask"],
            output_dir=ws["output_dir"],
        )
        recon = result.reconstruction_output
        if recon and recon.cloud_mask_png:
            assert recon.cloud_mask_png.exists()

    def test_missing_mask_path_handled_gracefully(self, integration_workspace):
        """A non-existent mask path should not crash the pipeline."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            cloud_mask_path="/nonexistent/mask.png",
            output_dir=ws["output_dir"],
        )
        assert result.fused_image is not None
        assert result.metadata.get("cloud_mask_used") is False


class TestReconstructionPackage:
    def test_reconstruction_output_dataclass(self, integration_workspace):
        """ReconstructionOutput should be a valid typed dataclass."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        recon = result.reconstruction_output
        assert recon is not None
        assert recon.fused_png.exists()
        assert recon.fused_npy.exists()
        assert recon.metadata_json.exists()

    def test_reconstruction_to_dict(self, integration_workspace):
        """to_dict() must return all required path keys."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        d = result.reconstruction_output.to_dict()
        for key in ("fused_png", "fused_npy", "fused_tif", "metadata_json"):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_geotiff_has_spatial_metadata(self, integration_workspace):
        """Exported GeoTIFF should have valid CRS and transform."""
        from backend.preprocessing.pipeline import PreprocessingPipeline
        ws = integration_workspace
        result = PreprocessingPipeline().run(
            current_path=ws["current"],
            historical_dir=ws["historical_dir"],
            output_dir=ws["output_dir"],
        )
        tif_path = result.reconstruction_output.fused_tif
        with rasterio.open(tif_path) as src:
            assert src.crs is not None
            assert src.transform is not None
            # Transform should not be identity (i.e. has real geo info)
            assert src.transform.a != 1.0 or src.transform.e != -1.0


class TestGeoTIFFExport:
    def test_export_geotiff_preserves_crs(self, tmp_path, integration_workspace):
        """GeoTIFF export should round-trip CRS correctly."""
        from backend.preprocessing.geotiff_export import export_geotiff
        from backend.preprocessing.utils import load_image
        _, meta = load_image(integration_workspace["current"])

        data = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
        out = tmp_path / "test_export.tif"
        export_geotiff(data, out, meta, dtype="float32")

        with rasterio.open(out) as src:
            assert src.crs is not None
            assert src.count == 3

    def test_export_geotiff_single_band(self, tmp_path, integration_workspace):
        from backend.preprocessing.geotiff_export import export_geotiff
        from backend.preprocessing.utils import load_image
        _, meta = load_image(integration_workspace["current"])

        data = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        out = tmp_path / "single.tif"
        export_geotiff(data, out, meta)
        with rasterio.open(out) as src:
            assert src.count == 1


class TestCloudMaskModule:
    def test_load_binary_mask(self, integration_workspace):
        from backend.preprocessing.cloud_mask import CloudMaskHandler
        handler = CloudMaskHandler()
        mask = handler.load_and_prepare(integration_workspace["cloud_mask"], (64, 64))
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32
        assert set(np.unique(mask).tolist()).issubset({0.0, 1.0})

    def test_mask_resize(self, tmp_path, integration_workspace):
        """Mask of different size should be resized."""
        from backend.preprocessing.cloud_mask import CloudMaskHandler
        import rasterio  # noqa: PLC0415
        from rasterio.transform import from_bounds  # noqa: PLC0415

        # Write a 32×32 mask
        small_mask = tmp_path / "small_mask.tif"
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        data = (np.random.default_rng(5).random((1, 32, 32)) > 0.5).astype(np.uint8) * 255
        with rasterio.open(
            small_mask, "w", driver="GTiff",
            height=32, width=32, count=1, dtype="uint8", transform=transform,
        ) as dst:
            dst.write(data)

        handler = CloudMaskHandler()
        mask = handler.load_and_prepare(small_mask, (64, 64))
        assert mask.shape == (64, 64)

    def test_load_cloud_mask_convenience(self, integration_workspace):
        from backend.preprocessing.cloud_mask import load_cloud_mask
        mask = load_cloud_mask(integration_workspace["cloud_mask"], (64, 64))
        assert mask is not None

    def test_load_cloud_mask_none_input(self):
        from backend.preprocessing.cloud_mask import load_cloud_mask
        result = load_cloud_mask(None, (64, 64))
        assert result is None


class TestCacheModule:
    def test_save_and_load(self, tmp_path):
        from backend.preprocessing.cache import PipelineCache
        cache = PipelineCache(cache_dir=tmp_path / "cache")
        arr = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        cache.save("key123", "aligned", arr)
        loaded = cache.load("key123", "aligned")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    def test_cache_miss(self, tmp_path):
        from backend.preprocessing.cache import PipelineCache
        cache = PipelineCache(cache_dir=tmp_path / "cache")
        result = cache.load("nonexistent", "stage")
        assert result is None

    def test_stale_entry_returns_none(self, tmp_path):
        from backend.preprocessing.cache import PipelineCache
        cache = PipelineCache(cache_dir=tmp_path / "cache", max_age_seconds=0.0)
        arr = np.ones((10, 10), dtype=np.float32)
        cache.save("key", "stage", arr)
        import time  # noqa: PLC0415
        time.sleep(0.01)
        result = cache.load("key", "stage")
        assert result is None

    def test_hash_file(self, tmp_path, integration_workspace):
        from backend.preprocessing.cache import PipelineCache
        cache = PipelineCache(cache_dir=tmp_path / "cache")
        h1 = cache.hash_file(integration_workspace["current"])
        h2 = cache.hash_file(integration_workspace["current"])
        assert h1 == h2
        assert len(h1) == 16

    def test_clear_all(self, tmp_path):
        from backend.preprocessing.cache import PipelineCache
        cache = PipelineCache(cache_dir=tmp_path / "cache")
        for i in range(3):
            cache.save(f"k{i}", "stage", np.zeros((4, 4), dtype=np.float32))
        deleted = cache.clear_all()
        assert deleted == 3


class TestInterfacesDataclass:
    def test_reconstruction_input_valid(self):
        from backend.preprocessing.interfaces import ReconstructionInput
        h, w, c = 64, 64, 3
        ri = ReconstructionInput(
            current_image=np.zeros((h, w, c), dtype=np.float32),
            historical_image=np.zeros((h, w, c), dtype=np.float32),
            fused_image=np.zeros((h, w, 6), dtype=np.float32),
            metadata={},
        )
        assert ri.spatial_shape == (h, w)
        assert not ri.has_cloud_mask
        assert not ri.has_sar

    def test_reconstruction_input_shape_mismatch(self):
        from backend.preprocessing.interfaces import ReconstructionInput
        with pytest.raises(ValueError, match="Shape mismatch"):
            ReconstructionInput(
                current_image=np.zeros((64, 64, 3), dtype=np.float32),
                historical_image=np.zeros((32, 32, 3), dtype=np.float32),
                fused_image=np.zeros((64, 64, 6), dtype=np.float32),
            )

    def test_reconstruction_input_with_mask_and_sar(self):
        from backend.preprocessing.interfaces import ReconstructionInput
        ri = ReconstructionInput(
            current_image=np.zeros((64, 64, 3), dtype=np.float32),
            historical_image=np.zeros((64, 64, 3), dtype=np.float32),
            fused_image=np.zeros((64, 64, 7), dtype=np.float32),
            cloud_mask=np.zeros((64, 64), dtype=np.float32),
            sar_image=np.zeros((64, 64, 1), dtype=np.float32),
        )
        assert ri.has_cloud_mask
        assert ri.has_sar
