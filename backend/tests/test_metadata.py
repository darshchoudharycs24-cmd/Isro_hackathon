"""Tests for MetadataManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.preprocessing.metadata import MetadataManager


class TestMetadataManager:
    def test_initial_keys_present(self):
        mgr = MetadataManager()
        d = mgr.to_dict()
        required = {
            "image_id", "resolution", "CRS", "bands",
            "historical_image_used", "registration_error",
            "normalization", "processing_time", "fusion_method", "SAR_used",
        }
        assert required.issubset(d.keys())

    def test_set_image_info(self, sample_meta):
        mgr = MetadataManager()
        mgr.set_image_info("test_image", sample_meta)
        d = mgr.to_dict()
        assert d["image_id"] == "test_image"
        assert d["resolution"] == sample_meta["resolution"]

    def test_set_registration_error(self):
        mgr = MetadataManager()
        mgr.set_registration_error(2.345)
        assert mgr.to_dict()["registration_error"] == 2.3450

    def test_set_fusion_info(self):
        mgr = MetadataManager()
        mgr.set_fusion_info(method="channel_stack", sar_used=True, sar_path="/data/sar.tif")
        d = mgr.to_dict()
        assert d["fusion_method"] == "channel_stack"
        assert d["SAR_used"] is True
        assert d["sar_path"] == "/data/sar.tif"

    def test_save_and_load_roundtrip(self, tmp_path, sample_meta):
        mgr = MetadataManager()
        mgr.set_image_info("roundtrip_test", sample_meta)
        mgr.set_registration_error(1.23)
        mgr.set_fusion_info(method="weighted_average")
        mgr.set_processing_time(5.678)

        json_path = tmp_path / "metadata.json"
        mgr.save(json_path)

        assert json_path.exists()
        loaded = MetadataManager.load(json_path)
        d = loaded.to_dict()
        assert d["image_id"] == "roundtrip_test"
        assert d["registration_error"] == 1.23
        assert d["fusion_method"] == "weighted_average"
        assert d["processing_time"] == 5.678

    def test_save_creates_parent_dirs(self, tmp_path):
        mgr = MetadataManager()
        deep_path = tmp_path / "a" / "b" / "c" / "meta.json"
        mgr.save(deep_path)
        assert deep_path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MetadataManager.load(tmp_path / "nonexistent.json")

    def test_update(self):
        mgr = MetadataManager()
        mgr.update({"custom_key": "custom_value"})
        assert mgr.to_dict()["custom_key"] == "custom_value"
