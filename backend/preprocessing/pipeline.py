"""
pipeline.py - Main preprocessing and fusion pipeline orchestrator.

Coordinates all preprocessing steps:
  [1/8] Load imagery
  [2/8] Radiometric calibration
  [3/8] Patch selection & historical load
  [4/8] Resampling + Geometric alignment
  [5/8] Normalisation
  [6/8] SAR fusion (optional) + Cloud mask (optional)
  [7/8] Feature fusion
  [8/8] Export + Preview + Metadata

New in Phase 2:
  - cloud_mask_path parameter (Task 2)
  - GeoTIFF export for all outputs (Task 3)
  - Reconstruction package via export_for_reconstruction() (Task 4)
  - Rich metadata: registration_rmse, feature_matches, ssim, histogram (Task 5)
  - Numbered step progress logging with timing + RAM (Task 6)
  - File-hash cache for aligned/normalised arrays (Task 7)
  - Band-flexible channel selection from config (Task 10)
  - ThreadPoolExecutor-parallelised historical scoring (Task 11, via PatchSelector)

Entry point::

    result = PreprocessingPipeline(config).run(
        current_path="data/current/image.tif",
        historical_dir="data/historical/",
        sar_path="data/sar/s1.tif",        # optional
        cloud_mask_path="data/mask.png",   # optional
        sample_id="my_scene",              # optional
    )
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .calibration import RadiometricCalibrator
from .alignment import GeometricAligner, AlignmentError
from .resampling import ImageResampler
from .normalization import BandNormalizer
from .fusion import FeatureFusion
from .patch_selector import PatchSelector
from .metadata import MetadataManager
from .sar_handler import SARHandler
from .cloud_mask import load_cloud_mask
from .geotiff_export import export_pipeline_geotiffs
from .exporter import export_for_reconstruction
from .preview import save_all_previews
from .cache import PipelineCache
from .progress import PipelineProgress
from .interfaces import ReconstructionInput, ReconstructionOutput
from .utils import load_image, save_image, to_uint8_png, to_float32_tensor

logger = logging.getLogger(__name__)

_TOTAL_STEPS = 8


class PipelineResult:
    """
    Container for all pipeline outputs.

    Backward-compatible with Phase 1: all existing attributes are preserved.
    Phase 2 adds ``reconstruction_output`` and ``warnings``.

    Attributes
    ----------
    fused_image : np.ndarray
        Float32 fused tensor (H, W, C).
    aligned_reference : np.ndarray
        Aligned historical image float32 (H, W, C).
    current_normalised : np.ndarray
        Normalised current image float32 (H, W, C).
    cloud_mask : np.ndarray or None
        Binary cloud mask (H, W) if provided.
    metadata : dict
        Full rich metadata dict.
    preview_paths : dict
        Paths to generated preview images.
    output_paths : dict
        Paths to all exported files (npy, png, tif, metadata, ...).
    reconstruction_output : ReconstructionOutput or None
        Typed reconstruction package (Phase 2).
    reconstruction_input : ReconstructionInput or None
        Typed input bundle for the reconstruction model (Phase 2).
    warnings : list[str]
        Non-fatal warnings accumulated during processing.
    """

    def __init__(self) -> None:
        self.fused_image: np.ndarray | None = None
        self.aligned_reference: np.ndarray | None = None
        self.current_normalised: np.ndarray | None = None
        self.cloud_mask: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.preview_paths: dict[str, str] = {}
        self.output_paths: dict[str, str] = {}
        self.reconstruction_output: ReconstructionOutput | None = None
        self.reconstruction_input: ReconstructionInput | None = None
        self.warnings: list[str] = []


class PreprocessingPipeline:
    """
    Full preprocessing and multi-temporal fusion pipeline.

    Parameters
    ----------
    config : dict, str, Path, or None
        Config dict or path to ``pipeline_config.yaml``.
        Falls back to ``backend/configs/pipeline_config.yaml`` if None.
    """

    def __init__(self, config: dict[str, Any] | str | Path | None = None) -> None:
        self.cfg = self._load_config(config)
        self._setup_logging()

        # ── Sub-components ─────────────────────────────────────────────────
        cal_cfg = self.cfg.get("calibration", {})
        self._calibrator = RadiometricCalibrator(
            gain=cal_cfg.get("gain", 0.0001),
            offset=cal_cfg.get("offset", 0.0),
            clip_min=cal_cfg.get("clip_min", 0.0),
            clip_max=cal_cfg.get("clip_max", 1.0),
            noise_method=cal_cfg.get("noise_reduction", {}).get("method", "gaussian"),
            noise_sigma=cal_cfg.get("noise_reduction", {}).get("sigma", 1.0),
        )

        align_cfg = self.cfg.get("alignment", {})
        self._aligner = GeometricAligner(
            detector=align_cfg.get("detector", "orb"),
            max_features=align_cfg.get("max_features", 5000),
            match_ratio=align_cfg.get("match_ratio", 0.75),
            min_match_count=align_cfg.get("min_match_count", 10),
            ecc_fallback=align_cfg.get("ecc_fallback", True),
            ecc_iterations=align_cfg.get("ecc_iterations", 200),
            ecc_epsilon=align_cfg.get("ecc_epsilon", 1e-5),
        )

        res_cfg = self.cfg.get("resampling", {})
        self._resampler = ImageResampler(method=res_cfg.get("method", "bilinear"))

        norm_cfg = self.cfg.get("normalization", {})
        self._normalizer = BandNormalizer(
            method=norm_cfg.get("method", "minmax"),
            clip_percentile=tuple(norm_cfg.get("clip_percentile", [2, 98])),
        )

        ps_cfg = self.cfg.get("patch_selector", {})
        weights = ps_cfg.get("weights", {})
        self._selector = PatchSelector(
            date_weight=weights.get("date_score", 0.25),
            ssim_weight=weights.get("ssim_score", 0.35),
            histogram_weight=weights.get("histogram_score", 0.25),
            feature_weight=weights.get("feature_score", 0.15),
            max_date_diff_days=ps_cfg.get("max_date_diff_days", 365),
        )

        fus_cfg = self.cfg.get("fusion", {})
        fus_weights = fus_cfg.get("weights", {})
        self._fusion = FeatureFusion(
            method=fus_cfg.get("method", "channel_stack"),
            weight_current=fus_weights.get("current", 0.5),
            weight_historical=fus_weights.get("historical", 0.5),
        )

        sar_cfg = self.cfg.get("sar", {})
        self._sar_handler = SARHandler(
            align_to_current=sar_cfg.get("align_to_current", True),
        )

        # Phase 2: cache
        cache_dir = self.cfg.get("cache", {}).get("dir", ".cache")
        cache_max_age = self.cfg.get("cache", {}).get("max_age_seconds", 7 * 24 * 3600)
        self._cache = PipelineCache(cache_dir=cache_dir, max_age_seconds=cache_max_age)

        self._meta_mgr = MetadataManager()

    # ─── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        current_path: str | Path,
        historical_dir: str | Path,
        sar_path: str | Path | None = None,
        current_date: date | None = None,
        output_dir: str | Path | None = None,
        # Phase 2 additions (all optional, backward-compatible)
        cloud_mask_path: str | Path | None = None,
        sample_id: str | None = None,
    ) -> PipelineResult:
        """
        Execute the full preprocessing and fusion pipeline.

        Parameters
        ----------
        current_path : str or Path
            Current (possibly cloudy) LISS-IV image.
        historical_dir : str or Path
            Directory of cloud-free historical images.
        sar_path : str or Path, optional
            Sentinel-1 SAR GeoTIFF.
        current_date : date, optional
            Acquisition date for temporal scoring.
        output_dir : str or Path, optional
            Output directory (overrides config).
        cloud_mask_path : str or Path, optional
            Cloud mask PNG/TIF from the Cloud Detection module.
        sample_id : str, optional
            Identifier for the reconstruction package sub-directory.

        Returns
        -------
        PipelineResult
            All outputs including ReconstructionOutput typed dataclass.
        """
        t_start = time.perf_counter()
        result = PipelineResult()
        progress = PipelineProgress(total_steps=_TOTAL_STEPS, logger_name=__name__)

        out_dir = Path(
            output_dir
            or self.cfg.get("paths", {}).get("output_dir", "backend/data/output")
        )
        preview_dir = out_dir / "previews"
        out_dir.mkdir(parents=True, exist_ok=True)

        _sample_id = sample_id or Path(current_path).stem

        # ── [1/8] Load imagery ─────────────────────────────────────────────
        with progress.step("Loading imagery"):
            current_raw, current_meta = load_image(current_path)
            self._meta_mgr.set_image_info(Path(current_path).stem, current_meta)
            logger.debug(
                "Current: %s | shape=%s CRS=%s",
                Path(current_path).name,
                current_raw.shape,
                "present" if current_meta.get("crs") else "MISSING",
            )

        # ── [2/8] Calibration ──────────────────────────────────────────────
        with progress.step("Calibration"):
            current_calib = self._calibrator.calibrate(current_raw, current_meta)

        # ── [3/8] Patch selection + historical load ────────────────────────
        with progress.step("Patch selection"):
            best_hist_path, selection_scores = self._selector.select_best(
                current_calib, historical_dir, current_date
            )
            self._meta_mgr.set_historical_image(best_hist_path)

            # Extract quality scores for best image
            best_scores = selection_scores.get(best_hist_path.name, {})
            self._meta_mgr.set_quality_scores(
                ssim_score=best_scores.get("ssim_score"),
                histogram_score=best_scores.get("histogram_score"),
            )

            hist_raw, hist_meta = load_image(best_hist_path)
            hist_calib = self._calibrator.calibrate(
                hist_raw, hist_meta, reference=current_calib
            )

        # ── [4/8] Resampling + Alignment ──────────────────────────────────
        with progress.step("Alignment"):
            # Resampling
            hist_resampled = self._resampler.resample_to_match(
                hist_calib, hist_meta, current_meta
            )

            # Cache key: hash of current + historical files
            cache_key = self._cache.make_key(current_path, best_hist_path)
            aligned_historical = self._cache.load(cache_key, "aligned")

            if aligned_historical is None:
                try:
                    aligned_historical, reg_error = self._aligner.align(
                        hist_resampled, current_calib
                    )
                    self._meta_mgr.set_registration_error(
                        reg_error, feature_matches=self._aligner.feature_matches
                    )
                    self._cache.save(cache_key, "aligned", aligned_historical)
                except AlignmentError as exc:
                    msg = f"Alignment failed: {exc} — using unaligned resampled image."
                    logger.error(msg)
                    result.warnings.append(msg)
                    aligned_historical = hist_resampled
                    self._meta_mgr.set_registration_error(-1.0)
            else:
                logger.info("Alignment loaded from cache.")
                self._meta_mgr.set_registration_error(0.0)

        # ── [5/8] Normalisation ────────────────────────────────────────────
        with progress.step("Normalisation"):
            norm_cfg = self.cfg.get("normalization", {})

            current_norm = self._cache.load(cache_key, "current_norm")
            if current_norm is None:
                current_norm = self._normalizer.normalise(current_calib)
                self._cache.save(cache_key, "current_norm", current_norm)

            historical_norm = self._normalizer.normalise(
                aligned_historical, stats=self._normalizer.band_stats
            )
            self._meta_mgr.set_normalization_stats(
                self._normalizer.band_stats,
                method=norm_cfg.get("method", "minmax"),
            )

        # ── [6/8] SAR + Cloud mask ─────────────────────────────────────────
        with progress.step("SAR / Cloud mask"):
            sar_norm: np.ndarray | None = None
            sar_used = sar_path is not None and Path(sar_path).exists()

            if sar_used:
                sar_cache_key = self._cache.make_key(current_path, sar_path)
                sar_norm = self._cache.load(sar_cache_key, "sar")
                if sar_norm is None:
                    sar_norm = self._sar_handler.load_and_prepare(sar_path, current_norm)
                    self._cache.save(sar_cache_key, "sar", sar_norm)
                else:
                    logger.info("SAR loaded from cache.")
            else:
                logger.info("SAR not provided; skipping.")

            # Cloud mask integration
            cloud_mask: np.ndarray | None = load_cloud_mask(
                cloud_mask_path, current_norm.shape[:2]
            )
            mask_used = cloud_mask is not None
            self._meta_mgr.set_cloud_mask_used(mask_used)
            if mask_used:
                logger.info(
                    "Cloud mask applied | coverage=%.1f%%",
                    float(cloud_mask.mean() * 100),
                )

        # ── [7/8] Feature fusion ───────────────────────────────────────────
        with progress.step("Feature fusion"):
            # Band selection from config
            band_cfg = self.cfg.get("bands", {})
            current_sel, historical_sel = self._select_bands(
                current_norm, historical_norm, band_cfg
            )

            fused = self._fusion.fuse(current_sel, historical_sel, sar_norm)
            self._meta_mgr.set_fusion_info(
                method=self.cfg.get("fusion", {}).get("method", "channel_stack"),
                sar_used=sar_used,
                sar_path=str(sar_path) if sar_used else None,
            )

        # ── [8/8] Export + Preview + Metadata ─────────────────────────────
        with progress.step("Export"):
            output_cfg = self.cfg.get("output", {})
            rgb_channels = tuple(output_cfg.get("png_channels", [0, 1, 2]))

            # Legacy outputs (backward compatible)
            fused_paths = self._fusion.save_outputs(fused, out_dir, rgb_channels)
            result.output_paths = fused_paths

            # GeoTIFF export (Task 3)
            tif_paths = export_pipeline_geotiffs(
                current_norm, aligned_historical, fused, current_meta, out_dir
            )
            result.output_paths.update(tif_paths)

            # Legacy best_reference.tif
            ref_tif = out_dir / "best_reference.tif"
            save_image(aligned_historical, ref_tif, current_meta, dtype="float32")
            result.output_paths["best_reference"] = str(ref_tif)

            # uint8 PNG + float32 tensor for reconstruction model
            current_u8 = to_uint8_png(current_norm)
            current_png = out_dir / "current_uint8.png"
            if current_u8.ndim == 3:
                cv2.imwrite(str(current_png), cv2.cvtColor(current_u8, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(current_png), current_u8)
            result.output_paths["current_uint8_png"] = str(current_png)

            tensor = to_float32_tensor(current_norm)
            tensor_path = out_dir / "current_tensor.npy"
            np.save(str(tensor_path), tensor)
            result.output_paths["current_tensor"] = str(tensor_path)

        # ── Preview images ─────────────────────────────────────────────────
        with progress.step("Preview generation"):
            preview_paths = save_all_previews(
                current_norm, aligned_historical, fused, preview_dir
            )
            result.preview_paths = preview_paths

        # ── Metadata ───────────────────────────────────────────────────────
        total_time = time.perf_counter() - t_start
        self._meta_mgr.set_processing_time(total_time)
        self._meta_mgr.update({
            "selection_scores": selection_scores,
            "warnings": result.warnings,
        })
        metadata_path = out_dir / "metadata.json"
        self._meta_mgr.save(metadata_path)
        result.output_paths["metadata"] = str(metadata_path)

        # ── Reconstruction package (Task 4) ───────────────────────────────
        recon_out = export_for_reconstruction(
            sample_id=_sample_id,
            current=current_norm,
            historical=historical_norm,
            fused=fused,
            reference_meta=current_meta,
            metadata_dict=self._meta_mgr.to_dict(),
            output_root=out_dir / "reconstruction",
            cloud_mask=cloud_mask,
            preview_sources=preview_paths,
            warnings=result.warnings,
        )
        result.reconstruction_output = recon_out

        # ── Typed ReconstructionInput (Task 1) ────────────────────────────
        try:
            result.reconstruction_input = ReconstructionInput(
                current_image=current_norm,
                historical_image=historical_norm,
                fused_image=fused,
                cloud_mask=cloud_mask,
                sar_image=sar_norm,
                metadata=self._meta_mgr.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build ReconstructionInput: %s", exc)

        # ── Populate result ────────────────────────────────────────────────
        result.fused_image = fused
        result.aligned_reference = aligned_historical
        result.current_normalised = current_norm
        result.cloud_mask = cloud_mask
        result.metadata = self._meta_mgr.to_dict()

        progress.complete(total_time)
        return result

    # ─── Band selection ────────────────────────────────────────────────────

    @staticmethod
    def _select_bands(
        current: np.ndarray,
        historical: np.ndarray,
        band_cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select or reorder bands according to config.

        Config example::

            bands:
              red: 3
              green: 2
              blue: 1
              nir: 4

        If no ``bands`` config is present, returns images unchanged.
        Indices are 1-based (matching LISS-IV convention).
        """
        if not band_cfg:
            return current, historical

        # Build index list in fixed RGB(+NIR) order
        order_keys = ["blue", "green", "red", "nir", "swir"]
        indices = []
        for k in order_keys:
            if k in band_cfg:
                idx = int(band_cfg[k]) - 1  # 1-based → 0-based
                if 0 <= idx < (current.shape[-1] if current.ndim == 3 else 1):
                    indices.append(idx)

        if not indices:
            return current, historical

        def _pick(arr: np.ndarray, idxs: list[int]) -> np.ndarray:
            if arr.ndim == 2:
                return arr
            return arr[:, :, idxs]

        logger.debug("Band selection: indices=%s", indices)
        return _pick(current, indices), _pick(historical, indices)

    # ─── Config helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_config(config: Any) -> dict[str, Any]:
        if config is None:
            default = Path("backend/configs/pipeline_config.yaml")
            if default.exists():
                return PreprocessingPipeline._load_yaml(default)
            return {}
        if isinstance(config, dict):
            return config
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return PreprocessingPipeline._load_yaml(path)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        logger.debug("Loaded config from %s", path)
        return data or {}

    @staticmethod
    def _setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
