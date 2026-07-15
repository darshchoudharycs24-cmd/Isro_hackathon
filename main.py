"""
main.py - FastAPI application entry point.

POST /preprocess
  Accepts multipart-form uploads of current image, historical images,
  optional SAR, and optional cloud mask.
  Returns processing metadata, file paths, and warnings.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LISS-IV Cloud Removal — Preprocessing API",
    description=(
        "Data Fusion & Preprocessing Pipeline for ISRO Bharatiya Antariksh Hackathon.\n"
        "Accepts LISS-IV imagery, performs preprocessing and multi-temporal fusion, "
        "and returns outputs ready for the Pix2Pix/Diffusion reconstruction model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single pipeline instance (re-used across requests)
_pipeline = PreprocessingPipeline()


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Quick liveness check."""
    return {"status": "ok", "service": "preprocessing-pipeline"}


@app.post("/preprocess", tags=["Pipeline"])
async def preprocess(
    current_image: Annotated[UploadFile, File(description="Current cloudy LISS-IV image (GeoTIFF/PNG/JPEG)")],
    historical_images: Annotated[
        list[UploadFile],
        File(description="One or more cloud-free historical images"),
    ],
    sar_image: Annotated[
        UploadFile | None,
        File(description="Optional Sentinel-1 SAR GeoTIFF"),
    ] = None,
    cloud_mask: Annotated[
        UploadFile | None,
        File(description="Optional cloud mask PNG/TIF from Cloud Detection module"),
    ] = None,
    current_date: Annotated[
        str | None,
        Form(description="Acquisition date of current image (YYYY-MM-DD)"),
    ] = None,
    sample_id: Annotated[
        str | None,
        Form(description="Optional sample identifier for the reconstruction package"),
    ] = None,
) -> JSONResponse:
    """
    Run the full preprocessing and fusion pipeline.

    Returns
    -------
    JSON::

        {
          "status": "success",
          "processing_time": 12.4,
          "output_directory": "...",
          "paths": {
            "fused_png": "...",
            "fused_tif": "...",
            "metadata": "...",
            "preview": "..."
          },
          "metadata": { ... },
          "warnings": []
        }
    """
    t0 = time.perf_counter()

    tmp_dir = Path(tempfile.mkdtemp(prefix="lissiv_preproc_"))
    try:
        cur_dir = tmp_dir / "current"
        hist_dir = tmp_dir / "historical"
        cur_dir.mkdir()
        hist_dir.mkdir()

        cur_path = cur_dir / current_image.filename
        _save_upload(current_image, cur_path)

        for hist_file in historical_images:
            _save_upload(hist_file, hist_dir / hist_file.filename)

        sar_path = None
        if sar_image and sar_image.filename:
            sar_dir = tmp_dir / "sar"
            sar_dir.mkdir()
            sar_path = sar_dir / sar_image.filename
            _save_upload(sar_image, sar_path)

        mask_path = None
        if cloud_mask and cloud_mask.filename:
            mask_dir = tmp_dir / "mask"
            mask_dir.mkdir()
            mask_path = mask_dir / cloud_mask.filename
            _save_upload(cloud_mask, mask_path)

        output_dir = Path("backend/data/output") / tmp_dir.name

        from datetime import date as _date, datetime  # noqa: PLC0415
        acq_date: _date | None = None
        if current_date:
            try:
                acq_date = datetime.strptime(current_date, "%Y-%m-%d").date()
            except ValueError:
                logger.warning("Could not parse current_date '%s'", current_date)

        try:
            result = _pipeline.run(
                current_path=cur_path,
                historical_dir=hist_dir,
                sar_path=sar_path,
                current_date=acq_date,
                output_dir=output_dir,
                cloud_mask_path=mask_path,
                sample_id=sample_id or cur_path.stem,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline error")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

        elapsed = time.perf_counter() - t0

        # Build enhanced response (Task 9)
        recon = result.reconstruction_output
        paths = {}
        if recon:
            d = recon.to_dict()
            paths = {
                "fused_png": d.get("fused_png"),
                "fused_tif": d.get("fused_tif"),
                "fused_npy": d.get("fused_npy"),
                "metadata": d.get("metadata_json"),
                "preview": d.get("preview_dir"),
                "aligned_reference_tif": d.get("aligned_reference_tif"),
                "current_png": d.get("current_png"),
                "cloud_mask_png": d.get("cloud_mask_png"),
            }
        else:
            paths = result.output_paths

        return JSONResponse({
            "status": "success",
            "processing_time": round(elapsed, 3),
            "output_directory": str(output_dir),
            "paths": paths,
            # Legacy keys preserved for backward compatibility
            "metadata": result.metadata,
            "output_paths": result.output_paths,
            "preview_paths": result.preview_paths,
            "warnings": result.warnings,
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/outputs/{filename}", tags=["Files"])
def serve_output(filename: str) -> FileResponse:
    """Serve a file from the output directory by name."""
    path = Path("backend/data/output") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.delete("/cache", tags=["Maintenance"])
def clear_cache() -> dict:
    """Clear all cached intermediate pipeline results."""
    from backend.preprocessing.cache import PipelineCache  # noqa: PLC0415
    cache = PipelineCache()
    deleted = cache.clear_all()
    return {"status": "ok", "files_deleted": deleted}


@app.get("/cache/stats", tags=["Maintenance"])
def cache_stats() -> dict:
    """Return cache directory statistics."""
    from backend.preprocessing.cache import PipelineCache  # noqa: PLC0415
    cache = PipelineCache()
    return cache.stats()


def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(upload.file, fh)
