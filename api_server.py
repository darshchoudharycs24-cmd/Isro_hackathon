"""
api_server.py — FastAPI backend for the CloudClear dashboard.

Supports two reconstruction modes, selected via the "mode" form field:
  - "reference": needs current + historical images. Clones real historical
    pixels into the cloud region. Higher fidelity (EdgeCorr ~0.91-0.95 in
    our tests) but requires a matching historical reference.
  - "hybrid": needs only the current (cloudy) image. Uses the CycleGAN
    checkpoint + classical inpainting to guess the hidden content. Lower
    fidelity (EdgeCorr ~0.44 in our tests) but works from a single image,
    matching the problem statement's baseline expected UX.

Run with:
    uvicorn api_server:app --reload --port 8000
"""

import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from evaluation.metrics import compute_metrics
from evaluation.confidence import generate_confidence_map
from refine import refine_with_reference, refine_reconstruction
from inference import CloudRemovalModel
from color_fix import match_color_stats

app = FastAPI(title="CloudClear API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
_gan_model = None  # lazy-loaded on first hybrid request


def _get_gan_model():
    global _gan_model
    if _gan_model is None:
        _gan_model = CloudRemovalModel(CHECKPOINT, gpu_id=-1)
    return _gan_model


def _read_upload_to_bgr(upload_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(upload_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image")
    return img


def _to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _make_mask_from_diff(current_bgr: np.ndarray, historical_bgr: np.ndarray) -> np.ndarray:
    """Ground-truth-diff cloud mask. Used for both modes here as a stand-in for
    Deeptanshu's detector, which needs separate band GeoTIFFs we don't get from
    a browser upload."""
    if current_bgr.shape != historical_bgr.shape:
        historical_bgr = cv2.resize(historical_bgr, (current_bgr.shape[1], current_bgr.shape[0]))
    diff = cv2.absdiff(current_bgr, historical_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = (diff_gray > 25).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _make_mask_from_brightness(current_bgr: np.ndarray) -> np.ndarray:
    """Fallback heuristic mask when no historical image is available at all
    (pure single-image mode): bright, low-saturation regions are treated as
    likely cloud. Rough — real deployments should use Deeptanshu's detector."""
    hsv = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v > 180) & (s < 60)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


@app.post("/predict")
async def predict(
    current: UploadFile = File(...),
    historical: Optional[UploadFile] = File(None),
    mode: str = Form("reference"),
):
    try:
        current_bgr = _read_upload_to_bgr(await current.read())

        if mode == "reference":
            if historical is None:
                return JSONResponse(
                    {"error": "Reference mode requires a historical image."},
                    status_code=400,
                )
            historical_bgr = _read_upload_to_bgr(await historical.read())
            mask = _make_mask_from_diff(current_bgr, historical_bgr)
            final = refine_with_reference(historical_bgr, current_bgr, mask)
            metrics = compute_metrics(historical_bgr, final)

        elif mode == "hybrid":
            if historical is not None:
                historical_bgr = _read_upload_to_bgr(await historical.read())
                mask = _make_mask_from_diff(current_bgr, historical_bgr)
                score_reference = historical_bgr
            else:
                mask = _make_mask_from_brightness(current_bgr)
                score_reference = current_bgr  # no ground truth available; scores vs itself as a placeholder

            model = _get_gan_model()
            raw_gan = model.reconstruct(current_bgr)
            gan_fixed = match_color_stats(raw_gan, current_bgr, mask=mask)
            final, _, _ = refine_reconstruction(gan_fixed, current_bgr, mask)
            metrics = compute_metrics(score_reference, final)

        else:
            return JSONResponse({"error": f"Unknown mode: {mode}"}, status_code=400)

        confidence, heatmap = generate_confidence_map(mask)

        return JSONResponse({
            "mode": mode,
            "cloud_mask": _to_data_url(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
            "reconstructed": _to_data_url(final),
            "confidence": _to_data_url(heatmap),
            "metrics": {
                "psnr": round(float(metrics["PSNR"]), 2),
                "ssim": round(float(metrics["SSIM"]), 4),
                "rmse": round(float(metrics["RMSE"]), 2),
                "sam": round(float(metrics["SAM"]), 2),
                "edgecorr": round(float(metrics["EdgeCorr"]), 4),
            },
        })
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}