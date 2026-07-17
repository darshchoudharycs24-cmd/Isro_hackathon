# CloudClear AI

**ISRO Bharatiya Antariksh Hackathon 2026 — Team Terror Byte**

Generative AI-based cloud removal and surface reconstruction for LISS-IV satellite imagery.

## Problem

Persistent cloud cover limits the usability of optical satellite imagery (LISS-IV) for
land use mapping, disaster monitoring, and environmental assessment — especially over
tropical and mountainous regions. This project reconstructs cloud-obscured regions of a
satellite scene into analysis-ready, cloud-free imagery.

## Approach

We implement and compare two reconstruction strategies, as encouraged by the problem
statement's guidance on using "temporal reference observations" as auxiliary data:

### 1. Reference-Based Reconstruction (primary method)
Clones real pixels from a historical cloud-free image of the same location into the
cloud-masked region of the current scene, using feathered alpha compositing. Because it
uses real ground truth rather than a generated guess, this method produces the highest
fidelity results.

- **Validated on real paired data:** EdgeCorr 0.91–0.95, SSIM 0.84–0.90
- Requires: current (cloudy) image + a historical cloud-free reference of the same location

### 2. Single-Image GAN Reconstruction (comparison / fallback method)
Uses a CycleGAN checkpoint plus classical inpainting to guess the hidden content directly
from the cloudy image, with no historical reference required.

- **Validated on the same test case:** EdgeCorr 0.44, SSIM 0.78
- Requires: only the current (cloudy) image
- Lower fidelity than the reference-based method, but usable when no historical image
  is available — matches the single-image UX implied by the problem statement's stated
  input/output, at a real accuracy cost we measured directly rather than assumed.

Both methods are exposed as a mode toggle in the dashboard so the tradeoff is visible
and testable, satisfying the "comparative assessment of different Generative AI
architectures" objective in the problem statement.

## Architecture

```
Cloudy Image ──┐
               ├─► Cloud Mask ──► Reconstruction ──► Cloud-Free Output
Historical Ref ┘   (Detection)    (Reference or GAN)      + Metrics
                                                        + Confidence Map
```

| Stage | Module | Owner |
|---|---|---|
| Preprocessing & fusion (alignment, calibration, resampling) | `backend/preprocessing/` | Omkrrish |
| Cloud detection (SimpleUNet, 4-band input) | `cloud_mask_wrapper.py`, `src/model.py` | Deeptanshu |
| Reconstruction (reference-based + GAN hybrid) | `refine.py`, `inference.py`, `color_fix.py` | Dhruv |
| Integration adapter (wires preprocessing → reconstruction) | `integration_adapter.py` | Dhruv |
| Evaluation metrics (PSNR, SSIM, RMSE, SAM, EdgeCorr) | `evaluation/metrics.py` | — |
| Confidence map generation | `evaluation/confidence.py` | — |
| API backend | `api_server.py` | Dhruv |
| Frontend dashboard | `frontend/` | Darsh |

## Results

Validated on real paired cloudy/clean samples from `raw_data/`:

| Sample | Method | PSNR | SSIM | RMSE | SAM | EdgeCorr |
|---|---|---|---|---|---|---|
| sample_70 | Reference-based | 19.95 | 0.9013 | 25.64 | 5.13 | **0.9543** |
| sample_57 | Reference-based | 14.83 | 0.8381 | 46.24 | 5.68 | **0.9111** |
| sample_57 | Single-image GAN | 17.70 | 0.7771 | 33.24 | 7.06 | 0.4424 |

EdgeCorr (structural edge correlation with ground truth) is the most informative metric
here: it shows the reference-based method preserves real terrain structure (rivers,
ridgelines) far better than pure generative guessing, at the cost of requiring a
historical image.

## Setup

```bash
git clone <repo-url>
cd cloud-reconstruction
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

## Running the dashboard

Start the backend:
```bash
uvicorn api_server:app --reload --port 8000
```

Open `frontend/index.html` in a browser. Upload a cloudy image (and, in Reference mode,
a historical cloud-free reference of the same location), then click **Process Image**.

## Running the validated tests directly

```bash
python test_pipeline_call_57.py    # reference-based, sample_57
python test_pipeline_call_70.py    # reference-based, sample_70
python test_hybrid_sample57.py     # single-image GAN, sample_57
```

## Known limitations / next steps

- **Cloud mask generation**: `cloud_mask_wrapper.py` wraps Deeptanshu's real SimpleUNet
  detector, which expects separate red/green/blue/NIR band GeoTIFFs (e.g. Landsat-style
  data). The dashboard currently derives masks from a current/historical image diff as
  a stand-in, since browser uploads don't provide separate bands. Wiring the real
  detector in requires band-separated input imagery.
- **Historical reference sourcing**: the dashboard currently requires the user to
  manually supply a historical image in Reference mode. `build_historical_library.py`
  prototypes an automatic lookup (color-histogram matching against a small reference
  library) for a more production-like single-upload UX; this should be replaced with
  real geolocation-based archive lookup for operational deployment.
- **GAN model quality**: the single-image path (EdgeCorr 0.44) would benefit from
  further training — more epochs, more diverse cloud patterns, or a larger architecture
  — if it's to be used as a primary method rather than a comparison/fallback.

## Tech stack

Python, PyTorch, FastAPI, OpenCV, Rasterio, GDAL, NumPy, CycleGAN/Pix2Pix, HTML/CSS/JS
(vanilla frontend).

## Team

- **Deeptanshu** — Cloud Detection & Integration
- **Dhruv** — Generative Reconstruction, API, Integration
- **Omkrrish** — Data Fusion & Preprocessing
- **Darsh** — Frontend Dashboard
