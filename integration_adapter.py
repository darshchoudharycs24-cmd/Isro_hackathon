"""
Integration adapter: Omkrrish's PreprocessingPipeline -> Dhruv's reconstruction model.
"""

import numpy as np
import cv2

from inference import CloudRemovalModel
from color_fix import match_color_stats
from refine import refine_reconstruction, refine_with_reference
from evaluation.metrics import compute_metrics
from evaluation.confidence import generate_confidence_map


def float01_rgb_to_uint8_bgr(arr):
    """Convert float32 [0,1] RGB array -> uint8 [0,255] BGR array (OpenCV convention)."""
    arr = np.clip(arr, 0, 1)
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    if arr_uint8.shape[-1] == 3:
        arr_uint8 = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR)
    return arr_uint8


def mask_to_uint8(mask):
    """Convert float32 binary mask (0/1) -> uint8 (0/255) mask."""
    if mask is None:
        return None
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return mask_uint8


def run_full_pipeline(reconstruction_input, checkpoint_path, method="reference"):
    """
    reconstruction_input: Omkrrish's ReconstructionInput dataclass
    checkpoint_path: path to trained CycleGAN generator checkpoint
    method: "reference" (primary, uses historical image) or "hybrid" (fallback)
    """
    current_cloudy = float01_rgb_to_uint8_bgr(reconstruction_input.current_image)
    historical_clean = float01_rgb_to_uint8_bgr(reconstruction_input.historical_image)
    mask = mask_to_uint8(reconstruction_input.cloud_mask)

    if mask is None:
        raise ValueError(
            "No cloud mask available in ReconstructionInput. "
            "Cloud detection module output is required for reconstruction."
        )

    if method == "reference":
        final = refine_with_reference(historical_clean, current_cloudy, mask)
    else:
        model = CloudRemovalModel(checkpoint_path, gpu_id=-1)
        raw_gan = model.reconstruct(current_cloudy)
        gan_fixed = match_color_stats(raw_gan, current_cloudy, mask=mask)
        final, _, _ = refine_reconstruction(gan_fixed, current_cloudy, mask)

    metrics = compute_metrics(historical_clean, final)
    confidence, heatmap = generate_confidence_map(mask)

    return {
        "final_image": final,
        "metrics": metrics,
        "confidence": confidence,
        "heatmap": heatmap,
        "mask": mask,
        "current_cloudy": current_cloudy,
        "historical_clean": historical_clean,
    }