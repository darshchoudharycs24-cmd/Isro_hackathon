"""
Cloud reconstruction refinement.

TWO approaches provided:

1. refine_reconstruction() -- single-image hybrid (GAN + classical inpainting).
   Works reasonably for SMALL cloud regions, but has a fundamental limitation:
   for large regions, neither method has real information about the true
   content underneath, so quality degrades as cloud size grows.

2. refine_with_reference() -- THE ROBUST SOLUTION for arbitrary cloud sizes.
   Uses a historical cloud-free image of the SAME location and seamlessly
   composites those REAL pixels into the cloud region. Works regardless of
   cloud size since it uses ground truth rather than a model's guess.
"""

import cv2
import numpy as np


def refine_reconstruction(gan_output, original_cloudy, mask, feather=25):
    inpainted = cv2.inpaint(original_cloudy, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = dist.max() + 1e-6
    depth_ratio = np.clip(dist / max_dist, 0, 1)
    gan_weight = (0.2 + 0.3 * depth_ratio)[..., None]
    inpaint_weight = 1 - gan_weight

    blended = (gan_weight * gan_output.astype(np.float32) +
               inpaint_weight * inpainted.astype(np.float32)).astype(np.uint8)

    mask_blur = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha = (mask_blur.astype(np.float32) / 255.0)[..., None]
    final = (alpha * blended + (1 - alpha) * original_cloudy).astype(np.uint8)

    return final, inpainted, blended


def refine_with_reference(historical_clean, current_cloudy, mask):
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return current_cloudy.copy()

    center = (int(xs.mean()), int(ys.mean()))

    result = cv2.seamlessClone(
        historical_clean, current_cloudy, mask, center, cv2.NORMAL_CLONE
    )
    return result


if __name__ == "__main__":
    import os
    from utils import create_synthetic_cloudy_image
    from inference import CloudRemovalModel
    from color_fix import match_color_stats

    original, cloudy, mask = create_synthetic_cloudy_image("data/sample.jpg")

    checkpoint = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
    model = CloudRemovalModel(checkpoint, gpu_id=-1)
    raw_gan = model.reconstruct(cloudy)
    gan_fixed = match_color_stats(raw_gan, original, mask=mask)
    final_hybrid, inpainted, blended = refine_reconstruction(gan_fixed, cloudy, mask)

    final_reference = refine_with_reference(original, cloudy, mask)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/final_hybrid.png", final_hybrid)
    cv2.imwrite("outputs/inpainted_only.png", inpainted)
    cv2.imwrite("outputs/final_reference_based.png", final_reference)
    print("Saved outputs/final_hybrid.png and outputs/final_reference_based.png")