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


def refine_with_reference(historical_clean, current_cloudy, mask, feather=31):
    """
    Robust reconstruction using a historical cloud-free reference image
    of the same location. Works regardless of cloud size/shape since it
    copies REAL ground-truth pixels rather than guessing.

    Uses feathered alpha compositing instead of Poisson blending
    (cv2.seamlessClone), because seamlessClone requires the masked
    region's bounding box to fit well within the image borders with
    margin -- real predicted masks can be large, irregular, and touch
    image edges, which breaks that assumption and throws an OpenCV
    assertion error. Alpha blending has no such constraint and is a
    more natural fit here anyway: we're compositing real ground-truth
    pixels, not seamlessly hiding a synthetic guess, so we don't need
    Poisson blending's gradient-domain color matching.
    """
    if mask.max() == 0:
        return current_cloudy.copy()  # no cloud, nothing to do

    if historical_clean.shape != current_cloudy.shape:
        historical_clean = cv2.resize(
            historical_clean, (current_cloudy.shape[1], current_cloudy.shape[0])
        )

    # Feather the mask edges so the seam isn't a hard cut
    feather = feather if feather % 2 == 1 else feather + 1  # must be odd
    mask_blur = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha = (mask_blur.astype(np.float32) / 255.0)[..., None]

    result = (
        alpha * historical_clean.astype(np.float32)
        + (1 - alpha) * current_cloudy.astype(np.float32)
    ).astype(np.uint8)

    return result
    """
    Robust reconstruction using a historical cloud-free reference image
    of the same location. Works regardless of cloud size since it copies
    REAL ground-truth pixels rather than guessing.
    """
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return current_cloudy.copy()  # no cloud, nothing to do

    # cv2.seamlessClone requires the cloned region to have margin from the
    # image border -- pad everything with a border first, clone, then crop
    # back. This handles real predicted masks whose cloud regions can
    # touch or extend to the image edge (unlike clean synthetic circles).
    pad = 20
    historical_padded = cv2.copyMakeBorder(historical_clean, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    current_padded = cv2.copyMakeBorder(current_cloudy, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mask_padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    ys_p, xs_p = np.where(mask_padded > 0)
    center = (int(xs_p.mean()), int(ys_p.mean()))

    result_padded = cv2.seamlessClone(
        historical_padded, current_padded, mask_padded, center, cv2.NORMAL_CLONE
    )

    # Crop back to original size
    result = result_padded[pad:pad + h, pad:pad + w]
    return result
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