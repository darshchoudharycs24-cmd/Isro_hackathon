"""
Color-matching post-process for GAN reconstruction output.
"""

import cv2
import numpy as np


def match_color_stats(reconstructed, reference, mask=None):
    result = reconstructed.copy().astype(np.float32)
    reference = reference.astype(np.float32)

    if mask is not None:
        valid = mask < 128  # non-cloud pixels
    else:
        valid = np.ones(reference.shape[:2], dtype=bool)

    for c in range(3):
        ref_channel = reference[:, :, c][valid]
        recon_channel = result[:, :, c]

        ref_mean, ref_std = ref_channel.mean(), ref_channel.std() + 1e-6
        recon_mean, recon_std = recon_channel.mean(), recon_channel.std() + 1e-6

        result[:, :, c] = (recon_channel - recon_mean) / recon_std * ref_std + ref_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


if __name__ == "__main__":
    reconstructed = cv2.imread("outputs/reconstructed.png")
    original = cv2.imread("data/sample.jpg")

    if reconstructed is None or original is None:
        print("Run demo_final.py first to generate outputs/reconstructed.png")
    else:
        fixed = match_color_stats(reconstructed, original)
        cv2.imwrite("outputs/reconstructed_colorfixed.png", fixed)
        print("Saved outputs/reconstructed_colorfixed.png")