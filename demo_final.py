"""
Final Cloud Reconstruction Pipeline (Dhruv's Module)
Terror Byte - ISRO Bharatiya Antariksh Hackathon 2026
"""

import os
import cv2
import matplotlib.pyplot as plt

from utils import create_synthetic_cloudy_image
from evaluation.metrics import compute_metrics
from evaluation.confidence import generate_confidence_map
from inference import CloudRemovalModel
from color_fix import match_color_stats
from refine import refine_reconstruction, refine_with_reference

CHECKPOINT_PATH = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
USE_TRAINED_MODEL = os.path.exists(CHECKPOINT_PATH)

# In production: `original` is replaced by Omkrrish's historical cloud-free
# reference for the same location; `cloudy`/`mask` come from a real current
# cloudy scene + Deeptanshu's predicted cloud mask.
original, cloudy, mask = create_synthetic_cloudy_image("data/sample.jpg")
historical_reference = original  # stand-in until Omkrrish's fusion output is ready

if USE_TRAINED_MODEL:
    print("Using trained CycleGAN model...")
    model = CloudRemovalModel(CHECKPOINT_PATH, gpu_id=-1)
    raw_gan = model.reconstruct(cloudy)
    gan_fixed = match_color_stats(raw_gan, original, mask=mask)
else:
    print("[!] No trained checkpoint found — using placeholder.")
    gan_fixed = original.copy()

hybrid_result, inpainted, _ = refine_reconstruction(gan_fixed, cloudy, mask)
reference_result = refine_with_reference(historical_reference, cloudy, mask)

hybrid_metrics = compute_metrics(original, hybrid_result)
reference_metrics = compute_metrics(original, reference_result)

print("\n========== HYBRID (GAN + Inpainting) ==========")
for k, v in hybrid_metrics.items():
    print(f"{k:10}: {v:.4f}")

print("\n========== REFERENCE-BASED (Primary Method) ==========")
for k, v in reference_metrics.items():
    print(f"{k:10}: {v:.4f}")
print()

confidence, heatmap = generate_confidence_map(mask)

os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/cloudy.png", cloudy)
cv2.imwrite("outputs/mask.png", mask)
cv2.imwrite("outputs/confidence.png", heatmap)
cv2.imwrite("outputs/final_hybrid.png", hybrid_result)
cv2.imwrite("outputs/final_reference_based.png", reference_result)

fig = plt.figure(figsize=(18, 10))

plt.subplot(241); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
plt.subplot(242); plt.imshow(cv2.cvtColor(cloudy, cv2.COLOR_BGR2RGB)); plt.title("Cloudy Input"); plt.axis("off")
plt.subplot(243); plt.imshow(mask, cmap="gray"); plt.title("Cloud Mask"); plt.axis("off")
plt.subplot(244); plt.imshow(confidence, cmap="jet"); plt.title("Confidence Map"); plt.axis("off")

plt.subplot(245); plt.imshow(cv2.cvtColor(hybrid_result, cv2.COLOR_BGR2RGB))
plt.title(f"Hybrid (fallback)\nSSIM={hybrid_metrics['SSIM']:.2f}  SAM={hybrid_metrics['SAM']:.1f}")
plt.axis("off")

plt.subplot(246); plt.imshow(cv2.cvtColor(reference_result, cv2.COLOR_BGR2RGB))
plt.title(f"Reference-based (primary)\nSSIM={reference_metrics['SSIM']:.2f}  SAM={reference_metrics['SAM']:.1f}")
plt.axis("off")

plt.subplot(247); plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)); plt.title("Confidence Heatmap"); plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/summary_figure.png", dpi=150)
plt.show()