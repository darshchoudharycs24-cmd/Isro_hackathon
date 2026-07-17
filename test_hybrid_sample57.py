"""
Test the single-image ('hybrid') GAN + inpainting reconstruction directly,
bypassing PreprocessingPipeline (which requires GeoTIFF CRS metadata we don't
need for this comparison). Uses raw_data/cloudy/57.png as the only input to
reconstruction; raw_data/clean/57 (1).png is used ONLY for scoring, not fed
into the model, so this is a fair single-image-in test.
"""
import cv2
import numpy as np

from inference import CloudRemovalModel
from color_fix import match_color_stats
from refine import refine_reconstruction
from evaluation.metrics import compute_metrics

current_cloudy = cv2.imread("backend/data/current/sample_57_rgb.png")
true_clean = cv2.imread("backend/data/historical/sample_57_rgb_hist.png")  # scoring only

# Build the same ground-truth-diff mask we used before (for evaluation purposes;
# in a real single-image-only deployment this would come from Deeptanshu's detector)
diff = cv2.absdiff(current_cloudy, true_clean)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
mask = (diff_gray > 25).astype(np.uint8) * 255
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

checkpoint = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
model = CloudRemovalModel(checkpoint, gpu_id=-1)

raw_gan = model.reconstruct(current_cloudy)
gan_fixed = match_color_stats(raw_gan, current_cloudy, mask=mask)
final, inpainted, blended = refine_reconstruction(gan_fixed, current_cloudy, mask)

cv2.imwrite("outputs/hybrid_sample57_final.png", final)

metrics = compute_metrics(true_clean, final)
print("\n=== Integration Metrics (single-image GAN hybrid, sample_57) ===")
for k, v in metrics.items():
    print(f"{k:10}: {v:.4f}")