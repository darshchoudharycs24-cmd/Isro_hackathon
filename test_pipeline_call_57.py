from datetime import date
from backend.preprocessing.pipeline import PreprocessingPipeline
from integration_adapter import run_full_pipeline
import cv2

result = PreprocessingPipeline().run(
    current_path="backend/data/current/sample_57_rgb.png",
    historical_dir="backend/data/historical/",
    cloud_mask_path="backend/data/sample_57_mask.png",
    current_date=date(2023, 6, 15),
    sample_id="real_test_57",
)

ri = result.reconstruction_input
print("cloud_mask shape:", None if ri.cloud_mask is None else ri.cloud_mask.shape)

checkpoint = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
output = run_full_pipeline(ri, checkpoint, method="reference")

cv2.imwrite("outputs/sample_57_final.png", output["final_image"])
cv2.imwrite("outputs/sample_57_heatmap.png", output["heatmap"])

print("\n=== Integration Metrics (sample_57) ===")
for k, v in output["metrics"].items():
    print(f"{k:10}: {v:.4f}")