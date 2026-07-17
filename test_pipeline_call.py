from datetime import date
from backend.preprocessing.pipeline import PreprocessingPipeline
from integration_adapter import run_full_pipeline
import cv2

result = PreprocessingPipeline().run(
    current_path="backend/data/current/lissiv.tif",
    historical_dir="backend/data/historical/",
    cloud_mask_path="backend/data/mask.png",
    current_date=date(2023, 6, 15),
    sample_id="scene_001",
)

ri = result.reconstruction_input
print("cloud_mask shape:", None if ri.cloud_mask is None else ri.cloud_mask.shape)

checkpoint = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
output = run_full_pipeline(ri, checkpoint, method="reference")

cv2.imwrite("outputs/real_integration_final.png", output["final_image"])
cv2.imwrite("outputs/real_integration_heatmap.png", output["heatmap"])

print("\n=== Integration Metrics ===")
for k, v in output["metrics"].items():
    print(f"{k:10}: {v:.4f}")
print("\nSaved outputs/real_integration_final.png")