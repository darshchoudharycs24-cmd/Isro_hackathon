import numpy as np
import rasterio
import torch

from src.model import SimpleUNet

# Same paths as infer.py
red_path = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_red_additional_to38cloud\red_patch_100_5_by_12_LC08_L1TP_006248_20160820_20170322_01_T1.TIF"

gt_path = red_path.replace(
    "train_red_additional_to38cloud",
    "train_gt_additional_to38cloud"
).replace("red_", "gt_")

# Load GT mask
with rasterio.open(gt_path) as src:
    gt = src.read(1)

gt = (gt > 0).astype(np.uint8)

print("GT Cloud Pixels:", gt.sum())
print("GT Shape:", gt.shape)