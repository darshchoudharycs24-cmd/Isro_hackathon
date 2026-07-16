import os
import numpy as np
import rasterio

gt_dir = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_gt_additional_to38cloud"

files = sorted(
    [f for f in os.listdir(gt_dir) if f.endswith(".TIF")]
)

for f in files:

    path = os.path.join(gt_dir, f)

    with rasterio.open(path) as src:
        mask = src.read(1)

    cloud_pixels = np.sum(mask > 0)

    if cloud_pixels > 1000:
        print(f)
        print("Cloud Pixels:", cloud_pixels)
        break