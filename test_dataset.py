import rasterio
import matplotlib.pyplot as plt

red_path = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_red_additional_to38cloud\red_patch_100_5_by_12_LC08_L1TP_006248_20160820_20170322_01_T1.TIF"

gt_path = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_gt_additional_to38cloud\gt_patch_100_5_by_12_LC08_L1TP_006248_20160820_20170322_01_T1.TIF"

with rasterio.open(red_path) as src:
    red = src.read(1)

with rasterio.open(gt_path) as src:
    mask = src.read(1)

print("Red shape :", red.shape)
print("Mask shape:", mask.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(red, cmap="gray")
plt.title("Red Band")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Cloud Mask")

plt.tight_layout()
plt.show()
import numpy as np
print("Unique mask values:", np.unique(mask))