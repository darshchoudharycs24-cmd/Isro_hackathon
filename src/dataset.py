import os
import numpy as np
import rasterio
from torch.utils.data import Dataset

class CloudDataset(Dataset):
    def __init__(self, red_dir, gt_dir):
        self.red_dir = red_dir
        self.gt_dir = gt_dir

        self.red_files = sorted([
            f for f in os.listdir(red_dir)
            if f.endswith(".TIF")
        ])

    def __len__(self):
        return len(self.red_files)

    def __getitem__(self, idx):
        red_name = self.red_files[idx]

        gt_name = red_name.replace("red_", "gt_")

        red_path = os.path.join(self.red_dir, red_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        with rasterio.open(red_path) as src:
            image = src.read(1).astype(np.float32)

        with rasterio.open(gt_path) as src:
            mask = src.read(1).astype(np.float32)

        image = image / image.max()

        if mask.max() > 1:
            mask = mask / 255.0

        return image, mask