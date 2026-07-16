import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(self, root_dir):
        self.red_dir = os.path.join(root_dir, "train_red_additional_to38cloud")
        self.green_dir = os.path.join(root_dir, "train_green_additional_to38cloud")
        self.blue_dir = os.path.join(root_dir, "train_blue_additional_to38cloud")
        self.nir_dir = os.path.join(root_dir, "train_nir_additional_to38cloud")
        self.gt_dir = os.path.join(root_dir, "train_gt_additional_to38cloud")

        self.red_files = sorted(
            [f for f in os.listdir(self.red_dir) if f.endswith(".TIF")]
        )

        print(f"Found {len(self.red_files)} training samples")

    def __len__(self):
        return len(self.red_files)

    def __getitem__(self, idx):

        red_file = self.red_files[idx]

        green_file = red_file.replace("red_", "green_")
        blue_file = red_file.replace("red_", "blue_")
        nir_file = red_file.replace("red_", "nir_")
        gt_file = red_file.replace("red_", "gt_")

        with rasterio.open(os.path.join(self.red_dir, red_file)) as src:
            red = src.read(1)

        with rasterio.open(os.path.join(self.green_dir, green_file)) as src:
            green = src.read(1)

        with rasterio.open(os.path.join(self.blue_dir, blue_file)) as src:
            blue = src.read(1)

        with rasterio.open(os.path.join(self.nir_dir, nir_file)) as src:
            nir = src.read(1)

        with rasterio.open(os.path.join(self.gt_dir, gt_file)) as src:
            mask = src.read(1)

        image = np.stack([red, green, blue, nir], axis=0)

        image = image.astype(np.float32)

        max_val = image.max()

        if max_val > 0:
            image = image / max_val
        else:
            image = np.zeros_like(image, dtype=np.float32)

        mask = mask.astype(np.float32)

        if mask.max() > 1:
            mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask