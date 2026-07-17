"""
Wrapper around Deeptanshu's SimpleUNet cloud detection model.
Produces a clean binary mask.png (uint8, 255=cloud, 0=clear).
"""

import torch
import numpy as np
import rasterio
import cv2
from src.model import SimpleUNet


def load_four_band_image(red_path, green_path, blue_path, nir_path):
    with rasterio.open(red_path) as src:
        red = src.read(1)
    with rasterio.open(green_path) as src:
        green = src.read(1)
    with rasterio.open(blue_path) as src:
        blue = src.read(1)
    with rasterio.open(nir_path) as src:
        nir = src.read(1)

    image = np.stack([red, green, blue, nir], axis=0).astype(np.float32)
    if image.max() > 0:
        image = image / image.max()

    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


def predict_cloud_mask(red_path, green_path, blue_path, nir_path,
                        checkpoint_path="cloud_detector.pth",
                        threshold=0.5):
    model = SimpleUNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    image = load_four_band_image(red_path, green_path, blue_path, nir_path)

    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output)

    probability = probability.squeeze().numpy()
    binary_mask = (probability > threshold).astype(np.uint8) * 255

    return binary_mask, probability


def save_mask_for_pipeline(binary_mask, output_path="mask.png"):
    cv2.imwrite(output_path, binary_mask)
    print(f"Saved clean binary mask to {output_path}")


if __name__ == "__main__":
    red_path = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_red_additional_to38cloud\red_patch_100_5_by_12_LC08_L1TP_006248_20160820_20170322_01_T1.TIF"
    green_path = red_path.replace("train_red_additional_to38cloud", "train_green_additional_to38cloud").replace("red_", "green_")
    blue_path = red_path.replace("train_red_additional_to38cloud", "train_blue_additional_to38cloud").replace("red_", "blue_")
    nir_path = red_path.replace("train_red_additional_to38cloud", "train_nir_additional_to38cloud").replace("red_", "nir_")

    binary_mask, probability = predict_cloud_mask(red_path, green_path, blue_path, nir_path)
    save_mask_for_pipeline(binary_mask, "mask.png")