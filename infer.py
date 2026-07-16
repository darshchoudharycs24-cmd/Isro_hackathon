import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from src.model import SimpleUNet

# -----------------------------
# Image paths
# -----------------------------

red_path = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_red_additional_to38cloud\red_patch_100_5_by_12_LC08_L1TP_006248_20160820_20170322_01_T1.TIF"

green_path = red_path.replace(
    "train_red_additional_to38cloud",
    "train_green_additional_to38cloud"
).replace("red_", "green_")

blue_path = red_path.replace(
    "train_red_additional_to38cloud",
    "train_blue_additional_to38cloud"
).replace("red_", "blue_")

nir_path = red_path.replace(
    "train_red_additional_to38cloud",
    "train_nir_additional_to38cloud"
).replace("red_", "nir_")

# -----------------------------
# Load bands
# -----------------------------

with rasterio.open(red_path) as src:
    red = src.read(1)

with rasterio.open(green_path) as src:
    green = src.read(1)

with rasterio.open(blue_path) as src:
    blue = src.read(1)

with rasterio.open(nir_path) as src:
    nir = src.read(1)

image = np.stack(
    [red, green, blue, nir],
    axis=0
).astype(np.float32)

if image.max() > 0:
    image = image / image.max()

image = torch.tensor(
    image,
    dtype=torch.float32
).unsqueeze(0)

# -----------------------------
# Load model
# -----------------------------

model = SimpleUNet()

model.load_state_dict(
    torch.load(
        "cloud_detector.pth",
        map_location="cpu"
    )
)

model.eval()

# -----------------------------
# Predict
# -----------------------------

with torch.no_grad():

    output = model(image)

    prediction = torch.sigmoid(output)

prediction = prediction.squeeze().numpy()

# -----------------------------
# Show mask
# -----------------------------

plt.figure(figsize=(8, 8))

plt.figure(figsize=(8, 8))
plt.imshow(prediction, cmap="gray")
plt.title("Predicted Cloud Mask")
plt.colorbar()

plt.savefig("predicted_mask.png")
plt.show()