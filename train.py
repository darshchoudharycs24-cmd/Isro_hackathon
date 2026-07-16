import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.dataset_multiband import CloudDataset
from src.model import SimpleUNet


ROOT = r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud"

# Load dataset
dataset = CloudDataset(ROOT)

# Train on first 500 samples for testing
dataset = Subset(dataset, range(500))

print(f"Training on {len(dataset)} samples")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Model
model = SimpleUNet().to(device)

# Loss
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    print(f"\nStarting Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, masks) in enumerate(loader):

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)}"
            )

        images = images.to(device)
        masks = masks.to(device)

        # Safety checks
        if torch.isnan(images).any():
            print("NaN detected in images")
            continue

        optimizer.zero_grad()

        outputs = model(images)

        if torch.isnan(outputs).any():
            print("NaN detected in outputs")
            continue

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "cloud_detector.pth"
)

print("\nTraining Complete")
print("Model saved as cloud_detector.pth")