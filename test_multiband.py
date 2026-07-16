from src.dataset_multiband import CloudDataset

dataset = CloudDataset(
    r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud"
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)