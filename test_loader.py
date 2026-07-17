from src.dataset import CloudDataset

dataset = CloudDataset(
    red_dir=r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_red_additional_to38cloud",
    gt_dir=r"C:\Users\user\Downloads\archive\95-cloud_training_only_additional_to38-cloud\train_gt_additional_to38cloud"
)

print("Dataset Size:", len(dataset))

image, mask = dataset[0]

print("Image Shape:", image.shape)
print("Mask Shape :", mask.shape)