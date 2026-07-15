"""
Organizes raw cloudy/clean images into the folder structure
required by pytorch-CycleGAN-and-pix2pix.

Put downloaded images in:
  raw_data/cloudy/
  raw_data/clean/

Then run: python prepare_dataset.py
"""

import os
import cv2
import random

RAW_CLOUDY_DIR = "raw_data/cloudy"
RAW_CLEAN_DIR = "raw_data/clean"
OUTPUT_ROOT = "pytorch-CycleGAN-and-pix2pix/datasets/cloud_removal"
IMG_SIZE = 256
TEST_SPLIT = 0.1
SEED = 42


def resize_and_save(src_path, dst_path, size=IMG_SIZE):
    img = cv2.imread(src_path)
    if img is None:
        print(f"  [skip] Could not read {src_path}")
        return False
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_path, img)
    return True


def process_folder(src_dir, train_dir, test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
    random.seed(SEED)
    random.shuffle(files)

    n_test = max(1, int(len(files) * TEST_SPLIT)) if files else 0
    test_files = set(files[:n_test])

    count_train, count_test = 0, 0
    for fname in files:
        src_path = os.path.join(src_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".png"
        if fname in test_files:
            count_test += resize_and_save(src_path, os.path.join(test_dir, out_name))
        else:
            count_train += resize_and_save(src_path, os.path.join(train_dir, out_name))

    return count_train, count_test


if __name__ == "__main__":
    print("Preparing cloudy (domain A) images...")
    a_train, a_test = process_folder(
        RAW_CLOUDY_DIR,
        os.path.join(OUTPUT_ROOT, "trainA"),
        os.path.join(OUTPUT_ROOT, "testA"),
    )

    print("Preparing clean (domain B) images...")
    b_train, b_test = process_folder(
        RAW_CLEAN_DIR,
        os.path.join(OUTPUT_ROOT, "trainB"),
        os.path.join(OUTPUT_ROOT, "testB"),
    )

    print("\n=== Dataset ready ===")
    print(f"trainA (cloudy): {a_train} images")
    print(f"trainB (clean):  {b_train} images")
    print(f"testA  (cloudy): {a_test} images")
    print(f"testB  (clean):  {b_test} images")
    print(f"\nLocation: {OUTPUT_ROOT}")

    if a_train < 100 or b_train < 100:
        print("\n[!] Warning: fewer than 100 training images per domain.")
        print("    CycleGAN will still run, but quality may be weak.")