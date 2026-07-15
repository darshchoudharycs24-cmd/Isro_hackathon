import cv2
import numpy as np
import os


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def create_synthetic_cloudy_image(path):
    """
    Creates: Original Image, Cloudy Image, Cloud Mask (binary)
    """
    img = load_image(path)
    h, w = img.shape[:2]

    binary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(binary_mask, (w // 3, h // 3), 35, 255, -1)
    cv2.circle(binary_mask, (w // 2, h // 2), 45, 255, -1)
    cv2.circle(binary_mask, (2 * w // 3, h // 3), 30, 255, -1)

    soft_mask = cv2.GaussianBlur(binary_mask, (41, 41), 15)

    cloudy = img.copy()
    alpha = soft_mask.astype(np.float32) / 255.0

    for c in range(3):
        cloudy[:, :, c] = (alpha * 255 + (1 - alpha) * cloudy[:, :, c])

    # Processing mask now matches the actual visible white halo,
    # not just the original tight circles
    _, processing_mask = cv2.threshold(soft_mask, 10, 255, cv2.THRESH_BINARY)

    return img, cloudy, processing_mask