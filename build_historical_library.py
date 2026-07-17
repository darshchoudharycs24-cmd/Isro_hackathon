"""
build_historical_library.py — one-time setup script.

Copies raw_data/clean/* into backend/data/historical_library/ and builds a
simple index (color-histogram signatures) so the API can auto-select the
best-matching historical reference for an arbitrary uploaded cloudy image,
without the user having to supply one manually.

Run once:
    python build_historical_library.py
"""

import json
import os
import shutil

import cv2
import numpy as np

CLEAN_DIR = "raw_data/clean"
LIBRARY_DIR = "backend/data/historical_library"
INDEX_PATH = "backend/data/historical_library_index.json"


def compute_signature(img_bgr: np.ndarray) -> list:
    """Cheap, rotation/crop-agnostic-ish signature: per-channel histogram."""
    hist = []
    for ch in range(3):
        h = cv2.calcHist([img_bgr], [ch], None, [32], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.extend(h.tolist())
    return hist


def main():
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    index = {}

    files = sorted(f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(".png"))
    for fname in files:
        src_path = os.path.join(CLEAN_DIR, fname)
        dst_path = os.path.join(LIBRARY_DIR, fname)
        shutil.copy2(src_path, dst_path)

        img = cv2.imread(src_path)
        if img is None:
            continue
        index[fname] = compute_signature(img)

    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)

    print(f"Library built: {len(index)} reference images in {LIBRARY_DIR}")
    print(f"Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()