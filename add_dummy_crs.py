import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import cv2
import numpy as np


def add_dummy_georeference(input_path, output_path, pixel_size=10.0):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {input_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    transform = from_origin(0, 0, pixel_size, pixel_size)
    crs = CRS.from_epsg(4326)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=3, dtype=img_rgb.dtype,
        crs=crs, transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(img_rgb[:, :, i], i + 1)

    print(f"Saved georeferenced version: {output_path}")


if __name__ == "__main__":
    add_dummy_georeference("backend/data/current/sample_1_rgb.png",
                            "backend/data/current/sample_1_rgb.tif")
    add_dummy_georeference("backend/data/historical/sample_1_rgb.png",
                            "backend/data/historical/sample_1_rgb_hist.tif")