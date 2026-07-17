import cv2
import numpy as np


def generate_confidence_map(cloud_mask):
    """
    Generates a confidence map where pixels deep inside the cloud region
    get low confidence, and pixels near clear areas get high confidence.
    """
    inv_mask = cv2.bitwise_not(cloud_mask)
    dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    confidence = 255 - dist_norm
    heatmap = cv2.applyColorMap(confidence, cv2.COLORMAP_JET)
    return confidence, heatmap