import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def spectral_angle_mapper(img1, img2):
    img1_f = img1.reshape(-1, img1.shape[-1]).astype(float)
    img2_f = img2.reshape(-1, img2.shape[-1]).astype(float)
    dot = np.sum(img1_f * img2_f, axis=1)
    norm1 = np.linalg.norm(img1_f, axis=1)
    norm2 = np.linalg.norm(img2_f, axis=1)
    cos_angle = dot / (norm1 * norm2 + 1e-8)
    angles = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(np.mean(angles))


def edge_preservation_score(original, reconstructed):
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    recon_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    orig_edges = cv2.Laplacian(orig_gray, cv2.CV_64F)
    recon_edges = cv2.Laplacian(recon_gray, cv2.CV_64F)
    correlation = np.corrcoef(orig_edges.flatten(), recon_edges.flatten())[0, 1]
    return correlation


def compute_metrics(original, reconstructed):
    rmse = np.sqrt(np.mean((original.astype(float) - reconstructed.astype(float)) ** 2))
    psnr_val = psnr(original, reconstructed, data_range=255)
    ssim_val = ssim(original, reconstructed, channel_axis=2, data_range=255)
    sam_val = spectral_angle_mapper(original, reconstructed)
    edge_val = edge_preservation_score(original, reconstructed)
    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "RMSE": rmse,
        "SAM": sam_val,
        "EdgeCorr": edge_val,
    }