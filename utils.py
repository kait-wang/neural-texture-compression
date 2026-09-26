import numpy as np 
import torch
from PIL import Image

def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def load_texture(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0

def psnr(original, recon):
    mse = np.mean((original - recon) ** 2)
    if mse == 0:
        return float("inf")
    return -10 * np.log10(mse)

def save_image(img, path):
    """
    Save a float image in [0, 1] as an 8-bit PNG.
    """
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)