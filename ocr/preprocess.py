import cv2
import torch
import numpy as np
from config import IMG_SIZE

def preprocess_char(img, enhance=True):
    """
    Preprocess a single character image for PyTorch model.
    
    Args:
        img: np.array, BGR image
        enhance: bool, apply CLAHE for contrast enhancement
    
    Returns:
        torch.Tensor: shape (1, 1, IMG_SIZE, IMG_SIZE)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if enhance:
        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    # Invert colors if background is white
    if np.mean(gray) > 127:
        gray = 255 - gray

    # Resize while keeping aspect ratio
    h, w = gray.shape
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to make square
    pad_h = (IMG_SIZE - new_h) // 2
    pad_w = (IMG_SIZE - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, IMG_SIZE - new_h - pad_h,
        pad_w, IMG_SIZE - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=0  # black background
    )

    # Normalize to [-1, 1]
    img_tensor = padded.astype("float32") / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5

    # Add channel and batch dimensions
    return torch.tensor(img_tensor).unsqueeze(0).unsqueeze(0)
