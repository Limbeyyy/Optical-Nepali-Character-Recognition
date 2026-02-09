# utils.py
import cv2
import os
import re
import numpy as np
import pytesseract

# ==========================================================
# TEXT CLEANING
# ==========================================================

def basic_text_cleanup(text: str) -> str:
    """
    Remove unwanted special characters while preserving:
    - Word chars
    - Nepali unicode range
    - -, /, +, :
    """
    text = re.sub(
        r"[^\w\s\-\/\+\:\u0900-\u097F]",
        "",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale and apply Otsu thresholding.
    Keeps preprocessing consistent and reusable.
    """
    if image is None or image.size == 0:
        return None

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh

# ==========================================================
# OCR EXECUTION
# ==========================================================

def run_tesseract(image: np.ndarray, lang: str, psm: int, oem: int) -> str:
    """
    Run Tesseract OCR with provided config.
    """
    config_str = f"--oem {oem} --psm {psm}"

    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=config_str
    ).strip()

    return text





