import cv2
import numpy as np

def preprocess_plate(plate_img):
    """
    Returns binarized, OCR-friendly plate image
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # upscale for OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # remove noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return th
