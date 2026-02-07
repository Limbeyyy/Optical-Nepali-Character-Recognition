import cv2
import pytesseract
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_image(img, lang, psm, oem=1, clean_text=True):
    """
    Core OCR engine. ALL OCR must go through this function.
    """

    # Load image if path provided
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            print("❌ Failed to read image")
            return ""

    # IMPORTANT: force contiguous memory
    img = img.copy()

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Normalize (stability booster)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    config = f"--oem {oem} --psm {psm} -c user_defined_dpi=300"

    text = pytesseract.image_to_string(
        thresh,
        lang=lang,
        config=config
    ).strip()

    if not clean_text:
        return text

    # Cleaning (same everywhere)
    text = re.sub(r'[^\w\s\-\/\+\:\u0900-\u097F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
