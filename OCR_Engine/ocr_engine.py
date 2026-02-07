import cv2
import pytesseract
import json
import os
from OCR_Engine.utils import crop_roi, normalize_wada_number
from config import LANG
from QR_Processor.qr_reader import read_qr_from_image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def run_plate_ocr(image_path_or_array, rois_json_path):
    """
    Run OCR + QR scan on a plate.
    
    Parameters:
        image_path_or_array: str or numpy array
            - str: path to image
            - numpy array: image already loaded (e.g., from webcam)
        rois_json_path: str
            - path to JSON defining ROIs
    Returns:
        dict of field -> text/QR values
    """
    print("\n================= OCR PIPELINE START =================")
    print(f"[INFO] ROI JSON path: {rois_json_path}")

    with open(rois_json_path, "r") as f:
        ROIS = json.load(f)

    print(f"[INFO] Loaded ROIs: {list(ROIS.keys())}")

    # Load image if path provided
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        print(f"[INFO] Image path: {image_path_or_array}")
        if img is None:
            print(" ERROR: Failed to read image from disk")
            return {}
    else:
        img = image_path_or_array
        print(f"[INFO] Using numpy array image. Shape: {img.shape}")

    results = {}

    for field, roi in ROIS.items():
        print("\n--------------------------------------")
        print(f"[FIELD] Processing: {field}")
        print(f"[ROI] {roi}")

        cropped = crop_roi(img, roi)

        if cropped is None or cropped.size == 0:
            print(" Cropped image EMPTY")
            results[field] = ""
            continue

        print(f"[INFO] Cropped shape: {cropped.shape}")

        # ================= QR FIELD =================
        if field == "QR_Code":
            print("🔍 Attempting QR decode on FULL image...")
            qr_value = read_qr_from_image(img, verbose=True)
            print(f"[RESULT] QR decoded value: '{qr_value}'")
            results["QR_Code"] = qr_value
            continue

        # ================= OCR FIELD =================
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        lang = "eng" if field in ["KID_No", "Plus_Code"] else LANG
        print(f"[INFO] OCR language: {lang}")

        text = pytesseract.image_to_string(
            thresh,
            lang=lang,
            config="--oem 3 --psm 6"
        ).strip()

        print(f"[RAW OCR] '{text}'")

        text = normalize_wada_number(text)
        results[field] = text

    print("\n================= OCR PIPELINE END =================\n")
    return results