import cv2
import pytesseract
import json
import os
import re
import sys
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)
from plate_preprocess import clean_ocr_text, post_process_ocr
from config import LANG
from QR_Processor.qr_reader import read_qr_from_image
from OCR_Engine.utils import normalize_wada_number, preprocess_number
from test_models import ocr_image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# def run_plate_ocr_on_roi(roi_dict, rois_json_path=None):
#     """
#     Run OCR + QR scan on already cropped ROI images.

#     Parameters:
#         roi_dict: dict
#             - field_name -> ROI image (numpy array)
#         rois_json_path: str, optional
#             - path to JSON defining ROIs (needed for field order)
#     Returns:
#         dict of field -> text/QR values
#     """
#     print("\n================= OCR PIPELINE START =================")
    
#     # Load ROIs keys if JSON path provided, else use roi_dict keys
#     if rois_json_path:
#         with open(rois_json_path, "r") as f:
#             ROIS = json.load(f)
#         print(f"[INFO] Loaded ROIs: {list(ROIS.keys())}")
#         fields = ROIS.keys()
#     else:
#         fields = roi_dict.keys()
#         print(f"[INFO] Using ROI dict keys: {list(fields)}")
    
#     results = {}

#     for field in fields:
#         print("\n--------------------------------------")
#         print(f"[FIELD] Processing: {field}")

#         if field not in roi_dict:
#             print(f" ⚠️ ROI image for field '{field}' not provided")
#             results[field] = ""
#             continue

#         cropped = roi_dict[field]
#         if cropped is None or cropped.size == 0:
#             print(" Cropped image EMPTY")
#             results[field] = ""
#             continue

#         # ================= Add padding around ROI =================
#         h, w = cropped.shape[:2]
#         pad_x = int(0.02 * w)  # 2% padding
#         pad_y = int(0.02 * h)
#         x1, y1 = max(0, -pad_x), max(0, -pad_y)
#         x2, y2 = min(w, w + pad_x), min(h, h + pad_y)
#         cropped = cropped[y1:y2, x1:x2]

#         print(f"[INFO] ROI shape: {cropped.shape}")

#         # ================= QR FIELD =================
#         if field == "QR_Code":
#             print("🔍 Attempting QR decode on ROI image...")
#             qr_value = read_qr_from_image(cropped, verbose=True)
#             print(f"[RESULT] QR decoded value: '{qr_value}'")
#             results["QR_Code"] = qr_value
#             continue

#         # ================= OCR FIELD =================
#         if len(cropped.shape) == 3:
#             gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = cropped.copy()

#         # ================= Apply CLAHE / adaptive lighting =================
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         gray = clahe.apply(gray)

#         _, thresh = cv2.threshold(
#             gray, 0, 255,
#             cv2.THRESH_BINARY + cv2.THRESH_OTSU
#         )

#         # ================= Set language and PSM per field =================
#         if field in ["KID_No", "Plus_Code"]:
#             lang = "eng"
#         elif field in ["Local_Address"]:
#             lang = "hin"
#         else:
#             lang = LANG  # from config.py

#         psm = 8 if field in ["Local_Address"] else 6  # single line for small fields
#         print(f"[INFO] OCR language: {lang}, PSM: {psm}")

#         text = pytesseract.image_to_string(
#             thresh,
#             lang=lang,
#             config=f"--oem 3 --psm {psm}"
#         ).strip()

#         print(f"[RAW OCR] '{text}'")

#         text = normalize_wada_number(text)
#         text = clean_ocr_text(field, text)
#         # text = preprocess_number(text)
#         text = post_process_ocr(field, text)
#         results[field] = text

#     print("\n================= OCR PIPELINE END =================\n")
#     return results



# import cv2
# import pytesseract
# import re
# import numpy as np

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def ocr_rois(roi_dict, clean_text=True):
#     results = {}

#     for field, cropped in roi_dict.items():
#         if cropped is None or cropped.size == 0:
#             results[field] = ""
#             continue

#         # ================= FIELD CONFIG =================
#         if field in ["KID_No", "Plus_Code"]:
#             lang, psm, oem = "eng", 6, 3

#         elif field == "Local_Address":
#             lang, psm, oem = "hin", 8, 3

#         elif field == "Ward_Address":
#             lang, psm, oem = "nep", 6, 1   # same as your working case

#         elif field == "QR_Code":
#             print("🔍 Attempting QR decode...")
#             results[field] = read_qr_from_image(cropped, verbose=True)
#             continue

#         else:
#             lang, psm, oem = LANG, 6, 3

#         print(f"[INFO] {field} → lang={lang}, psm={psm}, oem={oem}")

#         # ================= OCR (single path) =================
#         text = ocr_image(
#             img=cropped,
#             lang=lang,
#             psm=psm,
#             oem=oem,
#             clean_text=clean_text
#         )

#         results[field] = text

#     return results


def ocr_image_rois(roi_dict, clean_text=True):
    """
    Run Tesseract OCR on ROI dictionary and return cleaned text per field.
    
    Parameters:
        roi_dict : dict
            - field_name -> numpy array (BGR/gray)
        lang : str
            - Language code for Tesseract
        psm : int
            - Page segmentation mode (Tesseract)
        clean_text : bool
            - Whether to apply text cleanup
    
    Returns:
        dict : field_name -> OCR extracted text
    """
    results = {}

    for field, img in roi_dict.items():
        if img is None or img.size == 0:
            results[field] = ""
            continue

        if field in ["KID_No", "Plus_Code"]:
            lang, psm, oem = "eng", 6, 3

        elif field == "Local_Address":
            lang, psm, oem = "hin", 11, 3

        elif field  == "City":
            lang, psm, oem = "hin", 11, 3   # same as your working case

        elif field == "Ward_Address":
            lang, psm, oem = "nep",11 , 3

        elif field == "QR_Code":
            print("🔍 Attempting QR decode...")
            results[field] = read_qr_from_image(img, verbose=True)
            continue

        else:
            lang, psm, oem = LANG, 6, 3

        print(f"[INFO] {field} → lang={lang}, psm={psm}, oem={oem}")

        # Convert to grayscale if needed (SAME as ocr_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # Thresholding (SAME)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # OCR (SAME)
        text = pytesseract.image_to_string(
            thresh,
            lang=lang,
            config=f"--oem 1 --psm {psm}"
        ).strip()

        if clean_text:
            text = re.sub(
                r'[^\w\s\-\/\+\:\u0900-\u097F]', '',
                text
            )
            text = re.sub(r'\s+', ' ', text).strip()

        results[field] = text

    return results
