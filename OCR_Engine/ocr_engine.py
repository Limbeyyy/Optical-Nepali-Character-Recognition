
import os
import sys
import re
import logging
from typing import Dict

import cv2
import pytesseract
import numpy as np

# ==========================================================
# PROJECT PATH SETUP
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from OCR_Engine.plate_postprocess import post_process_ocr
from QR_Processor.QR_utils import read_qr_from_image
from config import DEFAULT_PSM, DEFAULT_OEM, FIELD_OCR_CONFIG   
from OCR_Engine.engine_utils import run_tesseract, preprocess_for_ocr, basic_text_cleanup
from config import TESSERACT_PATH, LANG, DEFAULT_OEM, DEFAULT_PSM

# ==========================================================
# TESSERACT CONFIGURATION
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH 
)

# ==========================================================
# LOGGING SETUP
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# MAIN OCR FUNCTION
# ==========================================================

def ocr_image_rois(
    roi_dict: Dict[str, np.ndarray],
    clean_text: bool = True
) -> Dict[str, str]:
    """
    Run OCR on ROI dictionary and return extracted text.

    Parameters
    ----------
    roi_dict : dict
        field_name -> numpy image
    clean_text : bool
        Apply basic regex cleanup

    Returns
    -------
    dict : field_name -> OCR text
    """

    results: Dict[str, str] = {}

    for field, image in roi_dict.items():

        # ---------------- EMPTY CHECK ----------------
        if image is None or image.size == 0:
            results[field] = ""
            continue

        # ---------------- QR HANDLING ----------------
        if field == "QR_Code":
            logger.info("Attempting QR decode...")
            results[field] = read_qr_from_image(image, verbose=True)
            continue

        # ---------------- CONFIG SELECTION ----------------
        config = FIELD_OCR_CONFIG.get(
            field,
            {"lang": LANG, "psm": DEFAULT_PSM, "oem": DEFAULT_OEM}
        )

        lang = config["lang"]
        psm = config["psm"]
        oem = config["oem"]

        logger.info(f"{field} → lang={lang}, psm={psm}, oem={oem}")

        # ---------------- PREPROCESS ----------------
        processed = preprocess_for_ocr(image)
        if processed is None:
            results[field] = ""
            continue

        # ---------------- OCR ----------------
        text = run_tesseract(processed, lang, psm, oem)

        # ---------------- CLEANING ----------------
        if clean_text:
            text = basic_text_cleanup(text)

        results[field] = text

    return results
