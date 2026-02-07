# utils.py
import cv2
import os
import re

# ---------------- POST-PROCESSING ----------------
import re

def preprocess_number(text: str) -> str:
    if not text:
        return ""

    # Fix common OCR misreads BEFORE trimming
    text = re.sub(r'[OoQ]', '0', text)

    #  Remove everything before the first digit
    text = re.sub(r'^[^\d]+', '', text)

    #  Final cleanup
    return text.strip()



def normalize_wada_number(text):
    """
    Replaces any word between 'वडा' and a Devanagari number with 'नं'.
    Example: 'वडार्ने २६' -> 'वडा नं २६'
    """
    devanagari_digits = r"[०१२३४५६७८९]+"
    # Match 'वडा', optional whitespace, any Devanagari chars except digits, optional whitespace, then number
    pattern = re.compile(r"(वडा)\s*[\u0900-\u097F]*\s*(" + devanagari_digits + r")")
    # Replace with 'वडा नं <number>'
    result = pattern.sub(r"\1 नं \2", text)
    return result

DEBUG_DIR = "debug_vis"

def save_debug(img, name, field="global"):
    folder = os.path.join(DEBUG_DIR, field)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), img)

def crop_roi(img, roi):
    h, w = img.shape[:2]
    x1n, y1n, x2n, y2n = roi
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    cropped = img[y1:y2, x1:x2]
    return cropped
