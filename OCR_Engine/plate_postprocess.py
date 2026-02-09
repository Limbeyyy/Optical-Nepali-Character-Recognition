# ---------------- POST-PROCESSING ----------------
import re
import cv2
import os

# ---------------- DEVANAGARI RANGES ----------------
DEV = r"\u0900-\u097F"
DEV_NUM = r"\u0966-\u096F"

# ---------------- HELPERS ----------------
def only_devanagari(text):
    return re.sub(fr"[^{DEV}\s]", "", text).strip()

def only_devanagari_words(text):
    text = re.sub(fr"[^{DEV}\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()

# ---------------- FIELD CLEANERS ----------------
def clean_local_address(text):
    text = only_devanagari_words(text)
    # Ensure starts & ends with Devanagari
    text = re.sub(fr"^[^{DEV}]+", "", text)
    text = re.sub(fr"[^{DEV}]+$", "", text)
    return text

def clean_kataho_address(text):
    text = re.sub(fr"[^{DEV}\s{DEV_NUM}]", "", text)
    text = normalize_spaces(text)

    # Match: 2 digit + word + word + 4 digit
    m = re.search(
        fr"([{DEV_NUM}]{{2}})\s+([{DEV}]+)\s+([{DEV}]+)\s+([{DEV_NUM}]{{4}})",
        text
    )
    return " ".join(m.groups()) if m else ""

def clean_plus_code(text):
    text = text.upper()
    text = text.replace("/", "7")
    text = re.sub(r"PLUS\s*CODE\s*[:\-]?", "", text, flags=re.I)
    text = re.sub(r"[^A-Z0-9\+]", "", text)

    # Google Plus Code pattern
    m = re.search(r"[2-9CFGHJMPQRVWX]{6,8}\+[2-9CFGHJMPQRVWX]{2,3}", text)
    return m.group(0) if m else ""

def clean_ward_address(text):
    text = re.sub(fr"[^{DEV}{DEV_NUM}\s,]", "", text)
    text = normalize_spaces(text)

    m = re.search(
        fr"([{DEV}\s]+),?\s*वडा\s*नें?\s*([{DEV_NUM}]{{1,2}})",
        text
    )
    return f"{m.group(1)}, वडा नें {m.group(2)}" if m else ""

def clean_city(text):
    return only_devanagari_words(text)

def clean_kid_no(text):
    text = re.sub(r"KID\s*[:\-]?", "", text, flags=re.I)
    text = re.sub(r"[^0-9\-]", "", text)

    m = re.search(
        r"(\d{2,3})-(\d{3,4})-(\d{3,4})-(\d{4})",
        text
    )
    return "-".join(m.groups()) if m else ""


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

import re

def clean_devanagari_noise(text: str) -> str:
    """
    Removes stray 1–2 character Devanagari words at the start/end of text.
    Keeps meaningful words (length >= 3) and all numeric words.
    """
    if not text:
        return ""

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split(" ")

    # Remove stray 1-2 char words only at start and end
    while words and re.fullmatch(r'[\u0900-\u097F]{1,2}', words[0]):
        words.pop(0)
    while words and re.fullmatch(r'[\u0900-\u097F]{1,2}', words[-1]):
        words.pop()

    return " ".join(words)

import re

def clean_ward_address(text: str) -> str:
    """
    Cleans stray Devanagari characters from text,
    keeps meaningful words (>= 3 chars) and preserves 'वडा नं <number>'.
    Adds a comma before 'वडा नं' if present.

    Example:
        "काठमाडौंं महानगरपालिकाा वडा नंं २६" 
        → "काठमाडौं महानगरपालिका, वडा नं 26"
    """
    if not text:
        return ""

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Match the ward pattern at the end
    match = re.search(r'(वडा\s+न[ं]?\s*\d+)$', text)
    if match:
        ward_part = match.group(1)
        main_part = text[:match.start()].strip()
        # Add comma between main part and ward part
        return f"{main_part}, {ward_part}"
    else:
        # If no ward pattern, just return cleaned text
        return text


import re

def normalize_nepali_ocr(text: str) -> str:
    """
    Post-process Nepali OCR text to fix common character errors.
    
    Examples:
        मागे → मार्ग
        काठमाडौीं → काठमाडौं
        वडा न → वडा नं
    """

    if not text:
        return text

    # -----------------------------------------
    # 1️⃣ Exact Word-Level Corrections
    # -----------------------------------------
    word_replacements = {
        "मागे": "मार्ग",
        "माग": "मार्ग",
        "काठमाडौीं": "काठमाडौं",
        "काठमाडौी": "काठमाडौं",
        "वडा न": "वडा नं",
        "वडान": "वडा नं",
    }

    for wrong, correct in word_replacements.items():
        text = text.replace(wrong, correct)

    # -----------------------------------------
    # 2️⃣ Regex-Based Structural Fixes
    # -----------------------------------------

    # Fix missing र् before ग (common OCR issue)
    text = re.sub(r'मा्ग', 'मार्ग', text)
    text = re.sub(r'माग([^\u0900-\u097F]|$)', r'मार्ग\1', text)

    # Normalize ward number variations
    text = re.sub(r'वडा\s*न\.?', 'वडा नं', text)

    # Fix duplicate matra issues
    text = re.sub(r'ीं+', 'ं', text)
    text = re.sub(r'ौीं', 'ौं', text)

    # Remove repeated characters (OCR noise)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# ---------------- MAIN DISPATCH ----------------
def post_process_ocr(field, text):
    if not text:
        return ""

    if field == "Local_Address":
        texts = clean_local_address(text)
        return normalize_nepali_ocr(texts)

    if field == "Kataho_Address":
        return clean_kataho_address(text)

    if field == "Plus_Code":
        return clean_plus_code(text)

    if field == "Ward_Address":
        # # text = clean_devanagari_noise(text)
        # text = normalize_nepali_ocr(text)
        text = clean_ward_address(text)
        # return normalize_wada_number(text)

    if field == "City":
        texts = clean_devanagari_noise(text)
        return clean_city(text)

    if field == "KID_No":
        return clean_kid_no(text)

    return text.strip()

