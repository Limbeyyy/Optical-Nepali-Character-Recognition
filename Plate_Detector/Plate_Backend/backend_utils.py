import requests
import re

# Helper Functions
def extract_qr_id(qr_text):
    """
    Extracts the unique code from QR URL.
    Example: 'https://kataho.app/c/ABCD123QRP' -> 'ABCD123QRP'
    """
    match = re.search(r'/c/([A-Za-z0-9]+)', qr_text)
    return match.group(1) if match else None


def normalize_kid(kid_text: str):
    if not kid_text:
        return None
    return kid_text.replace("KID:", "").strip()

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return str(text).strip().lower()



def fields_match(api_val, ocr_val) -> bool:
    return normalize_text(api_val) == normalize_text(ocr_val)
