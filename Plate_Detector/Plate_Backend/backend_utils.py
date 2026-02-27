from rapidfuzz import fuzz
from typing import Optional
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


def word_level_similarity(
    ocr_text: str,
    api_text: str,
    word_threshold: float = 80.0,   # per-word minimum similarity %
    match_mode: str = "all"          # "all" = every word must match, "ratio" = avg must exceed threshold
) -> tuple[bool, float]:
    """
    Word-by-word fuzzy similarity between OCR and API strings.
    
    Args:
        ocr_text:        Raw OCR output
        api_text:        API ground truth
        word_threshold:  Minimum similarity % per word (0–100)
        match_mode:      "all"   → every word pair must exceed threshold
                         "ratio" → average word similarity must exceed threshold
    
    Returns:
        (is_match: bool, avg_score: float)
    """
    if not ocr_text or not api_text:
        return False, 0.0

    ocr_words = ocr_text.strip().split()
    api_words = api_text.strip().split()

    # If word counts differ, align by position up to the shorter length
    # and penalize missing words
    min_len = min(len(ocr_words), len(api_words))
    max_len = max(len(ocr_words), len(api_words))

    scores = []
    for i in range(min_len):
        score = fuzz.ratio(ocr_words[i], api_words[i])
        scores.append(score)

    # Penalize length mismatch: missing words score 0
    for _ in range(max_len - min_len):
        scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if match_mode == "all":
        # Every word must individually pass the threshold
        per_word_pass = all(s >= word_threshold for s in scores)
        return per_word_pass, avg_score

    elif match_mode == "ratio":
        # Average of all word scores must pass the threshold
        return avg_score >= word_threshold, avg_score

    return False, avg_score


def fields_match(
    api_val: Optional[str],
    ocr_val: Optional[str],
    word_threshold: float = 80.0,
    match_mode: str = "all"
) -> bool:
    """
    Drop-in replacement for your existing fields_match.
    Uses word-level fuzzy similarity.
    """
    if not api_val or not ocr_val:
        return False

    is_match, avg_score = word_level_similarity(
        ocr_text=str(ocr_val),
        api_text=str(api_val),
        word_threshold=word_threshold,
        match_mode=match_mode
    )

    # Optional: uncomment to debug
    # print(f"  OCR: {ocr_val}")
    # print(f"  API: {api_val}")
    # print(f"  Scores → match={is_match}, avg={avg_score:.1f}")

    return is_match