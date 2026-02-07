import cv2
import re
import numpy as np

def straighten_plate(plate_img):
    """
    Detects plate contour and applies perspective transform
    Returns flattened (straightened) plate image
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return plate_img  # fallback

    # largest contour = plate
    cnt = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) != 4:
        return plate_img  # can't rectify safely

    pts = approx.reshape(4, 2)

    # order points: TL, TR, BR, BL
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(plate_img, M, (maxWidth, maxHeight))

    return warped

def draw_detections(img, detections, rois=None):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if rois:
        draw_rois(img, rois, color=(0, 255, 0))

    return img


def draw_plate_rois(frame, plate_box, rois, color=(255, 0, 0), thickness=2):
    """
    Draw ROI boxes (normalized to plate) onto full frame
    """
    px1, py1, px2, py2, _ = plate_box
    plate_w = px2 - px1
    plate_h = py2 - py1

    for field, coords in rois.items():
        # coords = [x1f, y1f, x2f, y2f] normalized
        x1f, y1f, x2f, y2f = coords

        # Convert to plate pixels
        rx1 = int(px1 + x1f * plate_w)
        ry1 = int(py1 + y1f * plate_h)
        rx2 = int(px1 + x2f * plate_w)
        ry2 = int(py1 + y2f * plate_h)

        # Safety clamp
        rx1, ry1 = max(rx1, 0), max(ry1, 0)
        rx2, ry2 = min(rx2, frame.shape[1]), min(ry2, frame.shape[0])

        if rx2 <= rx1 or ry2 <= ry1:
            continue

        # Draw ROI box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, thickness)

        # Label
        cv2.putText(
            frame,
            field,
            (rx1, ry1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )


def draw_rois(img, rois, color=(0, 255, 0), thickness=2):
    h_img, w_img = img.shape[:2]

    for name, roi in rois.items():

        # --- normalize ROI format ---
        if isinstance(roi, dict):
            x, y, w, h = roi.get("x"), roi.get("y"), roi.get("w"), roi.get("h")
        elif isinstance(roi, (list, tuple)) and len(roi) == 4:
            x, y, w, h = roi
        else:
            print(f"[WARN] Invalid ROI format for {name}: {roi}")
            continue

        # --- force int & clamp ---
        try:
            x = int(round(float(x)))
            y = int(round(float(y)))
            w = int(round(float(w)))
            h = int(round(float(h)))
        except Exception:
            print(f"[WARN] Non-numeric ROI values for {name}: {roi}")
            continue

        # clamp to image bounds
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        # --- draw ---
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )

        cv2.putText(
            img,
            name,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return img

import cv2
import numpy as np

def normalize_illumination(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def gamma_correction(img, gamma=1.5):
    inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


def lighting_robust_preprocess(img):
    img = normalize_illumination(img)
    img = gamma_correction(img, gamma=1.5)
    return img


def estimate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def adaptive_lighting(img):
    brightness = estimate_brightness(img)

    img = normalize_illumination(img)

    if brightness < 60:
        img = gamma_correction(img, 1.8)
    elif brightness < 100:
        img = gamma_correction(img, 1.4)
    elif brightness > 180:
        img = gamma_correction(img, 0.7)

    return img


    import re

def clean_ocr_text(field, text):
    """
    Refine OCR output for known pollution cases.
    
    Parameters:
        field: str - the field name (e.g., 'KID_No', 'Plus_Code', 'City')
        text: str - raw OCR output
    Returns:
        str - cleaned OCR text
    """
    if not text:
        return ""

    text = text.strip()  # remove leading/trailing whitespace

    # ----------------- Remove known prefixes -----------------
    prefixes = {
        "KID_No": ["KID:", "KID "],
        "Plus_Code": ["Plus code :", "Plus code:", "Plus Code:", "Plus code "],
        "City": [],
        "Ward_Address": [],
        "Local_Address": [],
        "Kataho_Address": [],
        "QR_Code": []
    }

    if field in prefixes:
        for p in prefixes[field]:
            if text.startswith(p):
                text = text[len(p):].strip()

    # ----------------- Remove timestamps / time-like patterns -----------------
    # Example: "[21:12 (210 ..." or "21:12 ..." etc.
    text = re.sub(r'\[?\d{1,2}:\d{2}.*?\]?', '', text)

    # ----------------- Remove trailing unwanted symbols -----------------
    # Keep only letters, numbers, Nepali chars, dashes, plus sign
    if field in ["KID_No", "Plus_Code"]:
        text = re.sub(r'[^A-Za-z0-9\-\+]', '', text)
    elif field in ["Local_Address", "Kataho_Address", "Ward_Address", "City"]:
        # Keep Nepali letters, digits, spaces, commas, periods
        text = re.sub(r'[^\u0900-\u097F0-9\s,.\-]', '', text)
    elif field == "QR_Code":
        # URLs: keep alphanumerics and URL symbols
        text = re.sub(r'[^\w:/\-\.\?\&\=]+', '', text)

    # ----------------- Normalize whitespace -----------------
    text = re.sub(r'\s+', ' ', text).strip()

    return text


import re

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


import re

def clean_devanagari_noise(text):
    """
    Removes stray 1–2 character Devanagari words
    from start/end or anywhere in text.
    Keeps only meaningful words (length >= 3).
    """
    if not text:
        return ""

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split(" ")

    cleaned_words = []
    for w in words:
        # Keep only words that are >= 3 chars OR non-Devanagari
        if re.fullmatch(r'[\u0900-\u097F]{1,2}', w):
            continue
        cleaned_words.append(w)

    return " ".join(cleaned_words)

import re

def clean_ward_address(text):
    """
    Cleans stray Devanagari characters from text,
    keeps meaningful words (>= 3 chars) and numbers at the end,
    and adds a comma before the ward number if present.
    
    Example:
        "काठमाडौंं महानगरपालिकाा वडा नंं २६" 
        → "काठमाडौं महानगरपालिका, वडा नं २६"
    """
    if not text:
        return ""

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Split words
    words = text.split(" ")

    cleaned_words = []
    last_number = None

    for w in words:
        # Check if it is a number (Devanagari or Arabic digits)
        if re.fullmatch(r'[\u0966-\u096F0-9]+', w):
            last_number = w
            continue

        # Ignore stray 1-2 char Devanagari words
        if re.fullmatch(r'[\u0900-\u097F]{1,2}', w):
            continue

        cleaned_words.append(w)

    # Add comma before ward number if it exists
    result = " ".join(cleaned_words)
    if last_number:
        result = f"{result}, {last_number}"

    return result



# ---------------- MAIN DISPATCH ----------------
def post_process_ocr(field, text):
    if not text:
        return ""

    if field == "Local_Address":
        return clean_local_address(text)

    if field == "Kataho_Address":
        return clean_kataho_address(text)

    if field == "Plus_Code":
        return clean_plus_code(text)

    # if field == "Ward_Address":
    #     return clean_ward_address(text)

    if field == "City":
        texts = clean_devanagari_noise(text)
        return clean_city(text)

    if field == "KID_No":
        return clean_kid_no(text)

    return text.strip()


import re

def normalize_nepali_ocr(text: str) -> str:
    if not text:
        return text

    # -------------------------------
    # 1️⃣ BASIC UNICODE CLEANUP
    # -------------------------------
    # Remove duplicated matras like ीं, ौीं, etc.
    text = re.sub(r'[ँं]+', 'ं', text)       # multiple anusvara → single
    text = re.sub(r'([ािीुूेैोौ])\1+', r'\1', text)  # repeated matras
    text = re.sub(r'\s+', ' ', text).strip()

    # -------------------------------
    # 2️⃣ COMMON OCR PATTERN FIXES
    # -------------------------------
    ocr_pattern_fixes = {
        "काठमाडौीं": "काठमाडौं",
        "काठमाडौ": "काठमाडौं",
        "काठमाण्डौ": "काठमाडौं",
        "काठमाण्डौं": "काठमाडौं",
        "महानगरपालीका": "महानगरपालिका",
        "महानगरपालिक": "महानगरपालिका",
        "वडा न": "वडा नं",
        "वडा न.": "वडा नं",
        "वडा नं.": "वडा नं",
    }

    for wrong, correct in ocr_pattern_fixes.items():
        text = text.replace(wrong, correct)

    # -------------------------------
    # 3️⃣ SMART WORD-LEVEL CLEANUP
    # -------------------------------
    words = text.split()
    cleaned_words = []

    for w in words:
        # Remove single or double junk characters
        if len(w) <= 2:
            continue
        cleaned_words.append(w)

    text = " ".join(cleaned_words)

    return text
