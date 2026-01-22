import cv2
import torch
import numpy as np

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from ocr.model import EnhancedBMCNNwHFCs
from config import IDX2CHAR, IMG_SIZE


import os

DEBUG_DIR = "debug_vis"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug(img, name, field):
    folder = os.path.join(DEBUG_DIR, field)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), img)


# ---------------- CONFIG ----------------
MODEL_PATH = "/home/kataho/Downloads/mallanet_ocr/models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROIS = {
    "kataho_address": (0.1031, 0.3169, 0.8938, 0.4804),
    "KID_No": (0.6281, 0.4787, 0.8888, 0.5321),
    "Plus_Code": (0.4530, 0.5688, 0.7656, 0.6555),
    "Address_Name": (0.2106, 0.6656, 0.7650, 0.8540),
}

# ---------------- MODEL ----------------
model = EnhancedBMCNNwHFCs(num_classes=58)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

# ---------------- BASIC UTILS ----------------

def binarize(img):
    _, th = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return th


def crop_text(img):
    coords = cv2.findNonZero(img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]



def is_valid_component(p):
    if p is None:
        return False
    if np.count_nonzero(p) < 20:
        return False
    h, w = p.shape
    return h >= 3 and w >= 3


def resize_and_center(img, size=32):
    h, w = img.shape
    if h == 0 or w == 0:
        return None

    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = img
    return canvas


# ---------------- HEADER ----------------

def detect_header(img):
    upper = int(0.55 * img.shape[0])
    return np.argmax(np.sum(img[:upper] > 0, axis=1))


def remove_header(img, row):
    img[row:row+2, :] = 0
    return img


def has_shirorekha(img):
    upper = int(0.55 * img.shape[0])
    row_sums = np.sum(img[:upper] > 0, axis=1)
    return row_sums.max() > 0.5 * img.shape[1]


# ---------------- COLUMN SEGMENT ----------------

def is_valid_column(col, min_pixels=40, min_width=4):
    if col is None:
        return False
    h, w = col.shape
    return w >= min_width and np.count_nonzero(col) >= min_pixels


def is_matra_only(col, header):
    rows = np.where(np.sum(col > 0, axis=1) > 0)[0]
    if len(rows) == 0:
        return True
    return rows[-1] < header - 2 or rows[0] > header + 6


def is_shirorekha_only(col):
    h, w = col.shape
    upper = int(0.55 * h)
    row_sum = np.sum(col[:upper] > 0, axis=1)
    return row_sum.max() > 0.8 * w and np.count_nonzero(col) < 60


def segment_columns(img):
    col_sum = np.sum(img > 0, axis=0)
    thresh = 0.03 * col_sum.max()
    splits, s = [], None

    for i, v in enumerate(col_sum):
        if v > thresh and s is None:
            s = i
        elif v <= thresh and s is not None:
            splits.append((s, i))
            s = None
    if s:
        splits.append((s, len(col_sum)))

    cols = []
    for a, b in splits:
        col = img[:, a:b]
        if is_valid_column(col):
            cols.append(col)
    return cols


# ---------------- MATRA ----------------

def classify_upper_matra(img, header):
    rows = np.where(np.sum(img > 0, axis=1) > 0)[0]
    if len(rows) == 0 or rows[0] > header - 3:
        return ""
    h = rows[-1] - rows[0]
    return "ि" if h <= 3 else "ी" if h <= 6 else "ै"


def classify_lower_matra(img, header):
    rows = np.where(np.sum(img > 0, axis=1) > 0)[0]
    if len(rows) == 0 or rows[-1] < header + 4:
        return ""
    h = rows[-1] - rows[0]
    return "ु" if h <= 4 else "ू" if h <= 7 else "ृ"


# ---------------- HALF FORM ----------------

def split_half_form(img, header):
    h, w = img.shape
    if w <= h or w < 6:
        return [img]

    cols = np.sum(img[header+1:] > 0, axis=0)
    cut = int(np.argmax(cols))

    parts = []
    if is_valid_component(img[:, :cut]):
        parts.append(img[:, :cut])
    if is_valid_component(img[:, cut:]):
        parts.append(img[:, cut:])
    return parts if parts else [img]


# ---------------- CNN ----------------

def infer_base(p):
    img = resize_and_center(p)
    img = (img / 255.0 - 0.5) * 2
    img = img[np.newaxis, np.newaxis, ...]
    x = torch.tensor(img, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        idx = torch.argmax(pred, 1).item()

    return IDX2CHAR[idx] if idx < len(IDX2CHAR) else "?"


# ---------------- OCR CORE ----------------

def recognize_word(image_or_path):
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image path")
    else:
        img = image_or_path.copy()

    img = binarize(img)
    img = crop_text(img)

    header = detect_header(img)
    img = remove_header(img, header)
    shirorekha_exists = has_shirorekha(img)

    columns = segment_columns(img)

    result = ""
    pending_matra = ""

    for col in columns:
        if is_matra_only(col, header) or is_shirorekha_only(col):
            continue

        parts = split_half_form(col, header)

        for p in parts:
            if not is_valid_component(p):
                continue

            rows = np.where(np.sum(p > 0, axis=1) > 0)[0]
            top, bottom = rows[0], rows[-1]

            if shirorekha_exists and top < header - 2:
                pending_matra = classify_upper_matra(p, header)
                continue

            if shirorekha_exists and bottom > header + int(0.35 * img.shape[0]):
                pending_matra = classify_lower_matra(p, header)
                continue

            base = infer_base(p)
            result += base
            if pending_matra:
                result += pending_matra
                pending_matra = ""

    return result


# ---------------- ROI OCR ----------------

def norm_to_pixel_roi(img, roi):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def recognize_from_rois(image_path):
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Invalid image path")

    h, w = original.shape[:2]
    results = {}

    for field, (rx1, ry1, rx2, ry2) in ROIS.items():
        # ---- normalized → pixel ----
        x1 = int(rx1 * w)
        y1 = int(ry1 * h)
        x2 = int(rx2 * w)
        y2 = int(ry2 * h)

        roi_color = original[y1:y2, x1:x2]

        if roi_color.size == 0:
            results[field] = ""
            continue

        # ---- visualize HARD ROI ----
        save_debug(roi_color, "00_roi_color.png", field)

        # ---- grayscale ONLY ROI ----
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        save_debug(roi_gray, "01_roi_gray.png", field)

        # ---- OCR ONLY ON ROI ----
        text = recognize_word(roi_gray)
        results[field] = text

    return results



# ---------------- MAIN ----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    results = recognize_from_rois(args.image)

    print("\n===== OCR RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v}")
