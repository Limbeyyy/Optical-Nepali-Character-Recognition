import cv2
import torch
import numpy as np
import os
import sys

# ---------------- PATH ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ocr.model import EnhancedBMCNNwHFCs
from config import IDX2CHAR, IMG_SIZE

# ---------------- DEBUG ----------------
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
    "Kataho_Address": (0.1031, 0.3169, 0.8938, 0.4804),
    "KID_No": (0.6281, 0.4787, 0.8888, 0.5321),
    "Plus_Code": (0.4530, 0.5688, 0.7656, 0.6555),
    "Local_Address": (0.2169, 0.1199, 0.7915, 0.2534),
    "Ward_Address": (0.2177, 0.6770, 0.7755, 0.7495),
    "Location": (0.4231, 0.7524, 0.5662, 0.8327)
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

def resize_and_center(img, size=32):
    """Resize image to fit into a square canvas while preserving aspect ratio."""
    h, w = img.shape
    if h == 0 or w == 0:
        # Empty image: return blank canvas
        return np.zeros((size, size), np.uint8)

    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size), np.uint8)
    y = (size - img.shape[0]) // 2
    x = (size - img.shape[1]) // 2
    canvas[y:y+img.shape[0], x:x+img.shape[1]] = img
    return canvas


# ---------------- CNN INFER ----------------
def infer_base(p):
    img = resize_and_center(p)
    img = (img / 255.0 - 0.5) * 2
    img = img[np.newaxis, np.newaxis, ...]
    x = torch.tensor(img, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, 1)
        conf, idx = torch.max(probs, 1)

    if conf.item() < 0.55:
        return ""

    return IDX2CHAR[idx.item()] if idx.item() < len(IDX2CHAR) else ""

# ---------------- CHARACTER RECOGNITION ----------------
from config import field_params
def recognize_word(gray_img, field="global", strict_char=False):
    """
    Unified OCR for different field types:
    KID, Plus Code, Kataho Address, Ward Address, City, Local Address.

    Parameters are dynamically set based on field:
        kid: threshold=0.77, min_gap=2
        plus_code: threshold=0.91, min_gap=2
        kataho_address: threshold=0.8, min_gap=2
        ward_address: threshold=0.65, min_gap=0.2
        city: threshold=0.65, min_gap=0.2
        local_address: threshold=0.82, min_gap=2
    """
    # ------------------- Set dynamic parameters -------------------
    params = field_params.get(field.lower(), field_params["global"])
    threshold_ratio = params["threshold_ratio"]
    min_gap_pixels = params["min_gap_pixels"]

    # ------------------- Preprocessing -------------------
    img = binarize(gray_img)
    img = crop_text(img)
    save_debug(img, "char_bin.png", field)

    # ------------------- Shirorekha removal for addresses/wards/city -------------------
    if field.lower() in ["Kataho_Address", "Ward_Address", "City", "Local_Address"]:
        row_sum = np.sum(img > 0, axis=1)
        shiro_threshold = 0.7 * row_sum.max()
        shiro_rows = np.where(row_sum > shiro_threshold)[0]
        if len(shiro_rows) > 0:
            img[shiro_rows, :] = 0
            save_debug(img, "char_no_shirorekha.png", field)

    char_images = []

    # ------------------- Character segmentation -------------------
    if strict_char:
        # Horizontal histogram segmentation
        col_sum = np.sum(img > 0, axis=0)
        threshold = threshold_ratio * col_sum.max()

        below_thresh = col_sum < threshold
        splits = []
        start = None
        for i, val in enumerate(below_thresh):
            if val:
                if start is None:
                    start = i
            else:
                if start is not None:
                    splits.append((start, i))
                    start = None
        if start is not None:
            splits.append((start, len(col_sum)))

        # Merge close splits
        merged_splits = []
        for s, e in splits:
            if not merged_splits:
                merged_splits.append((s, e))
            else:
                prev_s, prev_e = merged_splits[-1]
                if s - prev_e <= min_gap_pixels:
                    merged_splits[-1] = (prev_s, e)
                else:
                    merged_splits.append((s, e))

        # Extract characters
        for i, (s, e) in enumerate(merged_splits):
            char_img = img[:, s:e]
            if np.count_nonzero(char_img) < 10:
                continue
            char_images.append(char_img)
            save_debug(char_img, f"char_{i}.png", field)

        # Visualize histogram
        hist_vis = np.zeros((100, img.shape[1]), np.uint8)
        col_vis = (col_sum / col_sum.max() * 100).astype(np.int32)
        for x, h in enumerate(col_vis):
            cv2.line(hist_vis, (x, 100), (x, 100 - h), 255, 1)
        thresh_h = int(threshold / col_sum.max() * 100)
        cv2.line(hist_vis, (0, 100 - thresh_h), (img.shape[1]-1, 100 - thresh_h), 128, 1)
        for s, e in merged_splits:
            cv2.line(hist_vis, (s, 0), (s, 100), 0, 1)
            cv2.line(hist_vis, (e, 0), (e, 100), 0, 1)
        save_debug(hist_vis, "hist_threshold_splits.png", field)

    else:
        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20 or w < 3 or h < 3:
                continue
            char_img = img[y:y+h, x:x+w]
            components.append((x, char_img))
            save_debug(char_img, f"char_{i}.png", field)
        char_images = [img for x, img in sorted(components, key=lambda t: t[0])]

    # ------------------- CNN inference -------------------
    result = ""
    for char_img in char_images:
        result += infer_base(char_img)

    # ------------------- Visualization bounding boxes -------------------
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, char_img in enumerate(char_images):
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cv2.rectangle(vis, (x, 0), (x+w, img.shape[0]), (0, 255, 0), 1)
    save_debug(vis, "char_boxes.png", field)

    return result


# ---------------- DYNAMIC OCR ----------------
def recognize_text_dynamic(gray_img, field):
    strict_fields = ["KID_No", "Plus_Code", "Local_Address"]  # fields to split each character strictly
    strict_char = field in strict_fields
    return recognize_word(gray_img, field=field, strict_char=strict_char)

# ---------------- ROI OCR ----------------
def recognize_from_rois(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    results = {}

    for field, (x1n, y1n, x2n, y2n) in ROIS.items():
        x1, y1 = int(x1n*w), int(y1n*h)
        x2, y2 = int(x2n*w), int(y2n*h)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            results[field] = ""
            continue

        save_debug(roi, "00_roi_color.png", field)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        save_debug(gray, "00_roi_gray.png", field)

        results[field] = recognize_text_dynamic(gray, field)

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
