import cv2
import torch
import numpy as np
from main import EnhancedBMCNNwHFCs
from config import IDX2CHAR, IMG_SIZE

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/kataho/Downloads/mallanet_ocr/models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
model = EnhancedBMCNNwHFCs(num_classes=58)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

# ---------------- MATRA MAP ----------------
UPPER_MATRAS = {
    "small": "ि",
    "long": "ी",
    "ai": "ै",
    "anusvara": "ं",
    "chandra": "ँ"
}

LOWER_MATRAS = {
    "u": "ु",
    "uu": "ू",
    "r": "ृ"
}




# ---------------- BASIC UTILS ----------------

def binarize(img):
    _, th = cv2.threshold(img, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return th




def crop_text(img):
    rows = np.sum(img > 0, axis=1)
    cols = np.sum(img > 0, axis=0)
    t = np.argmax(rows > 0)
    b = len(rows) - np.argmax(rows[::-1] > 0)
    l = np.argmax(cols > 0)
    r = len(cols) - np.argmax(cols[::-1] > 0)
    return img[t:b, l:r]

def is_valid_component(p):
    if p is None:
        return False
    if np.count_nonzero(p) < 20:   # VERY IMPORTANT
        return False
    h, w = p.shape
    if h < 3 or w < 3:
        return False
    return True


def resize_and_center(img, size=32):
    h, w = img.shape

    if h == 0 or w == 0:
        return None

    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
    return canvas


# ---------------- HEADER LINE ----------------

def detect_header(img):
    upper = int(0.55 * img.shape[0])
    return np.argmax(np.sum(img[:upper] > 0, axis=1))

def remove_header(img, row):
    img[row:row+2, :] = 0
    return img



def is_valid_column(col, min_pixels=40, min_width=4):
    if col is None:
        return False

    h, w = col.shape

    # Too narrow → pollution
    if w < min_width:
        return False

    # Too few pixels → noise
    if np.count_nonzero(col) < min_pixels:
        return False

    return True


def is_matra_only(col, header):
    rows = np.where(np.sum(col > 0, axis=1) > 0)[0]
    if len(rows) == 0:
        return True

    top, bottom = rows[0], rows[-1]

    # Entire content above or below base zone
    if bottom < header - 2:
        return True
    if top > header + 6:
        return True

    return False


# ---------------- SEGMENT WORD ----------------

def segment_columns(img):
    col_sum = np.sum(img > 0, axis=0)
    thresh = 0.03 * col_sum.max()
    splits, s = [], None
    cols = []

    for i, v in enumerate(col_sum):
        if v > thresh and s is None:
            s = i
        elif v <= thresh and s is not None:
            splits.append((s, i))
            s = None
    if s:
        splits.append((s, len(col_sum)))
        

    for a, b in splits:
        col = img[:, a:b]
        if is_valid_column(col):
            cols.append(col)
    return cols


def is_shirorekha_only(col):
    h, w = col.shape
    upper = int(0.55 * h)
    row_sum = np.sum(col[:upper] > 0, axis=1)

    if row_sum.max() > 0.8 * w and np.count_nonzero(col) < 60:
        return True
    return False



# ---------------- MATRA DETECTION ----------------

def classify_upper_matra(img, header):
    """
    Rule-based upper matra detection.
    """
    h, w = img.shape
    rows = np.where(np.sum(img > 0, axis=1) > 0)[0]

    if len(rows) == 0:
        return ""

    top = rows[0]

    # Must lie ABOVE header
    if top > header - 3:
        return ""

    height = rows[-1] - rows[0]

    if height <= 3:
        return "ि"   # short i
    elif height <= 6:
        return "ी"   # long i
    else:
        return "ै"   # ai (fallback)


def classify_lower_matra(img, header):
    rows = np.where(np.sum(img > 0, axis=1) > 0)[0]
    if len(rows) == 0:
        return ""

    bottom = rows[-1]

    if bottom < header + 4:
        return ""

    height = rows[-1] - rows[0]

    if height <= 4:
        return "ु"
    elif height <= 7:
        return "ू"
    else:
        return "ृ"


# ---------------- SHIROREKHA CHECK ----------------

def has_shirorekha(img):
    """
    Return True if a horizontal shirorekha (headline) exists.
    """
    upper = int(0.55 * img.shape[0])
    row_sums = np.sum(img[:upper] > 0, axis=1)
    max_val = row_sums.max()
    # Line must cover at least 50% of width
    return max_val > 0.5 * img.shape[1]

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

# ---------------- CNN INFERENCE ----------------

def infer_base(p):
    img = resize_and_center(p)
    img = img.astype(np.float32)
    # Try normalization like training: [-1, 1]
    img = (img / 255.0 - 0.5) * 2
    img = np.expand_dims(img, 0)  # batch
    img = np.expand_dims(img, 0)  # channel
    x = torch.tensor(img).to(DEVICE)

    with torch.no_grad():
        preds = model(x)
        print("Raw model output:", preds)
        probs = torch.softmax(preds, dim=1)
        conf, idx = torch.max(probs, dim=1)
        conf = conf.item()
        idx = idx.item()
        print("Predicted index:", idx, "Confidence:", conf)
        print("IDX2CHAR length:", len(IDX2CHAR))

    if idx >= len(IDX2CHAR):
        print("Index out of range!")
        return "?"
    return IDX2CHAR[idx]


# ---------------- UPDATED COMPONENT LOOP ----------------

def recognize_word(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image")

    img = binarize(img)
    img = crop_text(img)

    header = detect_header(img)
    img = remove_header(img, header)

    shirorekha_exists = has_shirorekha(img)

    columns = segment_columns(img)
    print("Detected columns:", len(columns))

    # ---------------- CLEAN COLUMNS ONCE ----------------
    clean_columns = []
    for col in columns:
        if not is_valid_column(col):
            continue
        if is_matra_only(col, header):
            continue
        if is_shirorekha_only(col):
            continue
        clean_columns.append(col)

    columns = clean_columns
    print("Clean columns:", len(columns))

    # ---------------- RESULT INIT (FIXED) ----------------
    result = ""
    pending_matra = ""

    for i, col in enumerate(columns):
        print(f"Processing column {i}, pixels:", np.count_nonzero(col))
        cv2.imshow(f"Column {i}", col)
        cv2.waitKey(0)

        parts = split_half_form(col, header)

        for p in parts:
            if not is_valid_component(p):
                continue

            rows = np.where(np.sum(p > 0, axis=1) > 0)[0]
            if len(rows) == 0:
                continue

            top, bottom = rows[0], rows[-1]

            # Upper matra
            if shirorekha_exists and top < header - 2:
                pending_matra = classify_upper_matra(p, header)
                continue

            # Lower matra
            if shirorekha_exists and bottom > header + int(0.35 * img.shape[0]):
                pending_matra = classify_lower_matra(p, header)
                continue

            # Base character
            base = infer_base(p)
            if base:
                result += base
                if pending_matra:
                    result += pending_matra
                    pending_matra = ""

    return result


# ---------------- MAIN ----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    # Use args.image
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image path!")

    cv2.imshow("Original", img)
    cv2.waitKey(0)

    img_bin = binarize(img)
    cv2.imshow("Binarized", img_bin)
    cv2.waitKey(0)

    img_crop = crop_text(img_bin)
    cv2.imshow("Cropped", img_crop)
    cv2.waitKey(0)


    print("Recognized Word:", recognize_word(args.image))
