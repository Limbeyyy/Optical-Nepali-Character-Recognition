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
    return p is not None and np.count_nonzero(p) >= 20 and p.shape[0] >= 3 and p.shape[1] >= 3


def is_real_base(p, header):
    """Return True if component is likely a base consonant, not a matra."""
    rows = np.where(np.sum(p > 0, axis=1) > 0)[0]
    if len(rows) == 0:
        return False
    top, bottom = rows[0], rows[-1]
    # Heuristic: base should be around header ± small margin
    return top <= header + 4 and bottom >= header - 4


def resize_and_center(img, size=32):
    h, w = img.shape
    # Skip empty or zero-size images
    if h == 0 or w == 0:
        return None
    scale = size / max(h, w)
    if int(w*scale) == 0 or int(h*scale) == 0:
        return None
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y, x = (size-img.shape[0])//2, (size-img.shape[1])//2
    canvas[y:y+img.shape[0], x:x+img.shape[1]] = img
    return canvas


# ---------------- HEADER ----------------

def detect_header(img):
    upper = int(0.55 * img.shape[0])
    return np.argmax(np.sum(img[:upper] > 0, axis=1))

def remove_header(img, row):
    img[row:row+2, :] = 0
    return img

# ---------------- COLUMN FILTERS ----------------

def is_valid_column(col, min_pixels=10, min_width=2):
    return col is not None and col.shape[1] >= min_width and np.count_nonzero(col) >= min_pixels
    

def is_matra_only(col, header):
    rows = np.where(np.sum(col > 0, axis=1) > 0)[0]
    if len(rows) == 0:
        return False
    top, bottom = rows[0], rows[-1]
    return bottom < header - 2 or top > header + 6

def is_shirorekha_only(col):
    h, w = col.shape
    upper = int(0.55 * h)
    row_sum = np.sum(col[:upper] > 0, axis=1)
    return row_sum.max() > 0.8 * w and np.count_nonzero(col) < 60

# ---------------- SEGMENT WORD ----------------
from scipy.signal import find_peaks
import numpy as np
import cv2

def segment_columns(img, min_abs_pixels=3, min_width=2, valley_prominence=0.1, max_merge_gap=2):
    """
    Robust column segmentation for both thick and thin characters using valley detection,
    with improved merging to avoid splitting letters.

    Args:
        img: binary image (text=white, background=black)
        min_abs_pixels: minimum absolute pixels to avoid noise
        min_width: minimum width of a column to be valid
        valley_prominence: fraction of max column sum to detect valleys
        max_merge_gap: maximum gap (in pixels) to merge nearby narrow columns

    Returns:
        List of tuples: (column_img, start_x, end_x)
    """
    col_sum = np.sum(img > 0, axis=0)

    # Smooth with moving average
    kernel_size = max(2, img.shape[1] // 100)
    col_sum_smooth = np.convolve(col_sum, np.ones(kernel_size)/kernel_size, mode='same')

    # Invert column sum to find valleys
    inverted = col_sum_smooth.max() - col_sum_smooth
    prominence = valley_prominence * col_sum_smooth.max()
    valleys, _ = find_peaks(inverted, prominence=prominence, distance=2)

    # Define column boundaries using valleys
    boundaries = [0] + list(valleys) + [img.shape[1]]
    cols = []
    for i in range(len(boundaries)-1):
        start, end = boundaries[i], boundaries[i+1]
        col = img[:, start:end]
        if col.shape[1] >= 1 and np.count_nonzero(col) >= min_abs_pixels:
            cols.append((col, start, end))

    # ---------------- Merge very narrow columns / small gaps ----------------
    if not cols:
        return []

    merged_cols = [cols[0]]
    for c in cols[1:]:
        prev_col, prev_start, prev_end = merged_cols[-1]
        curr_col, curr_start, curr_end = c
        gap = curr_start - prev_end

        # Merge if gap is very small or previous column is narrow
        if gap <= max_merge_gap or prev_col.shape[1] < min_width:
            merged_img = np.zeros((img.shape[0], prev_col.shape[1] + gap + curr_col.shape[1]), dtype=np.uint8)
            merged_img[:, :prev_col.shape[1]] = prev_col
            merged_img[:, prev_col.shape[1]+gap:] = curr_col
            merged_cols[-1] = (merged_img, prev_start, curr_end)
        else:
            merged_cols.append(c)

    # Final filtering: remove empty or zero-width columns
    final_cols = []
    for col, a, b in merged_cols:
        if col.shape[1] > 0 and np.count_nonzero(col) > 0:
            final_cols.append((col, a, b))

    return final_cols

# ---------------- MATRA ----------------

def classify_upper_matra(img, header):
    rows = np.where(np.sum(img > 0, axis=1) > 0)[0]
    if len(rows) == 0 or rows[0] > header - 3:
        return ""
    h = rows[-1] - rows[0]
    relative_h = h / img.shape[0]  # fraction of image height

    if relative_h < 1.5:
        return "ि"
    elif relative_h < 1.6:
        return "ी"
    else:
        return "ै"


def classify_lower_matra(img, header):
    """
    Classify lower matras (ु, ू, ृ) based on relative height.
    
    img: binary image of the component
    header: row index of the shirorekha/header
    """
    rows = np.where(np.sum(img > 0, axis=1) > 0)[0]
    if len(rows) == 0 or rows[-1] < header + 4:
        return ""

    h = rows[-1] - rows[0]
    relative_h = h / img.shape[0]  # fraction of component height

    if relative_h < 1.5:
        return "ु"
    elif relative_h < 1.6:
        return "ू"
    else:
        return "ृ"


def has_shirorekha(img):
    upper = int(0.55 * img.shape[0])
    return np.sum(img[:upper] > 0, axis=1).max() > 0.5 * img.shape[1]

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
    if img is None:
        return "?"  # skip invalid slices
    img = img.astype(np.float32)
    img = (img / 255.0 - 0.5) * 2
    x = torch.tensor(img[None, None]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        idx = torch.argmax(probs, dim=1).item()
    return IDX2CHAR[idx] if idx < len(IDX2CHAR) else "?"

# ---------------- OCR ----------------

def recognize_word(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = crop_text(binarize(img))
    if img.size == 0:
        return ""

    header = detect_header(img)
    img = remove_header(img, header)

    shirorekha_exists = has_shirorekha(img)
    columns = segment_columns(img)
    columns = [(c,a,b) for c,a,b in columns
               if is_valid_column(c)
               and not is_matra_only(c, header)
               and not is_shirorekha_only(c)]

    # ---- space thresholds ----
    gaps = [(columns[i][1] - columns[i-1][2]) for i in range(1, len(columns))]
    avg_gap = np.mean(gaps) if gaps else 0
    ABS_SPACE_GAP = int(0.03 * img.shape[1])
    REL_SPACE_GAP = avg_gap * 1.4

    result = ""
    pending_pre_matra = ""

    for i, (col, start_x, end_x) in enumerate(columns):
        print(f"Processing column {i}, pixels:", np.count_nonzero(col))
        cv2.imshow(f"Column {i}", col)
        cv2.waitKey(0)

        if i > 0:
            gap = start_x - columns[i-1][2]
            if gap > REL_SPACE_GAP or gap > ABS_SPACE_GAP:
                result += " "

        for j, p in enumerate(split_half_form(col, header)):

            if not is_valid_component(p):
                continue

            # ---- Detect pre-base (upper) matra ----
            pre_matra = ""
            if shirorekha_exists:
                upper_region = p[:header, :]
                pre_matra = classify_upper_matra(upper_region, header)
                if pre_matra:
                    pending_pre_matra += pre_matra
                    # Remove just the matra rows
                    rows_m = np.where(np.sum(upper_region > 0, axis=1) > 0)[0]
                    if len(rows_m) > 0:
                        p = p[rows_m[-1]+1:, :]

            # ---- Detect post-base (lower) matra ----
            post_matra = ""
            if shirorekha_exists:
                lower_region = p[header:, :]
                post_matra = classify_lower_matra(lower_region, header)
                if post_matra:
                    rows_m = np.where(np.sum(lower_region > 0, axis=1) > 0)[0]
                    if len(rows_m) > 0:
                        p = p[:header + rows_m[0], :]

            # ---- Base character ----
            base = infer_base(p)

            if pending_pre_matra:
                if is_real_base(p, header):
                    result += pending_pre_matra + base
                    pending_pre_matra = ""
                else:
                    # This is not a valid base, skip attaching pending matra
                    continue
            else:
                if is_real_base(p, header):
                    result += base


            # ---- DEBUG OUTPUT ----
            print(f"Column: {i} Start-End: {start_x} {end_x} Pixels: {np.count_nonzero(col)}")
            print(f"Component: {j} Top-Bottom: {np.where(np.sum(p>0,axis=1)>0)[0][0] if np.count_nonzero(p) else 0} {np.where(np.sum(p>0,axis=1)>0)[0][-1] if np.count_nonzero(p) else 0}")
            print(f"Pre-base matra: {pre_matra} Post-base matra: {post_matra}")
            print(f"Base: {base}")
            print(f"Current result: {result}\n")

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
