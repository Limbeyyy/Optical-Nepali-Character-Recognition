import cv2
import torch
import os
import numpy as np
import argparse
import json

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Generate visualizations for each ROI in plate image")
parser.add_argument("--image", required=True, help="Path to plate image")
parser.add_argument(
    "--out_dir",
    default=r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Plate_Visualizations",
    help="Directory to save ROI visualizations"
)
args = parser.parse_args()


selected = True

if (selected == True):
    json_path = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Plate_Templates\default_next_plate_template.json"
else:
    json_path = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate\Plate_Templates\selected_plate_template.json"

if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found: {json_path}")

with open(json_path, "r") as f:
    ROIS_NORM = json.load(f)  # This loads the JSON as a dictionary

img_path = args.image
VIS_DIR = args.out_dir
os.makedirs(VIS_DIR, exist_ok=True)

# --- Load image ---
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

h, w = img.shape[:2]


def preprocess_char(c_img):
    """Grayscale + resize + tensor conversion for OCR"""
    c = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    c = cv2.resize(c, (32,32)) / 255.0
    t = torch.tensor(c).unsqueeze(0).unsqueeze(0).float()  # [1,1,32,32]
    return t

# --- Process each ROI ---
for field, (x1f, y1f, x2f, y2f) in ROIS_NORM.items():
    # Map normalized coords to absolute pixels
    x1, y1 = int(x1f * w), int(y1f * h)
    x2, y2 = int(x2f * w), int(y2f * h)
    roi = img[y1:y2, x1:x2].copy()

    # --- Create subfolder for this ROI ---
    roi_dir = os.path.join(VIS_DIR, field)
    os.makedirs(roi_dir, exist_ok=True)

    # --- Visualization preprocessing ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.dilate(binary, kernel, iterations=1)
    vis = roi.copy()

    # --- Save visualizations in the subfolder ---
    step_images = {
        "ROI": roi,
        "Gray": gray,
        "Binary": binary,
        "Morphology": morph,
        "Prediction": vis
    }
    for step, im in step_images.items():
        save_path = os.path.join(roi_dir, f"{step}.png")
        cv2.imwrite(save_path, im)

print(f"✅ All visualizations saved in '{VIS_DIR}' with subfolders for each ROI")
