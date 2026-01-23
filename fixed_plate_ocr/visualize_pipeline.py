import cv2
import torch
import sys
import os
import numpy as np

# --- Setup repo path ---
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_ROOT)

from ocr.model import EnhancedBMCNNwHFCs
from shirorekha import extract_characters
from label_map import CLASS_TO_CHAR

# --- Load OCR model ---
device = "cpu"
model = EnhancedBMCNNwHFCs(num_classes=58).to(device)
ckpt_path = "/home/kataho/Downloads/mallanet_ocr/models/best_model.pth"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# --- Load image ---
img_path = "/home/kataho/Downloads/mallanet_ocr/data/test_images/p_2.jpeg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

# --- Normalize ROIs as fractions of image size (0–1) ---
ROIS_NORM = {
    "kataho_address": (0.1031, 0.3169, 0.8938, 0.4804),
    "KID_No": (0.6281, 0.4787, 0.8888, 0.5321),
    "Plus_Code": (0.4530, 0.5688, 0.7656, 0.6555),
    "Address_Name": (0.2106, 0.6656, 0.7650, 0.8540),
    "QR Code": (0.7762, 0.6155, 0.9537, 0.8482),
    "Local_Address": (0.2184, 0.1228, 0.7746, 0.2666)
}

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

    # --- Visualization ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.dilate(binary, kernel, iterations=1)
    vis = roi.copy()

    # --- Character extraction and OCR ---
    chars = extract_characters(roi)
    text = ""
    for x,y,wc,hc in chars:
        c_img = roi[y:y+hc, x:x+wc]
        t = preprocess_char(c_img)
        with torch.no_grad():
            pred = model(t).argmax(1).item()
        char = CLASS_TO_CHAR.get(pred, "?")
        text += char
        cv2.rectangle(vis, (x,y), (x+wc, y+hc), (0,255,0), 1)
        cv2.putText(vis, char, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    print(f"{field}: {text}")

    # --- Show steps ---
    cv2.imshow(f"{field} - ROI", roi)
    cv2.imshow(f"{field} - Gray", gray)
    cv2.imshow(f"{field} - Binary", binary)
    cv2.imshow(f"{field} - Morphology", morph)
    cv2.imshow(f"{field} - Prediction", vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
