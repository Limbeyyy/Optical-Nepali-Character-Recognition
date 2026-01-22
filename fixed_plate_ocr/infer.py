
import cv2
import torch
import sys
import os

# Add the repo root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_ROOT)

# Now you can import
from ocr.model import EnhancedBMCNNwHFCs

from shirorekha import extract_characters
from label_map import CLASS_TO_CHAR

device = "cpu"
model = EnhancedBMCNNwHFCs(num_classes=58).to(device)
ckpt = torch.load("/home/kataho/Downloads/mallanet_ocr/models/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

img = cv2.imread("/home/kataho/Downloads/mallanet_ocr/data/test_images/lah.jpeg")

ROIS = {
    "kataho_address": (131, 431, 1472, 611),
    "KID_No": (1093, 628, 1511, 678),
    "Plus_Code": (694, 768, 1217, 828),
    "Address_Name": (402, 856, 1167, 1046),
    "QR Code": (1280, 819, 1509, 1044)
}

for field,(x1,y1,x2,y2) in ROIS.items():
    roi = img[y1:y2, x1:x2]
    chars = extract_characters(roi)
    text = ""

    for x,y,w,h in chars:
        c = roi[y:y+h, x:x+w]
        c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        c = cv2.resize(c, (32,32)) / 255.0
        t = torch.tensor(c).unsqueeze(0).unsqueeze(0).float()  # [1,1,32,32]

        with torch.no_grad():
            pred = model(t).argmax(1).item()

        text += CLASS_TO_CHAR.get(pred, "?")

    print(f"{field}: {text}")

