# run_ocr_local.py

import os
import json
import argparse
from OCR_Engine.ocr_engine import run_plate_ocr

# ---------------- Config ----------------
UPLOAD_DIR = "images"
TEMPLATE_DIR = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Plate_Templates"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# ---------------- CLI Parser ----------------
parser = argparse.ArgumentParser(description="Run Plate OCR locally")
parser.add_argument("--image", required=True, help="Path to plate image")
parser.add_argument(
    "--plate_type",
    choices=["default", "manual"],
    default="default",
    help="Template type"
)
parser.add_argument(
    "--rois",
    help="Path to manual ROI JSON file (required if plate_type=manual)"
)
args = parser.parse_args()

# ---------------- Validate ----------------
image_path = args.image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Plate image not found: {image_path}")

if args.plate_type == "default":
    rois_json_path = os.path.join(TEMPLATE_DIR, "default_plate_template.json")
    if not os.path.exists(rois_json_path):
        raise FileNotFoundError(f"Default template not found: {rois_json_path}")

elif args.plate_type == "manual":
    if not args.rois:
        raise ValueError("Manual mode requires --rois path")
    rois_json_path = args.rois
    if not os.path.exists(rois_json_path):
        raise FileNotFoundError(f"ROI JSON file not found: {rois_json_path}")

# ---------------- Run OCR ----------------
print("\n===== RUNNING OCR =====\n")
ocr_result = run_plate_ocr(image_path, rois_json_path)

# ---------------- Print Result ----------------
print("\n===== OCR RESULT =====\n")
for field, value in ocr_result.items():
    print(f"{field}: {value}")
