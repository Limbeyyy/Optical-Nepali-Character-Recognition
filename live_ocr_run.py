# live_plate_scanner.py
import cv2
import os
import json
import pytesseract
from OCR_Engine.ocr_engine import run_plate_ocr
from OCR_Engine.utils import crop_roi, normalize_wada_number
from QR_Processor.qr_reader import read_qr_from_image
from config import LANG
import datetime

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- CONFIG ----------------
# Normalized ROI for the plate (x1, y1, x2, y2)
# Adjust these coordinates based on where the plate appears in your camera feed
PLATE_ROI = [0.1, 0.1, 0.9, 0.9]  

# Output directories
SCAN_DIR = "live_scans"
os.makedirs(SCAN_DIR, exist_ok=True)

# ROI JSON template path (for OCR fields)
ROI_JSON_PATH = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Plate_Templates\default_plate_template.json"

# ---------------- FUNCTIONS ----------------
def crop_and_resize(frame, roi, width=600, height=200):
    cropped = crop_roi(frame, roi)
    resized = cv2.resize(cropped, (width, height))
    return resized

def run_ocr_and_qr(img):
    results = run_plate_ocr(img_path=None, rois_json_path=ROI_JSON_PATH)  # We'll modify OCR engine to accept numpy
    return results

# ---------------- WEBCAM LOOP ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Cannot open webcam")
    exit()

print("📸 Webcam live feed started. Press SPACE to capture, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    # Draw ROI rectangle on feed
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v * w if i % 2 == 0 else v * h) for i, v in enumerate(PLATE_ROI)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Live Feed - Press SPACE to Scan", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == 32:  # SPACE to capture
        cropped_plate = crop_and_resize(frame, PLATE_ROI)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SCAN_DIR, f"scan_{timestamp}.jpg")
        cv2.imwrite(save_path, cropped_plate)
        print(f"\n Saved scan: {save_path}")

        # Run QR decoding
        qr_value = read_qr_from_image(cropped_plate, verbose=True)
        print(f"[QR Result] {qr_value}")

        # Run OCR on cropped plate
        results = run_plate_ocr(save_path, ROI_JSON_PATH)
        results["QR_Code"] = qr_value
        print("\n===== OCR Result =====")
        import json
        print(json.dumps(results, indent=4, ensure_ascii=False))

cap.release()
cv2.destroyAllWindows()
