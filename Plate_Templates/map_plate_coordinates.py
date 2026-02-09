import cv2
import os
import json
import argparse
from config import COORDINATES_OUTPUT_DIR

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Draw ROIs on plate image and save normalized coordinates to JSON")
parser.add_argument("--image", required=True, help="Path to plate image(USE IMAGES FROM image_dataset FOLDER)")
JSON_DIR = COORDINATES_OUTPUT_DIR
args = parser.parse_args()

img_path = args.image
os.makedirs(JSON_DIR, exist_ok=True)

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

# --- Load image ---
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Failed to load image: {img_path}")

img_height, img_width = img.shape[:2]
print("Loaded image shape:", img.shape)
print("Loaded image path:", img_path)

# --- Ensure previous windows are destroyed ---
cv2.destroyAllWindows()
cv2.waitKey(1)

# --- Resize image for display ---
DISPLAY_WIDTH = 1200
scale = DISPLAY_WIDTH / img_width
display_height = int(img_height * scale)
img_display = cv2.resize(img, (DISPLAY_WIDTH, display_height))
clone = img_display.copy()

# --- ROI drawing variables ---
drawing = False
ix, iy = -1, -1
rois = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, clone, rois
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = clone.copy()
        cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("ROI Selection", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("ROI Selection", clone)

        # Map back to original image coordinates
        x1_orig = int(ix / scale)
        y1_orig = int(iy / scale)
        x2_orig = int(x / scale)
        y2_orig = int(y / scale)

        # --- Normalize to 0-1 ---
        x1_norm = x1_orig / img_width
        y1_norm = y1_orig / img_height
        x2_norm = x2_orig / img_width
        y2_norm = y2_orig / img_height

        rois.append((x1_norm, y1_norm, x2_norm, y2_norm))
        print(f"ROI added (normalized 0-1): ({x1_norm:.4f}, {y1_norm:.4f}, {x2_norm:.4f}, {y2_norm:.4f})")

# --- Setup window and mouse callback ---
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ROI Selection", draw_rectangle)
cv2.imshow("ROI Selection", clone)

print("Draw ROI by clicking and dragging. Press 'q' to finish.")

# --- Main loop ---
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# --- Cleanup ---
cv2.destroyAllWindows()
cv2.waitKey(1)

# --- Print all normalized ROIs ---
print("\nAll selected ROIs (normalized 0-1 coordinates):")
roi_dict = {}
for i, roi in enumerate(rois, 1):
    key_name = f"ROI_{i}"
    roi_dict[key_name] = roi
    print(f"{key_name}: {roi}")

# --- Save to JSON ---
json_path = os.path.join(JSON_DIR, "selected_plate_template.json")
with open(json_path, "w") as f:
    json.dump(roi_dict, f, indent=4)

print(f"\n✅ All ROIs saved to JSON file: {json_path}")
