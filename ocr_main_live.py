import cv2
import time
import os
import json


from Plate_Detector.plate_detector import PlateDetector
from OCR_Engine.plate_postprocess import  post_process_ocr
from Plate_Detector.detector_utils import shrink_bbox
from OCR_Engine.ocr_engine import ocr_image_rois
from Plate_Detector.Plate_Backend.backend_pipeline import validate_ocr_with_api
from config import YOLO_MODEL, JSON_PATH, OUTPUT_DIR, ROI_DIR, LIVE_CAMERA, HOLD_SECONDS, MAX_OCR_ROUNDS 

# ================= LOAD ROI TEMPLATE =================
with open(JSON_PATH, "r") as f:
    ROIS = json.load(f)


# ================= OCR ROUNDS =================
ocr_rounds_done = 0 # initialize OCR rounds counter
verified_fields = {}  # persistent across rounds


# ================= CAMERA INIT =================
cap = cv2.VideoCapture(LIVE_CAMERA, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4208)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3120)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

time.sleep(2)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

detector = PlateDetector(YOLO_MODEL)
round_count = 0
hold_active = False
hold_start = 0

print("Monitoring live feed...")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = cv2.resize(frame, (640, 480))
    detections = detector.detect(display)

    # Draw YOLO boxes and live ROIs
    for box in detections:
        x1, y1, x2, y2, conf = box
        x1, y1, x2, y2 = shrink_bbox(x1, y1, x2, y2, 0.01, 0.02)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plate_w = x2 - x1
        plate_h = y2 - y1

        for field, coords in ROIS.items():
            rx1 = int(x1 + coords[0] * plate_w)
            ry1 = int(y1 + coords[1] * plate_h)
            rx2 = int(x1 + coords[2] * plate_w)
            ry2 = int(y1 + coords[3] * plate_h)

            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)
            cv2.putText(display, field, (rx1, ry1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Detection hold logic
    if detections:
        if not hold_active:
            hold_active = True
            hold_start = time.time()
        elif time.time() - hold_start >= HOLD_SECONDS:
            round_count += 1
            hold_active = False
    else:
        round_count = 0
        hold_active = False

    ocr_results_rounds = []  # store OCR results of each round

    # ================= HIGH-RES SCAN =================
    if round_count >= 1:
        timestamp = int(time.time())
        scan_img = frame.copy()
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"scan_{timestamp}.png"), scan_img)

        plates = detector.detect(scan_img)
        if not plates:
            print("No plates detected in high-res scan")
            round_count = 0
            hold_active = False
            continue

        ocr_rounds_done += 1
        print(f"\nOCR ROUND {ocr_rounds_done} / {MAX_OCR_ROUNDS}")

        # Loop over plates
        for i, (x1, y1, x2, y2, conf) in enumerate(plates):
            x1, y1, x2, y2 = shrink_bbox(x1, y1, x2, y2, 0.01, 0.02)
            plate_crop = scan_img[y1:y2, x1:x2].copy()
            h, w = plate_crop.shape[:2]

            roi_dict = {}
            for field, coords in ROIS.items():
                rx1 = int(coords[0] * w)
                ry1 = int(coords[1] * h)
                rx2 = int(coords[2] * w)
                ry2 = int(coords[3] * h)

                roi = plate_crop[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    roi_dict[field] = None
                    continue

                roi_dict[field] = roi
                cv2.imwrite(
                    os.path.join(ROI_DIR, f"{timestamp}_r{ocr_rounds_done}_{field}.png"),
                    roi
                )

            # Run OCR
            results = ocr_image_rois(roi_dict, clean_text=True)
            processed_results = {
                field: post_process_ocr(field, text)
                for field, text in results.items()
            }

            print(" OCR RESULT (Round {ocr_rounds_done}, Plate {i+1}):")
            for k, v in processed_results.items():
                print(f"  {k}: {v}")

            # Append this plate OCR to the rounds list
            ocr_results_rounds.append(processed_results)

        # Reset for next round
        round_count = 0
        hold_active = False

        # Only validate API after at least 1 round and store updated values for next round
        final_results, verified_fields, remaining_fields = validate_ocr_with_api(
            ocr_results_rounds,
            previously_verified=verified_fields
        )
       
        print("\nFINAL VALIDATED OCR RESULT:")

        if verified_fields:
            for field in [
                "KID_No",
                "Kataho_Address",
                "Plus_Code",
                "QR_Code",
                "Local_Address",
                "Ward_Address",
                "City"
            ]:
                print(f"  {field}: {verified_fields.get(field)}")
        else:
            print("  No fields validated yet.")


        # Stop after MAX rounds
        if ocr_rounds_done >= MAX_OCR_ROUNDS:
            print("\n Completed all OCR rounds. Exiting.")
            break


    cv2.imshow("Live Feed", display)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done")
