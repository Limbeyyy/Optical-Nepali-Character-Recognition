import cv2
from plate_detector import PlateDetector
from plate_preprocess import preprocess_plate
from ocr_bridge import run_plate_ocr

# =====================
# CONFIG
# =====================
CAMERA_SOURCE = 0   # 0 = webcam, or IP camera URL
YOLO_MODEL = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\Plate_Detetctor\weights\plate_yolo.pt"
JSON_PATH = r"C:\Users\Hp\Desktop\Optical_Nepali_OCR\OCR_Plate_Processor\Plate_Templates\default_plate_template.json"
OCR_INTERVAL = 5    # OCR every N frames

# =====================
# INIT
# =====================
cap = cv2.VideoCapture(CAMERA_SOURCE)
detector = PlateDetector(YOLO_MODEL)

frame_count = 0
last_text = {}

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")


import time

HOLD_SECONDS = 10

hold_active = False
hold_start_time = 0
held_boxes = []


# =====================
# LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    if not hold_active:
        detections = detector.detect(frame)

        if detections:
            # Start holding once a plate is detected
            hold_active = True
            hold_start_time = current_time
            held_boxes = detections
    else:
        # During hold, reuse same detections
        detections = held_boxes

        # End hold after HOLD_SECONDS
        if current_time - hold_start_time >= HOLD_SECONDS:
            hold_active = False


    for box in detections:
        x1, y1, x2, y2, conf = box

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        if not hold_active and frame_count % OCR_INTERVAL == 0:
            plate = detector.crop(frame, box)
            clean_plate = preprocess_plate(plate)
            last_text = run_plate_ocr(clean_plate, JSON_PATH)

            cv2.imshow("Plate Crop", clean_plate)

        if hold_active:
            remaining = int(HOLD_SECONDS - (current_time - hold_start_time))
            cv2.putText(
                frame,
                f"HOLDING... OCR in {remaining}s",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        y = 30
        for field, value in last_text.items():
            if not value.strip():
                continue

            line = f"{field}: {value}"
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y += 25


    cv2.imshow("Live Plate Scanner", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
