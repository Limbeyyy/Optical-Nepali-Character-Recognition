import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Returns list of (x1,y1,x2,y2,confidence)
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

        return detections

    @staticmethod
    def crop(frame, box):
        x1, y1, x2, y2, _ = box
        return frame[y1:y2, x1:x2]
