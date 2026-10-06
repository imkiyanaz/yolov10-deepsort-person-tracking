from ultralytics import YOLO


class YoloDetector:
    """Person detector backed by an Ultralytics YOLO model."""

    def __init__(self, model_path: str = "yolov10m.pt", confidence: float = 0.2):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.allowed_classes = {"person"}

    def detect(self, image):
        """Return Deep-SORT formatted detections for people in an image."""
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        return self._make_detections(result)

    def _make_detections(self, result):
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            if class_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            width = x2 - x1
            height = y2 - y1
            confidence = float(box.conf[0])

            # deep-sort-realtime expects: ([left, top, width, height], confidence, class)
            detections.append(([x1, y1, width, height], confidence, class_name))

        return detections
