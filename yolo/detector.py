from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray):
        # Run detection
        results = self.model(frame)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            clss = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else []
            for box, conf, cls in zip(boxes, confs, clss):
                detections.append({
                    'box': box,  # [x1, y1, x2, y2]
                    'conf': float(conf),
                    'class_id': int(cls),
                    'class_name': self.model.names[int(cls)] if hasattr(self.model, 'names') else str(cls)
                })
        return detections 