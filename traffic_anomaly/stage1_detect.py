from __future__ import annotations

from ultralytics import YOLO

from .config import DetectConfig
from .contracts import Detection


class DetectionStage:
    def __init__(self, config: DetectConfig):
        self.config = config
        self.model = YOLO(str(config.weights))

    def run(self, frame) -> list[Detection]:
        results = self.model.predict(
            frame,
            classes=self.config.classes,
            conf=self.config.confidence,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        detections: list[Detection] = []
        for box, conf, class_id in zip(boxes, confs, classes):
            bbox = tuple(int(round(value)) for value in box)
            class_id = int(class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.config.class_name_for(class_id),
                    conf=float(conf),
                    bbox=bbox,
                )
            )
        return detections

