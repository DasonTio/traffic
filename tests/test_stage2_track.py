import numpy as np

from traffic_anomaly.config import SceneConfig
from traffic_anomaly.contracts import Detection
from traffic_anomaly.stage2_track import TrackingStage


def test_tracking_stage_keeps_track_ids_stable():
    scene = SceneConfig.load("configs/scene_config.yaml")
    stage = TrackingStage(scene.track, fps=30.0)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = [Detection(class_id=2, class_name="Car", conf=0.9, bbox=(100, 100, 150, 150))]

    first = stage.run(1, detections, frame)
    second = stage.run(2, detections, frame)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id

