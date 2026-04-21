import numpy as np
import pytest

from traffic_anomaly.tracker_backend import TrackerBackend, load_tracker_spec

pytest.importorskip("boxmot")


class DummyModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.last_conf = None

    def predict(self, frame, classes, conf, verbose):
        self.last_conf = conf
        return [DummyResult(self._boxes)]


class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes


class DummyBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


def test_load_bytetrack_spec_uses_low_score_confidence():
    spec = load_tracker_spec("configs/bytetrack.yaml", default_confidence=0.45)
    assert spec.tracker_type == "bytetrack"
    assert spec.detector_confidence == 0.1


def test_tracker_backend_tracks_with_bytetrack():
    boxes = DummyBoxes([[10, 10, 30, 30]], [0.9], [7])
    model = DummyModel(boxes)
    backend = TrackerBackend(
        model=model,
        tracker_config="configs/bytetrack.yaml",
        detect_classes=[2, 5, 7],
        default_confidence=0.45,
        fps=30.0,
    )

    tracks = backend.track(np.zeros((100, 100, 3), dtype=np.uint8))

    assert model.last_conf == 0.1
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_id == 7


def test_tracker_backend_tracks_with_ocsort():
    boxes = DummyBoxes([[10, 10, 30, 30]], [0.9], [7])
    model = DummyModel(boxes)
    backend = TrackerBackend(
        model=model,
        tracker_config="configs/ocsort.yaml",
        detect_classes=[2, 5, 7],
        default_confidence=0.45,
        fps=30.0,
    )

    tracks = backend.track(np.zeros((100, 100, 3), dtype=np.uint8))

    assert model.last_conf == 0.1
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_id == 7
