from traffic_anomaly.config import SceneConfig
from traffic_anomaly.tracklets import TrackManager


def _update_feature(manager: TrackManager, frame_idx: int, class_id: int, class_name: str, conf: float = 0.9):
    return manager.update(
        frame_idx=frame_idx,
        track_id=7,
        class_id=class_id,
        class_name=class_name,
        conf=conf,
        bbox=(10, 10, 20, 30),
        footpoint_px=(15.0, 30.0),
        footpoint_bev=(15.0 + frame_idx, 30.0),
        lane_id="fast_lane_1",
        lane_category="fast",
        ganomaly_score=0.0,
    )


def test_track_manager_resists_single_frame_class_flip():
    scene = SceneConfig.load("configs/scene_config.yaml")
    manager = TrackManager(scene, fps=30.0)

    feature = None
    for frame_idx in range(1, 6):
        feature = _update_feature(manager, frame_idx, class_id=7, class_name="Truck")
    assert feature is not None
    assert feature.class_name == "Truck"

    feature = _update_feature(manager, 6, class_id=5, class_name="Bus", conf=0.95)

    assert feature.raw_class_name == "Bus"
    assert feature.class_name == "Truck"
    assert feature.stable_class_score > 0.5


def test_track_manager_switches_after_sustained_new_evidence():
    scene = SceneConfig.load("configs/scene_config.yaml")
    manager = TrackManager(scene, fps=30.0)

    for frame_idx in range(1, 4):
        _update_feature(manager, frame_idx, class_id=7, class_name="Truck")

    feature = None
    for frame_idx in range(4, 9):
        feature = _update_feature(manager, frame_idx, class_id=5, class_name="Bus", conf=0.95)

    assert feature is not None
    assert feature.raw_class_name == "Bus"
    assert feature.class_name == "Bus"
    assert feature.class_id == 5
