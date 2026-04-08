from traffic_anomaly.config import SceneConfig


def test_scene_config_loads_stage_layout():
    scene = SceneConfig.load("configs/scene_config.yaml")
    assert scene.detect.weights.name == "yolo11n.pt"
    assert scene.detect.classes == [2, 5, 7]
    assert scene.detect.coco_names[5] == "Bus"
    assert scene.detect.coco_names[7] == "Truck"
    assert scene.track.state_ttl_frames == 30
    assert not hasattr(scene.features, "window_size")
    assert not hasattr(scene.features, "stride")
    assert len(scene.features.lanes) == 4
    assert scene.features.class_lane_policy["Car"] == ["emergency_lane_1", "emergency_lane_2"]
    assert scene.features.class_lane_policy["Bus"] == ["fast_lane_1", "fast_lane_2"]
    assert scene.features.class_lane_policy["Truck"] == ["fast_lane_1", "fast_lane_2"]
    assert scene.anomaly.checkpoint_path is not None
    assert scene.anomaly.checkpoint_path.name == "ganomaly_car.pt"
    assert scene.anomaly.image_size == 64
    assert scene.fusion.consecutive_frames_required == 3
