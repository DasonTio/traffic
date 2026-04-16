from traffic_anomaly.config import SceneConfig


def test_scene_config_loads_local_bytetrack():
    scene = SceneConfig.load("configs/scene_config.yaml")
    assert scene.tracker_config.name == "bytetrack.yaml"
    assert scene.yolo_weights.name == "yolo11n.pt"
    assert len(scene.lanes) == 4
    assert scene.class_lane_policy["Truck"] == ["fast_lane_1", "fast_lane_2"]
    assert scene.video_source_mode == "youtube"
    assert "youtube" in scene.video_sources
    assert "local" in scene.video_sources


def test_scene_config_can_select_local_video_source():
    scene = SceneConfig.load("configs/scene_config.yaml", source_mode="local")
    assert scene.video_source_mode == "local"
    assert scene.video_source.endswith("video.mp4.mp4")
