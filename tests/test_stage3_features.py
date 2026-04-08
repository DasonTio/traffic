from traffic_anomaly.config import SceneConfig
from traffic_anomaly.contracts import TrackObservation
from traffic_anomaly.stage3_features import FeatureStage


def test_feature_stage_flags_truck_in_fast_lane():
    scene = SceneConfig.load("configs/scene_config.yaml")
    stage = FeatureStage(scene.features, fps=30.0, state_ttl_frames=scene.track.state_ttl_frames)
    last_feature = None

    for frame_idx in range(1, 17):
        y_bottom = 220 + frame_idx * 2
        observation = TrackObservation(
            frame_idx=frame_idx,
            track_id=11,
            class_id=7,
            class_name="Truck",
            conf=0.9,
            bbox=(185, y_bottom - 36, 215, y_bottom),
        )
        features = stage.update(frame_idx, [observation])
        assert len(features) == 1
        last_feature = features[0]

    assert last_feature is not None
    assert last_feature.lane_id == "fast_lane_1"
    assert last_feature.class_group == "truck"
    assert last_feature.lane_violation_frames > 0


def test_feature_stage_flags_car_in_emergency_lane():
    scene = SceneConfig.load("configs/scene_config.yaml")
    stage = FeatureStage(scene.features, fps=30.0, state_ttl_frames=scene.track.state_ttl_frames)
    last_feature = None

    for frame_idx in range(1, 17):
        observation = TrackObservation(
            frame_idx=frame_idx,
            track_id=12,
            class_id=2,
            class_name="Car",
            conf=0.9,
            bbox=(40, 180, 60, 220),
        )
        features = stage.update(frame_idx, [observation])
        assert len(features) == 1
        last_feature = features[0]

    assert last_feature is not None
    assert last_feature.lane_id == "emergency_lane_1"
    assert last_feature.lane_violation_frames > 0
