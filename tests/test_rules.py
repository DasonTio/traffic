from traffic_anomaly.config import SceneConfig
from traffic_anomaly.rules import LaneSnapshot, RuleEngine
from traffic_anomaly.tracklets import TrackFeature


def make_feature(**overrides):
    base = dict(
        frame_idx=30,
        timestamp_s=1.0,
        track_id=7,
        class_id=7,
        class_name="Truck",
        conf=0.9,
        bbox=(10, 10, 20, 30),
        footpoint_px=(15.0, 30.0),
        footpoint_bev=(15.0, 30.0),
        lane_id="fast_lane_1",
        lane_category="fast",
        speed=12.0,
        acceleration=0.0,
        heading=(0.0, 1.0),
        heading_alignment=0.9,
        dwell_frames=20,
        lane_violation_frames=20,
        wrong_way_frames=0,
        stopped_frames=0,
        max_recent_speed=12.0,
        speed_drop=0.0,
        ganomaly_score=0.0,
    )
    base.update(overrides)
    return TrackFeature(**base)


def test_lane_violation_rule_triggers():
    scene = SceneConfig.load("configs/scene_config.yaml")
    engine = RuleEngine(scene, fps=30.0)
    hits = engine.evaluate(make_feature(lane_violation_frames=20), {})
    assert any(hit.anomaly_type == "lane_violation" for hit in hits)


def test_wrong_way_rule_triggers():
    scene = SceneConfig.load("configs/scene_config.yaml")
    engine = RuleEngine(scene, fps=30.0)
    hits = engine.evaluate(make_feature(heading_alignment=-0.8, wrong_way_frames=40), {})
    assert any(hit.anomaly_type == "wrong_way" for hit in hits)


def test_sudden_stop_suppressed_by_congestion():
    scene = SceneConfig.load("configs/scene_config.yaml")
    engine = RuleEngine(scene, fps=30.0)
    snapshots = {"fast_lane_1": LaneSnapshot(active_count=4, slow_count=3, avg_speed=1.0, congested=True)}
    hits = engine.evaluate(
        make_feature(speed=1.0, max_recent_speed=15.0, speed_drop=14.0),
        snapshots,
    )
    assert not any(hit.anomaly_type == "sudden_stop" for hit in hits)


def test_stopped_vehicle_in_emergency_lane_triggers():
    scene = SceneConfig.load("configs/scene_config.yaml")
    engine = RuleEngine(scene, fps=30.0)
    hits = engine.evaluate(
        make_feature(
            lane_id="emergency_lane_1",
            lane_category="emergency",
            speed=0.5,
            stopped_frames=80,
            lane_violation_frames=0,
            class_name="Car",
            class_id=2,
        ),
        {},
    )
    assert any(hit.anomaly_type == "stopped_vehicle" for hit in hits)
