from traffic_anomaly.config import SceneConfig
from traffic_anomaly.contracts import CAR_CLASS_GROUP, ModelHit, TrackFeature, class_group_for_name
from traffic_anomaly.stage5_fuse import FusionStage


def make_feature(**overrides) -> TrackFeature:
    base = dict(
        frame_idx=30,
        timestamp_s=1.0,
        track_id=7,
        class_id=2,
        class_name="Car",
        class_group=CAR_CLASS_GROUP,
        conf=0.9,
        bbox=(10, 10, 40, 50),
        center_px=(25.0, 30.0),
        footpoint_px=(25.0, 50.0),
        bev_point=(25.0, 30.0),
        dx=0.0,
        dy=0.0,
        speed=12.0,
        acceleration=0.0,
        heading=(0.0, 1.0),
        heading_alignment=0.9,
        lane_id="fast_lane_1",
        lane_category="fast",
        dwell_frames=20,
        lane_violation_frames=20,
        wrong_way_frames=0,
        stopped_frames=0,
        max_recent_speed=12.0,
        speed_drop=0.0,
        bbox_width=30.0,
        bbox_height=40.0,
    )
    base.update(overrides)
    return TrackFeature(**base)


def test_warning_rule_escalates_when_model_agrees():
    scene = SceneConfig.load("configs/scene_config.yaml")
    stage = FusionStage(scene.features, scene.fusion, fps=30.0)
    ganomaly_hit = ModelHit(
        frame_idx=30,
        track_id=7,
        class_name="Bus",
        class_group=class_group_for_name("Bus"),
        score=1.1,
        threshold=1.0,
        is_anomalous=True,
    )
    incidents = stage.evaluate(
        [
            make_feature(
                class_id=5,
                class_name="Bus",
                class_group=class_group_for_name("Bus"),
                lane_violation_frames=20,
            )
        ],
        {7: ganomaly_hit},
    )
    assert any(incident.anomaly_type == "lane_violation" and incident.severity == "critical" for incident in incidents)


def test_car_in_emergency_lane_triggers_lane_violation():
    scene = SceneConfig.load("configs/scene_config.yaml")
    stage = FusionStage(scene.features, scene.fusion, fps=30.0)
    incidents = stage.evaluate(
        [
            make_feature(
                lane_id="emergency_lane_1",
                lane_category="emergency",
                lane_violation_frames=20,
            )
        ],
        {},
    )
    assert any(incident.anomaly_type == "lane_violation" for incident in incidents)


def test_appearance_anomaly_requires_three_consecutive_frames():
    scene = SceneConfig.load("configs/scene_config.yaml")
    stage = FusionStage(scene.features, scene.fusion, fps=30.0)
    feature = make_feature(
        class_id=2,
        class_name="Car",
        class_group=CAR_CLASS_GROUP,
        lane_violation_frames=0,
        lane_id=None,
        lane_category=None,
        speed=8.0,
        max_recent_speed=8.0,
        speed_drop=0.0,
    )

    for frame_idx in (16, 20):
        incidents = stage.evaluate(
            [
                make_feature(
                    frame_idx=frame_idx,
                    class_id=feature.class_id,
                    class_name=feature.class_name,
                    class_group=feature.class_group,
                    conf=feature.conf,
                    bbox=feature.bbox,
                    center_px=feature.center_px,
                    footpoint_px=feature.footpoint_px,
                    bev_point=feature.bev_point,
                    dx=feature.dx,
                    dy=feature.dy,
                    speed=feature.speed,
                    acceleration=feature.acceleration,
                    heading=feature.heading,
                    heading_alignment=feature.heading_alignment,
                    lane_id=feature.lane_id,
                    lane_category=feature.lane_category,
                    dwell_frames=feature.dwell_frames,
                    lane_violation_frames=feature.lane_violation_frames,
                    wrong_way_frames=feature.wrong_way_frames,
                    stopped_frames=feature.stopped_frames,
                    max_recent_speed=feature.max_recent_speed,
                    speed_drop=feature.speed_drop,
                    bbox_width=feature.bbox_width,
                    bbox_height=feature.bbox_height,
                )
            ],
            {
                7: ModelHit(
                    frame_idx=frame_idx,
                    track_id=7,
                    class_name="Car",
                    class_group=CAR_CLASS_GROUP,
                    score=1.3,
                    threshold=1.0,
                    is_anomalous=True,
                )
            },
        )
        assert not any(incident.anomaly_type == "appearance_anomaly" for incident in incidents)

    incidents = stage.evaluate(
        [
            make_feature(
                frame_idx=24,
                class_id=feature.class_id,
                class_name=feature.class_name,
                class_group=feature.class_group,
                conf=feature.conf,
                bbox=feature.bbox,
                center_px=feature.center_px,
                footpoint_px=feature.footpoint_px,
                bev_point=feature.bev_point,
                dx=feature.dx,
                dy=feature.dy,
                speed=feature.speed,
                acceleration=feature.acceleration,
                heading=feature.heading,
                heading_alignment=feature.heading_alignment,
                lane_id=feature.lane_id,
                lane_category=feature.lane_category,
                dwell_frames=feature.dwell_frames,
                lane_violation_frames=feature.lane_violation_frames,
                wrong_way_frames=feature.wrong_way_frames,
                stopped_frames=feature.stopped_frames,
                max_recent_speed=feature.max_recent_speed,
                speed_drop=feature.speed_drop,
                bbox_width=feature.bbox_width,
                bbox_height=feature.bbox_height,
            )
        ],
        {
            7: ModelHit(
                frame_idx=24,
                track_id=7,
                class_name="Car",
                class_group=CAR_CLASS_GROUP,
                score=1.3,
                threshold=1.0,
                is_anomalous=True,
            )
        },
    )
    assert any(incident.anomaly_type == "appearance_anomaly" for incident in incidents)
