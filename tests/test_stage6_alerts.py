import csv
import json

import numpy as np

from traffic_anomaly.config import AlertsConfig, AnomalyConfig, FusionConfig
from traffic_anomaly.contracts import CAR_CLASS_GROUP, FusedIncident, ModelHit, TrackFeature
from traffic_anomaly.stage6_alerts import AlertStage


def make_anomaly_config(dataset_root) -> AnomalyConfig:
    return AnomalyConfig(
        enabled=True,
        checkpoint_path=dataset_root / "ganomaly_car.pt",
        image_size=32,
        crop_padding=4,
        latent_dim=16,
        channels=3,
        batch_size=4,
        epochs=1,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999,
        threshold_percentile=95.0,
        ema_alpha=0.3,
        aggregation_window=3,
        dataset_root=dataset_root,
        save_training_candidates=True,
        min_crop_width=8,
        min_crop_height=8,
    )


def make_feature(frame_idx: int) -> TrackFeature:
    return TrackFeature(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 30.0,
        track_id=9,
        class_id=2,
        class_name="Car",
        class_group=CAR_CLASS_GROUP,
        conf=0.9,
        bbox=(10, 10, 40, 50),
        center_px=(25.0, 30.0),
        footpoint_px=(25.0, 50.0),
        bev_point=(30.0, 40.0),
        dx=1.0,
        dy=0.0,
        speed=3.0,
        acceleration=0.1,
        heading=(1.0, 0.0),
        heading_alignment=0.8,
        lane_id="fast_lane_1",
        lane_category="fast",
        dwell_frames=30,
        lane_violation_frames=0,
        wrong_way_frames=60,
        stopped_frames=0,
        max_recent_speed=3.0,
        speed_drop=0.0,
        bbox_width=30.0,
        bbox_height=40.0,
    )


def test_alert_stage_writes_tracklets_incidents_and_summary(tmp_path):
    alerts = AlertsConfig(run_root=tmp_path)
    anomaly = make_anomaly_config(tmp_path / "dataset")
    fusion = FusionConfig(
        model_escalation_threshold=1.0,
        model_only_threshold=1.2,
        consecutive_frames_required=3,
        merge_gap_frames=5,
    )
    stage = AlertStage(alerts, fusion, anomaly, camera_id="camera_001", ganomaly_ready=True)
    feature_29 = make_feature(29)
    feature_30 = make_feature(30)
    incident = FusedIncident(
        frame_idx=30,
        track_id=9,
        class_name="Car",
        anomaly_type="wrong_way",
        anomaly_family="wrong_way",
        severity="critical",
        rule_score=1.0,
        ganomaly_score=0.7,
        fused_score=1.0,
        explanation="Car heading opposed lane direction.",
        lane_id="fast_lane_1",
        bbox=feature_30.bbox,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[6:64, 6:64] = 255
    stage.record_frame(
        frame_idx=29,
        features=[feature_29],
        ganomaly_hits={
            9: ModelHit(
                frame_idx=29,
                track_id=9,
                class_name="Car",
                class_group=CAR_CLASS_GROUP,
                score=0.2,
                threshold=1.0,
                is_anomalous=False,
            )
        },
        incidents=[],
        frame=frame,
    )
    stage.record_frame(
        frame_idx=30,
        features=[feature_30],
        ganomaly_hits={
            9: ModelHit(
                frame_idx=30,
                track_id=9,
                class_name="Car",
                class_group=CAR_CLASS_GROUP,
                score=0.7,
                threshold=1.0,
                is_anomalous=False,
            )
        },
        incidents=[incident],
        frame=frame,
    )
    stage.close()

    tracklets_path = stage.root / "tracklets.csv"
    incidents_path = stage.root / "incidents.csv"
    summary_path = stage.root / "run_summary.json"

    assert tracklets_path.exists()
    assert incidents_path.exists()
    assert summary_path.exists()
    manifest_path = anomaly.dataset_root / "review_manifest.csv"
    assert manifest_path.exists()

    with tracklets_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "ganomaly_score" in rows[0]

    with incidents_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["ganomaly_score"] == "0.7000"

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["status"] == "pending"

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["ganomaly_ready"] is True
    assert summary["candidate_rows"] == 1
