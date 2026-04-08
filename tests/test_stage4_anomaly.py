import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from traffic_anomaly.config import AnomalyConfig
from traffic_anomaly.contracts import CAR_CLASS_GROUP, TrackFeature
from traffic_anomaly.stage4_anomaly import GANomalyModel, GANomalyScorer, load_reviewed_crop_paths


def make_config(checkpoint_path: Path, dataset_root: Path) -> AnomalyConfig:
    return AnomalyConfig(
        enabled=True,
        checkpoint_path=checkpoint_path,
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


def make_feature(frame_idx: int = 16) -> TrackFeature:
    return TrackFeature(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 30.0,
        track_id=3,
        class_id=2,
        class_name="Car",
        class_group=CAR_CLASS_GROUP,
        conf=0.9,
        bbox=(10, 10, 42, 42),
        center_px=(26.0, 26.0),
        footpoint_px=(26.0, 42.0),
        bev_point=(20.0, 30.0),
        dx=1.0,
        dy=0.5,
        speed=4.0,
        acceleration=0.2,
        heading=(0.8, 0.2),
        heading_alignment=0.7,
        lane_id="fast_lane_1",
        lane_category="fast",
        dwell_frames=20,
        lane_violation_frames=0,
        wrong_way_frames=0,
        stopped_frames=0,
        max_recent_speed=4.0,
        speed_drop=0.0,
        bbox_width=32.0,
        bbox_height=32.0,
    )


def write_checkpoint(checkpoint_path: Path) -> None:
    model = GANomalyModel(channels=3, image_size=32, latent_dim=16)
    torch.save(
        {
            "generator_state_dict": model.state_dict(),
            "channels": 3,
            "image_size": 32,
            "latent_dim": 16,
            "threshold": 1.0,
            "class_group": CAR_CLASS_GROUP,
        },
        checkpoint_path,
    )


def test_ganomaly_scorer_loads_checkpoint_and_scores_frame(tmp_path):
    checkpoint_path = tmp_path / "ganomaly_car.pt"
    write_checkpoint(checkpoint_path)
    config = make_config(checkpoint_path, tmp_path / "dataset")
    scorer = GANomalyScorer(config)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[8:48, 8:48] = 255

    hits = scorer.score_frame(frame, [make_feature()])

    assert 3 in hits
    assert hits[3].track_id == 3
    assert hits[3].class_group == CAR_CLASS_GROUP
    assert hits[3].score >= 0.0


def test_missing_ganomaly_checkpoint_yields_no_hits(tmp_path):
    config = make_config(tmp_path / "missing.pt", tmp_path / "dataset")
    scorer = GANomalyScorer(config)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)

    assert not scorer.available()
    assert scorer.score_frame(frame, [make_feature()]) == {}


def test_reviewed_crop_loader_only_uses_approved_rows(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    approved_crop = (dataset_root / "approved_1.jpg").resolve()
    pending_crop = (dataset_root / "pending_1.jpg").resolve()
    rejected_crop = (dataset_root / "rejected_1.jpg").resolve()
    image = np.full((24, 24, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(approved_crop), image)
    cv2.imwrite(str(pending_crop), image)
    cv2.imwrite(str(rejected_crop), image)

    manifest_path = dataset_root / "review_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "class_name", "status", "crop_path"])
        writer.writeheader()
        writer.writerow({"sample_id": "a1", "class_name": "Car", "status": "approved", "crop_path": str(approved_crop)})
        writer.writerow({"sample_id": "p1", "class_name": "Car", "status": "pending", "crop_path": str(pending_crop)})
        writer.writerow({"sample_id": "r1", "class_name": "Car", "status": "rejected", "crop_path": str(rejected_crop)})

    crop_paths = load_reviewed_crop_paths(manifest_path=manifest_path, dataset_root=dataset_root)

    assert crop_paths == [approved_crop]
