from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_COCO_NAMES = {
    2: "Car",
    5: "Bus",
    7: "Truck",
}


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for root in (base_dir, base_dir.parent, Path.cwd()):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (base_dir.parent / candidate).resolve()


@dataclass(frozen=True)
class LaneConfig:
    id: str
    label: str
    category: str
    direction: np.ndarray
    polygon: np.ndarray


@dataclass(frozen=True)
class VideoConfig:
    source: str
    resolution: str | None
    fps: float


@dataclass(frozen=True)
class DetectConfig:
    weights: Path
    confidence: float
    classes: list[int]
    coco_names: dict[int, str]

    def class_name_for(self, class_id: int) -> str:
        return self.coco_names.get(class_id, "Vehicle")


@dataclass(frozen=True)
class TrackConfig:
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    match_thresh: float
    track_buffer: int
    fuse_score: bool
    state_ttl_frames: int


@dataclass(frozen=True)
class FeaturesConfig:
    homography: np.ndarray
    lanes: list[LaneConfig]
    lane_lookup: dict[str, LaneConfig]
    class_lane_policy: dict[str, list[str]]
    smoothing_window: int
    sudden_stop_window_seconds: float
    stopped_speed_threshold: float
    motion_floor: float
    wrong_way_alignment_threshold: float
    lane_violation_seconds: float
    wrong_way_seconds: float
    stopped_vehicle_seconds: float
    sudden_stop_delta: float
    congestion_slow_ratio: float
    congestion_min_active_tracks: int
    min_track_age_frames: int

    def lane_for_id(self, lane_id: str | None) -> LaneConfig | None:
        if lane_id is None:
            return None
        return self.lane_lookup.get(lane_id)


@dataclass(frozen=True)
class AnomalyConfig:
    enabled: bool
    checkpoint_path: Path | None
    image_size: int
    crop_padding: int
    latent_dim: int
    channels: int
    batch_size: int
    epochs: int
    learning_rate: float
    beta1: float
    beta2: float
    threshold_percentile: float
    ema_alpha: float
    aggregation_window: int
    dataset_root: Path
    save_training_candidates: bool
    min_crop_width: int
    min_crop_height: int


@dataclass(frozen=True)
class FusionConfig:
    model_escalation_threshold: float
    model_only_threshold: float
    consecutive_frames_required: int
    merge_gap_frames: int


@dataclass(frozen=True)
class AlertsConfig:
    run_root: Path


@dataclass(frozen=True)
class SceneConfig:
    camera_id: str
    config_path: Path
    video: VideoConfig
    detect: DetectConfig
    track: TrackConfig
    features: FeaturesConfig
    anomaly: AnomalyConfig
    fusion: FusionConfig
    alerts: AlertsConfig

    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        config_path = Path(path).resolve()
        base_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        video_raw = raw.get("video", {})
        detect_raw = raw.get("detect", {})
        track_raw = raw.get("track", {})
        features_raw = raw.get("features", {})
        anomaly_raw = raw.get("anomaly", {})
        fusion_raw = raw.get("fusion", {})
        alerts_raw = raw.get("alerts", {})

        homography_raw = features_raw.get("homography", {})
        src_points = np.asarray(homography_raw["src_points"], dtype=np.float32)
        dst_points = np.asarray(homography_raw["dst_points"], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(src_points, dst_points)

        lanes: list[LaneConfig] = []
        lane_lookup: dict[str, LaneConfig] = {}
        for lane_raw in features_raw.get("lanes", []):
            direction = np.asarray(lane_raw["direction"], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm > 0:
                direction = direction / norm
            lane = LaneConfig(
                id=lane_raw["id"],
                label=lane_raw.get("label", lane_raw["id"].replace("_", " ").title()),
                category=lane_raw.get("category", "active"),
                direction=direction,
                polygon=np.asarray(lane_raw["polygon"], dtype=np.int32),
            )
            lanes.append(lane)
            lane_lookup[lane.id] = lane

        checkpoint_value = anomaly_raw.get("checkpoint")
        if checkpoint_value is None:
            checkpoint_value = anomaly_raw.get("checkpoints", {}).get("car")

        detect_cfg = DetectConfig(
            weights=_resolve_path(base_dir, detect_raw.get("weights", "yolo11n.pt")),
            confidence=float(detect_raw.get("confidence", 0.45)),
            classes=[int(value) for value in detect_raw.get("classes", [2, 5, 7])],
            coco_names={int(key): value for key, value in detect_raw.get("coco_names", DEFAULT_COCO_NAMES).items()},
        )

        return cls(
            camera_id=raw.get("camera_id", "camera_001"),
            config_path=config_path,
            video=VideoConfig(
                source=video_raw.get("source", ""),
                resolution=video_raw.get("resolution"),
                fps=float(video_raw.get("fps", 30.0)),
            ),
            detect=detect_cfg,
            track=TrackConfig(
                track_high_thresh=float(track_raw.get("track_high_thresh", 0.25)),
                track_low_thresh=float(track_raw.get("track_low_thresh", 0.10)),
                new_track_thresh=float(track_raw.get("new_track_thresh", 0.25)),
                match_thresh=float(track_raw.get("match_thresh", 0.80)),
                track_buffer=int(track_raw.get("track_buffer", 30)),
                fuse_score=bool(track_raw.get("fuse_score", True)),
                state_ttl_frames=int(track_raw.get("state_ttl_frames", 30)),
            ),
            features=FeaturesConfig(
                homography=homography,
                lanes=lanes,
                lane_lookup=lane_lookup,
                class_lane_policy={
                    class_name: list(policy.get("forbidden_lanes", []))
                    for class_name, policy in features_raw.get("class_lane_policy", {}).items()
                },
                smoothing_window=int(features_raw.get("smoothing_window", 5)),
                sudden_stop_window_seconds=float(features_raw.get("sudden_stop_window_seconds", 1.5)),
                stopped_speed_threshold=float(features_raw.get("stopped_speed_threshold", 2.5)),
                motion_floor=float(features_raw.get("motion_floor", 8.0)),
                wrong_way_alignment_threshold=float(features_raw.get("wrong_way_alignment_threshold", -0.5)),
                lane_violation_seconds=float(features_raw.get("lane_violation_seconds", 0.5)),
                wrong_way_seconds=float(features_raw.get("wrong_way_seconds", 2.0)),
                stopped_vehicle_seconds=float(features_raw.get("stopped_vehicle_seconds", 2.0)),
                sudden_stop_delta=float(features_raw.get("sudden_stop_delta", 6.0)),
                congestion_slow_ratio=float(features_raw.get("congestion_slow_ratio", 0.65)),
                congestion_min_active_tracks=int(features_raw.get("congestion_min_active_tracks", 3)),
                min_track_age_frames=int(features_raw.get("min_track_age_frames", 15)),
            ),
            anomaly=AnomalyConfig(
                enabled=bool(anomaly_raw.get("enabled", True)),
                checkpoint_path=_resolve_path(base_dir, checkpoint_value) if checkpoint_value else None,
                image_size=int(anomaly_raw.get("image_size", 64)),
                crop_padding=int(anomaly_raw.get("crop_padding", 15)),
                latent_dim=int(anomaly_raw.get("latent_dim", 128)),
                channels=int(anomaly_raw.get("channels", 3)),
                batch_size=int(anomaly_raw.get("batch_size", 32)),
                epochs=int(anomaly_raw.get("epochs", 20)),
                learning_rate=float(anomaly_raw.get("learning_rate", 0.0002)),
                beta1=float(anomaly_raw.get("beta1", 0.5)),
                beta2=float(anomaly_raw.get("beta2", 0.999)),
                threshold_percentile=float(anomaly_raw.get("threshold_percentile", 95.0)),
                ema_alpha=float(anomaly_raw.get("ema_alpha", 0.3)),
                aggregation_window=int(anomaly_raw.get("aggregation_window", 20)),
                dataset_root=_resolve_path(base_dir, anomaly_raw.get("dataset_root", "dataset/ganomaly")),
                save_training_candidates=bool(anomaly_raw.get("save_training_candidates", True)),
                min_crop_width=int(anomaly_raw.get("min_crop_width", 24)),
                min_crop_height=int(anomaly_raw.get("min_crop_height", 24)),
            ),
            fusion=FusionConfig(
                model_escalation_threshold=float(fusion_raw.get("model_escalation_threshold", 1.0)),
                model_only_threshold=float(fusion_raw.get("model_only_threshold", 1.2)),
                consecutive_frames_required=int(fusion_raw.get("consecutive_frames_required", 3)),
                merge_gap_frames=int(fusion_raw.get("merge_gap_frames", 5)),
            ),
            alerts=AlertsConfig(
                run_root=_resolve_path(base_dir, alerts_raw.get("run_root", "runs")),
            ),
        )
