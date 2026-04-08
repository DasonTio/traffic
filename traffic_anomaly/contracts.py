from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CAR_CLASS_NAME = "Car"
CAR_CLASS_GROUP = "car"


def class_group_for_name(class_name: str) -> str:
    if class_name == CAR_CLASS_NAME:
        return CAR_CLASS_GROUP
    return class_name.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    conf: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackObservation:
    frame_idx: int
    track_id: int
    class_id: int
    class_name: str
    conf: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackFeature:
    frame_idx: int
    timestamp_s: float
    track_id: int
    class_id: int
    class_name: str
    class_group: str
    conf: float
    bbox: tuple[int, int, int, int]
    center_px: tuple[float, float]
    footpoint_px: tuple[float, float]
    bev_point: tuple[float, float]
    dx: float
    dy: float
    speed: float
    acceleration: float
    heading: tuple[float, float]
    heading_alignment: float
    lane_id: str | None
    lane_category: str | None
    dwell_frames: int
    lane_violation_frames: int
    wrong_way_frames: int
    stopped_frames: int
    max_recent_speed: float
    speed_drop: float
    bbox_width: float
    bbox_height: float

    def to_record(
        self,
        ganomaly_score: float = 0.0,
        active_anomalies: str = "",
        severity: str = "",
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        heading_x, heading_y = self.heading
        record = {
            "frame_idx": self.frame_idx,
            "timestamp_s": f"{self.timestamp_s:.3f}",
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_group": self.class_group,
            "class_id": self.class_id,
            "conf": f"{self.conf:.4f}",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": f"{self.center_px[0]:.4f}",
            "center_y": f"{self.center_px[1]:.4f}",
            "foot_x": f"{self.footpoint_px[0]:.4f}",
            "foot_y": f"{self.footpoint_px[1]:.4f}",
            "bev_x": f"{self.bev_point[0]:.4f}",
            "bev_y": f"{self.bev_point[1]:.4f}",
            "dx": f"{self.dx:.4f}",
            "dy": f"{self.dy:.4f}",
            "speed": f"{self.speed:.4f}",
            "acceleration": f"{self.acceleration:.4f}",
            "heading_x": f"{heading_x:.4f}",
            "heading_y": f"{heading_y:.4f}",
            "heading_alignment": f"{self.heading_alignment:.4f}",
            "lane_id": self.lane_id or "",
            "lane_category": self.lane_category or "",
            "dwell_frames": self.dwell_frames,
            "lane_violation_frames": self.lane_violation_frames,
            "wrong_way_frames": self.wrong_way_frames,
            "stopped_frames": self.stopped_frames,
            "max_recent_speed": f"{self.max_recent_speed:.4f}",
            "speed_drop": f"{self.speed_drop:.4f}",
            "bbox_width": f"{self.bbox_width:.4f}",
            "bbox_height": f"{self.bbox_height:.4f}",
            "ganomaly_score": f"{ganomaly_score:.4f}",
            "active_anomalies": active_anomalies,
            "severity": severity,
        }
        return record


@dataclass(frozen=True)
class RuleHit:
    anomaly_type: str
    anomaly_family: str
    rule_score: float
    severity: str
    explanation: str


@dataclass(frozen=True)
class ModelHit:
    frame_idx: int
    track_id: int
    class_name: str
    class_group: str
    score: float
    threshold: float
    is_anomalous: bool


@dataclass(frozen=True)
class FusedIncident:
    frame_idx: int
    track_id: int
    class_name: str
    anomaly_type: str
    anomaly_family: str
    severity: str
    rule_score: float
    ganomaly_score: float
    fused_score: float
    explanation: str
    lane_id: str
    bbox: tuple[int, int, int, int]
