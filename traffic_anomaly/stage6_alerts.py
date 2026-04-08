from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from .config import AlertsConfig, AnomalyConfig, FusionConfig
from .contracts import CAR_CLASS_NAME, FusedIncident, ModelHit, TrackFeature


TRACKLET_FIELDS = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "class_name",
    "class_group",
    "class_id",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "foot_x",
    "foot_y",
    "bev_x",
    "bev_y",
    "dx",
    "dy",
    "speed",
    "acceleration",
    "heading_x",
    "heading_y",
    "heading_alignment",
    "lane_id",
    "lane_category",
    "dwell_frames",
    "lane_violation_frames",
    "wrong_way_frames",
    "stopped_frames",
    "max_recent_speed",
    "speed_drop",
    "bbox_width",
    "bbox_height",
    "ganomaly_score",
    "active_anomalies",
    "severity",
]

INCIDENT_FIELDS = [
    "incident_id",
    "track_id",
    "class_name",
    "anomaly_type",
    "anomaly_family",
    "severity",
    "start_frame",
    "end_frame",
    "rule_score",
    "ganomaly_score",
    "fused_score",
    "lane_id",
    "explanation",
    "frame_path",
    "crop_path",
]

REVIEW_MANIFEST_FIELDS = [
    "sample_id",
    "run_tag",
    "frame_idx",
    "track_id",
    "class_name",
    "status",
    "crop_path",
    "crop_width",
    "crop_height",
    "lane_id",
    "source",
]


def _open_writer(path: Path, fieldnames: list[str]) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def _open_append_writer(path: Path, fieldnames: list[str]) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return handle, writer


@dataclass
class _ActiveIncident:
    incident_id: str
    track_id: int
    class_name: str
    anomaly_type: str
    anomaly_family: str
    severity: str
    start_frame: int
    end_frame: int
    last_seen_frame: int
    rule_score: float
    ganomaly_score: float
    fused_score: float
    lane_id: str
    explanation: str
    bbox: tuple[int, int, int, int]
    frame_path: str = ""
    crop_path: str = ""


class AlertStage:
    def __init__(
        self,
        alerts: AlertsConfig,
        fusion: FusionConfig,
        anomaly: AnomalyConfig,
        camera_id: str,
        ganomaly_ready: bool,
    ):
        self.alerts = alerts
        self.fusion = fusion
        self.anomaly = anomaly
        self.camera_id = camera_id
        self.ganomaly_ready = ganomaly_ready
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = alerts.run_root / self.run_tag
        self.frames_dir = self.root / "frames"
        self.crops_dir = self.root / "crops"
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.tracklet_handle, self.tracklet_writer = _open_writer(self.root / "tracklets.csv", TRACKLET_FIELDS)
        self.incident_handle, self.incident_writer = _open_writer(self.root / "incidents.csv", INCIDENT_FIELDS)

        self.review_manifest_path: Path | None = None
        self.candidate_dir: Path | None = None
        self.review_handle: object | None = None
        self.review_writer: csv.DictWriter | None = None
        if anomaly.save_training_candidates:
            self.candidate_dir = anomaly.dataset_root / "candidates"
            self.review_manifest_path = anomaly.dataset_root / "review_manifest.csv"
            self.candidate_dir.mkdir(parents=True, exist_ok=True)
            self.review_handle, self.review_writer = _open_append_writer(
                self.review_manifest_path,
                REVIEW_MANIFEST_FIELDS,
            )

        self.active_incidents: dict[tuple[int, str], _ActiveIncident] = {}
        self.finalized_counts: Counter[str] = Counter()
        self.tracklet_rows = 0
        self.incident_rows = 0
        self.candidate_rows = 0

    def record_frame(
        self,
        frame_idx: int,
        features: list[TrackFeature],
        ganomaly_hits: dict[int, ModelHit],
        incidents: list[FusedIncident],
        frame,
    ) -> None:
        incident_map: dict[int, list[FusedIncident]] = {}
        for incident in incidents:
            incident_map.setdefault(incident.track_id, []).append(incident)

        for feature in features:
            current = incident_map.get(feature.track_id, [])
            highest_severity = self._highest_severity([incident.severity for incident in current])
            ganomaly_score = ganomaly_hits.get(feature.track_id).score if feature.track_id in ganomaly_hits else 0.0
            active_anomalies = "|".join(incident.anomaly_type for incident in current)
            self.tracklet_writer.writerow(feature.to_record(ganomaly_score, active_anomalies, highest_severity))
            self.tracklet_rows += 1
            if self.review_writer is not None and not current:
                self._record_training_candidate(feature, frame)

        current_keys: set[tuple[int, str]] = set()
        for incident in incidents:
            key = (incident.track_id, incident.anomaly_family)
            current_keys.add(key)
            active = self.active_incidents.get(key)
            if active is None:
                incident_id = f"inc_{self.run_tag}_{incident.track_id}_{incident.anomaly_family}_{incident.frame_idx}"
                active = _ActiveIncident(
                    incident_id=incident_id,
                    track_id=incident.track_id,
                    class_name=incident.class_name,
                    anomaly_type=incident.anomaly_type,
                    anomaly_family=incident.anomaly_family,
                    severity=incident.severity,
                    start_frame=incident.frame_idx,
                    end_frame=incident.frame_idx,
                    last_seen_frame=incident.frame_idx,
                    rule_score=incident.rule_score,
                    ganomaly_score=incident.ganomaly_score,
                    fused_score=incident.fused_score,
                    lane_id=incident.lane_id,
                    explanation=incident.explanation,
                    bbox=incident.bbox,
                )
                self.active_incidents[key] = active
            else:
                active.end_frame = incident.frame_idx
                active.last_seen_frame = incident.frame_idx
                active.severity = self._highest_severity([active.severity, incident.severity])
                active.rule_score = max(active.rule_score, incident.rule_score)
                active.ganomaly_score = max(active.ganomaly_score, incident.ganomaly_score)
                active.fused_score = max(active.fused_score, incident.fused_score)
                active.explanation = incident.explanation
                active.bbox = incident.bbox

            if incident.fused_score >= active.fused_score:
                frame_path, crop_path = self._save_evidence(active.incident_id, frame, incident.bbox)
                active.frame_path = frame_path
                active.crop_path = crop_path
                active.fused_score = incident.fused_score

        self._finalize_inactive(frame_idx, current_keys)

    def close(self) -> None:
        for key in list(self.active_incidents):
            self._finalize(key)
        self.tracklet_handle.close()
        self.incident_handle.close()
        if self.review_handle is not None:
            self.review_handle.close()

        summary = {
            "camera_id": self.camera_id,
            "run_tag": self.run_tag,
            "ganomaly_ready": self.ganomaly_ready,
            "tracklet_rows": self.tracklet_rows,
            "incident_rows": self.incident_rows,
            "candidate_rows": self.candidate_rows,
            "incident_counts": dict(self.finalized_counts),
            "tracklets_csv": str(self.root / "tracklets.csv"),
            "incidents_csv": str(self.root / "incidents.csv"),
            "ganomaly_dataset_root": str(self.anomaly.dataset_root),
            "review_manifest_csv": str(self.review_manifest_path) if self.review_manifest_path else "",
        }
        with (self.root / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _record_training_candidate(self, feature: TrackFeature, frame) -> None:
        if feature.class_name != CAR_CLASS_NAME or self.review_writer is None or self.candidate_dir is None:
            return

        crop = self._crop_region(frame, feature.bbox)
        if crop is None:
            return

        height, width = crop.shape[:2]
        if width < self.anomaly.min_crop_width or height < self.anomaly.min_crop_height:
            return

        sample_id = f"{self.run_tag}_f{feature.frame_idx}_t{feature.track_id}"
        crop_path = (self.candidate_dir / f"{sample_id}.jpg").resolve()
        cv2.imwrite(str(crop_path), crop)
        self.review_writer.writerow(
            {
                "sample_id": sample_id,
                "run_tag": self.run_tag,
                "frame_idx": feature.frame_idx,
                "track_id": feature.track_id,
                "class_name": feature.class_name,
                "status": "pending",
                "crop_path": str(crop_path),
                "crop_width": width,
                "crop_height": height,
                "lane_id": feature.lane_id or "",
                "source": "candidate_normal",
            }
        )
        self.candidate_rows += 1

    def _save_evidence(self, incident_id: str, frame, bbox: tuple[int, int, int, int]) -> tuple[str, str]:
        crop = self._crop_region(frame, bbox)
        frame_path = self.frames_dir / f"{incident_id}.jpg"
        crop_path = self.crops_dir / f"{incident_id}.jpg"
        cv2.imwrite(str(frame_path), frame)
        if crop is not None and crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        return str(frame_path), str(crop_path)

    def _crop_region(self, frame, bbox: tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        pad = self.anomaly.crop_padding
        height, width = frame.shape[:2]
        crop = frame[
            max(0, y1 - pad) : min(height, y2 + pad),
            max(0, x1 - pad) : min(width, x2 + pad),
        ]
        if crop.size == 0:
            return None
        return crop

    def _finalize_inactive(self, frame_idx: int, current_keys: set[tuple[int, str]]) -> None:
        expired = [
            key
            for key, incident in self.active_incidents.items()
            if key not in current_keys and frame_idx - incident.last_seen_frame > self.fusion.merge_gap_frames
        ]
        for key in expired:
            self._finalize(key)

    def _finalize(self, key: tuple[int, str]) -> None:
        incident = self.active_incidents.pop(key, None)
        if incident is None:
            return
        self.incident_writer.writerow(
            {
                "incident_id": incident.incident_id,
                "track_id": incident.track_id,
                "class_name": incident.class_name,
                "anomaly_type": incident.anomaly_type,
                "anomaly_family": incident.anomaly_family,
                "severity": incident.severity,
                "start_frame": incident.start_frame,
                "end_frame": incident.end_frame,
                "rule_score": f"{incident.rule_score:.4f}",
                "ganomaly_score": f"{incident.ganomaly_score:.4f}",
                "fused_score": f"{incident.fused_score:.4f}",
                "lane_id": incident.lane_id,
                "explanation": incident.explanation,
                "frame_path": incident.frame_path,
                "crop_path": incident.crop_path,
            }
        )
        self.finalized_counts[incident.anomaly_type] += 1
        self.incident_rows += 1

    @staticmethod
    def _highest_severity(values: list[str]) -> str:
        if "critical" in values:
            return "critical"
        if "warning" in values:
            return "warning"
        return ""
