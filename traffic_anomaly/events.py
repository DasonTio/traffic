from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .storage import RunArtifacts
from .tracklets import TrackFeature


SEVERITY_RANK = {"warning": 1, "critical": 2}


@dataclass
class ActiveEvent:
    event_id: str
    camera_id: str
    track_id: int
    class_name: str
    anomaly_type: str
    severity: str
    start_frame: int
    end_frame: int
    last_seen_frame: int
    rule_score: float
    ganomaly_score: float
    fused_score: float
    lane_id: str
    explanation: str
    crop_path: str = ""
    frame_path: str = ""


class EventManager:
    def __init__(self, artifacts: RunArtifacts, camera_id: str, grace_frames: int = 5):
        self.artifacts = artifacts
        self.camera_id = camera_id
        self.grace_frames = grace_frames
        self.active_events: dict[tuple[int, str], ActiveEvent] = {}
        self.finalized_counts = defaultdict(int)

    def update_event(
        self,
        feature: TrackFeature,
        fused_hit: dict[str, float | str],
        frame,
        display,
        box: tuple[int, int, int, int],
    ) -> tuple[int, str]:
        key = (feature.track_id, str(fused_hit["anomaly_type"]))
        event = self.active_events.get(key)
        if event is None:
            event_id = f"evt_{self.artifacts.run_tag}_{feature.track_id}_{fused_hit['anomaly_type']}_{feature.frame_idx}"
            event = ActiveEvent(
                event_id=event_id,
                camera_id=self.camera_id,
                track_id=feature.track_id,
                class_name=feature.class_name,
                anomaly_type=str(fused_hit["anomaly_type"]),
                severity=str(fused_hit["severity"]),
                start_frame=feature.frame_idx,
                end_frame=feature.frame_idx,
                last_seen_frame=feature.frame_idx,
                rule_score=float(fused_hit["rule_score"]),
                ganomaly_score=float(fused_hit["ganomaly_score"]),
                fused_score=float(fused_hit["fused_score"]),
                lane_id=feature.lane_id or "",
                explanation=str(fused_hit["explanation"]),
            )
            self.active_events[key] = event
        else:
            event.end_frame = feature.frame_idx
            event.last_seen_frame = feature.frame_idx
            if SEVERITY_RANK.get(str(fused_hit["severity"]), 0) > SEVERITY_RANK.get(event.severity, 0):
                event.severity = str(fused_hit["severity"])
            event.rule_score = max(event.rule_score, float(fused_hit["rule_score"]))
            event.ganomaly_score = max(event.ganomaly_score, float(fused_hit["ganomaly_score"]))
            event.fused_score = max(event.fused_score, float(fused_hit["fused_score"]))
            event.explanation = str(fused_hit["explanation"])

        if float(fused_hit["fused_score"]) >= event.fused_score:
            evidence = self.artifacts.save_event_evidence(event.event_id, frame, display, box)
            event.crop_path = evidence["crop_path"]
            event.frame_path = evidence["frame_path"]
            event.fused_score = float(fused_hit["fused_score"])

        return key

    def finalize_inactive(self, active_keys: set[tuple[int, str]], frame_idx: int) -> None:
        expired_keys = [
            key
            for key, event in self.active_events.items()
            if key not in active_keys and frame_idx - event.last_seen_frame > self.grace_frames
        ]
        for key in expired_keys:
            self._finalize(key)

    def close_track(self, track_id: int) -> None:
        keys = [key for key in self.active_events if key[0] == track_id]
        for key in keys:
            self._finalize(key)

    def close_all(self) -> None:
        for key in list(self.active_events.keys()):
            self._finalize(key)

    def _finalize(self, key: tuple[int, str]) -> None:
        event = self.active_events.pop(key, None)
        if event is None:
            return
        self.artifacts.log_event(
            {
                "event_id": event.event_id,
                "camera_id": event.camera_id,
                "track_id": event.track_id,
                "class_name": event.class_name,
                "anomaly_type": event.anomaly_type,
                "severity": event.severity,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "rule_score": f"{event.rule_score:.4f}",
                "ganomaly_score": f"{event.ganomaly_score:.4f}",
                "fused_score": f"{event.fused_score:.4f}",
                "lane_id": event.lane_id,
                "explanation": event.explanation,
                "crop_path": event.crop_path,
                "frame_path": event.frame_path,
            }
        )
        self.finalized_counts[event.anomaly_type] += 1
