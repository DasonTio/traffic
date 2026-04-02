from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .config import SceneConfig
from .tracklets import TrackFeature


@dataclass
class LaneSnapshot:
    active_count: int = 0
    slow_count: int = 0
    avg_speed: float = 0.0
    congested: bool = False


@dataclass
class RuleHit:
    anomaly_type: str
    rule_score: float
    severity: str
    explanation: str


def build_lane_snapshots(features: list[TrackFeature], scene: SceneConfig) -> dict[str, LaneSnapshot]:
    totals = defaultdict(float)
    snapshots: dict[str, LaneSnapshot] = {}
    stopped_speed = scene.thresholds.get("stopped_speed_threshold", 2.5)
    slow_ratio_limit = scene.thresholds.get("congestion_slow_ratio", 0.65)
    min_active = int(scene.thresholds.get("congestion_min_active_tracks", 3))

    for feature in features:
        if not feature.lane_id:
            continue
        snapshot = snapshots.setdefault(feature.lane_id, LaneSnapshot())
        snapshot.active_count += 1
        totals[feature.lane_id] += feature.speed
        if feature.speed <= stopped_speed:
            snapshot.slow_count += 1

    for lane_id, snapshot in snapshots.items():
        snapshot.avg_speed = totals[lane_id] / max(snapshot.active_count, 1)
        slow_fraction = snapshot.slow_count / max(snapshot.active_count, 1)
        snapshot.congested = snapshot.active_count >= min_active and slow_fraction >= slow_ratio_limit

    return snapshots


class RuleEngine:
    def __init__(self, scene: SceneConfig, fps: float):
        self.scene = scene
        self.fps = max(fps, 1.0)
        self.thresholds = scene.thresholds
        self.lane_violation_frames = max(1, int(round(self.thresholds.get("lane_violation_seconds", 0.5) * self.fps)))
        self.wrong_way_frames = max(1, int(round(self.thresholds.get("wrong_way_seconds", 1.0) * self.fps)))
        self.stopped_frames = max(1, int(round(self.thresholds.get("stopped_vehicle_seconds", 2.0) * self.fps)))

    def evaluate(self, feature: TrackFeature, lane_snapshots: dict[str, LaneSnapshot]) -> list[RuleHit]:
        hits: list[RuleHit] = []
        lane_snapshot = lane_snapshots.get(feature.lane_id) if feature.lane_id else None
        lane_label = feature.lane_id or "unassigned lane"

        if feature.lane_violation_frames >= self.lane_violation_frames:
            duration = feature.lane_violation_frames / self.fps
            score = min(1.0, duration / max(self.thresholds.get("lane_violation_seconds", 0.5), 0.1))
            hits.append(
                RuleHit(
                    anomaly_type="lane_violation",
                    rule_score=score,
                    severity="warning",
                    explanation=f"{feature.class_name} remained inside forbidden lane {lane_label} for {duration:.1f}s.",
                )
            )

        if feature.wrong_way_frames >= self.wrong_way_frames:
            duration = feature.wrong_way_frames / self.fps
            alignment_penalty = min(1.0, abs(feature.heading_alignment))
            score = min(1.0, 0.5 * (duration / max(self.thresholds.get("wrong_way_seconds", 1.0), 0.1)) + 0.5 * alignment_penalty)
            hits.append(
                RuleHit(
                    anomaly_type="wrong_way",
                    rule_score=score,
                    severity="critical",
                    explanation=f"{feature.class_name} heading opposed lane direction in {lane_label} for {duration:.1f}s.",
                )
            )

        if (
            feature.lane_id
            and feature.speed <= self.thresholds.get("stopped_speed_threshold", 2.5)
            and feature.max_recent_speed >= self.thresholds.get("motion_floor", 4.0)
            and feature.speed_drop >= self.thresholds.get("sudden_stop_delta", 6.0)
            and not (lane_snapshot and lane_snapshot.congested)
        ):
            score = min(1.0, feature.speed_drop / max(self.thresholds.get("sudden_stop_delta", 6.0), 0.1))
            hits.append(
                RuleHit(
                    anomaly_type="sudden_stop",
                    rule_score=score,
                    severity="critical",
                    explanation=f"{feature.class_name} speed dropped by {feature.speed_drop:.1f} in lane {lane_label}.",
                )
            )

        if feature.lane_id and feature.stopped_frames >= self.stopped_frames:
            is_emergency_lane = feature.lane_category == "emergency"
            if is_emergency_lane or not (lane_snapshot and lane_snapshot.congested):
                duration = feature.stopped_frames / self.fps
                score = min(1.0, duration / max(self.thresholds.get("stopped_vehicle_seconds", 2.0), 0.1))
                severity = "critical" if is_emergency_lane else "warning"
                hits.append(
                    RuleHit(
                        anomaly_type="stopped_vehicle",
                        rule_score=score,
                        severity=severity,
                        explanation=f"{feature.class_name} remained nearly stationary in {lane_label} for {duration:.1f}s.",
                    )
                )

        return hits
