from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .config import FeaturesConfig, FusionConfig
from .contracts import FusedIncident, ModelHit, RuleHit, TrackFeature


@dataclass
class _LaneSnapshot:
    active_count: int = 0
    slow_count: int = 0
    avg_speed: float = 0.0
    congested: bool = False


class FusionStage:
    def __init__(self, features: FeaturesConfig, fusion: FusionConfig, fps: float):
        self.features = features
        self.fusion = fusion
        self.fps = max(fps, 1.0)
        self.lane_violation_frames = max(1, int(round(features.lane_violation_seconds * self.fps)))
        self.wrong_way_frames = max(1, int(round(features.wrong_way_seconds * self.fps)))
        self.stopped_frames = max(1, int(round(features.stopped_vehicle_seconds * self.fps)))
        self.ganomaly_only_streaks: dict[int, int] = {}

    def evaluate(self, features: list[TrackFeature], ganomaly_hits: dict[int, ModelHit]) -> list[FusedIncident]:
        snapshots = self._build_lane_snapshots(features)

        incidents: list[FusedIncident] = []
        for feature in features:
            ganomaly_score = ganomaly_hits.get(feature.track_id).score if feature.track_id in ganomaly_hits else 0.0
            if ganomaly_score >= self.fusion.model_only_threshold:
                self.ganomaly_only_streaks[feature.track_id] = self.ganomaly_only_streaks.get(feature.track_id, 0) + 1
            else:
                self.ganomaly_only_streaks[feature.track_id] = 0
            rule_hits = self._rule_hits(feature, snapshots)
            if rule_hits:
                for rule_hit in rule_hits:
                    severity = rule_hit.severity
                    explanation = rule_hit.explanation
                    if severity == "warning" and ganomaly_score >= self.fusion.model_escalation_threshold:
                        severity = "critical"
                        explanation += f" GANomaly score {ganomaly_score:.2f} escalated the alert."
                    incidents.append(
                        FusedIncident(
                            frame_idx=feature.frame_idx,
                            track_id=feature.track_id,
                            class_name=feature.class_name,
                            anomaly_type=rule_hit.anomaly_type,
                            anomaly_family=rule_hit.anomaly_family,
                            severity=severity,
                            rule_score=rule_hit.rule_score,
                            ganomaly_score=ganomaly_score,
                            fused_score=max(rule_hit.rule_score, ganomaly_score),
                            explanation=explanation,
                            lane_id=feature.lane_id or "",
                            bbox=feature.bbox,
                        )
                    )
                continue

            if feature.track_id not in ganomaly_hits:
                continue
            if (
                ganomaly_score >= self.fusion.model_only_threshold
                and self.ganomaly_only_streaks.get(feature.track_id, 0) >= self.fusion.consecutive_frames_required
            ):
                incidents.append(
                    FusedIncident(
                        frame_idx=feature.frame_idx,
                        track_id=feature.track_id,
                        class_name=feature.class_name,
                        anomaly_type="appearance_anomaly",
                        anomaly_family="appearance_anomaly",
                        severity="warning",
                        rule_score=0.0,
                        ganomaly_score=ganomaly_score,
                        fused_score=ganomaly_score,
                        explanation=f"{feature.class_name} appearance deviated from the reviewed normal GANomaly baseline.",
                        lane_id=feature.lane_id or "",
                        bbox=feature.bbox,
                    )
                )
        return incidents

    def _build_lane_snapshots(self, features: list[TrackFeature]) -> dict[str, _LaneSnapshot]:
        snapshots: dict[str, _LaneSnapshot] = {}
        totals = defaultdict(float)
        for feature in features:
            if not feature.lane_id:
                continue
            snapshot = snapshots.setdefault(feature.lane_id, _LaneSnapshot())
            snapshot.active_count += 1
            totals[feature.lane_id] += feature.speed
            if feature.speed <= self.features.stopped_speed_threshold:
                snapshot.slow_count += 1

        for lane_id, snapshot in snapshots.items():
            snapshot.avg_speed = totals[lane_id] / max(snapshot.active_count, 1)
            slow_ratio = snapshot.slow_count / max(snapshot.active_count, 1)
            snapshot.congested = (
                snapshot.active_count >= self.features.congestion_min_active_tracks
                and slow_ratio >= self.features.congestion_slow_ratio
            )
        return snapshots

    def _rule_hits(self, feature: TrackFeature, snapshots: dict[str, _LaneSnapshot]) -> list[RuleHit]:
        hits: list[RuleHit] = []
        lane_snapshot = snapshots.get(feature.lane_id) if feature.lane_id else None
        lane_label = feature.lane_id or "unassigned lane"

        if feature.lane_violation_frames >= self.lane_violation_frames:
            duration = feature.lane_violation_frames / self.fps
            score = min(1.0, duration / max(self.features.lane_violation_seconds, 0.1))
            hits.append(
                RuleHit(
                    anomaly_type="lane_violation",
                    anomaly_family="lane_violation",
                    rule_score=score,
                    severity="warning",
                    explanation=f"{feature.class_name} remained inside forbidden lane {lane_label} for {duration:.1f}s.",
                )
            )

        if feature.wrong_way_frames >= self.wrong_way_frames:
            duration = feature.wrong_way_frames / self.fps
            alignment_penalty = min(1.0, abs(feature.heading_alignment))
            score = min(1.0, 0.5 * (duration / max(self.features.wrong_way_seconds, 0.1)) + 0.5 * alignment_penalty)
            hits.append(
                RuleHit(
                    anomaly_type="wrong_way",
                    anomaly_family="wrong_way",
                    rule_score=score,
                    severity="critical",
                    explanation=f"{feature.class_name} heading opposed lane direction in {lane_label} for {duration:.1f}s.",
                )
            )

        if (
            feature.lane_id
            and feature.speed <= self.features.stopped_speed_threshold
            and feature.max_recent_speed >= self.features.motion_floor
            and feature.speed_drop >= self.features.sudden_stop_delta
            and not (lane_snapshot and lane_snapshot.congested)
        ):
            score = min(1.0, feature.speed_drop / max(self.features.sudden_stop_delta, 0.1))
            hits.append(
                RuleHit(
                    anomaly_type="sudden_stop",
                    anomaly_family="sudden_stop",
                    rule_score=score,
                    severity="critical",
                    explanation=f"{feature.class_name} speed dropped by {feature.speed_drop:.1f} in lane {lane_label}.",
                )
            )

        if feature.lane_id and feature.stopped_frames >= self.stopped_frames:
            is_emergency_lane = feature.lane_category == "emergency"
            if is_emergency_lane or not (lane_snapshot and lane_snapshot.congested):
                duration = feature.stopped_frames / self.fps
                score = min(1.0, duration / max(self.features.stopped_vehicle_seconds, 0.1))
                severity = "critical" if is_emergency_lane else "warning"
                hits.append(
                    RuleHit(
                        anomaly_type="stopped_vehicle",
                        anomaly_family="stopped_vehicle",
                        rule_score=score,
                        severity=severity,
                        explanation=f"{feature.class_name} remained nearly stationary in {lane_label} for {duration:.1f}s.",
                    )
                )
        return hits
