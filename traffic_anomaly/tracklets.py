from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import SceneConfig
from .geometry import heading_alignment, normalize_vector


@dataclass
class TrackFeature:
    frame_idx: int
    timestamp_s: float
    track_id: int
    class_id: int
    class_name: str
    conf: float
    bbox: tuple[int, int, int, int]
    footpoint_px: tuple[float, float]
    footpoint_bev: tuple[float, float]
    lane_id: str | None
    lane_category: str | None
    speed: float
    acceleration: float
    heading: tuple[float, float]
    heading_alignment: float
    dwell_frames: int
    lane_violation_frames: int
    wrong_way_frames: int
    stopped_frames: int
    max_recent_speed: float
    speed_drop: float
    ganomaly_score: float

    def to_record(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        hx, hy = self.heading
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": f"{self.timestamp_s:.3f}",
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "conf": f"{self.conf:.4f}",
            "bbox": f"{x1},{y1},{x2},{y2}",
            "footpoint_px": f"{self.footpoint_px[0]:.2f},{self.footpoint_px[1]:.2f}",
            "footpoint_bev": f"{self.footpoint_bev[0]:.2f},{self.footpoint_bev[1]:.2f}",
            "lane_id": self.lane_id or "",
            "speed": f"{self.speed:.4f}",
            "acceleration": f"{self.acceleration:.4f}",
            "heading": f"{hx:.4f},{hy:.4f}",
            "heading_alignment": f"{self.heading_alignment:.4f}",
            "dwell_frames": self.dwell_frames,
            "stopped_frames": self.stopped_frames,
            "ganomaly_score": f"{self.ganomaly_score:.4f}",
        }


@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    speed_history: deque[float]
    heading_history: deque[np.ndarray]
    recent_speeds: deque[float]
    ganomaly_history: deque[float]
    last_bev_point: tuple[float, float] | None = None
    speed_ema: float = 0.0
    accel_ema: float = 0.0
    ganomaly_ema: float = 0.0
    dwell_frames: int = 0
    lane_violation_frames: int = 0
    wrong_way_frames: int = 0
    stopped_frames: int = 0
    last_lane_id: str | None = None


class TrackManager:
    def __init__(self, scene: SceneConfig, fps: float):
        self.scene = scene
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps
        self.states: dict[int, TrackState] = {}
        smooth_window = int(scene.thresholds.get("smoothing_window", 5))
        self.ema_alpha = 2.0 / (smooth_window + 1.0)
        speed_window = int(round(scene.thresholds.get("sudden_stop_window_seconds", 1.5) * self.fps))
        self.speed_window = max(speed_window, 2)
        self.ganomaly_window = int(scene.ganomaly_settings.get("aggregation_window", 20))
        self.ganomaly_alpha = float(scene.ganomaly_settings.get("ema_alpha", 0.3))

    def stale_ids(self, active_ids: set[int]) -> list[int]:
        return [track_id for track_id in list(self.states.keys()) if track_id not in active_ids]

    def remove(self, track_id: int) -> None:
        self.states.pop(track_id, None)

    def update(
        self,
        frame_idx: int,
        track_id: int,
        class_id: int,
        class_name: str,
        conf: float,
        bbox: tuple[int, int, int, int],
        footpoint_px: tuple[float, float],
        footpoint_bev: tuple[float, float],
        lane_id: str | None,
        lane_category: str | None,
        ganomaly_score: float,
    ) -> TrackFeature:
        state = self.states.get(track_id)
        if state is None:
            state = TrackState(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                speed_history=deque(maxlen=5),
                heading_history=deque(maxlen=5),
                recent_speeds=deque(maxlen=self.speed_window),
                ganomaly_history=deque(maxlen=self.ganomaly_window),
            )
            self.states[track_id] = state

        raw_speed = 0.0
        movement = np.zeros(2, dtype=np.float32)
        if state.last_bev_point is not None:
            movement = np.asarray(
                [
                    footpoint_bev[0] - state.last_bev_point[0],
                    footpoint_bev[1] - state.last_bev_point[1],
                ],
                dtype=np.float32,
            )
            raw_speed = float(np.linalg.norm(movement) / self.dt)

        if state.speed_ema == 0.0:
            state.speed_ema = raw_speed
        else:
            state.speed_ema = self.ema_alpha * raw_speed + (1.0 - self.ema_alpha) * state.speed_ema

        raw_acceleration = (state.speed_ema - state.speed_history[-1]) / self.dt if state.speed_history else 0.0
        if state.accel_ema == 0.0:
            state.accel_ema = raw_acceleration
        else:
            state.accel_ema = self.ema_alpha * raw_acceleration + (1.0 - self.ema_alpha) * state.accel_ema

        state.speed_history.append(state.speed_ema)
        state.recent_speeds.append(state.speed_ema)

        heading = normalize_vector(movement)
        if float(np.linalg.norm(heading)) > 0.0:
            state.heading_history.append(heading)
        if state.heading_history:
            heading = normalize_vector(np.mean(np.stack(list(state.heading_history)), axis=0))

        lane = self.scene.lane_for_id(lane_id)
        alignment = heading_alignment(heading, lane)

        state.dwell_frames = state.dwell_frames + 1 if lane_id and lane_id == state.last_lane_id else int(lane_id is not None)
        forbidden_lanes = set(self.scene.class_lane_policy.get(class_name, []))
        state.lane_violation_frames = state.lane_violation_frames + 1 if lane_id in forbidden_lanes else 0

        motion_floor = self.scene.thresholds.get("motion_floor", 4.0)
        wrong_way_limit = self.scene.thresholds.get("wrong_way_alignment_threshold", -0.25)
        state.wrong_way_frames = (
            state.wrong_way_frames + 1
            if lane_id and state.speed_ema >= motion_floor and alignment <= wrong_way_limit
            else 0
        )

        stopped_speed = self.scene.thresholds.get("stopped_speed_threshold", 2.5)
        state.stopped_frames = state.stopped_frames + 1 if state.speed_ema <= stopped_speed else 0

        state.ganomaly_history.append(ganomaly_score)
        if state.ganomaly_ema == 0.0:
            state.ganomaly_ema = ganomaly_score
        else:
            state.ganomaly_ema = self.ganomaly_alpha * ganomaly_score + (1.0 - self.ganomaly_alpha) * state.ganomaly_ema
        aggregated_ganomaly = max(
            state.ganomaly_ema,
            float(np.percentile(list(state.ganomaly_history), 90)) if state.ganomaly_history else 0.0,
        )

        max_recent_speed = max(state.recent_speeds) if state.recent_speeds else 0.0
        speed_drop = max(0.0, max_recent_speed - state.speed_ema)

        state.last_bev_point = footpoint_bev
        state.last_lane_id = lane_id

        return TrackFeature(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps,
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            conf=float(conf),
            bbox=bbox,
            footpoint_px=footpoint_px,
            footpoint_bev=footpoint_bev,
            lane_id=lane_id,
            lane_category=lane_category,
            speed=float(state.speed_ema),
            acceleration=float(state.accel_ema),
            heading=(float(heading[0]), float(heading[1])) if heading.size == 2 else (0.0, 0.0),
            heading_alignment=float(alignment),
            dwell_frames=state.dwell_frames,
            lane_violation_frames=state.lane_violation_frames,
            wrong_way_frames=state.wrong_way_frames,
            stopped_frames=state.stopped_frames,
            max_recent_speed=float(max_recent_speed),
            speed_drop=float(speed_drop),
            ganomaly_score=float(aggregated_ganomaly),
        )
