from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import FeaturesConfig, LaneConfig
from .contracts import TrackFeature, TrackObservation, class_group_for_name


@dataclass
class _TrackState:
    last_frame_idx: int | None = None
    last_bev_point: tuple[float, float] | None = None
    speed_ema: float = 0.0
    accel_ema: float = 0.0
    dwell_frames: int = 0
    lane_violation_frames: int = 0
    wrong_way_frames: int = 0
    stopped_frames: int = 0
    heading_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    recent_speed_points: deque[tuple[int, float]] = field(default_factory=deque)


class FeatureStage:
    def __init__(self, config: FeaturesConfig, fps: float, state_ttl_frames: int):
        self.config = config
        self.fps = max(fps, 1.0)
        self.state_ttl_frames = max(state_ttl_frames, 1)
        smooth_window = max(config.smoothing_window, 1)
        self.ema_alpha = 2.0 / (smooth_window + 1.0)
        self.states: dict[int, _TrackState] = {}

    def update(self, frame_idx: int, observations: list[TrackObservation]) -> list[TrackFeature]:
        features: list[TrackFeature] = []
        active_ids = {obs.track_id for obs in observations}

        for observation in observations:
            features.append(self._update_track(frame_idx, observation))

        self._expire_missing(frame_idx, active_ids)
        return features

    def _update_track(self, frame_idx: int, observation: TrackObservation) -> TrackFeature:
        state = self.states.get(observation.track_id)
        if state is None:
            state = _TrackState()
            self.states[observation.track_id] = state

        frame_delta = 1 if state.last_frame_idx is None else max(frame_idx - state.last_frame_idx, 1)
        dt = frame_delta / self.fps
        x1, y1, x2, y2 = observation.bbox
        center_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        footpoint_px = ((x1 + x2) / 2.0, float(y2))
        lane = self._find_lane(footpoint_px)
        bev_point = self._project_point(footpoint_px)

        dx = 0.0
        dy = 0.0
        raw_speed = 0.0
        movement = np.zeros(2, dtype=np.float32)
        if state.last_bev_point is not None:
            dx = float(bev_point[0] - state.last_bev_point[0])
            dy = float(bev_point[1] - state.last_bev_point[1])
            movement = np.asarray([dx, dy], dtype=np.float32)
            raw_speed = float(np.linalg.norm(movement) / max(dt, 1e-6))

        previous_speed = state.speed_ema
        if state.speed_ema == 0.0:
            state.speed_ema = raw_speed
        else:
            state.speed_ema = self.ema_alpha * raw_speed + (1.0 - self.ema_alpha) * state.speed_ema

        raw_acceleration = 0.0 if state.last_frame_idx is None else (state.speed_ema - previous_speed) / max(dt, 1e-6)
        if state.accel_ema == 0.0:
            state.accel_ema = raw_acceleration
        else:
            state.accel_ema = self.ema_alpha * raw_acceleration + (1.0 - self.ema_alpha) * state.accel_ema

        heading = self._normalize(movement)
        if float(np.linalg.norm(heading)) > 0.0:
            state.heading_history.append(heading)
        if state.heading_history:
            heading = self._normalize(np.mean(np.stack(list(state.heading_history)), axis=0))

        alignment = self._heading_alignment(heading, lane)

        state.dwell_frames += frame_delta

        forbidden_lanes = set(self.config.class_lane_policy.get(observation.class_name, []))
        state.lane_violation_frames = state.lane_violation_frames + frame_delta if lane and lane.id in forbidden_lanes else 0
        state.wrong_way_frames = (
            state.wrong_way_frames + frame_delta
            if (
                lane is not None
                and state.speed_ema >= self.config.motion_floor
                and alignment <= self.config.wrong_way_alignment_threshold
                and state.dwell_frames >= self.config.min_track_age_frames
            )
            else 0
        )
        state.stopped_frames = (
            state.stopped_frames + frame_delta if state.speed_ema <= self.config.stopped_speed_threshold else 0
        )

        recent_window_frames = int(round(self.config.sudden_stop_window_seconds * self.fps))
        state.recent_speed_points.append((frame_idx, state.speed_ema))
        while state.recent_speed_points and frame_idx - state.recent_speed_points[0][0] > recent_window_frames:
            state.recent_speed_points.popleft()
        max_recent_speed = max((value for _, value in state.recent_speed_points), default=0.0)
        speed_drop = max(0.0, max_recent_speed - state.speed_ema)

        bbox_width = float(max(x2 - x1, 0))
        bbox_height = float(max(y2 - y1, 0))
        feature = TrackFeature(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps,
            track_id=observation.track_id,
            class_id=observation.class_id,
            class_name=observation.class_name,
            class_group=class_group_for_name(observation.class_name),
            conf=observation.conf,
            bbox=observation.bbox,
            center_px=center_px,
            footpoint_px=footpoint_px,
            bev_point=bev_point,
            dx=dx,
            dy=dy,
            speed=float(state.speed_ema),
            acceleration=float(state.accel_ema),
            heading=(float(heading[0]), float(heading[1])) if heading.size == 2 else (0.0, 0.0),
            heading_alignment=float(alignment),
            lane_id=lane.id if lane else None,
            lane_category=lane.category if lane else None,
            dwell_frames=state.dwell_frames,
            lane_violation_frames=state.lane_violation_frames,
            wrong_way_frames=state.wrong_way_frames,
            stopped_frames=state.stopped_frames,
            max_recent_speed=float(max_recent_speed),
            speed_drop=float(speed_drop),
            bbox_width=bbox_width,
            bbox_height=bbox_height,
        )
        state.last_frame_idx = frame_idx
        state.last_bev_point = bev_point
        return feature

    def _expire_missing(self, frame_idx: int, active_ids: set[int]) -> None:
        expired = [
            track_id
            for track_id, state in self.states.items()
            if track_id not in active_ids and state.last_frame_idx is not None and frame_idx - state.last_frame_idx > self.state_ttl_frames
        ]
        for track_id in expired:
            self.states.pop(track_id, None)

    def _find_lane(self, point: tuple[float, float]) -> LaneConfig | None:
        for lane in self.config.lanes:
            if cv2.pointPolygonTest(lane.polygon, point, False) >= 0:
                return lane
        return None

    def _project_point(self, point: tuple[float, float]) -> tuple[float, float]:
        vector = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(vector, self.config.homography)[0][0]
        return float(transformed[0]), float(transformed[1])

    @staticmethod
    def _normalize(vector) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm <= 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        return array / norm

    def _heading_alignment(self, heading: np.ndarray, lane: LaneConfig | None) -> float:
        if lane is None:
            return 0.0
        if float(np.linalg.norm(heading)) <= 1e-6:
            return 0.0
        return float(np.clip(np.dot(heading, lane.direction), -1.0, 1.0))
