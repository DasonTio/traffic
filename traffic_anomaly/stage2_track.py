from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from .config import TrackConfig
from .contracts import Detection, TrackObservation


class _TrackerDetections:
    def __init__(self, detections: list[Detection]):
        self._detections = detections
        if detections:
            self.xyxy = np.asarray([d.bbox for d in detections], dtype=np.float32)
            self.conf = np.asarray([d.conf for d in detections], dtype=np.float32)
            self.cls = np.asarray([d.class_id for d in detections], dtype=np.float32)
        else:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1 = self.xyxy[:, 0]
        y1 = self.xyxy[:, 1]
        x2 = self.xyxy[:, 2]
        y2 = self.xyxy[:, 3]
        centers_x = (x1 + x2) / 2.0
        centers_y = (y1 + y2) / 2.0
        widths = x2 - x1
        heights = y2 - y1
        return np.stack([centers_x, centers_y, widths, heights], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, item) -> "_TrackerDetections":
        clone = object.__new__(_TrackerDetections)
        clone._detections = self._detections
        clone.xyxy = np.asarray(self.xyxy[item], dtype=np.float32)
        clone.conf = np.asarray(self.conf[item], dtype=np.float32)
        clone.cls = np.asarray(self.cls[item], dtype=np.float32)
        if clone.xyxy.ndim == 1:
            clone.xyxy = clone.xyxy.reshape(1, -1)
        if clone.conf.ndim == 0:
            clone.conf = clone.conf.reshape(1)
        if clone.cls.ndim == 0:
            clone.cls = clone.cls.reshape(1)
        return clone


class TrackingStage:
    def __init__(self, config: TrackConfig, fps: float):
        tracker_args = SimpleNamespace(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            match_thresh=config.match_thresh,
            track_buffer=config.track_buffer,
            fuse_score=config.fuse_score,
        )
        self.tracker = BYTETracker(tracker_args, frame_rate=max(int(round(fps)), 1))

    def run(self, frame_idx: int, detections: list[Detection], frame) -> list[TrackObservation]:
        tracker_input = _TrackerDetections(detections)
        tracked = self.tracker.update(tracker_input, img=frame)
        observations: list[TrackObservation] = []

        for row in tracked:
            x1, y1, x2, y2, track_id, score, class_id, det_index = row.tolist()
            base = detections[int(det_index)]
            observations.append(
                TrackObservation(
                    frame_idx=frame_idx,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=base.class_name,
                    conf=float(score),
                    bbox=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                )
            )
        return observations

