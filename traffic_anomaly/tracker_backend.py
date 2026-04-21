from __future__ import annotations

from difflib import get_close_matches
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TrackerSpec:
    tracker_type: str
    config_path: Path
    settings: dict[str, Any]
    detector_confidence: float


@dataclass(frozen=True)
class TrackedDetection:
    bbox: tuple[int, int, int, int]
    track_id: int
    class_id: int
    conf: float


TRACKER_ALIASES = {
    "byte": "bytetrack",
    "bytetrack": "bytetrack",
    "oc-sort": "ocsort",
    "ocsort": "ocsort",
}


def _normalize_tracker_type(value: str) -> str:
    tracker_type = value.strip().lower()
    return TRACKER_ALIASES.get(tracker_type, tracker_type)


def _flatten_tracker_settings(raw: dict[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "default" in value:
            settings[key] = value["default"]
        else:
            settings[key] = value
    return settings


def _detector_confidence(tracker_type: str, settings: dict[str, Any], default_confidence: float) -> float:
    if tracker_type == "bytetrack":
        low_conf = float(settings.get("min_conf", settings.get("track_low_thresh", 0.1)))
        return min(default_confidence, low_conf)
    if tracker_type == "ocsort":
        low_conf = float(settings.get("min_conf", settings.get("track_low_thresh", 0.1)))
        return min(default_confidence, low_conf)
    return default_confidence


def load_tracker_spec(config_path: str | Path, default_confidence: float) -> TrackerSpec:
    path = Path(config_path).resolve()
    if not path.exists():
        siblings = sorted(candidate.name for candidate in path.parent.glob("*.yaml")) if path.parent.exists() else []
        suggestion = get_close_matches(path.name, siblings, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        available = f" Available tracker configs: {', '.join(siblings)}." if siblings else ""
        raise FileNotFoundError(f"Tracker config not found: {path}.{hint}{available}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Tracker config at {path} must be a YAML mapping.")

    settings = _flatten_tracker_settings(raw)
    tracker_type = _normalize_tracker_type(str(settings.get("tracker_type", path.stem)))
    detector_confidence = _detector_confidence(tracker_type, settings, default_confidence)
    return TrackerSpec(
        tracker_type=tracker_type,
        config_path=path,
        settings=settings,
        detector_confidence=detector_confidence,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _results_to_boxmot_dets(result: Any) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = _to_numpy(getattr(boxes, "xyxy", None))
    conf = _to_numpy(getattr(boxes, "conf", None))
    cls = _to_numpy(getattr(boxes, "cls", None))
    if xyxy.size == 0 or conf.size == 0 or cls.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    conf = conf.reshape(-1, 1)
    cls = cls.reshape(-1, 1)
    return np.hstack([xyxy, conf, cls]).astype(np.float32, copy=False)


def _boxmot_outputs_to_tracked(outputs: np.ndarray) -> list[TrackedDetection]:
    if outputs.size == 0:
        return []
    tracks: list[TrackedDetection] = []
    for row in outputs:
        x1, y1, x2, y2, track_id, conf, class_id = row[:7]
        tracks.append(
            TrackedDetection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                track_id=int(track_id),
                class_id=int(class_id),
                conf=float(conf),
            )
        )
    return tracks


def _build_boxmot_tracker(spec: TrackerSpec, fps: float):
    try:
        from boxmot.trackers.bytetrack.bytetrack import ByteTrack
        from boxmot.trackers.ocsort.ocsort import OcSort
    except ImportError as exc:
        raise RuntimeError(
            "boxmot is required for tracker backends. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    settings = spec.settings
    frame_rate = max(1, int(round(fps)))

    if spec.tracker_type == "bytetrack":
        return ByteTrack(
            min_conf=float(settings.get("min_conf", settings.get("track_low_thresh", 0.1))),
            track_thresh=float(settings.get("track_thresh", settings.get("track_high_thresh", 0.45))),
            match_thresh=float(settings.get("match_thresh", 0.8)),
            track_buffer=int(settings.get("track_buffer", 30)),
            frame_rate=frame_rate,
            max_age=int(settings.get("max_age", settings.get("track_buffer", 30))),
            max_obs=int(settings.get("max_obs", max(int(settings.get("track_buffer", 30)) + 5, 50))),
            min_hits=int(settings.get("min_hits", 3)),
            iou_threshold=float(settings.get("iou_threshold", 0.3)),
            asso_func=str(settings.get("asso_func", "iou")),
        )

    if spec.tracker_type == "ocsort":
        return OcSort(
            det_thresh=float(settings.get("det_thresh", settings.get("track_high_thresh", 0.45))),
            min_conf=float(settings.get("min_conf", settings.get("track_low_thresh", 0.1))),
            max_age=int(settings.get("max_age", settings.get("track_buffer", 30))),
            max_obs=int(settings.get("max_obs", max(int(settings.get("max_age", settings.get("track_buffer", 30))) + 5, 50))),
            min_hits=int(settings.get("min_hits", 3)),
            iou_threshold=float(settings.get("iou_threshold", 0.3)),
            delta_t=int(settings.get("delta_t", 3)),
            asso_func=str(settings.get("asso_func", "iou")),
            use_byte=bool(settings.get("use_byte", False)),
            inertia=float(settings.get("inertia", 0.2)),
            Q_xy_scaling=float(settings.get("Q_xy_scaling", 0.01)),
            Q_s_scaling=float(settings.get("Q_s_scaling", 0.0001)),
        )

    raise ValueError(
        f"Unsupported tracker_type '{spec.tracker_type}'. "
        "Supported tracker types are: bytetrack, ocsort."
    )


class TrackerBackend:
    def __init__(
        self,
        model: Any,
        tracker_config: str | Path,
        detect_classes: list[int],
        default_confidence: float,
        fps: float,
    ):
        self.model = model
        self.detect_classes = detect_classes
        self.spec = load_tracker_spec(tracker_config, default_confidence)
        self.detector_confidence = self.spec.detector_confidence
        self.tracker = _build_boxmot_tracker(self.spec, fps=fps)

    @property
    def tracker_type(self) -> str:
        return self.spec.tracker_type

    def describe(self) -> str:
        return f"{self.spec.tracker_type} ({self.spec.config_path.name})"

    def track(self, frame: np.ndarray) -> list[TrackedDetection]:
        results = self.model.predict(
            frame,
            classes=self.detect_classes,
            conf=self.detector_confidence,
            verbose=False,
        )
        result = results[0] if results else None
        dets = _results_to_boxmot_dets(result)
        outputs = self.tracker.update(dets, frame)
        return _boxmot_outputs_to_tracked(outputs)
