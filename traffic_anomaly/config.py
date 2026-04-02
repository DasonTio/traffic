from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


COCO_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


def _resolve_path(base_dir: Path, value: str, prefer_parent: bool = False) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate

    search_roots = [
        base_dir,
        base_dir.parent,
        Path.cwd(),
    ]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    fallback_root = base_dir.parent if prefer_parent else base_dir
    return (fallback_root / candidate).resolve()


@dataclass(frozen=True)
class LaneConfig:
    id: str
    polygon: np.ndarray
    direction: np.ndarray
    category: str
    label: str


@dataclass
class SceneConfig:
    camera_id: str
    config_path: Path
    video_source: str
    video_resolution: str | None
    source_fps: float
    yolo_weights: Path
    confidence: float
    detect_classes: list[int]
    tracker_config: Path
    run_root: Path
    dataset_dir: Path
    homography: np.ndarray
    lanes: list[LaneConfig]
    lane_lookup: dict[str, LaneConfig]
    class_lane_policy: dict[str, list[str]]
    thresholds: dict[str, float]
    ganomaly_settings: dict[str, Any]
    ganomaly_checkpoints: dict[str, Path]
    coco_names: dict[int, str]

    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        config_path = Path(path).resolve()
        base_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        video_cfg = raw.get("video", {})
        model_cfg = raw.get("model", {})
        tracking_cfg = raw.get("tracking", {})
        output_cfg = raw.get("output", {})
        ganomaly_cfg = raw.get("ganomaly", {})

        src_points = np.asarray(raw["homography"]["src_points"], dtype=np.float32)
        dst_points = np.asarray(raw["homography"]["dst_points"], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(src_points, dst_points)

        lanes: list[LaneConfig] = []
        lane_lookup: dict[str, LaneConfig] = {}
        for lane in raw.get("lanes", []):
            direction = np.asarray(lane["direction"], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm > 0:
                direction = direction / norm
            compiled = LaneConfig(
                id=lane["id"],
                polygon=np.asarray(lane["polygon"], dtype=np.int32),
                direction=direction,
                category=lane.get("category", "active"),
                label=lane.get("label", lane["id"].replace("_", " ").title()),
            )
            lanes.append(compiled)
            lane_lookup[compiled.id] = compiled

        checkpoints = {
            name: _resolve_path(base_dir, value, prefer_parent=True)
            for name, value in ganomaly_cfg.get("checkpoints", {}).items()
        }

        return cls(
            camera_id=raw.get("camera_id", "camera_001"),
            config_path=config_path,
            video_source=video_cfg.get("source", ""),
            video_resolution=video_cfg.get("resolution"),
            source_fps=float(video_cfg.get("fps", raw.get("thresholds", {}).get("fps_fallback", 30.0))),
            yolo_weights=_resolve_path(base_dir, model_cfg.get("weights", "yolo11n.pt"), prefer_parent=True),
            confidence=float(model_cfg.get("confidence", 0.45)),
            detect_classes=list(model_cfg.get("detect_classes", [2, 5, 7])),
            tracker_config=_resolve_path(base_dir, tracking_cfg.get("tracker_config", "bytetrack.yaml")),
            run_root=_resolve_path(base_dir, output_cfg.get("run_root", "runs"), prefer_parent=True),
            dataset_dir=_resolve_path(base_dir, output_cfg.get("dataset_dir", "dataset"), prefer_parent=True),
            homography=homography,
            lanes=lanes,
            lane_lookup=lane_lookup,
            class_lane_policy={
                class_name: list(policy.get("forbidden_lanes", []))
                for class_name, policy in raw.get("class_lane_policy", {}).items()
            },
            thresholds={key: float(value) for key, value in raw.get("thresholds", {}).items()},
            ganomaly_settings={key: value for key, value in ganomaly_cfg.items() if key != "checkpoints"},
            ganomaly_checkpoints=checkpoints,
            coco_names={int(key): value for key, value in raw.get("coco_names", COCO_NAMES).items()},
        )

    def lane_for_id(self, lane_id: str | None) -> LaneConfig | None:
        if lane_id is None:
            return None
        return self.lane_lookup.get(lane_id)
