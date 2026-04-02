from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .config import LaneConfig


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(vector), dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return array / norm


def bottom_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def project_point(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    vector = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(vector, homography)[0][0]
    return float(transformed[0]), float(transformed[1])


def point_in_lane(point: tuple[float, float], lane: LaneConfig) -> bool:
    return cv2.pointPolygonTest(lane.polygon, point, False) >= 0


def find_lane(point: tuple[float, float], lanes: list[LaneConfig]) -> LaneConfig | None:
    for lane in lanes:
        if point_in_lane(point, lane):
            return lane
    return None


def heading_alignment(heading: np.ndarray, lane: LaneConfig | None) -> float:
    if lane is None:
        return 0.0
    heading = normalize_vector(heading)
    if float(np.linalg.norm(heading)) <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(heading, lane.direction), -1.0, 1.0))
