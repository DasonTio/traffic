from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch


APPROVED_REVIEW_STATUSES = {"approved", "usable", "yes", "true"}
APPEARANCE_CLASS_GROUPS = ("car", "bus", "truck")


@dataclass(frozen=True)
class AppearanceScore:
    model_name: str
    class_name: str
    class_group: str
    raw_score: float
    threshold: float
    normalized_score: float

    @property
    def is_anomaly(self) -> bool:
        return self.normalized_score >= 1.0


class AppearanceScorer(Protocol):
    def available(self) -> bool: ...

    def score_crop(self, crop: np.ndarray, class_name: str) -> float: ...

    def score_crop_details(self, crop: np.ndarray, class_name: str) -> AppearanceScore: ...


def class_group_for_name(class_name: str) -> str:
    normalized = class_name.strip().lower()
    if normalized in APPEARANCE_CLASS_GROUPS:
        return normalized
    return normalized or "unknown"


def preprocess_crop(crop: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 127.5 - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def approved_frame_paths(dataset_dir: Path, class_group: str) -> list[Path]:
    review_path = dataset_dir / "sequence_review.csv"
    if not review_path.exists():
        return []

    frame_paths: list[Path] = []
    with review_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = row.get("review_status", "").strip().lower()
            class_name = row.get("class_name", "Unknown")
            if status not in APPROVED_REVIEW_STATUSES:
                continue
            if class_group_for_name(class_name) != class_group:
                continue
            seq_dir_name = Path(row["sequence_path"]).name
            sequence_path = dataset_dir / "sequences" / seq_dir_name
            frame_paths.extend(sorted(sequence_path.glob("*.jpg")))
    return frame_paths


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image at {path}")
    return image
