import csv
from pathlib import Path

import cv2
import numpy as np

from traffic_anomaly.appearance import AppearanceScore, class_group_for_name
from traffic_anomaly.appearance_eval import evaluate_appearance_model, write_appearance_outputs


class StubScorer:
    def available(self) -> bool:
        return True

    def score_crop(self, crop: np.ndarray, class_name: str) -> float:
        return self.score_crop_details(crop, class_name).normalized_score

    def score_crop_details(self, crop: np.ndarray, class_name: str) -> AppearanceScore:
        normalized_score = 1.5 if float(crop.mean()) > 127 else 0.3
        return AppearanceScore(
            model_name="stub",
            class_name=class_name,
            class_group=class_group_for_name(class_name),
            raw_score=normalized_score,
            threshold=1.0,
            normalized_score=normalized_score,
        )


def _write_image(path: Path, value: int) -> None:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_evaluate_appearance_model_writes_outputs(tmp_path):
    gt_path = tmp_path / "appearance_ground_truth.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = [
        ("normal_car", image_dir / "normal_car.jpg", "Car", "car", "normal"),
        ("anomaly_car", image_dir / "anomaly_car.jpg", "Car", "car", "appearance_anomaly"),
        ("normal_truck", image_dir / "normal_truck.jpg", "Truck", "truck", "normal"),
        ("anomaly_truck", image_dir / "anomaly_truck.jpg", "Truck", "truck", "appearance_anomaly"),
    ]
    _write_image(rows[0][1], 20)
    _write_image(rows[1][1], 240)
    _write_image(rows[2][1], 30)
    _write_image(rows[3][1], 220)

    with gt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "image_path",
                "class_name",
                "class_group",
                "label",
                "source_type",
                "source_run",
                "event_id",
                "track_id",
                "frame_idx",
                "notes",
            ],
        )
        writer.writeheader()
        for sample_id, image_path, class_name, class_group, label in rows:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "class_name": class_name,
                    "class_group": class_group,
                    "label": label,
                    "source_type": "test",
                    "source_run": "",
                    "event_id": "",
                    "track_id": "",
                    "frame_idx": "",
                    "notes": "",
                }
            )

    summary = evaluate_appearance_model(StubScorer(), gt_path)
    output_dir = write_appearance_outputs(summary, tmp_path / "reports")

    assert summary.model_name == "stub"
    assert summary.metrics.total == 4
    assert summary.metrics.precision == 1.0
    assert summary.metrics.recall == 1.0
    assert summary.metrics.f1 == 1.0
    assert summary.per_group["car"].precision == 1.0
    assert summary.per_group["truck"].recall == 1.0
    assert (output_dir / "stub_predictions.csv").exists()
    assert (output_dir / "stub_summary.json").exists()
    assert (output_dir / "stub_report.md").exists()
