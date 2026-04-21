from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


TRACKLET_FIELDS = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "class_name",
    "class_id",
    "raw_class_name",
    "raw_class_id",
    "stable_class_score",
    "conf",
    "bbox",
    "footpoint_px",
    "footpoint_bev",
    "lane_id",
    "speed",
    "acceleration",
    "heading",
    "heading_alignment",
    "dwell_frames",
    "stopped_frames",
    "ganomaly_score",
    "rule_hits",
    "severity",
]

EVENT_FIELDS = [
    "event_id",
    "camera_id",
    "track_id",
    "class_name",
    "anomaly_type",
    "severity",
    "start_frame",
    "end_frame",
    "rule_score",
    "ganomaly_score",
    "fused_score",
    "lane_id",
    "explanation",
    "crop_path",
    "frame_path",
]

NORMAL_FIELDS = [
    "timestamp",
    "sequence_id",
    "track_id",
    "class_name",
    "first_frame",
    "last_frame",
    "num_frames",
    "confidence",
    "sequence_path",
    "label",
    "source_run",
]

SEQUENCE_MANIFEST_FIELDS = NORMAL_FIELDS + ["review_status", "review_notes"]
REVIEW_FIELDS = ["sequence_id", "class_name", "num_frames", "sequence_path", "review_status", "review_notes", "last_updated"]


def _open_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not file_exists or path.stat().st_size == 0:
        writer.writeheader()
    return handle, writer


def _load_rows(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row[key_field]: row for row in reader if row.get(key_field)}


def _pick_value(*values: str, default: str = "") -> str:
    for value in values:
        if value and value != "Unknown":
            return value
    return default


def sync_sequence_manifests(dataset_dir: Path) -> None:
    sequence_dir = dataset_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "sequence_manifest.csv"
    review_path = dataset_dir / "sequence_review.csv"
    legacy_path = dataset_dir / "normal_sequences.csv"

    existing_manifest = _load_rows(manifest_path, "sequence_id")
    existing_review = _load_rows(review_path, "sequence_id")
    legacy_rows: dict[str, dict[str, str]] = {}
    if legacy_path.exists():
        with legacy_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sequence_id = row.get("Sequence_ID") or row.get("sequence_id")
                if sequence_id:
                    legacy_rows[sequence_id] = row

    merged_manifest: list[dict[str, str]] = []
    merged_review: list[dict[str, str]] = []
    for sequence_path in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
        sequence_id = sequence_path.name
        num_frames = str(len(sorted(sequence_path.glob("*.jpg"))))
        manifest_row = existing_manifest.get(sequence_id, {})
        legacy_row = legacy_rows.get(sequence_id, {})
        merged = {
            "timestamp": _pick_value(manifest_row.get("timestamp", ""), legacy_row.get("Timestamp", ""), default=""),
            "sequence_id": sequence_id,
            "track_id": _pick_value(manifest_row.get("track_id", ""), legacy_row.get("Track_ID", ""), default=""),
            "class_name": _pick_value(manifest_row.get("class_name", ""), legacy_row.get("Class", ""), default="Unknown"),
            "first_frame": _pick_value(manifest_row.get("first_frame", ""), legacy_row.get("First_Frame", ""), default=""),
            "last_frame": _pick_value(manifest_row.get("last_frame", ""), legacy_row.get("Last_Frame", ""), default=""),
            "num_frames": num_frames,
            "confidence": _pick_value(manifest_row.get("confidence", ""), legacy_row.get("Confidence", ""), default=""),
            "sequence_path": str(sequence_path),
            "label": _pick_value(manifest_row.get("label", ""), legacy_row.get("Label", ""), default="normal"),
            "source_run": _pick_value(manifest_row.get("source_run", ""), default="legacy"),
            "review_status": _pick_value(
                manifest_row.get("review_status", ""),
                existing_review.get(sequence_id, {}).get("review_status", ""),
                default="pending",
            ),
            "review_notes": _pick_value(
                manifest_row.get("review_notes", ""),
                existing_review.get(sequence_id, {}).get("review_notes", ""),
                default="",
            ),
        }
        merged_manifest.append(merged)

        review_row = existing_review.get(sequence_id, {})
        merged_review.append(
            {
                "sequence_id": sequence_id,
                "class_name": merged["class_name"],
                "num_frames": num_frames,
                "sequence_path": str(sequence_path),
                "review_status": review_row.get("review_status", merged["review_status"]),
                "review_notes": review_row.get("review_notes", merged["review_notes"]),
                "last_updated": review_row.get("last_updated", ""),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEQUENCE_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(merged_manifest)

    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(merged_review)


@dataclass
class RunArtifacts:
    run_root: Path
    dataset_dir: Path
    min_sequence_frames: int

    def __post_init__(self) -> None:
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = self.run_root / self.run_tag
        self.frames_dir = self.root / "frames"
        self.crops_dir = self.root / "crops"
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "sequences").mkdir(parents=True, exist_ok=True)
        self.tmp_sequence_dir = self.dataset_dir / f"_tmp_sequences_{self.run_tag}"
        self.tmp_sequence_dir.mkdir(parents=True, exist_ok=True)
        sync_sequence_manifests(self.dataset_dir)

        self.tracklet_handle, self.tracklet_writer = _open_writer(self.root / "tracklets.csv", TRACKLET_FIELDS)
        self.event_handle, self.event_writer = _open_writer(self.root / "events.csv", EVENT_FIELDS)
        self.normal_handle, self.normal_writer = _open_writer(self.root / "normal_sequences.csv", NORMAL_FIELDS)

    def log_tracklet(self, record: dict[str, Any]) -> None:
        self.tracklet_writer.writerow(record)
        self.tracklet_handle.flush()

    def log_event(self, record: dict[str, Any]) -> None:
        self.event_writer.writerow(record)
        self.event_handle.flush()

    def log_normal_sequence(self, record: dict[str, Any]) -> None:
        self.normal_writer.writerow(record)
        self.normal_handle.flush()
        self._upsert_sequence_manifest(record)

    def create_sequence_candidate(self, track_id: int, class_name: str, frame_idx: int) -> dict[str, Any]:
        tag = f"track_{track_id}_{frame_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp_dir = self.tmp_sequence_dir / tag
        temp_dir.mkdir(parents=True, exist_ok=True)
        return {
            "track_id": track_id,
            "class_name": class_name,
            "first_frame": frame_idx,
            "last_frame": frame_idx,
            "max_conf": 0.0,
            "num_frames": 0,
            "temp_dir": temp_dir,
        }

    def append_sequence_frame(
        self,
        candidate: dict[str, Any],
        crop,
        frame_idx: int,
        conf: float,
        class_name: str,
    ) -> None:
        candidate["last_frame"] = frame_idx
        candidate["class_name"] = class_name
        candidate["max_conf"] = max(candidate["max_conf"], float(conf))
        next_index = candidate["num_frames"] + 1
        frame_path = candidate["temp_dir"] / f"{next_index:06d}.jpg"
        if cv2.imwrite(str(frame_path), crop):
            candidate["num_frames"] = next_index

    def discard_sequence_candidate(self, track_id: int, candidates: dict[int, dict[str, Any]]) -> None:
        candidate = candidates.pop(track_id, None)
        if candidate is not None:
            shutil.rmtree(candidate["temp_dir"], ignore_errors=True)

    def finalize_sequence_candidate(
        self,
        track_id: int,
        candidates: dict[int, dict[str, Any]],
    ) -> dict[str, Any] | None:
        candidate = candidates.pop(track_id, None)
        if candidate is None:
            return None
        if candidate["num_frames"] < self.min_sequence_frames:
            shutil.rmtree(candidate["temp_dir"], ignore_errors=True)
            return None

        sequence_id = (
            f"seq_{self.run_tag}_{candidate['track_id']}_"
            f"{candidate['first_frame']}_{candidate['last_frame']}"
        )
        final_dir = self.dataset_dir / "sequences" / sequence_id
        suffix = 0
        while final_dir.exists():
            suffix += 1
            final_dir = self.dataset_dir / "sequences" / f"{sequence_id}_{suffix:02d}"
        shutil.move(str(candidate["temp_dir"]), str(final_dir))
        sequence_id = final_dir.name

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sequence_id": sequence_id,
            "track_id": candidate["track_id"],
            "class_name": candidate["class_name"],
            "first_frame": candidate["first_frame"],
            "last_frame": candidate["last_frame"],
            "num_frames": candidate["num_frames"],
            "confidence": f"{candidate['max_conf']:.4f}",
            "sequence_path": str(final_dir),
            "label": "normal",
            "source_run": self.run_tag,
        }
        self.log_normal_sequence(record)
        return record

    def save_event_evidence(
        self,
        event_id: str,
        frame,
        display,
        box: tuple[int, int, int, int],
    ) -> dict[str, str]:
        x1, y1, x2, y2 = box
        pad = 15
        height, width = frame.shape[:2]
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(width, x2 + pad)
        cy2 = min(height, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        frame_path = self.frames_dir / f"{event_id}.jpg"
        crop_path = self.crops_dir / f"{event_id}.jpg"
        cv2.imwrite(str(frame_path), display)
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        return {"frame_path": str(frame_path), "crop_path": str(crop_path)}

    def close(self) -> None:
        self.tracklet_handle.close()
        self.event_handle.close()
        self.normal_handle.close()
        shutil.rmtree(self.tmp_sequence_dir, ignore_errors=True)
        sync_sequence_manifests(self.dataset_dir)

    def _upsert_sequence_manifest(self, record: dict[str, Any]) -> None:
        manifest_path = self.dataset_dir / "sequence_manifest.csv"
        review_path = self.dataset_dir / "sequence_review.csv"
        manifest_rows = _load_rows(manifest_path, "sequence_id")
        review_rows = _load_rows(review_path, "sequence_id")

        manifest_rows[record["sequence_id"]] = {
            **{field: str(record.get(field, "")) for field in NORMAL_FIELDS},
            "review_status": review_rows.get(record["sequence_id"], {}).get("review_status", "pending"),
            "review_notes": review_rows.get(record["sequence_id"], {}).get("review_notes", ""),
        }

        review_rows.setdefault(
            record["sequence_id"],
            {
                "sequence_id": str(record["sequence_id"]),
                "class_name": str(record["class_name"]),
                "num_frames": str(record["num_frames"]),
                "sequence_path": str(record["sequence_path"]),
                "review_status": "pending",
                "review_notes": "",
                "last_updated": "",
            },
        )

        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SEQUENCE_MANIFEST_FIELDS)
            writer.writeheader()
            for sequence_id in sorted(manifest_rows):
                writer.writerow(manifest_rows[sequence_id])

        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for sequence_id in sorted(review_rows):
                writer.writerow(review_rows[sequence_id])
