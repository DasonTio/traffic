from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.appearance import APPROVED_REVIEW_STATUSES, class_group_for_name


APPEARANCE_GT_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or refresh the appearance ground-truth CSV from approved normal sequences and event crops."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset directory containing sequence review files.")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing pipeline run folders.")
    parser.add_argument("--output", default="dataset/appearance_ground_truth.csv", help="Output CSV path.")
    parser.add_argument(
        "--frames-per-sequence",
        type=int,
        default=3,
        help="How many frames to seed from each approved normal sequence.",
    )
    return parser.parse_args()


def _load_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("sample_id")}


def _load_manifest_rows(dataset_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = dataset_dir / "sequence_manifest.csv"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["sequence_id"]: row for row in csv.DictReader(handle) if row.get("sequence_id")}


def _pick_frame_samples(frame_paths: list[Path], frames_per_sequence: int) -> list[Path]:
    if frames_per_sequence <= 0 or len(frame_paths) <= frames_per_sequence:
        return frame_paths
    if frames_per_sequence == 1:
        return [frame_paths[len(frame_paths) // 2]]

    selected: list[Path] = []
    max_index = len(frame_paths) - 1
    for idx in range(frames_per_sequence):
        position = round(idx * max_index / (frames_per_sequence - 1))
        candidate = frame_paths[position]
        if candidate not in selected:
            selected.append(candidate)
    return selected


def _seed_normal_rows(
    dataset_dir: Path,
    frames_per_sequence: int,
    existing: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    review_path = dataset_dir / "sequence_review.csv"
    if not review_path.exists():
        return []
    manifest_rows = _load_manifest_rows(dataset_dir)
    seeded: list[dict[str, str]] = []

    with review_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = row.get("review_status", "").strip().lower()
            if status not in APPROVED_REVIEW_STATUSES:
                continue
            sequence_path = Path(row.get("sequence_path", ""))
            if not sequence_path.exists():
                continue
            sequence_id = row.get("sequence_id", sequence_path.name)
            manifest_row = manifest_rows.get(sequence_id, {})
            class_name = row.get("class_name", manifest_row.get("class_name", "Unknown"))
            frame_paths = sorted(sequence_path.glob("*.jpg"))
            for frame_path in _pick_frame_samples(frame_paths, frames_per_sequence):
                sample_id = f"normal::{sequence_id}::{frame_path.name}"
                base = {
                    "sample_id": sample_id,
                    "image_path": str(frame_path),
                    "class_name": class_name,
                    "class_group": class_group_for_name(class_name),
                    "label": "normal",
                    "source_type": "approved_sequence",
                    "source_run": manifest_row.get("source_run", ""),
                    "event_id": "",
                    "track_id": manifest_row.get("track_id", ""),
                    "frame_idx": "",
                    "notes": "Seeded from approved normal sequence",
                }
                base.update(existing.get(sample_id, {}))
                seeded.append({field: base.get(field, "") for field in APPEARANCE_GT_FIELDS})
    return seeded


def _seed_event_rows(runs_root: Path, existing: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    if not runs_root.exists():
        return []
    seeded: list[dict[str, str]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        events_path = run_dir / "events.csv"
        if not events_path.exists():
            continue
        with events_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                crop_path = row.get("crop_path", "").strip()
                if not crop_path:
                    continue
                crop = Path(crop_path)
                if not crop.exists():
                    continue
                event_id = row.get("event_id", "").strip()
                sample_id = f"event::{run_dir.name}::{event_id or crop.name}"
                class_name = row.get("class_name", "Unknown").strip() or "Unknown"
                base = {
                    "sample_id": sample_id,
                    "image_path": str(crop),
                    "class_name": class_name,
                    "class_group": class_group_for_name(class_name),
                    "label": "",
                    "source_type": "event_crop",
                    "source_run": run_dir.name,
                    "event_id": event_id,
                    "track_id": row.get("track_id", ""),
                    "frame_idx": row.get("start_frame", ""),
                    "notes": row.get("anomaly_type", ""),
                }
                base.update(existing.get(sample_id, {}))
                seeded.append({field: base.get(field, "") for field in APPEARANCE_GT_FIELDS})
    return seeded


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    runs_root = Path(args.runs_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_rows(output_path)
    rows = _seed_normal_rows(dataset_dir, args.frames_per_sequence, existing)
    rows.extend(_seed_event_rows(runs_root, existing))
    seen_ids = {row["sample_id"] for row in rows}
    for sample_id, row in existing.items():
        if sample_id in seen_ids:
            continue
        rows.append({field: row.get(field, "") for field in APPEARANCE_GT_FIELDS})
    rows.sort(key=lambda row: (row["source_type"], row["class_group"], row["sample_id"]))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=APPEARANCE_GT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labeled = sum(1 for row in rows if row.get("label") in {"normal", "appearance_anomaly"})
    print(f"Saved appearance ground truth to: {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"Labeled rows: {labeled}")


if __name__ == "__main__":
    main()
