"""
Seed (or refresh) dataset/appearance_ground_truth.csv from two sources:

  normal             — approved sequences in sequence_review.csv.
                       All sequences recorded by the pipeline are clean-normal crop windows
                       (the pipeline only saves frames from tracks that complete without
                       triggering an anomaly event). One representative middle frame per sequence.

  appearance_anomaly — event crop images stored in runs/*/crops/*.jpg.
                       Each crop was saved at the moment the pipeline raised an anomaly event,
                       so they represent vehicles that were flagged as anomalous.

The script targets a balanced split (up to --max-per-label samples per label) to prevent class
imbalance from distorting AUROC / AUPRC computation.

Usage:
    python scripts/seed_appearance_ground_truth.py
    python scripts/seed_appearance_ground_truth.py --max-per-label 200
    python scripts/seed_appearance_ground_truth.py --runs-root runs --dry-run
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GT_FIELDNAMES = [
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

APPROVED_STATUSES = {"approved", "usable", "yes", "true"}
CLASS_GROUPS = {"car", "bus", "truck"}


def _class_group(name: str) -> str:
    n = name.strip().lower()
    return n if n in CLASS_GROUPS else n


def _pick_representative_image(sequence_path: Path) -> Path | None:
    """Return the middle .jpg frame from a sequence directory, or None if empty."""
    frames = sorted(sequence_path.glob("*.jpg"))
    if not frames:
        frames = sorted(sequence_path.glob("*.png"))
    if not frames:
        return None
    return frames[len(frames) // 2]


def collect_normal_candidates(review_path: Path) -> list[dict[str, object]]:
    """One representative image per approved sequence → label=normal."""
    candidates: list[dict[str, object]] = []
    if not review_path.exists():
        return candidates
    with review_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = row.get("review_status", "").strip().lower()
            if status not in APPROVED_STATUSES:
                continue
            seq_id = row.get("sequence_id", "")
            class_name = row.get("class_name", "Unknown")
            seq_path = Path(row.get("sequence_path", ""))
            img = _pick_representative_image(seq_path)
            if img is None:
                continue
            candidates.append(
                {
                    "sequence_id": seq_id,
                    "class_name": class_name,
                    "class_group": _class_group(class_name),
                    "image_path": str(img),
                    "source_run": "",
                    "event_id": "",
                    "track_id": "",
                    "frame_idx": "",
                }
            )
    return candidates


def collect_anomaly_candidates(
    runs_root: Path,
    events_csv_name: str = "events.csv",
) -> list[dict[str, object]]:
    """
    Collect event crop images from all run directories → label=appearance_anomaly.
    Each crop was saved by the pipeline at the moment of an anomaly detection.
    """
    candidates: list[dict[str, object]] = []
    if not runs_root.exists():
        return candidates
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        events_path = run_dir / events_csv_name
        if not events_path.exists():
            continue
        with events_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                crop_path = Path(row.get("crop_path", ""))
                if not crop_path.exists():
                    continue
                class_name = row.get("class_name", "Unknown")
                candidates.append(
                    {
                        "class_name": class_name,
                        "class_group": _class_group(class_name),
                        "image_path": str(crop_path),
                        "source_run": run_dir.name,
                        "event_id": row.get("event_id", ""),
                        "track_id": str(row.get("track_id", "")),
                        "frame_idx": str(row.get("start_frame", "")),
                        "anomaly_type": row.get("anomaly_type", ""),
                    }
                )
    return candidates


def build_rows(
    review_path: Path,
    runs_root: Path,
    *,
    max_per_label: int = 300,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Return labeled rows ready to write into appearance_ground_truth.csv."""
    rng = random.Random(seed)

    normal_candidates = collect_normal_candidates(review_path)
    anomaly_candidates = collect_anomaly_candidates(runs_root)

    rng.shuffle(normal_candidates)
    rng.shuffle(anomaly_candidates)
    normal_sample = normal_candidates[:max_per_label]
    anomaly_sample = anomaly_candidates[:max_per_label]

    output_rows: list[dict[str, object]] = []

    for idx, entry in enumerate(normal_sample, start=1):
        output_rows.append(
            {
                "sample_id": f"normal_{idx:04d}_{entry['sequence_id']}",
                "image_path": entry["image_path"],
                "class_name": entry["class_name"],
                "class_group": entry["class_group"],
                "label": "normal",
                "source_type": "sequence_crop",
                "source_run": entry["source_run"],
                "event_id": entry["event_id"],
                "track_id": entry["track_id"],
                "frame_idx": entry["frame_idx"],
                "notes": "approved_normal_sequence",
            }
        )

    for idx, entry in enumerate(anomaly_sample, start=1):
        output_rows.append(
            {
                "sample_id": f"anomaly_{idx:04d}_{entry['event_id']}",
                "image_path": entry["image_path"],
                "class_name": entry["class_name"],
                "class_group": entry["class_group"],
                "label": "appearance_anomaly",
                "source_type": "event_crop",
                "source_run": entry["source_run"],
                "event_id": entry["event_id"],
                "track_id": entry["track_id"],
                "frame_idx": entry["frame_idx"],
                "notes": f"predicted_{entry.get('anomaly_type', 'anomaly')}",
            }
        )

    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed appearance_ground_truth.csv from approved sequences + event crops.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--review-csv",
        default="dataset/sequence_review.csv",
        help="Path to sequence_review.csv.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Directory containing run folders (for event crops).",
    )
    parser.add_argument(
        "--output",
        default="dataset/appearance_ground_truth.csv",
        help="Output path for the appearance ground truth CSV.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=300,
        help="Maximum number of samples per label (normal / appearance_anomaly).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling candidates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_path = Path(args.review_csv).resolve()
    runs_root = Path(args.runs_root).resolve()
    output_path = Path(args.output).resolve()

    print(f"Review CSV : {review_path}")
    print(f"Runs root  : {runs_root}")
    print(f"Output     : {output_path}")
    if args.dry_run:
        print("[DRY RUN — no files will be written]")

    rows = build_rows(review_path, runs_root, max_per_label=args.max_per_label, seed=args.seed)

    normal_count = sum(1 for r in rows if r["label"] == "normal")
    anomaly_count = sum(1 for r in rows if r["label"] == "appearance_anomaly")
    print(f"\nSamples generated:")
    print(f"  normal             : {normal_count}")
    print(f"  appearance_anomaly : {anomaly_count}")
    print(f"  total              : {len(rows)}")

    if anomaly_count == 0:
        print(
            "\nWARNING: No anomaly samples found. Run inference first to populate runs/*/crops/, "
            "then re-run this script."
        )
    if normal_count == 0:
        print(
            "\nWARNING: No normal samples found. Ensure sequences are marked 'approved' in "
            "sequence_review.csv (run approve_all_sequences.py first)."
        )

    if not args.dry_run and rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
