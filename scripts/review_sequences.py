from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.storage import REVIEW_FIELDS, sync_sequence_manifests


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review exported normal sequences for GANomaly training.")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset directory containing sequence_review.csv.")
    parser.add_argument("--sequence-id", action="append", default=[], help="Sequence ID to update. Repeat for multiple.")
    parser.add_argument("--status", choices=["pending", "approved", "rejected"], help="New review status.")
    parser.add_argument("--notes", default="", help="Optional review notes.")
    parser.add_argument("--approve-all-pending", action="store_true", help="Approve all pending rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    sync_sequence_manifests(dataset_dir)
    review_path = dataset_dir / "sequence_review.csv"
    rows = load_rows(review_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for row in rows:
        update = False
        if args.approve_all_pending and row.get("review_status", "pending") == "pending":
            row["review_status"] = "approved"
            update = True
        if args.sequence_id and row.get("sequence_id") in set(args.sequence_id):
            if args.status:
                row["review_status"] = args.status
            if args.notes:
                row["review_notes"] = args.notes
            update = True
        if update:
            row["last_updated"] = now

    save_rows(review_path, rows)
    summary = {}
    for row in rows:
        summary[row.get("review_status", "pending")] = summary.get(row.get("review_status", "pending"), 0) + 1
    print(f"Updated {review_path}")
    print(summary)


if __name__ == "__main__":
    main()
