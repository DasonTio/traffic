"""
GT-aware sequence labeler.

Reads dataset/sequence_review.csv and dataset/ground_truth_events.csv, then marks every
'pending' sequence as either:

  approved  — frame range does NOT overlap any GT anomaly event for the same vehicle class.
              These are clean normal-appearance crops used to train GANomaly / VAE.

  rejected  — frame range OVERLAPS at least one GT anomaly event. Excluded from training;
              used as positive appearance-anomaly samples in the appearance ground truth.

Sequences already set to 'approved' or 'rejected' are left untouched.

Usage:
    python scripts/approve_all_sequences.py
    python scripts/approve_all_sequences.py --config configs/scene_config.yaml
    python scripts/approve_all_sequences.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_SEQ_FRAME_RE = re.compile(r"_(\d+)_(\d+)$")  # …_<start>_<end>


def _parse_seq_frames(sequence_id: str) -> tuple[int, int] | None:
    """Extract (start_frame, end_frame) from the sequence_id suffix."""
    m = _SEQ_FRAME_RE.search(sequence_id)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _frames_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return True when [a_start, a_end] and [b_start, b_end] share at least one frame."""
    return a_start <= b_end and b_start <= a_end


def _normalise_class(name: str) -> str:
    return name.strip().lower()


# --------------------------------------------------------------------------- #
# GT event loading                                                             #
# --------------------------------------------------------------------------- #

def load_gt_events(gt_path: Path) -> list[dict[str, object]]:
    """Return a list of GT events as {class_lower, start_frame, end_frame}."""
    events: list[dict[str, object]] = []
    if not gt_path.exists():
        return events
    with gt_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                events.append(
                    {
                        "class": _normalise_class(row.get("actual_class", "")),
                        "start": int(float(row.get("start_frame", 0) or 0)),
                        "end": int(float(row.get("end_frame", 0) or 0)),
                        "anomaly_type": row.get("anomaly_type", ""),
                    }
                )
            except ValueError:
                continue
    return events


# --------------------------------------------------------------------------- #
# Core logic                                                                   #
# --------------------------------------------------------------------------- #

def is_anomalous_sequence(
    seq_class: str,
    seq_start: int,
    seq_end: int,
    gt_events: list[dict[str, object]],
) -> bool:
    """Return True if this sequence overlaps any GT event of the same vehicle class."""
    norm = _normalise_class(seq_class)
    for ev in gt_events:
        if ev["class"] != norm:
            continue
        if _frames_overlap(seq_start, seq_end, int(ev["start"]), int(ev["end"])):
            return True
    return False


def label_sequences(
    review_path: Path,
    gt_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Read, label, and (unless dry_run) overwrite the sequence_review CSV.

    Returns a counter dict with keys: approved / rejected / skipped.
    """
    gt_events = load_gt_events(gt_path)
    if not gt_events:
        print(f"WARNING: No GT events loaded from {gt_path}. All pending sequences will be approved.")

    if not review_path.exists():
        raise FileNotFoundError(f"Sequence review CSV not found: {review_path}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with review_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counters: Counter[str] = Counter()

    for row in rows:
        status = row.get("review_status", "").strip().lower()
        if status in {"approved", "rejected"}:
            counters["skipped"] += 1
            continue

        sequence_id = row.get("sequence_id", "")
        class_name = row.get("class_name", "Unknown")
        frame_range = _parse_seq_frames(sequence_id)

        if frame_range is None:
            # Can't determine frame range — approve conservatively
            new_status = "approved"
        else:
            seq_start, seq_end = frame_range
            if is_anomalous_sequence(class_name, seq_start, seq_end, gt_events):
                new_status = "rejected"
            else:
                new_status = "approved"

        row["review_status"] = new_status
        row["last_updated"] = now_str
        counters[new_status] += 1

    if not dry_run:
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return dict(counters)


# --------------------------------------------------------------------------- #
# Per-class summary                                                            #
# --------------------------------------------------------------------------- #

def print_class_summary(review_path: Path) -> None:
    class_approved: Counter[str] = Counter()
    class_rejected: Counter[str] = Counter()

    with review_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = row.get("review_status", "").strip().lower()
            cls = row.get("class_name", "Unknown")
            if status == "approved":
                class_approved[cls] += 1
            elif status == "rejected":
                class_rejected[cls] += 1

    all_classes = sorted(set(list(class_approved) + list(class_rejected)))
    print(f"\n{'Class':<12} {'Approved':>10} {'Rejected':>10}")
    print("-" * 34)
    for cls in all_classes:
        print(f"{cls:<12} {class_approved[cls]:>10} {class_rejected[cls]:>10}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT-aware approval of pending vehicle sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--review-csv",
        default="dataset/sequence_review.csv",
        help="Path to sequence_review.csv.",
    )
    parser.add_argument(
        "--gt",
        default="dataset/ground_truth_events.csv",
        help="Path to ground_truth_events.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_path = Path(args.review_csv).resolve()
    gt_path = Path(args.gt).resolve()

    print(f"Review CSV : {review_path}")
    print(f"GT events  : {gt_path}")
    if args.dry_run:
        print("[DRY RUN — no files will be modified]")

    counters = label_sequences(review_path, gt_path, dry_run=args.dry_run)

    print(f"\nResults:")
    print(f"  approved   : {counters.get('approved', 0)}")
    print(f"  rejected   : {counters.get('rejected', 0)}")
    print(f"  skipped    : {counters.get('skipped', 0)}")

    if not args.dry_run:
        print_class_summary(review_path)
        print(f"\nWrote updated sequence review to: {review_path}")


if __name__ == "__main__":
    main()
