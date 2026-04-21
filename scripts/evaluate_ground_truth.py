"""
Compare predicted anomaly events against manually-labeled ground truth events.

The matcher is event-level and intentionally ignores tracker IDs.
Prediction and ground truth events are matched by anomaly type and temporal overlap.

Usage:
    python scripts/evaluate_ground_truth.py --run 20260416_080235
    python scripts/evaluate_ground_truth.py --run-dir runs\\20260416_080235 --gt dataset\\ground_truth_events.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class EventSpan:
    event_id: str
    start_frame: int
    end_frame: int
    anomaly_type: str
    class_name: str
    lane_id: str
    source: str
    raw: dict[str, str]


@dataclass(frozen=True)
class EvaluationSummary:
    run_dir: Path
    gt_path: Path
    events_path: Path
    total_gt: int
    total_pred: int
    matches: list[dict[str, object]]
    false_negatives: list[EventSpan]
    false_positives: list[EventSpan]
    precision: float
    recall: float
    f1: float
    lane_agreement: float
    class_agreement: float


def find_latest_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    subdirs = sorted((d for d in runs_root.iterdir() if d.is_dir()), key=lambda p: p.name, reverse=True)
    return subdirs[0] if subdirs else None


def load_gt_events(path: Path) -> list[EventSpan]:
    rows: list[EventSpan] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    EventSpan(
                        event_id=row.get("event_gt_id", ""),
                        start_frame=int(float(row.get("start_frame", 0) or 0)),
                        end_frame=int(float(row.get("end_frame", 0) or 0)),
                        anomaly_type=row.get("anomaly_type", ""),
                        class_name=row.get("actual_class", ""),
                        lane_id=row.get("lane_id", ""),
                        source="gt",
                        raw=row,
                    )
                )
            except ValueError:
                continue
    return rows


def load_pred_events(path: Path) -> list[EventSpan]:
    rows: list[EventSpan] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    EventSpan(
                        event_id=row.get("event_id", ""),
                        start_frame=int(float(row.get("start_frame", 0) or 0)),
                        end_frame=int(float(row.get("end_frame", 0) or 0)),
                        anomaly_type=row.get("anomaly_type", ""),
                        class_name=row.get("class_name", ""),
                        lane_id=row.get("lane_id", ""),
                        source="pred",
                        raw=row,
                    )
                )
            except ValueError:
                continue
    return rows


def temporal_intersection(a: EventSpan, b: EventSpan) -> int:
    return max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame) + 1)


def temporal_union(a: EventSpan, b: EventSpan) -> int:
    return max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame) + 1


def temporal_iou(a: EventSpan, b: EventSpan) -> float:
    inter = temporal_intersection(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1, temporal_union(a, b))


def gt_coverage_ratio(gt: EventSpan, pred: EventSpan) -> float:
    inter = temporal_intersection(gt, pred)
    span = max(1, gt.end_frame - gt.start_frame + 1)
    return inter / span


def pred_coverage_ratio(gt: EventSpan, pred: EventSpan) -> float:
    inter = temporal_intersection(gt, pred)
    span = max(1, pred.end_frame - pred.start_frame + 1)
    return inter / span


def within_window(event: EventSpan, frame_start: int | None, frame_end: int | None) -> bool:
    if frame_start is not None and event.end_frame < frame_start:
        return False
    if frame_end is not None and event.start_frame > frame_end:
        return False
    return True


def match_events(
    gt_events: list[EventSpan],
    pred_events: list[EventSpan],
    min_iou: float,
    min_gt_coverage: float,
    require_lane: bool,
) -> tuple[list[dict[str, object]], list[EventSpan], list[EventSpan]]:
    candidates: list[tuple[float, float, float, int, int]] = []
    for gt_idx, gt in enumerate(gt_events):
        for pred_idx, pred in enumerate(pred_events):
            if gt.anomaly_type != pred.anomaly_type:
                continue
            if require_lane and gt.lane_id and pred.lane_id and gt.lane_id != pred.lane_id:
                continue
            iou = temporal_iou(gt, pred)
            gt_cov = gt_coverage_ratio(gt, pred)
            pred_cov = pred_coverage_ratio(gt, pred)
            if iou < min_iou or gt_cov < min_gt_coverage:
                continue
            candidates.append((iou, gt_cov, pred_cov, gt_idx, pred_idx))

    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, object]] = []
    for iou, gt_cov, pred_cov, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        gt = gt_events[gt_idx]
        pred = pred_events[pred_idx]
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append(
            {
                "gt_event_id": gt.event_id,
                "pred_event_id": pred.event_id,
                "anomaly_type": gt.anomaly_type,
                "gt_start": gt.start_frame,
                "gt_end": gt.end_frame,
                "pred_start": pred.start_frame,
                "pred_end": pred.end_frame,
                "iou": iou,
                "gt_coverage": gt_cov,
                "pred_coverage": pred_cov,
                "lane_match": "yes" if gt.lane_id == pred.lane_id else "no",
                "class_match": "yes" if gt.class_name == pred.class_name else "no",
                "gt_lane": gt.lane_id,
                "pred_lane": pred.lane_id,
                "gt_class": gt.class_name,
                "pred_class": pred.class_name,
            }
        )

    false_negatives = [gt for idx, gt in enumerate(gt_events) if idx not in matched_gt]
    false_positives = [pred for idx, pred in enumerate(pred_events) if idx not in matched_pred]
    return matches, false_negatives, false_positives


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    print(f"\n{title}")
    print("  ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(
    run_dir: Path,
    gt_path: Path,
    total_gt: int,
    total_pred: int,
    matches: list[dict[str, object]],
    false_negatives: list[EventSpan],
    false_positives: list[EventSpan],
) -> str:
    tp = len(matches)
    fn = len(false_negatives)
    fp = len(false_positives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    lane_ok = sum(1 for item in matches if item["lane_match"] == "yes")
    class_ok = sum(1 for item in matches if item["class_match"] == "yes")

    lines = [
        "# Ground Truth Evaluation",
        "",
        f"Run: `{run_dir.name}`",
        f"Ground truth: `{gt_path}`",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall",
        "",
        f"- GT events: {total_gt}",
        f"- Predicted events: {total_pred}",
        f"- True positives: {tp}",
        f"- False negatives: {fn}",
        f"- False positives: {fp}",
        f"- Precision: {precision*100:.1f}%",
        f"- Recall: {recall*100:.1f}%",
        f"- F1: {f1*100:.1f}%",
        "",
        "## Match Quality",
        "",
        f"- Lane agreement on matched events: {lane_ok}/{tp} ({(lane_ok / tp * 100) if tp else 0.0:.1f}%)",
        f"- Class agreement on matched events: {class_ok}/{tp} ({(class_ok / tp * 100) if tp else 0.0:.1f}%)",
        "",
    ]

    by_type_gt = Counter(event.anomaly_type for event in false_negatives)
    by_type_gt.update(item["anomaly_type"] for item in matches)
    types = sorted(by_type_gt)
    if types:
        lines.extend(["## Per Type", "", "| Type | GT | TP | FN |", "|------|----|----|----|"])
        matched_counter = Counter(str(item["anomaly_type"]) for item in matches)
        fn_counter = Counter(event.anomaly_type for event in false_negatives)
        gt_counter = Counter(event.anomaly_type for event in false_negatives)
        gt_counter.update(str(item["anomaly_type"]) for item in matches)
        for anomaly_type in types:
            lines.append(f"| {anomaly_type} | {gt_counter[anomaly_type]} | {matched_counter[anomaly_type]} | {fn_counter[anomaly_type]} |")
        lines.append("")

    return "\n".join(lines)


def _compute_metrics(
    matches: list[dict[str, object]],
    false_negatives: list[EventSpan],
    false_positives: list[EventSpan],
) -> tuple[float, float, float, float, float]:
    tp = len(matches)
    fn = len(false_negatives)
    fp = len(false_positives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    lane_ok = sum(1 for item in matches if item["lane_match"] == "yes")
    class_ok = sum(1 for item in matches if item["class_match"] == "yes")
    lane_agreement = lane_ok / tp if tp else 0.0
    class_agreement = class_ok / tp if tp else 0.0
    return precision, recall, f1, lane_agreement, class_agreement


def evaluate_run(
    run_dir: Path,
    gt_path: Path,
    *,
    min_iou: float = 0.10,
    min_gt_coverage: float = 0.30,
    require_lane: bool = False,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> EvaluationSummary:
    run_dir = run_dir.resolve()
    gt_path = gt_path.resolve()
    events_path = run_dir / "events.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Predicted events not found: {events_path}")

    gt_events = [event for event in load_gt_events(gt_path) if within_window(event, frame_start, frame_end)]
    pred_events = [event for event in load_pred_events(events_path) if within_window(event, frame_start, frame_end)]
    matches, false_negatives, false_positives = match_events(
        gt_events=gt_events,
        pred_events=pred_events,
        min_iou=min_iou,
        min_gt_coverage=min_gt_coverage,
        require_lane=require_lane,
    )
    precision, recall, f1, lane_agreement, class_agreement = _compute_metrics(
        matches,
        false_negatives,
        false_positives,
    )
    return EvaluationSummary(
        run_dir=run_dir,
        gt_path=gt_path,
        events_path=events_path,
        total_gt=len(gt_events),
        total_pred=len(pred_events),
        matches=matches,
        false_negatives=false_negatives,
        false_positives=false_positives,
        precision=precision,
        recall=recall,
        f1=f1,
        lane_agreement=lane_agreement,
        class_agreement=class_agreement,
    )


def write_evaluation_outputs(summary: EvaluationSummary) -> Path:
    eval_dir = summary.run_dir / "ground_truth_eval"
    save_csv(
        eval_dir / "matched_events.csv",
        [
            "gt_event_id", "pred_event_id", "anomaly_type", "gt_start", "gt_end", "pred_start", "pred_end",
            "iou", "gt_coverage", "pred_coverage", "lane_match", "class_match", "gt_lane", "pred_lane", "gt_class", "pred_class",
        ],
        summary.matches,
    )
    save_csv(
        eval_dir / "false_negatives.csv",
        ["event_id", "start_frame", "end_frame", "anomaly_type", "class_name", "lane_id"],
        [
            {
                "event_id": event.event_id,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "anomaly_type": event.anomaly_type,
                "class_name": event.class_name,
                "lane_id": event.lane_id,
            }
            for event in summary.false_negatives
        ],
    )
    save_csv(
        eval_dir / "false_positives.csv",
        ["event_id", "start_frame", "end_frame", "anomaly_type", "class_name", "lane_id"],
        [
            {
                "event_id": event.event_id,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "anomaly_type": event.anomaly_type,
                "class_name": event.class_name,
                "lane_id": event.lane_id,
            }
            for event in summary.false_positives
        ],
    )

    report = build_report(
        summary.run_dir,
        summary.gt_path,
        summary.total_gt,
        summary.total_pred,
        summary.matches,
        summary.false_negatives,
        summary.false_positives,
    )
    report_path = eval_dir / "ground_truth_report.md"
    report_path.write_text(report, encoding="utf-8")
    return eval_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction events against ground truth.")
    parser.add_argument("--run", default=None, help="Run directory name under runs/.")
    parser.add_argument("--run-dir", default=None, help="Explicit path to run directory.")
    parser.add_argument("--runs-root", default="runs", help="Directory containing run folders.")
    parser.add_argument("--gt", default="dataset/ground_truth_events.csv", help="Ground truth CSV path.")
    parser.add_argument("--min-iou", type=float, default=0.10, help="Minimum temporal IoU for a match.")
    parser.add_argument("--min-gt-coverage", type=float, default=0.30, help="Minimum GT coverage ratio for a match.")
    parser.add_argument("--require-lane", action="store_true", help="Require lane_id to match when both are present.")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional evaluation window start frame.")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional evaluation window end frame.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.run:
        run_dir = (Path(args.runs_root) / args.run).resolve()
    else:
        latest = find_latest_run(Path(args.runs_root))
        if latest is None:
            print("No runs found.")
            sys.exit(1)
        run_dir = latest.resolve()

    try:
        summary = evaluate_run(
            run_dir=run_dir,
            gt_path=Path(args.gt),
            min_iou=args.min_iou,
            min_gt_coverage=args.min_gt_coverage,
            require_lane=args.require_lane,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    matches = summary.matches
    false_negatives = summary.false_negatives
    false_positives = summary.false_positives
    tp = len(matches)
    fn = len(false_negatives)
    fp = len(false_positives)
    precision = summary.precision
    recall = summary.recall
    f1 = summary.f1

    print(f"\n{'=' * 68}")
    print(f" Ground Truth Evaluation - {summary.run_dir.name}")
    print(f"{'=' * 68}")
    print(f"Ground truth: {summary.gt_path}")
    print(f"Predictions:  {summary.events_path}")
    if args.frame_start is not None or args.frame_end is not None:
        print(f"Window:       {args.frame_start or '-inf'} to {args.frame_end or 'inf'}")
    print(f"Match rule:   anomaly_type + IoU>={args.min_iou:.2f} + GT coverage>={args.min_gt_coverage:.2f}")
    print(f"Lane strict:  {'yes' if args.require_lane else 'no'}")

    print_table(
        "Overall",
        ["Metric", "Value"],
        [
            ["GT events", str(summary.total_gt)],
            ["Pred events", str(summary.total_pred)],
            ["TP", str(tp)],
            ["FN", str(fn)],
            ["FP", str(fp)],
            ["Precision", f"{precision*100:.1f}%"],
            ["Recall", f"{recall*100:.1f}%"],
            ["F1", f"{f1*100:.1f}%"],
        ],
    )

    gt_counter = Counter(event.anomaly_type for event in load_gt_events(summary.gt_path) if within_window(event, args.frame_start, args.frame_end))
    tp_counter = Counter(str(item["anomaly_type"]) for item in matches)
    fn_counter = Counter(event.anomaly_type for event in false_negatives)
    type_rows = []
    for anomaly_type in sorted(gt_counter):
        type_rows.append([anomaly_type, str(gt_counter[anomaly_type]), str(tp_counter[anomaly_type]), str(fn_counter[anomaly_type])])
    if type_rows:
        print_table("Per Type", ["Type", "GT", "TP", "FN"], type_rows)

    lane_ok = sum(1 for item in matches if item["lane_match"] == "yes")
    class_ok = sum(1 for item in matches if item["class_match"] == "yes")
    match_rows = [
        ["Lane agreement", f"{lane_ok}/{tp}", f"{summary.lane_agreement * 100:.1f}%"],
        ["Class agreement", f"{class_ok}/{tp}", f"{summary.class_agreement * 100:.1f}%"],
    ]
    print_table("Matched Event Quality", ["Metric", "Count", "Rate"], match_rows)

    eval_dir = write_evaluation_outputs(summary)
    print(f"\nSaved report to: {eval_dir / 'ground_truth_report.md'}")


if __name__ == "__main__":
    main()
