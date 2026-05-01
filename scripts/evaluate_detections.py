"""
Evaluate detection performance from manually-labeled anomaly results.

Reports TWO separate metrics:
  1. Anomaly Detection Accuracy — are the alerts correct? (TP vs FP)
  2. YOLO Classification Accuracy — did YOLO label the vehicle class correctly?

Usage:
    python scripts/evaluate_detections.py                   # latest run
    python scripts/evaluate_detections.py --run 20260407_002145
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def find_latest_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    subdirs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def load_labels(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        return []
    with labels_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_anomaly_metrics(rows: list[dict[str, str]]) -> dict:
    tp = sum(1 for r in rows if r.get("label") == "TP")
    fp = sum(1 for r in rows if r.get("label") == "FP")
    skip = sum(1 for r in rows if r.get("label") == "skip")
    total_labeled = tp + fp
    precision = tp / total_labeled if total_labeled > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "skip": skip,
        "total_labeled": total_labeled,
        "precision": precision,
    }


def compute_classification_metrics(rows: list[dict[str, str]]) -> dict:
    """Compute YOLO classification accuracy from yolo_class vs actual_class."""
    checked = 0
    correct = 0
    misclass: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # yolo_class -> actual_class -> count

    for row in rows:
        actual = row.get("actual_class", "")
        yolo = row.get("yolo_class", "")
        class_correct = row.get("class_correct", "")

        if not actual or actual == "Unknown" or not yolo:
            continue

        checked += 1
        if class_correct == "yes" or actual == yolo:
            correct += 1
        else:
            misclass[yolo][actual] += 1

    accuracy = correct / checked if checked > 0 else 0.0
    return {
        "checked": checked,
        "correct": correct,
        "incorrect": checked - correct,
        "accuracy": accuracy,
        "misclassifications": dict(misclass),
    }


def group_by_field(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get(field, "unknown")
        groups[key].append(row)
    return dict(groups)


def print_table(title: str, headers: list[str], rows_data: list[list[str]]):
    widths = [len(h) for h in headers]
    for row in rows_data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)

    print(f"\n{title}")
    print(header_line)
    print(sep)
    for row in rows_data:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def generate_report(run_dir: Path, rows: list[dict[str, str]], threshold: float) -> str:
    """Generate a markdown report covering both anomaly accuracy and YOLO classification."""
    lines: list[str] = []
    lines.append(f"# Detection Performance Report")
    lines.append(f"")
    lines.append(f"**Run**: `{run_dir.name}`")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Acceptance threshold**: {threshold*100:.0f}%")
    lines.append("")

    # ── Section 1: Anomaly Detection ──
    overall = compute_anomaly_metrics(rows)
    anomaly_verdict = "✅ ACCEPTABLE" if overall["precision"] >= threshold else "❌ NOT ACCEPTABLE"

    lines.append(f"## 1. Anomaly Detection Accuracy")
    lines.append(f"")
    lines.append(f"*Is the alert actually a real anomaly?*")
    lines.append(f"")
    lines.append(f"| Metric         | Value |")
    lines.append(f"|----------------|-------|")
    lines.append(f"| True Positives | {overall['tp']} |")
    lines.append(f"| False Positives| {overall['fp']} |")
    lines.append(f"| Skipped        | {overall['skip']} |")
    lines.append(f"| Total Labeled  | {overall['total_labeled']} |")
    lines.append(f"| **Precision**  | **{overall['precision']*100:.1f}%** |")
    lines.append(f"| **Verdict**    | **{anomaly_verdict}** |")
    lines.append("")

    # Per anomaly type
    by_type = group_by_field(rows, "anomaly_type")
    lines.append("### Per Anomaly Type")
    lines.append("")
    lines.append("| Type | TP | FP | Skip | Precision |")
    lines.append("|------|----|----|------|-----------|")
    for atype in sorted(by_type.keys()):
        m = compute_anomaly_metrics(by_type[atype])
        pct = f"{m['precision']*100:.1f}%" if m["total_labeled"] > 0 else "—"
        lines.append(f"| {atype} | {m['tp']} | {m['fp']} | {m['skip']} | {pct} |")
    lines.append("")

    # ── Section 2: YOLO Classification ──
    class_metrics = compute_classification_metrics(rows)
    class_verdict = "✅ ACCEPTABLE" if class_metrics["accuracy"] >= threshold else "❌ NOT ACCEPTABLE"

    lines.append(f"## 2. YOLO Classification Accuracy")
    lines.append(f"")
    lines.append(f"*Did YOLO correctly identify the vehicle type?*")
    lines.append(f"")
    lines.append(f"| Metric          | Value |")
    lines.append(f"|-----------------|-------|")
    lines.append(f"| Checked         | {class_metrics['checked']} |")
    lines.append(f"| Correct         | {class_metrics['correct']} |")
    lines.append(f"| Incorrect       | {class_metrics['incorrect']} |")
    lines.append(f"| **Accuracy**    | **{class_metrics['accuracy']*100:.1f}%** |")
    lines.append(f"| **Verdict**     | **{class_verdict}** |")
    lines.append("")

    # Confusion details
    if class_metrics["misclassifications"]:
        lines.append("### Misclassification Details")
        lines.append("")
        lines.append("| YOLO Predicted | Actually Was | Count |")
        lines.append("|----------------|-------------|-------|")
        for yolo_class, actuals in sorted(class_metrics["misclassifications"].items()):
            for actual_class, count in sorted(actuals.items(), key=lambda x: -x[1]):
                lines.append(f"| {yolo_class} | {actual_class} | {count} |")
        lines.append("")

    # ── Section 3: Combined verdict ──
    lines.append("## 3. Overall Verdict")
    lines.append("")
    anomaly_ok = overall["precision"] >= threshold
    class_ok = class_metrics["accuracy"] >= threshold if class_metrics["checked"] > 0 else True
    if anomaly_ok and class_ok:
        lines.append("> ✅ **SYSTEM ACCEPTABLE** — Both anomaly detection and YOLO classification meet the threshold.")
    elif anomaly_ok and not class_ok:
        lines.append("> ⚠️ **PARTIAL** — Anomaly detection is acceptable, but YOLO misclassifies too many vehicles.")
        lines.append("> Consider: fine-tuning YOLO on highway CCTV data, or adjusting camera angle compatibility (Step 3).")
    elif not anomaly_ok and class_ok:
        lines.append("> ⚠️ **PARTIAL** — YOLO classification is fine, but anomaly rules produce too many false alerts.")
        lines.append("> Consider: tuning threshold parameters in scene_config.yaml.")
    else:
        lines.append("> ❌ **NOT ACCEPTABLE** — Both anomaly detection and YOLO classification are below threshold.")
        lines.append("> Consider: dataset analysis (Step 3), image enhancement (Step 4), or model fine-tuning.")
    lines.append("")

    return "\n".join(lines)


def detect_false_negatives(classification_rows: list[dict[str, str]], forbidden_policy: dict[str, list[str]]) -> list[dict[str, str]]:
    """
    Find vehicles from the classification sampler where YOLO's misclassification
    likely caused a violation to be MISSED (False Negative).

    E.g., a truck YOLO called "Car" in a fast lane → no lane_violation was triggered.
    """
    false_negatives: list[dict[str, str]] = []
    for row in classification_rows:
        actual = row.get("actual_class", "")
        yolo = row.get("yolo_class", "")
        lane = row.get("lane_id", "")
        correct = row.get("class_correct", "")

        if correct == "yes" or not actual or actual == "Unknown":
            continue

        # Check if the actual class is forbidden in the lane but YOLO's class isn't
        actual_forbidden = lane in forbidden_policy.get(actual, [])
        yolo_forbidden = lane in forbidden_policy.get(yolo, [])

        if actual_forbidden and not yolo_forbidden:
            false_negatives.append({
                "track_id": row.get("track_id", ""),
                "frame_idx": row.get("frame_idx", ""),
                "lane_id": lane,
                "yolo_class": yolo,
                "actual_class": actual,
                "explanation": f"YOLO called it {yolo} (allowed in {lane}) but it's actually {actual} (forbidden)",
            })
    return false_negatives


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection + YOLO classification performance")
    parser.add_argument("--run", default=None, help="Run ID (directory name under runs/)")
    parser.add_argument("--run-dir", default=None, help="Full path to run directory")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing run folders")
    parser.add_argument("--threshold", type=float, default=0.80, help="Acceptable precision threshold (default: 0.80)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.run:
        run_dir = Path(args.runs_root) / args.run
    else:
        run_dir = find_latest_run(Path(args.runs_root))
        if run_dir is None:
            print("No runs found. Run the pipeline first.")
            sys.exit(1)

    # Load anomaly labels
    anomaly_labels_path = run_dir / "anomaly_labels.csv"
    anomaly_rows = load_labels(anomaly_labels_path) if anomaly_labels_path.exists() else []

    # Load classification sampler labels
    class_labels_path = run_dir / "classification_labels.csv"
    class_rows = load_labels(class_labels_path) if class_labels_path.exists() else []

    if not anomaly_rows and not class_rows:
        print(f"No labels found in {run_dir}")
        print("Run the labeling tools first:")
        print("  python scripts/label_anomalies.py")
        print("  python scripts/sample_classifications.py")
        sys.exit(1)

    # ════════════════════════════════════════════
    # Console output
    # ════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f" Detection Performance Report — {run_dir.name}")
    print(f"{'='*65}")

    # 1. Anomaly accuracy (from anomaly_labels.csv)
    if anomaly_rows:
        overall = compute_anomaly_metrics(anomaly_rows)
        anomaly_ok = overall["precision"] >= args.threshold
        print_table(
            "1. ANOMALY DETECTION ACCURACY (from anomaly labeler)",
            ["Metric", "Value"],
            [
                ["True Positives", str(overall["tp"])],
                ["False Positives", str(overall["fp"])],
                ["Skipped", str(overall["skip"])],
                ["Precision", f"{overall['precision']*100:.1f}%"],
                ["Verdict", f"{'✅ ACCEPTABLE' if anomaly_ok else '❌ NOT ACCEPTABLE'}"],
            ],
        )

        by_type = group_by_field(anomaly_rows, "anomaly_type")
        type_rows = []
        for atype in sorted(by_type.keys()):
            m = compute_anomaly_metrics(by_type[atype])
            pct = f"{m['precision']*100:.1f}%" if m["total_labeled"] > 0 else "—"
            type_rows.append([atype, str(m["tp"]), str(m["fp"]), str(m["skip"]), pct])
        print_table("  Per Anomaly Type", ["Type", "TP", "FP", "Skip", "Precision"], type_rows)
    else:
        print("\n[No anomaly labels found — run: python scripts/label_anomalies.py]")

    # 2. YOLO classification accuracy (merge BOTH sources)
    # From anomaly_labels.csv: class accuracy on anomaly events only
    # From classification_labels.csv: class accuracy on ALL vehicles (sampled)
    all_class_rows = anomaly_rows + class_rows
    class_metrics = compute_classification_metrics(all_class_rows)

    source_note = ""
    if anomaly_rows and class_rows:
        anom_metrics = compute_classification_metrics(anomaly_rows)
        samp_metrics = compute_classification_metrics(class_rows)
        source_note = (
            f"  Sources: anomaly events ({anom_metrics['checked']} checked) + "
            f"random sample ({samp_metrics['checked']} checked)"
        )
    elif class_rows:
        source_note = "  Source: random vehicle sample (classification_labels.csv)"
    else:
        source_note = "  Source: anomaly events only (anomaly_labels.csv)"

    if class_metrics["checked"] > 0:
        class_ok = class_metrics["accuracy"] >= args.threshold
        print_table(
            f"\n2. YOLO CLASSIFICATION ACCURACY",
            ["Metric", "Value"],
            [
                ["Checked", str(class_metrics["checked"])],
                ["Correct", str(class_metrics["correct"])],
                ["Incorrect", str(class_metrics["incorrect"])],
                ["Accuracy", f"{class_metrics['accuracy']*100:.1f}%"],
                ["Verdict", f"{'✅ ACCEPTABLE' if class_ok else '❌ NOT ACCEPTABLE'}"],
            ],
        )
        if source_note:
            print(source_note)

        if class_metrics["misclassifications"]:
            misclass_rows = []
            for yolo_cls, actuals in sorted(class_metrics["misclassifications"].items()):
                for actual_cls, count in sorted(actuals.items(), key=lambda x: -x[1]):
                    misclass_rows.append([yolo_cls, actual_cls, str(count)])
            print_table("  Misclassification Details (Confusion Matrix)",
                        ["YOLO Predicted", "Actually Was", "Count"], misclass_rows)
    else:
        print("\n[No YOLO classification data — set actual class in labeling tools]")

    # 3. False Negatives (from classification sampler)
    if class_rows:
        # Default lane policy
        forbidden_policy = {
            "Bus": ["fast_lane_1", "fast_lane_2"],
            "Truck": ["fast_lane_1", "fast_lane_2"],
        }
        fn_list = detect_false_negatives(class_rows, forbidden_policy)
        if fn_list:
            fn_rows = [
                [fn["track_id"], fn["frame_idx"], fn["yolo_class"], fn["actual_class"],
                 fn["lane_id"], fn["explanation"]]
                for fn in fn_list
            ]
            print_table(
                "\n3. LIKELY FALSE NEGATIVES (missed violations due to misclassification)",
                ["Track", "Frame", "YOLO Said", "Actually", "Lane", "Explanation"],
                fn_rows,
            )
            print(f"\n  ⚠️  {len(fn_list)} potential violations were MISSED because YOLO misclassified the vehicle.")
        else:
            print("\n3. FALSE NEGATIVES: None detected from sampled vehicles ✅")

    # Save full report
    report = generate_report(run_dir, anomaly_rows or all_class_rows, args.threshold)
    report_path = run_dir / "detection_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{'='*65}")
    print(f" Report saved to: {report_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

