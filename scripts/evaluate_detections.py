"""
Evaluate detection performance from manually-labeled anomaly results.

Reads anomaly_labels.csv (produced by label_anomalies.py) and computes:
  - Overall TP / FP / rates
  - Per-anomaly-type breakdown
  - Per-class breakdown
  - Verdict: ACCEPTABLE (≥80% precision) or NOT ACCEPTABLE

Usage:
    python scripts/evaluate_detections.py                   # latest run
    python scripts/evaluate_detections.py --run 20260406_233816
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


def compute_metrics(rows: list[dict[str, str]]) -> dict:
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


def generate_report(run_dir: Path, rows: list[dict[str, str]]) -> str:
    """Generate a markdown report and return the text."""
    lines: list[str] = []
    lines.append(f"# Detection Performance Report")
    lines.append(f"")
    lines.append(f"**Run**: `{run_dir.name}`")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    overall = compute_metrics(rows)
    verdict = "✅ ACCEPTABLE" if overall["precision"] >= 0.80 else "❌ NOT ACCEPTABLE"

    lines.append(f"## Overall Results")
    lines.append("")
    lines.append(f"| Metric         | Value |")
    lines.append(f"|----------------|-------|")
    lines.append(f"| True Positives | {overall['tp']} |")
    lines.append(f"| False Positives| {overall['fp']} |")
    lines.append(f"| Skipped        | {overall['skip']} |")
    lines.append(f"| Total Labeled  | {overall['total_labeled']} |")
    lines.append(f"| **Precision**  | **{overall['precision']*100:.1f}%** |")
    lines.append(f"| **Verdict**    | **{verdict}** |")
    lines.append("")

    # Per anomaly type
    by_type = group_by_field(rows, "anomaly_type")
    lines.append("## Per Anomaly Type")
    lines.append("")
    lines.append("| Type | TP | FP | Skip | Precision |")
    lines.append("|------|----|----|------|-----------|")
    for atype in sorted(by_type.keys()):
        m = compute_metrics(by_type[atype])
        pct = f"{m['precision']*100:.1f}%" if m["total_labeled"] > 0 else "—"
        lines.append(f"| {atype} | {m['tp']} | {m['fp']} | {m['skip']} | {pct} |")
    lines.append("")

    # Per class
    by_class = group_by_field(rows, "class_name")
    lines.append("## Per Vehicle Class")
    lines.append("")
    lines.append("| Class | TP | FP | Skip | Precision |")
    lines.append("|-------|----|----|------|-----------|")
    for cls in sorted(by_class.keys()):
        m = compute_metrics(by_class[cls])
        pct = f"{m['precision']*100:.1f}%" if m["total_labeled"] > 0 else "—"
        lines.append(f"| {cls} | {m['tp']} | {m['fp']} | {m['skip']} | {pct} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection performance from labels")
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

    labels_path = run_dir / "anomaly_labels.csv"
    if not labels_path.exists():
        print(f"No labels found at {labels_path}")
        print("Run the labeling tool first: python scripts/label_anomalies.py")
        sys.exit(1)

    rows = load_labels(labels_path)
    labeled_rows = [r for r in rows if r.get("label") in ("TP", "FP")]
    if not labeled_rows:
        print("No TP/FP labels found. Label some anomalies first.")
        sys.exit(1)

    # Console output
    overall = compute_metrics(rows)
    verdict = "ACCEPTABLE" if overall["precision"] >= args.threshold else "NOT ACCEPTABLE"
    verdict_emoji = "✅" if overall["precision"] >= args.threshold else "❌"

    print(f"\n{'='*60}")
    print(f" Detection Performance Report — {run_dir.name}")
    print(f"{'='*60}")

    print_table(
        "Overall",
        ["Metric", "Value"],
        [
            ["True Positives", str(overall["tp"])],
            ["False Positives", str(overall["fp"])],
            ["Skipped", str(overall["skip"])],
            ["Precision", f"{overall['precision']*100:.1f}%"],
            ["Verdict", f"{verdict_emoji} {verdict}"],
        ],
    )

    by_type = group_by_field(rows, "anomaly_type")
    type_rows = []
    for atype in sorted(by_type.keys()):
        m = compute_metrics(by_type[atype])
        pct = f"{m['precision']*100:.1f}%" if m["total_labeled"] > 0 else "—"
        type_rows.append([atype, str(m["tp"]), str(m["fp"]), str(m["skip"]), pct])
    print_table("Per Anomaly Type", ["Type", "TP", "FP", "Skip", "Precision"], type_rows)

    by_class = group_by_field(rows, "class_name")
    class_rows = []
    for cls in sorted(by_class.keys()):
        m = compute_metrics(by_class[cls])
        pct = f"{m['precision']*100:.1f}%" if m["total_labeled"] > 0 else "—"
        class_rows.append([cls, str(m["tp"]), str(m["fp"]), str(m["skip"]), pct])
    print_table("Per Vehicle Class", ["Class", "TP", "FP", "Skip", "Precision"], class_rows)

    # Save markdown report
    report = generate_report(run_dir, rows)
    report_path = run_dir / "detection_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    print()


if __name__ == "__main__":
    main()
