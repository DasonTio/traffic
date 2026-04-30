from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ground_truth import EvaluationSummary, evaluate_run, write_evaluation_outputs
from traffic_anomaly.pipeline import TrafficAnomalyPipeline


def _resolve_tracker_config(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix:
        return (ROOT / candidate).resolve()
    return (ROOT / "configs" / f"{value}.yaml").resolve()


def _write_run_config(base_config: str | Path, output_path: Path, source: str | Path, report_dir: Path) -> Path:
    base_path = Path(base_config).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("video", {})["default"] = "local"
    raw["video"]["local"] = str(Path(source).resolve())
    raw.setdefault("output", {})["run_root"] = str((report_dir / "runs").resolve())
    raw["output"]["dataset_dir"] = str((report_dir / "dataset").resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return output_path


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summary_dict(
    *,
    detector: str,
    detector_weights: Path | str,
    tracker: str,
    run_dir: Path,
    summary: EvaluationSummary,
) -> dict[str, Any]:
    return {
        "pipeline": f"{detector}/{tracker}",
        "detector": detector,
        "detector_weights": str(detector_weights),
        "tracker": tracker,
        "appearance_model": "none",
        "run_dir": str(run_dir),
        "gt_events": summary.total_gt,
        "pred_events": summary.total_pred,
        "true_positives": len(summary.matches),
        "false_positives": len(summary.false_positives),
        "false_negatives": len(summary.false_negatives),
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
        "lane_agreement": summary.lane_agreement,
        "class_agreement": summary.class_agreement,
    }


def _build_report(rows: list[dict[str, Any]], frame_end: int | None) -> str:
    ranked = sorted(rows, key=lambda row: row["f1"], reverse=True)
    best = ranked[0] if ranked else None
    lines = [
        "# Detector + Tracker Event-Based Anomaly Comparison",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Appearance models: disabled (`none`).",
        f"Evaluation window: frames 1-{frame_end}" if frame_end else "Evaluation window: full run.",
        "",
    ]
    if best:
        lines += [
            "## Best Pipeline",
            "",
            (
                f"Best by F1: **{best['pipeline']}** with F1 {_pct(best['f1'])}, "
                f"precision {_pct(best['precision'])}, recall {_pct(best['recall'])}."
            ),
            "",
        ]

    lines += [
        "## Metrics",
        "",
        "| Pipeline | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Lane Agree | Class Agree | Run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['pipeline']} | {row['gt_events']} | {row['pred_events']} | "
            f"{row['true_positives']} | {row['false_positives']} | {row['false_negatives']} | "
            f"{_pct(row['precision'])} | {_pct(row['recall'])} | {_pct(row['f1'])} | "
            f"{_pct(row['lane_agreement'])} | {_pct(row['class_agreement'])} | `{Path(row['run_dir']).name}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_outputs(rows: list[dict[str, Any]], report_dir: Path, frame_end: int | None) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "detector_tracker_metrics.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "appearance_model": "none",
                "frame_end": frame_end,
                "metrics": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = report_dir / "detector_tracker_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    report_path = report_dir / "detector_tracker_report.md"
    report_path.write_text(_build_report(rows, frame_end), encoding="utf-8")
    print(f"Report:  {report_path}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLO11-m vs RT-DETR with ByteTrack vs OC-Sort, without VAE/GAN."
    )
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Base scene config.")
    parser.add_argument("--source", default=".video/video.mp4.mp4", help="Video source path.")
    parser.add_argument("--gt", default="dataset/ground_truth_events.csv", help="Event ground-truth CSV.")
    parser.add_argument("--yolo-weights", default="yolo11m.pt", help="YOLO11 weights path.")
    parser.add_argument("--rtdetr-weights", default="rtdetr-l.pt", help="RT-DETR weights path or name.")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["bytetrack", "ocsort"],
        help="Tracker configs or short names under configs/.",
    )
    parser.add_argument("--report-dir", default=None, help="Output report directory.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for inference.")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional frame cap for evaluation.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--min-iou", type=float, default=0.10, help="Minimum temporal IoU for matching.")
    parser.add_argument("--min-gt-coverage", type=float, default=0.30, help="Minimum GT coverage for matching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else (ROOT / "reports" / "detector_tracker_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    config_path = _write_run_config(
        base_config=args.config,
        output_path=report_dir / "configs" / "scene_config.yaml",
        source=args.source,
        report_dir=report_dir,
    )

    detectors = [
        ("yolo11m", "yolo", args.yolo_weights),
        ("rtdetr", "rtdetr", args.rtdetr_weights),
    ]
    trackers = [(Path(_resolve_tracker_config(value)).stem, _resolve_tracker_config(value)) for value in args.trackers]

    rows: list[dict[str, Any]] = []
    for detector_name, detector_family, detector_weights in detectors:
        for tracker_name, tracker_config in trackers:
            print(f"\n[{detector_name}/{tracker_name}] source={args.source}")
            pipeline = TrafficAnomalyPipeline(
                config_path=config_path,
                max_frames=args.max_frames,
                display=False,
                source_mode="local",
                source_override=args.source,
                tracker_config_override=str(tracker_config),
                skip_frames=args.skip_frames,
                device=args.device,
                appearance_model="none",
                detector=detector_family,
                detector_weights=detector_weights,
                save_evidence=False,
                save_normal_sequences=False,
                save_tracklets=False,
            )
            run_dir = pipeline.run()
            summary = evaluate_run(
                run_dir=run_dir,
                gt_path=Path(args.gt),
                min_iou=args.min_iou,
                min_gt_coverage=args.min_gt_coverage,
                frame_end=args.frame_end,
            )
            write_evaluation_outputs(summary)
            rows.append(
                _summary_dict(
                    detector=detector_name,
                    detector_weights=detector_weights,
                    tracker=tracker_name,
                    run_dir=run_dir,
                    summary=summary,
                )
            )
            print(
                f"Metrics: GT={summary.total_gt} Pred={summary.total_pred} "
                f"TP={len(summary.matches)} FP={len(summary.false_positives)} "
                f"FN={len(summary.false_negatives)} F1={summary.f1 * 100:.1f}%"
            )
            _write_outputs(rows, report_dir, args.frame_end)
            _release_torch_cache()

    print("\nComparison complete.")
    _write_outputs(rows, report_dir, args.frame_end)


if __name__ == "__main__":
    main()
