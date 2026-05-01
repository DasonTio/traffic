from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ground_truth import (
    EvaluationSummary,
    evaluate_run,
    find_latest_run,
    load_gt_events,
    within_window,
    write_evaluation_outputs,
)
from traffic_anomaly.appearance_eval import (
    AppearanceEvaluationError,
    AppearanceEvaluationSummary,
    evaluate_appearance_model,
    write_appearance_outputs,
)
from traffic_anomaly.config import SceneConfig
from traffic_anomaly.ganomaly import GANomalyScorer
from traffic_anomaly.pipeline import TrafficAnomalyPipeline
from traffic_anomaly.vae import VAEScorer


class EvaluationSetupError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the traffic anomaly model and score it against ground-truth events."
    )
    parser.add_argument("--mode", choices=["system", "appearance"], default="system", help="Evaluation mode.")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Scene config path.")
    parser.add_argument(
        "--source-mode",
        choices=["youtube", "local"],
        default="local",
        help="Named source from config. Defaults to local for reproducible testing.",
    )
    parser.add_argument("--source", default=None, help="Optional direct source override.")
    parser.add_argument("--tracker-config", default=None, help="Optional tracker config override.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for faster test runs.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--run", default=None, help="Existing run directory name under runs/. Skips inference.")
    parser.add_argument("--run-dir", default=None, help="Existing run directory path. Skips inference.")
    parser.add_argument("--runs-root", default="runs", help="Directory containing run folders.")
    parser.add_argument("--gt", default="dataset/ground_truth_events.csv", help="Ground truth CSV path.")
    parser.add_argument(
        "--appearance-gt",
        default="dataset/appearance_ground_truth.csv",
        help="Appearance ground truth CSV used when --mode appearance.",
    )
    parser.add_argument(
        "--appearance-model",
        choices=["ganomaly", "vae"],
        default="ganomaly",
        help="Appearance model to test when --mode appearance.",
    )
    parser.add_argument("--report-dir", default=None, help="Optional report output directory.")
    parser.add_argument("--min-iou", type=float, default=0.10, help="Minimum temporal IoU for a match.")
    parser.add_argument("--min-gt-coverage", type=float, default=0.30, help="Minimum GT coverage ratio for a match.")
    parser.add_argument("--require-lane", action="store_true", help="Require lane_id to match when both are present.")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional evaluation window start frame.")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional evaluation window end frame.")
    parser.add_argument("--min-precision", type=float, default=0.80, help="Pass threshold for precision.")
    parser.add_argument("--min-recall", type=float, default=0.70, help="Pass threshold for recall.")
    parser.add_argument("--min-f1", type=float, default=0.75, help="Pass threshold for F1.")
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace) -> Path | None:
    if args.run_dir:
        return Path(args.run_dir).resolve()
    if args.run:
        return (Path(args.runs_root) / args.run).resolve()
    return None


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _csv_int_values(path: Path, *fields: str) -> list[int]:
    if not path.exists():
        return []
    values: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in fields:
                value = row.get(field, "")
                if value:
                    values.append(int(float(value)))
    return values


def get_gt_frame_range(gt_path: Path, frame_start: int | None = None, frame_end: int | None = None) -> tuple[int | None, int | None]:
    events = [event for event in load_gt_events(gt_path.resolve()) if within_window(event, frame_start, frame_end)]
    if not events:
        return None, None
    return min(event.start_frame for event in events), max(event.end_frame for event in events)


def get_run_frame_range(run_dir: Path) -> tuple[int | None, int | None]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_min = data.get("processed_frame_min")
        frame_max = data.get("processed_frame_max")
        if frame_min is not None and frame_max is not None:
            return int(frame_min), int(frame_max)

    frames = _csv_int_values(run_dir / "tracklets.csv", "frame_idx")
    if frames:
        return min(frames), max(frames)

    frames = _csv_int_values(run_dir / "events.csv", "start_frame", "end_frame")
    if frames:
        return min(frames), max(frames)
    return None, None


def validate_run_against_ground_truth(
    run_dir: Path,
    gt_path: Path,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> None:
    gt_min, gt_max = get_gt_frame_range(gt_path, frame_start=frame_start, frame_end=frame_end)
    if gt_min is None or gt_max is None:
        return

    run_min, run_max = get_run_frame_range(run_dir)
    if run_min is not None and run_max is not None:
        if run_max < gt_min or run_min > gt_max:
            raise EvaluationSetupError(
                f"No frame overlap between run and ground truth. Run covers frames {run_min}-{run_max}, "
                f"but ground truth covers {gt_min}-{gt_max}. Test the same video that was annotated."
            )
        return

    tracklet_rows = _count_csv_rows(run_dir / "tracklets.csv")
    event_rows = _count_csv_rows(run_dir / "events.csv")
    if tracklet_rows == 0 and event_rows == 0:
        raise EvaluationSetupError(
            "Run produced no tracklets or events, so it cannot be meaningfully compared to non-empty ground truth. "
            "This usually means the wrong source video was tested or the source clip is too short."
        )


def build_test_result(
    summary: EvaluationSummary,
    *,
    min_precision: float,
    min_recall: float,
    min_f1: float,
) -> dict[str, object]:
    checks = {
        "precision": summary.precision >= min_precision,
        "recall": summary.recall >= min_recall,
        "f1": summary.f1 >= min_f1,
    }
    passed = all(checks.values())
    return {
        "run_dir": str(summary.run_dir),
        "ground_truth": str(summary.gt_path),
        "events_path": str(summary.events_path),
        "metrics": {
            "gt_events": summary.total_gt,
            "pred_events": summary.total_pred,
            "true_positives": len(summary.matches),
            "false_negatives": len(summary.false_negatives),
            "false_positives": len(summary.false_positives),
            "precision": round(summary.precision, 6),
            "recall": round(summary.recall, 6),
            "f1": round(summary.f1, 6),
            "lane_agreement": round(summary.lane_agreement, 6),
            "class_agreement": round(summary.class_agreement, 6),
        },
        "thresholds": {
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_f1": min_f1,
        },
        "checks": checks,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def build_test_report(summary: EvaluationSummary, result: dict[str, object]) -> str:
    metrics = result["metrics"]
    thresholds = result["thresholds"]
    checks = result["checks"]
    verdict = str(result["verdict"])
    return "\n".join(
        [
            "# Model Test Report",
            "",
            f"Run: `{summary.run_dir.name}`",
            f"Ground truth: `{summary.gt_path}`",
            f"Verdict: **{verdict}**",
            "",
            "## Metrics",
            "",
            f"- GT events: {metrics['gt_events']}",
            f"- Predicted events: {metrics['pred_events']}",
            f"- True positives: {metrics['true_positives']}",
            f"- False negatives: {metrics['false_negatives']}",
            f"- False positives: {metrics['false_positives']}",
            f"- Precision: {summary.precision*100:.1f}%",
            f"- Recall: {summary.recall*100:.1f}%",
            f"- F1: {summary.f1*100:.1f}%",
            f"- Lane agreement: {summary.lane_agreement*100:.1f}%",
            f"- Class agreement: {summary.class_agreement*100:.1f}%",
            "",
            "## Acceptance Gates",
            "",
            f"- Precision >= {thresholds['min_precision']*100:.1f}%: {'PASS' if checks['precision'] else 'FAIL'}",
            f"- Recall >= {thresholds['min_recall']*100:.1f}%: {'PASS' if checks['recall'] else 'FAIL'}",
            f"- F1 >= {thresholds['min_f1']*100:.1f}%: {'PASS' if checks['f1'] else 'FAIL'}",
            "",
            "## Artifacts",
            "",
            f"- Ground-truth evaluation report: `{summary.run_dir / 'ground_truth_eval' / 'ground_truth_report.md'}`",
            f"- Matched events CSV: `{summary.run_dir / 'ground_truth_eval' / 'matched_events.csv'}`",
            f"- False negatives CSV: `{summary.run_dir / 'ground_truth_eval' / 'false_negatives.csv'}`",
            f"- False positives CSV: `{summary.run_dir / 'ground_truth_eval' / 'false_positives.csv'}`",
            "",
        ]
    )


def _appearance_report_dir(model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path("reports") / "appearance_test" / f"{timestamp}_{model_name}").resolve()


def _load_appearance_scorer(args: argparse.Namespace):
    scene = SceneConfig.load(args.config)
    if args.appearance_model == "ganomaly":
        return GANomalyScorer(
            scene.ganomaly_checkpoints,
            image_size=int(scene.ganomaly_settings.get("image_size", 64)),
            default_threshold=float(scene.ganomaly_settings.get("default_threshold", 0.02)),
        )
    return VAEScorer(
        scene.vae_checkpoints,
        image_size=int(scene.vae_settings.get("image_size", 64)),
        default_threshold=float(scene.vae_settings.get("default_threshold", 0.02)),
    )


def build_appearance_test_result(
    summary: AppearanceEvaluationSummary,
    *,
    min_precision: float,
    min_recall: float,
    min_f1: float,
) -> dict[str, object]:
    checks = {
        "precision": summary.metrics.precision >= min_precision,
        "recall": summary.metrics.recall >= min_recall,
        "f1": summary.metrics.f1 >= min_f1,
    }
    passed = all(checks.values())
    return {
        "model_name": summary.model_name,
        "ground_truth": str(summary.gt_path),
        "threshold": summary.threshold,
        "metrics": {
            "samples": summary.metrics.total,
            "positives": summary.metrics.positives,
            "negatives": summary.metrics.negatives,
            "true_positives": summary.metrics.true_positives,
            "false_positives": summary.metrics.false_positives,
            "false_negatives": summary.metrics.false_negatives,
            "precision": round(summary.metrics.precision, 6),
            "recall": round(summary.metrics.recall, 6),
            "f1": round(summary.metrics.f1, 6),
            "auroc": round(summary.metrics.auroc, 6),
            "auprc": round(summary.metrics.auprc, 6),
        },
        "thresholds": {
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_f1": min_f1,
        },
        "checks": checks,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def build_appearance_test_report(summary: AppearanceEvaluationSummary, result: dict[str, object]) -> str:
    metrics = result["metrics"]
    thresholds = result["thresholds"]
    checks = result["checks"]
    return "\n".join(
        [
            "# Appearance Model Test Report",
            "",
            f"Model: `{summary.model_name}`",
            f"Ground truth: `{summary.gt_path}`",
            f"Verdict: **{result['verdict']}**",
            "",
            "## Metrics",
            "",
            f"- Samples: {metrics['samples']}",
            f"- Positives: {metrics['positives']}",
            f"- Negatives: {metrics['negatives']}",
            f"- Precision: {summary.metrics.precision*100:.1f}%",
            f"- Recall: {summary.metrics.recall*100:.1f}%",
            f"- F1: {summary.metrics.f1*100:.1f}%",
            f"- AUROC: {summary.metrics.auroc:.3f}",
            f"- AUPRC: {summary.metrics.auprc:.3f}",
            "",
            "## Acceptance Gates",
            "",
            f"- Precision >= {thresholds['min_precision']*100:.1f}%: {'PASS' if checks['precision'] else 'FAIL'}",
            f"- Recall >= {thresholds['min_recall']*100:.1f}%: {'PASS' if checks['recall'] else 'FAIL'}",
            f"- F1 >= {thresholds['min_f1']*100:.1f}%: {'PASS' if checks['f1'] else 'FAIL'}",
            "",
        ]
    )


def run_appearance_test(args: argparse.Namespace) -> None:
    scorer = _load_appearance_scorer(args)
    try:
        summary = evaluate_appearance_model(scorer, args.appearance_gt)
    except (FileNotFoundError, AppearanceEvaluationError) as exc:
        print(str(exc))
        sys.exit(1)

    result = build_appearance_test_result(
        summary,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
    )
    output_dir = Path(args.report_dir).resolve() if args.report_dir else _appearance_report_dir(summary.model_name)
    write_appearance_outputs(summary, output_dir)

    summary_path = output_dir / f"{summary.model_name}_test_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = output_dir / f"{summary.model_name}_test_report.md"
    report_path.write_text(build_appearance_test_report(summary, result), encoding="utf-8")

    print(f"Model:      {summary.model_name}")
    print(f"GT:         {summary.gt_path}")
    print(f"Precision:  {summary.metrics.precision*100:.1f}%")
    print(f"Recall:     {summary.metrics.recall*100:.1f}%")
    print(f"F1:         {summary.metrics.f1*100:.1f}%")
    print(f"AUROC:      {summary.metrics.auroc:.3f}")
    print(f"AUPRC:      {summary.metrics.auprc:.3f}")
    print(f"Verdict:    {result['verdict']}")
    print(f"Saved appearance summary to: {summary_path}")
    print(f"Saved appearance report to:  {report_path}")
    sys.exit(0 if bool(result["passed"]) else 2)


def main() -> None:
    args = parse_args()
    if args.mode == "appearance":
        run_appearance_test(args)

    run_dir = _resolve_run_dir(args)

    if run_dir is None:
        pipeline = TrafficAnomalyPipeline(
            config_path=args.config,
            max_frames=args.max_frames,
            display=False,
            source_mode=args.source_mode,
            source_override=args.source,
            tracker_config_override=args.tracker_config,
            skip_frames=args.skip_frames,
            appearance_model=args.appearance_model,
        )
        run_dir = pipeline.run()
    elif not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    try:
        validate_run_against_ground_truth(
            run_dir=run_dir,
            gt_path=Path(args.gt),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
        summary = evaluate_run(
            run_dir=run_dir,
            gt_path=Path(args.gt),
            min_iou=args.min_iou,
            min_gt_coverage=args.min_gt_coverage,
            require_lane=args.require_lane,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    except (FileNotFoundError, EvaluationSetupError) as exc:
        print(str(exc))
        sys.exit(1)

    eval_dir = write_evaluation_outputs(summary)
    result = build_test_result(
        summary,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
    )

    summary_path = eval_dir / "model_test_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = eval_dir / "model_test_report.md"
    report_path.write_text(build_test_report(summary, result), encoding="utf-8")

    latest_run = find_latest_run(Path(args.runs_root))
    print(f"Run directory: {summary.run_dir}")
    if latest_run is not None:
        print(f"Latest run under {Path(args.runs_root).resolve()}: {latest_run.resolve()}")
    print(f"Precision: {summary.precision*100:.1f}%")
    print(f"Recall:    {summary.recall*100:.1f}%")
    print(f"F1:        {summary.f1*100:.1f}%")
    print(f"Verdict:   {result['verdict']}")
    print(f"Saved test summary to: {summary_path}")
    print(f"Saved test report to:  {report_path}")
    sys.exit(0 if bool(result["passed"]) else 2)


if __name__ == "__main__":
    main()
