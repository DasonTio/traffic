"""
Full evaluation orchestrator.

Runs (in order):
  1. System-level evaluation  — predicted events vs ground_truth_events.csv
  2. GANomaly appearance eval — appearance/anomaly classification metrics
  3. VAE appearance eval      — same
  4. Side-by-side comparison  — written to reports/full_evaluation_<timestamp>/

Usage:
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --run 20260421_193704 --skip-system
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ground_truth import (
    EvaluationSummary,
    evaluate_run,
    find_latest_run,
    write_evaluation_outputs,
)
from traffic_anomaly.appearance_eval import (
    AppearanceEvaluationError,
    AppearanceEvaluationSummary,
    build_comparison_report,
    evaluate_appearance_model,
    write_appearance_outputs,
    write_comparison_outputs,
)
from traffic_anomaly.config import SceneConfig
from traffic_anomaly.ganomaly import GANomalyScorer
from traffic_anomaly.vae import VAEScorer


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _report_dir(label: str = "full_evaluation") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path("reports") / f"{ts}_{label}").resolve()


def _metrics_block(metrics: dict[str, object], indent: str = "  ") -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{indent}{k}: {v:.4f}")
        else:
            lines.append(f"{indent}{k}: {v}")
    return "\n".join(lines)


def _system_summary_dict(summary: EvaluationSummary) -> dict[str, object]:
    return {
        "run_dir": str(summary.run_dir),
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
    }


def _appearance_summary_dict(summary: AppearanceEvaluationSummary) -> dict[str, object]:
    m = summary.metrics
    return {
        "model": summary.model_name,
        "samples": m.total,
        "positives": m.positives,
        "negatives": m.negatives,
        "true_positives": m.true_positives,
        "false_positives": m.false_positives,
        "false_negatives": m.false_negatives,
        "precision": round(m.precision, 6),
        "recall": round(m.recall, 6),
        "f1": round(m.f1, 6),
        "auroc": round(m.auroc, 6),
        "auprc": round(m.auprc, 6),
        "per_group": {
            grp: {
                "precision": round(gm.precision, 6),
                "recall": round(gm.recall, 6),
                "f1": round(gm.f1, 6),
                "auroc": round(gm.auroc, 6),
                "auprc": round(gm.auprc, 6),
            }
            for grp, gm in summary.per_group.items()
        },
    }


def _build_full_report(
    system: dict[str, object] | None,
    appearance_summaries: list[AppearanceEvaluationSummary],
) -> str:
    lines = [
        "# Full Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # ── System-level ────────────────────────────────────────────────────────
    if system:
        lines += [
            "## System-Level Detection (vs ground_truth_events.csv)",
            "",
            f"Run: `{system['run_dir']}`",
            "",
            f"| Metric          | Value |",
            f"|-----------------|-------|",
            f"| GT events       | {system['gt_events']} |",
            f"| Pred events     | {system['pred_events']} |",
            f"| True positives  | {system['true_positives']} |",
            f"| False negatives | {system['false_negatives']} |",
            f"| False positives | {system['false_positives']} |",
            f"| **Precision**   | **{system['precision']*100:.1f}%** |",  # type: ignore[operator]
            f"| **Recall**      | **{system['recall']*100:.1f}%** |",  # type: ignore[operator]
            f"| **F1**          | **{system['f1']*100:.1f}%** |",  # type: ignore[operator]
            f"| Lane agreement  | {system['lane_agreement']*100:.1f}% |",  # type: ignore[operator]
            f"| Class agreement | {system['class_agreement']*100:.1f}% |",  # type: ignore[operator]
            "",
        ]
    else:
        lines += ["## System-Level Detection", "", "_Skipped._", ""]

    # ── Appearance models ────────────────────────────────────────────────────
    if appearance_summaries:
        lines += [
            "## Appearance Model Comparison",
            "",
            "| Model | Samples | Precision | Recall | F1 | AUROC | AUPRC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in appearance_summaries:
            m = s.metrics
            lines.append(
                f"| {s.model_name} | {m.total} | {m.precision*100:.1f}% | "
                f"{m.recall*100:.1f}% | {m.f1*100:.1f}% | "
                f"{m.auroc:.3f} | {m.auprc:.3f} |"
            )
        lines.append("")

        # Per-group breakdown
        for s in appearance_summaries:
            lines += [f"### {s.model_name} — per class group", ""]
            lines += [
                "| Class | Precision | Recall | F1 | AUROC | AUPRC |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for grp, gm in sorted(s.per_group.items()):
                lines.append(
                    f"| {grp} | {gm.precision*100:.1f}% | {gm.recall*100:.1f}% | "
                    f"{gm.f1*100:.1f}% | {gm.auroc:.3f} | {gm.auprc:.3f} |"
                )
            lines.append("")
    else:
        lines += ["## Appearance Models", "", "_No appearance models evaluated._", ""]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full evaluation: system-level + GANomaly + VAE appearance scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Scene config path.")
    parser.add_argument("--run", default=None, help="Run directory name under runs/.")
    parser.add_argument("--run-dir", default=None, help="Explicit path to run directory.")
    parser.add_argument("--runs-root", default="runs", help="Directory containing run folders.")
    parser.add_argument(
        "--gt", default="dataset/ground_truth_events.csv", help="Ground truth events CSV."
    )
    parser.add_argument(
        "--appearance-gt",
        default="dataset/appearance_ground_truth.csv",
        help="Appearance ground truth CSV.",
    )
    parser.add_argument("--min-iou", type=float, default=0.10)
    parser.add_argument("--min-gt-coverage", type=float, default=0.30)
    parser.add_argument(
        "--skip-system",
        action="store_true",
        help="Skip system-level evaluation (useful if re-running appearance only).",
    )
    parser.add_argument(
        "--skip-appearance",
        action="store_true",
        help="Skip appearance model evaluation.",
    )
    parser.add_argument("--report-dir", default=None, help="Optional override for output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = SceneConfig.load(args.config)

    report_dir = Path(args.report_dir).resolve() if args.report_dir else _report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    full_payload: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # ── 1. Resolve run dir ───────────────────────────────────────────────────
    if not args.skip_system:
        if args.run_dir:
            run_dir = Path(args.run_dir).resolve()
        elif args.run:
            run_dir = (Path(args.runs_root) / args.run).resolve()
        else:
            run_dir = find_latest_run(Path(args.runs_root))
            if run_dir is None:
                print("ERROR: No runs found. Run inference first.", file=sys.stderr)
                sys.exit(1)
            run_dir = run_dir.resolve()

        print(f"\n[1/3] System evaluation — run: {run_dir.name}")
        try:
            summary = evaluate_run(
                run_dir=run_dir,
                gt_path=Path(args.gt),
                min_iou=args.min_iou,
                min_gt_coverage=args.min_gt_coverage,
            )
            write_evaluation_outputs(summary)
            sys_dict = _system_summary_dict(summary)
            full_payload["system"] = sys_dict
            print(f"  Precision : {summary.precision*100:.1f}%")
            print(f"  Recall    : {summary.recall*100:.1f}%")
            print(f"  F1        : {summary.f1*100:.1f}%")
        except (FileNotFoundError, Exception) as exc:
            print(f"  WARNING: System evaluation failed — {exc}", file=sys.stderr)
            sys_dict = None
            full_payload["system"] = None
    else:
        print("\n[1/3] System evaluation — SKIPPED")
        sys_dict = None
        full_payload["system"] = None

    # ── 2. Appearance evaluation ─────────────────────────────────────────────
    appearance_summaries: list[AppearanceEvaluationSummary] = []

    if not args.skip_appearance:
        appearance_gt = Path(args.appearance_gt)
        scorers = [
            (
                "ganomaly",
                GANomalyScorer(
                    scene.ganomaly_checkpoints,
                    image_size=int(scene.ganomaly_settings.get("image_size", 64)),
                    default_threshold=float(scene.ganomaly_settings.get("default_threshold", 0.02)),
                ),
            ),
            (
                "vae",
                VAEScorer(
                    scene.vae_checkpoints,
                    image_size=int(scene.vae_settings.get("image_size", 64)),
                    default_threshold=float(scene.vae_settings.get("default_threshold", 0.02)),
                ),
            ),
        ]

        for step, (model_name, scorer) in enumerate(scorers, start=2):
            print(f"\n[{step}/3] {model_name.upper()} appearance evaluation")
            if not scorer.available():
                print(f"  WARNING: No valid checkpoints loaded for {model_name}. Skipping.")
                continue
            try:
                app_summary = evaluate_appearance_model(scorer, appearance_gt)
                model_report_dir = report_dir / model_name
                write_appearance_outputs(app_summary, model_report_dir)
                appearance_summaries.append(app_summary)
                full_payload[model_name] = _appearance_summary_dict(app_summary)
                m = app_summary.metrics
                print(f"  Precision : {m.precision*100:.1f}%")
                print(f"  Recall    : {m.recall*100:.1f}%")
                print(f"  F1        : {m.f1*100:.1f}%")
                print(f"  AUROC     : {m.auroc:.3f}")
                print(f"  AUPRC     : {m.auprc:.3f}")
            except (FileNotFoundError, AppearanceEvaluationError) as exc:
                print(f"  WARNING: {exc}", file=sys.stderr)
    else:
        print("\n[2/3] GANomaly appearance evaluation — SKIPPED")
        print("[3/3] VAE appearance evaluation — SKIPPED")

    # ── 3. Comparison + full report ──────────────────────────────────────────
    if len(appearance_summaries) >= 2:
        write_comparison_outputs(appearance_summaries, report_dir)

    full_report = _build_full_report(sys_dict, appearance_summaries)
    full_report_path = report_dir / "full_evaluation_report.md"
    full_report_path.write_text(full_report, encoding="utf-8")

    payload_path = report_dir / "full_evaluation_summary.json"
    payload_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f" Full evaluation complete")
    print(f" Report dir : {report_dir}")
    print(f" Report     : {full_report_path}")
    print(f" Summary    : {payload_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
