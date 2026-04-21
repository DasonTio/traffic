from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from traffic_anomaly.appearance import AppearanceScore, AppearanceScorer, class_group_for_name, read_image


LABELED_APPEARANCE_CLASSES = {"normal", "appearance_anomaly"}


class AppearanceEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppearanceGroundTruth:
    sample_id: str
    image_path: Path
    class_name: str
    class_group: str
    label: str
    source_type: str
    source_run: str
    event_id: str
    track_id: str
    frame_idx: int | None
    notes: str
    raw: dict[str, str]


@dataclass(frozen=True)
class AppearancePrediction:
    sample_id: str
    image_path: str
    class_name: str
    class_group: str
    label: str
    source_type: str
    source_run: str
    event_id: str
    track_id: str
    frame_idx: int | None
    model_name: str
    raw_score: float
    threshold: float
    normalized_score: float
    predicted_label: str


@dataclass(frozen=True)
class AppearanceMetrics:
    total: int
    positives: int
    negatives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float


@dataclass(frozen=True)
class AppearanceEvaluationSummary:
    model_name: str
    gt_path: Path
    threshold: float
    total_samples: int
    metrics: AppearanceMetrics
    predictions: list[AppearancePrediction]
    per_group: dict[str, AppearanceMetrics]


def _resolve_image_path(gt_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    search_roots = [gt_path.parent, gt_path.parent.parent, Path.cwd()]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (gt_path.parent / candidate).resolve()


def load_appearance_ground_truth(path: str | Path, *, labeled_only: bool = True) -> list[AppearanceGroundTruth]:
    gt_path = Path(path).resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Appearance ground truth not found: {gt_path}")

    rows: list[AppearanceGroundTruth] = []
    with gt_path.open("r", newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            label = row.get("label", "").strip().lower()
            if labeled_only and label not in LABELED_APPEARANCE_CLASSES:
                continue
            image_value = row.get("image_path", "").strip()
            if not image_value:
                continue
            class_name = row.get("class_name", "Unknown").strip() or "Unknown"
            class_group = class_group_for_name(class_name)
            frame_idx_raw = row.get("frame_idx", "").strip()
            rows.append(
                AppearanceGroundTruth(
                    sample_id=row.get("sample_id", f"sample_{idx}"),
                    image_path=_resolve_image_path(gt_path, image_value),
                    class_name=class_name,
                    class_group=class_group,
                    label=label,
                    source_type=row.get("source_type", "").strip(),
                    source_run=row.get("source_run", "").strip(),
                    event_id=row.get("event_id", "").strip(),
                    track_id=row.get("track_id", "").strip(),
                    frame_idx=int(frame_idx_raw) if frame_idx_raw else None,
                    notes=row.get("notes", "").strip(),
                    raw=row,
                )
            )
    if labeled_only and not rows:
        raise AppearanceEvaluationError(
            f"No labeled appearance rows found in {gt_path}. Add labels 'normal' or 'appearance_anomaly' first."
        )
    return rows


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise AppearanceEvaluationError("Appearance evaluation needs both positive and negative labeled samples.")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, labels.size - 1]

    tps = np.cumsum(sorted_labels)[threshold_indices]
    fps = 1 + threshold_indices - tps

    tpr = tps / positives
    fpr = fps / negatives
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives

    auroc = float(np.trapz(np.r_[0.0, tpr, 1.0], np.r_[0.0, fpr, 1.0]))
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auroc, auprc


def compute_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 1.0,
    *,
    allow_single_class: bool = False,
) -> AppearanceMetrics:
    y_true = np.asarray(list(labels), dtype=np.int32)
    y_score = np.asarray(list(scores), dtype=np.float64)
    if y_true.size == 0:
        raise AppearanceEvaluationError("Appearance evaluation received zero samples.")

    try:
        auroc, auprc = _binary_auc(y_true, y_score)
    except AppearanceEvaluationError:
        if not allow_single_class:
            raise
        auroc, auprc = 0.0, 0.0
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return AppearanceMetrics(
        total=int(y_true.size),
        positives=int(np.sum(y_true == 1)),
        negatives=int(np.sum(y_true == 0)),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        auprc=auprc,
    )


def evaluate_appearance_model(
    scorer: AppearanceScorer,
    gt_path: str | Path,
    *,
    threshold: float = 1.0,
) -> AppearanceEvaluationSummary:
    if not scorer.available():
        raise AppearanceEvaluationError("Appearance scorer has no valid checkpoints loaded.")

    samples = load_appearance_ground_truth(gt_path, labeled_only=True)
    predictions: list[AppearancePrediction] = []
    labels: list[int] = []
    scores: list[float] = []
    group_labels: dict[str, list[int]] = {}
    group_scores: dict[str, list[float]] = {}
    model_name: str | None = None

    for sample in samples:
        image = read_image(sample.image_path)
        score = scorer.score_crop_details(image, sample.class_name)
        model_name = model_name or score.model_name
        label_value = 1 if sample.label == "appearance_anomaly" else 0

        labels.append(label_value)
        scores.append(score.normalized_score)
        group_labels.setdefault(sample.class_group, []).append(label_value)
        group_scores.setdefault(sample.class_group, []).append(score.normalized_score)
        predictions.append(
            AppearancePrediction(
                sample_id=sample.sample_id,
                image_path=str(sample.image_path),
                class_name=sample.class_name,
                class_group=sample.class_group,
                label=sample.label,
                source_type=sample.source_type,
                source_run=sample.source_run,
                event_id=sample.event_id,
                track_id=sample.track_id,
                frame_idx=sample.frame_idx,
                model_name=score.model_name,
                raw_score=score.raw_score,
                threshold=score.threshold,
                normalized_score=score.normalized_score,
                predicted_label="appearance_anomaly" if score.is_anomaly else "normal",
            )
        )

    per_group: dict[str, AppearanceMetrics] = {}
    for class_group, group_y_true in group_labels.items():
        per_group[class_group] = compute_metrics(
            group_y_true,
            group_scores[class_group],
            threshold=threshold,
            allow_single_class=True,
        )

    return AppearanceEvaluationSummary(
        model_name=model_name or "appearance_model",
        gt_path=Path(gt_path).resolve(),
        threshold=threshold,
        total_samples=len(predictions),
        metrics=compute_metrics(labels, scores, threshold=threshold),
        predictions=predictions,
        per_group=per_group,
    )


def _metrics_to_dict(metrics: AppearanceMetrics) -> dict[str, float | int]:
    return asdict(metrics)


def build_appearance_report(summary: AppearanceEvaluationSummary) -> str:
    metrics = summary.metrics
    lines = [
        f"# {summary.model_name.upper()} Appearance Evaluation",
        "",
        f"Ground truth: `{summary.gt_path}`",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Decision threshold: normalized score >= {summary.threshold:.2f}",
        "",
        "## Overall",
        "",
        f"- Samples: {metrics.total}",
        f"- Positives: {metrics.positives}",
        f"- Negatives: {metrics.negatives}",
        f"- Precision: {metrics.precision*100:.1f}%",
        f"- Recall: {metrics.recall*100:.1f}%",
        f"- F1: {metrics.f1*100:.1f}%",
        f"- AUROC: {metrics.auroc:.3f}",
        f"- AUPRC: {metrics.auprc:.3f}",
        "",
        "## Per Group",
        "",
    ]
    for class_group, group_metrics in sorted(summary.per_group.items()):
        lines.extend(
            [
                f"### {class_group}",
                "",
                f"- Samples: {group_metrics.total}",
                f"- Precision: {group_metrics.precision*100:.1f}%",
                f"- Recall: {group_metrics.recall*100:.1f}%",
                f"- F1: {group_metrics.f1*100:.1f}%",
                f"- AUROC: {group_metrics.auroc:.3f}",
                f"- AUPRC: {group_metrics.auprc:.3f}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_appearance_outputs(summary: AppearanceEvaluationSummary, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / f"{summary.model_name}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summary.predictions[0]).keys()))
        writer.writeheader()
        for prediction in summary.predictions:
            writer.writerow(asdict(prediction))

    payload = {
        "model_name": summary.model_name,
        "ground_truth": str(summary.gt_path),
        "threshold": summary.threshold,
        "total_samples": summary.total_samples,
        "metrics": _metrics_to_dict(summary.metrics),
        "per_group": {group: _metrics_to_dict(metrics) for group, metrics in summary.per_group.items()},
    }
    json_path = output_path / f"{summary.model_name}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = output_path / f"{summary.model_name}_report.md"
    report_path.write_text(build_appearance_report(summary), encoding="utf-8")
    return output_path


def build_comparison_report(summaries: list[AppearanceEvaluationSummary]) -> str:
    lines = [
        "# Appearance Model Comparison",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Model | Samples | Precision | Recall | F1 | AUROC | AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        metrics = summary.metrics
        lines.append(
            "| "
            f"{summary.model_name} | "
            f"{metrics.total} | "
            f"{metrics.precision*100:.1f}% | "
            f"{metrics.recall*100:.1f}% | "
            f"{metrics.f1*100:.1f}% | "
            f"{metrics.auroc:.3f} | "
            f"{metrics.auprc:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_comparison_outputs(summaries: list[AppearanceEvaluationSummary], output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    write_appearance_outputs(summaries[0], output_path)
    for summary in summaries[1:]:
        write_appearance_outputs(summary, output_path)

    comparison_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": [
            {
                "model_name": summary.model_name,
                "metrics": _metrics_to_dict(summary.metrics),
                "per_group": {group: _metrics_to_dict(metrics) for group, metrics in summary.per_group.items()},
            }
            for summary in summaries
        ],
    }
    (output_path / "comparison_summary.json").write_text(
        json.dumps(comparison_payload, indent=2),
        encoding="utf-8",
    )
    (output_path / "comparison_report.md").write_text(build_comparison_report(summaries), encoding="utf-8")
    return output_path
