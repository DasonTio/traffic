from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ground_truth import EvaluationSummary, evaluate_run, write_evaluation_outputs
from traffic_anomaly.pipeline import TrafficAnomalyPipeline


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float


@dataclass(frozen=True)
class RunMetric:
    source_label: str
    tracker: str
    appearance_model: str
    run_dir: Path
    summary: EvaluationSummary
    config_path: Path

    @property
    def pipeline_name(self) -> str:
        return f"{self.tracker}/{self.appearance_model}"


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_original() -> Path:
    return _first_existing(
        [
            ROOT.parent / "traffic_data_5mins.mp4",
            ROOT / ".video" / "video.mp4.mp4",
        ]
    )


def _default_upscaled() -> Path:
    return _first_existing(
        [
            ROOT.parent / "1920x1440-traffic.mp4",
            ROOT.parent / "upscaled_5min_traffic.mp4",
            ROOT / ".video" / "1920x1440-traffic.mp4",
        ]
    )


def video_info(path: str | Path) -> VideoInfo:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video not found: {resolved}")
    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {resolved}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoInfo(
        path=resolved,
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration_s=frames / fps if fps > 0 else 0.0,
    )


def _scale_point(point: list[Any], scale_x: float, scale_y: float, *, as_int: bool) -> list[int | float]:
    x = float(point[0]) * scale_x
    y = float(point[1]) * scale_y
    if as_int:
        return [int(round(x)), int(round(y))]
    return [round(x, 6), round(y, 6)]


def write_scaled_scene_config(
    *,
    base_config: str | Path,
    output_path: str | Path,
    source_path: str | Path,
    scale_x: float,
    scale_y: float,
    run_root: str | Path,
    dataset_dir: str | Path,
) -> Path:
    base_path = Path(base_config).resolve()
    output = Path(output_path).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    scaled = copy.deepcopy(raw)
    video_cfg = scaled.setdefault("video", {})
    video_cfg["default"] = "local"
    video_cfg["local"] = str(Path(source_path).resolve())

    output_cfg = scaled.setdefault("output", {})
    output_cfg["run_root"] = str(Path(run_root).resolve())
    output_cfg["dataset_dir"] = str(Path(dataset_dir).resolve())

    homography = scaled.get("homography", {})
    if "src_points" in homography:
        homography["src_points"] = [
            _scale_point(point, scale_x, scale_y, as_int=False)
            for point in homography["src_points"]
        ]

    for lane in scaled.get("lanes", []):
        if "polygon" in lane:
            lane["polygon"] = [
                _scale_point(point, scale_x, scale_y, as_int=True)
                for point in lane["polygon"]
            ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scaled, handle, sort_keys=False)
    return output


def _resolve_tracker_config(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix:
        return (ROOT / candidate).resolve()
    return (ROOT / "configs" / f"{value}.yaml").resolve()


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summary_payload(metric: RunMetric) -> dict[str, Any]:
    summary = metric.summary
    return {
        "source": metric.source_label,
        "tracker": metric.tracker,
        "appearance_model": metric.appearance_model,
        "pipeline": metric.pipeline_name,
        "run_dir": str(metric.run_dir),
        "config_path": str(metric.config_path),
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


def build_report(
    *,
    original: VideoInfo,
    upscaled: VideoInfo,
    metrics: list[RunMetric],
    frame_start: int | None,
    frame_end: int | None,
) -> str:
    by_pipeline: dict[str, dict[str, RunMetric]] = {}
    for metric in metrics:
        by_pipeline.setdefault(metric.pipeline_name, {})[metric.source_label] = metric

    lines = [
        "# Original vs Upscaled Video Metrics",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Sources",
        "",
        "| Source | Path | Resolution | FPS | Frames | Duration |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| original | `{original.path}` | {original.width}x{original.height} | {original.fps:.3f} | {original.frames} | {original.duration_s:.1f}s |",
        f"| upscaled | `{upscaled.path}` | {upscaled.width}x{upscaled.height} | {upscaled.fps:.3f} | {upscaled.frames} | {upscaled.duration_s:.1f}s |",
        "",
        f"Evaluation window: frames {frame_start if frame_start is not None else 1} to {frame_end if frame_end is not None else 'end'}.",
        "The upscaled config scales homography source points and lane polygons to match the upscaled resolution.",
        "",
        "## Metrics",
        "",
        "| Pipeline | Source | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Lane Agree | Class Agree |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for pipeline in sorted(by_pipeline):
        for source in ("original", "upscaled"):
            metric = by_pipeline[pipeline].get(source)
            if metric is None:
                continue
            summary = metric.summary
            lines.append(
                f"| {pipeline} | {source} | {summary.total_gt} | {summary.total_pred} | "
                f"{len(summary.matches)} | {len(summary.false_positives)} | {len(summary.false_negatives)} | "
                f"{_pct(summary.precision)} | {_pct(summary.recall)} | {_pct(summary.f1)} | "
                f"{_pct(summary.lane_agreement)} | {_pct(summary.class_agreement)} |"
            )

    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| Pipeline | Delta Pred | Delta TP | Delta FP | Delta FN | Delta Precision | Delta Recall | Delta F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pipeline in sorted(by_pipeline):
        original_metric = by_pipeline[pipeline].get("original")
        upscaled_metric = by_pipeline[pipeline].get("upscaled")
        if original_metric is None or upscaled_metric is None:
            continue
        orig = original_metric.summary
        up = upscaled_metric.summary
        lines.append(
            f"| {pipeline} | {up.total_pred - orig.total_pred:+d} | "
            f"{len(up.matches) - len(orig.matches):+d} | "
            f"{len(up.false_positives) - len(orig.false_positives):+d} | "
            f"{len(up.false_negatives) - len(orig.false_negatives):+d} | "
            f"{(up.precision - orig.precision) * 100:+.1f} pp | "
            f"{(up.recall - orig.recall) * 100:+.1f} pp | "
            f"{(up.f1 - orig.f1) * 100:+.1f} pp |"
        )

    lines.extend(
        [
            "",
            "## Run Artifacts",
            "",
            "| Pipeline | Source | Run Dir | Evaluation Report | Config |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for metric in sorted(metrics, key=lambda item: (item.pipeline_name, item.source_label)):
        report = metric.run_dir / "ground_truth_eval" / "ground_truth_report.md"
        lines.append(
            f"| {metric.pipeline_name} | {metric.source_label} | `{metric.run_dir}` | "
            f"`{report}` | `{metric.config_path}` |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tracker/appearance model combinations on original and upscaled videos."
    )
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Base scene config.")
    parser.add_argument("--original", default=str(_default_original()), help="Original video path.")
    parser.add_argument("--upscaled", default=str(_default_upscaled()), help="Upscaled video path.")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["bytetrack", "ocsort"],
        help="Tracker configs or short names under configs/.",
    )
    parser.add_argument(
        "--appearance-models",
        nargs="+",
        choices=["ganomaly", "vae"],
        default=["ganomaly", "vae"],
        help="Appearance scorers to run.",
    )
    parser.add_argument("--gt", default="dataset/ground_truth_events.csv", help="Ground truth CSV.")
    parser.add_argument("--report-dir", default=None, help="Output report directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--frame-start", type=int, default=None, help="Evaluation window start frame.")
    parser.add_argument("--frame-end", type=int, default=None, help="Evaluation window end frame.")
    parser.add_argument("--min-iou", type=float, default=0.10, help="Minimum temporal IoU for GT matching.")
    parser.add_argument("--min-gt-coverage", type=float, default=0.30, help="Minimum GT coverage for matching.")
    return parser.parse_args()


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def main() -> None:
    args = parse_args()
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else (ROOT / "reports" / "video_source_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    configs_dir = report_dir / "configs"
    runs_root = report_dir / "runs"
    datasets_root = report_dir / "datasets"
    report_dir.mkdir(parents=True, exist_ok=True)

    original = video_info(args.original)
    upscaled = video_info(args.upscaled)
    frame_end = args.frame_end if args.frame_end is not None else min(original.frames, upscaled.frames)
    if args.max_frames is not None:
        frame_end = min(frame_end, args.max_frames)

    original_config = write_scaled_scene_config(
        base_config=args.config,
        output_path=configs_dir / "original_scene_config.yaml",
        source_path=original.path,
        scale_x=1.0,
        scale_y=1.0,
        run_root=runs_root,
        dataset_dir=datasets_root / "original",
    )
    upscaled_config = write_scaled_scene_config(
        base_config=args.config,
        output_path=configs_dir / "upscaled_scene_config.yaml",
        source_path=upscaled.path,
        scale_x=upscaled.width / max(1, original.width),
        scale_y=upscaled.height / max(1, original.height),
        run_root=runs_root,
        dataset_dir=datasets_root / "upscaled",
    )

    tracker_configs = [_resolve_tracker_config(value) for value in args.trackers]
    metrics: list[RunMetric] = []
    sources = [
        ("original", original.path, original_config),
        ("upscaled", upscaled.path, upscaled_config),
    ]

    for source_label, source_path, config_path in sources:
        for tracker_config in tracker_configs:
            for appearance_model in args.appearance_models:
                tracker_name = tracker_config.stem
                print(
                    f"\n[{source_label}] tracker={tracker_name} appearance={appearance_model} "
                    f"source={source_path}"
                )
                pipeline = TrafficAnomalyPipeline(
                    config_path=config_path,
                    max_frames=args.max_frames,
                    display=False,
                    source_mode="local",
                    source_override=str(source_path),
                    tracker_config_override=str(tracker_config),
                    skip_frames=args.skip_frames,
                    device=args.device,
                    appearance_model=appearance_model,
                )
                run_dir = pipeline.run()
                summary = evaluate_run(
                    run_dir=run_dir,
                    gt_path=Path(args.gt),
                    min_iou=args.min_iou,
                    min_gt_coverage=args.min_gt_coverage,
                    frame_start=args.frame_start,
                    frame_end=frame_end,
                )
                write_evaluation_outputs(summary)
                metrics.append(
                    RunMetric(
                        source_label=source_label,
                        tracker=tracker_name,
                        appearance_model=appearance_model,
                        run_dir=run_dir,
                        summary=summary,
                        config_path=config_path,
                    )
                )
                print(
                    f"Metrics: GT={summary.total_gt} Pred={summary.total_pred} "
                    f"TP={len(summary.matches)} FP={len(summary.false_positives)} "
                    f"FN={len(summary.false_negatives)} F1={summary.f1 * 100:.1f}%"
                )
                _release_torch_cache()

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "original": original.__dict__ | {"path": str(original.path)},
        "upscaled": upscaled.__dict__ | {"path": str(upscaled.path)},
        "frame_start": args.frame_start,
        "frame_end": frame_end,
        "metrics": [_summary_payload(metric) for metric in metrics],
    }
    summary_path = report_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_path = report_dir / "comparison_report.md"
    report_path.write_text(
        build_report(
            original=original,
            upscaled=upscaled,
            metrics=metrics,
            frame_start=args.frame_start,
            frame_end=frame_end,
        ),
        encoding="utf-8",
    )

    print(f"\nComparison complete: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
