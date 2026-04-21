from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ground_truth import EventSpan, load_gt_events, within_window
from traffic_anomaly.config import SceneConfig
from traffic_anomaly.pipeline import TrafficAnomalyPipeline, enhance_frame, extract_crop, open_capture
from traffic_anomaly.storage import RunArtifacts
from traffic_anomaly.tracker_backend import TrackerBackend


APPEARANCE_CLASSES = {"Car", "Bus", "Truck"}


class MiningSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameInterval:
    start_frame: int
    end_frame: int


def merge_intervals(intervals: list[FrameInterval]) -> list[FrameInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.start_frame, item.end_frame))
    merged: list[FrameInterval] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_frame <= previous.end_frame + 1:
            merged[-1] = FrameInterval(previous.start_frame, max(previous.end_frame, current.end_frame))
            continue
        merged.append(current)
    return merged


def build_exclusion_intervals(
    events: list[EventSpan],
    *,
    buffer_frames: int,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> list[FrameInterval]:
    raw: list[FrameInterval] = []
    for event in events:
        if not within_window(event, frame_start, frame_end):
            continue
        start_frame = max(1, event.start_frame - buffer_frames)
        end_frame = event.end_frame + buffer_frames
        if frame_start is not None:
            start_frame = max(start_frame, frame_start)
        if frame_end is not None:
            end_frame = min(end_frame, frame_end)
        if start_frame <= end_frame:
            raw.append(FrameInterval(start_frame, end_frame))
    return merge_intervals(raw)


def frame_is_excluded(frame_idx: int, intervals: list[FrameInterval], interval_index: int) -> tuple[bool, int]:
    while interval_index < len(intervals) and frame_idx > intervals[interval_index].end_frame:
        interval_index += 1
    if interval_index >= len(intervals):
        return False, interval_index
    interval = intervals[interval_index]
    return interval.start_frame <= frame_idx <= interval.end_frame, interval_index


def validate_video_against_gt(total_frames: int, gt_path: Path, intervals: list[FrameInterval], source: str) -> None:
    if not intervals:
        raise MiningSetupError(f"No anomaly intervals found in {gt_path}.")
    if total_frames <= 0:
        return
    required_frames = max(interval.end_frame for interval in intervals)
    if total_frames < required_frames:
        raise MiningSetupError(
            f"Video '{source}' has {total_frames} frames, but ground truth reaches frame {required_frames}. "
            "Use the annotated source video instead of the short local smoke clip."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine normal training sequences by excluding anomaly windows defined in ground truth."
    )
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Scene config path.")
    parser.add_argument("--gt", default="dataset/ground_truth_events.csv", help="Ground truth events CSV.")
    parser.add_argument(
        "--source-mode",
        choices=["youtube", "local"],
        default="local",
        help="Named source from config. Defaults to local for reproducible mining.",
    )
    parser.add_argument("--source", default=None, help="Optional direct source override.")
    parser.add_argument("--tracker-config", default=None, help="Optional tracker config override.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional frame start.")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional frame end.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=30,
        help="Frames to exclude before and after each anomaly interval.",
    )
    parser.add_argument(
        "--min-sequence-frames",
        type=int,
        default=None,
        help="Override minimum frames required to save a sequence.",
    )
    return parser.parse_args()


def _open_video_capture(scene: SceneConfig):
    source = scene.video_source
    try:
        cap = open_capture(source, scene.video_resolution)
        TrafficAnomalyPipeline._validate_capture(cap, source)
        return cap
    except Exception as exc:
        raise MiningSetupError(TrafficAnomalyPipeline._format_source_error(source, exc)) from exc


def _build_tracker_backend(scene: SceneConfig, model, fps: float):
    return TrackerBackend(
        model=model,
        tracker_config=scene.tracker_config,
        detect_classes=scene.detect_classes,
        default_confidence=scene.confidence,
        fps=fps,
    )


def _finalize_all_candidates(
    artifacts: RunArtifacts,
    sequence_candidates: dict[int, dict],
    saved_counts: Counter[str],
) -> None:
    for track_id in list(sequence_candidates):
        record = artifacts.finalize_sequence_candidate(track_id, sequence_candidates)
        if record is not None:
            saved_counts[str(record["class_name"])] += 1


def main() -> None:
    args = parse_args()
    scene = SceneConfig.load(args.config, source_mode=args.source_mode)
    if args.source:
        scene.video_source = args.source
        scene.video_source_mode = "override"
    if args.tracker_config:
        scene.tracker_config = Path(args.tracker_config).resolve()

    events = load_gt_events(Path(args.gt).resolve())
    intervals = build_exclusion_intervals(
        events,
        buffer_frames=max(0, args.buffer_frames),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    cap = _open_video_capture(scene)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    validate_video_against_gt(total_frames, Path(args.gt).resolve(), intervals, scene.video_source)

    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = reported_fps if reported_fps and reported_fps > 1.0 else scene.source_fps
    from ultralytics import YOLO

    model = YOLO(str(scene.yolo_weights))
    tracker_backend = _build_tracker_backend(scene, model=model, fps=video_fps)

    artifacts = RunArtifacts(
        run_root=scene.run_root,
        dataset_dir=scene.dataset_dir,
        min_sequence_frames=(
            args.min_sequence_frames
            if args.min_sequence_frames is not None
            else int(scene.thresholds.get("min_sequence_frames", 16))
        ),
    )

    enhancement = scene.raw_config.get("enhancement", {})
    enhance_enabled = bool(enhancement.get("enabled", False))
    skip_frames = max(1, args.skip_frames)
    sequence_candidates: dict[int, dict] = {}
    saved_counts: Counter[str] = Counter()

    interval_index = 0
    previous_excluded = False
    frame_idx = 0
    processed_frames = 0
    mined_frames = 0

    print(f"Video source: {scene.video_source}")
    print(f"Tracker config: {scene.tracker_config}")
    print(f"GT path: {Path(args.gt).resolve()}")
    print(f"Excluded intervals: {len(intervals)}")
    print(f"Frame buffer: {max(0, args.buffer_frames)}")
    print(f"Mining output run: {artifacts.root}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if args.max_frames and frame_idx > args.max_frames:
                break
            if args.frame_start is not None and frame_idx < args.frame_start:
                continue
            if args.frame_end is not None and frame_idx > args.frame_end:
                break
            if skip_frames > 1 and frame_idx % skip_frames != 0:
                continue

            processed_frames += 1
            excluded, interval_index = frame_is_excluded(frame_idx, intervals, interval_index)
            if excluded:
                if not previous_excluded:
                    _finalize_all_candidates(artifacts, sequence_candidates, saved_counts)
                    tracker_backend = _build_tracker_backend(scene, model=model, fps=video_fps)
                previous_excluded = True
                continue

            if previous_excluded:
                tracker_backend = _build_tracker_backend(scene, model=model, fps=video_fps)
            previous_excluded = False

            if enhance_enabled:
                frame = enhance_frame(frame, enhancement)

            tracked_detections = tracker_backend.track(frame)
            mined_frames += 1
            active_ids: set[int] = set()

            for tracked in tracked_detections:
                class_name = scene.coco_names.get(tracked.class_id, str(tracked.class_id))
                if class_name not in APPEARANCE_CLASSES:
                    continue
                crop = extract_crop(frame, tracked.bbox)
                if crop.size == 0:
                    continue
                active_ids.add(tracked.track_id)
                candidate = sequence_candidates.get(tracked.track_id)
                if candidate is None:
                    candidate = artifacts.create_sequence_candidate(tracked.track_id, class_name, frame_idx)
                    sequence_candidates[tracked.track_id] = candidate
                artifacts.append_sequence_frame(candidate, crop, frame_idx, tracked.conf, class_name)

            for track_id in list(sequence_candidates):
                if track_id in active_ids:
                    continue
                record = artifacts.finalize_sequence_candidate(track_id, sequence_candidates)
                if record is not None:
                    saved_counts[str(record["class_name"])] += 1
    finally:
        _finalize_all_candidates(artifacts, sequence_candidates, saved_counts)
        cap.release()
        artifacts.close()

    print(f"Processed frames: {processed_frames}")
    print(f"Mined frames outside anomaly windows: {mined_frames}")
    print(f"Saved normal sequences: {sum(saved_counts.values())}")
    for class_name in sorted(saved_counts):
        print(f"  {class_name}: {saved_counts[class_name]}")
    print(f"Dataset sequences dir: {scene.dataset_dir / 'sequences'}")
    print(f"Review file: {scene.dataset_dir / 'sequence_review.csv'}")


if __name__ == "__main__":
    main()
