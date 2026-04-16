from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import SceneConfig
from .events import EventManager
from .ganomaly import GANomalyScorer
from .geometry import bottom_center, find_lane, project_point
from .rules import RuleEngine, build_lane_snapshots
from .storage import RunArtifacts
from .tracklets import TrackManager
from .visualization import draw_hud_panel, draw_scene_overlay, draw_track_box, draw_trail, COLOR_CRITICAL, COLOR_WARNING, COLOR_TRAIL


def extract_crop(frame, box: tuple[int, int, int, int], pad: int = 15):
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(width, x2 + pad)
    cy2 = min(height, y2 + pad)
    return frame[cy1:cy2, cx1:cx2]


def open_capture(source: str, resolution: str | None):
    if source.startswith("http") and "youtube" in source:
        from cap_from_youtube import cap_from_youtube

        return cap_from_youtube(source, resolution=resolution or "360p")
    return cv2.VideoCapture(source)


def _format_anomaly_label(name: str) -> str:
    return name.replace("_", " ").upper()


def enhance_frame(frame: np.ndarray, settings: dict) -> np.ndarray:
    """Apply image enhancements: CLAHE, denoising, sharpening."""
    result = frame
    if settings.get("clahe_enabled", False):
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clip_limit = float(settings.get("clahe_clip_limit", 2.0))
        tile_size = int(settings.get("clahe_tile_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if settings.get("denoise_enabled", False):
        strength = int(settings.get("denoise_strength", 6))
        result = cv2.fastNlMeansDenoisingColored(result, None, strength, strength, 7, 21)
    if settings.get("sharpen_enabled", False):
        amount = float(settings.get("sharpen_amount", 0.5))
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)
    return result


class TrafficAnomalyPipeline:
    def __init__(
        self,
        config_path: str | Path,
        max_frames: int | None = None,
        display: bool = True,
        source_mode: str | None = None,
        source_override: str | None = None,
        skip_frames: int = 1,
    ):
        self.scene = SceneConfig.load(config_path, source_mode=source_mode)
        if source_override:
            self.scene.video_source = source_override
            self.scene.video_source_mode = "override"
        self.max_frames = max_frames
        self.display = display
        self.skip_frames = max(1, skip_frames)
        self.enhancement = self.scene.raw_config.get("enhancement", {})

    def run(self) -> None:
        cap = open_capture(self.scene.video_source, self.scene.video_resolution)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.scene.video_source}")

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = reported_fps if reported_fps and reported_fps > 1.0 else self.scene.source_fps
        artifacts = RunArtifacts(
            run_root=self.scene.run_root,
            dataset_dir=self.scene.dataset_dir,
            min_sequence_frames=int(self.scene.thresholds.get("min_sequence_frames", 16)),
        )
        track_manager = TrackManager(self.scene, video_fps)
        rule_engine = RuleEngine(self.scene, video_fps)
        event_manager = EventManager(
            artifacts,
            camera_id=self.scene.camera_id,
            grace_frames=int(self.scene.thresholds.get("event_gap_frames", 5)),
        )
        ganomaly = GANomalyScorer(
            checkpoint_paths=self.scene.ganomaly_checkpoints,
            image_size=int(self.scene.ganomaly_settings.get("image_size", 64)),
            default_threshold=float(self.scene.ganomaly_settings.get("default_threshold", 0.02)),
        )

        model = YOLO(str(self.scene.yolo_weights))
        sequence_candidates: dict[int, dict] = {}
        anomalous_track_ids: set[int] = set()

        display_fps = 0.0
        prev_time = time.time()
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_mode = not self.display
        enhance_enabled = bool(self.enhancement.get("enabled", False))

        print(f"Camera: {self.scene.camera_id}")
        print(f"Video source: {self.scene.video_source}")
        print(f"Video source mode: {self.scene.video_source_mode}")
        print(f"Tracker config: {self.scene.tracker_config}")
        print(f"Run output: {artifacts.root}")
        print(f"Dataset root: {self.scene.dataset_dir}")
        if self.skip_frames > 1:
            print(f"Frame skip: processing every {self.skip_frames} frame(s)")
        if enhance_enabled:
            print(f"Image enhancement: enabled")
        if batch_mode and total_frames > 0:
            effective_total = total_frames // self.skip_frames
            print(f"Batch mode: {total_frames} total frames → ~{effective_total} to process")

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if self.max_frames and frame_idx > self.max_frames:
                break

            # Skip frames for faster batch processing
            if self.skip_frames > 1 and (frame_idx % self.skip_frames) != 0:
                continue

            # Progress reporting for batch mode
            if batch_mode and total_frames > 0 and frame_idx % 500 == 0:
                elapsed = time.time() - start_time
                pct = frame_idx / total_frames * 100
                fps_real = frame_idx / max(elapsed, 0.01)
                remaining = (total_frames - frame_idx) / max(fps_real, 0.01)
                mins_left = remaining / 60
                sys.stdout.write(
                    f"\r[{pct:5.1f}%] Frame {frame_idx}/{total_frames} | "
                    f"{fps_real:.0f} raw fps | ETA {mins_left:.1f}min"
                )
                sys.stdout.flush()

            # Apply image enhancement if enabled
            if enhance_enabled:
                frame = enhance_frame(frame, self.enhancement)

            now = time.time()
            display_fps = 0.8 * display_fps + 0.2 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            results = model.track(
                frame,
                persist=True,
                classes=self.scene.detect_classes,
                conf=self.scene.confidence,
                tracker=str(self.scene.tracker_config),
                verbose=False,
            )

            display_frame = frame.copy()
            draw_scene_overlay(display_frame, self.scene.lanes)

            detections: list[tuple] = []
            active_ids: set[int] = set()
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                    if float(conf) < self.scene.confidence:
                        continue
                    track_id = int(track_id)
                    class_id = int(class_id)
                    bbox = tuple(int(value) for value in box)
                    class_name = self.scene.coco_names.get(class_id, "Vehicle")
                    foot_px = bottom_center(bbox)
                    lane = find_lane(foot_px, self.scene.lanes)
                    lane_id = lane.id if lane else None
                    lane_category = lane.category if lane else None
                    foot_bev = project_point(self.scene.homography, foot_px)
                    crop = extract_crop(frame, bbox)
                    ganomaly_score = ganomaly.score_crop(crop, class_name)
                    feature = track_manager.update(
                        frame_idx=frame_idx,
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        conf=float(conf),
                        bbox=bbox,
                        footpoint_px=foot_px,
                        footpoint_bev=foot_bev,
                        lane_id=lane_id,
                        lane_category=lane_category,
                        ganomaly_score=ganomaly_score,
                    )
                    detections.append((feature, crop, bbox))
                    active_ids.add(track_id)

            lane_snapshots = build_lane_snapshots([feature for feature, _, _ in detections], self.scene)
            current_event_keys: set[tuple[int, str]] = set()

            for feature, crop, bbox in detections:
                rule_hits = rule_engine.evaluate(feature, lane_snapshots)
                fused_hits = self._fuse_hits(feature, rule_hits)

                # Draw trajectory trail
                trail = track_manager.get_trail(feature.track_id)
                if fused_hits:
                    primary_severity = max(fused_hits, key=lambda h: self._severity_rank(str(h["severity"])))["severity"]
                    trail_color = COLOR_CRITICAL if primary_severity == "critical" else COLOR_WARNING
                else:
                    trail_color = COLOR_TRAIL
                draw_trail(display_frame, trail, color=trail_color)

                if fused_hits:
                    anomalous_track_ids.add(feature.track_id)
                    artifacts.discard_sequence_candidate(feature.track_id, sequence_candidates)
                    primary = max(
                        fused_hits,
                        key=lambda item: (self._severity_rank(str(item["severity"])), float(item["fused_score"])),
                    )
                    draw_track_box(
                        display_frame,
                        bbox,
                        f"{feature.class_name} #{feature.track_id} {_format_anomaly_label(str(primary['anomaly_type']))}",
                        severity=str(primary["severity"]),
                    )
                    for fused_hit in fused_hits:
                        key = event_manager.update_event(feature, fused_hit, frame, display_frame, bbox)
                        current_event_keys.add(key)
                else:
                    if feature.track_id not in anomalous_track_ids and crop.size > 0:
                        candidate = sequence_candidates.get(feature.track_id)
                        if candidate is None:
                            candidate = artifacts.create_sequence_candidate(feature.track_id, feature.class_name, frame_idx)
                            sequence_candidates[feature.track_id] = candidate
                        artifacts.append_sequence_frame(candidate, crop, frame_idx, feature.conf, feature.class_name)
                    draw_track_box(display_frame, bbox, f"{feature.class_name} #{feature.track_id}")

                tracklet_record = feature.to_record()
                tracklet_record["rule_hits"] = "|".join(hit["anomaly_type"] for hit in fused_hits)
                tracklet_record["severity"] = "|".join(hit["severity"] for hit in fused_hits)
                artifacts.log_tracklet(tracklet_record)

            event_manager.finalize_inactive(current_event_keys, frame_idx)

            for stale_id in track_manager.stale_ids(active_ids):
                if stale_id not in anomalous_track_ids:
                    artifacts.finalize_sequence_candidate(stale_id, sequence_candidates)
                else:
                    artifacts.discard_sequence_candidate(stale_id, sequence_candidates)
                track_manager.remove(stale_id)
                event_manager.close_track(stale_id)
                anomalous_track_ids.discard(stale_id)

            draw_hud_panel(
                display_frame,
                display_fps,
                dict(event_manager.finalized_counts),
                active_events=len(event_manager.active_events),
                ganomaly_ready=ganomaly.available(),
            )

            if self.display:
                cv2.imshow("Traffic Anomaly Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if self.display:
            cv2.destroyAllWindows()

        for track_id in list(sequence_candidates.keys()):
            if track_id not in anomalous_track_ids:
                artifacts.finalize_sequence_candidate(track_id, sequence_candidates)
            else:
                artifacts.discard_sequence_candidate(track_id, sequence_candidates)
        event_manager.close_all()
        artifacts.close()

        print("Run complete.")
        print(f"Tracklet log: {artifacts.root / 'tracklets.csv'}")
        print(f"Event log: {artifacts.root / 'events.csv'}")
        print(f"Normal sequence log: {artifacts.root / 'normal_sequences.csv'}")

    @staticmethod
    def _severity_rank(value: str) -> int:
        return 2 if value == "critical" else 1 if value == "warning" else 0

    def _fuse_hits(self, feature, rule_hits) -> list[dict[str, float | str]]:
        ganomaly_threshold = float(self.scene.thresholds.get("ganomaly_high_threshold", 1.0))
        ganomaly_high = feature.ganomaly_score >= ganomaly_threshold
        fused: list[dict[str, float | str]] = []

        if rule_hits:
            for hit in rule_hits:
                severity = "critical" if ganomaly_high else hit.severity
                explanation = hit.explanation
                if ganomaly_high:
                    explanation += f" GANomaly support score {feature.ganomaly_score:.2f} exceeded threshold."
                fused.append(
                    {
                        "anomaly_type": hit.anomaly_type,
                        "severity": severity,
                        "rule_score": hit.rule_score,
                        "ganomaly_score": feature.ganomaly_score,
                        "fused_score": max(hit.rule_score, feature.ganomaly_score if ganomaly_high else 0.0),
                        "explanation": explanation,
                    }
                )
            return fused

        if ganomaly_high:
            fused.append(
                {
                    "anomaly_type": "appearance_anomaly",
                    "severity": "warning",
                    "rule_score": 0.0,
                    "ganomaly_score": feature.ganomaly_score,
                    "fused_score": feature.ganomaly_score,
                    "explanation": f"{feature.class_name} appearance score {feature.ganomaly_score:.2f} exceeded learned normal threshold.",
                }
            )
        return fused


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic anomaly detection MVP")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Path to the scene configuration YAML.")
    parser.add_argument(
        "--source-mode",
        choices=["youtube", "local"],
        default=None,
        help="Select a named video source from the config. --source still overrides this.",
    )
    parser.add_argument("--source", default=None, help="Optional override for the configured video source.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV preview window.")
    parser.add_argument("--batch", action="store_true", help="Batch mode: disable display, show progress bar.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (default: 1 = all frames).")
    return parser

