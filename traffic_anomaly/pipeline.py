from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .appearance import AppearanceScorer
from .config import SceneConfig
from .events import EventManager
from .ganomaly import GANomalyScorer
from .geometry import bottom_center, find_lane, project_point
from .rules import RuleEngine, build_lane_snapshots
from .storage import RunArtifacts
from .tracker_backend import TrackerBackend
from .tracklets import TrackManager
from .vae import VAEScorer
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
        tracker_config_override: str | None = None,
        skip_frames: int = 1,
        device: str | None = None,
        appearance_model: str = "ganomaly",
    ):
        self.scene = SceneConfig.load(config_path, source_mode=source_mode)
        if source_override:
            self.scene.video_source = source_override
            self.scene.video_source_mode = "override"
        if tracker_config_override:
            self.scene.tracker_config = Path(tracker_config_override).resolve()
        self.max_frames = max_frames
        self.display = display
        self.skip_frames = max(1, skip_frames)
        self.enhancement = self.scene.raw_config.get("enhancement", {})
        self.device = device
        self.appearance_model = appearance_model.strip().lower()
        if self.appearance_model not in {"ganomaly", "vae"}:
            raise ValueError("appearance_model must be one of: ganomaly, vae")

    def _build_appearance_scorer(self) -> AppearanceScorer:
        if self.appearance_model == "ganomaly":
            return GANomalyScorer(
                checkpoint_paths=self.scene.ganomaly_checkpoints,
                image_size=int(self.scene.ganomaly_settings.get("image_size", 64)),
                default_threshold=float(self.scene.ganomaly_settings.get("default_threshold", 0.02)),
            )
        return VAEScorer(
            checkpoint_paths=self.scene.vae_checkpoints,
            image_size=int(self.scene.vae_settings.get("image_size", 64)),
            default_threshold=float(self.scene.vae_settings.get("default_threshold", 0.02)),
        )

    def _open_video_capture(self):
        source = self.scene.video_source
        try:
            cap = open_capture(source, self.scene.video_resolution)
            self._validate_capture(cap, source)
            return cap
        except Exception as exc:
            fallback_source = self.scene.video_sources.get("local")
            if (
                self.scene.video_source_mode != "local"
                and fallback_source
                and fallback_source != source
                and Path(fallback_source).exists()
            ):
                print(
                    f"Warning: failed to open {self.scene.video_source_mode} source "
                    f"'{source}': {exc}"
                )
                print(f"Falling back to local video: {fallback_source}")
                cap = open_capture(fallback_source, self.scene.video_resolution)
                self._validate_capture(cap, fallback_source)
                self.scene.video_source = fallback_source
                self.scene.video_source_mode = "local-fallback"
                return cap
            raise RuntimeError(self._format_source_error(source, exc)) from exc

    @staticmethod
    def _validate_capture(cap, source: str) -> None:
        if cap is not None and cap.isOpened():
            return
        if source.startswith("http://") or source.startswith("https://"):
            raise RuntimeError(f"Could not open remote video source: {source}")
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Local video source not found: {source_path}")
        raise RuntimeError(f"Could not open video file: {source_path}")

    @staticmethod
    def _format_source_error(source: str, exc: Exception) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            if "youtube" in source:
                return (
                    f"Could not open YouTube source '{source}'. This environment may be offline "
                    "or YouTube download failed. Run with '--source-mode local' after placing a "
                    "video at the configured local path, or pass '--source /absolute/path/to/video.mp4'."
                )
            return (
                f"Could not open remote video source '{source}' ({exc}). "
                "Pass '--source /absolute/path/to/video.mp4' to use a local file instead."
            )
        source_path = Path(source)
        if not source_path.exists():
            return (
                f"Local video source not found: {source_path}. Update 'video.local' in "
                "configs/scene_config.yaml or pass '--source /absolute/path/to/video.mp4'."
            )
        return (
            f"Could not open video file '{source_path}' ({exc}). "
            "Make sure the file is readable and encoded in a format OpenCV can open."
        )

    def run(self) -> Path:
        cap = self._open_video_capture()

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
        appearance_scorer = self._build_appearance_scorer()

        model = YOLO(str(self.scene.yolo_weights))
        tracker_backend = TrackerBackend(
            model=model,
            tracker_config=self.scene.tracker_config,
            detect_classes=self.scene.detect_classes,
            default_confidence=self.scene.confidence,
            fps=video_fps,
            device=self.device,
        )
        sequence_candidates: dict[int, dict] = {}
        anomalous_track_ids: set[int] = set()

        display_fps = 0.0
        prev_time = time.time()
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_mode = not self.display
        enhance_enabled = bool(self.enhancement.get("enabled", False))
        processed_frame_count = 0
        processed_frame_min: int | None = None
        processed_frame_max: int | None = None

        print(f"Camera: {self.scene.camera_id}")
        print(f"Video source: {self.scene.video_source}")
        print(f"Video source mode: {self.scene.video_source_mode}")
        print(f"Tracker config: {self.scene.tracker_config}")
        print(f"Tracker backend: {tracker_backend.describe()}")
        print(f"Appearance model: {self.appearance_model}")
        print(f"Inference device: {self.device or 'auto'}")
        if tracker_backend.detector_confidence != self.scene.confidence:
            print(
                f"Detector confidence: {self.scene.confidence:.2f} "
                f"(tracker ingest threshold {tracker_backend.detector_confidence:.2f})"
            )
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
            processed_frame_count += 1
            if processed_frame_min is None:
                processed_frame_min = frame_idx
            processed_frame_max = frame_idx

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

            display_frame = frame.copy()
            draw_scene_overlay(display_frame, self.scene.lanes)

            tracked_detections = tracker_backend.track(frame)
            detections: list[tuple] = []
            active_ids: set[int] = set()
            for tracked in tracked_detections:
                bbox = tracked.bbox
                track_id = tracked.track_id
                class_id = tracked.class_id
                conf = tracked.conf
                class_name = self.scene.coco_names.get(class_id, "Vehicle")
                foot_px = bottom_center(bbox)
                lane = find_lane(foot_px, self.scene.lanes)
                lane_id = lane.id if lane else None
                lane_category = lane.category if lane else None
                foot_bev = project_point(self.scene.homography, foot_px)
                crop = extract_crop(frame, bbox)
                appearance_score = appearance_scorer.score_crop(crop, class_name)
                feature = track_manager.update(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    conf=conf,
                    bbox=bbox,
                    footpoint_px=foot_px,
                    footpoint_bev=foot_bev,
                    lane_id=lane_id,
                    lane_category=lane_category,
                    ganomaly_score=appearance_score,
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
                ganomaly_ready=appearance_scorer.available(),
                appearance_label=self.appearance_model.upper() if self.appearance_model == "vae" else "GANomaly",
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
        run_metadata = {
            "camera_id": self.scene.camera_id,
            "video_source": self.scene.video_source,
            "video_source_mode": self.scene.video_source_mode,
            "video_frame_count": total_frames,
            "processed_frame_count": processed_frame_count,
            "processed_frame_min": processed_frame_min,
            "processed_frame_max": processed_frame_max,
            "skip_frames": self.skip_frames,
            "tracker_config": str(self.scene.tracker_config),
            "appearance_model": self.appearance_model,
            "device": self.device or "auto",
            "display": self.display,
        }
        (artifacts.root / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        return artifacts.root

    @staticmethod
    def _severity_rank(value: str) -> int:
        return 2 if value == "critical" else 1 if value == "warning" else 0

    def _fuse_hits(self, feature, rule_hits) -> list[dict[str, float | str]]:
        threshold_key = f"{self.appearance_model}_high_threshold"
        appearance_threshold = float(
            self.scene.thresholds.get(threshold_key, self.scene.thresholds.get("ganomaly_high_threshold", 1.0))
        )
        appearance_high = feature.ganomaly_score >= appearance_threshold
        appearance_label = "GANomaly" if self.appearance_model == "ganomaly" else "VAE"
        fused: list[dict[str, float | str]] = []

        if rule_hits:
            for hit in rule_hits:
                severity = "critical" if appearance_high else hit.severity
                explanation = hit.explanation
                if appearance_high:
                    explanation += f" {appearance_label} support score {feature.ganomaly_score:.2f} exceeded threshold."
                fused.append(
                    {
                        "anomaly_type": hit.anomaly_type,
                        "severity": severity,
                        "rule_score": hit.rule_score,
                        "ganomaly_score": feature.ganomaly_score,
                        "fused_score": max(hit.rule_score, feature.ganomaly_score if appearance_high else 0.0),
                        "explanation": explanation,
                    }
                )
            return fused

        if appearance_high:
            fused.append(
                {
                    "anomaly_type": "appearance_anomaly",
                    "severity": "warning",
                    "rule_score": 0.0,
                    "ganomaly_score": feature.ganomaly_score,
                    "fused_score": feature.ganomaly_score,
                    "explanation": f"{feature.class_name} {appearance_label} score {feature.ganomaly_score:.2f} exceeded learned normal threshold.",
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
    parser.add_argument(
        "--tracker-config",
        default=None,
        help="Optional override for the tracker YAML, e.g. configs/bytetrack.yaml or configs/ocsort.yaml.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV preview window.")
    parser.add_argument("--batch", action="store_true", help="Batch mode: disable display, show progress bar.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (default: 1 = all frames).")
    parser.add_argument("--device", default=None, help="Inference device override, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--appearance-model",
        choices=["ganomaly", "vae"],
        default="ganomaly",
        help="Appearance scorer used for fused appearance anomaly scoring.",
    )
    return parser
