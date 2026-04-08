from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .config import SceneConfig
from .stage1_detect import DetectionStage
from .stage2_track import TrackingStage
from .stage3_features import FeatureStage
from .stage4_anomaly import GANomalyScorer
from .stage5_fuse import FusionStage
from .stage6_alerts import AlertStage


def open_capture(source: str, resolution: str | None):
    if source.startswith("http") and "youtube" in source:
        from cap_from_youtube import cap_from_youtube

        return cap_from_youtube(source, resolution=resolution or "360p")
    return cv2.VideoCapture(source)


class TrafficAnomalyPipeline:
    def __init__(
        self,
        config_path: str | Path,
        max_frames: int | None = None,
        display: bool = True,
        source_override: str | None = None,
        skip_frames: int = 1,
    ):
        self.scene = SceneConfig.load(config_path)
        self.video_source = source_override or self.scene.video.source
        self.max_frames = max_frames
        self.display = display
        self.skip_frames = max(skip_frames, 1)

    def run(self) -> None:
        cap = open_capture(self.video_source, self.scene.video.resolution)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_source}")

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = reported_fps if reported_fps and reported_fps > 1.0 else self.scene.video.fps
        detect_stage = DetectionStage(self.scene.detect)
        track_stage = TrackingStage(self.scene.track, video_fps)
        feature_stage = FeatureStage(self.scene.features, video_fps, self.scene.track.state_ttl_frames)
        anomaly_stage = GANomalyScorer(self.scene.anomaly)
        fuse_stage = FusionStage(self.scene.features, self.scene.fusion, video_fps)
        alert_stage = AlertStage(
            self.scene.alerts,
            self.scene.fusion,
            self.scene.anomaly,
            self.scene.camera_id,
            anomaly_stage.available(),
        )

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_mode = not self.display
        display_fps = 0.0
        prev_time = time.time()
        start_time = time.time()

        print(f"Camera: {self.scene.camera_id}")
        print(f"Video source: {self.video_source}")
        print(f"Run output: {alert_stage.root}")
        if self.skip_frames > 1:
            print(f"Frame skip: processing every {self.skip_frames} frame(s)")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if self.max_frames and frame_idx > self.max_frames:
                break
            if self.skip_frames > 1 and frame_idx % self.skip_frames != 0:
                continue

            if batch_mode and total_frames > 0 and frame_idx % 500 == 0:
                elapsed = time.time() - start_time
                raw_fps = frame_idx / max(elapsed, 0.01)
                remaining = (total_frames - frame_idx) / max(raw_fps, 0.01)
                sys.stdout.write(
                    f"\r[{frame_idx / total_frames * 100:5.1f}%] Frame {frame_idx}/{total_frames} | "
                    f"{raw_fps:.0f} raw fps | ETA {remaining / 60:.1f} min"
                )
                sys.stdout.flush()

            now = time.time()
            display_fps = 0.8 * display_fps + 0.2 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            detections = detect_stage.run(frame)
            observations = track_stage.run(frame_idx, detections, frame)
            features = feature_stage.update(frame_idx, observations)
            ganomaly_hits = anomaly_stage.score_frame(frame, features)
            incidents = fuse_stage.evaluate(features, ganomaly_hits)
            alert_stage.record_frame(frame_idx, features, ganomaly_hits, incidents, frame)

            if self.display:
                display_frame = self._draw_frame(frame.copy(), features, incidents, display_fps, anomaly_stage.available())
                cv2.imshow("Traffic Anomaly Pipeline", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if self.display:
            cv2.destroyAllWindows()
        alert_stage.close()

        print("\nRun complete.")
        print(f"Tracklet log: {alert_stage.root / 'tracklets.csv'}")
        print(f"Incident log: {alert_stage.root / 'incidents.csv'}")
        print(f"Run summary: {alert_stage.root / 'run_summary.json'}")

    def _draw_frame(self, frame, features, incidents, fps: float, ganomaly_ready: bool):
        overlay = frame.copy()
        for lane in self.scene.features.lanes:
            color = (0, 200, 255) if lane.category == "emergency" else (255, 80, 80)
            cv2.fillPoly(overlay, [lane.polygon], color)
            cv2.polylines(frame, [lane.polygon], True, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        primary_by_track = {}
        for incident in incidents:
            previous = primary_by_track.get(incident.track_id)
            if previous is None or incident.fused_score >= previous.fused_score:
                primary_by_track[incident.track_id] = incident

        for feature in features:
            x1, y1, x2, y2 = feature.bbox
            incident = primary_by_track.get(feature.track_id)
            if incident is None:
                color = (200, 200, 200)
                label = f"{feature.class_name} #{feature.track_id}"
            else:
                color = (60, 60, 255) if incident.severity == "critical" else (0, 220, 255)
                label = f"{feature.class_name} #{feature.track_id} {incident.anomaly_type.upper()}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        hud_lines = [
            "Readable 6-Stage Pipeline",
            f"FPS: {fps:.1f}",
            f"GANomaly: {'ready' if ganomaly_ready else 'missing checkpoint'}",
            f"Incidents this frame: {len(incidents)}",
        ]
        for idx, text in enumerate(hud_lines):
            cv2.putText(frame, text, (10, 20 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic anomaly detection pipeline")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Path to the scene configuration YAML.")
    parser.add_argument("--source", default=None, help="Optional override for the configured video source.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV preview window.")
    parser.add_argument("--batch", action="store_true", help="Batch mode: disable display and show progress.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    return parser
