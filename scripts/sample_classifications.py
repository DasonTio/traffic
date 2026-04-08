"""
YOLO Classification Sampler — Evaluate YOLO accuracy on ALL tracked vehicles.

Unlike the anomaly labeler (which only sees detected anomalies), this tool
samples from the FULL tracklets.csv to verify YOLO's class predictions on
all vehicles — including ones that were never flagged.

It extracts crops from the video at sampled frames and lets you verify
each vehicle's actual class. This catches False Negatives: e.g., a truck
YOLO called "Car" that should have triggered a lane violation.

Usage:
    python scripts/sample_classifications.py
    python scripts/sample_classifications.py --run 20260407_002145 --samples 50

Keyboard shortcuts:
    C / 1 = Car         B / 2 = Bus
    K / 3 = Truck       M / 4 = Motorcycle
    U / 5 = Unknown     S     = Skip
    Left  = Previous    Right = Next
    Q     = Quit and save
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


VEHICLE_CLASSES = ["Car", "Bus", "Truck", "Motorcycle", "Unknown"]
CLASS_KEYS = {"c": "Car", "b": "Bus", "k": "Truck", "m": "Motorcycle", "u": "Unknown"}


@dataclass
class TrackletSample:
    frame_idx: int
    track_id: str
    class_name: str
    class_id: str
    conf: str
    bbox: str  # "x1,y1,x2,y2"
    lane_id: str
    rule_hits: str
    speed: str


def find_latest_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    subdirs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def load_tracklets(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "tracklets.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_unique_tracks(
    tracklet_rows: list[dict[str, str]],
    num_samples: int,
    seed: int = 42,
) -> list[TrackletSample]:
    """Pick one representative row per unique track_id, then sample N of them."""
    # Group by track_id, pick the row with highest confidence per track
    best_per_track: dict[str, dict[str, str]] = {}
    for row in tracklet_rows:
        tid = row.get("track_id", "")
        if not tid:
            continue
        existing = best_per_track.get(tid)
        if existing is None or float(row.get("conf", 0)) > float(existing.get("conf", 0)):
            best_per_track[tid] = row

    candidates = list(best_per_track.values())
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:min(num_samples, len(candidates))]

    return [
        TrackletSample(
            frame_idx=int(row.get("frame_idx", 0)),
            track_id=row.get("track_id", ""),
            class_name=row.get("class_name", ""),
            class_id=row.get("class_id", ""),
            conf=row.get("conf", ""),
            bbox=row.get("bbox", ""),
            lane_id=row.get("lane_id", ""),
            rule_hits=row.get("rule_hits", ""),
            speed=row.get("speed", ""),
        )
        for row in selected
    ]


@dataclass
class ClassLabel:
    actual_class: str


LABEL_FIELDS = [
    "track_id",
    "frame_idx",
    "yolo_class",
    "yolo_conf",
    "actual_class",
    "class_correct",
    "lane_id",
    "rule_hits",
    "labeled_at",
]


def load_existing_labels(path: Path) -> dict[str, ClassLabel]:
    if not path.exists():
        return {}
    labels: dict[str, ClassLabel] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row.get('track_id', '')}_{row.get('frame_idx', '')}"
            labels[key] = ClassLabel(actual_class=row.get("actual_class", ""))
    return labels


class ClassificationSampler:
    CROP_W = 300
    CROP_H = 300
    FRAME_W = 640
    FRAME_H = 400

    def __init__(self, run_dir: Path, video_source: str, samples: list[TrackletSample]):
        self.run_dir = run_dir
        self.video_source = video_source
        self.samples = samples
        self.index = 0
        self.labels_path = run_dir / "classification_labels.csv"
        self.labels = load_existing_labels(self.labels_path)
        self._cap = None

        # Skip to first unlabeled
        for i, s in enumerate(self.samples):
            key = f"{s.track_id}_{s.frame_idx}"
            if key not in self.labels:
                self.index = i
                break

        self.root = tk.Tk()
        self.root.title(f"YOLO Classification Sampler — {run_dir.name}")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1100x720")

        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _open_video(self):
        if self._cap is None:
            if self.video_source.startswith("http") and "youtube" in self.video_source:
                from cap_from_youtube import cap_from_youtube
                self._cap = cap_from_youtube(self.video_source, resolution="360p")
            else:
                self._cap = cv2.VideoCapture(self.video_source)
        return self._cap

    def _extract_crop(self, frame_idx: int, bbox_str: str):
        """Seek to frame and extract crop."""
        cap = self._open_video()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = cap.read()
        if not ret:
            return None, None

        try:
            parts = bbox_str.split(",")
            x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            pad = 20
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]

            # Draw box on frame for context
            display = frame.copy()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            return crop, display
        except (ValueError, IndexError):
            return None, frame

    def _cv2_to_tk(self, cv_img, max_w, max_h):
        if cv_img is None or cv_img.size == 0:
            return None
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"), foreground="#cba6f7")
        style.configure("Meta.TLabel", font=("Segoe UI", 9), foreground="#a6adc8")
        style.configure("Stats.TLabel", font=("Segoe UI", 10, "bold"), foreground="#a6e3a1")
        style.configure("TFrame", background="#1e1e2e")

        # Top bar
        top = ttk.Frame(self.root, style="TFrame")
        top.pack(fill="x", padx=12, pady=(8, 0))
        self.title_label = ttk.Label(top, style="Title.TLabel")
        self.title_label.pack(side="left")
        self.stats_label = ttk.Label(top, style="Stats.TLabel")
        self.stats_label.pack(side="right")

        # Loading label
        self.loading_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.loading_var, bg="#1e1e2e", fg="#f9e2af",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12)

        # Image area
        img_frame = ttk.Frame(self.root, style="TFrame")
        img_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.frame_canvas = tk.Canvas(img_frame, bg="#181825", highlightthickness=0)
        self.frame_canvas.pack(side="left", fill="both", expand=True, padx=(0, 4))

        right_panel = ttk.Frame(img_frame, style="TFrame")
        right_panel.pack(side="right", fill="y", padx=(4, 0))

        self.crop_canvas = tk.Canvas(right_panel, bg="#181825", width=self.CROP_W, height=self.CROP_H,
                                     highlightthickness=0)
        self.crop_canvas.pack()

        self.meta_label = ttk.Label(right_panel, style="Meta.TLabel", wraplength=260, justify="left")
        self.meta_label.pack(pady=(8, 0), fill="x")

        # Current label display
        self.current_label_var = tk.StringVar(value="")
        self.current_label_display = tk.Label(
            right_panel, textvariable=self.current_label_var,
            font=("Segoe UI", 12, "bold"), fg="#6c7086", bg="#1e1e2e"
        )
        self.current_label_display.pack(pady=(6, 0))

        # Class buttons
        class_frame = ttk.Frame(self.root, style="TFrame")
        class_frame.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Label(class_frame, text="What is this vehicle?", style="Title.TLabel").pack(side="left", padx=(0, 12))
        for cls in VEHICLE_CLASSES:
            key_hint = {"Car": "C", "Bus": "B", "Truck": "K", "Motorcycle": "M", "Unknown": "U"}[cls]
            ttk.Button(
                class_frame, text=f"{cls} ({key_hint})",
                command=lambda c=cls: self._label_class(c)
            ).pack(side="left", padx=3)

        # Nav buttons
        nav_frame = ttk.Frame(self.root, style="TFrame")
        nav_frame.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Button(nav_frame, text="◀ Prev", command=self._prev).pack(side="left", padx=4)
        ttk.Button(nav_frame, text="Next ▶", command=self._next).pack(side="left", padx=4)
        ttk.Button(nav_frame, text="⊘ Skip (S)", command=lambda: self._label_class("Unknown")).pack(side="right", padx=4)

    def _bind_keys(self):
        self.root.bind("c", lambda e: self._label_class("Car"))
        self.root.bind("1", lambda e: self._label_class("Car"))
        self.root.bind("b", lambda e: self._label_class("Bus"))
        self.root.bind("2", lambda e: self._label_class("Bus"))
        self.root.bind("k", lambda e: self._label_class("Truck"))
        self.root.bind("3", lambda e: self._label_class("Truck"))
        self.root.bind("m", lambda e: self._label_class("Motorcycle"))
        self.root.bind("4", lambda e: self._label_class("Motorcycle"))
        self.root.bind("u", lambda e: self._label_class("Unknown"))
        self.root.bind("s", lambda e: self._label_class("Unknown"))
        self.root.bind("5", lambda e: self._label_class("Unknown"))
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("q", lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())

    def _show_current(self):
        if not self.samples:
            self.title_label.config(text="No samples to label.")
            return

        sample = self.samples[self.index]

        self.title_label.config(
            text=f"[{self.index + 1}/{len(self.samples)}]  "
                 f"YOLO says: {sample.class_name}  —  Track #{sample.track_id}"
        )

        # Extract crop from video
        self.loading_var.set("Loading frame from video...")
        self.root.update_idletasks()
        crop, frame = self._extract_crop(sample.frame_idx, sample.bbox)
        self.loading_var.set("")

        # Show frame
        self._frame_photo = self._cv2_to_tk(frame, self.FRAME_W, self.FRAME_H)
        self.frame_canvas.delete("all")
        if self._frame_photo:
            self.frame_canvas.create_image(
                self.FRAME_W // 2, self.FRAME_H // 2,
                image=self._frame_photo, anchor="center"
            )
        else:
            self.frame_canvas.create_text(
                self.FRAME_W // 2, self.FRAME_H // 2,
                text="Could not load frame", fill="#585b70", font=("Segoe UI", 12)
            )

        # Show crop
        self._crop_photo = self._cv2_to_tk(crop, self.CROP_W, self.CROP_H)
        self.crop_canvas.delete("all")
        if self._crop_photo:
            self.crop_canvas.create_image(
                self.CROP_W // 2, self.CROP_H // 2,
                image=self._crop_photo, anchor="center"
            )

        # Metadata
        anomaly_info = f"Anomalies: {sample.rule_hits}" if sample.rule_hits else "No anomaly detected"
        meta = (
            f"YOLO class: {sample.class_name}\n"
            f"Confidence: {sample.conf}\n"
            f"Lane: {sample.lane_id or 'none'}\n"
            f"Speed: {sample.speed}\n"
            f"Frame: {sample.frame_idx}\n\n"
            f"{anomaly_info}"
        )
        self.meta_label.config(text=meta)

        # Existing label
        key = f"{sample.track_id}_{sample.frame_idx}"
        existing = self.labels.get(key)
        if existing and existing.actual_class:
            cls = existing.actual_class
            match = cls == sample.class_name
            color = "#a6e3a1" if match else "#f38ba8"
            label_text = f"Labeled: {cls}" + (" ✓" if match else f" ✗ (YOLO: {sample.class_name})")
            self.current_label_var.set(label_text)
            self.current_label_display.config(fg=color)
        else:
            self.current_label_var.set("Not labeled")
            self.current_label_display.config(fg="#6c7086")

        self._update_stats()

    def _update_stats(self):
        total = len(self.labels)
        correct = 0
        incorrect = 0
        for i, sample in enumerate(self.samples):
            key = f"{sample.track_id}_{sample.frame_idx}"
            data = self.labels.get(key)
            if data and data.actual_class and data.actual_class != "Unknown":
                if data.actual_class == sample.class_name:
                    correct += 1
                else:
                    incorrect += 1
        checked = correct + incorrect
        acc = f"{correct/checked*100:.0f}%" if checked > 0 else "—"
        self.stats_label.config(
            text=f"Labeled: {total}/{len(self.samples)}  |  "
                 f"Correct: {correct}  Wrong: {incorrect}  |  YOLO accuracy: {acc}"
        )

    def _label_class(self, actual_class: str):
        if not self.samples:
            return
        sample = self.samples[self.index]
        key = f"{sample.track_id}_{sample.frame_idx}"
        self.labels[key] = ClassLabel(actual_class=actual_class)
        self._save_labels()
        self._next()

    def _next(self):
        if self.index < len(self.samples) - 1:
            self.index += 1
            self._show_current()
        else:
            self._update_stats()

    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._show_current()

    def _save_labels(self):
        with self.labels_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            for sample in self.samples:
                key = f"{sample.track_id}_{sample.frame_idx}"
                data = self.labels.get(key)
                if data and data.actual_class:
                    writer.writerow({
                        "track_id": sample.track_id,
                        "frame_idx": sample.frame_idx,
                        "yolo_class": sample.class_name,
                        "yolo_conf": sample.conf,
                        "actual_class": data.actual_class,
                        "class_correct": "yes" if data.actual_class == sample.class_name else "no",
                        "lane_id": sample.lane_id,
                        "rule_hits": sample.rule_hits,
                        "labeled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

    def _quit(self):
        self._save_labels()
        if self._cap is not None:
            self._cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Sample and verify YOLO classifications on ALL tracked vehicles")
    parser.add_argument("--run", default=None, help="Run ID (directory name under runs/)")
    parser.add_argument("--run-dir", default=None, help="Full path to run directory")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing run folders")
    parser.add_argument("--samples", type=int, default=50, help="Number of unique tracks to sample (default: 50)")
    parser.add_argument("--source", default=None, help="Video source (default: from scene config)")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Scene config path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
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

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Get video source
    video_source = args.source
    if not video_source:
        import yaml
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            video_source = cfg.get("video", {}).get("source", "")
        if not video_source:
            print("No video source. Use --source or ensure scene_config.yaml exists.")
            sys.exit(1)

    # Load and sample tracklets
    tracklet_rows = load_tracklets(run_dir)
    if not tracklet_rows:
        print(f"No tracklets found in {run_dir}/tracklets.csv")
        sys.exit(1)

    samples = sample_unique_tracks(tracklet_rows, args.samples, args.seed)
    print(f"Sampled {len(samples)} unique tracks from {len(tracklet_rows)} tracklet rows")
    print(f"Video source: {video_source}")

    app = ClassificationSampler(run_dir, video_source, samples)
    app.run()
    print(f"Labels saved to {app.labels_path}")


if __name__ == "__main__":
    main()
