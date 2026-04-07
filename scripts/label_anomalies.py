"""
Anomaly Labeling Tool — Label detected anomalies as TP / FP / Skip.

Scans a run directory (runs/<run_id>/) for event evidence (crops + frames),
shows them side-by-side, and lets you rapidly label each event.

Usage:
    python scripts/label_anomalies.py                   # auto-picks latest run
    python scripts/label_anomalies.py --run 20260406_233816
    python scripts/label_anomalies.py --run-dir runs/20260406_233816

Keyboard shortcuts:
    T / 1      = True Positive (real anomaly)
    F / 0      = False Positive (wrong detection)
    S / Space  = Skip / Unsure
    Left       = Previous event
    Right      = Next event
    Q / Esc    = Quit and save
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


@dataclass
class EventRow:
    event_id: str
    anomaly_type: str
    class_name: str
    severity: str
    lane_id: str
    explanation: str
    crop_path: str
    frame_path: str
    rule_score: str
    fused_score: str
    start_frame: str
    end_frame: str


def find_latest_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    subdirs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def load_events(run_dir: Path) -> list[EventRow]:
    events_csv = run_dir / "events.csv"
    if not events_csv.exists():
        return []
    rows: list[EventRow] = []
    with events_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                EventRow(
                    event_id=row.get("event_id", ""),
                    anomaly_type=row.get("anomaly_type", ""),
                    class_name=row.get("class_name", ""),
                    severity=row.get("severity", ""),
                    lane_id=row.get("lane_id", ""),
                    explanation=row.get("explanation", ""),
                    crop_path=row.get("crop_path", ""),
                    frame_path=row.get("frame_path", ""),
                    rule_score=row.get("rule_score", ""),
                    fused_score=row.get("fused_score", ""),
                    start_frame=row.get("start_frame", ""),
                    end_frame=row.get("end_frame", ""),
                )
            )
    return rows


def load_existing_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["event_id"]] = row.get("label", "skip")
    return labels


LABEL_FIELDS = [
    "event_id",
    "anomaly_type",
    "class_name",
    "lane_id",
    "severity",
    "label",
    "labeled_at",
]


class LabelingApp:
    FRAME_MAX_W = 640
    FRAME_MAX_H = 400
    CROP_MAX_W = 250
    CROP_MAX_H = 250

    def __init__(self, run_dir: Path, events: list[EventRow]):
        self.run_dir = run_dir
        self.events = events
        self.index = 0
        self.labels_path = run_dir / "anomaly_labels.csv"
        self.labels = load_existing_labels(self.labels_path)

        # Skip to first unlabeled
        for i, ev in enumerate(self.events):
            if ev.event_id not in self.labels:
                self.index = i
                break

        self.root = tk.Tk()
        self.root.title(f"Anomaly Labeler — {run_dir.name}")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1080x720")

        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"), foreground="#89b4fa")
        style.configure("Meta.TLabel", font=("Segoe UI", 9), foreground="#a6adc8")
        style.configure("Stats.TLabel", font=("Segoe UI", 10, "bold"), foreground="#a6e3a1")
        style.configure("TP.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("FP.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("Skip.TButton", font=("Segoe UI", 11))

        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(8, 0))
        self.title_label = ttk.Label(top, style="Title.TLabel")
        self.title_label.pack(side="left")
        self.stats_label = ttk.Label(top, style="Stats.TLabel")
        self.stats_label.pack(side="right")

        # Image area
        img_frame = ttk.Frame(self.root)
        img_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.frame_canvas = tk.Canvas(img_frame, bg="#181825", highlightthickness=0)
        self.frame_canvas.pack(side="left", fill="both", expand=True, padx=(0, 4))

        right_panel = ttk.Frame(img_frame)
        right_panel.pack(side="right", fill="y", padx=(4, 0))

        self.crop_canvas = tk.Canvas(right_panel, bg="#181825", width=self.CROP_MAX_W, height=self.CROP_MAX_H, highlightthickness=0)
        self.crop_canvas.pack()

        self.meta_label = ttk.Label(right_panel, style="Meta.TLabel", wraplength=240, justify="left")
        self.meta_label.pack(pady=(8, 0), fill="x")

        # Label showing current label
        self.current_label_var = tk.StringVar(value="")
        self.current_label_display = ttk.Label(right_panel, textvariable=self.current_label_var,
                                                font=("Segoe UI", 12, "bold"), foreground="#f9e2af",
                                                background="#1e1e2e")
        self.current_label_display.pack(pady=(6, 0))

        # Bottom buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=12, pady=(0, 10))

        ttk.Button(btn_frame, text="◀ Prev", command=self._prev).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Next ▶", command=self._next).pack(side="left", padx=4)

        ttk.Button(btn_frame, text="✓ True Positive (T)", style="TP.TButton",
                   command=lambda: self._label("TP")).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="✗ False Positive (F)", style="FP.TButton",
                   command=lambda: self._label("FP")).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="⊘ Skip (S)", style="Skip.TButton",
                   command=lambda: self._label("skip")).pack(side="right", padx=4)

    def _bind_keys(self):
        self.root.bind("t", lambda e: self._label("TP"))
        self.root.bind("1", lambda e: self._label("TP"))
        self.root.bind("f", lambda e: self._label("FP"))
        self.root.bind("0", lambda e: self._label("FP"))
        self.root.bind("s", lambda e: self._label("skip"))
        self.root.bind("<space>", lambda e: self._label("skip"))
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("q", lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())

    def _load_image(self, path: str, max_w: int, max_h: int):
        if not path or not Path(path).exists():
            return None
        img = Image.open(path)
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _show_current(self):
        if not self.events:
            self.title_label.config(text="No events found in this run.")
            return

        ev = self.events[self.index]

        self.title_label.config(
            text=f"[{self.index + 1}/{len(self.events)}]  {ev.anomaly_type.replace('_', ' ').upper()}  —  "
                 f"{ev.class_name} #{ev.event_id[-8:]}"
        )

        # Frame image
        self._frame_photo = self._load_image(ev.frame_path, self.FRAME_MAX_W, self.FRAME_MAX_H)
        self.frame_canvas.delete("all")
        if self._frame_photo:
            self.frame_canvas.create_image(
                self.FRAME_MAX_W // 2, self.FRAME_MAX_H // 2,
                image=self._frame_photo, anchor="center"
            )
        else:
            self.frame_canvas.create_text(
                self.FRAME_MAX_W // 2, self.FRAME_MAX_H // 2,
                text="Frame not found", fill="#585b70", font=("Segoe UI", 12)
            )

        # Crop image
        self._crop_photo = self._load_image(ev.crop_path, self.CROP_MAX_W, self.CROP_MAX_H)
        self.crop_canvas.delete("all")
        if self._crop_photo:
            self.crop_canvas.create_image(
                self.CROP_MAX_W // 2, self.CROP_MAX_H // 2,
                image=self._crop_photo, anchor="center"
            )
        else:
            self.crop_canvas.create_text(
                self.CROP_MAX_W // 2, self.CROP_MAX_H // 2,
                text="Crop not found", fill="#585b70", font=("Segoe UI", 10)
            )

        # Metadata
        meta = (
            f"Type: {ev.anomaly_type}\n"
            f"Class: {ev.class_name}\n"
            f"Severity: {ev.severity}\n"
            f"Lane: {ev.lane_id}\n"
            f"Frames: {ev.start_frame}–{ev.end_frame}\n"
            f"Rule score: {ev.rule_score}\n"
            f"Fused score: {ev.fused_score}\n\n"
            f"{ev.explanation}"
        )
        self.meta_label.config(text=meta)

        # Current label
        existing = self.labels.get(ev.event_id)
        if existing:
            color = "#a6e3a1" if existing == "TP" else "#f38ba8" if existing == "FP" else "#f9e2af"
            self.current_label_var.set(f"Labeled: {existing}")
            self.current_label_display.config(foreground=color)
        else:
            self.current_label_var.set("Not labeled")
            self.current_label_display.config(foreground="#6c7086")

        self._update_stats()

    def _update_stats(self):
        tp = sum(1 for v in self.labels.values() if v == "TP")
        fp = sum(1 for v in self.labels.values() if v == "FP")
        skip = sum(1 for v in self.labels.values() if v == "skip")
        total = tp + fp + skip
        labeled_of_total = f"{total}/{len(self.events)}"
        tp_rate = f"{tp/(tp+fp)*100:.0f}%" if (tp + fp) > 0 else "—"
        self.stats_label.config(
            text=f"Labeled: {labeled_of_total}  |  TP: {tp}  FP: {fp}  Skip: {skip}  |  TP rate: {tp_rate}"
        )

    def _label(self, value: str):
        if not self.events:
            return
        ev = self.events[self.index]
        self.labels[ev.event_id] = value
        self._save_labels()
        self._next()

    def _next(self):
        if self.index < len(self.events) - 1:
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
            for ev in self.events:
                label = self.labels.get(ev.event_id)
                if label:
                    writer.writerow({
                        "event_id": ev.event_id,
                        "anomaly_type": ev.anomaly_type,
                        "class_name": ev.class_name,
                        "lane_id": ev.lane_id,
                        "severity": ev.severity,
                        "label": label,
                        "labeled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

    def _quit(self):
        self._save_labels()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Label anomaly detections as TP/FP")
    parser.add_argument("--run", default=None, help="Run ID (directory name under runs/)")
    parser.add_argument("--run-dir", default=None, help="Full path to run directory")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing run folders")
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

    events = load_events(run_dir)
    if not events:
        print(f"No events found in {run_dir}/events.csv")
        sys.exit(1)

    print(f"Loading {len(events)} events from {run_dir}")
    app = LabelingApp(run_dir, events)
    app.run()
    print(f"Labels saved to {app.labels_path}")


if __name__ == "__main__":
    main()
