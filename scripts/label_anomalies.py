"""
Anomaly Labeling Tool — Label detected anomalies as TP / FP / Skip.

Also tracks YOLO classification correctness (actual vehicle class).

Usage:
    python scripts/label_anomalies.py                   # auto-picks latest run
    python scripts/label_anomalies.py --run 20260407_002145

Keyboard shortcuts:
    T / 1      = True Positive (real anomaly)
    F / 0      = False Positive (wrong detection)
    S / Space  = Skip / Unsure
    Left       = Previous event
    Right      = Next event
    Q / Esc    = Quit and save
    C          = Cycle through correct class options
"""
from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk


VEHICLE_CLASSES = ["Car", "Bus", "Truck", "Motorcycle", "Unknown"]


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


@dataclass
class LabelData:
    label: str  # TP, FP, skip
    actual_class: str  # what the vehicle actually is (Car, Truck, Bus, etc.)


LABEL_FIELDS = [
    "event_id",
    "anomaly_type",
    "yolo_class",
    "actual_class",
    "class_correct",
    "lane_id",
    "severity",
    "label",
    "labeled_at",
]


def load_existing_labels(path: Path) -> dict[str, LabelData]:
    if not path.exists():
        return {}
    labels: dict[str, LabelData] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["event_id"]] = LabelData(
                label=row.get("label", "skip"),
                actual_class=row.get("actual_class", row.get("yolo_class", "")),
            )
    return labels


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
        self.root.geometry("1100x760")

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
        style.configure("ClassLabel.TLabel", font=("Segoe UI", 10, "bold"), foreground="#cba6f7", background="#1e1e2e")
        style.configure("TP.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("FP.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("Skip.TButton", font=("Segoe UI", 11))
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TCombobox", font=("Segoe UI", 10))

        # Top bar
        top = ttk.Frame(self.root, style="TFrame")
        top.pack(fill="x", padx=12, pady=(8, 0))
        self.title_label = ttk.Label(top, style="Title.TLabel")
        self.title_label.pack(side="left")
        self.stats_label = ttk.Label(top, style="Stats.TLabel")
        self.stats_label.pack(side="right")

        # Image area
        img_frame = ttk.Frame(self.root, style="TFrame")
        img_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.frame_canvas = tk.Canvas(img_frame, bg="#181825", highlightthickness=0)
        self.frame_canvas.pack(side="left", fill="both", expand=True, padx=(0, 4))

        right_panel = ttk.Frame(img_frame, style="TFrame")
        right_panel.pack(side="right", fill="y", padx=(4, 0))

        self.crop_canvas = tk.Canvas(right_panel, bg="#181825", width=self.CROP_MAX_W, height=self.CROP_MAX_H, highlightthickness=0)
        self.crop_canvas.pack()

        self.meta_label = ttk.Label(right_panel, style="Meta.TLabel", wraplength=240, justify="left")
        self.meta_label.pack(pady=(8, 0), fill="x")

        # Correct class selector
        class_frame = ttk.Frame(right_panel, style="TFrame")
        class_frame.pack(pady=(10, 0), fill="x")

        ttk.Label(class_frame, text="Actual class:", style="ClassLabel.TLabel").pack(side="left")
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(
            class_frame, textvariable=self.class_var,
            values=VEHICLE_CLASSES, state="readonly", width=12,
            style="TCombobox",
        )
        self.class_combo.pack(side="left", padx=(6, 0))

        # Classification match indicator
        self.class_match_var = tk.StringVar(value="")
        self.class_match_label = tk.Label(
            class_frame, textvariable=self.class_match_var,
            font=("Segoe UI", 10, "bold"), bg="#1e1e2e", fg="#a6e3a1"
        )
        self.class_match_label.pack(side="left", padx=(8, 0))

        # Bind class change
        self.class_combo.bind("<<ComboboxSelected>>", self._on_class_change)

        # Label showing current label
        self.current_label_var = tk.StringVar(value="")
        self.current_label_display = tk.Label(
            right_panel, textvariable=self.current_label_var,
            font=("Segoe UI", 12, "bold"), fg="#f9e2af", bg="#1e1e2e"
        )
        self.current_label_display.pack(pady=(6, 0))

        # Bottom buttons
        btn_frame = ttk.Frame(self.root, style="TFrame")
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
        self.root.bind("c", lambda e: self._cycle_class())

    def _cycle_class(self):
        """Cycle through vehicle classes with the C key."""
        current = self.class_var.get()
        try:
            idx = VEHICLE_CLASSES.index(current)
            next_idx = (idx + 1) % len(VEHICLE_CLASSES)
        except ValueError:
            next_idx = 0
        self.class_var.set(VEHICLE_CLASSES[next_idx])
        self._on_class_change()

    def _on_class_change(self, event=None):
        ev = self.events[self.index]
        actual = self.class_var.get()
        if actual and actual != "Unknown":
            if actual == ev.class_name:
                self.class_match_var.set("✓ Correct")
                self.class_match_label.config(fg="#a6e3a1")
            else:
                self.class_match_var.set(f"✗ YOLO said {ev.class_name}")
                self.class_match_label.config(fg="#f38ba8")
        else:
            self.class_match_var.set("")
        # Save the actual class immediately
        label_data = self.labels.get(ev.event_id)
        if label_data:
            label_data.actual_class = actual

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
            f"YOLO class: {ev.class_name}\n"
            f"Severity: {ev.severity}\n"
            f"Lane: {ev.lane_id}\n"
            f"Frames: {ev.start_frame}–{ev.end_frame}\n"
            f"Rule score: {ev.rule_score}\n"
            f"Fused score: {ev.fused_score}\n\n"
            f"{ev.explanation}"
        )
        self.meta_label.config(text=meta)

        # Restore saved actual class, or default to YOLO's class
        existing = self.labels.get(ev.event_id)
        if existing and existing.actual_class:
            self.class_var.set(existing.actual_class)
        else:
            self.class_var.set(ev.class_name)  # default: assume YOLO is correct
        self._on_class_change()

        # Current label
        if existing:
            label = existing.label
            color = "#a6e3a1" if label == "TP" else "#f38ba8" if label == "FP" else "#f9e2af"
            self.current_label_var.set(f"Labeled: {label}")
            self.current_label_display.config(fg=color)
        else:
            self.current_label_var.set("Not labeled")
            self.current_label_display.config(fg="#6c7086")

        self._update_stats()

    def _update_stats(self):
        tp = sum(1 for v in self.labels.values() if v.label == "TP")
        fp = sum(1 for v in self.labels.values() if v.label == "FP")
        skip = sum(1 for v in self.labels.values() if v.label == "skip")
        total = tp + fp + skip

        # Classification accuracy (how often YOLO class matches actual class)
        class_checked = 0
        class_correct = 0
        for ev in self.events:
            data = self.labels.get(ev.event_id)
            if data and data.actual_class and data.actual_class != "Unknown":
                class_checked += 1
                if data.actual_class == ev.class_name:
                    class_correct += 1
        class_acc = f"{class_correct/class_checked*100:.0f}%" if class_checked > 0 else "—"

        labeled_of_total = f"{total}/{len(self.events)}"
        tp_rate = f"{tp/(tp+fp)*100:.0f}%" if (tp + fp) > 0 else "—"
        self.stats_label.config(
            text=f"Labeled: {labeled_of_total}  |  TP: {tp}  FP: {fp}  Skip: {skip}  |  "
                 f"TP rate: {tp_rate}  |  YOLO class acc: {class_acc}"
        )

    def _label(self, value: str):
        if not self.events:
            return
        ev = self.events[self.index]
        actual_class = self.class_var.get() or ev.class_name
        self.labels[ev.event_id] = LabelData(label=value, actual_class=actual_class)
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
                label_data = self.labels.get(ev.event_id)
                if label_data:
                    actual = label_data.actual_class or ev.class_name
                    writer.writerow({
                        "event_id": ev.event_id,
                        "anomaly_type": ev.anomaly_type,
                        "yolo_class": ev.class_name,
                        "actual_class": actual,
                        "class_correct": "yes" if actual == ev.class_name else "no",
                        "lane_id": ev.lane_id,
                        "severity": ev.severity,
                        "label": label_data.label,
                        "labeled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

    def _quit(self):
        self._save_labels()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Label anomaly detections as TP/FP + verify YOLO class")
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
