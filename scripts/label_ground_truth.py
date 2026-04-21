"""
Ground-truth event labeler for direct video annotation.

This tool labels anomaly events from the raw video instead of from a model run.
Use short chunks first, then continue labeling the next chunk with the same CSV.

Usage examples:
    python scripts/label_ground_truth.py --source-mode local
    python scripts/label_ground_truth.py --source-mode local --start-minute 0 --duration-minutes 15
    python scripts/label_ground_truth.py --source .video\\video.mp4.mp4 --output dataset\\ground_truth_events.csv

Keyboard shortcuts:
    Right / D        = next frame
    Left / A         = previous frame
    Up / W           = jump forward by large step
    Down / S         = jump backward by large step
    PageDown / E     = jump forward by medium step
    PageUp / Q       = jump backward by medium step
    I                = mark event start at current frame
    O                = mark event end at current frame
    Enter            = save current event
    Delete           = delete selected saved event
    R                = reset current event form
    Esc              = quit
"""
from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.config import SceneConfig
from traffic_anomaly.visualization import draw_scene_overlay


ANOMALY_TYPES = [
    "lane_violation",
    "wrong_way",
    "stopped_vehicle",
    "sudden_stop",
    "appearance_anomaly",
]

VEHICLE_CLASSES = ["Car", "Bus", "Truck", "Motorcycle", "Unknown"]

CSV_FIELDS = [
    "event_gt_id",
    "source_video",
    "chunk_start_frame",
    "chunk_end_frame",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "anomaly_type",
    "actual_class",
    "lane_id",
    "notes",
]


@dataclass
class GroundTruthEvent:
    event_gt_id: str
    source_video: str
    chunk_start_frame: int
    chunk_end_frame: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    anomaly_type: str
    actual_class: str
    lane_id: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-truth event labeler for anomaly comparison.")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Scene config path.")
    parser.add_argument("--source-mode", choices=["youtube", "local"], default="local", help="Named source from config.")
    parser.add_argument("--source", default=None, help="Optional direct source override.")
    parser.add_argument("--output", default="dataset/ground_truth_events.csv", help="CSV output path.")
    parser.add_argument("--start-minute", type=float, default=0.0, help="Chunk start minute.")
    parser.add_argument("--duration-minutes", type=float, default=15.0, help="Chunk duration in minutes.")
    parser.add_argument("--frame-step", type=int, default=1, help="Small navigation step in frames.")
    parser.add_argument("--medium-step", type=int, default=30, help="Medium navigation step in frames.")
    parser.add_argument("--large-step", type=int, default=150, help="Large navigation step in frames.")
    return parser.parse_args()


def load_existing_events(path: Path) -> list[GroundTruthEvent]:
    if not path.exists():
        return []
    rows: list[GroundTruthEvent] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    GroundTruthEvent(
                        event_gt_id=row.get("event_gt_id", ""),
                        source_video=row.get("source_video", ""),
                        chunk_start_frame=int(float(row.get("chunk_start_frame", 0) or 0)),
                        chunk_end_frame=int(float(row.get("chunk_end_frame", 0) or 0)),
                        start_frame=int(float(row.get("start_frame", 0) or 0)),
                        end_frame=int(float(row.get("end_frame", 0) or 0)),
                        start_time_s=float(row.get("start_time_s", 0.0) or 0.0),
                        end_time_s=float(row.get("end_time_s", 0.0) or 0.0),
                        anomaly_type=row.get("anomaly_type", ""),
                        actual_class=row.get("actual_class", ""),
                        lane_id=row.get("lane_id", ""),
                        notes=row.get("notes", ""),
                    )
                )
            except ValueError:
                continue
    return rows


def write_events(path: Path, events: list[GroundTruthEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for event in sorted(events, key=lambda item: (item.start_frame, item.end_frame, item.event_gt_id)):
            writer.writerow(
                {
                    "event_gt_id": event.event_gt_id,
                    "source_video": event.source_video,
                    "chunk_start_frame": event.chunk_start_frame,
                    "chunk_end_frame": event.chunk_end_frame,
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "start_time_s": f"{event.start_time_s:.3f}",
                    "end_time_s": f"{event.end_time_s:.3f}",
                    "anomaly_type": event.anomaly_type,
                    "actual_class": event.actual_class,
                    "lane_id": event.lane_id,
                    "notes": event.notes,
                }
            )


class GroundTruthLabeler:
    FRAME_W = 960
    FRAME_H = 540

    def __init__(
        self,
        scene: SceneConfig,
        source: str,
        output_path: Path,
        start_minute: float,
        duration_minutes: float,
        frame_step: int,
        medium_step: int,
        large_step: int,
    ):
        self.scene = scene
        self.source = source
        self.output_path = output_path.resolve()
        self.events = load_existing_events(self.output_path)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = reported_fps if reported_fps and reported_fps > 1.0 else self.scene.source_fps
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.chunk_start_frame = max(1, int(start_minute * 60.0 * self.video_fps) + 1)
        chunk_length = max(1, int(duration_minutes * 60.0 * self.video_fps))
        if self.total_frames > 0:
            self.chunk_end_frame = min(self.total_frames, self.chunk_start_frame + chunk_length - 1)
        else:
            self.chunk_end_frame = self.chunk_start_frame + chunk_length - 1

        self.current_frame_idx = self.chunk_start_frame
        self.frame_step = max(1, frame_step)
        self.medium_step = max(1, medium_step)
        self.large_step = max(1, large_step)
        self.current_frame = None

        self.root = tk.Tk()
        self.root.title("Ground Truth Event Labeler")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1500x920")

        self.start_frame_var = tk.StringVar(value="")
        self.end_frame_var = tk.StringVar(value="")
        self.anomaly_type_var = tk.StringVar(value=ANOMALY_TYPES[0])
        self.actual_class_var = tk.StringVar(value=VEHICLE_CLASSES[0])
        lane_options = [lane.id for lane in self.scene.lanes] + ["unassigned"]
        self.lane_id_var = tk.StringVar(value=lane_options[0] if lane_options else "unassigned")
        self.notes_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.frame_info_var = tk.StringVar(value="")

        self._build_ui(lane_options)
        self._bind_keys()
        self._show_frame(self.current_frame_idx)

    def _build_ui(self, lane_options: list[str]) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 13, "bold"))
        style.configure("Meta.TLabel", background="#1e1e2e", foreground="#a6adc8", font=("Segoe UI", 9))
        style.configure("TFrame", background="#1e1e2e")

        top = ttk.Frame(self.root, style="TFrame")
        top.pack(fill="x", padx=12, pady=(10, 6))

        ttk.Label(top, text="Ground Truth Event Labeler", style="Title.TLabel").pack(side="left")
        ttk.Label(top, textvariable=self.frame_info_var, style="Meta.TLabel").pack(side="right")

        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        left = ttk.Frame(body, style="TFrame")
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body, style="TFrame")
        right.pack(side="right", fill="y", padx=(12, 0))

        self.frame_canvas = tk.Canvas(left, width=self.FRAME_W, height=self.FRAME_H, bg="#11111b", highlightthickness=0)
        self.frame_canvas.pack(fill="both", expand=False)

        nav = ttk.Frame(left, style="TFrame")
        nav.pack(fill="x", pady=(8, 0))
        ttk.Button(nav, text=f"Prev {self.large_step}", command=lambda: self._move(-self.large_step)).pack(side="left", padx=3)
        ttk.Button(nav, text=f"Prev {self.medium_step}", command=lambda: self._move(-self.medium_step)).pack(side="left", padx=3)
        ttk.Button(nav, text=f"Prev {self.frame_step}", command=lambda: self._move(-self.frame_step)).pack(side="left", padx=3)
        ttk.Button(nav, text=f"Next {self.frame_step}", command=lambda: self._move(self.frame_step)).pack(side="left", padx=3)
        ttk.Button(nav, text=f"Next {self.medium_step}", command=lambda: self._move(self.medium_step)).pack(side="left", padx=3)
        ttk.Button(nav, text=f"Next {self.large_step}", command=lambda: self._move(self.large_step)).pack(side="left", padx=3)

        help_text = (
            "Workflow: label 15-minute chunks, mark start/end frame for each real event, save to one CSV.\n"
            "Use frame ranges as ground truth. Ignore tracker IDs here."
        )
        ttk.Label(left, text=help_text, style="Meta.TLabel", justify="left").pack(anchor="w", pady=(8, 0))

        ttk.Label(right, text="Current Event", style="Title.TLabel").pack(anchor="w")

        form = ttk.Frame(right, style="TFrame")
        form.pack(fill="x", pady=(8, 10))

        self._add_field(form, "Start frame", self.start_frame_var)
        self._add_field(form, "End frame", self.end_frame_var)

        ttk.Label(form, text="Anomaly type").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(form, textvariable=self.anomaly_type_var, values=ANOMALY_TYPES, state="readonly", width=28).pack(anchor="w")

        ttk.Label(form, text="Actual class").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(form, textvariable=self.actual_class_var, values=VEHICLE_CLASSES, state="readonly", width=28).pack(anchor="w")

        ttk.Label(form, text="Lane ID").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(form, textvariable=self.lane_id_var, values=lane_options, state="readonly", width=28).pack(anchor="w")

        ttk.Label(form, text="Notes").pack(anchor="w", pady=(6, 0))
        self.notes_entry = ttk.Entry(form, textvariable=self.notes_var, width=32)
        self.notes_entry.pack(anchor="w")

        actions = ttk.Frame(right, style="TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="Mark Start (I)", command=self._mark_start).pack(fill="x", pady=2)
        ttk.Button(actions, text="Mark End (O)", command=self._mark_end).pack(fill="x", pady=2)
        ttk.Button(actions, text="Save Event (Enter)", command=self._save_current_event).pack(fill="x", pady=2)
        ttk.Button(actions, text="Reset Form (R)", command=self._reset_form).pack(fill="x", pady=2)
        ttk.Button(actions, text="Delete Selected (Del)", command=self._delete_selected_event).pack(fill="x", pady=2)

        ttk.Label(right, text="Saved Events", style="Title.TLabel").pack(anchor="w")
        self.event_list = tk.Listbox(right, width=58, height=20, bg="#11111b", fg="#cdd6f4", selectbackground="#45475a")
        self.event_list.pack(fill="both", expand=True, pady=(8, 0))
        self.event_list.bind("<<ListboxSelect>>", self._on_select_event)

        ttk.Label(right, textvariable=self.status_var, style="Meta.TLabel", justify="left", wraplength=380).pack(anchor="w", pady=(8, 0))

        self._refresh_event_list()

    def _add_field(self, parent, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=var, width=32).pack(anchor="w")

    def _bind_keys(self) -> None:
        self.root.bind("<Right>", lambda e: self._move(self.frame_step))
        self.root.bind("<Left>", lambda e: self._move(-self.frame_step))
        self.root.bind("<Up>", lambda e: self._move(self.large_step))
        self.root.bind("<Down>", lambda e: self._move(-self.large_step))
        self.root.bind("d", lambda e: self._move(self.frame_step))
        self.root.bind("a", lambda e: self._move(-self.frame_step))
        self.root.bind("e", lambda e: self._move(self.medium_step))
        self.root.bind("q", lambda e: self._move(-self.medium_step))
        self.root.bind("w", lambda e: self._move(self.large_step))
        self.root.bind("s", lambda e: self._move(-self.large_step))
        self.root.bind("i", lambda e: self._mark_start())
        self.root.bind("o", lambda e: self._mark_end())
        self.root.bind("<Return>", lambda e: self._save_current_event())
        self.root.bind("<Delete>", lambda e: self._delete_selected_event())
        self.root.bind("r", lambda e: self._reset_form())
        self.root.bind("<Escape>", lambda e: self._quit())

    def _seek_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
        ret, frame = self.cap.read()
        if not ret:
            return None
        display = frame.copy()
        draw_scene_overlay(display, self.scene.lanes)
        return display

    def _show_frame(self, frame_idx: int) -> None:
        bounded = max(self.chunk_start_frame, min(frame_idx, self.chunk_end_frame))
        frame = self._seek_frame(bounded)
        if frame is None:
            self.status_var.set(f"Could not read frame {bounded}.")
            return
        self.current_frame_idx = bounded
        self.current_frame = frame
        self._frame_photo = self._cv2_to_tk(frame, self.FRAME_W, self.FRAME_H)
        self.frame_canvas.delete("all")
        if self._frame_photo:
            self.frame_canvas.create_image(self.FRAME_W // 2, self.FRAME_H // 2, image=self._frame_photo, anchor="center")
        else:
            self.frame_canvas.create_text(
                self.FRAME_W // 2,
                self.FRAME_H // 2,
                text="Could not render frame",
                fill="#cdd6f4",
                font=("Segoe UI", 12),
            )

        timestamp = (self.current_frame_idx - 1) / max(self.video_fps, 1e-6)
        chunk_start_s = (self.chunk_start_frame - 1) / max(self.video_fps, 1e-6)
        chunk_end_s = (self.chunk_end_frame - 1) / max(self.video_fps, 1e-6)
        self.frame_info_var.set(
            f"Frame {self.current_frame_idx}/{self.total_frames or '?'}  |  "
            f"Time {timestamp:.1f}s  |  Chunk {chunk_start_s/60:.1f}-{chunk_end_s/60:.1f} min"
        )

    def _cv2_to_tk(self, frame, max_w: int, max_h: int):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((max_w, max_h), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _move(self, delta: int) -> None:
        self._show_frame(self.current_frame_idx + delta)

    def _mark_start(self) -> None:
        self.start_frame_var.set(str(self.current_frame_idx))
        self.status_var.set(f"Start frame set to {self.current_frame_idx}.")

    def _mark_end(self) -> None:
        self.end_frame_var.set(str(self.current_frame_idx))
        self.status_var.set(f"End frame set to {self.current_frame_idx}.")

    def _save_current_event(self) -> None:
        try:
            start_frame = int(self.start_frame_var.get().strip())
            end_frame = int(self.end_frame_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid frame range", "Start frame and end frame must be integers.")
            return

        if start_frame > end_frame:
            messagebox.showerror("Invalid frame range", "Start frame must be <= end frame.")
            return

        if start_frame < self.chunk_start_frame or end_frame > self.chunk_end_frame:
            messagebox.showerror("Out of chunk", "Event must stay inside the selected chunk.")
            return

        next_id = self._next_event_id()
        event = GroundTruthEvent(
            event_gt_id=next_id,
            source_video=self.source,
            chunk_start_frame=self.chunk_start_frame,
            chunk_end_frame=self.chunk_end_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_s=(start_frame - 1) / max(self.video_fps, 1e-6),
            end_time_s=(end_frame - 1) / max(self.video_fps, 1e-6),
            anomaly_type=self.anomaly_type_var.get().strip(),
            actual_class=self.actual_class_var.get().strip(),
            lane_id=self.lane_id_var.get().strip(),
            notes=self.notes_var.get().strip(),
        )
        self.events.append(event)
        write_events(self.output_path, self.events)
        self._refresh_event_list()
        self._reset_form()
        self.status_var.set(f"Saved {event.event_gt_id}: frames {start_frame}-{end_frame}.")

    def _refresh_event_list(self) -> None:
        self.event_list.delete(0, tk.END)
        for event in sorted(self.events, key=lambda item: (item.start_frame, item.end_frame, item.event_gt_id)):
            label = (
                f"{event.event_gt_id} | {event.start_frame}-{event.end_frame} | "
                f"{event.anomaly_type} | {event.actual_class} | {event.lane_id}"
            )
            self.event_list.insert(tk.END, label)

    def _on_select_event(self, event=None) -> None:
        selection = self.event_list.curselection()
        if not selection:
            return
        ordered = sorted(self.events, key=lambda item: (item.start_frame, item.end_frame, item.event_gt_id))
        chosen = ordered[selection[0]]
        self.start_frame_var.set(str(chosen.start_frame))
        self.end_frame_var.set(str(chosen.end_frame))
        self.anomaly_type_var.set(chosen.anomaly_type)
        self.actual_class_var.set(chosen.actual_class)
        self.lane_id_var.set(chosen.lane_id)
        self.notes_var.set(chosen.notes)
        self._show_frame(chosen.start_frame)
        self.status_var.set(f"Loaded {chosen.event_gt_id} into the form.")

    def _delete_selected_event(self) -> None:
        selection = self.event_list.curselection()
        if not selection:
            self.status_var.set("No saved event selected.")
            return
        ordered = sorted(self.events, key=lambda item: (item.start_frame, item.end_frame, item.event_gt_id))
        chosen = ordered[selection[0]]
        self.events = [event for event in self.events if event.event_gt_id != chosen.event_gt_id]
        write_events(self.output_path, self.events)
        self._refresh_event_list()
        self.status_var.set(f"Deleted {chosen.event_gt_id}.")

    def _reset_form(self) -> None:
        self.start_frame_var.set("")
        self.end_frame_var.set("")
        self.anomaly_type_var.set(ANOMALY_TYPES[0])
        self.actual_class_var.set(VEHICLE_CLASSES[0])
        if self.scene.lanes:
            self.lane_id_var.set(self.scene.lanes[0].id)
        else:
            self.lane_id_var.set("unassigned")
        self.notes_var.set("")

    def _next_event_id(self) -> str:
        max_id = 0
        for event in self.events:
            raw = event.event_gt_id.removeprefix("gt_")
            if raw.isdigit():
                max_id = max(max_id, int(raw))
        return f"gt_{max_id + 1:04d}"

    def _quit(self) -> None:
        self.cap.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    scene = SceneConfig.load(args.config, source_mode=args.source_mode)
    source = args.source or scene.video_source
    output = Path(args.output)

    labeler = GroundTruthLabeler(
        scene=scene,
        source=source,
        output_path=output,
        start_minute=args.start_minute,
        duration_minutes=args.duration_minutes,
        frame_step=args.frame_step,
        medium_step=args.medium_step,
        large_step=args.large_step,
    )
    print(f"Video source: {source}")
    print(f"Output CSV: {output.resolve()}")
    print(
        f"Chunk: start={args.start_minute:.2f} min, duration={args.duration_minutes:.2f} min, "
        f"frames={labeler.chunk_start_frame}-{labeler.chunk_end_frame}"
    )
    labeler.run()
    write_events(output.resolve(), labeler.events)
    print(f"Saved {len(labeler.events)} events to {output.resolve()}")


if __name__ == "__main__":
    main()
