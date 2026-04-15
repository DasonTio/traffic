from __future__ import annotations

import cv2
import numpy as np

from .config import LaneConfig


COLOR_BG_DARK = (20, 20, 20)
COLOR_FAST = (255, 80, 80)
COLOR_EMERGENCY = (0, 200, 255)
COLOR_WARNING = (0, 220, 255)
COLOR_CRITICAL = (60, 60, 255)
COLOR_NORMAL = (200, 200, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_TRAIL = (0, 255, 200)


def draw_trail(img, points: list[tuple[int, int]], color=COLOR_TRAIL, max_thickness: int = 3) -> None:
    """Draw a fading trail polyline. Older segments are thinner and more transparent."""
    if len(points) < 2:
        return
    num_segments = len(points) - 1
    overlay = img.copy()
    for i in range(num_segments):
        progress = (i + 1) / num_segments  # 0→1, older→newer
        alpha = 0.15 + 0.85 * progress
        thickness = max(1, int(max_thickness * progress))
        faded_color = tuple(int(c * alpha) for c in color)
        cv2.line(overlay, points[i], points[i + 1], faded_color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


def draw_label(img, text: str, position: tuple[int, int], color, font_scale: float = 0.45, thickness: int = 1) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 6, y + 4), COLOR_BG_DARK, -1)
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 6, y + 4), color, 1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius: int = 8) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)


def draw_scene_overlay(img, lanes: list[LaneConfig]) -> None:
    overlay = img.copy()
    for lane in lanes:
        fill = COLOR_EMERGENCY if lane.category == "emergency" else COLOR_FAST
        cv2.fillPoly(overlay, [lane.polygon], fill)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    for lane in lanes:
        color = COLOR_EMERGENCY if lane.category == "emergency" else COLOR_FAST
        cv2.polylines(img, [lane.polygon], True, color, 2, cv2.LINE_AA)
        moments = cv2.moments(lane.polygon)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            draw_label(img, lane.label, (cx - 40, cy), color, 0.35)


def draw_track_box(img, box: tuple[int, int, int, int], label: str, severity: str | None = None) -> None:
    x1, y1, x2, y2 = box
    if severity == "critical":
        color = COLOR_CRITICAL
        thickness = 3
    elif severity == "warning":
        color = COLOR_WARNING
        thickness = 2
    else:
        color = COLOR_NORMAL
        thickness = 1
    draw_rounded_rect(img, (x1, y1), (x2, y2), color, thickness, radius=6)
    draw_label(img, label, (x1, max(20, y1 - 8)), color, 0.38)


def draw_hud_panel(img, fps: float, counts: dict[str, int], active_events: int, ganomaly_ready: bool) -> None:
    panel_height = 88
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], panel_height), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, "TRAFFIC ANOMALY MVP", (10, 20), font, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 42), font, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"Active alerts: {active_events}", (10, 62), font, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
    gan_label = "GANomaly: ready" if ganomaly_ready else "GANomaly: checkpoints missing"
    cv2.putText(img, gan_label, (10, 82), font, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    summary = [
        ("Lane", counts.get("lane_violation", 0)),
        ("Wrong-way", counts.get("wrong_way", 0)),
        ("Sudden stop", counts.get("sudden_stop", 0)),
        ("Stopped", counts.get("stopped_vehicle", 0)),
        ("Appearance", counts.get("appearance_anomaly", 0)),
    ]
    x = 220
    for label, value in summary:
        cv2.putText(img, f"{label}: {value}", (x, 28), font, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
        x += 130
