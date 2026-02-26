import cv2
import numpy as np
import math
import time
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube

# ───────────────────────────────────────────────────────────────
# CONFIG — adjust these values as needed
# ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45          # min detection confidence (0.0 - 1.0)
STOPPED_THRESHOLD_FRAMES = 30      # frames to confirm a vehicle is stopped
MOVEMENT_THRESHOLD = 5.0           # max pixel movement to count as "stopped"
DWELL_FRAMES_REQUIRED = 3          # consecutive frames in zone before flagging

# Color palette (BGR)
COLOR_BG_DARK = (20, 20, 20)
COLOR_FAST_LANE = (255, 80, 80)       # soft blue
COLOR_EMERGENCY = (0, 200, 255)       # amber/yellow
COLOR_ALERT_RED = (60, 60, 255)       # vibrant red
COLOR_ALERT_YELLOW = (0, 220, 255)    # bright yellow
COLOR_NORMAL_BOX = (200, 200, 200)    # light gray
COLOR_GREEN_DOT = (100, 255, 120)     # green
COLOR_WHITE = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)

COCO_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Output directory for anomaly evidence
OUTPUT_DIR = "anomalies"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "crops"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)

# ───────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # Four corner arcs + four lines
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)


def draw_label(img, text, position, color, font_scale=0.45, thickness=1):
    """Draw text with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Background pill
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 6, y + 4), COLOR_BG_DARK, -1)
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 6, y + 4), color, 1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_hud_panel(img, fast_count, emg_count, fps, frame_w):
    """Draw a translucent HUD panel at the top of the frame."""
    panel_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, panel_h), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Divider line
    cv2.line(img, (0, panel_h), (frame_w, panel_h), (60, 60, 60), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(img, "TRAFFIC MONITOR", (10, 20), font, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # FPS badge
    fps_text = f"FPS: {fps:.0f}"
    (fw, _), _ = cv2.getTextSize(fps_text, font, 0.4, 1)
    cv2.putText(img, fps_text, (frame_w - fw - 10, 20), font, 0.4, COLOR_GREEN_DOT, 1, cv2.LINE_AA)

    # --- Fast Lane Violations ---
    # Icon: small filled square
    cv2.rectangle(img, (10, 32), (22, 44), COLOR_ALERT_RED, -1)
    cv2.putText(img, f"Fast Lane Violations: {fast_count}", (28, 44), font, 0.5, COLOR_ALERT_RED, 1, cv2.LINE_AA)

    # --- Emergency Lane Stops ---
    cv2.rectangle(img, (10, 52), (22, 64), COLOR_ALERT_YELLOW, -1)
    cv2.putText(img, f"Emergency Lane Stops: {emg_count}", (28, 64), font, 0.5, COLOR_ALERT_YELLOW, 1, cv2.LINE_AA)


def draw_roi_zones(img, fast_zones, emg_zones):
    """Draw semi-transparent ROI polygons with labels."""
    overlay = img.copy()

    # Fast lane zones — soft blue fill
    cv2.fillPoly(overlay, fast_zones, COLOR_FAST_LANE)
    # Emergency lane zones — amber fill
    cv2.fillPoly(overlay, emg_zones, COLOR_EMERGENCY)

    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Outlines
    cv2.polylines(img, fast_zones, isClosed=True, color=COLOR_FAST_LANE, thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(img, emg_zones, isClosed=True, color=COLOR_EMERGENCY, thickness=2, lineType=cv2.LINE_AA)

    # Zone labels
    for i, zone in enumerate(fast_zones):
        M = cv2.moments(zone)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            draw_label(img, f"FAST LANE {i+1}", (cx - 40, cy), COLOR_FAST_LANE, 0.35)

    for i, zone in enumerate(emg_zones):
        M = cv2.moments(zone)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            draw_label(img, f"EMERGENCY {i+1}", (cx - 40, cy), COLOR_EMERGENCY, 0.35)


def save_anomaly(frame, display, box, track_id, class_name, anomaly_type, conf, csv_writer, frame_count):
    """Save cropped vehicle, full annotated frame, and log to CSV."""
    x1, y1, x2, y2 = box
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{anomaly_type}_{int(track_id)}_{timestamp}"

    # Crop the vehicle from the original frame (with padding)
    h, w = frame.shape[:2]
    pad = 15
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]

    crop_path = os.path.join(OUTPUT_DIR, "crops", f"{tag}.jpg")
    frame_path = os.path.join(OUTPUT_DIR, "frames", f"{tag}.jpg")

    cv2.imwrite(crop_path, crop)
    cv2.imwrite(frame_path, display)

    csv_writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        frame_count,
        int(track_id),
        class_name,
        anomaly_type,
        f"{conf:.3f}",
        f"({x1},{y1})-({x2},{y2})",
        crop_path,
        frame_path
    ])


# ───────────────────────────────────────────────────────────────
# SETUP
# ───────────────────────────────────────────────────────────────
model = YOLO("yolo11n.pt")
video_url = "https://www.youtube.com/watch?v=wWSSUfL2LpE"
cap = cap_from_youtube(video_url, resolution='360p')

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define ROI polygons
fast_lane_1 = np.array([[116, 303], [152, 261], [194, 217], [219, 197], [252, 195],
                         [241, 217], [234, 239], [222, 285], [212, 327], [106, 321]], np.int32)
fast_lane_2 = np.array([[278, 193], [315, 193], [332, 221], [350, 257], [368, 284],
                         [395, 324], [404, 338], [285, 339], [280, 295], [277, 248], [277, 219]], np.int32)
emergency_lane_1 = np.array([[4, 199], [51, 180], [120, 167], [152, 188],
                              [113, 203], [70, 225], [37, 239], [3, 256]], np.int32)
emergency_lane_2 = np.array([[382, 181], [425, 206], [462, 221], [479, 220],
                              [479, 178], [434, 163], [405, 156], [382, 147]], np.int32)

fast_zones = [fast_lane_1, fast_lane_2]
emg_zones = [emergency_lane_1, emergency_lane_2]

# Tracking state
track_history = {}
dwell_counter = {}  # track_id -> consecutive frames in fast lane
fast_lane_anomalies_count = 0
emergency_lane_anomalies_count = 0
counted_fast_lane_ids = set()
counted_emergency_ids = set()

prev_time = time.time()
fps = 0.0
frame_count = 0

# Open CSV log file
csv_path = os.path.join(OUTPUT_DIR, "anomaly_log.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Frame", "Track_ID", "Class", "Anomaly_Type",
                     "Confidence", "BBox", "Crop_Path", "Frame_Path"])

print(f"Frame size: {frame_w}x{frame_h}")
print(f"Anomaly output: {os.path.abspath(OUTPUT_DIR)}/")
print("Processing traffic data... Press 'q' to stop.")

# ───────────────────────────────────────────────────────────────
# MAIN LOOP
# ───────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # FPS calculation
    curr_time = time.time()
    fps = 0.8 * fps + 0.2 * (1.0 / max(curr_time - prev_time, 1e-6))
    prev_time = curr_time

    results = model.track(frame, persist=True, classes=[2, 5, 7], conf=CONFIDENCE_THRESHOLD, verbose=False)
    display = frame.copy()

    # Draw ROI zones
    draw_roi_zones(display, fast_zones, emg_zones)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        # Clean up old track IDs (memory management)
        active_ids = set(track_ids.tolist())
        stale_ids = [tid for tid in track_history if tid not in active_ids]
        for tid in stale_ids:
            del track_history[tid]
            dwell_counter.pop(tid, None)

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            center_x = (x1 + x2) // 2
            center_y = y2  # bottom-center

            class_name = COCO_NAMES.get(int(class_id), "Vehicle")

            # Update track history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((center_x, center_y))
            if len(track_history[track_id]) > STOPPED_THRESHOLD_FRAMES:
                track_history[track_id].pop(0)

            # ── Check: Truck in Fast Lane ──
            is_anomaly = False
            if int(class_id) == 7:
                in_fl1 = cv2.pointPolygonTest(fast_lane_1, (center_x, center_y), False) >= 0
                in_fl2 = cv2.pointPolygonTest(fast_lane_2, (center_x, center_y), False) >= 0

                if in_fl1 or in_fl2:
                    dwell_counter[track_id] = dwell_counter.get(track_id, 0) + 1

                    if dwell_counter[track_id] >= DWELL_FRAMES_REQUIRED:
                        is_anomaly = True
                        if track_id not in counted_fast_lane_ids:
                            fast_lane_anomalies_count += 1
                            counted_fast_lane_ids.add(track_id)
                            save_anomaly(frame, display, (x1, y1, x2, y2),
                                         track_id, class_name, "truck_in_fast_lane",
                                         conf, csv_writer, frame_count)

                        # Pulsing border effect
                        pulse = int(180 + 75 * math.sin(time.time() * 6))
                        border_color = (60, 60, pulse)
                        draw_rounded_rect(display, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_color, 3, radius=6)
                        draw_label(display, "TRUCK IN FAST LANE", (x1, y1 - 14), COLOR_ALERT_RED, 0.4, 1)
                else:
                    dwell_counter.pop(track_id, None)

            # ── Check: Stopped in Emergency Lane ──
            in_el1 = cv2.pointPolygonTest(emergency_lane_1, (center_x, center_y), False) >= 0
            in_el2 = cv2.pointPolygonTest(emergency_lane_2, (center_x, center_y), False) >= 0

            if (in_el1 or in_el2) and len(track_history[track_id]) >= STOPPED_THRESHOLD_FRAMES:
                start_pt = track_history[track_id][0]
                end_pt = track_history[track_id][-1]
                dist = math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)

                if dist < MOVEMENT_THRESHOLD:
                    is_anomaly = True
                    if track_id not in counted_emergency_ids:
                        emergency_lane_anomalies_count += 1
                        counted_emergency_ids.add(track_id)
                        save_anomaly(frame, display, (x1, y1, x2, y2),
                                     track_id, class_name, "stopped_in_emergency",
                                     conf, csv_writer, frame_count)

                    pulse = int(180 + 75 * math.sin(time.time() * 6))
                    border_color = (0, pulse, 255)
                    draw_rounded_rect(display, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_color, 3, radius=6)
                    draw_label(display, "STOPPED VEHICLE", (x1, y1 - 14), COLOR_ALERT_YELLOW, 0.4, 1)

            # ── Draw normal vehicle boxes ──
            if not is_anomaly:
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_NORMAL_BOX, 1, cv2.LINE_AA)
                label = f"{class_name} #{int(track_id)}"
                draw_label(display, label, (x1, y1 - 4), COLOR_NORMAL_BOX, 0.35)

            # Reference dot
            cv2.circle(display, (center_x, center_y), 3, COLOR_GREEN_DOT, -1, cv2.LINE_AA)

    # Draw HUD panel
    draw_hud_panel(display, fast_lane_anomalies_count, emergency_lane_anomalies_count, fps, frame_w)

    cv2.imshow("YOLO Traffic Analysis - Tol", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print(f"\n{'='*50}")
print(f"  Total Fast Lane Violations : {fast_lane_anomalies_count}")
print(f"  Total Emergency Lane Stops : {emergency_lane_anomalies_count}")
print(f"  Anomaly log saved to       : {os.path.abspath(csv_path)}")
print(f"  Cropped evidence in        : {os.path.abspath(os.path.join(OUTPUT_DIR, 'crops'))}/")
print(f"  Full frames in             : {os.path.abspath(os.path.join(OUTPUT_DIR, 'frames'))}/")
print(f"{'='*50}")