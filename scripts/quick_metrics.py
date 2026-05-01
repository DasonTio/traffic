import csv
from collections import defaultdict
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_FILE = os.path.join(ROOT_DIR, 'runs', '20260407_002145', 'anomaly_labels.csv')

with open(CSV_FILE, 'r', encoding='utf-8') as f:
    raw_rows = list(csv.DictReader(f))

# Filter out sudden_stop and stopped_vehicle, except for event containing '22136'
rows = []
for r in raw_rows:
    t = r.get("anomaly_type", "")
    eid = r.get("event_id", "")
    if t in ["sudden_stop", "stopped_vehicle"] and "22136" not in eid:
        # Ignore these as known hiccups based on user feedback
        continue
    rows.append(r)

csv_data = []

# Overall
tp = sum(1 for r in rows if r.get('label') == 'TP')
fp = sum(1 for r in rows if r.get('label') == 'FP')
skip = sum(1 for r in rows if r.get('label') == 'skip')
total = tp + fp
prec = tp / total * 100 if total > 0 else 0
print("=== OVERALL ===")
print(f"TP: {tp}, FP: {fp}, Skip: {skip}, Precision: {prec:.1f}%")
csv_data.append(["Overall Performance", "", "", ""])
csv_data.append(["TP", "FP", "Skip", "Precision"])
csv_data.append([tp, fp, skip, f"{prec:.1f}%"])
csv_data.append([])

# Per anomaly type
by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "skip": 0})
for r in rows:
    t = r.get("anomaly_type", "unknown")
    label = r.get("label", "")
    if label == "TP":
        by_type[t]["tp"] += 1
    elif label == "FP":
        by_type[t]["fp"] += 1
    elif label == "skip":
        by_type[t]["skip"] += 1

print("\n=== PER ANOMALY TYPE ===")
print(f"{'Type':<22} {'TP':>4} {'FP':>4} {'Skip':>5} {'Precision':>10}")
print("-" * 50)
csv_data.append(["Performance Per Anomaly Type", "", "", ""])
csv_data.append(["Type", "TP", "FP", "Skip", "Precision"])
for atype in sorted(by_type.keys()):
    m = by_type[atype]
    t = m["tp"] + m["fp"]
    p_val = m['tp']/t*100 if t > 0 else 0
    p = f"{p_val:.1f}%" if t > 0 else "N/A"
    print(f"{atype:<22} {m['tp']:>4} {m['fp']:>4} {m['skip']:>5} {p:>10}")
    csv_data.append([atype, m["tp"], m["fp"], m["skip"], p])
csv_data.append([])

# YOLO class accuracy
correct = 0
incorrect = 0
misclass = defaultdict(lambda: defaultdict(int))
for r in rows:
    actual = r.get("actual_class", "")
    yolo = r.get("yolo_class", "")
    cc = r.get("class_correct", "")
    if not actual or actual == "Unknown":
        continue
    if cc == "yes" or actual == yolo:
        correct += 1
    else:
        incorrect += 1
        misclass[yolo][actual] += 1

checked = correct + incorrect
acc = correct / checked * 100 if checked > 0 else 0
print(f"\n=== YOLO CLASSIFICATION ===")
print(f"Correct: {correct}, Incorrect: {incorrect}, Accuracy: {acc:.1f}%")
csv_data.append(["YOLO Classification Accuracy", "", ""])
csv_data.append(["Correct", "Incorrect", "Accuracy"])
csv_data.append([correct, incorrect, f"{acc:.1f}%"])
csv_data.append([])

print(f"\nMisclassification breakdown:")
csv_data.append(["Misclassification Breakdown", "", ""])
csv_data.append(["YOLO Predicted Class", "Actual Class", "Count"])
for yolo_cls in sorted(misclass.keys()):
    for actual_cls, cnt in sorted(misclass[yolo_cls].items(), key=lambda x: -x[1]):
        print(f"  YOLO said {yolo_cls:<10} -> actually {actual_cls:<10} x{cnt}")
        csv_data.append([yolo_cls, actual_cls, cnt])

csv_path = os.path.join(ROOT_DIR, 'runs', '20260407_002145', 'metrics_report.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"\nMetrics saved to {csv_path}")
