# Traffic Anomaly Pipeline

Small active pipeline:

`YOLO -> ByteTrack -> Tracklet Features -> GANomaly -> Fusion -> Alert/Log`

Current behavior:
- `Bus` / `Truck` in fast lanes are rule-based anomalies
- `Car` in emergency lanes is a rule-based anomaly
- GANomaly is `Car`-only and uses image crops, not tracklet vectors

## 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The default config is:

```bash
configs/scene_config.yaml
```

## 2. Collect Data

Run the pipeline on mostly normal traffic so it can generate candidate car crops for GANomaly.

Prefer a local video file over YouTube while building the dataset:

```bash
python main.py --config configs/scene_config.yaml --source /path/to/normal_traffic.mp4 --batch
```

This run produces:
- `runs/<run_tag>/tracklets.csv`
- `runs/<run_tag>/incidents.csv`
- `runs/<run_tag>/run_summary.json`
- candidate GANomaly crops in `dataset/ganomaly/candidates/`
- crop review metadata in `dataset/ganomaly/review_manifest.csv`

## 3. Review Crops

Open:

```bash
dataset/ganomaly/review_manifest.csv
```

For each row:
- keep `approved` only for true normal car crops
- set `rejected` for bad crops, occluded crops, wrong detections, or anomalous behavior
- leave `pending` only while still reviewing

GANomaly training uses only rows with:
- `class_name=Car`
- `status=approved`

If you do not want to edit the CSV by hand, use the built-in reviewer:

```bash
python -m traffic_anomaly.review_ganomaly --manifest dataset/ganomaly/review_manifest.csv
```

Keys:
- `a` approve
- `r` reject
- `p` set back to pending
- `s` skip
- `b` go back
- `q` save and quit

## 4. Train GANomaly

Train the checkpoint from the reviewed crop manifest:

```bash
python -m traffic_anomaly.stage4_anomaly \
  --manifest dataset/ganomaly/review_manifest.csv \
  --dataset-root dataset/ganomaly \
  --output models/ganomaly_car.pt
```

This writes:

```bash
models/ganomaly_car.pt
```

## 5. Run With GANomaly Enabled

After the checkpoint exists, run the pipeline again:

```bash
python main.py --config configs/scene_config.yaml --source /path/to/eval_video.mp4 --batch
```

Check:
- `runs/<run_tag>/tracklets.csv`
- `runs/<run_tag>/incidents.csv`
- `runs/<run_tag>/run_summary.json`

What you want to see:
- non-zero `ganomaly_score` values in `tracklets.csv`
- rule anomalies still firing for lane restrictions
- GANomaly supporting or adding `appearance_anomaly` when appropriate

## 6. Tune Thresholds

Main config knobs:
- `anomaly.threshold_percentile`
- `fusion.model_escalation_threshold`
- `fusion.model_only_threshold`
- `fusion.consecutive_frames_required`
- `features.lane_violation_seconds`
- `features.wrong_way_seconds`
- `features.stopped_vehicle_seconds`

All of these live in:

```bash
configs/scene_config.yaml
```

## Notes

- Tracklet features are still important, but they are used for rules and fusion, not as GANomaly input.
- The active GANomaly dataset path is `dataset/ganomaly/`.
- Older sequence-style files such as `dataset/normal_sequences.csv` are legacy and are not part of the active training flow.
