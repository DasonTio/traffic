# Traffic Anomaly Architecture

This document describes the active runtime in this repository.

It does not follow the older `stage1_detect.py -> stage6_alerts.py` layout.
The current code path is centered around:

- `main.py`
- `traffic_anomaly/pipeline.py`

## One-Sentence Summary

The system watches a traffic video, detects and tracks vehicles, converts each
vehicle into motion and lane-aware features, applies rule-based anomaly logic,
adds appearance-based anomaly support with GANomaly, and stores alerts plus
clean sequences for future model training.

## End-to-End Flow

```text
Video Source
  -> Optional Image Enhancement
  -> YOLO Detection + ByteTrack Tracking
  -> Lane Mapping + Homography Projection
  -> Per-Track Motion / Behavior Features
  -> Rule Evaluation
  -> GANomaly Appearance Scoring
  -> Rule / GANomaly Fusion
  -> Event Management + Evidence Saving
  -> CSV Logging + Clean Sequence Export
```

## Why The Pipeline Is Structured This Way

The pipeline mixes two kinds of signals:

- explicit traffic logic
  - examples: wrong-way driving, bus or truck staying in a forbidden lane
- learned visual abnormality
  - examples: a crop that looks unusual compared with normal vehicle crops

This hybrid design is practical for traffic CCTV:

- rules are interpretable and easy to debug
- learned appearance scoring can catch things rules do not describe well
- tracking makes the system temporal instead of frame-only

## Active Runtime Modules

### `main.py`

Small entrypoint that parses CLI arguments and starts
`TrafficAnomalyPipeline`.

### `traffic_anomaly/config.py`

Loads the YAML config and converts it into a runtime-friendly `SceneConfig`.

What it prepares:

- video source selection
- YOLO settings
- ByteTrack config path
- output directories
- homography matrix
- lane polygons
- lane direction vectors
- class-to-forbidden-lane policy
- thresholds
- GANomaly checkpoint paths

Important config sections:

- `video`
- `model`
- `tracking`
- `output`
- `homography`
- `lanes`
- `class_lane_policy`
- `ganomaly`
- `thresholds`
- `enhancement`

### `traffic_anomaly/pipeline.py`

This is the orchestrator. It owns the full runtime loop.

Per processed frame it does:

1. Read one frame from the selected source.
2. Optionally apply enhancement.
3. Run `YOLO.track(..., persist=True)` so detections keep stable IDs.
4. For each tracked object:
   - compute a road-contact proxy point
   - assign a lane
   - project that point into BEV
   - crop the vehicle image
   - score the crop with GANomaly
   - update the per-track feature state
5. Build lane-level congestion snapshots.
6. Evaluate rule anomalies.
7. Fuse rule output with GANomaly support.
8. Update active events and save evidence.
9. Log per-track rows.
10. Save clean non-anomalous image sequences for future GANomaly training.

### `traffic_anomaly/geometry.py`

Geometry helpers that turn raw image-space observations into lane-aware motion
context.

Main helpers:

- `bottom_center(box)`
- `project_point(homography, point)`
- `find_lane(point, lanes)`
- `heading_alignment(heading, lane)`

### `traffic_anomaly/tracklets.py`

This is the behavior feature generator.

`TrackManager` does not create IDs. YOLO + ByteTrack already do that. Instead,
`TrackManager` keeps rolling state per `track_id` and turns detections into
motion features, counters, and smoothed signals.

### `traffic_anomaly/rules.py`

Contains:

- `build_lane_snapshots()`
- `RuleEngine.evaluate()`

This is the rule-based anomaly layer.

### `traffic_anomaly/ganomaly.py`

Contains both:

- GANomaly training
- GANomaly inference

This is the appearance-based anomaly layer.

### `traffic_anomaly/events.py`

Merges repeated frame-level anomaly hits into persistent events so alerts are
not emitted as isolated one-frame spikes.

### `traffic_anomaly/storage.py`

Owns:

- run outputs
- evidence saving
- clean-sequence export
- sequence manifest maintenance

### `traffic_anomaly/visualization.py`

Presentation only. Draws overlays, boxes, trails, and HUD summaries. It does
not affect anomaly logic.

## Step-By-Step Data Flow

### 1. Video Source

The video source comes from `configs/scene_config.yaml`.

Current options:

- `youtube`
- `local`

The default source in the current config is `youtube`.

### 2. Optional Image Enhancement

If enabled, the frame can be improved before detection:

- CLAHE
  - increases local contrast
- denoising
  - reduces sensor or compression noise
- sharpening
  - makes edges more pronounced

Why this exists:

- CCTV video is often noisy or low-contrast
- better visual quality can improve both detection and crop quality

Current default in the YAML:

- enhancement block exists
- `enhancement.enabled: false`

### 3. Detection And Tracking

Detection uses Ultralytics YOLO.

Tracking uses ByteTrack through YOLO's `track()` API.

Important current settings:

- YOLO weights: `yolo11m.pt`
- confidence threshold: `0.45`
- tracked classes: `Car`, `Bus`, `Truck`
- tracker config: `configs/bytetrack.yaml`

Current ByteTrack settings:

- `track_high_thresh: 0.25`
- `track_low_thresh: 0.1`
- `new_track_thresh: 0.25`
- `track_buffer: 30`
- `match_thresh: 0.8`

Why tracking matters:

- a single frame cannot tell you wrong-way movement
- a single frame cannot tell you if a car has stopped for 2 seconds
- persistent `track_id` is required for temporal features

### 4. Footpoint Extraction

For each box `(x1, y1, x2, y2)`, the code uses:

```text
footpoint_px = ((x1 + x2) / 2, y2)
```

This is the bottom-center of the box.

Why this point is used:

- it is a practical approximation of where the vehicle touches the road
- it is more stable for lane assignment than box center
- lane polygons represent road areas, so the road-contact proxy is a better fit

### 5. Lane Assignment

The footpoint is checked against each configured lane polygon with polygon
containment.

Output:

- `lane_id`
- `lane_category`

Current configured lanes:

- `fast_lane_1`
- `fast_lane_2`
- `emergency_lane_1`
- `emergency_lane_2`

### 6. Homography Projection

The image-space footpoint is projected into a bird's-eye-view space using the
configured homography matrix.

Why this matters:

- image-space motion is distorted by perspective
- a vehicle far from the camera appears to move fewer pixels
- BEV makes motion comparisons more consistent across the scene

Important note:

- this BEV is geometrically more useful than raw pixels
- it is not a fully calibrated physical world model
- speed values are therefore in BEV-coordinate units per second, not km/h

### 7. Crop Extraction

The pipeline crops the detected vehicle box with a small padding margin.

Why this exists:

- GANomaly works on the vehicle crop, not the full frame
- padding keeps some context and avoids overly tight crops

## Per-Track Features

These are the core features produced by `TrackManager`.

### Speed

The code first computes movement in BEV:

```text
movement = footpoint_bev_t - footpoint_bev_(t-1)
```

Then raw speed:

```text
raw_speed = ||movement|| / dt
```

Where:

- `||movement||` is Euclidean distance
- `dt = 1 / fps`

Why this feature matters:

- needed for stopped-vehicle logic
- needed for sudden-stop logic
- helps suppress wrong-way checks on almost stationary vehicles

### Smoothed Speed

Raw speed is noisy because detection boxes jitter. So the code smooths it with
an exponential moving average:

```text
speed_ema = alpha * raw_speed + (1 - alpha) * previous_speed_ema
```

Where:

- `alpha = 2 / (window + 1)`
- current `smoothing_window = 5`
- current `alpha ≈ 0.333`

Why smoothing is used:

- one-frame box jitter should not create false motion spikes
- smoother speed gives more stable stop and drop calculations

### Acceleration

Acceleration is computed from the change in smoothed speed:

```text
raw_acceleration = (speed_ema_t - speed_ema_(t-1)) / dt
```

Then it is also smoothed with EMA.

Why this exists:

- acceleration is useful context for abrupt behavior
- even when not directly thresholded now, it is valuable in logs

### Heading

Heading is the normalized movement direction in BEV:

```text
heading = movement / ||movement||
```

The code keeps a short heading history and averages it before normalizing again.

Why this is important:

- wrong-way detection depends on direction, not just motion magnitude
- averaging reduces zig-zag noise from frame-to-frame box movement

### Heading Alignment

Each lane has a configured direction vector.

The code computes:

```text
heading_alignment = dot(normalized_heading, lane_direction)
```

Interpretation:

- `+1` means same direction as lane flow
- `0` means unclear or sideways
- `-1` means opposite direction

Why this is powerful:

- it turns geometric motion into explicit traffic semantics
- wrong-way detection becomes a simple directional test

### Dwell Frames

`dwell_frames` is the total number of processed frames that a track has existed.

Why it matters:

- prevents early decisions on very young tracks
- helps the pipeline wait for a stable motion estimate

### Lane Violation Frames

This is a consecutive-frame counter.

It increments while the current `lane_id` is in the class-specific
`forbidden_lanes`, otherwise it resets to zero.

Current policy:

- `Car`
  - no forbidden lanes configured
- `Bus`
  - forbidden in `fast_lane_1`, `fast_lane_2`
- `Truck`
  - forbidden in `fast_lane_1`, `fast_lane_2`

### Wrong-Way Frames

This counter increments only when all of these conditions hold:

- vehicle has a lane
- smoothed speed is at least `motion_floor`
- heading alignment is below the wrong-way threshold
- track age is at least `min_track_age_frames`

Current thresholds:

- `motion_floor = 8.0`
- `wrong_way_alignment_threshold = -0.5`
- `min_track_age_frames = 15`

Why this design is good:

- avoids calling nearly stationary objects "wrong-way"
- avoids early unstable direction decisions

### Stopped Frames

This counter increments while:

```text
speed <= stopped_speed_threshold
```

Current threshold:

- `stopped_speed_threshold = 2.5`

### Recent Max Speed

The code keeps a rolling speed window whose length is:

```text
sudden_stop_window_seconds * fps
```

Current window:

- `1.5` seconds

Then:

```text
max_recent_speed = max(recent_speeds)
```

Why this exists:

- sudden stop is not about low speed alone
- it is about low speed after meaningful prior motion

### Speed Drop

The code computes:

```text
speed_drop = max_recent_speed - current_speed
```

clipped at zero.

Why this exists:

- a stopped vehicle and a sudden stop are different concepts
- this feature captures the abruptness of deceleration

### GANomaly Score Aggregation

The raw GANomaly crop score is not used directly as a one-frame signal.

The track state keeps:

- a GANomaly EMA
- a short history window

The final track-level score is:

```text
max(ganomaly_ema, percentile(history, 90))
```

Why this is useful:

- EMA suppresses flicker
- the 90th percentile keeps brief suspicious spikes from being averaged away

## Lane-Level Features

Before evaluating rules, the pipeline summarizes each lane with
`build_lane_snapshots()`.

Per lane it computes:

- `active_count`
- `slow_count`
- `avg_speed`
- `congested`

`slow_count` means speed below the stopped-speed threshold.

Congestion rule:

```text
congested = active_count >= congestion_min_active_tracks
             and slow_fraction >= congestion_slow_ratio
```

Current thresholds:

- `congestion_min_active_tracks = 3`
- `congestion_slow_ratio = 0.65`

Why congestion is modeled:

- traffic jams should not be treated like isolated anomalies
- sudden-stop and stopped-vehicle rules become more realistic

## Rule-Based Anomaly Logic

`RuleEngine.evaluate()` emits `RuleHit` objects with:

- anomaly type
- score
- severity
- explanation

### 1. Lane Violation

Meaning:

- a vehicle stayed long enough in a forbidden lane

Current threshold:

- `lane_violation_seconds = 0.5`

The score is approximately:

```text
min(1.0, duration / threshold)
```

Default severity:

- `warning`

### 2. Wrong-Way

Meaning:

- the vehicle has been moving opposite the lane direction long enough

Current threshold:

- `wrong_way_seconds = 2.0`

The score combines:

- duration term
- direction-opposition term

Current scoring form:

```text
score = 0.5 * duration_term + 0.5 * abs(heading_alignment)
```

clipped to `1.0`.

Default severity:

- `critical`

### 3. Sudden Stop

Meaning:

- the vehicle was moving
- then speed dropped sharply
- and the lane is not behaving like congestion

Current conditions include:

- current speed <= `stopped_speed_threshold`
- recent max speed >= `motion_floor`
- `speed_drop >= sudden_stop_delta`
- lane not congested

Current threshold:

- `sudden_stop_delta = 6.0`

Default severity:

- `critical`

### 4. Stopped Vehicle

Meaning:

- the vehicle remained nearly stationary long enough

Current threshold:

- `stopped_vehicle_seconds = 2.0`

Special behavior:

- stop in an emergency lane is treated as more severe

Default severity:

- `critical` in emergency lane
- `warning` otherwise

## GANomaly Appearance Scoring

GANomaly is used as an appearance anomaly model on vehicle crops.

### Class Groups

The code groups classes into:

- `car`
- `truck_bus`

This is a practical compromise:

- smaller models
- more class-specific normality modeling
- easier training data management

### Crop Preprocessing

Each crop is:

1. resized to `image_size x image_size`
2. converted from BGR to RGB
3. normalized from `[0, 255]` to `[-1, 1]`
4. converted to channel-first tensor format

Current default image size:

- `64`

### Model Shape

The generator is:

- encoder 1
- decoder
- encoder 2

At inference time the important quantities are:

- `latent`
- `latent_reconstructed`

The raw anomaly score is:

```text
mean((latent - latent_reconstructed)^2)
```

Why this works:

- normal samples should reconstruct to a similar latent representation
- abnormal samples tend to create a larger latent mismatch

### Score Normalization

The checkpoint stores a learned threshold estimated from validation normal data.

At runtime:

```text
normalized_score = raw_score / learned_threshold
```

Interpretation:

- near `1.0`
  - close to the learned anomaly threshold
- much greater than `1.0`
  - looks more abnormal relative to the learned normal set

### Training Threshold Estimation

During training, the threshold is set to the:

- 95th percentile of validation scores

This means the model is calibrated against observed normal examples rather than
using a hardcoded raw number.

### Training Data Source

GANomaly training uses approved clean frames from:

- `dataset/sequences/`
- `dataset/sequence_review.csv`

Only reviewed and approved normal sequences are used.

## Fusion Logic

Fusion happens in `TrafficAnomalyPipeline._fuse_hits()`.

Current logic:

- if a rule hit exists, it is kept
- if GANomaly is also high, the rule severity is escalated to `critical`
- if no rule hit exists but GANomaly is high, emit `appearance_anomaly`

Current GANomaly support threshold:

- `ganomaly_high_threshold = 3.5`

So the system is intentionally hybrid:

- rules provide interpretable traffic reasoning
- GANomaly provides supporting or appearance-only evidence

## Event Lifecycle

`EventManager` keeps one active event per:

- `(track_id, anomaly_type)`

It stores:

- start frame
- end frame
- last seen frame
- severity
- best scores
- explanation
- evidence paths

Important behavior:

- repeated hits extend the same event
- stronger hits can upgrade severity
- strongest evidence frame and crop are saved
- events are finalized after a short grace window

Current grace window:

- `event_gap_frames = 5`

## Outputs

### Per-Run Outputs

For each run, the pipeline writes:

- `runs/<run_tag>/tracklets.csv`
- `runs/<run_tag>/events.csv`
- `runs/<run_tag>/frames/`
- `runs/<run_tag>/crops/`
- `runs/<run_tag>/normal_sequences.csv`

### Reusable Dataset Outputs

The pipeline also maintains:

- `dataset/sequences/`
- `dataset/sequence_manifest.csv`
- `dataset/sequence_review.csv`

Only tracks that stay clean are exported as normal sequences.

If a track later becomes anomalous:

- its temporary clean-sequence candidate is deleted

This is important because it protects the GANomaly training set from polluted
examples.

## How Left And Right Traffic Are Represented

The code does not use separate left-side and right-side tracker subsystems.

Instead:

- all vehicles are tracked in one global tracking pass
- lane membership is determined by polygon lookup
- travel direction is determined by the lane's configured direction vector

In the current scene:

- `fast_lane_1` and `emergency_lane_1` share one direction
- `fast_lane_2` and `emergency_lane_2` share the opposite direction

So "left vs right" is represented by geometry, not by separate pipelines.

## Current Configuration Snapshot

The current YAML expresses:

- default source: `youtube`
- YOLO weights: `yolo11m.pt`
- detection classes: `Car`, `Bus`, `Truck`
- lane policy:
  - buses and trucks are forbidden from fast lanes
  - cars have no forbidden lanes configured
- GANomaly checkpoints:
  - `models/ganomaly_car.pt`
  - `models/ganomaly_truck_bus.pt`

Important runtime note:

- the scorer only loads checkpoints that actually exist on disk
- if a checkpoint is missing, that class group silently scores `0.0`
- in the current workspace, `models/ganomaly_car.pt` exists, so car
  appearance scoring is available
- if `models/ganomaly_truck_bus.pt` is missing, bus and truck appearance
  scoring is inactive until that checkpoint is added

## Important Caveats In The Current Branch

### 1. This is not the staged runtime

If you see references to:

- `stage1_detect.py`
- `stage2_track.py`
- `stage3_features.py`
- `stage4_anomaly.py`

those refer to an older or different layout and are not the active runtime
described here.

### 2. Rule enable flags are read but not enforced

The config contains:

- `lane_violation_enabled`
- `wrong_way_enabled`
- `stopped_vehicle_enabled`
- `sudden_stop_enabled`

`RuleEngine.__init__()` reads these values, but `RuleEngine.evaluate()` does
not currently gate rule emission on them. So changing those flags in YAML does
not fully disable the rules at runtime.

### 3. Speed is not real-world speed

The current speed feature is derived from BEV coordinates, not a metric road
calibration.

So:

- it is useful for relative motion logic
- it should not be interpreted as km/h or m/s

### 4. Cars are not currently forbidden from emergency lanes

In the active class policy:

- `Car: forbidden_lanes: []`

So emergency-lane cars are not currently treated as lane violations by this
rule set.

## Short Mental Model

If you want one compact and accurate mental model of the system, it is:

```text
YOLO + ByteTrack gives stable vehicle identities
-> geometry maps each identity into lanes and BEV
-> TrackManager converts identity history into behavior features
-> RuleEngine checks explicit traffic violations
-> GANomaly scores visual abnormality
-> the pipeline fuses both signals
-> EventManager and RunArtifacts persist alerts and clean training sequences
```
