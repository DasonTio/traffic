<div align="center">

<h1>Traffic Anomaly Detection</h1>

<p>
Fixed-scene traffic CCTV anomaly detection with detector/tracker comparison,
lane-aware rule logic, and appearance-model experiment support.
</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-111F68?style=for-the-badge)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

<p>
  <a href="./detector_tracker_pipeline_report.pdf">Detector + Tracker Report</a>
  |
  <a href="./old_vs_1920x1440_pipeline_report.pdf">Resolution Comparison Report</a>
  |
  <a href="./ARCHITECTURE.md">Architecture Notes</a>
</p>

</div>

## Table Of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [PDF Report Results](#pdf-report-results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training Workflows](#training-workflows)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Traffic Anomaly Detection is a Python computer-vision project for analyzing a
fixed traffic CCTV scene. It detects and tracks vehicles, maps them into
configured lane geometry, computes per-track motion features, applies rule-based
event logic, and can run GANomaly or VAE appearance scoring experiments.

The repository is organized as a reproducible research pipeline: run inference
on a local video or YouTube source, write event outputs under `runs/`, compare
detector/tracker choices, train appearance models, and generate report
artifacts.

## Features

| Area | What The Project Provides |
| --- | --- |
| Video input | Local MP4 and YouTube source support through YAML config and CLI overrides |
| Detection | YOLO11 by default, with RT-DETR available through `--detector rtdetr` |
| Tracking | ByteTrack and OC-Sort workflows through tracker YAML files |
| Scene geometry | Lane polygons, lane categories, direction vectors, and homography projection |
| Rule logic | Lane-violation and wrong-way logic, with optional stopped-vehicle and sudden-stop rules |
| Appearance experiments | GANomaly and VAE training/scoring modules for vehicle crop anomaly experiments |
| Outputs | Event CSVs, tracklets, evidence crops, normal sequence exports, metadata, and PDF/Markdown reports |

## PDF Report Results

This README reports the results from the PDF files currently present in the
project root:

- [`detector_tracker_pipeline_report.pdf`](detector_tracker_pipeline_report.pdf)
- [`old_vs_1920x1440_pipeline_report.pdf`](old_vs_1920x1440_pipeline_report.pdf)

### Detector + Tracker Event-Based Anomaly Report

Source: [`detector_tracker_pipeline_report.pdf`](detector_tracker_pipeline_report.pdf)

Generated from: `detector_tracker_metrics.json`

Appearance models: `none`

Summary from the report:

- Best pipeline by F1: `YOLO11-m + OC-Sort`.
- YOLO11-m outperformed RT-DETR-L by F1 for both tracker choices.
- OC-Sort improved recall for both detectors.
- RT-DETR-L + OC-Sort produced the highest recall but also the highest false-positive count.
- These metrics isolate detector, tracker, and rule-based event logic only.

Ranking by F1:

| Rank | Pipeline | F1 | Precision | Recall | TP | FP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | YOLO11-m + OC-Sort | 3.2% | 1.9% | 8.7% | 11 | 560 |
| 2 | YOLO11-m + ByteTrack | 2.9% | 1.8% | 7.1% | 9 | 491 |
| 3 | RT-DETR-L + OC-Sort | 1.7% | 0.9% | 10.2% | 13 | 1399 |
| 4 | RT-DETR-L + ByteTrack | 1.6% | 1.0% | 4.7% | 6 | 605 |

Detailed metrics:

| Pipeline | Run | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Lane Agree | Class Agree |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11-m + OC-Sort | `20260430_031045` | 127 | 571 | 11 | 560 | 116 | 1.9% | 8.7% | 3.2% | 54.5% | 100.0% |
| YOLO11-m + ByteTrack | `20260430_020640` | 127 | 500 | 9 | 491 | 118 | 1.8% | 7.1% | 2.9% | 66.7% | 100.0% |
| RT-DETR-L + OC-Sort | `20260430_054651` | 127 | 1412 | 13 | 1399 | 114 | 0.9% | 10.2% | 1.7% | 61.5% | 84.6% |
| RT-DETR-L + ByteTrack | `20260430_035924` | 127 | 611 | 6 | 605 | 121 | 1.0% | 4.7% | 1.6% | 50.0% | 100.0% |

The PDF also includes metric charts for F1, precision, recall, and false
positives.

### Original Size vs 1920x1440 Traffic Pipeline Report

Source: [`old_vs_1920x1440_pipeline_report.pdf`](old_vs_1920x1440_pipeline_report.pdf)

Evaluation window: frames `1-139079`

Ground-truth events: `127`

Summary from the report:

- Original-size video performed better overall than 1920x1440 upscaled video.
- Best overall pipeline: `OC-Sort + VAE` on original video.
- Best original-size result: `OC-Sort + VAE`, F1 `3.0%`.
- Best 1920x1440 result: `OC-Sort + VAE`, F1 `1.3%`.
- The largest upscaled-video failure mode was false positives, especially for GANomaly-backed pipelines.

Best result ranking by F1:

| Rank | Pipeline | Source | F1 | Precision | Recall | TP | FP |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | OC-Sort + VAE | Original | 3.0% | 1.8% | 8.7% | 11 | 590 |
| 2 | ByteTrack + VAE | Original | 2.7% | 1.7% | 7.1% | 9 | 520 |
| 3 | OC-Sort + GANomaly | Original | 2.0% | 1.2% | 8.7% | 11 | 938 |
| 4 | ByteTrack + GANomaly | Original | 2.0% | 1.2% | 7.1% | 9 | 755 |
| 5 | OC-Sort + VAE | 1920x1440 | 1.3% | 0.8% | 4.7% | 6 | 757 |
| 6 | ByteTrack + VAE | 1920x1440 | 0.5% | 0.3% | 1.6% | 2 | 641 |
| 7 | OC-Sort + GANomaly | 1920x1440 | 0.1% | 0.0% | 4.7% | 6 | 12144 |
| 8 | ByteTrack + GANomaly | 1920x1440 | 0.0% | 0.0% | 1.6% | 2 | 12291 |

Detailed metrics:

| Pipeline | Source | Run | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Lane Agree | Class Agree |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack + GANomaly | Original | `20260428_131429` | 127 | 764 | 9 | 755 | 118 | 1.2% | 7.1% | 2.0% | 66.7% | 100.0% |
| ByteTrack + GANomaly | 1920x1440 | `20260428_212825` | 127 | 12293 | 2 | 12291 | 125 | 0.0% | 1.6% | 0.0% | 50.0% | 100.0% |
| ByteTrack + VAE | Original | `20260428_151120` | 127 | 529 | 9 | 520 | 118 | 1.7% | 7.1% | 2.7% | 66.7% | 100.0% |
| ByteTrack + VAE | 1920x1440 | `20260429_030727` | 127 | 643 | 2 | 641 | 125 | 0.3% | 1.6% | 0.5% | 50.0% | 100.0% |
| OC-Sort + GANomaly | Original | `20260428_163617` | 127 | 949 | 11 | 938 | 116 | 1.2% | 8.7% | 2.0% | 54.5% | 100.0% |
| OC-Sort + GANomaly | 1920x1440 | `20260429_064324` | 127 | 12150 | 6 | 12144 | 121 | 0.0% | 4.7% | 0.1% | 50.0% | 100.0% |
| OC-Sort + VAE | Original | `20260428_183232` | 127 | 601 | 11 | 590 | 116 | 1.8% | 8.7% | 3.0% | 54.5% | 100.0% |
| OC-Sort + VAE | 1920x1440 | `20260429_103232` | 127 | 763 | 6 | 757 | 121 | 0.8% | 4.7% | 1.3% | 50.0% | 100.0% |

Deltas, calculated as `1920x1440 minus Original`:

| Pipeline | Delta Pred | Delta TP | Delta FP | Delta FN | Delta Precision | Delta Recall | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack + GANomaly | +11529 | -7 | +11536 | +7 | -1.2 pp | -5.5 pp | -2.0 pp |
| ByteTrack + VAE | +114 | -7 | +121 | +7 | -1.4 pp | -5.5 pp | -2.2 pp |
| OC-Sort + GANomaly | +11201 | -5 | +11206 | +5 | -1.1 pp | -3.9 pp | -1.9 pp |
| OC-Sort + VAE | +162 | -5 | +167 | +5 | -1.0 pp | -3.9 pp | -1.7 pp |

Report interpretation:

- Original-size video is the safer input for the current configuration.
- GANomaly on 1920x1440 is especially unstable, with false positives above 12k for both trackers.
- VAE is more conservative than GANomaly on 1920x1440, but still trails the original-size runs.
- If 1920x1440 input must be used, thresholds and geometry should be tuned for that resolution before relying on the metrics.

The PDF also includes visual charts comparing F1 score and false positives by
pipeline and source resolution.

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.10+ |
| Computer vision | OpenCV, NumPy |
| Detectors | Ultralytics YOLO11, RT-DETR |
| Trackers | ByteTrack, OC-Sort tracker configs |
| Deep learning | PyTorch, torchvision, torchaudio |
| Evaluation | CSV/JSON/Markdown/PDF report outputs |
| Testing | pytest |

## Getting Started

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a local-video inference pass without the OpenCV preview window:

```bash
python main.py --batch --source-mode local
```

Run the configured YouTube source:

```bash
python main.py --batch --source-mode youtube
```

Override the video source for one run:

```bash
python main.py --batch --source "/path/to/video.mp4"
```

## Usage

Run YOLO11-m with ByteTrack and GANomaly appearance scoring:

```bash
python main.py \
  --batch \
  --source-mode local \
  --detector yolo \
  --detector-weights yolo11m.pt \
  --tracker-config configs/bytetrack.yaml \
  --appearance-model ganomaly
```

Run RT-DETR-L with OC-Sort and no appearance model:

```bash
python main.py \
  --batch \
  --source-mode local \
  --detector rtdetr \
  --detector-weights rtdetr-l.pt \
  --tracker-config configs/ocsort.yaml \
  --appearance-model none
```

Run a bounded inference pass:

```bash
python main.py --batch --source-mode youtube --max-frames 5000
```

Compare detector and tracker choices:

```bash
python scripts/compare_detectors_trackers.py \
  --source .video/video.mp4.mp4 \
  --max-frames 5000 \
  --frame-end 5000
```

Evaluate an existing run:

```bash
python scripts/evaluate_detections.py --run <run_id>
```

## Training Workflows

Approve normal training sequences:

```bash
python scripts/approve_all_sequences.py
```

Train GANomaly checkpoints:

```bash
python scripts/train_ganomaly.py --group car --epochs 20
python scripts/train_ganomaly.py --group bus --epochs 20
python scripts/train_ganomaly.py --group truck --epochs 20
```

Train VAE checkpoints:

```bash
python scripts/train_vae.py --group car --epochs 20
python scripts/train_vae.py --group bus --epochs 20
python scripts/train_vae.py --group truck --epochs 20
```

Generated model files are local artifacts and should not be committed unless
the project explicitly decides to publish a checkpoint.

## Configuration

Primary settings live in [`configs/scene_config.yaml`](configs/scene_config.yaml).

| Section | Controls |
| --- | --- |
| `video` | Default source mode, YouTube URL, local video path, FPS |
| `model` | Default detector weights, confidence threshold, detected COCO classes |
| `tracking` | Default tracker YAML |
| `classification` | Temporal class-vote smoothing |
| `homography` | Image-to-bird's-eye projection points |
| `lanes` | Lane polygons, labels, categories, and direction vectors |
| `class_lane_policy` | Vehicle classes forbidden in configured lanes |
| `ganomaly` | Image size, aggregation settings, thresholds, checkpoint paths |
| `vae` | Image size, latent dimension, beta, thresholds, checkpoint paths |
| `thresholds` | Rule toggles, event timing, speed thresholds, event gaps |
| `enhancement` | Optional CLAHE, denoising, and sharpening |

Frequently used CLI overrides:

| Flag | Purpose |
| --- | --- |
| `--source-mode local\|youtube` | Select a named source from config |
| `--source <path-or-url>` | Override the configured source |
| `--tracker-config <path>` | Override tracker YAML |
| `--detector yolo\|rtdetr` | Select detector family |
| `--detector-weights <path>` | Override detector weights |
| `--appearance-model ganomaly\|vae\|none` | Select appearance scorer |
| `--max-frames N` | Stop after N frames |
| `--batch` | Disable display for batch runs |

## Project Structure

```text
traffic/
  main.py
  README.md
  ARCHITECTURE.md
  requirements.txt
  detector_tracker_pipeline_report.pdf
  old_vs_1920x1440_pipeline_report.pdf
  configs/
    scene_config.yaml
    bytetrack.yaml
    ocsort.yaml
  traffic_anomaly/
    pipeline.py
    config.py
    geometry.py
    tracklets.py
    rules.py
    events.py
    storage.py
    ganomaly.py
    vae.py
    visualization.py
  scripts/
    approve_all_sequences.py
    compare_detectors_trackers.py
    evaluate_detections.py
    run_full_evaluation.py
    train_ganomaly.py
    train_vae.py
  dataset/
    ground_truth_events.csv
    sequence_review.csv
  reports/
  runs/
  tests/
```

`runs/`, generated datasets, checkpoints, model weights, and local videos are
runtime artifacts. Keep them local unless a report or checkpoint is explicitly
being published.

## Testing

Run the test suite:

```bash
pytest -q
```

The tests cover configuration parsing, rule behavior, tracker helpers,
appearance-model utilities, sequence mining, and evaluation logic.

## Contributing

Keep pull requests focused on one concern. Include:

- The scenario changed.
- Any config, threshold, source, tracker, or model-artifact impact.
- The exact commands used for verification.
- Screenshots only when overlay, labeling, or visual report output changes.

## License

No license file is currently present. Add a license before distributing this
project publicly.
