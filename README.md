# Traffic Anomaly Detection

A real-time traffic anomaly detection system built on YOLOv11 + multi-object tracking + rule-based violation logic, with optional appearance-anomaly scoring via **GANomaly** and **VAE** models.

---

## Overview

The pipeline processes a live YouTube CCTV stream (or a local MP4) and detects:
- **Lane violations** — trucks/buses entering fast lanes
- **Stopped vehicles** — vehicles halting in or near the emergency lane
- **Wrong-way driving** — vehicles with heading misaligned to lane direction

Detected events are saved under `runs/<timestamp>/` alongside evaluation reports.

---

## Requirements

- Python 3.10+
- macOS / Linux (Windows works too — replace `/` with `\` in paths)

---

## Quick Start

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run detection (live YouTube stream)

```bash
python main.py --batch --source-mode youtube
```

Run on the local MP4 instead (faster / reproducible):

```bash
python main.py --batch --source-mode local
```

Override the source for a single run:

```bash
python main.py --batch --source "https://www.youtube.com/watch?v=wWSSUfL2LpE"
```

`--source` overrides `--source-mode`. `--batch` disables the OpenCV preview window.

---

## End-to-End Training & Testing

Follow these steps **in order** to train both appearance-anomaly models and evaluate them.

---

### Step 1 — Approve training sequences

The appearance models train only on sequences marked `approved` in `dataset/sequence_review.csv`.
Run the helper script to label sequences automatically:
- Sequences overlapping a GT anomaly window → `rejected` (kept out of training)
- Everything else → `approved` (clean normal-vehicle crops)

```bash
python scripts/approve_all_sequences.py
```

You will see a per-class summary:

```
Class          Approved   Rejected
----------------------------------
Bus                  20          0
Car                 245          0
Truck               107          0
```

Use `--dry-run` to preview changes without writing any file.

---

### Step 2 — Train GANomaly (one checkpoint per vehicle class)

```bash
python scripts/train_ganomaly.py --group car   --epochs 20
python scripts/train_ganomaly.py --group bus   --epochs 20
python scripts/train_ganomaly.py --group truck --epochs 20
```

Checkpoints are saved to the paths set in `configs/scene_config.yaml`:

| Class | Output path |
|-------|-------------|
| car   | `models/ganomaly_car.pt` |
| bus   | `models/ganomaly_bus.pt` |
| truck | `models/ganomaly_truck.pt` |

A sibling `.csv` file with per-epoch metrics is saved alongside each checkpoint.

**Optional flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 10 | Number of training epochs |
| `--batch-size N` | 32 | Mini-batch size |
| `--lr F` | 2e-4 | Learning rate |
| `--workers N` | 4 | DataLoader worker threads (set `0` on Windows) |
| `--device cpu\|cuda` | auto | Force CPU or GPU |
| `--output PATH` | from config | Override checkpoint save path |

---

### Step 3 — Train VAE (same class groups)

```bash
python scripts/train_vae.py --group car   --epochs 20
python scripts/train_vae.py --group bus   --epochs 20
python scripts/train_vae.py --group truck --epochs 20
```

VAE checkpoints go to `models/vae_{car,bus,truck}.pt`.

**Additional VAE flag**

| Flag | Default | Description |
|------|---------|-------------|
| `--lr F` | 1e-3 | Learning rate (higher than GANomaly's default) |

All other flags are the same as GANomaly.

---

### Step 4 — Run inference on the YouTube stream

Collect a fresh run of predicted events:

```bash
python main.py --batch --source-mode youtube --max-frames 5000
```

`--max-frames 5000` caps at ≈ 2 min 46 s of 30 fps video — enough to cover multiple GT anomaly windows while keeping the run fast. The run is saved under `runs/<timestamp>/`.

---

### Step 5 — Seed the appearance ground truth

After inference, populate `dataset/appearance_ground_truth.csv`:

```bash
python scripts/seed_appearance_ground_truth.py
```

This writes two label classes:
- `normal` — one representative frame from each approved sequence
- `appearance_anomaly` — event crop images saved by the pipeline in `runs/*/crops/`

Use `--dry-run` to check counts before writing.

---

### Step 6 — Full evaluation (system + both models)

Run everything in one shot:

```bash
python scripts/run_full_evaluation.py
```

This produces a report directory under `reports/<timestamp>_full_evaluation/` containing:

| File | Contents |
|------|----------|
| `full_evaluation_report.md` | System Precision / Recall / F1 + GANomaly vs VAE comparison table |
| `full_evaluation_summary.json` | Machine-readable metrics payload |
| `ganomaly/ganomaly_report.md` | GANomaly AUROC / AUPRC per class group |
| `vae/vae_report.md` | VAE AUROC / AUPRC per class group |
| `comparison_report.md` | Side-by-side model comparison table |

**Useful flags**

| Flag | Description |
|------|-------------|
| `--run 20260421_171727` | Evaluate a specific existing run (skips inference) |
| `--skip-system` | Skip system-level GT evaluation, run appearance only |
| `--skip-appearance` | Skip appearance model evaluation |
| `--report-dir PATH` | Override the output directory |

---

### Step 7 — Evaluate a single model (optional)

Score one model against the appearance ground truth:

```bash
python scripts/test_model.py --mode appearance --appearance-model ganomaly
python scripts/test_model.py --mode appearance --appearance-model vae
```

Reports are written under `reports/appearance_test/<timestamp>_<model>/`.

Evaluate system detection only (against `dataset/ground_truth_events.csv`):

```bash
python scripts/test_model.py --source-mode local
```

Or skip inference and evaluate an existing run:

```bash
python scripts/test_model.py --run 20260421_171727
```

---

## Video Source Configuration

Edit `configs/scene_config.yaml` to change sources or resolution:

```yaml
video:
  default: youtube
  youtube: https://www.youtube.com/watch?v=wWSSUfL2LpE
  local: .video/video.mp4.mp4
  resolution: 360p
  fps: 30.0
```

`--source` on the CLI overrides the config for a single run.

---

## Running the Tests

```bash
pytest -q
```

---

## Project Layout

```
traffic/
├── main.py                        # CLI entry point
├── configs/
│   └── scene_config.yaml          # All tuneable parameters
├── traffic_anomaly/               # Runtime package
│   ├── pipeline.py                # Main inference orchestrator
│   ├── ganomaly.py                # GANomaly model + trainer + scorer
│   ├── vae.py                     # VAE model + trainer + scorer
│   ├── rules.py                   # Violation detection rules
│   ├── events.py                  # Event lifecycle management
│   └── ...
├── scripts/
│   ├── approve_all_sequences.py   # GT-aware sequence labeler (Step 1)
│   ├── train_ganomaly.py          # GANomaly trainer CLI (Step 2)
│   ├── train_vae.py               # VAE trainer CLI (Step 3)
│   ├── seed_appearance_ground_truth.py  # Appearance GT builder (Step 5)
│   ├── run_full_evaluation.py     # Full evaluation orchestrator (Step 6)
│   ├── test_model.py              # Single-model / system evaluator (Step 7)
│   └── evaluate_ground_truth.py  # System-level GT evaluation
├── dataset/
│   ├── ground_truth_events.csv    # 128 labeled traffic anomaly events
│   ├── appearance_ground_truth.csv # Crop-level normal/anomaly labels
│   └── sequence_review.csv        # Approved / rejected training sequences
├── models/                        # Saved checkpoints (gitignored)
├── runs/                          # Inference output (gitignored)
└── tests/                         # pytest unit tests
```

---

## Model Comparison: GANomaly vs VAE

| | GANomaly | VAE |
|---|---|---|
| **Architecture** | Encoder → Decoder → Encoder (adversarial) | Encoder → Latent (μ, σ) → Decoder |
| **Anomaly score** | MSE between latent codes | Reconstruction loss + β·KL divergence |
| **Strengths** | Sharper reconstructions; better fine-grained anomaly sensitivity | Simpler, more stable training; easier to reproduce |
| **Weaknesses** | Generator/discriminator balance is fragile | Blurrier reconstructions; weaker on texture anomalies |
| **Best for** | Production, when detection quality matters most | Rapid experimentation and stable baselines |

The live `main.py` pipeline uses GANomaly for the fused anomaly score. VAE is available as an offline benchmark baseline via `scripts/test_model.py --mode appearance --appearance-model vae`.
