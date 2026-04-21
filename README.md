# Traffic Anomaly Detection

This project runs YOLO + tracking + rule-based traffic anomaly detection with GANomaly support scoring.

## Video Source Selection

The scene config now keeps both a YouTube source and a local MP4 source:

- `youtube`: configured in [configs/scene_config.yaml](configs/scene_config.yaml)
- `local`: currently set to `.video/video.mp4.mp4`

The default source mode is also configured in `configs/scene_config.yaml`:

```yaml
video:
  default: youtube
  youtube: https://www.youtube.com/watch?v=wWSSUfL2LpE
  local: .video/video.mp4.mp4
```

## Usage

Run with the default source from the config:

```powershell
python main.py --batch
```

Run with the configured YouTube source:

```powershell
python main.py --batch --source-mode youtube
```

Run with the configured local MP4:

```powershell
python main.py --batch --source-mode local
```

Override with any custom source path or URL:

```powershell
python main.py --batch --source ".video\\video.mp4.mp4"
```

`--source` has higher priority than `--source-mode`.

## Notes

- If you rename the local file, update `video.local` in `configs/scene_config.yaml`.
- For final evaluation runs, prefer the local MP4 so the inference is reproducible and does not depend on the YouTube stream.

## Model Testing

Use the ground-truth annotations to run a full model test in one command:

```powershell
python scripts/test_model.py --source-mode local --tracker-config configs/ocsort.yaml
```

This will:
- run the pipeline in batch mode
- compare `runs/<run>/events.csv` against `dataset/ground_truth_events.csv`
- save evaluation files under `runs/<run>/ground_truth_eval/`
- emit a `PASS` or `FAIL` verdict using precision, recall, and F1 thresholds

To test an existing run without rerunning inference:

```powershell
python scripts/test_model.py --run 20260421_171727
```

## Appearance Models: GANomaly and VAE

Use `dataset/appearance_ground_truth.csv` when you want to compare `GANomaly` and `VAE` as appearance-anomaly models. This benchmark is separate from `dataset/ground_truth_events.csv`, which is still the correct test set for the full rule-based traffic pipeline.

### Step 1: Prepare normal training sequences

The trainers only learn from sequences marked `approved` in `dataset/sequence_review.csv`. Review and approve good normal sequences first:

```powershell
python scripts/review_sequences.py --dataset-dir dataset --sequence-id <SEQ_ID> --status approved
```

If you already checked the whole file and want a bulk shortcut:

```powershell
python scripts/review_sequences.py --dataset-dir dataset --approve-all-pending
```

### Step 2: Build the appearance ground truth

Bootstrap the crop-level labeling sheet:

```powershell
python scripts/bootstrap_appearance_ground_truth.py
```

This creates or refreshes `dataset/appearance_ground_truth.csv` with:
- `normal` crops sampled from approved normal sequences
- unlabeled event crops from previous `runs/*/events.csv`

Label the `label` column with:
- `normal`
- `appearance_anomaly`

### Step 3: Train GANomaly

Train one checkpoint per class group:

```powershell
python scripts/train_ganomaly.py --group car
python scripts/train_ganomaly.py --group bus
python scripts/train_ganomaly.py --group truck
```

Optional flags:
- `--epochs`
- `--batch-size`
- `--lr`
- `--device cpu|cuda`
- `--output <checkpoint_path>`

### Step 4: Train VAE

Train the VAE on the same approved normal-sequence source:

```powershell
python scripts/train_vae.py --group car
python scripts/train_vae.py --group bus
python scripts/train_vae.py --group truck
```

Use the same optional flags as the GAN trainer when you need to tune training.

### Step 5: Run one model by itself

Evaluate a single appearance model against `dataset/appearance_ground_truth.csv`:

```powershell
python scripts/test_model.py --mode appearance --appearance-model ganomaly
python scripts/test_model.py --mode appearance --appearance-model vae
```

This writes per-model reports under `reports/appearance_test/`.

Important: the live `main.py` pipeline currently uses `GANomaly` for appearance scoring. The `VAE` path is implemented as an offline benchmark baseline, not yet as the default runtime scorer.

### Step 6: Compare GANomaly vs VAE directly

Run both on the same labeled crop set:

```powershell
python scripts/compare_appearance_models.py
```

Outputs are written under `reports/appearance_eval/` and include:
- per-model prediction CSVs
- per-model markdown reports
- one combined comparison summary

### Differences, Pros, and Cons

`GANomaly`
- What it is: an adversarial reconstruction model with latent consistency.
- Pros: often better at separating subtle visual anomalies; usually sharper reconstructions.
- Cons: harder to train, more sensitive to checkpoint quality, and less stable than a VAE.

`VAE`
- What it is: a probabilistic autoencoder trained with reconstruction loss plus KL regularization.
- Pros: simpler, more stable to train, easier to debug and reproduce.
- Cons: usually blurrier reconstructions and weaker sensitivity to fine-grained texture anomalies.

Practical guidance:
- Choose `GANomaly` when detection quality is the priority and you can afford more training instability.
- Choose `VAE` when you want a cleaner, easier baseline and more predictable experimentation.
- Keep using `dataset/ground_truth_events.csv` for the full end-to-end traffic-system test.
