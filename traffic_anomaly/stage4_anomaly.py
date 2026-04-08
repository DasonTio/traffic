from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .config import AnomalyConfig
from .contracts import CAR_CLASS_GROUP, CAR_CLASS_NAME, ModelHit, TrackFeature


ADVERSARIAL_LOSS_WEIGHT = 1.0
CONTEXTUAL_LOSS_WEIGHT = 50.0
LATENT_LOSS_WEIGHT = 1.0
DEFAULT_THRESHOLD = 1.0


def load_reviewed_crop_paths(
    manifest_path: Path | None = None,
    dataset_root: Path | None = None,
) -> list[Path]:
    if manifest_path is None and dataset_root is None:
        raise ValueError("Either manifest_path or dataset_root must be provided.")

    crop_paths: list[Path] = []
    if manifest_path is not None and manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status", "").strip().lower() != "approved":
                    continue
                if row.get("class_name", CAR_CLASS_NAME).strip() != CAR_CLASS_NAME:
                    continue
                crop_path_value = row.get("crop_path", "").strip()
                if not crop_path_value:
                    continue
                crop_path = Path(crop_path_value)
                if not crop_path.is_absolute():
                    base_dir = dataset_root if dataset_root is not None else manifest_path.parent
                    crop_path = (base_dir / crop_path).resolve()
                else:
                    crop_path = crop_path.resolve()
                if crop_path.exists():
                    crop_paths.append(crop_path)
        if crop_paths or dataset_root is None:
            return sorted(crop_paths)

    if dataset_root is None:
        return []

    approved_dir = dataset_root / "approved"
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        crop_paths.extend(sorted(approved_dir.glob(pattern)))
    return sorted({path.resolve() for path in crop_paths if path.exists()})


def _image_to_tensor(image, image_size: int, channels: int) -> torch.Tensor:
    if channels == 1:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        array = resized[np.newaxis, :, :]
    else:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        array = np.transpose(resized, (2, 0, 1))

    tensor = torch.as_tensor(array, dtype=torch.float32)
    return tensor / 127.5 - 1.0


class _CropDataset(Dataset):
    def __init__(self, crop_paths: list[Path], image_size: int, channels: int):
        self.crop_paths = crop_paths
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return len(self.crop_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        flag = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        image = cv2.imread(str(self.crop_paths[idx]), flag)
        if image is None:
            raise RuntimeError(f"Could not read training crop: {self.crop_paths[idx]}")
        return _image_to_tensor(image, self.image_size, self.channels)


def _build_feature_extractor(channels: int, image_size: int) -> tuple[nn.Sequential, tuple[int, int, int]]:
    if image_size < 16 or image_size % 16 != 0:
        raise ValueError("GANomaly image_size must be a multiple of 16 and at least 16.")

    layers: list[nn.Module] = [
        nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    current_channels = 64
    current_size = image_size // 2

    while current_size > 4:
        next_channels = min(current_channels * 2, 512)
        layers.extend(
            [
                nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        current_channels = next_channels
        current_size //= 2

    return nn.Sequential(*layers), (current_channels, current_size, current_size)


class _Encoder(nn.Module):
    def __init__(self, channels: int, image_size: int, latent_dim: int):
        super().__init__()
        self.features, self.feature_shape = _build_feature_extractor(channels, image_size)
        feature_size = int(np.prod(self.feature_shape))
        self.projection = nn.Linear(feature_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.projection(features.flatten(1))


class _Decoder(nn.Module):
    def __init__(self, channels: int, image_size: int, latent_dim: int, feature_shape: tuple[int, int, int]):
        super().__init__()
        feature_channels, feature_height, feature_width = feature_shape
        self.feature_shape = feature_shape
        self.expand = nn.Linear(latent_dim, feature_channels * feature_height * feature_width)

        upsample_layers: list[nn.Module] = []
        current_channels = feature_channels
        current_size = feature_height
        while current_size < image_size // 2:
            next_channels = max(current_channels // 2, 64)
            upsample_layers.extend(
                [
                    nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = next_channels
            current_size *= 2

        upsample_layers.extend(
            [
                nn.ConvTranspose2d(current_channels, channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*upsample_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(latent)
        expanded = expanded.view(latent.size(0), *self.feature_shape)
        return self.decoder(expanded)


class _Discriminator(nn.Module):
    def __init__(self, channels: int, image_size: int):
        super().__init__()
        self.features, self.feature_shape = _build_feature_extractor(channels, image_size)
        feature_size = int(np.prod(self.feature_shape))
        self.classifier = nn.Linear(feature_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x).flatten(1)
        logits = self.classifier(features)
        return logits, features


class GANomalyModel(nn.Module):
    def __init__(self, channels: int = 3, image_size: int = 64, latent_dim: int = 128):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.encoder_in = _Encoder(channels, image_size, latent_dim)
        self.decoder = _Decoder(channels, image_size, latent_dim, self.encoder_in.feature_shape)
        self.encoder_out = _Encoder(channels, image_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_in = self.encoder_in(x)
        reconstructed = self.decoder(latent_in)
        latent_out = self.encoder_out(reconstructed)
        return latent_in, reconstructed, latent_out


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    threshold: float
    train_samples: int
    val_samples: int


class GANomalyTrainer:
    def __init__(
        self,
        config: AnomalyConfig,
        manifest_path: Path | None = None,
        dataset_root: Path | None = None,
        device: str | None = None,
    ):
        self.config = config
        self.manifest_path = manifest_path
        self.dataset_root = dataset_root or config.dataset_root
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def train(self, checkpoint_path: Path | None = None) -> TrainingResult:
        crop_paths = load_reviewed_crop_paths(self.manifest_path, self.dataset_root)
        if len(crop_paths) < 8:
            raise RuntimeError(f"Need at least 8 approved normal car crops, found {len(crop_paths)}.")

        dataset = _CropDataset(crop_paths, self.config.image_size, self.config.channels)
        val_size = max(1, int(round(len(dataset) * 0.1)))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

        generator = GANomalyModel(
            channels=self.config.channels,
            image_size=self.config.image_size,
            latent_dim=self.config.latent_dim,
        ).to(self.device)
        discriminator = _Discriminator(
            channels=self.config.channels,
            image_size=self.config.image_size,
        ).to(self.device)

        optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        adversarial_criterion = nn.BCEWithLogitsLoss()
        contextual_criterion = nn.L1Loss()
        latent_criterion = nn.MSELoss()

        for epoch in range(self.config.epochs):
            generator.train()
            discriminator.train()
            running_g = 0.0
            running_d = 0.0
            batches = 0

            for batch in train_loader:
                real = batch.to(self.device)
                valid = torch.ones((real.size(0), 1), device=self.device)
                fake_target = torch.zeros((real.size(0), 1), device=self.device)

                latent_in, reconstructed, latent_out = generator(real)

                optimizer_d.zero_grad()
                real_logits, _ = discriminator(real)
                fake_logits, _ = discriminator(reconstructed.detach())
                d_loss = 0.5 * (
                    adversarial_criterion(real_logits, valid) + adversarial_criterion(fake_logits, fake_target)
                )
                d_loss.backward()
                optimizer_d.step()

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)

                optimizer_g.zero_grad()
                _, real_features = discriminator(real)
                _, fake_features = discriminator(reconstructed)
                adversarial_loss = latent_criterion(fake_features, real_features.detach())
                contextual_loss = contextual_criterion(reconstructed, real)
                latent_loss = latent_criterion(latent_out, latent_in.detach())
                g_loss = (
                    ADVERSARIAL_LOSS_WEIGHT * adversarial_loss
                    + CONTEXTUAL_LOSS_WEIGHT * contextual_loss
                    + LATENT_LOSS_WEIGHT * latent_loss
                )
                g_loss.backward()
                optimizer_g.step()

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

                running_g += float(g_loss.item())
                running_d += float(d_loss.item())
                batches += 1

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"g_loss {running_g / max(batches, 1):.6f} - d_loss {running_d / max(batches, 1):.6f}"
            )

        threshold = self._estimate_threshold(generator, val_loader)
        checkpoint_path = (checkpoint_path or self.config.checkpoint_path or Path("models/ganomaly_car.pt")).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "channels": self.config.channels,
                "image_size": self.config.image_size,
                "latent_dim": self.config.latent_dim,
                "threshold": threshold,
                "class_group": CAR_CLASS_GROUP,
                "ema_alpha": self.config.ema_alpha,
                "aggregation_window": self.config.aggregation_window,
            },
            checkpoint_path,
        )
        return TrainingResult(
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            train_samples=train_size,
            val_samples=val_size,
        )

    def _estimate_threshold(self, generator: GANomalyModel, loader: DataLoader) -> float:
        scores = self._collect_scores(generator, loader)
        if not scores:
            return DEFAULT_THRESHOLD
        return float(np.percentile(scores, self.config.threshold_percentile))

    def _collect_scores(self, generator: GANomalyModel, loader: DataLoader) -> list[float]:
        generator.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch in loader:
                real = batch.to(self.device)
                latent_in, _, latent_out = generator(real)
                batch_scores = torch.mean((latent_in - latent_out) ** 2, dim=1)
                scores.extend(batch_scores.detach().cpu().tolist())
        return scores


@dataclass
class _LoadedModel:
    generator: GANomalyModel
    threshold: float


@dataclass
class _TrackScoreState:
    history: deque[float]
    ema: float = 0.0
    last_frame_idx: int = 0


class GANomalyScorer:
    def __init__(self, config: AnomalyConfig, device: str | None = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: _LoadedModel | None = None
        self.states: dict[int, _TrackScoreState] = {}

        if not config.enabled or config.checkpoint_path is None or not config.checkpoint_path.exists():
            return

        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        generator = GANomalyModel(
            channels=int(checkpoint.get("channels", config.channels)),
            image_size=int(checkpoint.get("image_size", config.image_size)),
            latent_dim=int(checkpoint.get("latent_dim", config.latent_dim)),
        ).to(self.device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()
        self.model = _LoadedModel(
            generator=generator,
            threshold=float(checkpoint.get("threshold", DEFAULT_THRESHOLD)),
        )

    def available(self) -> bool:
        return self.model is not None

    def score_frame(self, frame, features: list[TrackFeature]) -> dict[int, ModelHit]:
        hits: dict[int, ModelHit] = {}
        if self.model is None:
            return hits

        latest_frame_idx = 0
        for feature in features:
            latest_frame_idx = max(latest_frame_idx, feature.frame_idx)
            if feature.class_name != CAR_CLASS_NAME:
                continue

            crop = self._crop_region(frame, feature.bbox)
            if crop is None:
                continue

            raw_score = self._score_crop(crop)
            state = self.states.get(feature.track_id)
            if state is None:
                state = _TrackScoreState(history=deque(maxlen=self.config.aggregation_window))
                self.states[feature.track_id] = state

            state.history.append(raw_score)
            state.ema = raw_score if state.ema == 0.0 else self.config.ema_alpha * raw_score + (1.0 - self.config.ema_alpha) * state.ema
            state.last_frame_idx = feature.frame_idx

            history_score = float(np.percentile(list(state.history), 90)) if state.history else 0.0
            aggregated = max(state.ema, history_score)
            normalized = aggregated / max(self.model.threshold, 1e-6)
            hits[feature.track_id] = ModelHit(
                frame_idx=feature.frame_idx,
                track_id=feature.track_id,
                class_name=feature.class_name,
                class_group=feature.class_group,
                score=float(normalized),
                threshold=1.0,
                is_anomalous=normalized >= 1.0,
            )

        self._expire_states(latest_frame_idx)
        return hits

    def _score_crop(self, crop) -> float:
        tensor = _image_to_tensor(crop, self.config.image_size, self.config.channels).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent_in, _, latent_out = self.model.generator(tensor)
            score = torch.mean((latent_in - latent_out) ** 2).item()
        return float(score)

    def _crop_region(self, frame, bbox: tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        pad = self.config.crop_padding
        height, width = frame.shape[:2]
        crop = frame[
            max(0, y1 - pad) : min(height, y2 + pad),
            max(0, x1 - pad) : min(width, x2 + pad),
        ]
        if crop.size == 0:
            return None
        crop_height, crop_width = crop.shape[:2]
        if crop_width < self.config.min_crop_width or crop_height < self.config.min_crop_height:
            return None
        return crop

    def _expire_states(self, latest_frame_idx: int) -> None:
        if latest_frame_idx <= 0:
            return
        stale_after = max(self.config.aggregation_window, 1) * 2
        stale_ids = [
            track_id
            for track_id, state in self.states.items()
            if latest_frame_idx - state.last_frame_idx > stale_after
        ]
        for track_id in stale_ids:
            self.states.pop(track_id, None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train car-only GANomaly from reviewed crop data.")
    parser.add_argument("--manifest", default=None, help="Optional review_manifest.csv path.")
    parser.add_argument("--dataset-root", default="dataset/ganomaly", help="GANomaly dataset root.")
    parser.add_argument("--output", default="models/ganomaly_car.pt", help="Checkpoint output path.")
    parser.add_argument("--image-size", type=int, default=64, help="Square GANomaly input size.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension size.")
    parser.add_argument("--channels", type=int, default=3, help="Input channels, usually 3.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="Adam learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--threshold-percentile", type=float, default=95.0, help="Validation percentile for threshold.")
    parser.add_argument("--device", default=None, help="Optional torch device override.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = AnomalyConfig(
        enabled=True,
        checkpoint_path=Path(args.output).resolve(),
        image_size=args.image_size,
        crop_padding=15,
        latent_dim=args.latent_dim,
        channels=args.channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        threshold_percentile=args.threshold_percentile,
        ema_alpha=0.3,
        aggregation_window=20,
        dataset_root=Path(args.dataset_root).resolve(),
        save_training_candidates=True,
        min_crop_width=24,
        min_crop_height=24,
    )
    trainer = GANomalyTrainer(
        config=config,
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
        dataset_root=Path(args.dataset_root).resolve(),
        device=args.device,
    )
    result = trainer.train(Path(args.output).resolve())
    print(
        f"Saved GANomaly checkpoint to {result.checkpoint_path} "
        f"(threshold={result.threshold:.6f}, train={result.train_samples}, val={result.val_samples})"
    )


if __name__ == "__main__":
    main()
