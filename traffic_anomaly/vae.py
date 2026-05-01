from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from traffic_anomaly.appearance import (
    AppearanceScore,
    approved_frame_paths,
    class_group_for_name,
    preprocess_crop,
)


class SequenceFrameDataset(Dataset):
    def __init__(self, frame_paths: list[Path], image_size: int = 64):
        self.frame_paths = frame_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = cv2.imread(str(self.frame_paths[idx]))
        if image is None:
            raise RuntimeError(f"Could not read {self.frame_paths[idx]}")
        return preprocess_crop(image, self.image_size)


class VariationalAutoencoder(nn.Module):
    def __init__(self, image_size: int = 64, latent_dim: int = 96):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for the current VAE architecture")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_side = image_size // 16
        self.hidden_channels = 256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flattened_dim = self.hidden_channels * self.hidden_side * self.hidden_side
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flattened_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_channels, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        flat = features.flatten(1)
        return self.fc_mu(flat), self.fc_logvar(flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.fc_decode(z)
        decoded = decoded.view(-1, self.hidden_channels, self.hidden_side, self.hidden_side)
        return self.decoder(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


@dataclass
class TrainingResult:
    checkpoint_path: Path
    threshold: float
    train_frames: int
    val_frames: int


def _per_sample_loss(
    batch: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = torch.mean((reconstruction - batch) ** 2, dim=(1, 2, 3))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    total = recon + beta * kl
    return total, recon, kl


class VAETrainer:
    def __init__(
        self,
        dataset_dir: str | Path,
        class_group: str,
        image_size: int = 64,
        latent_dim: int = 96,
        beta: float = 1.0,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str | None = None,
        workers: int = 4,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.class_group = class_group
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.workers = workers
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def train(self, checkpoint_path: str | Path) -> TrainingResult:
        frame_paths = approved_frame_paths(self.dataset_dir, self.class_group)
        if len(frame_paths) < 8:
            raise RuntimeError(
                f"Need at least 8 approved frames for group '{self.class_group}', found {len(frame_paths)}."
            )

        dataset = SequenceFrameDataset(frame_paths, image_size=self.image_size)
        val_size = max(1, int(round(len(dataset) * 0.1)))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        pin_memory = self.device.type == "cuda"
        persistent_workers = self.workers > 0
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model = VariationalAutoencoder(image_size=self.image_size, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        metrics_history: list[dict[str, float | int | str]] = []

        model.train()
        for epoch in range(self.epochs):
            import time

            epoch_start = time.time()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                batch = batch.to(self.device, non_blocking=True)
                reconstruction, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = _per_sample_loss(batch, reconstruction, mu, logvar, self.beta)
                batch_loss = loss.mean()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += float(batch_loss.item())
                total_recon += float(recon_loss.mean().item())
                total_kl += float(kl_loss.mean().item())
                num_batches += 1
                pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

            duration = time.time() - epoch_start
            mins = int(duration // 60)
            secs = int(duration % 60)
            metrics_history.append(
                {
                    "epoch": epoch + 1,
                    "time_duration": f"{mins:02d}:{secs:02d}",
                    "total_iterations": num_batches,
                    "iterations_per_sec": round(num_batches / duration, 2) if duration > 0 else 0.0,
                    "avg_loss": total_loss / max(num_batches, 1),
                    "avg_recon_loss": total_recon / max(num_batches, 1),
                    "avg_kl_loss": total_kl / max(num_batches, 1),
                }
            )

        threshold = self._estimate_threshold(model, val_loader)
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": self.image_size,
                "latent_dim": self.latent_dim,
                "threshold": threshold,
                "class_group": self.class_group,
                "beta": self.beta,
            },
            checkpoint_path,
        )

        metrics_path = checkpoint_path.with_suffix(".csv")
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "time_duration",
                    "total_iterations",
                    "iterations_per_sec",
                    "avg_loss",
                    "avg_recon_loss",
                    "avg_kl_loss",
                ],
            )
            writer.writeheader()
            writer.writerows(metrics_history)

        return TrainingResult(
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            train_frames=train_size,
            val_frames=val_size,
        )

    def _estimate_threshold(self, model: VariationalAutoencoder, data_loader: DataLoader) -> float:
        model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device, non_blocking=True)
                reconstruction, mu, logvar = model(batch)
                total, _, _ = _per_sample_loss(batch, reconstruction, mu, logvar, self.beta)
                scores.extend(total.detach().cpu().tolist())
        model.train()
        if not scores:
            return 0.02
        return float(np.percentile(scores, 95))


class VAEScorer:
    def __init__(
        self,
        checkpoint_paths: dict[str, Path],
        image_size: int = 64,
        default_threshold: float = 0.02,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.default_threshold = default_threshold
        self.models: dict[str, VariationalAutoencoder] = {}
        self.thresholds: dict[str, float] = {}
        self.betas: dict[str, float] = {}

        for class_group, checkpoint_path in checkpoint_paths.items():
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                continue
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                latent_dim = int(checkpoint.get("latent_dim", 96))
                image_size = int(checkpoint.get("image_size", self.image_size))
                state_dict = checkpoint.get("model_state_dict")
                if not isinstance(state_dict, dict):
                    raise KeyError("model_state_dict")
                model = VariationalAutoencoder(image_size=image_size, latent_dim=latent_dim).to(self.device)
                model.load_state_dict(state_dict)
            except Exception as exc:
                reason = "incompatible or invalid checkpoint format"
                if not isinstance(exc, (RuntimeError, KeyError, ValueError, TypeError)):
                    reason = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"Skipping VAE checkpoint for '{class_group}' at {checkpoint_path}: {reason}",
                    RuntimeWarning,
                )
                continue
            model.eval()
            self.models[class_group] = model
            self.thresholds[class_group] = float(checkpoint.get("threshold", self.default_threshold))
            self.betas[class_group] = float(checkpoint.get("beta", 1.0))
            self.image_size = image_size

    def available(self) -> bool:
        return bool(self.models)

    def score_crop(self, crop: np.ndarray, class_name: str) -> float:
        return self.score_crop_details(crop, class_name).normalized_score

    def score_crop_details(self, crop: np.ndarray, class_name: str) -> AppearanceScore:
        class_group = class_group_for_name(class_name)
        threshold = max(self.thresholds.get(class_group, self.default_threshold), 1e-6)
        if crop is None or crop.size == 0:
            return AppearanceScore(
                model_name="vae",
                class_name=class_name,
                class_group=class_group,
                raw_score=0.0,
                threshold=threshold,
                normalized_score=0.0,
            )
        model = self.models.get(class_group)
        if model is None:
            return AppearanceScore(
                model_name="vae",
                class_name=class_name,
                class_group=class_group,
                raw_score=0.0,
                threshold=threshold,
                normalized_score=0.0,
            )
        tensor = preprocess_crop(crop, self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstruction, mu, logvar = model(tensor)
            total, _, _ = _per_sample_loss(tensor, reconstruction, mu, logvar, self.betas.get(class_group, 1.0))
            raw_score = float(total.item())
        return AppearanceScore(
            model_name="vae",
            class_name=class_name,
            class_group=class_group,
            raw_score=raw_score,
            threshold=threshold,
            normalized_score=float(raw_score / threshold),
        )
