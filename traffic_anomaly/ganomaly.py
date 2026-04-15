from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def class_group_for_name(class_name: str) -> str:
    return "car" if class_name.lower() == "car" else "truck_bus"


def preprocess_crop(crop: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 127.5 - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _load_review_rows(dataset_dir: Path) -> list[dict[str, str]]:
    review_path = dataset_dir / "sequence_review.csv"
    if not review_path.exists():
        return []
    with review_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def approved_frame_paths(dataset_dir: Path, class_group: str) -> list[Path]:
    frames: list[Path] = []
    for row in _load_review_rows(dataset_dir):
        status = row.get("review_status", "").lower()
        class_name = row.get("class_name", "Unknown")
        if status not in {"approved", "usable", "yes", "true"}:
            continue
        if class_group_for_name(class_name) != class_group:
            continue
        sequence_path = Path(row["sequence_path"])
        frames.extend(sorted(sequence_path.glob("*.jpg")))
    return frames


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


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, latent_dim, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder_1 = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.encoder_2 = Encoder(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder_1(x)
        reconstructed = self.decoder(latent)
        latent_reconstructed = self.encoder_2(reconstructed)
        return reconstructed, latent, latent_reconstructed


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        logits = self.classifier(features).view(-1, 1)
        return logits, features


@dataclass
class TrainingResult:
    checkpoint_path: Path
    threshold: float
    train_frames: int
    val_frames: int


class GANomalyTrainer:
    def __init__(
        self,
        dataset_dir: str | Path,
        class_group: str,
        image_size: int = 64,
        latent_dim: int = 128,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 2e-4,
        device: str | None = None,
        workers: int = 4,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.class_group = class_group
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.workers = workers
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True # Aggressive CNN acceleration

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
        
        # I/O Optimizations
        pin_memory = self.device.type == "cuda"
        persistent_workers = self.workers > 0
        
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.workers, drop_last=False, 
            pin_memory=pin_memory, persistent_workers=persistent_workers
        )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.workers, drop_last=False,
            pin_memory=pin_memory, persistent_workers=persistent_workers
        )

        generator = Generator(latent_dim=self.latent_dim).to(self.device)
        discriminator = Discriminator().to(self.device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        bce = nn.BCELoss()
        l1 = nn.L1Loss()
        mse = nn.MSELoss()

        generator.train()
        discriminator.train()
        
        metrics_history = []
        
        for epoch in range(self.epochs):
            import time
            epoch_start_time = time.time()
            running_g = 0.0
            running_d = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                batch = batch.to(self.device, non_blocking=True)
                batch_size = batch.size(0)
                real_targets = torch.ones((batch_size, 1), device=self.device)
                fake_targets = torch.zeros((batch_size, 1), device=self.device)

                reconstructed, latent, latent_reconstructed = generator(batch)
                real_logits, real_features = discriminator(batch)
                fake_logits_detached, _ = discriminator(reconstructed.detach())
                loss_d = bce(real_logits, real_targets) + bce(fake_logits_detached, fake_targets)
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

                fake_logits, fake_features = discriminator(reconstructed)
                adv_loss = l1(fake_features, real_features.detach())
                con_loss = l1(reconstructed, batch)
                enc_loss = mse(latent, latent_reconstructed)
                loss_g = adv_loss + 50.0 * con_loss + enc_loss
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                running_g += float(loss_g.item())
                running_d += float(loss_d.item())
                num_batches += 1
                
                pbar.set_postfix({
                    "G_loss": f"{loss_g.item():.4f}",
                    "D_loss": f"{loss_d.item():.4f}"
                })

            avg_g = running_g / max(num_batches, 1)
            avg_d = running_d / max(num_batches, 1)
            
            epoch_duration_sec = time.time() - epoch_start_time
            mins = int(epoch_duration_sec // 60)
            secs = int(epoch_duration_sec % 60)
            duration_str = f"{mins:02d}:{secs:02d}"
            
            it_per_sec = num_batches / epoch_duration_sec if epoch_duration_sec > 0 else 0.0
            
            metrics_history.append({
                "epoch": epoch + 1,
                "time_duration": duration_str,
                "total_iterations": num_batches,
                "iterations_per_sec": round(it_per_sec, 2),
                "avg_g_loss": avg_g,
                "avg_d_loss": avg_d
            })
            
            print(
                f"Epoch {epoch + 1}/{self.epochs} Summary - "
                f"Avg G: {avg_g:.4f} | "
                f"Avg D: {avg_d:.4f}"
            )

        threshold = self._estimate_threshold(generator, val_loader)
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "latent_dim": self.latent_dim,
                "image_size": self.image_size,
                "threshold": threshold,
                "class_group": self.class_group,
            },
            checkpoint_path,
        )
        
        import csv
        metrics_path = checkpoint_path.with_suffix('.csv')
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "time_duration", "total_iterations", "iterations_per_sec", "avg_g_loss", "avg_d_loss"
            ])
            writer.writeheader()
            writer.writerows(metrics_history)
            
        return TrainingResult(
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            train_frames=train_size,
            val_frames=val_size,
        )

    def _estimate_threshold(self, generator: Generator, data_loader: DataLoader) -> float:
        generator.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device, non_blocking=True)
                _, latent, latent_reconstructed = generator(batch)
                batch_scores = torch.mean((latent - latent_reconstructed) ** 2, dim=(1, 2, 3))
                scores.extend(batch_scores.detach().cpu().tolist())
        generator.train()
        if not scores:
            return 0.02
        return float(np.percentile(scores, 95))


class GANomalyScorer:
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
        self.models: dict[str, Generator] = {}
        self.thresholds: dict[str, float] = {}

        for class_group, checkpoint_path in checkpoint_paths.items():
            if not Path(checkpoint_path).exists():
                continue
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            latent_dim = int(checkpoint.get("latent_dim", 128))
            model = Generator(latent_dim=latent_dim).to(self.device)
            model.load_state_dict(checkpoint["generator_state_dict"])
            model.eval()
            self.models[class_group] = model
            self.thresholds[class_group] = float(checkpoint.get("threshold", self.default_threshold))
            self.image_size = int(checkpoint.get("image_size", self.image_size))

    def available(self) -> bool:
        return bool(self.models)

    def score_crop(self, crop: np.ndarray, class_name: str) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        class_group = class_group_for_name(class_name)
        model = self.models.get(class_group)
        if model is None:
            return 0.0
        tensor = preprocess_crop(crop, self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, latent, latent_reconstructed = model(tensor)
            raw_score = torch.mean((latent - latent_reconstructed) ** 2).item()
        threshold = max(self.thresholds.get(class_group, self.default_threshold), 1e-6)
        return float(raw_score / threshold)
