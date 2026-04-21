from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.appearance import APPEARANCE_CLASS_GROUPS
from traffic_anomaly.config import SceneConfig
from traffic_anomaly.vae import VAETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE checkpoint from approved traffic sequences.")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Path to the scene config.")
    parser.add_argument("--group", required=True, choices=list(APPEARANCE_CLASS_GROUPS), help="Class group to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of background workers for DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument("--output", default=None, help="Optional checkpoint output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = SceneConfig.load(args.config)
    output = args.output or str(scene.vae_checkpoints[args.group])
    trainer = VAETrainer(
        dataset_dir=scene.dataset_dir,
        class_group=args.group,
        image_size=int(scene.vae_settings.get("image_size", 64)),
        latent_dim=int(scene.vae_settings.get("latent_dim", 96)),
        beta=float(scene.vae_settings.get("beta", 1.0)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        workers=args.workers,
    )
    result = trainer.train(Path(output))
    print(
        f"Saved {args.group} VAE checkpoint to {result.checkpoint_path} "
        f"(threshold={result.threshold:.6f}, train={result.train_frames}, val={result.val_frames})"
    )


if __name__ == "__main__":
    main()
