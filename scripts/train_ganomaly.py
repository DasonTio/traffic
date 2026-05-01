from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.appearance import APPEARANCE_CLASS_GROUPS
from traffic_anomaly.config import SceneConfig
from traffic_anomaly.ganomaly import GANomalyTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GANomaly checkpoint from approved traffic sequences.")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Path to the scene config.")
    parser.add_argument("--group", required=True, choices=list(APPEARANCE_CLASS_GROUPS), help="Class group to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of background threads for DataLoader.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument("--output", default=None, help="Optional checkpoint output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = SceneConfig.load(args.config)
    output = args.output or str(scene.ganomaly_checkpoints[args.group])
    trainer = GANomalyTrainer(
        dataset_dir=scene.dataset_dir,
        class_group=args.group,
        image_size=int(scene.ganomaly_settings.get("image_size", 64)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        workers=args.workers,
    )
    result = trainer.train(Path(output))
    print(
        f"Saved {args.group} GANomaly checkpoint to {result.checkpoint_path} "
        f"(threshold={result.threshold:.6f}, train={result.train_frames}, val={result.val_frames})"
    )


if __name__ == "__main__":
    main()
