from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.appearance_eval import (
    AppearanceEvaluationError,
    evaluate_appearance_model,
    write_comparison_outputs,
)
from traffic_anomaly.config import SceneConfig
from traffic_anomaly.ganomaly import GANomalyScorer
from traffic_anomaly.vae import VAEScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GANomaly and VAE on labeled appearance ground truth.")
    parser.add_argument("--config", default="configs/scene_config.yaml", help="Scene config path.")
    parser.add_argument(
        "--gt",
        default="dataset/appearance_ground_truth.csv",
        help="Appearance ground truth CSV with 'normal' and 'appearance_anomaly' labels.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional report directory. Defaults to reports/appearance_eval/<timestamp>.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    return parser.parse_args()


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("reports") / "appearance_eval" / timestamp


def main() -> None:
    args = parse_args()
    scene = SceneConfig.load(args.config)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir().resolve()

    scorers = [
        GANomalyScorer(
            scene.ganomaly_checkpoints,
            image_size=int(scene.ganomaly_settings.get("image_size", 64)),
            default_threshold=float(scene.ganomaly_settings.get("default_threshold", 0.02)),
            device=args.device,
        ),
        VAEScorer(
            scene.vae_checkpoints,
            image_size=int(scene.vae_settings.get("image_size", 64)),
            default_threshold=float(scene.vae_settings.get("default_threshold", 0.02)),
            device=args.device,
        ),
    ]

    summaries = []
    for scorer in scorers:
        if not scorer.available():
            continue
        summaries.append(evaluate_appearance_model(scorer, args.gt))

    if not summaries:
        raise AppearanceEvaluationError("No valid GANomaly or VAE checkpoints were available for comparison.")

    write_comparison_outputs(summaries, output_dir)
    print(f"Saved appearance comparison to: {output_dir}")
    for summary in summaries:
        metrics = summary.metrics
        print(
            f"{summary.model_name}: precision={metrics.precision*100:.1f}% "
            f"recall={metrics.recall*100:.1f}% f1={metrics.f1*100:.1f}% "
            f"auroc={metrics.auroc:.3f} auprc={metrics.auprc:.3f}"
        )


if __name__ == "__main__":
    main()
