import warnings

import numpy as np
import torch

from traffic_anomaly.vae import VAEScorer, VariationalAutoencoder


def test_vae_scorer_skips_incompatible_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "bad_vae.pt"
    torch.save(
        {
            "latent_dim": 16,
            "model_state_dict": {"wrong.weight": torch.zeros(1)},
        },
        checkpoint_path,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scorer = VAEScorer({"car": checkpoint_path}, device="cpu")

    assert not scorer.available()
    assert any("Skipping VAE checkpoint" in str(item.message) for item in caught)


def test_vae_scorer_returns_structured_score(tmp_path):
    checkpoint_path = tmp_path / "vae_car.pt"
    model = VariationalAutoencoder(image_size=64, latent_dim=16)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": 64,
            "latent_dim": 16,
            "threshold": 0.5,
            "class_group": "car",
            "beta": 1.0,
        },
        checkpoint_path,
    )

    scorer = VAEScorer({"car": checkpoint_path}, device="cpu")
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    score = scorer.score_crop_details(crop, "Car")

    assert scorer.available()
    assert score.model_name == "vae"
    assert score.class_group == "car"
    assert score.threshold == 0.5
    assert score.normalized_score >= 0.0
