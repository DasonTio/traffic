import warnings

import torch

from traffic_anomaly.ganomaly import GANomalyScorer


def test_ganomaly_scorer_skips_incompatible_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "bad_ganomaly.pt"
    torch.save(
        {
            "latent_dim": 128,
            "generator_state_dict": {"wrong.weight": torch.zeros(1)},
        },
        checkpoint_path,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scorer = GANomalyScorer({"car": checkpoint_path}, device="cpu")

    assert not scorer.available()
    assert any("Skipping GANomaly checkpoint" in str(item.message) for item in caught)
