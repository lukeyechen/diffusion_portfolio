import numpy as np
from core.neural_score import train_score_model


def test_validation_fields():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(50, 3)).astype(float) * 0.02
    result = train_score_model(
        x,
        horizon=0.2,
        beta=1.0,
        epochs=5,
        batch_size=16,
        hidden_dim=16,
        time_dim=8,
        n_hidden_layers=1,
        validation_fraction=0.2,
        patience=3,
        min_delta=1e-4,
        eval_repeats=1,
        seed=1,
        device="cpu",
    )
    assert len(result.train_losses) >= 1
    assert len(result.train_losses) == len(result.val_losses)
    assert result.best_epoch >= 1
    assert result.zero_score_baseline_train > 0
    assert result.zero_score_baseline_val > 0
