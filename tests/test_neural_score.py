import numpy as np

from core.neural_score import (
    train_score_model,
    generate_neural_reverse_diffusion_samples,
)


def test_neural_score_training_and_sampling():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(40, 3)) * 0.02

    result = train_score_model(
        x,
        horizon=0.2,
        beta=1.0,
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        time_dim=8,
        n_hidden_layers=1,
        seed=1,
        device="cpu",
    )

    assert len(result.losses) == 2

    samples = generate_neural_reverse_diffusion_samples(
        result,
        m=5,
        horizon=0.2,
        beta=1.0,
        n_steps=3,
        seed=2,
    )

    assert samples.shape == (5, 3)
    assert np.isfinite(samples).all()
