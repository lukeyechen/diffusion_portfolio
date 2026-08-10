import numpy as np
from core.diffusion import diffusion_augmented_moments, generate_reverse_diffusion_samples

def test_diffusion_sample_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 3))
    fake = generate_reverse_diffusion_samples(x, m=25, horizon=1.0, beta=1.0, n_steps=20, seed=1)
    assert fake.shape == (25, 3)

def test_augmented_moment_shapes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    mu, sigma, fake, combined = diffusion_augmented_moments(
        x, m=20, horizon=1.0, beta=1.0, n_steps=20, seed=2
    )
    assert mu.shape == (4,)
    assert sigma.shape == (4, 4)
    assert fake.shape == (20, 4)
    assert combined.shape == (100, 4)
