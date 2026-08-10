import numpy as np
from core.moments import sample_moments

def test_sample_moments_shapes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 5))
    mu, sigma = sample_moments(x, mle=True)
    assert mu.shape == (5,)
    assert sigma.shape == (5, 5)
    assert np.allclose(sigma, sigma.T)
