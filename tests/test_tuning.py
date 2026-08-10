import numpy as np

from core.tuning import theoretical_horizon


def test_theoretical_horizon_boundary():
    # Construct a case where b*/n is very large, hence constrained T=0.
    mu = np.array([0.001, 0.001])
    sigma = np.eye(2) * 0.01
    T, b, signal = theoretical_horizon(mu, sigma, n_obs=60, beta=1.0)
    assert T == 0.0
    assert signal == 1.0
    assert b > 0


def test_theoretical_horizon_positive_when_feasible():
    # Artificial scale chosen so b*/n < 1.
    mu = np.array([1.0, 0.5])
    sigma = np.eye(2) * 0.1
    T, b, signal = theoretical_horizon(mu, sigma, n_obs=1000, beta=1.0)
    assert T >= 0
    assert 0 < signal <= 1
