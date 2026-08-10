import numpy as np
from core.portfolio_rules import equal_weight, solve_mv

def test_equal_weight_sums_to_one():
    w = equal_weight(5)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, np.ones(5) / 5)

def test_mv_identity_covariance():
    mu = np.array([0.01, 0.02])
    sigma = np.eye(2)
    w = solve_mv(mu, sigma, gamma=2.0)
    assert np.allclose(w, mu / 2.0)
