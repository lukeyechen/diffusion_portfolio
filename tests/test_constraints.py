import numpy as np

from core.portfolio_rules import solve_mv_constrained


def test_long_only_weights():
    mu = np.array([0.01, 0.02, 0.015, 0.005])
    sigma = np.eye(4) * 0.02

    w = solve_mv_constrained(
        mu,
        sigma,
        gamma=3.0,
        mode="Long-only",
        max_long_weight=0.40,
    )

    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.all(w >= -1e-8)
    assert np.all(w <= 0.4000001)


def test_limited_long_short_gross_limit():
    mu = np.array([0.04, 0.02, -0.02, 0.01])
    sigma = np.eye(4) * 0.03

    w = solve_mv_constrained(
        mu,
        sigma,
        gamma=3.0,
        mode="Limited Long-Short",
        max_long_weight=0.70,
        max_short_weight=0.30,
        max_gross_exposure=1.50,
    )

    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert np.abs(w).sum() <= 1.500001
