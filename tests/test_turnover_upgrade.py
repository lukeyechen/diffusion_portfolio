import numpy as np

from core.moments import sample_moments
from core.turnover_upgrade import (
    exact_diffusion_augmented_moments,
    exact_reverse_diffusion_moments,
    nested_exact_horizon,
    partial_rebalance,
    portfolio_turnover,
    solve_mv_turnover_aware,
)


def test_exact_horizon_zero_preserves_sample_moments():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 4))
    mu0, sigma0 = sample_moments(x, mle=True)
    mu, sigma, mu_fake, sigma_fake = exact_diffusion_augmented_moments(
        x, m=500, horizon=0.0, beta=1.0, n_steps=50
    )
    assert np.allclose(mu, mu0, atol=1e-10)
    assert np.allclose(sigma, sigma0, atol=1e-8)
    assert np.allclose(mu_fake, mu0, atol=1e-10)
    assert np.allclose(sigma_fake, sigma0, atol=1e-8)


def test_exact_reverse_moments_are_finite_psd():
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.03, size=(180, 5))
    mu, sigma = exact_reverse_diffusion_moments(x, horizon=0.5, n_steps=40)
    assert mu.shape == (5,)
    assert sigma.shape == (5, 5)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    assert np.min(np.linalg.eigvalsh(sigma)) > 0


def test_turnover_penalty_reduces_trade_size():
    mu = np.array([0.05, 0.04, 0.01, 0.005, 0.0])
    sigma = np.eye(5) * 0.02
    prev = np.ones(5) / 5

    w0 = solve_mv_turnover_aware(
        mu,
        sigma,
        prev,
        turnover_penalty=0.0,
        gamma=3.0,
        mode="Long-only",
        max_long_weight=0.40,
    )
    wp = solve_mv_turnover_aware(
        mu,
        sigma,
        prev,
        turnover_penalty=0.02,
        gamma=3.0,
        mode="Long-only",
        max_long_weight=0.40,
    )
    assert portfolio_turnover(wp, prev) <= portfolio_turnover(w0, prev) + 1e-8


def test_partial_rebalance_scales_turnover():
    prev = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    target = np.array([0.40, 0.30, 0.10, 0.10, 0.10])
    half = partial_rebalance(target, prev, alpha=0.5)
    assert np.isclose(portfolio_turnover(half, prev), 0.5 * portfolio_turnover(target, prev))
    assert np.isclose(half.sum(), 1.0)


def test_nested_exact_horizon_returns_grid_value():
    rng = np.random.default_rng(11)
    x = rng.normal(scale=0.02, size=(120, 5))
    grid = [0.0, 0.1, 0.3]
    t, results = nested_exact_horizon(
        x,
        gamma=3.0,
        m=100,
        candidate_T=grid,
        inner_folds=4,
        validation_size=12,
        min_train_size=50,
        n_steps=20,
        selection_rule="one_se",
    )
    assert t in grid
    assert len(results) == len(grid)
    assert sum(bool(r["selected_by_rule"]) for r in results) == 1
