import numpy as np

from core.tuning import nested_rolling_validation_tuned_horizon, theoretical_horizon


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


def test_nested_rolling_validation_returns_fold_diagnostics():
    rng = np.random.default_rng(123)
    returns = rng.normal(
        loc=np.array([0.01, 0.008]),
        scale=np.array([0.04, 0.03]),
        size=(72, 2),
    )

    best_T, results = nested_rolling_validation_tuned_horizon(
        returns,
        gamma=3.0,
        rule="Mean-Variance",
        m=20,
        beta=1.0,
        n_steps=3,
        candidate_T=[0.0, 0.10],
        inner_folds=3,
        validation_size=10,
        min_train_size=30,
        seed=7,
        constraint_mode="Long-only",
        max_long_weight=0.80,
    )

    assert best_T in {0.0, 0.10}
    assert len(results) == 2
    for result in results:
        assert result["fold_count"] == 3
        assert len(result["fold_CERs"]) == 3
        assert len(result["folds"]) == 3
        assert np.isfinite(result["mean_validation_CER"])
        assert np.isfinite(result["std_validation_CER"])
        assert [fold["train_n"] for fold in result["folds"]] == [42, 52, 62]
        assert all(fold["validation_n"] == 10 for fold in result["folds"])
