from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .moments import regularize_covariance


def solve_mv(mu: np.ndarray, sigma: np.ndarray, gamma: float = 3.0) -> np.ndarray:
    """Unconstrained mean-variance solution."""
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    sigma = regularize_covariance(sigma)
    return np.linalg.solve(sigma, np.asarray(mu, dtype=float)) / gamma


def equal_weight(n_assets: int) -> np.ndarray:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")
    return np.ones(n_assets) / n_assets


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    model = LedoitWolf(assume_centered=False).fit(np.asarray(returns, dtype=float))
    return model.covariance_


def _mv_objective(w, mu, sigma, gamma):
    return -(float(w @ mu) - 0.5 * gamma * float(w @ sigma @ w))


def _gross_exposure(w):
    return float(np.abs(w).sum())


def solve_mv_constrained(
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: float = 3.0,
    mode: str = "Long-only",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
) -> np.ndarray:
    """
    Practical constrained mean-variance portfolio.

    Long-only:
        sum(w)=1, 0 <= w_i <= max_long_weight.

    Limited Long-Short:
        sum(w)=1,
        -max_short_weight <= w_i <= max_long_weight,
        sum(|w_i|) <= max_gross_exposure.

    Research / Unconstrained:
        returns the closed-form paper-style solution.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = regularize_covariance(np.asarray(sigma, dtype=float))
    n = mu.size

    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    if mode == "Research / Unconstrained":
        return solve_mv(mu, sigma, gamma)

    if max_long_weight <= 0:
        raise ValueError("max_long_weight must be positive.")

    # Feasibility for fully invested long-only portfolios.
    if mode == "Long-only" and max_long_weight * n < 1.0 - 1e-12:
        raise ValueError(
            f"Long-only problem is infeasible: with {n} assets and "
            f"max weight {max_long_weight:.0%}, the weights cannot sum to 100%. "
            f"Increase the maximum asset weight to at least {1/n:.1%}."
        )

    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if mode == "Long-only":
        bounds = [(0.0, float(max_long_weight))] * n

    elif mode == "Limited Long-Short":
        if max_short_weight < 0:
            raise ValueError("max_short_weight must be nonnegative.")
        if max_gross_exposure < 1.0:
            raise ValueError("max_gross_exposure must be at least 1.0.")
        bounds = [(-float(max_short_weight), float(max_long_weight))] * n
        constraints.append({
            "type": "ineq",
            "fun": lambda w: float(max_gross_exposure) - np.sum(np.abs(w)),
        })
    else:
        raise ValueError(f"Unknown constraint mode: {mode}")

    result = minimize(
        _mv_objective,
        x0=x0,
        args=(mu, sigma, gamma),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        raise ValueError(f"Constrained optimization failed: {result.message}")

    return np.asarray(result.x, dtype=float)


def compute_weights(
    rule: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: float = 3.0,
    returns: np.ndarray | None = None,
    constraint_mode: str = "Research / Unconstrained",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
) -> np.ndarray:
    """
    Compute portfolio weights.

    Equal Weight is unchanged by constraint mode.
    Mean-Variance and Ledoit-Wolf Mean-Variance can use practical constraints.
    """
    key = rule.strip().lower()

    if key in {"equal weight", "equal-weight", "1/n"}:
        return equal_weight(len(mu))

    if key in {"mean-variance", "mean variance", "first plug-in", "1st plug-in"}:
        return solve_mv_constrained(
            mu, sigma, gamma,
            mode=constraint_mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )

    if key in {"ledoit-wolf mean-variance", "ledoit wolf mean-variance", "lw"}:
        if returns is None:
            raise ValueError("returns are required for the Ledoit-Wolf rule.")
        sigma_lw = ledoit_wolf_covariance(returns)
        return solve_mv_constrained(
            mu, sigma_lw, gamma,
            mode=constraint_mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )

    raise ValueError(f"Unsupported portfolio rule: {rule}")
