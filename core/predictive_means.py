"""Point-in-time, per-asset Gaussian diffusion regression forecasts.

Inputs/outputs are simple holding-period returns in decimals (as in the app).
Internally all features and responses use fixed percentage-point units, without
sample demeaning or whitening. The zero-mean prior's origin is thus preserved.
This is an empirical extension, not a portfolio-dominance theorem.
"""
from __future__ import annotations

import numpy as np

MEAN_MODELS = ("Original diffusion mean", "OLS forecast", "Diffusion forecast")
B_GRID = (0.0, 0.25, 1.0, 4.0)
MIN_PAIRS = 12


def regression_coefficients(x: np.ndarray, y: np.ndarray, a: float):
    """Exact terminal-moment regression; a=0 is OLS, not T=0.

    Singular designs explicitly use least-squares OLS and return a status.
    No covariance regularization is silently attributed to the paper.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) <= x.shape[1] + 2:
        raise ValueError("Regression needs n > p+2 aligned observations.")
    if not np.isfinite(a) or not 0 <= a < 1:
        raise ValueError("Forecast signal a must satisfy 0 <= a < 1.")
    z = np.column_stack([x, y])
    if not np.isfinite(z).all():
        raise ValueError("Regression observations must be finite.")
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / len(z)
    vals, vecs = np.linalg.eigh(cov)
    if vals[0] <= max(1e-12, vals[-1] * 1e-10):
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        return float(coef[0]), coef[1:], "OLS fallback: singular joint covariance"
    c = 1 - a + a * vals
    # Stable equivalent of lambda - a^2 lambda^3/c^2 (avoids cancellation).
    svals = vals * (1 - a) * (1 - a + 2 * a * vals) / c**2
    terminal_cov = (vecs * svals) @ vecs.T
    terminal_mean = vecs @ (((1 - a) / c) * (vecs.T @ mean))
    slope = np.linalg.solve(terminal_cov[:-1, :-1], terminal_cov[:-1, -1])
    intercept = terminal_mean[-1] - terminal_mean[:-1] @ slope
    return float(intercept), slope, "OLS" if a == 0 else "Diffusion"


def regression_dataset(returns: np.ndarray):
    """At origin u: last return, trailing 3-period mean/std -> return u+1."""
    r = np.asarray(returns, float)
    if r.ndim != 2 or r.shape[1] < 1 or len(r) < MIN_PAIRS + 3:
        raise ValueError(f"Forecasting needs at least {MIN_PAIRS + 3} return observations.")
    if not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("Returns must be finite decimal simple returns >= -1.")
    features = np.stack([
        np.stack([r[u], r[u-2:u+1].mean(axis=0), r[u-2:u+1].std(axis=0)], axis=-1)
        for u in range(2, len(r))
    ]) * 100.0
    return features[:-1], r[3:] * 100.0, features[-1]


def _fit_forecast(features, targets, latest, b):
    a = float(b) / len(targets)
    forecasts, statuses = [], []
    for j in range(targets.shape[1]):
        intercept, slope, status = regression_coefficients(features[:, j], targets[:, j], a)
        forecasts.append((intercept + latest[j] @ slope) / 100.0)
        statuses.append(status)
    return np.asarray(forecasts), statuses


def forecast_means(returns: np.ndarray, *, tune: bool = True) -> dict:
    """Tune shared b on at most 12 past one-step origins, then refit all pairs.

    Every validation fit ends before its held-out response; its predictors are
    already observed. b=0 permits selection of OLS. No outer response is used.
    """
    features, targets, latest = regression_dataset(returns)
    n = len(targets)
    scores = []
    if tune:
        start = max(MIN_PAIRS, n - min(12, max(1, n // 5)))
        for b in B_GRID:
            errors = []
            for k in range(start, n):
                pred, _ = _fit_forecast(features[:k], targets[:k], features[k], b)
                errors.append(np.mean((pred - targets[k] / 100.0)**2))
            if errors:
                scores.append({"b": b, "validation_mse": float(np.mean(errors)), "origins": len(errors)})
    best_b = min(scores, key=lambda row: (row["validation_mse"], row["b"]))["b"] if scores else 0.0
    ols, ols_status = _fit_forecast(features, targets, latest, 0.0)
    diff, status = _fit_forecast(features, targets, latest, best_b)
    return {
        "ols": ols, "diffusion": diff, "b": best_b, "a": best_b / n,
        "n_pairs": n, "validation": scores, "status": status, "ols_status": ols_status,
        "selection": ("Chronological validation MSE" if scores else
                      "OLS: insufficient validation history" if tune else "OLS (no tuning)"),
    }
