from __future__ import annotations
import numpy as np
import pandas as pd


def sample_moments(returns: pd.DataFrame | np.ndarray, mle: bool = True):
    """Estimate sample mean and covariance; mle=True divides by n."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2D array.")
    if x.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    mu = x.mean(axis=0)
    centered = x - mu
    denom = x.shape[0] if mle else x.shape[0] - 1
    sigma = centered.T @ centered / denom
    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma


def regularize_covariance(sigma: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    scale = max(float(np.trace(sigma) / max(sigma.shape[0], 1)), 1.0)
    return sigma + ridge * scale * np.eye(sigma.shape[0])


def covariance_condition_number(sigma: np.ndarray) -> float:
    return float(np.linalg.cond(np.asarray(sigma, dtype=float)))
