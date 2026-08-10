from __future__ import annotations

import numpy as np

from .diffusion import diffusion_augmented_moments
from .metrics import certainty_equivalent
from .moments import sample_moments
from .portfolio_rules import compute_weights


def optimal_b_star(mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Theoretical interior tuning constant

        b* = A / Q,

    where
        A = (N+2)||mu||^2 + tr(Sigma),
        Q = mu' Sigma mu.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float)
    n_assets = mu.size

    A = (n_assets + 2.0) * float(mu @ mu) + float(np.trace(sigma))
    Q = float(mu @ sigma @ mu)

    if Q <= 0 or not np.isfinite(Q):
        raise ValueError("mu' Sigma mu must be positive to compute b*.")

    return A / Q


def theoretical_signal_level(mu: np.ndarray, sigma: np.ndarray, n_obs: int) -> float:
    """
    Constrained terminal signal:
        s_T^2 = min(1, b*/n).
    """
    b_star = optimal_b_star(mu, sigma)
    return float(min(1.0, b_star / float(n_obs)))


def theoretical_horizon(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_obs: int,
    beta: float = 1.0,
) -> tuple[float, float, float]:
    """
    Constrained theoretical horizon.

        T* = max{0, (1/beta) log(n/b*)}

    Returns
    -------
    T_star, b_star, signal_level
    """
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")

    b_star = optimal_b_star(mu, sigma)
    raw_ratio = b_star / float(n_obs)
    signal = min(1.0, raw_ratio)

    if signal >= 1.0:
        T_star = 0.0
    else:
        T_star = float(-np.log(signal) / beta)

    return T_star, b_star, signal


def constant_beta_horizon(
    b_star: float,
    n_obs: int,
    beta: float = 1.0,
    clip: bool = True,
) -> float:
    """
    Backward-compatible helper.

        exp(-beta T) = b*/n.
    """
    if beta <= 0:
        raise ValueError("beta must be positive.")
    ratio = b_star / float(n_obs)
    if ratio <= 0:
        raise ValueError("b*/n must be positive.")
    if ratio >= 1:
        if clip:
            return 0.0
        raise ValueError("b*/n >= 1 implies the constrained optimum T=0.")
    return float(-np.log(ratio) / beta)


def terminal_signal_level(b_star: float, n_obs: int) -> float:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    return float(min(1.0, b_star / float(n_obs)))


def validation_tuned_horizon(
    returns: np.ndarray,
    gamma: float,
    rule: str,
    m: int,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_T: list[float] | None = None,
    validation_fraction: float = 0.20,
    seed: int = 42,
    constraint_mode: str = "Long-only",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
):
    """
    Choose T by a simple train/validation CER criterion.

    The historical window is split chronologically:
      - first part: train the Gaussian diffusion estimator;
      - final part: validation observations.

    For each candidate T:
      1. augment the training sample with M diffusion-generated observations;
      2. estimate augmented moments;
      3. construct portfolio weights;
      4. evaluate those fixed weights on the validation returns;
      5. compute validation CER.

    This is deliberately simple and transparent for the web app.
    For research-grade deployment, replace this with nested rolling
    validation to avoid overfitting T.
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]

    if candidate_T is None:
        candidate_T = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]

    if not (0.10 <= validation_fraction <= 0.40):
        raise ValueError("validation_fraction should be between 0.10 and 0.40.")

    n_val = max(10, int(round(n * validation_fraction)))
    n_train = n - n_val

    if n_train < max(30, 3 * x.shape[1]):
        raise ValueError(
            "Not enough observations remain in the training split for validation tuning."
        )

    train = x[:n_train]
    valid = x[n_train:]

    mu_val, sigma_val = sample_moments(valid, mle=True)

    results = []

    for j, T in enumerate(candidate_T):
        mu_aug, sigma_aug, _, combined = diffusion_augmented_moments(
            train,
            m=int(m),
            horizon=float(T),
            beta=float(beta),
            n_steps=int(n_steps),
            seed=int(seed + j),
        )

        w = compute_weights(
            rule,
            mu_aug,
            sigma_aug,
            gamma=float(gamma),
            returns=combined,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        )

        cer = certainty_equivalent(
            w,
            mu_val,
            sigma_val,
            gamma=float(gamma),
        )

        results.append(
            {
                "T": float(T),
                "validation_CER": float(cer),
                "train_n": int(n_train),
                "validation_n": int(n_val),
            }
        )

    best = max(results, key=lambda r: r["validation_CER"])
    return float(best["T"]), results
