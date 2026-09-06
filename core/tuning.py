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
    Choose T by one chronological train/validation CER split.

    This function is retained for the lightweight web-app tuning mode.
    For stronger research validation, use nested_rolling_validation_tuned_horizon.
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


def nested_rolling_validation_tuned_horizon(
    returns: np.ndarray,
    gamma: float,
    rule: str,
    m: int,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_T: list[float] | None = None,
    inner_folds: int = 4,
    validation_size: int = 12,
    min_train_size: int | None = None,
    seed: int = 42,
    constraint_mode: str = "Long-only",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
):
    """
    Choose the diffusion horizon with rolling-origin inner validation.

    The outer estimation window is never mixed with the future OOS return.
    Inside that outer window, create several expanding-window folds. For
    example, with n=120, inner_folds=4, validation_size=12:

        fold 1: train months 1:72   -> validate 73:84
        fold 2: train months 1:84   -> validate 85:96
        fold 3: train months 1:96   -> validate 97:108
        fold 4: train months 1:108  -> validate 109:120

    Each candidate T is fit independently in every inner fold. Its score is
    the mean validation certainty equivalent across folds. The T with the
    highest mean CER is selected; exact ties prefer the smaller T.

    Returns
    -------
    best_T : float
        Selected horizon.
    results : list[dict]
        Candidate-level mean/std CER plus per-fold diagnostics.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2D array of observations by assets.")
    n, n_assets = x.shape

    if candidate_T is None:
        candidate_T = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    candidate_T = [float(T) for T in candidate_T]
    if not candidate_T:
        raise ValueError("candidate_T must contain at least one horizon.")
    if any(T < 0 for T in candidate_T):
        raise ValueError("candidate_T values must be nonnegative.")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2 for nested rolling validation.")
    if validation_size < 2:
        raise ValueError("validation_size must be at least 2 observations.")

    required_train = max(30, 3 * n_assets) if min_train_size is None else int(min_train_size)
    if required_train < max(3, n_assets + 1):
        raise ValueError("min_train_size is too small for stable moment estimation.")

    first_train_n = n - int(inner_folds) * int(validation_size)
    if first_train_n < required_train:
        raise ValueError(
            "Not enough observations for the requested nested rolling folds: "
            f"first inner training sample would have {first_train_n}, "
            f"but at least {required_train} are required."
        )

    folds = []
    for fold_idx in range(int(inner_folds)):
        train_end = first_train_n + fold_idx * int(validation_size)
        valid_start = train_end
        valid_end = valid_start + int(validation_size)
        folds.append((train_end, valid_start, valid_end))

    results = []
    for candidate_idx, T in enumerate(candidate_T):
        fold_cers = []
        fold_details = []

        for fold_idx, (train_end, valid_start, valid_end) in enumerate(folds):
            train = x[:train_end]
            valid = x[valid_start:valid_end]

            mu_aug, sigma_aug, _, combined = diffusion_augmented_moments(
                train,
                m=int(m),
                horizon=float(T),
                beta=float(beta),
                n_steps=int(n_steps),
                seed=int(seed + candidate_idx * 1000 + fold_idx),
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

            mu_val, sigma_val = sample_moments(valid, mle=True)
            cer = certainty_equivalent(
                w,
                mu_val,
                sigma_val,
                gamma=float(gamma),
            )
            fold_cers.append(float(cer))
            fold_details.append(
                {
                    "fold": int(fold_idx + 1),
                    "train_n": int(train_end),
                    "validation_n": int(valid_end - valid_start),
                    "validation_start": int(valid_start),
                    "validation_end": int(valid_end),
                    "validation_CER": float(cer),
                }
            )

        mean_cer = float(np.mean(fold_cers))
        std_cer = float(np.std(fold_cers, ddof=1)) if len(fold_cers) > 1 else 0.0
        results.append(
            {
                "T": float(T),
                "mean_validation_CER": mean_cer,
                "std_validation_CER": std_cer,
                "min_validation_CER": float(np.min(fold_cers)),
                "max_validation_CER": float(np.max(fold_cers)),
                "fold_count": int(len(fold_cers)),
                "fold_CERs": fold_cers,
                "folds": fold_details,
            }
        )

    best = max(results, key=lambda r: (r["mean_validation_CER"], -r["T"]))
    return float(best["T"]), results
