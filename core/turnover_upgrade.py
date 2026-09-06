from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .diffusion import forward_moments
from .metrics import certainty_equivalent
from .moments import regularize_covariance, sample_moments
from .portfolio_rules import solve_mv_constrained


def exact_reverse_diffusion_moments(
    returns: np.ndarray,
    horizon: float,
    beta: float = 1.0,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic moments of the Euler reverse-SDE used by the app.

    This propagates the mean/covariance recursion of the existing
    Euler-Maruyama sampler exactly, so it removes Monte-Carlo noise while
    preserving the same finite-step reverse dynamics and prior initialization.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be a 2D array with at least two observations.")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    mu_hat, sigma_hat = sample_moments(x, mle=True)
    sigma_hat = regularize_covariance(sigma_hat)
    n_assets = x.shape[1]

    if horizon == 0:
        return mu_hat.copy(), sigma_hat.copy()

    eye = np.eye(n_assets)
    prior_var = 1.0 - np.exp(-beta * horizon)
    mean = np.zeros(n_assets, dtype=float)
    cov = max(float(prior_var), 1e-12) * eye
    dt = float(horizon) / int(n_steps)

    for k in range(int(n_steps)):
        t = k * dt
        u = max(float(horizon) - t, 0.0)
        mu_u, sigma_u = forward_moments(mu_hat, sigma_hat, u, beta)
        sigma_u = regularize_covariance(sigma_u)
        a_u = np.linalg.inv(sigma_u)
        b_u = a_u @ mu_u

        # X_{k+1} = F X_k + beta*b_u*dt + sqrt(beta*dt) eps.
        drift_matrix = -beta * (a_u - 0.5 * eye)
        f = eye + drift_matrix * dt
        mean = f @ mean + beta * b_u * dt
        cov = f @ cov @ f.T + beta * dt * eye

    cov = regularize_covariance(0.5 * (cov + cov.T))
    return mean, cov


def exact_diffusion_augmented_moments(
    returns: np.ndarray,
    m: int,
    horizon: float,
    beta: float = 1.0,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact population-moment analogue of adding ``m`` diffusion samples.

    The real sample keeps its MLE moments. The synthetic block is represented
    by its deterministic Gaussian mean/covariance rather than a random finite
    draw. The real/synthetic blocks are mixed with weights n/(n+m) and m/(n+m).
    At T=0 this leaves the sample moments unchanged exactly.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be a 2D array with at least two observations.")
    if m < 0:
        raise ValueError("m must be nonnegative.")

    mu_real, sigma_real = sample_moments(x, mle=True)
    sigma_real = regularize_covariance(sigma_real)
    if m == 0:
        return mu_real, sigma_real, mu_real.copy(), sigma_real.copy()

    mu_fake, sigma_fake = exact_reverse_diffusion_moments(
        x,
        horizon=float(horizon),
        beta=float(beta),
        n_steps=int(n_steps),
    )

    n = x.shape[0]
    total = float(n + int(m))
    wr = float(n) / total
    wf = float(m) / total
    mu_aug = wr * mu_real + wf * mu_fake

    dr = mu_real - mu_aug
    df = mu_fake - mu_aug
    sigma_aug = (
        wr * (sigma_real + np.outer(dr, dr))
        + wf * (sigma_fake + np.outer(df, df))
    )
    sigma_aug = regularize_covariance(0.5 * (sigma_aug + sigma_aug.T))
    return mu_aug, sigma_aug, mu_fake, sigma_fake


def portfolio_turnover(target: np.ndarray, previous: np.ndarray) -> float:
    target = np.asarray(target, dtype=float).reshape(-1)
    previous = np.asarray(previous, dtype=float).reshape(-1)
    if target.shape != previous.shape:
        raise ValueError("target and previous weights must have the same shape.")
    return 0.5 * float(np.abs(target - previous).sum())


def solve_mv_turnover_aware(
    mu: np.ndarray,
    sigma: np.ndarray,
    previous_weights: np.ndarray,
    turnover_penalty: float,
    gamma: float = 3.0,
    mode: str = "Long-only",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
) -> np.ndarray:
    """Constrained MV with an exact L1 turnover penalty.

    Objective:
        max  w'mu - gamma/2 w'Sigma w
             - turnover_penalty * 0.5 * ||w - w_prev||_1.

    Auxiliary variables represent the absolute weight changes, avoiding a
    non-differentiable ``abs`` directly inside the SLSQP objective.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = regularize_covariance(np.asarray(sigma, dtype=float))
    prev = np.asarray(previous_weights, dtype=float).reshape(-1)
    n = mu.size

    if prev.size != n:
        raise ValueError("previous_weights has the wrong dimension.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if turnover_penalty < 0:
        raise ValueError("turnover_penalty must be nonnegative.")
    if not np.all(np.isfinite(prev)):
        raise ValueError("previous_weights must be finite.")

    prev_sum = float(prev.sum())
    if abs(prev_sum) < 1e-12:
        prev = np.ones(n) / n
    else:
        prev = prev / prev_sum

    if turnover_penalty == 0:
        return solve_mv_constrained(
            mu,
            sigma,
            gamma=gamma,
            mode=mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )

    if mode == "Long-only" and max_long_weight * n < 1.0 - 1e-12:
        raise ValueError("Long-only problem is infeasible under max_long_weight.")

    if mode == "Long-only":
        w_bounds = [(0.0, float(max_long_weight))] * n
    elif mode == "Limited Long-Short":
        if max_short_weight < 0 or max_gross_exposure < 1.0:
            raise ValueError("Invalid long-short constraints.")
        w_bounds = [(-float(max_short_weight), float(max_long_weight))] * n
    else:
        raise ValueError("Turnover-aware solver supports Long-only or Limited Long-Short.")

    # Equal weight is a robust feasible starting point for the user's 5-asset,
    # 40%-cap setting even if drift has pushed current holdings above the cap.
    w0 = np.ones(n) / n
    z0 = np.abs(w0 - prev)
    y0 = np.concatenate([w0, z0])
    bounds = w_bounds + [(0.0, None)] * n

    def objective(y):
        w = y[:n]
        z = y[n:]
        utility = float(w @ mu) - 0.5 * gamma * float(w @ sigma @ w)
        return -utility + 0.5 * float(turnover_penalty) * float(z.sum())

    constraints = [
        {"type": "eq", "fun": lambda y: float(np.sum(y[:n]) - 1.0)},
        {"type": "ineq", "fun": lambda y: y[n:] - (y[:n] - prev)},
        {"type": "ineq", "fun": lambda y: y[n:] + (y[:n] - prev)},
    ]
    if mode == "Limited Long-Short":
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y: float(max_gross_exposure) - np.abs(y[:n]).sum(),
            }
        )

    result = minimize(
        objective,
        x0=y0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1500, "ftol": 1e-11},
    )
    if not result.success:
        raise ValueError(f"Turnover-aware optimization failed: {result.message}")
    return np.asarray(result.x[:n], dtype=float)


def partial_rebalance(
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """Move only ``alpha`` of the way from current drifted weights to target."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    target = np.asarray(target_weights, dtype=float).reshape(-1)
    prev = np.asarray(previous_weights, dtype=float).reshape(-1)
    if target.shape != prev.shape:
        raise ValueError("target_weights and previous_weights must match.")
    if abs(float(prev.sum())) < 1e-12:
        raise ValueError("previous_weights must have nonzero total weight.")
    prev = prev / float(prev.sum())
    w = (1.0 - float(alpha)) * prev + float(alpha) * target
    return w / float(w.sum())


def nested_exact_horizon(
    returns: np.ndarray,
    gamma: float,
    m: int,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_T: list[float] | None = None,
    inner_folds: int = 4,
    validation_size: int = 12,
    min_train_size: int | None = None,
    selection_rule: str = "one_se",
    constraint_mode: str = "Long-only",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
) -> tuple[float, list[dict]]:
    """Nested rolling T selection using deterministic diffusion moments.

    ``selection_rule='best'`` chooses maximum mean validation CER.
    ``selection_rule='one_se'`` chooses the smallest T whose mean CER is within
    one standard error of the best candidate, reducing horizon instability.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2D array.")
    n, n_assets = x.shape
    if candidate_T is None:
        candidate_T = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    candidate_T = [float(t) for t in candidate_T]
    if not candidate_T or any(t < 0 for t in candidate_T):
        raise ValueError("candidate_T must contain nonnegative horizons.")
    if inner_folds < 2 or validation_size < 2:
        raise ValueError("Need at least two folds and two validation observations.")

    required = max(30, 3 * n_assets) if min_train_size is None else int(min_train_size)
    first_train_n = n - int(inner_folds) * int(validation_size)
    if first_train_n < required:
        raise ValueError("Not enough observations for requested nested folds.")

    folds = []
    for fold_idx in range(int(inner_folds)):
        train_end = first_train_n + fold_idx * int(validation_size)
        valid_start = train_end
        valid_end = valid_start + int(validation_size)
        folds.append((train_end, valid_start, valid_end))

    results = []
    for t in candidate_T:
        cers = []
        for train_end, valid_start, valid_end in folds:
            train = x[:train_end]
            valid = x[valid_start:valid_end]
            mu_aug, sigma_aug, _, _ = exact_diffusion_augmented_moments(
                train,
                m=int(m),
                horizon=t,
                beta=float(beta),
                n_steps=int(n_steps),
            )
            w = solve_mv_constrained(
                mu_aug,
                sigma_aug,
                gamma=float(gamma),
                mode=constraint_mode,
                max_long_weight=float(max_long_weight),
                max_short_weight=float(max_short_weight),
                max_gross_exposure=float(max_gross_exposure),
            )
            mu_val, sigma_val = sample_moments(valid, mle=True)
            cers.append(float(certainty_equivalent(w, mu_val, sigma_val, gamma=float(gamma))))

        std = float(np.std(cers, ddof=1)) if len(cers) > 1 else 0.0
        results.append(
            {
                "T": t,
                "mean_validation_CER": float(np.mean(cers)),
                "std_validation_CER": std,
                "standard_error_CER": std / np.sqrt(len(cers)) if cers else np.nan,
                "fold_CERs": cers,
                "fold_count": len(cers),
            }
        )

    best = max(results, key=lambda r: (r["mean_validation_CER"], -r["T"]))
    rule = selection_rule.strip().lower()
    if rule == "best":
        chosen = best
    elif rule in {"one_se", "one-se", "1se"}:
        threshold = best["mean_validation_CER"] - best["standard_error_CER"]
        eligible = [r for r in results if r["mean_validation_CER"] >= threshold - 1e-15]
        chosen = min(eligible, key=lambda r: r["T"])
    else:
        raise ValueError("selection_rule must be 'best' or 'one_se'.")

    for r in results:
        r["best_mean_CER"] = float(best["mean_validation_CER"])
        r["selected_by_rule"] = bool(r["T"] == chosen["T"])
        r["selection_rule"] = rule
    return float(chosen["T"]), results
