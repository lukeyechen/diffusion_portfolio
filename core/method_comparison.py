from __future__ import annotations

import numpy as np
import pandas as pd

from .diffusion import diffusion_augmented_moments
from .neural_score import neural_diffusion_augmented_moments
from .moments import sample_moments
from .portfolio_rules import compute_weights
from .tuning import theoretical_horizon, validation_tuned_horizon


DEFAULT_T_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
DEFAULT_SAFE_T = 0.25


def _summary_from_returns(r: pd.Series, turnover: pd.Series, gamma: float) -> dict:
    r = pd.Series(r, dtype=float).dropna()

    if len(r) == 0:
        return {
            "Return / period": np.nan,
            "Volatility / period": np.nan,
            "Sharpe / period": np.nan,
            "CER / period": np.nan,
            "Average turnover": np.nan,
            "Max drawdown": np.nan,
            "OOS periods": 0,
        }

    mean = float(r.mean())
    var = float(r.var(ddof=0))
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = mean / vol if vol > 0 else np.nan
    cer = mean - 0.5 * float(gamma) * var

    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0

    valid_turnover = pd.Series(turnover, dtype=float).dropna()

    return {
        "Return / period": mean,
        "Volatility / period": vol,
        "Sharpe / period": sharpe,
        "CER / period": cer,
        "Average turnover": (
            float(valid_turnover.mean()) if len(valid_turnover) else np.nan
        ),
        "Max drawdown": float(drawdown.min()),
        "OOS periods": int(len(r)),
    }


def _adaptive_validation_fraction(n_obs: int) -> float:
    """
    Smaller validation share for short rolling samples.
    """
    if n_obs < 30:
        return 0.10
    if n_obs < 50:
        return 0.15
    return 0.20


def _can_validation_tune(n_obs: int, n_assets: int, val_fraction: float) -> bool:
    """
    Match the practical constraints expected by validation_tuned_horizon.
    """
    n_val = max(10, int(round(n_obs * val_fraction)))
    n_train = n_obs - n_val
    return n_train >= max(30, 3 * n_assets)


def rolling_three_method_comparison(
    returns: pd.DataFrame,
    lookback: int,
    test_periods: int,
    gamma: float,
    rule: str,
    m: int,
    beta: float,
    n_steps: int,
    constraint_mode: str,
    max_long_weight: float,
    max_short_weight: float,
    max_gross_exposure: float,
    neural_epochs: int = 500,
    neural_batch: int = 32,
    neural_learning_rate: float = 1e-3,
    neural_hidden_dim: int = 128,
    neural_validation_fraction: float = 0.20,
    neural_patience: int = 75,
    neural_min_delta: float = 1e-3,
    fallback_T_grid: list[float] | None = None,
    safe_fallback_T: float = DEFAULT_SAFE_T,
    seed: int = 42,
):
    """
    Same-date rolling comparison of Historical, Gaussian Diffusion,
    and Neural Diffusion.

    Horizon hierarchy per rolling window:
      1. Theoretical T, if positive.
      2. Validation-tuned T, if theory gives T=0 and sample is large enough.
      3. Safe fixed positive T, if validation tuning is not feasible.

    The realized OOS observation is never used in horizon selection.
    """
    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)

    n_total = len(returns)
    n_assets = returns.shape[1]

    if lookback < 20:
        raise ValueError("Comparison lookback must be at least 20 observations.")
    if lookback >= n_total:
        raise ValueError("Comparison lookback must be smaller than the full sample.")

    available = n_total - lookback
    test_periods = min(int(test_periods), available)
    if test_periods < 2:
        raise ValueError("At least two out-of-sample periods are required.")

    if fallback_T_grid is None:
        fallback_T_grid = DEFAULT_T_GRID

    if safe_fallback_T <= 0:
        raise ValueError("safe_fallback_T must be positive.")

    first_t = max(lookback, n_total - test_periods)

    method_names = [
        "Historical",
        "Gaussian Diffusion",
        "Neural Diffusion",
    ]

    records = []
    previous_weights = {name: None for name in method_names}

    for step_index, t in enumerate(range(first_t, n_total)):
        window = returns.iloc[t - lookback:t]
        x = window.to_numpy(dtype=float)
        realized_vector = returns.iloc[t].to_numpy(dtype=float)
        date = returns.index[t]

        mu_hist, sigma_hist = sample_moments(x, mle=True)

        T_theory, b_star, signal = theoretical_horizon(
            mu_hist,
            sigma_hist,
            n_obs=len(x),
            beta=float(beta),
        )

        raw_ratio = float(b_star / len(x))

        fallback_results = None
        fallback_val_fraction = np.nan

        if T_theory > 0:
            T = float(T_theory)
            T_source = "Theoretical"

        else:
            adaptive_val_fraction = _adaptive_validation_fraction(len(x))
            fallback_val_fraction = adaptive_val_fraction

            if _can_validation_tune(
                len(x),
                n_assets=n_assets,
                val_fraction=adaptive_val_fraction,
            ):
                try:
                    T, fallback_results = validation_tuned_horizon(
                        x,
                        gamma=float(gamma),
                        rule=rule,
                        m=int(m),
                        beta=float(beta),
                        n_steps=int(n_steps),
                        candidate_T=list(fallback_T_grid),
                        validation_fraction=float(adaptive_val_fraction),
                        seed=int(seed + 500 + step_index),
                        constraint_mode=constraint_mode,
                        max_long_weight=float(max_long_weight),
                        max_short_weight=float(max_short_weight),
                        max_gross_exposure=float(max_gross_exposure),
                    )
                    T_source = "Validation fallback"
                except Exception:
                    T = float(safe_fallback_T)
                    T_source = "Safe fallback"
            else:
                T = float(safe_fallback_T)
                T_source = "Safe fallback"

        if T <= 0:
            T = float(safe_fallback_T)
            T_source = "Safe fallback"

        # ------------------------------------------------------------
        # Historical
        # ------------------------------------------------------------
        w_hist = compute_weights(
            rule,
            mu_hist,
            sigma_hist,
            gamma=float(gamma),
            returns=x,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        )

        # ------------------------------------------------------------
        # Gaussian diffusion
        # ------------------------------------------------------------
        mu_g, sigma_g, _, combined_g = diffusion_augmented_moments(
            x,
            m=int(m),
            horizon=float(T),
            beta=float(beta),
            n_steps=int(n_steps),
            seed=int(seed + 1000 + step_index),
        )

        w_g = compute_weights(
            rule,
            mu_g,
            sigma_g,
            gamma=float(gamma),
            returns=combined_g,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        )

        # ------------------------------------------------------------
        # Neural diffusion
        # ------------------------------------------------------------
        mu_n, sigma_n, _, combined_n, training_result = (
            neural_diffusion_augmented_moments(
                x,
                m=int(m),
                horizon=float(T),
                beta=float(beta),
                n_steps=int(n_steps),
                epochs=int(neural_epochs),
                batch_size=int(neural_batch),
                learning_rate=float(neural_learning_rate),
                hidden_dim=int(neural_hidden_dim),
                validation_fraction=float(neural_validation_fraction),
                patience=int(neural_patience),
                min_delta=float(neural_min_delta),
                seed=int(seed + 2000 + step_index),
            )
        )

        w_n = compute_weights(
            rule,
            mu_n,
            sigma_n,
            gamma=float(gamma),
            returns=combined_n,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        )

        weights_by_method = {
            "Historical": np.asarray(w_hist, dtype=float),
            "Gaussian Diffusion": np.asarray(w_g, dtype=float),
            "Neural Diffusion": np.asarray(w_n, dtype=float),
        }

        neural_best_val = min(training_result.val_losses)
        neural_zero_val = training_result.zero_score_baseline_val

        for method, w in weights_by_method.items():
            realized_return = float(w @ realized_vector)

            prev = previous_weights[method]
            turnover = (
                np.nan
                if prev is None
                else 0.5 * float(np.abs(w - prev).sum())
            )
            previous_weights[method] = w.copy()

            records.append(
                {
                    "Date": date,
                    "Method": method,
                    "Realized return": realized_return,
                    "Turnover": turnover,
                    "T": float(T),
                    "T source": T_source,
                    "b*": float(b_star),
                    "b*/n": raw_ratio,
                    "s_T^2": float(signal),
                    "Fallback validation fraction": (
                        float(fallback_val_fraction)
                        if np.isfinite(fallback_val_fraction)
                        else np.nan
                    ),
                    "Neural best validation DSM": (
                        float(neural_best_val)
                        if method == "Neural Diffusion"
                        else np.nan
                    ),
                    "Neural zero-score baseline": (
                        float(neural_zero_val)
                        if method == "Neural Diffusion"
                        else np.nan
                    ),
                }
            )

    detail = pd.DataFrame(records)

    summaries = []
    for method in method_names:
        sub = detail[detail["Method"] == method].copy()
        summary = _summary_from_returns(
            sub["Realized return"],
            sub["Turnover"],
            gamma=float(gamma),
        )
        summary["Method"] = method
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)[
        [
            "Method",
            "Return / period",
            "Volatility / period",
            "Sharpe / period",
            "CER / period",
            "Average turnover",
            "Max drawdown",
            "OOS periods",
        ]
    ]

    pivot = detail.pivot(
        index="Date",
        columns="Method",
        values="Realized return",
    ).sort_index()

    wealth = (1.0 + pivot).cumprod()

    tuning_summary = (
        detail[
            [
                "Date",
                "T",
                "T source",
                "b*",
                "b*/n",
                "s_T^2",
                "Fallback validation fraction",
            ]
        ]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return summary_df, detail, wealth, tuning_summary
