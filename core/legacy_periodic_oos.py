from __future__ import annotations

import numpy as np
import pandas as pd

from .diffusion import diffusion_augmented_moments
from .moments import covariance_condition_number, sample_moments
from .neural_score import neural_diffusion_augmented_moments
from .portfolio_rules import compute_weights
from .tuning import theoretical_horizon, validation_tuned_horizon


DEFAULT_T_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
DEFAULT_SAFE_T = 0.25


LEGACY_HORIZON_PRESETS: dict[str, dict[str, int | str]] = {
    "1 week": {
        "source": "weekly",
        "block_size": 1,
        "periods_per_year": 52,
        "lookback": 520,
    },
    "2 weeks": {
        "source": "weekly",
        "block_size": 2,
        "periods_per_year": 26,
        "lookback": 260,
    },
    "1 month": {
        "source": "monthly",
        "block_size": 1,
        "periods_per_year": 12,
        "lookback": 120,
    },
    "3 months": {
        "source": "monthly",
        "block_size": 3,
        "periods_per_year": 4,
        "lookback": 40,
    },
}


def get_legacy_horizon_preset(label: str) -> dict[str, int | str]:
    if label not in LEGACY_HORIZON_PRESETS:
        raise ValueError(f"Unsupported legacy holding period: {label}")
    return dict(LEGACY_HORIZON_PRESETS[label])


def _adaptive_validation_fraction(n_obs: int) -> float:
    if n_obs < 30:
        return 0.10
    if n_obs < 50:
        return 0.15
    return 0.20


def _can_validation_tune(n_obs: int, n_assets: int, val_fraction: float) -> bool:
    n_val = max(10, int(round(n_obs * val_fraction)))
    n_train = n_obs - n_val
    return n_train >= max(30, 3 * n_assets)


def _portfolio_weights(
    rule: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    returns: np.ndarray,
    *,
    gamma: float,
    constraint_mode: str,
    max_long_weight: float,
    max_short_weight: float,
    max_gross_exposure: float,
) -> np.ndarray:
    w = compute_weights(
        rule,
        mu,
        sigma,
        gamma=float(gamma),
        returns=returns,
        constraint_mode=constraint_mode,
        max_long_weight=float(max_long_weight),
        max_short_weight=float(max_short_weight),
        max_gross_exposure=float(max_gross_exposure),
    )
    return np.asarray(w, dtype=float).reshape(-1)


def _summary_metrics(
    returns: np.ndarray,
    turnover: np.ndarray,
    *,
    gamma: float,
    periods_per_year: int,
) -> dict[str, float]:
    r = np.asarray(returns, dtype=float).reshape(-1)
    to = np.asarray(turnover, dtype=float).reshape(-1)
    if r.size == 0:
        raise ValueError("No realized returns to summarize.")

    mean = float(np.mean(r))
    var = float(np.var(r, ddof=0))
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe_period = mean / vol if vol > 0 else np.nan
    cer = mean - 0.5 * float(gamma) * var

    wealth = np.cumprod(1.0 + r)
    if wealth[-1] > 0:
        cagr = float(wealth[-1] ** (float(periods_per_year) / len(r)) - 1.0)
    else:
        cagr = np.nan
    ann_vol = float(vol * np.sqrt(float(periods_per_year)))
    ann_mean = float(mean * float(periods_per_year))
    ann_sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
    dd = wealth / np.maximum.accumulate(wealth) - 1.0

    to_cost = np.nan_to_num(to, nan=0.0)
    net10 = r - 0.0010 * to_cost
    net25 = r - 0.0025 * to_cost

    def _net_cagr(x: np.ndarray) -> float:
        w = np.cumprod(1.0 + x)
        if w[-1] <= 0:
            return np.nan
        return float(w[-1] ** (float(periods_per_year) / len(x)) - 1.0)

    finite_to = to[np.isfinite(to)]
    return {
        "Return / period": mean,
        "Volatility / period": vol,
        "Sharpe / period": sharpe_period,
        "CER / period": cer,
        "CAGR": cagr,
        "Annualized volatility": ann_vol,
        "Annualized Sharpe": ann_sharpe,
        "Average turnover": float(np.mean(finite_to)) if finite_to.size else np.nan,
        "Max drawdown": float(np.min(dd)),
        "Net CAGR 10bps": _net_cagr(net10),
        "Net CAGR 25bps": _net_cagr(net25),
        "Final $10,000": float(10000.0 * wealth[-1]),
        "OOS periods": int(len(r)),
    }


def legacy_periodic_oos_comparison(
    returns: pd.DataFrame,
    *,
    lookback: int,
    test_periods: int,
    periods_per_year: int,
    gamma: float,
    rule: str,
    m: int,
    beta: float,
    n_steps: int,
    constraint_mode: str,
    max_long_weight: float,
    max_short_weight: float,
    max_gross_exposure: float,
    neural_epochs: int,
    neural_batch: int,
    neural_learning_rate: float,
    neural_hidden_dim: int,
    neural_validation_fraction: float,
    neural_patience: int,
    neural_min_delta: float,
    safe_fallback_T: float = DEFAULT_SAFE_T,
    fallback_T_grid: list[float] | None = None,
    seed: int = 42,
):
    """Legacy Historical/Gaussian/Neural OOS comparison at one holding period.

    Unlike ``monthly_rebalance_oos_comparison``, estimation and realized returns
    use the same sampling frequency. A 1-week model is therefore tested on the
    next real 1-week return, a 2-week model on the next non-overlapping 2-week
    return, and so on.

    The legacy T hierarchy is intentionally preserved:
      1. use a positive theoretical T when available;
      2. otherwise use the old single-split validation tuner;
      3. otherwise use a positive safe fallback T.

    This function deliberately retains Monte-Carlo Gaussian augmentation and
    the Learned Neural Score so the page remains a true legacy-method study.
    """
    if fallback_T_grid is None:
        fallback_T_grid = list(DEFAULT_T_GRID)

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    data = returns.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.shape[1] < 1:
        raise ValueError("No asset return columns are available.")

    lookback = int(lookback)
    test_periods = int(test_periods)
    periods_per_year = int(periods_per_year)
    if lookback < 2 or lookback >= len(data):
        raise ValueError("lookback must be at least 2 and smaller than the data length.")
    if test_periods < 1:
        raise ValueError("test_periods must be positive.")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive.")

    candidate_positions = list(range(lookback, len(data)))
    candidate_positions = candidate_positions[-min(test_periods, len(candidate_positions)) :]
    if not candidate_positions:
        raise ValueError("No OOS periods are available after the selected lookback.")

    method_names = ["Historical", "Gaussian Diffusion", "Neural Diffusion"]
    previous_weights = {name: None for name in method_names}
    records: list[dict] = []
    regime_records: list[dict] = []

    diagnostics = {
        "Requested OOS periods": int(test_periods),
        "Candidate OOS periods": int(len(candidate_positions)),
        "Evaluated OOS periods": 0,
        "Theoretical T windows": 0,
        "Validation fallback windows": 0,
        "Safe fallback windows": 0,
    }

    for j, pos in enumerate(candidate_positions):
        train = data.iloc[pos - lookback : pos]
        x = train.to_numpy(dtype=float)
        realized_date = data.index[pos]
        realized_vector = data.iloc[pos].to_numpy(dtype=float)

        mu_hist, sigma_hist = sample_moments(x, mle=True)

        try:
            T_theory, b_star, signal_theory = theoretical_horizon(
                mu_hist,
                sigma_hist,
                n_obs=len(x),
                beta=float(beta),
            )
        except Exception:
            T_theory = 0.0
            b_star = np.nan
            signal_theory = np.nan

        ratio = float(b_star / len(x)) if np.isfinite(b_star) else np.nan

        if T_theory > 0:
            T = float(T_theory)
            T_source = "Theoretical"
            diagnostics["Theoretical T windows"] += 1
        else:
            vf = _adaptive_validation_fraction(len(x))
            if _can_validation_tune(len(x), data.shape[1], vf):
                try:
                    T, _ = validation_tuned_horizon(
                        x,
                        gamma=float(gamma),
                        rule=rule,
                        m=int(m),
                        beta=float(beta),
                        n_steps=int(n_steps),
                        candidate_T=list(fallback_T_grid),
                        validation_fraction=float(vf),
                        seed=int(seed + 500 + j),
                        constraint_mode=constraint_mode,
                        max_long_weight=float(max_long_weight),
                        max_short_weight=float(max_short_weight),
                        max_gross_exposure=float(max_gross_exposure),
                    )
                    T_source = "Validation fallback"
                    diagnostics["Validation fallback windows"] += 1
                except Exception:
                    T = float(safe_fallback_T)
                    T_source = "Safe fallback"
                    diagnostics["Safe fallback windows"] += 1
            else:
                T = float(safe_fallback_T)
                T_source = "Safe fallback"
                diagnostics["Safe fallback windows"] += 1

        if not np.isfinite(T) or T <= 0:
            T = float(safe_fallback_T)
            T_source = "Safe fallback"

        actual_signal = float(np.exp(-float(beta) * float(T)))

        w_hist = _portfolio_weights(
            rule,
            mu_hist,
            sigma_hist,
            x,
            gamma=gamma,
            constraint_mode=constraint_mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )

        mu_g, sigma_g, _, combined_g = diffusion_augmented_moments(
            x,
            m=int(m),
            horizon=float(T),
            beta=float(beta),
            n_steps=int(n_steps),
            seed=int(seed + 1000 + j),
        )
        w_g = _portfolio_weights(
            rule,
            mu_g,
            sigma_g,
            combined_g,
            gamma=gamma,
            constraint_mode=constraint_mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )

        mu_n, sigma_n, _, combined_n, neural_result = neural_diffusion_augmented_moments(
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
            seed=int(seed + 2000 + j),
        )
        w_n = _portfolio_weights(
            rule,
            mu_n,
            sigma_n,
            combined_n,
            gamma=gamma,
            constraint_mode=constraint_mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )

        weights = {
            "Historical": w_hist,
            "Gaussian Diffusion": w_g,
            "Neural Diffusion": w_n,
        }

        regime_records.append(
            {
                "Realized date": realized_date,
                "T": float(T),
                "T source": T_source,
                "b*": float(b_star) if np.isfinite(b_star) else np.nan,
                "b*/n": ratio,
                "s_T^2": actual_signal,
                "Equal-weight realized return": float(np.mean(realized_vector)),
                "Average asset mean": float(np.mean(mu_hist)),
                "Cross-sectional dispersion": float(np.std(realized_vector, ddof=0)),
                "Average asset volatility": float(
                    np.mean(np.sqrt(np.clip(np.diag(sigma_hist), 0.0, None)))
                ),
                "Mean-vector norm": float(np.linalg.norm(mu_hist)),
                "Covariance trace": float(np.trace(sigma_hist)),
                "Covariance condition number": float(covariance_condition_number(sigma_hist)),
            }
        )

        for method, w in weights.items():
            prev = previous_weights[method]
            turnover = np.nan if prev is None else 0.5 * float(np.abs(w - prev).sum())
            previous_weights[method] = w.copy()
            records.append(
                {
                    "Realized date": realized_date,
                    "Method": method,
                    "Realized return": float(w @ realized_vector),
                    "Turnover": turnover,
                    "T": float(T),
                    "T source": T_source,
                    "b*": float(b_star) if np.isfinite(b_star) else np.nan,
                    "b*/n": ratio,
                    "s_T^2": actual_signal,
                    "Neural best validation DSM": (
                        float(min(neural_result.val_losses))
                        if method == "Neural Diffusion" and len(neural_result.val_losses)
                        else np.nan
                    ),
                }
            )

        diagnostics["Evaluated OOS periods"] += 1

    detail = pd.DataFrame(records)
    if detail.empty:
        raise ValueError("No OOS windows could be evaluated.")

    summary_rows = []
    for method in method_names:
        sub = detail[detail["Method"] == method].copy()
        metrics = _summary_metrics(
            sub["Realized return"].to_numpy(dtype=float),
            sub["Turnover"].to_numpy(dtype=float),
            gamma=float(gamma),
            periods_per_year=periods_per_year,
        )
        summary_rows.append({"Method": method, **metrics})
    summary = pd.DataFrame(summary_rows)

    pivot = detail.pivot(
        index="Realized date",
        columns="Method",
        values="Realized return",
    ).sort_index()
    wealth = (1.0 + pivot).cumprod()

    regime_detail = pd.DataFrame(regime_records)
    source_counts = (
        regime_detail["T source"].value_counts().rename_axis("T source").reset_index(name="Windows")
    )

    return summary, detail, wealth, diagnostics, source_counts, regime_detail
