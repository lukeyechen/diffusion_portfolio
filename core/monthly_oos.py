from __future__ import annotations

import numpy as np
import pandas as pd

from .diffusion import diffusion_augmented_moments
from .neural_score import neural_diffusion_augmented_moments
from .moments import sample_moments, covariance_condition_number
from .portfolio_rules import compute_weights
from .tuning import theoretical_horizon, validation_tuned_horizon


DEFAULT_T_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
DEFAULT_SAFE_T = 0.25


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


def monthly_rebalance_oos_comparison(
    model_returns: pd.DataFrame,
    monthly_realized_returns: pd.DataFrame,
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
    """
    Estimate from overlapping horizon returns, rebalance monthly, and evaluate
    on the next actual 1M return.

    Horizon hierarchy for each monthly rolling window:
      1. Theoretical positive T.
      2. Validation-tuned T if theory gives T=0 and sample is sufficient.
      3. Safe fallback T if validation tuning is not feasible.

    Returns summary, detail, wealth, diagnostics.
    """
    if fallback_T_grid is None:
        fallback_T_grid = DEFAULT_T_GRID

    common_cols = [c for c in model_returns.columns if c in monthly_realized_returns.columns]
    model_returns = model_returns[common_cols].copy()
    monthly_realized_returns = monthly_realized_returns[common_cols].copy()

    eligible_dates = model_returns.index.intersection(monthly_realized_returns.index).sort_values()

    if len(eligible_dates) <= lookback:
        raise ValueError("Not enough aligned observations for monthly OOS comparison.")

    # Need current model date plus next monthly realized date.
    candidate_dates = []
    for d in eligible_dates:
        try:
            p = monthly_realized_returns.index.get_loc(d)
            if isinstance(p, slice):
                p = p.start
            if p + 1 < len(monthly_realized_returns):
                candidate_dates.append(d)
        except KeyError:
            pass

    if len(candidate_dates) < 2:
        raise ValueError("Not enough candidate monthly OOS dates.")

    test_dates = candidate_dates[-int(min(test_periods, len(candidate_dates))):]

    method_names = ["Historical", "Gaussian Diffusion", "Neural Diffusion"]
    records = []
    regime_records = []
    previous_weights = {m: None for m in method_names}

    diagnostics = {
        "Requested OOS months": int(test_periods),
        "Candidate OOS months": int(len(test_dates)),
        "Evaluated OOS months": 0,
        "Theoretical T windows": 0,
        "Validation fallback windows": 0,
        "Safe fallback windows": 0,
    }

    for j, date in enumerate(test_dates):
        pos = model_returns.index.get_loc(date)
        if isinstance(pos, slice):
            pos = pos.start
        if pos < lookback:
            continue

        train = model_returns.iloc[pos - lookback:pos]
        x = train.to_numpy(dtype=float)

        monthly_pos = monthly_realized_returns.index.get_loc(date)
        if isinstance(monthly_pos, slice):
            monthly_pos = monthly_pos.start
        if monthly_pos + 1 >= len(monthly_realized_returns):
            continue

        realized_date = monthly_realized_returns.index[monthly_pos + 1]
        realized_vector = monthly_realized_returns.iloc[monthly_pos + 1].to_numpy(dtype=float)

        mu_hist, sigma_hist = sample_moments(x, mle=True)
        T_theory, b_star, signal_theory = theoretical_horizon(
            mu_hist,
            sigma_hist,
            n_obs=len(x),
            beta=float(beta),
        )
        ratio = float(b_star / len(x))

        if T_theory > 0:
            T = float(T_theory)
            T_source = "Theoretical"
            diagnostics["Theoretical T windows"] += 1
        else:
            vf = _adaptive_validation_fraction(len(x))
            if _can_validation_tune(len(x), len(common_cols), vf):
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

        if T <= 0:
            T = float(safe_fallback_T)
            T_source = "Safe fallback"

        actual_signal = float(np.exp(-float(beta) * float(T)))

        # ------------------------------------------------------------
        # Window-level market / estimation regime diagnostics
        # ------------------------------------------------------------
        # Equal-weight realized market proxy for the NEXT monthly observation.
        market_return = float(np.mean(realized_vector))

        # Cross-sectional dispersion of the next realized monthly asset returns.
        cross_sectional_dispersion = float(np.std(realized_vector, ddof=0))

        # Average asset return estimated from the training window.
        avg_asset_return = float(np.mean(mu_hist))

        # Average per-asset volatility in the training window.
        avg_asset_volatility = float(np.mean(np.sqrt(np.clip(np.diag(sigma_hist), 0.0, None))))

        mu_norm = float(np.linalg.norm(mu_hist))
        sigma_trace = float(np.trace(sigma_hist))
        sigma_condition = float(covariance_condition_number(sigma_hist))

        regime_records.append(
            {
                "Model date": date,
                "Realized date": realized_date,
                "T": float(T),
                "T source": T_source,
                "b*": float(b_star),
                "b*/n": ratio,
                "s_T^2": actual_signal,
                "Market return / month": market_return,
                "Average asset mean": avg_asset_return,
                "Cross-sectional dispersion": cross_sectional_dispersion,
                "Average asset volatility": avg_asset_volatility,
                "Mean-vector norm": mu_norm,
                "Covariance trace": sigma_trace,
                "Covariance condition number": sigma_condition,
            }
        )

        w_hist = compute_weights(
            rule, mu_hist, sigma_hist,
            gamma=float(gamma),
            returns=x,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        )

        mu_g, sigma_g, _, combined_g = diffusion_augmented_moments(
            x,
            m=int(m),
            horizon=float(T),
            beta=float(beta),
            n_steps=int(n_steps),
            seed=int(seed + 1000 + j),
        )
        w_g = compute_weights(
            rule, mu_g, sigma_g,
            gamma=float(gamma),
            returns=combined_g,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
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
        w_n = compute_weights(
            rule, mu_n, sigma_n,
            gamma=float(gamma),
            returns=combined_n,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        )

        weights = {
            "Historical": np.asarray(w_hist, dtype=float),
            "Gaussian Diffusion": np.asarray(w_g, dtype=float),
            "Neural Diffusion": np.asarray(w_n, dtype=float),
        }

        for method, w in weights.items():
            prev = previous_weights[method]
            turnover = np.nan if prev is None else 0.5 * float(np.abs(w - prev).sum())
            previous_weights[method] = w.copy()

            records.append(
                {
                    "Model date": date,
                    "Realized date": realized_date,
                    "Method": method,
                    "Realized monthly return": float(w @ realized_vector),
                    "Turnover": turnover,
                    "T": float(T),
                    "T source": T_source,
                    "b*": float(b_star),
                    "b*/n": ratio,
                    "s_T^2": actual_signal,
                    "Neural best validation DSM": (
                        float(min(neural_result.val_losses))
                        if method == "Neural Diffusion"
                        else np.nan
                    ),
                }
            )

        diagnostics["Evaluated OOS months"] += 1

    detail = pd.DataFrame(records)
    if detail.empty:
        raise ValueError("No monthly OOS windows could be evaluated.")

    summaries = []
    for method in method_names:
        sub = detail[detail["Method"] == method].copy()
        r = sub["Realized monthly return"].astype(float)
        mean = float(r.mean())
        var = float(r.var(ddof=0))
        vol = float(np.sqrt(max(var, 0.0)))
        sharpe = mean / vol if vol > 0 else np.nan
        cer = mean - 0.5 * float(gamma) * var
        wealth = (1.0 + r).cumprod()
        dd = wealth / wealth.cummax() - 1.0

        summaries.append(
            {
                "Method": method,
                "Return / month": mean,
                "Volatility / month": vol,
                "Sharpe / month": sharpe,
                "CER / month": cer,
                "Average turnover": float(sub["Turnover"].dropna().mean())
                if sub["Turnover"].notna().any() else np.nan,
                "Max drawdown": float(dd.min()),
                "OOS months": int(len(sub)),
            }
        )

    summary = pd.DataFrame(summaries)

    # Split OOS performance by how T was selected.
    split_rows = []
    source_groups = [
        ("All windows", None),
        ("Theoretical only", "Theoretical"),
        ("Validation fallback only", "Validation fallback"),
        ("Safe fallback only", "Safe fallback"),
    ]

    for source_label, source_value in source_groups:
        for method in method_names:
            sub = detail[detail["Method"] == method].copy()
            if source_value is not None:
                sub = sub[sub["T source"] == source_value].copy()

            r = sub["Realized monthly return"].astype(float)
            if len(r) == 0:
                split_rows.append({
                    "Window subset": source_label,
                    "Method": method,
                    "Return / month": np.nan,
                    "Volatility / month": np.nan,
                    "Sharpe / month": np.nan,
                    "CER / month": np.nan,
                    "Average turnover": np.nan,
                    "Max drawdown": np.nan,
                    "OOS months": 0,
                })
                continue

            mean = float(r.mean())
            var = float(r.var(ddof=0))
            vol = float(np.sqrt(max(var, 0.0)))
            sharpe = mean / vol if vol > 0 else np.nan
            cer = mean - 0.5 * float(gamma) * var
            wealth_subset = (1.0 + r).cumprod()
            dd = wealth_subset / wealth_subset.cummax() - 1.0

            split_rows.append({
                "Window subset": source_label,
                "Method": method,
                "Return / month": mean,
                "Volatility / month": vol,
                "Sharpe / month": sharpe,
                "CER / month": cer,
                "Average turnover": (
                    float(sub["Turnover"].dropna().mean())
                    if sub["Turnover"].notna().any() else np.nan
                ),
                "Max drawdown": float(dd.min()),
                "OOS months": int(len(sub)),
            })

    split_summary = pd.DataFrame(split_rows)

    pivot = detail.pivot(
        index="Realized date",
        columns="Method",
        values="Realized monthly return",
    ).sort_index()
    wealth = (1.0 + pivot).cumprod()

    regime_detail = pd.DataFrame(regime_records)

    # Aggregate diagnostics by T source to understand which market/data regimes
    # are associated with theoretical feasibility vs fallback behavior.
    regime_rows = []
    regime_groups = [
        ("All windows", None),
        ("Theoretical only", "Theoretical"),
        ("Validation fallback only", "Validation fallback"),
        ("Safe fallback only", "Safe fallback"),
    ]

    metric_columns = [
        "Market return / month",
        "Average asset mean",
        "Cross-sectional dispersion",
        "Average asset volatility",
        "Mean-vector norm",
        "Covariance trace",
        "Covariance condition number",
        "b*",
        "b*/n",
        "T",
    ]

    for label, source in regime_groups:
        sub = regime_detail.copy()
        if source is not None:
            sub = sub[sub["T source"] == source].copy()

        row = {
            "Window subset": label,
            "Windows": int(len(sub)),
        }

        for col in metric_columns:
            row[col] = float(sub[col].mean()) if len(sub) else np.nan

        regime_rows.append(row)

    regime_summary = pd.DataFrame(regime_rows)

    return (
        summary,
        detail,
        wealth,
        diagnostics,
        split_summary,
        regime_summary,
        regime_detail,
    )
