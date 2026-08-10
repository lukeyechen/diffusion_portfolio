from __future__ import annotations
import numpy as np
import pandas as pd
from .diffusion import diffusion_augmented_moments
from .moments import sample_moments
from .portfolio_rules import compute_weights
from .tuning import constant_beta_horizon, optimal_b_star


def rolling_backtest(
    returns: pd.DataFrame,
    lookback: int = 120,
    gamma: float = 3.0,
    rule: str = "Mean-Variance",
    estimator: str = "Historical",
    m: int = 500,
    beta: float = 1.0,
    n_steps: int = 100,
    seed: int = 42,
    constraint_mode: str = "Long-only",
    max_long_weight: float = 0.40,
    max_short_weight: float = 0.20,
    max_gross_exposure: float = 1.50,
) -> pd.DataFrame:
    if lookback >= len(returns):
        raise ValueError("lookback must be smaller than the data length.")

    rows, cols = [], list(returns.columns)

    for t in range(lookback, len(returns)):
        window = returns.iloc[t - lookback:t]
        x = window.to_numpy()
        mu_hat, sigma_hat = sample_moments(x, mle=True)
        used_returns = x
        horizon = np.nan

        if estimator.lower().startswith("diff"):
            try:
                b_star = optimal_b_star(mu_hat, sigma_hat)
                horizon = constant_beta_horizon(b_star, len(x), beta=beta, clip=True)
            except ValueError:
                horizon = 0.0

            mu_hat, sigma_hat, _, combined = diffusion_augmented_moments(
                x, m=m, horizon=horizon, beta=beta, n_steps=n_steps, seed=seed + t
            )
            used_returns = combined

        w = compute_weights(
            rule,
            mu_hat,
            sigma_hat,
            gamma=gamma,
            returns=used_returns,
            constraint_mode=constraint_mode,
            max_long_weight=max_long_weight,
            max_short_weight=max_short_weight,
            max_gross_exposure=max_gross_exposure,
        )
        realized = float(w @ returns.iloc[t].to_numpy())

        row = {
            "date": returns.index[t],
            "portfolio_return": realized,
            "horizon_T": horizon,
            "gross_exposure": float(np.abs(w).sum()),
            "net_exposure": float(w.sum()),
        }
        for c, weight in zip(cols, w):
            row[f"w_{c}"] = float(weight)
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def backtest_summary(bt: pd.DataFrame, gamma: float = 3.0, periods_per_year: int = 12):
    r = bt["portfolio_return"].dropna()
    mean = float(r.mean())
    var = float(r.var(ddof=0))
    vol = float(np.sqrt(var))
    sharpe = mean / vol if vol > 0 else np.nan
    cer = mean - 0.5 * gamma * var
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return {
        "Mean / period": mean,
        "Volatility / period": vol,
        "Sharpe / period": sharpe,
        "CER / period": cer,
        "Annualized mean (simple)": mean * periods_per_year,
        "Annualized volatility": vol * np.sqrt(periods_per_year),
        "Max drawdown": float(drawdown.min()),
        "Final wealth": float(wealth.iloc[-1]),
    }
