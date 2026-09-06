from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_monthly_returns
from core.diffusion import diffusion_augmented_moments
from core.moments import sample_moments
from core.portfolio_rules import compute_weights
from core.tuning import theoretical_horizon

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
START = "2000-01-01"
END = "2026-09-01"  # last full month before the current partial month
LOOKBACK = 120        # 120 monthly observations = ~10 years
GAMMA = 3.0
M = 500
BETA = 1.0
N_STEPS = 100
SEED = 42
MAX_LONG = 0.40
COST_RATES = [0.0010, 0.0025]


def drifted_weights(w, asset_returns):
    gross = 1.0 + float(np.dot(w, asset_returns))
    if gross <= 0:
        return w.copy()
    return w * (1.0 + asset_returns) / gross


def perf_metrics(r, periods_per_year=12):
    r = np.asarray(r, dtype=float)
    wealth = np.cumprod(1.0 + r)
    total_return = float(wealth[-1] - 1.0)
    cagr = float(wealth[-1] ** (periods_per_year / len(r)) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan
    mean_ann = float(np.mean(r) * periods_per_year)
    sharpe = mean_ann / vol if np.isfinite(vol) and vol > 0 else np.nan
    peak = np.maximum.accumulate(wealth)
    max_dd = float(np.min(wealth / peak - 1.0))
    win_rate = float(np.mean(r > 0))
    return {
        "Total return": total_return,
        "CAGR": cagr,
        "Annualized vol": vol,
        "Annualized Sharpe (rf=0)": sharpe,
        "Max drawdown": max_dd,
        "Positive months": win_rate,
        "Final $10,000": float(10000.0 * wealth[-1]),
    }


def main():
    returns = download_yahoo_monthly_returns(
        TICKERS,
        start=START,
        end=END,
    )
    returns = returns[TICKERS].dropna().copy()

    if len(returns) <= LOOKBACK:
        raise RuntimeError(f"Need more than {LOOKBACK} 1M returns, got {len(returns)}")

    records = []
    prev_post_diff = None
    prev_post_ew = None
    t_zero = 0
    t_positive = 0
    all_weights = []

    for i in range(LOOKBACK, len(returns)):
        train = returns.iloc[i - LOOKBACK:i]
        realized = returns.iloc[i].to_numpy(dtype=float)
        x = train.to_numpy(dtype=float)

        mu_hist, sigma_hist = sample_moments(x, mle=True)
        T, b_star, signal = theoretical_horizon(mu_hist, sigma_hist, n_obs=len(x), beta=BETA)
        if T <= 0:
            t_zero += 1
        else:
            t_positive += 1

        mu_d, sigma_d, _, combined = diffusion_augmented_moments(
            x,
            m=M,
            horizon=float(T),
            beta=BETA,
            n_steps=N_STEPS,
            seed=SEED + i,
        )
        w_diff = compute_weights(
            "Mean-Variance",
            mu_d,
            sigma_d,
            gamma=GAMMA,
            returns=combined,
            constraint_mode="Long-only",
            max_long_weight=MAX_LONG,
            max_short_weight=0.20,
            max_gross_exposure=1.50,
        )
        w_ew = np.ones(len(TICKERS), dtype=float) / len(TICKERS)

        turnover_diff = 0.0 if prev_post_diff is None else 0.5 * float(np.abs(w_diff - prev_post_diff).sum())
        turnover_ew = 0.0 if prev_post_ew is None else 0.5 * float(np.abs(w_ew - prev_post_ew).sum())

        r_diff = float(w_diff @ realized)
        r_ew = float(w_ew @ realized)

        row = {
            "date": str(returns.index[i].date()),
            "diffusion_return": r_diff,
            "equal_weight_return": r_ew,
            "diff_minus_ew": r_diff - r_ew,
            "turnover_diffusion": turnover_diff,
            "turnover_equal_weight": turnover_ew,
            "T": float(T),
            "b_star": float(b_star),
            "b_star_over_n": float(b_star / LOOKBACK),
            "signal": float(signal),
        }
        for j, ticker in enumerate(TICKERS):
            row[f"w_{ticker}"] = float(w_diff[j])
        records.append(row)
        all_weights.append(w_diff.copy())

        prev_post_diff = drifted_weights(w_diff, realized)
        prev_post_ew = drifted_weights(w_ew, realized)

    detail = pd.DataFrame(records)
    gross_summary = pd.DataFrame({
        "Gaussian Diffusion": perf_metrics(detail["diffusion_return"].to_numpy()),
        "20% Equal Weight": perf_metrics(detail["equal_weight_return"].to_numpy()),
    }).T

    print("\n=== 1-MONTH OUT-OF-SAMPLE BACKTEST ===")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Source history requested: {START} to {END}")
    print(f"Common 1M return observations: {len(returns)}")
    print(f"Rolling lookback: {LOOKBACK} monthly observations (~10 years)")
    print(f"OOS months: {len(detail)}")
    print(f"OOS start: {detail['date'].iloc[0]} | OOS end: {detail['date'].iloc[-1]}")
    print(f"Diffusion settings: Mean-Variance, gamma={GAMMA}, long-only, max asset={MAX_LONG:.0%}, M={M}, beta={BETA}, steps={N_STEPS}")
    print(f"Theoretical T positive windows: {t_positive}; T=0 windows: {t_zero}")

    print("\n=== GROSS PERFORMANCE (NO TRADING COSTS) ===")
    display_cols = ["Total return", "CAGR", "Annualized vol", "Annualized Sharpe (rf=0)", "Max drawdown", "Positive months", "Final $10,000"]
    print(gross_summary[display_cols].to_string(float_format=lambda x: f"{x:.6f}"))

    diff_beats = float((detail["diff_minus_ew"] > 0).mean())
    avg_excess = float(detail["diff_minus_ew"].mean())
    cumulative_diff = float(np.prod(1.0 + detail["diffusion_return"]) - 1.0)
    cumulative_ew = float(np.prod(1.0 + detail["equal_weight_return"]) - 1.0)
    print("\n=== DIRECT COMPARISON ===")
    print(f"Diffusion beats equal-weight months: {diff_beats:.2%}")
    print(f"Average diffusion minus equal-weight return per 1M period: {avg_excess:.4%}")
    print(f"Cumulative wealth advantage (return points): {(cumulative_diff - cumulative_ew):.4%}")
    print(f"Average diffusion turnover per month: {detail['turnover_diffusion'].mean():.4f}")
    print(f"Average equal-weight turnover per month: {detail['turnover_equal_weight'].mean():.4f}")

    for cost in COST_RATES:
        diff_net = detail["diffusion_return"].to_numpy() - cost * detail["turnover_diffusion"].to_numpy()
        ew_net = detail["equal_weight_return"].to_numpy() - cost * detail["turnover_equal_weight"].to_numpy()
        net_summary = pd.DataFrame({
            "Gaussian Diffusion": perf_metrics(diff_net),
            "20% Equal Weight": perf_metrics(ew_net),
        }).T
        print(f"\n=== NET PERFORMANCE WITH {cost*10000:.0f} BPS PER UNIT TURNOVER ===")
        print(net_summary[["Total return", "CAGR", "Annualized Sharpe (rf=0)", "Max drawdown", "Final $10,000"]].to_string(float_format=lambda x: f"{x:.6f}"))

    weights_df = pd.DataFrame(all_weights, columns=TICKERS)
    print("\n=== AVERAGE DIFFUSION WEIGHTS OVER OOS WINDOWS ===")
    print(weights_df.mean().to_string(float_format=lambda x: f"{x:.4%}"))

    x_latest = returns.iloc[-LOOKBACK:].to_numpy(dtype=float)
    mu_latest, sigma_latest = sample_moments(x_latest, mle=True)
    T_latest, b_latest, signal_latest = theoretical_horizon(mu_latest, sigma_latest, n_obs=LOOKBACK, beta=BETA)
    mu_d_latest, sigma_d_latest, _, combined_latest = diffusion_augmented_moments(
        x_latest, m=M, horizon=float(T_latest), beta=BETA, n_steps=N_STEPS, seed=SEED + 9999
    )
    w_latest = compute_weights(
        "Mean-Variance", mu_d_latest, sigma_d_latest, gamma=GAMMA, returns=combined_latest,
        constraint_mode="Long-only", max_long_weight=MAX_LONG, max_short_weight=0.20,
        max_gross_exposure=1.50,
    )
    print("\n=== LATEST 1M DIFFUSION WEIGHTS FROM THE SAME RULE ===")
    print(f"T={T_latest:.6f}, b*/n={b_latest/LOOKBACK:.6f}, s_T^2={signal_latest:.6f}")
    for ticker, w in zip(TICKERS, w_latest):
        print(f"{ticker}: {w:.4%}")

    detail.to_csv("one_month_backtest_detail.csv", index=False)
    gross_summary.to_csv("one_month_backtest_summary.csv")


if __name__ == "__main__":
    main()
