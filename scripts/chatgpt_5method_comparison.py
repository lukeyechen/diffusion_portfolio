from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_monthly_returns
from core.diffusion import diffusion_augmented_moments
from core.moments import sample_moments
from core.portfolio_rules import compute_weights
from core.tuning import nested_rolling_validation_tuned_horizon

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
START = "2000-01-01"
END = "2026-09-01"  # last full month before the current partial month
OOS_START = pd.Timestamp("2016-01-01")
LOOKBACK = 120  # 10 years of monthly observations
GAMMA = 3.0
M = 500
BETA = 1.0
N_STEPS = 100
SEED = 42
MAX_LONG = 0.40
INNER_FOLDS = 4
INNER_VALIDATION_SIZE = 12
CANDIDATE_T = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
COST_RATES = [0.0010, 0.0025]  # 10 and 25 bps per unit turnover

METHODS = [
    "20% Equal Weight",
    "Classical MV",
    "Classical MV + LW",
    "Diffusion MV (Nested-T)",
    "50% Diffusion + 50% EW",
]


def drifted_weights(w: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross = 1.0 + float(np.dot(w, asset_returns))
    if gross <= 0:
        return w.copy()
    return w * (1.0 + asset_returns) / gross


def perf_metrics(r, gamma=GAMMA, periods_per_year=12):
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
    var_ann = float(np.var(r, ddof=1) * periods_per_year) if len(r) > 1 else np.nan
    realized_cer = mean_ann - 0.5 * gamma * var_ann if np.isfinite(var_ann) else np.nan
    return {
        "Total return": total_return,
        "CAGR": cagr,
        "Annualized vol": vol,
        "Annualized Sharpe (rf=0)": sharpe,
        "Realized CER (gamma=3)": realized_cer,
        "Max drawdown": max_dd,
        "Positive months": win_rate,
        "Final $10,000": float(10000.0 * wealth[-1]),
    }


def mv_weights(mu, sigma, returns):
    return compute_weights(
        "Mean-Variance",
        mu,
        sigma,
        gamma=GAMMA,
        returns=returns,
        constraint_mode="Long-only",
        max_long_weight=MAX_LONG,
        max_short_weight=0.20,
        max_gross_exposure=1.50,
    )


def lw_weights(mu, sigma, returns):
    return compute_weights(
        "Ledoit-Wolf Mean-Variance",
        mu,
        sigma,
        gamma=GAMMA,
        returns=returns,
        constraint_mode="Long-only",
        max_long_weight=MAX_LONG,
        max_short_weight=0.20,
        max_gross_exposure=1.50,
    )


def main():
    returns = download_yahoo_monthly_returns(
        TICKERS,
        start=START,
        end=END,
    )
    returns = returns[TICKERS].dropna().copy()

    eligible = [
        i for i in range(LOOKBACK, len(returns))
        if pd.Timestamp(returns.index[i]).tz_localize(None) >= OOS_START
    ]
    if not eligible:
        raise RuntimeError("No OOS observations available after applying the lookback and OOS start date.")

    records = []
    prev_post = {name: None for name in METHODS}
    t_values = []
    latest_weights = {}

    for i in eligible:
        train_df = returns.iloc[i - LOOKBACK:i]
        x = train_df.to_numpy(dtype=float)
        realized = returns.iloc[i].to_numpy(dtype=float)
        mu_hist, sigma_hist = sample_moments(x, mle=True)

        w_ew = np.ones(len(TICKERS), dtype=float) / len(TICKERS)
        w_classical = mv_weights(mu_hist, sigma_hist, x)
        w_lw = lw_weights(mu_hist, sigma_hist, x)

        T_val, nested_results = nested_rolling_validation_tuned_horizon(
            x,
            gamma=GAMMA,
            rule="Mean-Variance",
            m=M,
            beta=BETA,
            n_steps=N_STEPS,
            candidate_T=CANDIDATE_T,
            inner_folds=INNER_FOLDS,
            validation_size=INNER_VALIDATION_SIZE,
            seed=SEED + i * 100,
            constraint_mode="Long-only",
            max_long_weight=MAX_LONG,
            max_short_weight=0.20,
            max_gross_exposure=1.50,
        )
        t_values.append(float(T_val))

        mu_d, sigma_d, _, combined = diffusion_augmented_moments(
            x,
            m=M,
            horizon=float(T_val),
            beta=BETA,
            n_steps=N_STEPS,
            seed=SEED + i * 1000 + 7,
        )
        w_diff = mv_weights(mu_d, sigma_d, combined)
        w_blend = 0.5 * w_diff + 0.5 * w_ew

        weights = {
            "20% Equal Weight": w_ew,
            "Classical MV": w_classical,
            "Classical MV + LW": w_lw,
            "Diffusion MV (Nested-T)": w_diff,
            "50% Diffusion + 50% EW": w_blend,
        }
        latest_weights = {k: v.copy() for k, v in weights.items()}

        best_nested = max(
            nested_results,
            key=lambda r: (r["mean_validation_CER"], -r["T"]),
        )
        row = {
            "date": str(pd.Timestamp(returns.index[i]).date()),
            "selected_T": float(T_val),
            "best_nested_mean_CER": float(best_nested["mean_validation_CER"]),
            "best_nested_std_CER": float(best_nested["std_validation_CER"]),
            "inner_folds": int(INNER_FOLDS),
            "inner_validation_size": int(INNER_VALIDATION_SIZE),
        }

        for name, w in weights.items():
            turnover = 0.0 if prev_post[name] is None else 0.5 * float(np.abs(w - prev_post[name]).sum())
            port_r = float(w @ realized)
            key = name.replace("%", "pct").replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_")
            row[f"return__{key}"] = port_r
            row[f"turnover__{key}"] = turnover
            for j, ticker in enumerate(TICKERS):
                row[f"weight__{key}__{ticker}"] = float(w[j])
            prev_post[name] = drifted_weights(w, realized)

        records.append(row)

    detail = pd.DataFrame(records)

    summary_rows = {}
    turnover_means = {}
    for name in METHODS:
        key = name.replace("%", "pct").replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_")
        r = detail[f"return__{key}"].to_numpy(dtype=float)
        summary_rows[name] = perf_metrics(r)
        turnover_means[name] = float(detail[f"turnover__{key}"].mean())

    summary = pd.DataFrame(summary_rows).T
    summary["Avg monthly turnover"] = pd.Series(turnover_means)

    print("\n=== EXACT 5-METHOD MONTHLY OOS COMPARISON: NESTED ROLLING T ===")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Source history requested: {START} to {END}")
    print(f"Common monthly return observations: {len(returns)}")
    print(f"Rolling estimation window: {LOOKBACK} months (~10 years)")
    print(f"OOS start: {detail['date'].iloc[0]} | OOS end: {detail['date'].iloc[-1]}")
    print(f"OOS months: {len(detail)}")
    print(f"Constraints: long-only, fully invested, max asset weight={MAX_LONG:.0%}, gamma={GAMMA}")
    print(f"Diffusion: nested rolling validation T, M={M}, beta={BETA}, steps={N_STEPS}")
    print(f"Inner rolling folds: {INNER_FOLDS}; validation block per fold: {INNER_VALIDATION_SIZE} months")
    first_inner_train = LOOKBACK - INNER_FOLDS * INNER_VALIDATION_SIZE
    print(f"Inner train sizes: {[first_inner_train + j * INNER_VALIDATION_SIZE for j in range(INNER_FOLDS)]}")
    print(f"T grid: {CANDIDATE_T}")

    print("\n=== GROSS OOS PERFORMANCE ===")
    cols = [
        "Total return", "CAGR", "Annualized vol", "Annualized Sharpe (rf=0)",
        "Realized CER (gamma=3)", "Max drawdown", "Positive months",
        "Avg monthly turnover", "Final $10,000",
    ]
    print(summary[cols].to_string(float_format=lambda z: f"{z:.6f}"))

    t_series = pd.Series(t_values)
    print("\n=== NESTED-ROLLING SELECTED T ===")
    print(f"T>0 windows: {(t_series > 0).sum()} / {len(t_series)} ({(t_series > 0).mean():.2%})")
    print(f"T=0 windows: {(t_series == 0).sum()} / {len(t_series)} ({(t_series == 0).mean():.2%})")
    print(f"Average selected T: {t_series.mean():.6f}")
    print(f"Median selected T: {t_series.median():.6f}")
    print("Selected-T frequency:")
    print(t_series.value_counts().sort_index().to_string())

    def direct_compare(a: str, b: str):
        ka = a.replace("%", "pct").replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_")
        kb = b.replace("%", "pct").replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_")
        ra = detail[f"return__{ka}"].to_numpy(dtype=float)
        rb = detail[f"return__{kb}"].to_numpy(dtype=float)
        print(f"\n=== {a} VS {b} ===")
        print(f"{a} beats {b} months: {np.mean(ra > rb):.2%}")
        print(f"Average monthly excess return: {np.mean(ra - rb):.4%}")
        print(f"CAGR difference: {(summary.loc[a, 'CAGR'] - summary.loc[b, 'CAGR']):.4%}")
        print(f"Realized CER difference: {(summary.loc[a, 'Realized CER (gamma=3)'] - summary.loc[b, 'Realized CER (gamma=3)']):.4%}")
        print(f"Ending-$10k difference: ${(summary.loc[a, 'Final $10,000'] - summary.loc[b, 'Final $10,000']):,.2f}")

    direct_compare("Diffusion MV (Nested-T)", "Classical MV")
    direct_compare("Diffusion MV (Nested-T)", "20% Equal Weight")
    direct_compare("50% Diffusion + 50% EW", "20% Equal Weight")

    for cost in COST_RATES:
        print(f"\n=== NET PERFORMANCE WITH {cost * 10000:.0f} BPS PER UNIT TURNOVER ===")
        net_rows = {}
        for name in METHODS:
            key = name.replace("%", "pct").replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_")
            r = detail[f"return__{key}"].to_numpy(dtype=float)
            to = detail[f"turnover__{key}"].to_numpy(dtype=float)
            net_rows[name] = perf_metrics(r - cost * to)
        net_summary = pd.DataFrame(net_rows).T
        print(net_summary[["CAGR", "Annualized Sharpe (rf=0)", "Realized CER (gamma=3)", "Max drawdown", "Final $10,000"]].to_string(float_format=lambda z: f"{z:.6f}"))

    print("\n=== LATEST PORTFOLIO WEIGHTS ===")
    latest_table = pd.DataFrame(latest_weights, index=TICKERS).T
    print(latest_table.to_string(float_format=lambda z: f"{z:.4%}"))

    summary.to_csv("five_method_comparison_summary.csv")
    detail.to_csv("five_method_comparison_detail.csv", index=False)
    latest_table.to_csv("five_method_latest_weights.csv")


if __name__ == "__main__":
    main()
