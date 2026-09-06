from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_returns
from core.diffusion import diffusion_augmented_moments
from core.moments import sample_moments
from core.portfolio_rules import compute_weights
from core.tuning import nested_rolling_validation_tuned_horizon

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
START = "2000-01-01"
END = "2026-09-01"
OOS_START = pd.Timestamp("2019-01-01")
GAMMA = 3.0
M = 500
BETA = 1.0
N_STEPS = 100
SEED = 42
MAX_LONG = 0.40
CANDIDATE_T = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
COST_RATES = [0.0010, 0.0025]

# Horizon-specific settings preserve a long estimation history while keeping
# several chronological inner validation forecasts. Sparse long-horizon data
# necessarily use fewer observations than the 2-week case.
HORIZONS = {
    "2 weeks": {
        "periods_per_year": 26,
        "lookback": 260,             # ~10 years of non-overlapping 2-week returns
        "inner_folds": 4,
        "validation_size": 26,       # ~1 year per inner validation block
        "min_train_size": 100,
    },
    "3 months": {
        "periods_per_year": 4,
        "lookback": 48,              # ~12 years
        "inner_folds": 4,
        "validation_size": 4,        # 1 year
        "min_train_size": 30,
    },
    "6 months": {
        "periods_per_year": 2,
        "lookback": 30,              # ~15 years
        "inner_folds": 4,
        "validation_size": 2,        # 1 year
        "min_train_size": 20,
    },
    "1 year": {
        "periods_per_year": 1,
        "lookback": 15,              # ~15 years
        "inner_folds": 2,
        "validation_size": 2,        # 2 years; >=2 needed for covariance/CER
        "min_train_size": 10,
    },
}

METHODS = [
    "20% Equal Weight",
    "Classical MV",
    "Classical MV + LW",
    "Diffusion MV (Nested-T)",
    "50% Diffusion + 50% EW",
]


def _method_key(name: str) -> str:
    return (
        name.replace("%", "pct")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def drifted_weights(w: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross = 1.0 + float(np.dot(w, asset_returns))
    if gross <= 0:
        return w.copy()
    return w * (1.0 + asset_returns) / gross


def perf_metrics(r, *, gamma: float, periods_per_year: int):
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
        "Positive periods": win_rate,
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


def load_horizon_returns(label: str) -> pd.DataFrame:
    if label == "2 weeks":
        weekly = download_yahoo_returns(
            TICKERS,
            start=START,
            end=END,
            interval="1wk",
        )[TICKERS].dropna()
        # Aggregate complete, non-overlapping pairs of weekly returns.
        n_complete = (len(weekly) // 2) * 2
        weekly = weekly.iloc[:n_complete].copy()
        values = weekly.to_numpy(dtype=float)
        two_week = (1.0 + values[0::2]) * (1.0 + values[1::2]) - 1.0
        idx = weekly.index[1::2]
        return pd.DataFrame(two_week, index=idx, columns=TICKERS)

    months = {"3 months": 3, "6 months": 6, "1 year": 12}[label]
    return download_yahoo_returns(
        TICKERS,
        start=START,
        end=END,
        interval="1mo",
        return_horizon_months=months,
    )[TICKERS].dropna()


def run_horizon(label: str, cfg: dict):
    returns = load_horizon_returns(label)
    lookback = int(cfg["lookback"])
    ppy = int(cfg["periods_per_year"])
    inner_folds = int(cfg["inner_folds"])
    validation_size = int(cfg["validation_size"])
    min_train_size = int(cfg["min_train_size"])

    eligible = [
        i for i in range(lookback, len(returns))
        if pd.Timestamp(returns.index[i]).tz_localize(None) >= OOS_START
    ]
    if not eligible:
        raise RuntimeError(f"No OOS observations for {label} after lookback/OOS filters.")

    records = []
    prev_post = {name: None for name in METHODS}
    t_values = []
    latest_weights = {}

    for i in eligible:
        x = returns.iloc[i - lookback:i].to_numpy(dtype=float)
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
            inner_folds=inner_folds,
            validation_size=validation_size,
            min_train_size=min_train_size,
            seed=SEED + i * 100 + ppy,
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
            seed=SEED + i * 1000 + ppy,
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
            "horizon": label,
            "date": str(pd.Timestamp(returns.index[i]).date()),
            "selected_T": float(T_val),
            "best_nested_mean_CER": float(best_nested["mean_validation_CER"]),
            "best_nested_std_CER": float(best_nested["std_validation_CER"]),
        }

        for name, w in weights.items():
            turnover = 0.0 if prev_post[name] is None else 0.5 * float(np.abs(w - prev_post[name]).sum())
            port_r = float(w @ realized)
            key = _method_key(name)
            row[f"return__{key}"] = port_r
            row[f"turnover__{key}"] = turnover
            for j, ticker in enumerate(TICKERS):
                row[f"weight__{key}__{ticker}"] = float(w[j])
            prev_post[name] = drifted_weights(w, realized)

        records.append(row)

    detail = pd.DataFrame(records)
    summary_rows = []
    for name in METHODS:
        key = _method_key(name)
        r = detail[f"return__{key}"].to_numpy(dtype=float)
        metrics = perf_metrics(r, gamma=GAMMA, periods_per_year=ppy)
        metrics["Avg turnover per period"] = float(detail[f"turnover__{key}"].mean())
        metrics.update(
            {
                "Horizon": label,
                "Method": name,
                "OOS periods": int(len(detail)),
                "OOS start": detail["date"].iloc[0],
                "OOS end": detail["date"].iloc[-1],
                "Lookback": lookback,
                "Periods/year": ppy,
            }
        )
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    t_series = pd.Series(t_values, dtype=float)
    t_diag = {
        "Horizon": label,
        "T>0 count": int((t_series > 0).sum()),
        "T=0 count": int((t_series == 0).sum()),
        "T>0 fraction": float((t_series > 0).mean()),
        "Average T": float(t_series.mean()),
        "Median T": float(t_series.median()),
        "OOS periods": int(len(detail)),
        "OOS start": detail["date"].iloc[0],
        "OOS end": detail["date"].iloc[-1],
        "Lookback": lookback,
        "Inner folds": inner_folds,
        "Validation size": validation_size,
        "Min inner train": min_train_size,
    }

    latest = pd.DataFrame(latest_weights, index=TICKERS).T
    latest.insert(0, "Horizon", label)
    latest.insert(1, "Selected T", float(t_values[-1]))

    print(f"\n=== {label.upper()} HOLDING PERIOD ===")
    print(
        f"Observations={len(returns)} | lookback={lookback} | OOS={len(detail)} "
        f"({detail['date'].iloc[0]} to {detail['date'].iloc[-1]})"
    )
    print(
        f"Nested T: folds={inner_folds}, validation_size={validation_size}, "
        f"T>0={(t_series > 0).sum()}/{len(t_series)} ({(t_series > 0).mean():.2%}), "
        f"median T={t_series.median():.2f}"
    )
    show_cols = [
        "Method", "CAGR", "Annualized vol", "Annualized Sharpe (rf=0)",
        "Realized CER (gamma=3)", "Max drawdown", "Avg turnover per period",
        "Final $10,000",
    ]
    print(summary[show_cols].to_string(index=False, float_format=lambda z: f"{z:.6f}"))

    for cost in COST_RATES:
        print(f"\nNet at {cost * 10000:.0f} bps per unit turnover:")
        net_rows = []
        for name in METHODS:
            key = _method_key(name)
            gross_r = detail[f"return__{key}"].to_numpy(dtype=float)
            turnover = detail[f"turnover__{key}"].to_numpy(dtype=float)
            met = perf_metrics(gross_r - cost * turnover, gamma=GAMMA, periods_per_year=ppy)
            net_rows.append({"Method": name, **met})
        net_df = pd.DataFrame(net_rows)
        print(net_df[["Method", "CAGR", "Annualized Sharpe (rf=0)", "Final $10,000"]].to_string(index=False, float_format=lambda z: f"{z:.6f}"))

    return summary, detail, pd.DataFrame([t_diag]), latest


def main():
    all_summary = []
    all_detail = []
    all_t = []
    all_latest = []

    print("=== NESTED-T MULTI-HORIZON HOLDING COMPARISON ===")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"History requested: {START} to {END}; common OOS floor: {OOS_START.date()}")
    print(f"T grid: {CANDIDATE_T}; M={M}; beta={BETA}; SDE steps={N_STEPS}")

    for label, cfg in HORIZONS.items():
        summary, detail, t_diag, latest = run_horizon(label, cfg)
        all_summary.append(summary)
        all_detail.append(detail)
        all_t.append(t_diag)
        all_latest.append(latest.reset_index(names="Method"))

    summary_all = pd.concat(all_summary, ignore_index=True)
    detail_all = pd.concat(all_detail, ignore_index=True, sort=False)
    t_all = pd.concat(all_t, ignore_index=True)
    latest_all = pd.concat(all_latest, ignore_index=True)

    print("\n=== DIFFUSION NESTED-T ACROSS HOLDING HORIZONS ===")
    diff = summary_all[summary_all["Method"] == "Diffusion MV (Nested-T)"].copy()
    ew = summary_all[summary_all["Method"] == "20% Equal Weight"].copy()
    comp = diff.merge(ew, on="Horizon", suffixes=("_Diff", "_EW"))
    comp["CAGR advantage vs EW"] = comp["CAGR_Diff"] - comp["CAGR_EW"]
    comp["Ending-$10k advantage vs EW"] = comp["Final $10,000_Diff"] - comp["Final $10,000_EW"]
    print(
        comp[[
            "Horizon", "OOS periods_Diff", "OOS start_Diff", "OOS end_Diff",
            "CAGR_Diff", "CAGR_EW", "CAGR advantage vs EW",
            "Annualized Sharpe (rf=0)_Diff", "Annualized Sharpe (rf=0)_EW",
            "Final $10,000_Diff", "Final $10,000_EW", "Ending-$10k advantage vs EW",
        ]].to_string(index=False, float_format=lambda z: f"{z:.6f}")
    )

    summary_all.to_csv("nested_holding_summary.csv", index=False)
    detail_all.to_csv("nested_holding_detail.csv", index=False)
    t_all.to_csv("nested_holding_t_diagnostics.csv", index=False)
    latest_all.to_csv("nested_holding_latest_weights.csv", index=False)


if __name__ == "__main__":
    main()
