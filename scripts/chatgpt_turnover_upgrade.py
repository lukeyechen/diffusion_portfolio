from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_returns
from core.moments import sample_moments
from core.portfolio_rules import compute_weights, solve_mv_constrained
from core.turnover_upgrade import (
    exact_diffusion_augmented_moments,
    nested_exact_horizon,
    partial_rebalance,
    portfolio_turnover,
    solve_mv_turnover_aware,
)

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
START = "2000-01-01"
END = "2026-09-01"
OOS_START = pd.Timestamp("2019-01-01")
GAMMA = 3.0
M = 500
BETA = 1.0
N_STEPS = 100
MAX_LONG = 0.40
CANDIDATE_T = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
COSTS = [0.0010, 0.0025]

HORIZONS = {
    "1 week": {
        "block": 1,
        "ppy": 52,
        "lookback": 520,
        "validation_size": 52,
        "min_train": 200,
    },
    "2 weeks": {
        "block": 2,
        "ppy": 26,
        "lookback": 260,
        "validation_size": 26,
        "min_train": 100,
    },
}

METHODS = [
    "20% Equal Weight",
    "Classical MV",
    "Classical MV + LW",
    "Exact Diffusion (Best-T)",
    "Exact Diffusion (1SE-T)",
    "50% Exact Diff + 50% EW",
    "Exact Diff + TC10",
    "Exact Diff + TC10 + 50% Step",
    "Exact Diff + TC25 + 50% Step",
]


def key(name: str) -> str:
    return (
        name.replace("%", "pct")
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def drifted_weights(w: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross = 1.0 + float(w @ asset_returns)
    if gross <= 0:
        return np.asarray(w, dtype=float).copy()
    return np.asarray(w, dtype=float) * (1.0 + asset_returns) / gross


def aggregate_nonoverlap(weekly: pd.DataFrame, block: int) -> pd.DataFrame:
    if block == 1:
        return weekly.copy()
    n_complete = (len(weekly) // block) * block
    base = weekly.iloc[:n_complete]
    values = base.to_numpy(dtype=float).reshape(-1, block, len(TICKERS))
    compounded = np.prod(1.0 + values, axis=1) - 1.0
    idx = base.index[block - 1 :: block]
    return pd.DataFrame(compounded, index=idx, columns=TICKERS)


def perf_metrics(r: np.ndarray, ppy: int) -> dict:
    r = np.asarray(r, dtype=float)
    wealth = np.cumprod(1.0 + r)
    cagr = float(wealth[-1] ** (ppy / len(r)) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(ppy))
    mean_ann = float(np.mean(r) * ppy)
    sharpe = mean_ann / vol if vol > 0 else np.nan
    peak = np.maximum.accumulate(wealth)
    max_dd = float(np.min(wealth / peak - 1.0))
    var_ann = float(np.var(r, ddof=1) * ppy)
    cer = mean_ann - 0.5 * GAMMA * var_ann
    return {
        "CAGR": cagr,
        "Annualized vol": vol,
        "Sharpe": sharpe,
        "Realized CER": cer,
        "Max drawdown": max_dd,
        "Final $10,000": float(10000.0 * wealth[-1]),
    }


def mv(mu, sigma):
    return solve_mv_constrained(
        mu,
        sigma,
        gamma=GAMMA,
        mode="Long-only",
        max_long_weight=MAX_LONG,
    )


def run_horizon(label: str, cfg: dict, weekly: pd.DataFrame):
    returns = aggregate_nonoverlap(weekly, int(cfg["block"]))
    ppy = int(cfg["ppy"])
    lookback = int(cfg["lookback"])
    validation_size = int(cfg["validation_size"])
    min_train = int(cfg["min_train"])

    eligible = [
        i for i in range(lookback, len(returns))
        if pd.Timestamp(returns.index[i]).tz_localize(None) >= OOS_START
    ]
    if not eligible:
        raise RuntimeError(f"No OOS observations for {label}.")

    prev_post = {name: None for name in METHODS}
    rows = []
    t_best_series = []
    t_1se_series = []

    for i in eligible:
        x = returns.iloc[i - lookback:i].to_numpy(dtype=float)
        realized = returns.iloc[i].to_numpy(dtype=float)
        mu_hist, sigma_hist = sample_moments(x, mle=True)
        w_ew = np.ones(len(TICKERS)) / len(TICKERS)

        w_classical = mv(mu_hist, sigma_hist)
        w_lw = compute_weights(
            "Ledoit-Wolf Mean-Variance",
            mu_hist,
            sigma_hist,
            gamma=GAMMA,
            returns=x,
            constraint_mode="Long-only",
            max_long_weight=MAX_LONG,
        )

        t_1se, t_results = nested_exact_horizon(
            x,
            gamma=GAMMA,
            m=M,
            beta=BETA,
            n_steps=N_STEPS,
            candidate_T=CANDIDATE_T,
            inner_folds=4,
            validation_size=validation_size,
            min_train_size=min_train,
            selection_rule="one_se",
            constraint_mode="Long-only",
            max_long_weight=MAX_LONG,
        )
        best_rec = max(t_results, key=lambda r: (r["mean_validation_CER"], -r["T"]))
        t_best = float(best_rec["T"])
        t_best_series.append(t_best)
        t_1se_series.append(float(t_1se))

        mu_best, sigma_best, _, _ = exact_diffusion_augmented_moments(
            x, m=M, horizon=t_best, beta=BETA, n_steps=N_STEPS
        )
        if abs(t_best - t_1se) < 1e-15:
            mu_1se, sigma_1se = mu_best, sigma_best
        else:
            mu_1se, sigma_1se, _, _ = exact_diffusion_augmented_moments(
                x, m=M, horizon=float(t_1se), beta=BETA, n_steps=N_STEPS
            )

        w_best = mv(mu_best, sigma_best)
        w_1se = mv(mu_1se, sigma_1se)
        w_blend = 0.5 * w_1se + 0.5 * w_ew

        prev_tc10 = prev_post["Exact Diff + TC10"]
        prev_tc10 = w_ew if prev_tc10 is None else prev_tc10
        w_tc10 = solve_mv_turnover_aware(
            mu_1se,
            sigma_1se,
            previous_weights=prev_tc10,
            turnover_penalty=0.0010,
            gamma=GAMMA,
            mode="Long-only",
            max_long_weight=MAX_LONG,
        )

        prev_tc10_step = prev_post["Exact Diff + TC10 + 50% Step"]
        prev_tc10_step = w_ew if prev_tc10_step is None else prev_tc10_step
        target_tc10_step = solve_mv_turnover_aware(
            mu_1se,
            sigma_1se,
            previous_weights=prev_tc10_step,
            turnover_penalty=0.0010,
            gamma=GAMMA,
            mode="Long-only",
            max_long_weight=MAX_LONG,
        )
        w_tc10_step = partial_rebalance(target_tc10_step, prev_tc10_step, alpha=0.50)

        prev_tc25_step = prev_post["Exact Diff + TC25 + 50% Step"]
        prev_tc25_step = w_ew if prev_tc25_step is None else prev_tc25_step
        target_tc25_step = solve_mv_turnover_aware(
            mu_1se,
            sigma_1se,
            previous_weights=prev_tc25_step,
            turnover_penalty=0.0025,
            gamma=GAMMA,
            mode="Long-only",
            max_long_weight=MAX_LONG,
        )
        w_tc25_step = partial_rebalance(target_tc25_step, prev_tc25_step, alpha=0.50)

        weights = {
            "20% Equal Weight": w_ew,
            "Classical MV": w_classical,
            "Classical MV + LW": w_lw,
            "Exact Diffusion (Best-T)": w_best,
            "Exact Diffusion (1SE-T)": w_1se,
            "50% Exact Diff + 50% EW": w_blend,
            "Exact Diff + TC10": w_tc10,
            "Exact Diff + TC10 + 50% Step": w_tc10_step,
            "Exact Diff + TC25 + 50% Step": w_tc25_step,
        }

        row = {
            "Horizon": label,
            "Date": str(pd.Timestamp(returns.index[i]).date()),
            "T_best": t_best,
            "T_1SE": float(t_1se),
        }
        for name, w in weights.items():
            old = prev_post[name]
            turnover = 0.0 if old is None else portfolio_turnover(w, old)
            r = float(w @ realized)
            k = key(name)
            row[f"return__{k}"] = r
            row[f"turnover__{k}"] = turnover
            for j, ticker in enumerate(TICKERS):
                row[f"weight__{k}__{ticker}"] = float(w[j])
            prev_post[name] = drifted_weights(w, realized)
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for name in METHODS:
        k = key(name)
        gross = detail[f"return__{k}"].to_numpy(dtype=float)
        turnover = detail[f"turnover__{k}"].to_numpy(dtype=float)
        base = perf_metrics(gross, ppy)
        out = {
            "Horizon": label,
            "Method": name,
            "OOS periods": len(detail),
            "OOS start": detail["Date"].iloc[0],
            "OOS end": detail["Date"].iloc[-1],
            "Avg turnover": float(turnover.mean()),
            **base,
        }
        for cost in COSTS:
            net = perf_metrics(gross - cost * turnover, ppy)
            bps = int(cost * 10000)
            out[f"Net CAGR {bps}bps"] = net["CAGR"]
            out[f"Net Sharpe {bps}bps"] = net["Sharpe"]
            out[f"Net Final $10,000 {bps}bps"] = net["Final $10,000"]
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    tdiag = pd.DataFrame([
        {
            "Horizon": label,
            "OOS periods": len(detail),
            "Best-T >0 fraction": float(np.mean(np.asarray(t_best_series) > 0)),
            "Best-T median": float(np.median(t_best_series)),
            "1SE-T >0 fraction": float(np.mean(np.asarray(t_1se_series) > 0)),
            "1SE-T median": float(np.median(t_1se_series)),
            "Best and 1SE same fraction": float(np.mean(np.asarray(t_best_series) == np.asarray(t_1se_series))),
        }
    ])

    print(f"\n=== {label.upper()} ===")
    print(
        summary[[
            "Method", "CAGR", "Sharpe", "Avg turnover",
            "Net CAGR 10bps", "Net CAGR 25bps", "Final $10,000"
        ]].to_string(index=False, float_format=lambda z: f"{z:.6f}")
    )
    print("\nT diagnostics:")
    print(tdiag.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    return summary, detail, tdiag


def main():
    weekly = download_yahoo_returns(
        TICKERS,
        start=START,
        end=END,
        interval="1wk",
    )[TICKERS].dropna()

    all_summary = []
    all_detail = []
    all_tdiag = []
    for label, cfg in HORIZONS.items():
        summary, detail, tdiag = run_horizon(label, cfg, weekly)
        all_summary.append(summary)
        all_detail.append(detail)
        all_tdiag.append(tdiag)

    summary_all = pd.concat(all_summary, ignore_index=True)
    detail_all = pd.concat(all_detail, ignore_index=True, sort=False)
    tdiag_all = pd.concat(all_tdiag, ignore_index=True)

    # Turnover reductions relative to exact 1SE diffusion are easy to inspect.
    base = summary_all[summary_all["Method"] == "Exact Diffusion (1SE-T)"][
        ["Horizon", "Avg turnover"]
    ].rename(columns={"Avg turnover": "Baseline exact turnover"})
    compare = summary_all.merge(base, on="Horizon", how="left")
    compare["Turnover reduction vs exact"] = 1.0 - compare["Avg turnover"] / compare["Baseline exact turnover"]

    print("\n=== UPGRADE SUMMARY ===")
    focus = compare[compare["Method"].isin([
        "Classical MV + LW",
        "Exact Diffusion (1SE-T)",
        "Exact Diff + TC10",
        "Exact Diff + TC10 + 50% Step",
        "Exact Diff + TC25 + 50% Step",
    ])]
    print(
        focus[[
            "Horizon", "Method", "CAGR", "Net CAGR 10bps", "Net CAGR 25bps",
            "Avg turnover", "Turnover reduction vs exact", "Sharpe"
        ]].to_string(index=False, float_format=lambda z: f"{z:.6f}")
    )

    summary_all.to_csv("turnover_upgrade_summary.csv", index=False)
    detail_all.to_csv("turnover_upgrade_detail.csv", index=False)
    tdiag_all.to_csv("turnover_upgrade_t_diagnostics.csv", index=False)
    compare.to_csv("turnover_upgrade_comparison.csv", index=False)


if __name__ == "__main__":
    main()
