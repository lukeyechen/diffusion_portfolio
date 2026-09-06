from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_returns
from core.moments import sample_moments
from core.portfolio_rules import compute_weights
from core.turnover_upgrade import (
    exact_diffusion_augmented_moments,
    nested_exact_horizon,
    partial_rebalance,
    portfolio_turnover,
    solve_mv_turnover_aware,
)
from scripts.chatgpt_turnover_upgrade import (
    BETA,
    CANDIDATE_T,
    END,
    GAMMA,
    HORIZONS,
    M,
    MAX_LONG,
    N_STEPS,
    OOS_START,
    START,
    TICKERS,
    aggregate_nonoverlap,
    drifted_weights,
    key,
    mv,
    perf_metrics,
)

COSTS = [0.0010, 0.0025]
METHODS = [
    "20% Equal Weight",
    "Classical MV",
    "Classical MV + LW",
    "Exact Diffusion (Best-T)",
    "50% Exact Diff + 50% EW",
    "Best-T Diff + TC10",
    "Best-T Diff + TC10 + 50% Step",
    "Best-T Diff + TC25",
    "Best-T Diff + TC25 + 50% Step",
]


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
    t_series = []

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

        _, t_results = nested_exact_horizon(
            x,
            gamma=GAMMA,
            m=M,
            beta=BETA,
            n_steps=N_STEPS,
            candidate_T=CANDIDATE_T,
            inner_folds=4,
            validation_size=validation_size,
            min_train_size=min_train,
            selection_rule="best",
            constraint_mode="Long-only",
            max_long_weight=MAX_LONG,
        )
        best_rec = max(t_results, key=lambda r: (r["mean_validation_CER"], -r["T"]))
        t_best = float(best_rec["T"])
        t_series.append(t_best)
        mu_d, sigma_d, _, _ = exact_diffusion_augmented_moments(
            x, m=M, horizon=t_best, beta=BETA, n_steps=N_STEPS
        )
        w_diff = mv(mu_d, sigma_d)
        w_blend = 0.5 * w_diff + 0.5 * w_ew

        def prev_for(name: str):
            old = prev_post[name]
            return w_ew if old is None else old

        p10 = prev_for("Best-T Diff + TC10")
        w_tc10 = solve_mv_turnover_aware(
            mu_d, sigma_d, p10, 0.0010,
            gamma=GAMMA, mode="Long-only", max_long_weight=MAX_LONG,
        )

        p10s = prev_for("Best-T Diff + TC10 + 50% Step")
        target10s = solve_mv_turnover_aware(
            mu_d, sigma_d, p10s, 0.0010,
            gamma=GAMMA, mode="Long-only", max_long_weight=MAX_LONG,
        )
        w_tc10_step = partial_rebalance(target10s, p10s, alpha=0.50)

        p25 = prev_for("Best-T Diff + TC25")
        w_tc25 = solve_mv_turnover_aware(
            mu_d, sigma_d, p25, 0.0025,
            gamma=GAMMA, mode="Long-only", max_long_weight=MAX_LONG,
        )

        p25s = prev_for("Best-T Diff + TC25 + 50% Step")
        target25s = solve_mv_turnover_aware(
            mu_d, sigma_d, p25s, 0.0025,
            gamma=GAMMA, mode="Long-only", max_long_weight=MAX_LONG,
        )
        w_tc25_step = partial_rebalance(target25s, p25s, alpha=0.50)

        weights = {
            "20% Equal Weight": w_ew,
            "Classical MV": w_classical,
            "Classical MV + LW": w_lw,
            "Exact Diffusion (Best-T)": w_diff,
            "50% Exact Diff + 50% EW": w_blend,
            "Best-T Diff + TC10": w_tc10,
            "Best-T Diff + TC10 + 50% Step": w_tc10_step,
            "Best-T Diff + TC25": w_tc25,
            "Best-T Diff + TC25 + 50% Step": w_tc25_step,
        }

        row = {
            "Horizon": label,
            "Date": str(pd.Timestamp(returns.index[i]).date()),
            "T_best": t_best,
        }
        for name, w in weights.items():
            old = prev_post[name]
            turnover = 0.0 if old is None else portfolio_turnover(w, old)
            row[f"return__{key(name)}"] = float(w @ realized)
            row[f"turnover__{key(name)}"] = turnover
            prev_post[name] = drifted_weights(w, realized)
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for name in METHODS:
        k = key(name)
        gross = detail[f"return__{k}"].to_numpy(dtype=float)
        turnover = detail[f"turnover__{k}"].to_numpy(dtype=float)
        out = {
            "Horizon": label,
            "Method": name,
            "OOS periods": len(detail),
            "OOS start": detail["Date"].iloc[0],
            "OOS end": detail["Date"].iloc[-1],
            "Avg turnover": float(turnover.mean()),
            **perf_metrics(gross, ppy),
        }
        for cost in COSTS:
            bps = int(cost * 10000)
            net = perf_metrics(gross - cost * turnover, ppy)
            out[f"Net CAGR {bps}bps"] = net["CAGR"]
            out[f"Net Sharpe {bps}bps"] = net["Sharpe"]
            out[f"Net Final $10,000 {bps}bps"] = net["Final $10,000"]
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    t_arr = np.asarray(t_series, dtype=float)
    tdiag = pd.DataFrame([{
        "Horizon": label,
        "OOS periods": len(detail),
        "T>0 fraction": float(np.mean(t_arr > 0)),
        "Median T": float(np.median(t_arr)),
        "Mean T": float(np.mean(t_arr)),
    }])

    print(f"\n=== {label.upper()} BEST-T TURNOVER UPGRADE ===")
    print(summary[[
        "Method", "CAGR", "Sharpe", "Avg turnover", "Net CAGR 10bps", "Net CAGR 25bps"
    ]].to_string(index=False, float_format=lambda z: f"{z:.6f}"))
    print(tdiag.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    return summary, detail, tdiag


def main():
    weekly = download_yahoo_returns(
        TICKERS, start=START, end=END, interval="1wk"
    )[TICKERS].dropna()
    summaries, details, tdiags = [], [], []
    for label, cfg in HORIZONS.items():
        s, d, t = run_horizon(label, cfg, weekly)
        summaries.append(s)
        details.append(d)
        tdiags.append(t)

    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True, sort=False)
    tdiag = pd.concat(tdiags, ignore_index=True)
    base = summary[summary["Method"] == "Exact Diffusion (Best-T)"][
        ["Horizon", "Avg turnover"]
    ].rename(columns={"Avg turnover": "Baseline Best-T turnover"})
    compare = summary.merge(base, on="Horizon", how="left")
    compare["Turnover reduction"] = 1.0 - compare["Avg turnover"] / compare["Baseline Best-T turnover"]

    print("\n=== BEST-T UPGRADE SUMMARY ===")
    focus_names = [
        "Classical MV + LW",
        "Exact Diffusion (Best-T)",
        "Best-T Diff + TC10",
        "Best-T Diff + TC10 + 50% Step",
        "Best-T Diff + TC25",
        "Best-T Diff + TC25 + 50% Step",
    ]
    focus = compare[compare["Method"].isin(focus_names)]
    print(focus[[
        "Horizon", "Method", "CAGR", "Net CAGR 10bps", "Net CAGR 25bps",
        "Avg turnover", "Turnover reduction", "Sharpe"
    ]].to_string(index=False, float_format=lambda z: f"{z:.6f}"))

    summary.to_csv("turnover_bestT_summary.csv", index=False)
    detail.to_csv("turnover_bestT_detail.csv", index=False)
    tdiag.to_csv("turnover_bestT_t_diagnostics.csv", index=False)
    compare.to_csv("turnover_bestT_comparison.csv", index=False)


if __name__ == "__main__":
    main()
