from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_returns
from core.turnover_upgrade import (
    exact_diffusion_augmented_moments,
    nested_exact_horizon,
    partial_rebalance,
    solve_mv_turnover_aware,
)
from scripts.chatgpt_turnover_upgrade import (
    BETA,
    CANDIDATE_T,
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
)


def _best_t(x: np.ndarray, cfg: dict) -> float:
    _, results = nested_exact_horizon(
        x,
        gamma=GAMMA,
        m=M,
        beta=BETA,
        n_steps=N_STEPS,
        candidate_T=CANDIDATE_T,
        inner_folds=4,
        validation_size=int(cfg["validation_size"]),
        min_train_size=int(cfg["min_train"]),
        selection_rule="best",
        constraint_mode="Long-only",
        max_long_weight=MAX_LONG,
    )
    best = max(results, key=lambda r: (r["mean_validation_CER"], -r["T"]))
    return float(best["T"])


def _target_weights(x: np.ndarray, previous_drifted: np.ndarray, cfg: dict, label: str):
    t_best = _best_t(x, cfg)
    mu_d, sigma_d, _, _ = exact_diffusion_augmented_moments(
        x,
        m=M,
        horizon=t_best,
        beta=BETA,
        n_steps=N_STEPS,
    )

    # Recommended variants from the completed turnover experiment:
    # 1 week: Best-T Diffusion + TC25 + 50% partial rebalance.
    # 2 weeks: Best-T Diffusion + TC25 (full step).
    raw_tc25 = solve_mv_turnover_aware(
        mu_d,
        sigma_d,
        previous_weights=previous_drifted,
        turnover_penalty=0.0025,
        gamma=GAMMA,
        mode="Long-only",
        max_long_weight=MAX_LONG,
    )
    if label == "1 week":
        final = partial_rebalance(raw_tc25, previous_drifted, alpha=0.50)
        method = "Best-T Diff + TC25 + 50% Step"
    elif label == "2 weeks":
        final = raw_tc25
        method = "Best-T Diff + TC25"
    else:
        raise ValueError(label)
    return t_best, raw_tc25, final, method


def run_one(label: str, cfg: dict, weekly: pd.DataFrame) -> dict:
    returns = aggregate_nonoverlap(weekly, int(cfg["block"]))
    lookback = int(cfg["lookback"])
    eligible = [
        i for i in range(lookback, len(returns))
        if pd.Timestamp(returns.index[i]).tz_localize(None) >= OOS_START
    ]
    if not eligible:
        raise RuntimeError(f"No OOS observations for {label}.")

    ew = np.ones(len(TICKERS)) / len(TICKERS)
    previous_drifted = ew.copy()
    last_target = None
    last_t = None
    last_method = None

    # Replay the same historical state path so the turnover-aware optimizer's
    # current holdings are correct before computing the next recommendation.
    for i in eligible:
        x = returns.iloc[i - lookback:i].to_numpy(dtype=float)
        realized = returns.iloc[i].to_numpy(dtype=float)
        t_best, _, target, method = _target_weights(x, previous_drifted, cfg, label)
        previous_drifted = drifted_weights(target, realized)
        last_target = target
        last_t = t_best
        last_method = method

    # Recommendation for the NEXT holding period: use every return currently
    # available, but no future return. Yahoo is downloaded through the latest
    # available market close (end=None), which is appropriate when run live.
    x_next = returns.iloc[-lookback:].to_numpy(dtype=float)
    t_next, raw_tc25_next, target_next, method_next = _target_weights(
        x_next, previous_drifted, cfg, label
    )

    last_date = pd.Timestamp(returns.index[-1])
    out = {
        "Horizon": label,
        "Method": method_next,
        "Data through": str(last_date.date()),
        "Next selected T": t_next,
        "Previous selected T": last_t,
    }
    for ticker, weight in zip(TICKERS, target_next):
        out[ticker] = float(weight)
    for ticker, weight in zip(TICKERS, raw_tc25_next):
        out[f"Raw TC25 target {ticker}"] = float(weight)
    if last_target is not None:
        for ticker, weight in zip(TICKERS, last_target):
            out[f"Last rebalance {ticker}"] = float(weight)
    return out


def main():
    weekly = download_yahoo_returns(
        TICKERS,
        start=START,
        end=None,
        interval="1wk",
    )[TICKERS].dropna()

    rows = []
    for label in ("1 week", "2 weeks"):
        rows.append(run_one(label, HORIZONS[label], weekly))

    df = pd.DataFrame(rows)
    df.to_csv("latest_recommended_weights.csv", index=False)

    print("\n=== LATEST RECOMMENDED WEIGHTS ===")
    cols = ["Horizon", "Method", "Data through", "Next selected T", *TICKERS]
    shown = df[cols].copy()
    for ticker in TICKERS:
        shown[ticker] = 100.0 * shown[ticker]
    print(shown.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print("\nWeights shown above are percentages and sum to 100% per row.")

    print("\n=== RAW TC25 TARGETS BEFORE 50% STEP (1W only materially differs) ===")
    raw_cols = ["Horizon", *[f"Raw TC25 target {t}" for t in TICKERS]]
    raw = df[raw_cols].copy()
    for c in raw_cols[1:]:
        raw[c] = 100.0 * raw[c]
    print(raw.to_string(index=False, float_format=lambda z: f"{z:.4f}"))


if __name__ == "__main__":
    main()
