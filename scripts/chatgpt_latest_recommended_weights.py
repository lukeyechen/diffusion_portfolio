from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_yahoo_returns
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    aggregate_nonoverlapping,
    get_horizon_preset,
    replay_latest_recommendation,
)


TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
START = "2000-01-01"
GAMMA = 3.0
M = 500
BETA = 1.0
N_STEPS = 100
MAX_LONG = 0.40
TURNOVER_PENALTY = 0.0025
REPLAY_START = "2019-01-01"


def run_one(label: str, weekly: pd.DataFrame) -> dict:
    cfg = get_horizon_preset(label)
    returns = aggregate_nonoverlapping(weekly, int(cfg["block_size"]))
    alpha = 0.50 if label == "1 week" else 1.00

    rec = replay_latest_recommendation(
        returns,
        cfg,
        gamma=GAMMA,
        m=M,
        beta=BETA,
        n_steps=N_STEPS,
        candidate_t=CANDIDATE_T,
        turnover_penalty=TURNOVER_PENALTY,
        rebalance_alpha=alpha,
        max_long_weight=MAX_LONG,
        replay_start=REPLAY_START,
    )

    method = (
        "Best-T Diff + TC25 + 50% Step"
        if label == "1 week"
        else "Best-T Diff + TC25"
    )
    out = {
        "Horizon": label,
        "Method": method,
        "Data through": str(pd.Timestamp(rec["data_through"]).date()),
        "Next selected T": float(rec["T"]),
        "Previous selected T": float(rec["previous_T"]),
        "Turnover": float(rec["turnover"]),
    }
    for ticker, weight in zip(TICKERS, rec["weights"]):
        out[ticker] = float(weight)
    for ticker, weight in zip(TICKERS, rec["raw_target"]):
        out[f"Raw TC25 target {ticker}"] = float(weight)
    for ticker, weight in zip(TICKERS, rec["previous_drifted"]):
        out[f"Previous drifted {ticker}"] = float(weight)
    return out


def main():
    weekly = download_yahoo_returns(
        TICKERS,
        start=START,
        end=None,
        interval="1wk",
    )[TICKERS].dropna()

    df = pd.DataFrame([run_one("1 week", weekly), run_one("2 weeks", weekly)])
    df.to_csv("latest_recommended_weights.csv", index=False)

    print("\n=== LATEST RECOMMENDED WEIGHTS — UPGRADED UI ENGINE ===")
    cols = ["Horizon", "Method", "Data through", "Next selected T", "Turnover", *TICKERS]
    shown = df[cols].copy()
    for ticker in TICKERS:
        shown[ticker] = 100.0 * shown[ticker]
    shown["Turnover"] = 100.0 * shown["Turnover"]
    print(shown.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print("\nWeights and turnover are shown as percentages; each weight row sums to 100%.")

    print("\n=== RAW TC25 TARGETS ===")
    raw_cols = ["Horizon", *[f"Raw TC25 target {t}" for t in TICKERS]]
    raw = df[raw_cols].copy()
    for c in raw_cols[1:]:
        raw[c] = 100.0 * raw[c]
    print(raw.to_string(index=False, float_format=lambda z: f"{z:.4f}"))


if __name__ == "__main__":
    main()
