from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    PortfolioRecommendationRequest,
    PortfolioRecommendationResponse,
    TValidationRow,
    WeightRow,
)
from core.data import download_yahoo_returns
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    aggregate_nonoverlapping,
    get_horizon_preset,
    replay_latest_recommendation,
)


API_VERSION = "0.1.0"

app = FastAPI(
    title="Diffusion Portfolio API",
    version=API_VERSION,
    description=(
        "Shared backend for the Streamlit research app and the iPhone client. "
        "The production portfolio endpoint uses nested Best-T, deterministic "
        "diffusion moments, turnover-aware optimization, and optional partial rebalancing."
    ),
)

# Native iOS clients do not require browser CORS, but permissive CORS is useful
# during the prototype stage for local/web clients. Restrict this before a
# public multi-user deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _holding_returns(tickers: list[str], start_date: str, holding_period: str) -> pd.DataFrame:
    cfg = get_horizon_preset(holding_period)
    interval = "1wk" if cfg["source"] == "weekly" else "1mo"
    base = download_yahoo_returns(
        tickers,
        start=start_date,
        end=None,
        interval=interval,
    )
    base = base[tickers].dropna()
    return aggregate_nonoverlapping(base, int(cfg["block_size"]))


def _finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", service="diffusion-portfolio-api", version=API_VERSION)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="diffusion-portfolio-api", version=API_VERSION)


@app.get("/config/horizons")
def horizons() -> dict:
    labels = ["1 week", "2 weeks", "1 month", "2 months", "3 months"]
    return {
        "holding_periods": [
            {"label": label, **get_horizon_preset(label)}
            for label in labels
        ],
        "candidate_T": list(CANDIDATE_T),
        "validated_presets": {
            "1 week": {"turnover_penalty_bps": 25.0, "rebalance_step_pct": 50.0},
            "2 weeks": {"turnover_penalty_bps": 25.0, "rebalance_step_pct": 100.0},
        },
    }


@app.post("/portfolio/recommendation", response_model=PortfolioRecommendationResponse)
def portfolio_recommendation(
    request: PortfolioRecommendationRequest,
) -> PortfolioRecommendationResponse:
    try:
        cfg = get_horizon_preset(request.holding_period)

        if request.max_weight * len(request.tickers) < 1.0 - 1e-12:
            raise ValueError(
                "max_weight is infeasible for a fully invested long-only portfolio: "
                "max_weight * number_of_assets must be at least 1."
            )

        returns = _holding_returns(
            request.tickers,
            request.start_date,
            request.holding_period,
        )

        required_min = (
            int(cfg["min_train_size"])
            + int(cfg["inner_folds"]) * int(cfg["validation_size"])
        )
        if len(returns) <= max(int(cfg["lookback"]), required_min):
            raise ValueError(
                f"Not enough {request.holding_period} history. "
                f"Found {len(returns)} observations; need more than "
                f"{max(int(cfg['lookback']), required_min)}."
            )

        rebalance_pct = request.rebalance_step_pct
        if rebalance_pct is None:
            rebalance_pct = 50.0 if request.holding_period == "1 week" else 100.0

        rec = replay_latest_recommendation(
            returns,
            cfg,
            gamma=float(request.gamma),
            m=int(request.synthetic_equivalent_m),
            beta=float(request.beta),
            n_steps=int(request.reverse_steps),
            candidate_t=list(CANDIDATE_T),
            turnover_penalty=float(request.turnover_penalty_bps) / 10000.0,
            rebalance_alpha=float(rebalance_pct) / 100.0,
            max_long_weight=float(request.max_weight),
            replay_start=request.replay_start,
        )

        previous = np.asarray(rec["previous_drifted"], dtype=float)
        raw = np.asarray(rec["raw_target"], dtype=float)
        final = np.asarray(rec["weights"], dtype=float)

        weight_rows = [
            WeightRow(
                ticker=ticker,
                previous_drifted=float(previous[i]),
                raw_target=float(raw[i]),
                recommended_weight=float(final[i]),
            )
            for i, ticker in enumerate(returns.columns)
        ]

        selected_t = float(rec["T"])
        validation_rows = []
        for row in rec.get("t_results", []):
            t = float(row["T"])
            validation_rows.append(
                TValidationRow(
                    T=t,
                    mean_validation_CER=float(row["mean_validation_CER"]),
                    std_validation_CER=float(row["std_validation_CER"]),
                    standard_error_CER=_finite_or_none(row.get("standard_error_CER")),
                    selected=abs(t - selected_t) <= 1e-12,
                )
            )

        return PortfolioRecommendationResponse(
            holding_period=request.holding_period,
            data_through=str(pd.Timestamp(rec["data_through"]).date()),
            observations=int(len(returns)),
            lookback=int(rec["lookback"]),
            selected_T=selected_t,
            previous_T=_finite_or_none(rec.get("previous_T")),
            turnover=float(rec["turnover"]),
            turnover_penalty_bps=float(request.turnover_penalty_bps),
            rebalance_step_pct=float(rebalance_pct),
            gamma=float(request.gamma),
            max_weight=float(request.max_weight),
            weights=weight_rows,
            t_validation=validation_rows,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
