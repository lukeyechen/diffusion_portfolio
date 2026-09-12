from __future__ import annotations

from datetime import date
import os
import re
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field, field_validator

from config import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_M, DEFAULT_MAX_LONG_WEIGHT
from core.data import download_yahoo_returns
from core.metrics import (
    certainty_equivalent,
    portfolio_mean,
    portfolio_volatility,
    sharpe_ratio,
)
from core.moments import covariance_condition_number, sample_moments
from core.portfolio_rules import compute_weights
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    aggregate_nonoverlapping,
    get_horizon_preset,
    performance_metrics,
    replay_latest_recommendation,
    run_oos_comparison,
)


HoldingPeriod = Literal["1 week", "2 weeks", "1 month", "2 months", "3 months"]

app = FastAPI(
    title="Diffusion Portfolio API",
    version="1.0.0",
    description=(
        "Private JSON API for the same exact-moment, nested Best-T and "
        "turnover-controlled portfolio engine used by the Streamlit app."
    ),
)

_google_token_request = google_requests.Request()


def _environment_list(name: str) -> list[str]:
    """Read private list settings separated by commas or semicolons."""
    return [
        value.strip()
        for value in re.split(r"[;,]", os.getenv(name, ""))
        if value.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_environment_list("ALLOWED_WEB_ORIGINS")
    or ["https://lukeyechen.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _authentication_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_google_user(
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Verify a Google OpenID Connect token and enforce the email allowlist."""
    client_ids = _environment_list("GOOGLE_OAUTH_CLIENT_IDS")
    allowed_emails = {
        email.casefold() for email in _environment_list("ALLOWED_GOOGLE_EMAILS")
    }
    if not client_ids or not allowed_emails:
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured on the server.",
        )

    scheme, separator, token = (authorization or "").partition(" ")
    if separator != " " or scheme.casefold() != "bearer" or not token.strip():
        raise _authentication_error("Sign in with Google to use this endpoint.")

    claims: dict[str, Any] | None = None
    for client_id in client_ids:
        try:
            claims = google_id_token.verify_oauth2_token(
                token.strip(),
                _google_token_request,
                audience=client_id,
            )
            break
        except ValueError:
            continue
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Google token verification is temporarily unavailable.",
            ) from exc

    if claims is None:
        raise _authentication_error("Your Google sign-in has expired or is invalid.")

    email = str(claims.get("email", "")).strip().casefold()
    if claims.get("email_verified") is not True or not email:
        raise _authentication_error("Google did not provide a verified email address.")
    if email not in allowed_emails:
        raise HTTPException(
            status_code=403,
            detail="This Google account is not allowed to use the portfolio API.",
        )
    if not claims.get("sub"):
        raise _authentication_error("Google did not provide a valid account identifier.")

    return claims


class PortfolioSettings(BaseModel):
    tickers: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        min_length=1,
        max_length=50,
    )
    start_date: date = date(2000, 1, 1)
    holding_period: HoldingPeriod = "1 week"
    lookback: int | None = Field(default=None, ge=1)
    gamma: float = Field(default=DEFAULT_GAMMA, gt=0.0, le=100.0)
    synthetic_equivalent_m: int = Field(default=DEFAULT_M, ge=0, le=100_000)
    beta: float = Field(default=DEFAULT_BETA, gt=0.0, le=100.0)
    reverse_steps: int = Field(default=100, ge=10, le=10_000)
    turnover_penalty_bps: float = Field(default=25.0, ge=0.0, le=1_000.0)
    rebalance_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    max_long_weight: float = Field(
        default=DEFAULT_MAX_LONG_WEIGHT,
        gt=0.0,
        le=1.0,
    )

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, values: list[str]) -> list[str]:
        tickers: list[str] = []
        for value in values:
            ticker = str(value).strip().upper()
            if ticker and ticker not in tickers:
                tickers.append(ticker)
        if not tickers:
            raise ValueError("Provide at least one ticker.")
        return tickers


class RecommendationRequest(PortfolioSettings):
    replay_start: date = date(2019, 1, 1)


class BacktestRequest(PortfolioSettings):
    strategy: Literal[
        "Equal Weight",
        "Classical MV",
        "Classical MV + LW",
        "Exact Diffusion (Best-T)",
        "50% Exact Diff + 50% EW",
        "Turnover-Controlled Exact Diffusion",
    ] = "Turnover-Controlled Exact Diffusion"
    oos_start: date = date(2019, 1, 1)
    transaction_cost_bps: float = Field(default=25.0, ge=0.0, le=1_000.0)
    initial_capital: float = Field(default=10_000.0, gt=0.0)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _finite_or_none(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _rebalance_alpha(settings: PortfolioSettings) -> float:
    if settings.rebalance_percent is not None:
        return float(settings.rebalance_percent) / 100.0
    return 0.50 if settings.holding_period == "1 week" else 1.0


def _load_and_configure_returns(
    settings: PortfolioSettings,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    if float(settings.max_long_weight) * len(settings.tickers) < 1.0 - 1e-12:
        required = 1.0 / len(settings.tickers)
        raise ValueError(
            "Maximum weight is infeasible for a fully invested long-only portfolio. "
            f"Use at least {required:.4f} for {len(settings.tickers)} assets."
        )

    cfg = get_horizon_preset(settings.holding_period)
    interval = "1wk" if cfg["source"] == "weekly" else "1mo"
    base = download_yahoo_returns(
        settings.tickers,
        start=settings.start_date.isoformat(),
        end=None,
        interval=interval,
    )
    missing = [ticker for ticker in settings.tickers if ticker not in base.columns]
    if missing:
        raise ValueError(
            "Yahoo Finance returned no complete history for: " + ", ".join(missing)
        )
    base = base.loc[:, settings.tickers]
    returns = aggregate_nonoverlapping(base, int(cfg["block_size"]))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    required_min = int(cfg["min_train_size"]) + int(cfg["inner_folds"]) * int(
        cfg["validation_size"]
    )
    if len(returns) <= required_min:
        raise ValueError(
            f"Not enough {settings.holding_period} observations for nested validation. "
            f"Need more than {required_min}; found {len(returns)}."
        )

    lookback = (
        min(int(cfg["lookback"]), len(returns) - 1)
        if settings.lookback is None
        else int(settings.lookback)
    )
    if lookback < required_min:
        raise ValueError(
            f"lookback must be at least {required_min} for {settings.holding_period}."
        )
    if lookback >= len(returns):
        raise ValueError(
            f"lookback must be smaller than the {len(returns)} available observations."
        )
    cfg["lookback"] = lookback
    return returns, cfg


def _request_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/")
def api_root() -> dict[str, str]:
    return {
        "name": "Diffusion Portfolio API",
        "status": "ok",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "diffusion-portfolio-api"}


@app.get("/v1/auth/me")
def authenticated_user(
    user: dict[str, Any] = Depends(require_google_user),
) -> dict[str, str]:
    return {
        "status": "ok",
        "email": str(user["email"]),
        "name": str(user.get("name", "")),
    }


@app.post("/v1/portfolio/recommendation")
def portfolio_recommendation(
    request: RecommendationRequest,
    _user: dict[str, Any] = Depends(require_google_user),
) -> dict[str, Any]:
    try:
        returns, cfg = _load_and_configure_returns(request)
        alpha = _rebalance_alpha(request)
        rec = replay_latest_recommendation(
            returns,
            cfg,
            gamma=float(request.gamma),
            m=int(request.synthetic_equivalent_m),
            beta=float(request.beta),
            n_steps=int(request.reverse_steps),
            candidate_t=CANDIDATE_T,
            turnover_penalty=float(request.turnover_penalty_bps) / 10_000.0,
            rebalance_alpha=alpha,
            max_long_weight=float(request.max_long_weight),
            replay_start=request.replay_start.isoformat(),
        )

        x = returns.iloc[-int(cfg["lookback"]) :].to_numpy(dtype=float)
        mu_hist, sigma_hist = sample_moments(x, mle=True)
        w_classical = compute_weights(
            "Mean-Variance",
            mu_hist,
            sigma_hist,
            gamma=float(request.gamma),
            returns=x,
            constraint_mode="Long-only",
            max_long_weight=float(request.max_long_weight),
        )
        w_lw = compute_weights(
            "Ledoit-Wolf Mean-Variance",
            mu_hist,
            sigma_hist,
            gamma=float(request.gamma),
            returns=x,
            constraint_mode="Long-only",
            max_long_weight=float(request.max_long_weight),
        )

        assets = list(returns.columns)
        recommended = np.asarray(rec["weights"], dtype=float)
        raw_target = np.asarray(rec["raw_target"], dtype=float)
        previous = np.asarray(rec["previous_drifted"], dtype=float)
        mu_used = np.asarray(rec["mu"], dtype=float)
        sigma_used = np.asarray(rec["sigma"], dtype=float)
        equal_weight = np.ones(len(assets), dtype=float) / len(assets)

        weight_rows = []
        for index, asset in enumerate(assets):
            weight_rows.append(
                {
                    "asset": asset,
                    "previous_drifted": float(previous[index]),
                    "raw_target": float(raw_target[index]),
                    "recommended": float(recommended[index]),
                    "classical_mv": float(np.asarray(w_classical)[index]),
                    "classical_mv_ledoit_wolf": float(np.asarray(w_lw)[index]),
                    "equal_weight": float(equal_weight[index]),
                }
            )

        return {
            "data": {
                "assets": assets,
                "observations": len(returns),
                "periods_per_year": int(cfg["periods_per_year"]),
                "lookback": int(cfg["lookback"]),
                "data_through": pd.Timestamp(rec["data_through"]).date().isoformat(),
            },
            "model": {
                "selected_t": float(rec["T"]),
                "previous_selected_t": _finite_or_none(float(rec["previous_T"])),
                "turnover": float(rec["turnover"]),
                "turnover_penalty_bps": float(request.turnover_penalty_bps),
                "rebalance_percent": 100.0 * alpha,
            },
            "weights": weight_rows,
            "diagnostics": {
                "expected_return_per_period": float(
                    portfolio_mean(recommended, mu_used)
                ),
                "volatility_per_period": float(
                    portfolio_volatility(recommended, sigma_used)
                ),
                "sharpe_per_period": _finite_or_none(
                    sharpe_ratio(recommended, mu_used, sigma_used)
                ),
                "certainty_equivalent_per_period": float(
                    certainty_equivalent(
                        recommended, mu_used, sigma_used, float(request.gamma)
                    )
                ),
                "covariance_condition_number": _finite_or_none(
                    covariance_condition_number(sigma_used)
                ),
            },
            "disclaimer": "Research output only; not individualized investment advice.",
        }
    except ValueError as exc:
        raise _request_error(exc) from exc


@app.post("/v1/backtest")
def portfolio_backtest(
    request: BacktestRequest,
    _user: dict[str, Any] = Depends(require_google_user),
) -> dict[str, Any]:
    try:
        returns, cfg = _load_and_configure_returns(request)
        alpha = _rebalance_alpha(request)
        summary, detail, latest, t_diagnostics = run_oos_comparison(
            returns,
            cfg,
            gamma=float(request.gamma),
            m=int(request.synthetic_equivalent_m),
            beta=float(request.beta),
            n_steps=int(request.reverse_steps),
            candidate_t=CANDIDATE_T,
            turnover_penalty=float(request.turnover_penalty_bps) / 10_000.0,
            rebalance_alpha=alpha,
            max_long_weight=float(request.max_long_weight),
            oos_start=request.oos_start.isoformat(),
            costs=(float(request.transaction_cost_bps) / 10_000.0,),
        )

        gross = detail[f"return__{request.strategy}"].to_numpy(dtype=float)
        turnover = detail[f"turnover__{request.strategy}"].to_numpy(dtype=float)
        cost = float(request.transaction_cost_bps) / 10_000.0
        net = gross - cost * turnover
        ppy = int(cfg["periods_per_year"])
        gross_metrics = performance_metrics(
            gross,
            gamma=float(request.gamma),
            periods_per_year=ppy,
        )
        net_metrics = performance_metrics(
            net,
            gamma=float(request.gamma),
            periods_per_year=ppy,
        )

        gross_wealth = float(request.initial_capital) * np.cumprod(1.0 + gross)
        net_wealth = float(request.initial_capital) * np.cumprod(1.0 + net)
        wealth = [
            {
                "date": pd.Timestamp(value).date().isoformat(),
                "gross": float(gross_wealth[index]),
                "net": float(net_wealth[index]),
            }
            for index, value in enumerate(detail["Date"])
        ]

        periods = [
            {
                "date": pd.Timestamp(value).date().isoformat(),
                "gross_return": float(gross[index]),
                "turnover": float(turnover[index]),
                "trading_cost": float(cost * turnover[index]),
                "net_return": float(net[index]),
                "selected_t": float(detail["T"].iloc[index]),
                "year": int(pd.Timestamp(value).year),
            }
            for index, value in enumerate(detail["Date"])
        ]
        calendar_year_returns = [
            {
                "year": int(year),
                "net_return": float(
                    np.prod(1.0 + group["net_return"].to_numpy(dtype=float))
                    - 1.0
                ),
            }
            for year, group in pd.DataFrame(periods).groupby("year")
        ]

        t_values = detail["T"].to_numpy(dtype=float)
        t_counts = pd.Series(t_values).value_counts().sort_index()
        max_t_count = int(t_counts.max())
        most_frequent_t = float(
            t_counts[t_counts == max_t_count].index.min()
        )
        t_selection = {
            "latest": float(t_values[-1]),
            "most_frequent": most_frequent_t,
            "median": float(np.median(t_values)),
            "mean": float(np.mean(t_values)),
            "positive_fraction": float(np.mean(t_values > 0)),
            "frequency": [
                {
                    "t": float(t_value),
                    "count": int(count),
                    "fraction": float(count / len(t_values)),
                }
                for t_value, count in t_counts.items()
            ],
        }

        return {
            "strategy": request.strategy,
            "holding_period": request.holding_period,
            "transaction_cost_bps": float(request.transaction_cost_bps),
            "initial_capital": float(request.initial_capital),
            "gross_final_value": float(gross_wealth[-1]),
            "net_final_value": float(net_wealth[-1]),
            "gross_metrics": {
                key: _finite_or_none(value) for key, value in gross_metrics.items()
            },
            "net_metrics": {
                key: _finite_or_none(value) for key, value in net_metrics.items()
            },
            "wealth": wealth,
            "periods": periods,
            "calendar_year_returns": calendar_year_returns,
            "all_method_summary": _records(summary),
            "latest_weights": _records(latest),
            "t_diagnostics": _records(t_diagnostics),
            "t_selection": t_selection,
            "disclaimer": "Historical research simulation; past performance does not guarantee future results.",
        }
    except ValueError as exc:
        raise _request_error(exc) from exc
