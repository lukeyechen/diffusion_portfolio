from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo
from dataclasses import replace
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
from core.feasible_tuning import TuningSettings, TradingComparison, trace_calibration, historical_backtest, latest_portfolios
from core.data import download_yahoo_returns
from core.diffusion_exposure import (
    backtest_exposure,
    build_yahoo_exposure_data,
    completed_yahoo_returns,
    latest_exposure,
    prepare_exposure_data,
    summarize_exposure,
)
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


class BacktestComparisonRequest(PortfolioSettings):
    oos_start: date = date(2019, 1, 1)


class DiffusionExposureRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=10)
    start_date: date = date(2000, 1, 1)
    holding_period: HoldingPeriod = "1 week"
    annual_risk_free_percent: float = Field(default=4.0, gt=-100.0, le=100.0)
    window: int = Field(default=120, ge=7, le=2000)
    gamma: float = Field(default=5.0, gt=0.0, le=100.0)
    cap: float = Field(default=0.4, gt=0.0, le=1.0)
    cost_bps: float = Field(default=10.0, ge=0.0, le=500.0)
    response_scale: float = Field(default=1.0, gt=0.0)
    tuning_mode: Literal["Fixed b", "Past-only validation"] = "Fixed b"
    fixed_b: float = Field(default=1.0, ge=0.0)
    b_grid: list[float] = Field(
        default_factory=lambda: [50, 60, 70, 89, 90, 91, 92, 93, 94, 95]
    )
    validation: int = Field(default=24, ge=1)
    oos_start: date = date(2019, 1, 1)

    @field_validator("tickers")
    @classmethod
    def normalize_exposure_tickers(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(str(value).strip().upper() for value in values))
        if not normalized or any(not value for value in normalized):
            raise ValueError("Enter at least one valid ticker.")
        return normalized


@app.post("/v1/diffusion-ols/exposure")
def diffusion_ols_exposure(
    request: DiffusionExposureRequest,
    _user: dict[str, Any] = Depends(require_google_user),
) -> dict[str, Any]:
    """Use the same Yahoo preparation and exposure engine as the Streamlit tab."""
    try:
        preset = get_horizon_preset(request.holding_period)
        ppy = int(preset["periods_per_year"])
        interval = "1wk" if preset["source"] == "weekly" else "1mo"
        downloaded = download_yahoo_returns(
            request.tickers,
            start=request.start_date.isoformat(),
            end=None,
            interval=interval,
        )
        downloaded = completed_yahoo_returns(downloaded, source=str(preset["source"]))
        holding_returns = aggregate_nonoverlapping(
            downloaded, int(preset["block_size"])
        )
        risk_free = (1 + request.annual_risk_free_percent / 100) ** (1 / ppy) - 1
        raw_frame, current = build_yahoo_exposure_data(
            holding_returns, risk_free_return=risk_free
        )
        frame, return_columns, _ = prepare_exposure_data(raw_frame, request.tickers)
        predictor_columns = [
            column
            for asset in request.tickers
            for column in (f"x_{asset}_lag1", f"x_{asset}_mean3", f"x_{asset}_vol3")
        ]
        if request.window <= len(predictor_columns) + 2 or len(frame) <= request.window:
            raise ValueError("The estimation window needs more observations and predictors.")
        if request.tuning_mode == "Fixed b":
            if request.fixed_b >= request.window:
                raise ValueError("Fixed b must be smaller than the window.")
            grid = None
        else:
            smallest_inner = request.window - request.validation
            if smallest_inner <= len(predictor_columns) + 2:
                raise ValueError("Reduce inner validation periods or increase the window.")
            if not request.b_grid or any(
                not np.isfinite(b) or b < 0 or b >= smallest_inner
                for b in request.b_grid
            ):
                raise ValueError(f"Each candidate b must satisfy 0 <= b < {smallest_inner}.")
            grid = request.b_grid

        recommendation = latest_exposure(
            frame, return_columns, predictor_columns,
            current[predictor_columns].to_numpy(dtype=float),
            window=request.window, b=request.fixed_b, b_grid=grid,
            validation=request.validation, response_scale=request.response_scale,
            gamma=request.gamma, cap=request.cap,
        )
        results, forecasts = backtest_exposure(
            frame, return_columns, predictor_columns,
            window=request.window, b=request.fixed_b, b_grid=grid,
            validation=request.validation, response_scale=request.response_scale,
            gamma=request.gamma, cap=request.cap, cost_bps=request.cost_bps,
            oos_start=request.oos_start.isoformat(),
        )
        summary = summarize_exposure(
            results, forecasts, periods_per_year=ppy, gamma=request.gamma
        )
        weights = recommendation["weights"]["Diffusion OLS"]
        predicted = recommendation["forecasts"]["Diffusion OLS"]
        variance = np.diag(recommendation["covariance"])
        return {
            "data": {
                "assets": request.tickers,
                "observations": len(frame),
                "predictors": len(predictor_columns),
                "periods_per_year": ppy,
                "data_through": pd.Timestamp(frame["date"].iloc[-1]).date().isoformat(),
                "evaluation_start": pd.Timestamp(results["date"].min()).date().isoformat(),
                "evaluation_end": pd.Timestamp(results["date"].max()).date().isoformat(),
            },
            "recommendation": {
                "selected_b": float(recommendation["selected_b"]),
                "a": float(recommendation["a"]),
                "cash_weight": max(0.0, 1.0 - float(np.sum(weights))),
                "assets": [
                    {
                        "asset": asset,
                        "forecast_excess_return": float(predicted[i]),
                        "estimated_variance": float(variance[i]),
                        "risky_weight": float(weights[i]),
                    }
                    for i, asset in enumerate(request.tickers)
                ],
            },
            "summary": _records(summary),
            "rolling_results": _records(results),
            "rolling_forecasts": _records(forecasts),
        }
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        raise _request_error(exc) from exc


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


@app.post("/v1/backtest/horizon-comparison")
def backtest_horizon_comparison(
    request: BacktestComparisonRequest,
    _user: dict[str, Any] = Depends(require_google_user),
) -> dict[str, Any]:
    """Compare 1W/2W methods with a fixed 25 bps realized trading cost."""
    try:
        base_cfg = get_horizon_preset(request.holding_period)
        base_lookback = (
            int(request.lookback)
            if request.lookback is not None
            else int(base_cfg["lookback"])
        )
        rows: list[dict[str, Any]] = []
        dates: list[dict[str, str]] = []
        for label, period in (("1W", "1 week"), ("2W", "2 weeks")):
            period_cfg = get_horizon_preset(period)
            lookback = round(
                base_lookback * int(period_cfg["periods_per_year"])
                / int(base_cfg["periods_per_year"])
            )
            required_min = (
                int(period_cfg["min_train_size"])
                + int(period_cfg["inner_folds"]) * int(period_cfg["validation_size"])
            )
            period_request = request.model_copy(update={
                "holding_period": period,
                "lookback": max(required_min, lookback),
            })
            returns, cfg = _load_and_configure_returns(period_request)
            summary, detail, _, _ = run_oos_comparison(
                returns,
                cfg,
                gamma=float(request.gamma),
                m=int(request.synthetic_equivalent_m),
                beta=float(request.beta),
                n_steps=int(request.reverse_steps),
                candidate_t=CANDIDATE_T,
                turnover_penalty=0.0025,
                rebalance_alpha=1.0,
                additional_rebalance_alpha=0.5,
                max_long_weight=float(request.max_long_weight),
                oos_start=request.oos_start.isoformat(),
                costs=(0.0025,),
            )
            dates.append({
                "holding": label,
                "start": pd.Timestamp(detail["Date"].iloc[0]).date().isoformat(),
                "end": pd.Timestamp(detail["Date"].iloc[-1]).date().isoformat(),
            })
            for name, method in (
                ("MV + LW", "Classical MV + LW"),
                ("Exact Diffusion", "Exact Diffusion (Best-T)"),
                ("Exact Diffusion + TC25", "Turnover-Controlled Exact Diffusion"),
                ("Exact Diffusion + TC25 + 50% step", "Turnover-Controlled Exact Diffusion (50% step)"),
            ):
                result = summary.loc[summary["Method"] == method].iloc[0]
                rows.append({
                    "holding": label,
                    "method": name,
                    "gross": _finite_or_none(result["CAGR"]),
                    "net_25bp": _finite_or_none(result["Net CAGR 25bps"]),
                    "turnover": _finite_or_none(result["Average turnover"]),
                    "sharpe": _finite_or_none(result["Net Sharpe 25bps"]),
                })
        return {"rows": rows, "dates": dates}
    except ValueError as exc:
        raise _request_error(exc) from exc


def _trace_today() -> date:
    return datetime.now(ZoneInfo("America/Havana")).date()


class FeasibleTraceRequest(PortfolioSettings):
    oos_start: date = Field(default_factory=lambda: (pd.Timestamp(_trace_today()) - pd.DateOffset(months=6)).date())
    end_date: date = Field(default_factory=_trace_today)
    annual_risk_free_percent: float = Field(default=0.0, gt=-100.0, le=100.0)
    transaction_cost_bps: float = Field(default=25.0, ge=0.0, le=1000.0)
    cs: list[float] = Field(default_factory=lambda: [.25, .5, 1, 2, 4], min_length=1, max_length=12)
    epsilons: list[float] = Field(default_factory=lambda: [.001, .01, .05, .1, .25], min_length=1, max_length=12)


@app.post("/v1/feasible-tuning")
def feasible_trace(
    request: FeasibleTraceRequest,
    _user: dict[str, Any] = Depends(require_google_user),
) -> dict[str, Any]:
    """Shared automatic Trace calibration for the iPhone PWA and Android."""
    try:
        cfg = get_horizon_preset(request.holding_period)
        interval = "1wk" if cfg["source"] == "weekly" else "1mo"
        base = download_yahoo_returns(request.tickers, start=request.start_date.isoformat(), end=None, interval=interval)
        missing = set(request.tickers) - set(base.columns)
        if missing:
            raise ValueError("Missing Yahoo history: " + ", ".join(sorted(missing)))
        base = completed_yahoo_returns(base.loc[:, request.tickers], source=cfg["source"])
        returns = aggregate_nonoverlapping(base, int(cfg["block_size"]))
        returns = returns.loc[returns.index <= pd.Timestamp(request.end_date)]
        window = request.lookback or int(cfg["lookback"])
        ppy = int(cfg["periods_per_year"])
        rf = (1 + request.annual_risk_free_percent / 100) ** (1 / ppy) - 1
        trading = TradingComparison(
            inner_folds=int(cfg["inner_folds"]), validation_size=int(cfg["validation_size"]),
            min_train_size=int(cfg["min_train_size"]), cap=request.max_long_weight,
            turnover_penalty=request.turnover_penalty_bps / 10000,
            rebalance_alpha=_rebalance_alpha(request), m=request.synthetic_equivalent_m,
            beta=request.beta, n_steps=request.reverse_steps,
        )
        settings = TuningSettings(gamma=request.gamma, a_max=.95)
        calibrated = trace_calibration(
            returns, settings, request.epsilons, cs=request.cs, window=window,
            calibration_start=request.start_date.isoformat(), evaluation_start=request.oos_start.isoformat(),
            periods_per_year=ppy, rf=rf, cost_bps=request.transaction_cost_bps,
            cap=request.max_long_weight, trading=trading, end_date=request.end_date.isoformat(),
        )
        settings = replace(settings, c=calibrated["selected_c"], epsilon=calibrated["selected_epsilon"])
        history = historical_backtest(
            returns, settings, window=window, oos_start=request.oos_start.isoformat(),
            rf=rf, cost_bps=request.transaction_cost_bps, cap=request.max_long_weight, trading=trading,
        )
        latest, tuning, raw = latest_portfolios(
            returns, settings, history, window=window, rf=rf, cap=request.max_long_weight,
            trading=trading, return_raw=True,
        )
        label = lambda name: "Classical MV" if name == "Classical (main sample)" else name
        allocations = lambda weights: [
            {"Method": label(name), **dict(zip(returns.columns, w)), "Cash": 1-float(w.sum())}
            for name, w in weights.items()
        ]
        summary = []
        for method, group in history.groupby("Method", sort=False):
            net = group["Net return"].to_numpy()
            metrics = performance_metrics(net, gamma=request.gamma, periods_per_year=ppy)
            excess = net-rf
            summary.append({"Method": label(method), **metrics,
                            "Annualized MV excess": ppy*(excess.mean()-request.gamma/2*np.var(excess, ddof=1)),
                            "Average turnover": group["Turnover"].mean()})
        tuning["Trace c"] = np.where(tuning["Method"] == "Trace tuning", settings.c, np.nan)
        tuning["Trace ε"] = np.where(tuning["Method"] == "Trace tuning", settings.epsilon, np.nan)
        return {
            "selected_c": settings.c, "selected_epsilon": settings.epsilon,
            "calibration_through": pd.Timestamp(calibrated["calibration_through"]).date().isoformat(),
            "data_through": pd.Timestamp(returns.index[-1]).date().isoformat(),
            "calibration": _records(calibrated["calibration"]), "summary": _records(pd.DataFrame(summary)),
            "raw_allocations": _records(pd.DataFrame(allocations(raw))),
            "allocations": _records(pd.DataFrame(allocations(latest))),
            "tuning": _records(tuning.drop(columns=["a", "Effective b = n a"], errors="ignore")), "history": _records(history),
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _request_error(exc) from exc
