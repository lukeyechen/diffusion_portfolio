from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import api


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values = rng.normal(0.01, 0.04, size=(130, 5))
    return pd.DataFrame(
        values,
        index=pd.date_range("2014-01-31", periods=130, freq="ME"),
        columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    )


def test_health_contract():
    assert api.health() == {
        "status": "ok",
        "service": "diffusion-portfolio-api",
    }


def test_openapi_contract():
    paths = api.app.openapi()["paths"]
    assert "/health" in paths
    assert "/v1/auth/me" in paths
    assert "/v1/portfolio/recommendation" in paths
    assert "/v1/backtest" in paths
    assert "/v1/backtest/horizon-comparison" in paths
    assert "/v1/diffusion-ols/exposure" in paths


def test_pwa_origin_has_cors_access():
    cors = next(
        middleware
        for middleware in api.app.user_middleware
        if middleware.cls is api.CORSMiddleware
    )
    assert "https://lukeyechen.github.io" in cors.kwargs["allow_origins"]
    assert "Authorization" in cors.kwargs["allow_headers"]


def test_google_auth_allows_only_configured_email(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_IDS", "web-client.apps.googleusercontent.com")
    monkeypatch.setenv(
        "ALLOWED_GOOGLE_EMAILS",
        "first@example.com;second@example.com",
    )
    monkeypatch.setattr(
        api.google_id_token,
        "verify_oauth2_token",
        lambda *args, **kwargs: {
            "sub": "google-account-id",
            "email": "second@example.com",
            "email_verified": True,
        },
    )

    user = api.require_google_user("Bearer signed-google-id-token")
    assert user["email"] == "second@example.com"


def test_google_auth_rejects_other_email(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_IDS", "web-client.apps.googleusercontent.com")
    monkeypatch.setenv("ALLOWED_GOOGLE_EMAILS", "first@example.com")
    monkeypatch.setattr(
        api.google_id_token,
        "verify_oauth2_token",
        lambda *args, **kwargs: {
            "sub": "different-google-account",
            "email": "someone@example.com",
            "email_verified": True,
        },
    )

    with pytest.raises(HTTPException) as error:
        api.require_google_user("Bearer signed-google-id-token")
    assert error.value.status_code == 403


def test_google_auth_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_IDS", "web-client.apps.googleusercontent.com")
    monkeypatch.setenv("ALLOWED_GOOGLE_EMAILS", "first@example.com")

    with pytest.raises(HTTPException) as error:
        api.require_google_user(None)
    assert error.value.status_code == 401


def test_recommendation_contract(monkeypatch):
    returns = _returns()
    monkeypatch.setattr(api, "download_yahoo_returns", lambda *args, **kwargs: returns)

    def fake_recommendation(frame, cfg, **kwargs):
        n_assets = frame.shape[1]
        weights = np.ones(n_assets) / n_assets
        sigma = np.eye(n_assets) * 0.01
        return {
            "T": 0.10,
            "previous_T": 0.05,
            "turnover": 0.02,
            "weights": weights,
            "raw_target": weights,
            "previous_drifted": weights,
            "mu": np.full(n_assets, 0.01),
            "sigma": sigma,
            "data_through": frame.index[-1],
        }

    monkeypatch.setattr(api, "replay_latest_recommendation", fake_recommendation)
    request = api.RecommendationRequest(
        holding_period="1 month",
        start_date=date(2000, 1, 1),
    )
    result = api.portfolio_recommendation(request)

    assert result["data"]["assets"] == ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    assert result["model"]["selected_t"] == 0.10
    assert len(result["weights"]) == 5
    assert np.isclose(
        sum(row["recommended"] for row in result["weights"]),
        1.0,
    )


def test_backtest_contract(monkeypatch):
    returns = _returns()
    monkeypatch.setattr(api, "download_yahoo_returns", lambda *args, **kwargs: returns)

    def fake_backtest(frame, cfg, **kwargs):
        dates = frame.index[-3:]
        strategy = "Turnover-Controlled Exact Diffusion"
        detail = pd.DataFrame(
            {
                "Date": dates,
                "T": [0.0, 0.10, 0.10],
                f"return__{strategy}": [0.02, -0.01, 0.03],
                f"turnover__{strategy}": [0.10, 0.08, 0.12],
            }
        )
        summary = pd.DataFrame([{"Method": strategy, "OOS periods": 3}])
        latest = pd.DataFrame([{"Method": strategy, "AAPL": 0.20}])
        tdiag = pd.DataFrame([{"OOS periods": 3, "Median T": 0.10}])
        return summary, detail, latest, tdiag

    monkeypatch.setattr(api, "run_oos_comparison", fake_backtest)
    request = api.BacktestRequest(
        holding_period="1 month",
        start_date=date(2000, 1, 1),
    )
    result = api.portfolio_backtest(request)

    assert result["strategy"] == "Turnover-Controlled Exact Diffusion"
    assert len(result["wealth"]) == 3
    assert len(result["periods"]) == 3
    assert result["periods"][-1]["selected_t"] == 0.10
    assert result["periods"][0]["trading_cost"] == pytest.approx(0.00025)
    assert result["calendar_year_returns"][0]["year"] == returns.index[-1].year
    assert result["net_metrics"]["Final $10,000"] > 0
    assert result["latest_weights"][0]["AAPL"] == 0.20
    assert result["t_selection"]["latest"] == 0.10
    assert result["t_selection"]["most_frequent"] == 0.10
    assert sum(row["count"] for row in result["t_selection"]["frequency"]) == 3


def test_backtest_horizon_comparison_uses_both_steps_and_horizons(monkeypatch):
    observed = []
    frame = _returns()

    def fake_load(settings):
        return frame, {"periods_per_year": 52 if settings.holding_period == "1 week" else 26,
                       "lookback": settings.lookback}

    def fake_run(returns, cfg, **kwargs):
        observed.append((cfg["periods_per_year"], cfg["lookback"],
                         kwargs["rebalance_alpha"], kwargs["additional_rebalance_alpha"]))
        methods = ["Classical MV + LW", "Exact Diffusion (Best-T)",
                   "Turnover-Controlled Exact Diffusion",
                   "Turnover-Controlled Exact Diffusion (50% step)"]
        summary = pd.DataFrame([
            {"Method": method, "CAGR": 0.4, "Net CAGR 25bps": 0.39,
             "Average turnover": 0.02, "Net Sharpe 25bps": 1.3}
            for method in methods
        ])
        detail = pd.DataFrame({"Date": frame.index[-2:]})
        return summary, detail, None, None

    monkeypatch.setattr(api, "_load_and_configure_returns", fake_load)
    monkeypatch.setattr(api, "run_oos_comparison", fake_run)
    result = api.backtest_horizon_comparison(api.BacktestComparisonRequest(
        holding_period="1 week", lookback=520,
    ))
    assert observed == [(52, 520, 1.0, 0.5), (26, 260, 1.0, 0.5)]
    assert len(result["rows"]) == 8
    assert {row["holding"] for row in result["rows"]} == {"1W", "2W"}
    assert result["rows"][0]["net_25bp"] == pytest.approx(0.39)
    assert result["rows"][3]["method"].endswith("50% step")


def test_diffusion_ols_api_uses_shared_exposure_engine(monkeypatch):
    dates = pd.date_range("2020-01-03", periods=155, freq="W-FRI")
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        {"AAPL": rng.normal(0.002, 0.025, len(dates))}, index=dates
    )
    monkeypatch.setattr(api, "download_yahoo_returns", lambda *args, **kwargs: returns)
    request = api.DiffusionExposureRequest(
        tickers=["aapl"], holding_period="1 week", window=40,
        oos_start=date(2022, 1, 1), fixed_b=1.0,
    )
    output = api.diffusion_ols_exposure(request)
    assert output["data"]["assets"] == ["AAPL"]
    assert output["data"]["evaluation_start"] >= "2022-01-01"
    assert output["recommendation"]["selected_b"] == 1.0
    assert sum(row["Periods"] for row in output["summary"]) == 5 * len(
        {row["date"] for row in output["rolling_forecasts"]}
    )
    assert {row["Method"] for row in output["summary"]} == {
        "Diffusion OLS", "OLS", "Historical mean", "Buy and hold", "Cash"
    }
    assert output["recommendation"]["cash_weight"] + sum(
        row["risky_weight"] for row in output["recommendation"]["assets"]
    ) == pytest.approx(1.0)


def test_feasible_trace_matches_shared_engine_and_excludes_removed_methods(monkeypatch):
    frame = pd.DataFrame(
        np.random.default_rng(9).normal(.002, .02, (90, 5)),
        index=pd.date_range("2020-01-03", periods=90, freq="W-FRI"),
        columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    )
    monkeypatch.setattr(api, "download_yahoo_returns", lambda *args, **kwargs: frame)
    monkeypatch.setattr(api, "get_horizon_preset", lambda period: dict(
        source="weekly", block_size=1, lookback=30, periods_per_year=52,
        inner_folds=2, validation_size=5, min_train_size=10,
    ))
    request = api.FeasibleTraceRequest(
        tickers=list(frame.columns), lookback=30, start_date=date(2020, 1, 1),
        oos_start=frame.index[70].date(), end_date=frame.index[-1].date(),
        cs=[.25, 1], epsilons=[.01, .25], reverse_steps=10,
        synthetic_equivalent_m=10,
    )
    result = api.feasible_trace(request, {})
    winner = max(result["calibration"], key=lambda row: row["Annualized MV excess"])
    assert result["selected_c"] == winner["c"]
    assert result["selected_epsilon"] == winner["ε"]
    assert pd.Timestamp(result["calibration_through"]) < pd.Timestamp(request.oos_start)
    assert {row["Method"] for row in result["summary"]} == {
        "Classical MV", "Trace tuning", "Trace tuning (no TC)", "Old Diffusion (Best-T, no TC)", "Old Portfolio (Best-T + turnover control)"
    }
    assert not {"a cap active", "a", "Effective b = n a"} & result["tuning"][0].keys()
    trace = next(row for row in result["tuning"] if row["Method"] == "Trace tuning")
    assert trace["Trace c"] == result["selected_c"]
    assert trace["Trace ε"] == result["selected_epsilon"]
    assert "/v1/feasible-tuning" in api.app.openapi()["paths"]
    assert api.app.openapi()["paths"]["/v1/feasible-tuning"]["post"]["parameters"][0]["name"] == "authorization"


def test_feasible_trace_defaults(monkeypatch):
    request = api.FeasibleTraceRequest(tickers=["AAPL"])
    today = pd.Timestamp(api._trace_today())
    assert request.oos_start == (today - pd.DateOffset(months=6)).date()
    assert request.end_date == today.date()
