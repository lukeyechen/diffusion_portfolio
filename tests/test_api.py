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
