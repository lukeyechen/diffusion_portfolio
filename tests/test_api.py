from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main


client = TestClient(api_main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "diffusion-portfolio-api"


def test_horizon_config_endpoint():
    response = client.get("/config/horizons")
    assert response.status_code == 200
    body = response.json()
    labels = [row["label"] for row in body["holding_periods"]]
    assert labels == ["1 week", "2 weeks", "1 month", "2 months", "3 months"]
    assert body["validated_presets"]["1 week"]["rebalance_step_pct"] == 50.0
    assert body["validated_presets"]["2 weeks"]["rebalance_step_pct"] == 100.0


def test_portfolio_recommendation_response(monkeypatch):
    dates = pd.date_range("2010-01-01", periods=600, freq="W-FRI")
    returns = pd.DataFrame(
        np.zeros((600, 5)),
        index=dates,
        columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    )

    monkeypatch.setattr(api_main, "_holding_returns", lambda *args, **kwargs: returns)

    fake_rec = {
        "T": 0.50,
        "previous_T": 0.30,
        "data_through": dates[-1],
        "lookback": 520,
        "turnover": 0.01,
        "previous_drifted": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
        "raw_target": np.array([0.18, 0.17, 0.40, 0.13, 0.12]),
        "weights": np.array([0.19, 0.185, 0.30, 0.165, 0.16]),
        "t_results": [
            {
                "T": 0.0,
                "mean_validation_CER": 0.01,
                "std_validation_CER": 0.002,
                "standard_error_CER": 0.001,
            },
            {
                "T": 0.5,
                "mean_validation_CER": 0.02,
                "std_validation_CER": 0.003,
                "standard_error_CER": 0.0015,
            },
        ],
    }
    monkeypatch.setattr(
        api_main,
        "replay_latest_recommendation",
        lambda *args, **kwargs: fake_rec,
    )

    response = client.post(
        "/portfolio/recommendation",
        json={
            "tickers": ["aapl", "msft", "nvda", "googl", "amzn"],
            "holding_period": "1 week",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selected_T"] == 0.5
    assert body["rebalance_step_pct"] == 50.0
    assert body["turnover_penalty_bps"] == 25.0
    assert [row["ticker"] for row in body["weights"]] == [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"
    ]
    assert abs(sum(row["recommended_weight"] for row in body["weights"]) - 1.0) < 1e-12
    selected = [row for row in body["t_validation"] if row["selected"]]
    assert len(selected) == 1
    assert selected[0]["T"] == 0.5


def test_infeasible_cap_returns_422(monkeypatch):
    response = client.post(
        "/portfolio/recommendation",
        json={
            "tickers": ["AAPL", "MSFT"],
            "holding_period": "1 week",
            "max_weight": 0.40,
        },
    )
    assert response.status_code == 422
    assert "infeasible" in response.json()["detail"].lower()
