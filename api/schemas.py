from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


HoldingPeriod = Literal["1 week", "2 weeks", "1 month", "2 months", "3 months"]


class PortfolioRecommendationRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"], min_length=1, max_length=25)
    start_date: str = "2000-01-01"
    holding_period: HoldingPeriod = "1 week"
    gamma: float = Field(default=3.0, gt=0.0, le=50.0)
    max_weight: float = Field(default=0.40, gt=0.0, le=1.0)
    synthetic_equivalent_m: int = Field(default=500, ge=0, le=100000)
    beta: float = Field(default=1.0, gt=0.0, le=100.0)
    reverse_steps: int = Field(default=100, ge=10, le=5000)
    turnover_penalty_bps: float = Field(default=25.0, ge=0.0, le=500.0)
    rebalance_step_pct: float | None = Field(default=None, gt=0.0, le=100.0)
    replay_start: str = "2019-01-01"

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            ticker = str(value).strip().upper()
            if ticker and ticker not in cleaned:
                cleaned.append(ticker)
        if not cleaned:
            raise ValueError("Provide at least one non-empty ticker.")
        return cleaned


class WeightRow(BaseModel):
    ticker: str
    previous_drifted: float
    raw_target: float
    recommended_weight: float


class TValidationRow(BaseModel):
    T: float
    mean_validation_CER: float
    std_validation_CER: float
    standard_error_CER: float | None = None
    selected: bool


class PortfolioRecommendationResponse(BaseModel):
    holding_period: HoldingPeriod
    data_through: str
    observations: int
    lookback: int
    selected_T: float
    previous_T: float | None = None
    turnover: float
    turnover_penalty_bps: float
    rebalance_step_pct: float
    gamma: float
    max_weight: float
    weights: list[WeightRow]
    t_validation: list[TValidationRow]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
