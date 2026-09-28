"""Diffusion-regression forecasts and stock/cash exposure backtests.

Each CSV row describes one completed holding period.  ``x_*`` values must be
known before that period starts, ``ret_*`` values are simple total returns, and
``rf`` is the matching per-period risk-free return.  All calculations are
chronological and use only observations preceding the return being forecast.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


METHODS = (
    "Diffusion OLS",
    "OLS",
    "Historical mean",
    "Buy and hold",
    "Cash",
)


def build_yahoo_exposure_data(
    returns: pd.DataFrame,
    *,
    risk_free_return: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build point-in-time regression rows from downloaded asset returns.

    For the return realized at date ``t``, every feature uses returns dated no
    later than ``t-1``.  The returned current vector uses the latest completed
    returns and is therefore aligned to the next, not-yet-realized period.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("Yahoo returns must be a non-empty DataFrame.")
    if len(returns) < 5 or returns.shape[1] < 1:
        raise ValueError("Need at least five downloaded return observations.")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Yahoo return dates must be unique and increasing.")
    clean = returns.copy()
    clean.columns = [str(column).strip().upper() for column in clean.columns]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("Downloaded returns must be finite and greater than -100%.")
    if not np.isfinite(risk_free_return) or risk_free_return <= -1.0:
        raise ValueError("The per-period risk-free return must be finite and greater than -100%.")

    past = clean.shift(1)
    predictors: dict[str, pd.Series] = {}
    current: dict[str, float] = {}
    for asset in clean.columns:
        predictors[f"x_{asset}_lag1"] = past[asset]
        predictors[f"x_{asset}_mean3"] = past[asset].rolling(3).mean()
        predictors[f"x_{asset}_vol3"] = past[asset].rolling(3).std(ddof=0)
        current[f"x_{asset}_lag1"] = float(clean[asset].iloc[-1])
        current[f"x_{asset}_mean3"] = float(clean[asset].iloc[-3:].mean())
        current[f"x_{asset}_vol3"] = float(clean[asset].iloc[-3:].std(ddof=0))

    frame = pd.DataFrame(index=clean.index)
    frame["date"] = pd.to_datetime(clean.index)
    frame["rf"] = float(risk_free_return)
    for asset in clean.columns:
        frame[f"ret_{asset}"] = clean[asset]
    for column, series in predictors.items():
        frame[column] = series
    frame = frame.dropna().reset_index(drop=True)
    return frame, pd.Series(current, dtype=float)


def prepare_exposure_data(
    source: Any,
    assets: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate an exposure CSV/DataFrame and return aligned model columns."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if "date" not in frame or "rf" not in frame:
        raise ValueError("Data must contain date and rf columns.")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("Dates must be unique and strictly increasing.")

    requested = [str(asset).strip() for asset in (assets or ()) if str(asset).strip()]
    return_columns = (
        [f"ret_{asset}" for asset in requested]
        if requested
        else [column for column in frame if column.startswith("ret_")]
    )
    predictor_columns = [column for column in frame if column.startswith("x_")]
    missing = [column for column in return_columns if column not in frame]
    if missing:
        raise ValueError("Missing selected return columns: " + ", ".join(missing))
    if not return_columns or not predictor_columns:
        raise ValueError("Select at least one ret_SYMBOL column and provide x_ predictors.")

    numeric_columns = ["rf", *return_columns, *predictor_columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite values found; align data explicitly without backfilling.")
    if (frame[["rf", *return_columns]].to_numpy(dtype=float) <= -1.0).any():
        raise ValueError("Every simple return must be greater than -100%.")
    return frame, return_columns, predictor_columns


def fit_diffusion_forecast(
    predictors: np.ndarray,
    excess_returns: np.ndarray,
    current_predictors: np.ndarray,
    *,
    b: float = 0.0,
    response_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit exact zero-mean diffusion regressions; ``b=0`` is ordinary OLS.

    The returned covariance is the OLS residual covariance with divisor
    ``n-p-1``.  It estimates the next-period conditional innovation covariance
    used by every allocation method, so forecast comparisons do not also change
    their risk estimator.
    """
    x = np.asarray(predictors, dtype=float)
    y = np.asarray(excess_returns, dtype=float)
    current = np.asarray(current_predictors, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2:
        raise ValueError("Predictors must be a two-dimensional array.")
    n, p = x.shape
    if n <= p + 2 or y.shape[0] != n or current.shape != (p,):
        raise ValueError("Need n > p+2, aligned predictors/returns, and one current p-vector.")
    if not np.isfinite(response_scale) or response_scale <= 0:
        raise ValueError("Response scale must be positive and finite.")
    if not np.isfinite(b) or not 0 <= b < n:
        raise ValueError("Require 0 <= b < the training sample size.")
    if not all(np.isfinite(value).all() for value in (x, y, current)):
        raise ValueError("Model inputs must be finite.")

    design = np.column_stack([np.ones(n), x])
    if np.linalg.matrix_rank(design) != p + 1:
        raise ValueError("Predictors are collinear. Remove redundant predictor columns.")

    ols_coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ ols_coefficients
    residual_covariance = residuals.T @ residuals / (n - p - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-12)
    residual_covariance = (
        eigenvectors * np.maximum(eigenvalues, floor)
    ) @ eigenvectors.T

    forecasts: list[float] = []
    a = float(b) / n
    for asset_index in range(y.shape[1]):
        joint = np.column_stack([x, y[:, asset_index] * response_scale])
        sample_mean = joint.mean(axis=0)
        centered = joint - sample_mean
        sample_covariance = centered.T @ centered / n
        values, vectors = np.linalg.eigh(sample_covariance)
        if values[0] <= max(1e-14, values[-1] * 1e-12):
            raise ValueError(
                "Joint predictor/return covariance is poorly conditioned; "
                "check units, duplicate predictors, or response scaling."
            )

        denominator = 1.0 - a + a * values
        terminal_mean = vectors @ (
            ((1.0 - a) / denominator) * (vectors.T @ sample_mean)
        )
        terminal_values = (
            values
            * (1.0 - a)
            * (1.0 - a + 2.0 * a * values)
            / denominator**2
        )
        terminal_covariance = (vectors * terminal_values) @ vectors.T
        slope = np.linalg.solve(
            terminal_covariance[:p, :p], terminal_covariance[:p, p]
        )
        intercept = terminal_mean[p] - terminal_mean[:p] @ slope
        forecasts.append(float((intercept + current @ slope) / response_scale))

    return np.asarray(forecasts), residual_covariance


def allocate_exposure(
    forecast: np.ndarray,
    covariance: np.ndarray,
    *,
    gamma: float = 5.0,
    cap: float = 1.0,
) -> np.ndarray:
    """Choose long-only risky weights; unallocated wealth remains in cash."""
    mean = np.asarray(forecast, dtype=float).reshape(-1)
    risk = np.asarray(covariance, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("Require gamma > 0 and 0 < cap <= 1.")
    if risk.shape != (len(mean), len(mean)) or not np.isfinite(risk).all():
        raise ValueError("Covariance shape must match the finite forecast vector.")

    if len(mean) == 1:
        variance = float(risk[0, 0])
        if variance <= 0:
            raise ValueError("Estimated excess-return variance must be positive.")
        return np.asarray([np.clip(mean[0] / (gamma * variance), 0.0, cap)])

    scale = max(
        float(np.max(np.abs(mean))),
        float(gamma * np.max(np.abs(risk))),
        1e-8,
    )
    result = minimize(
        lambda weights: (
            0.5 * gamma * weights @ risk @ weights - weights @ mean
        )
        / scale,
        np.zeros(len(mean)),
        jac=lambda weights: (gamma * risk @ weights - mean) / scale,
        bounds=[(0.0, cap)] * len(mean),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda weights: 1.0 - weights.sum(),
                "jac": lambda weights: -np.ones(len(weights)),
            }
        ],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success or result.x.sum() > 1.0 + 1e-7:
        raise RuntimeError(f"Exposure optimization failed: {result.message}")
    return np.clip(result.x, 0.0, cap)


def choose_b(
    predictors: np.ndarray,
    excess_returns: np.ndarray,
    candidates: Iterable[float],
    *,
    validation: int,
    response_scale: float,
) -> float:
    """Select a shared b by expanding, past-only forecast validation."""
    x = np.asarray(predictors, dtype=float)
    y = np.asarray(excess_returns, dtype=float)
    candidate_values = [float(value) for value in candidates]
    start = len(x) - int(validation)
    if validation < 1 or start <= x.shape[1] + 2:
        raise ValueError("Insufficient inner training observations for b validation.")
    if not candidate_values or any(
        not np.isfinite(value) or value < 0 or value >= start
        for value in candidate_values
    ):
        raise ValueError("Each b candidate must satisfy 0 <= b < smallest inner sample size.")

    scores: list[tuple[float, float]] = []
    for candidate in candidate_values:
        errors = []
        for origin in range(start, len(x)):
            prediction, _ = fit_diffusion_forecast(
                x[:origin],
                y[:origin],
                x[origin],
                b=candidate,
                response_scale=response_scale,
            )
            errors.append(float(np.mean((prediction - y[origin]) ** 2)))
        scores.append((float(np.mean(errors)), candidate))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def _validate_backtest_settings(
    frame: pd.DataFrame,
    predictor_columns: list[str],
    window: int,
    gamma: float,
    cap: float,
    cost_bps: float,
) -> None:
    if len(frame) <= window or window <= len(predictor_columns) + 2:
        raise ValueError("Need more rows than the window and window > predictor count + 2.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("Trading cost must be in [0, 10000) bps.")
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("Invalid risk aversion or exposure cap.")


def backtest_exposure(
    frame: pd.DataFrame,
    return_columns: list[str],
    predictor_columns: list[str],
    *,
    window: int = 120,
    b: float = 1.0,
    b_grid: Iterable[float] | None = None,
    validation: int = 24,
    response_scale: float = 1.0,
    gamma: float = 5.0,
    cap: float = 1.0,
    cost_bps: float = 10.0,
    oos_start: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a chronological stock/cash backtest with identical risk estimates.

    Rows before ``oos_start`` remain available to the rolling estimation
    window, but no portfolio returns or turnover are recorded before that date.
    All evaluated strategies begin the selected test interval in cash.
    """
    _validate_backtest_settings(frame, predictor_columns, window, gamma, cap, cost_bps)
    x = frame[predictor_columns].to_numpy(dtype=float)
    risky_returns = frame[return_columns].to_numpy(dtype=float)
    risk_free = frame["rf"].to_numpy(dtype=float)
    excess = risky_returns - risk_free[:, None]
    asset_count = len(return_columns)
    candidate_values = None if b_grid is None else tuple(float(value) for value in b_grid)

    first_origin = int(window)
    if oos_start is not None:
        start = pd.Timestamp(oos_start)
        eligible = np.flatnonzero((frame["date"] >= start).to_numpy())
        if len(eligible) == 0:
            raise ValueError("Backtest start date must not be after the final data date.")
        first_origin = max(first_origin, int(eligible[0]))
    if first_origin >= len(frame):
        raise ValueError("Backtest start leaves no out-of-sample periods.")

    holdings = {
        method: np.r_[np.zeros(asset_count), 1.0]
        for method in METHODS
    }
    records: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []

    for origin in range(first_origin, len(frame)):
        train_x = x[origin - window : origin]
        train_y = excess[origin - window : origin]
        selected_b = (
            choose_b(
                train_x,
                train_y,
                candidate_values,
                validation=validation,
                response_scale=response_scale,
            )
            if candidate_values is not None
            else float(b)
        )
        diffusion, covariance = fit_diffusion_forecast(
            train_x,
            train_y,
            x[origin],
            b=selected_b,
            response_scale=response_scale,
        )
        ols, _ = fit_diffusion_forecast(
            train_x,
            train_y,
            x[origin],
            b=0.0,
            response_scale=response_scale,
        )
        forecasts = {
            "Diffusion OLS": diffusion,
            "OLS": ols,
            "Historical mean": train_y.mean(axis=0),
        }

        for asset_index, return_column in enumerate(return_columns):
            forecast_records.append(
                {
                    "date": frame["date"].iloc[origin],
                    "asset": return_column.removeprefix("ret_"),
                    "actual_excess": excess[origin, asset_index],
                    "diffusion_ols": diffusion[asset_index],
                    "ols": ols[asset_index],
                    "historical_mean": forecasts["Historical mean"][asset_index],
                    "selected_b": selected_b,
                    "a": selected_b / window,
                }
            )

        for method in METHODS:
            pretrade = holdings[method]
            if method in forecasts:
                risky_weights = allocate_exposure(
                    forecasts[method], covariance, gamma=gamma, cap=cap
                )
            elif method == "Cash":
                risky_weights = np.zeros(asset_count)
            elif origin == first_origin:
                risky_weights = np.full(asset_count, min(1.0 / asset_count, cap))
            else:
                risky_weights = pretrade[:asset_count].copy()

            target = np.r_[risky_weights, max(0.0, 1.0 - risky_weights.sum())]
            target /= target.sum()
            turnover = 0.5 * np.abs(target - pretrade).sum()
            fee = cost_bps * 1e-4 * turnover
            gross_asset_values = 1.0 + np.r_[risky_returns[origin], risk_free[origin]]
            gross_growth = float(target @ gross_asset_values)
            net_return = (1.0 - fee) * gross_growth - 1.0
            holdings[method] = target * gross_asset_values / gross_growth

            row: dict[str, Any] = {
                "date": frame["date"].iloc[origin],
                "method": method,
                "gross_return": gross_growth - 1.0,
                "net_return": net_return,
                "net_excess": net_return - risk_free[origin],
                "turnover": turnover,
                "cost_fraction": fee,
                "cash_weight": target[-1],
            }
            row.update(
                {
                    f"weight_{column.removeprefix('ret_')}": target[index]
                    for index, column in enumerate(return_columns)
                }
            )
            records.append(row)

    return pd.DataFrame(records), pd.DataFrame(forecast_records)


def latest_exposure(
    frame: pd.DataFrame,
    return_columns: list[str],
    predictor_columns: list[str],
    current_predictors: np.ndarray,
    *,
    window: int = 120,
    b: float = 1.0,
    b_grid: Iterable[float] | None = None,
    validation: int = 24,
    response_scale: float = 1.0,
    gamma: float = 5.0,
    cap: float = 1.0,
) -> dict[str, Any]:
    """Forecast the next excess return and convert it to current exposure."""
    if len(frame) < window:
        raise ValueError("The latest recommendation needs at least window rows.")
    x = frame[predictor_columns].to_numpy(dtype=float)[-window:]
    risky_returns = frame[return_columns].to_numpy(dtype=float)[-window:]
    risk_free = frame["rf"].to_numpy(dtype=float)[-window:]
    excess = risky_returns - risk_free[:, None]
    candidate_values = None if b_grid is None else tuple(float(value) for value in b_grid)
    selected_b = (
        choose_b(
            x,
            excess,
            candidate_values,
            validation=validation,
            response_scale=response_scale,
        )
        if candidate_values is not None
        else float(b)
    )
    diffusion, covariance = fit_diffusion_forecast(
        x,
        excess,
        current_predictors,
        b=selected_b,
        response_scale=response_scale,
    )
    ols, _ = fit_diffusion_forecast(
        x,
        excess,
        current_predictors,
        b=0.0,
        response_scale=response_scale,
    )
    forecasts = {
        "Diffusion OLS": diffusion,
        "OLS": ols,
        "Historical mean": excess.mean(axis=0),
    }
    weights = {
        method: allocate_exposure(value, covariance, gamma=gamma, cap=cap)
        for method, value in forecasts.items()
    }
    return {
        "selected_b": selected_b,
        "a": selected_b / window,
        "forecasts": forecasts,
        "weights": weights,
        "covariance": covariance,
        "window": window,
    }


def summarize_exposure(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    periods_per_year: int = 12,
    gamma: float = 5.0,
) -> pd.DataFrame:
    """Return net performance and prediction metrics for each method."""
    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")
    forecast_columns = {
        "Diffusion OLS": "diffusion_ols",
        "OLS": "ols",
        "Historical mean": "historical_mean",
    }
    rows = []
    for method, method_results in results.groupby("method", sort=False):
        net = method_results["net_return"].to_numpy(dtype=float)
        excess = method_results["net_excess"].to_numpy(dtype=float)
        wealth = np.r_[1.0, np.cumprod(1.0 + net)]
        standard_deviation = np.std(excess, ddof=1) if len(excess) > 1 else 0.0
        forecast_column = forecast_columns.get(method)
        rows.append(
            {
                "Method": method,
                "Periods": len(net),
                "Total return": wealth[-1] - 1.0,
                "CAGR": wealth[-1] ** (periods_per_year / len(net)) - 1.0,
                "Annualized volatility": (
                    np.std(net, ddof=1) * np.sqrt(periods_per_year)
                    if len(net) > 1
                    else 0.0
                ),
                "Excess Sharpe": (
                    excess.mean() / standard_deviation * np.sqrt(periods_per_year)
                    if standard_deviation > 1e-15
                    else np.nan
                ),
                "Annualized MV excess": periods_per_year
                * (excess.mean() - gamma / 2.0 * standard_deviation**2),
                "Maximum drawdown": np.min(
                    wealth / np.maximum.accumulate(wealth) - 1.0
                ),
                "Average turnover": method_results["turnover"].mean(),
                "Forecast MSE": (
                    np.mean(
                        (forecasts[forecast_column] - forecasts["actual_excess"]) ** 2
                    )
                    if forecast_column is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def demo_exposure_data(n: int = 260, seed: int = 20260928) -> pd.DataFrame:
    """Create a synthetic, clearly fictional three-asset demonstration."""
    generator = np.random.default_rng(seed)
    predictors = generator.normal(size=(n, 2))
    coefficients = np.array([[0.007, 0.004, -0.003], [-0.004, 0.005, 0.002]])
    covariance = np.array(
        [[0.0025, 0.0008, 0.0004], [0.0008, 0.0016, 0.0003], [0.0004, 0.0003, 0.0012]]
    )
    excess = (
        np.array([0.006, 0.004, 0.003])
        + predictors @ coefficients
        + generator.multivariate_normal(np.zeros(3), covariance, n)
    )
    data: dict[str, Any] = {
        "date": pd.date_range("2000-01-01", periods=n, freq="MS"),
        "rf": np.full(n, 0.001),
    }
    data.update(
        {
            f"ret_{name}": excess[:, index] + 0.001
            for index, name in enumerate(("A", "B", "C"))
        }
    )
    data.update({f"x_{index + 1}": predictors[:, index] for index in range(2)})
    return pd.DataFrame(data)
