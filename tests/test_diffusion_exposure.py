import numpy as np
import pandas as pd
import pytest

from core.diffusion_exposure import (
    allocate_exposure,
    backtest_exposure,
    demo_exposure_data,
    fit_diffusion_forecast,
    latest_exposure,
    prepare_exposure_data,
    summarize_exposure,
)


def test_b_zero_recovers_ols_and_exact_terminal_moments():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(100, 2))
    y = 0.3 + x @ np.array([0.2, -0.1]) + rng.normal(size=100) * 0.4
    current = np.array([0.1, -0.3])

    prediction, _ = fit_diffusion_forecast(x, y, current, b=0)
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), y, rcond=None
    )[0]
    np.testing.assert_allclose(prediction[0], np.r_[1, current] @ coefficients)

    a = 0.2
    joint = np.column_stack([x, y])
    mean = joint.mean(axis=0)
    covariance = np.cov(joint, rowvar=False, bias=True)
    c_matrix = (1 - a) * np.eye(3) + a * covariance
    inverse = np.linalg.inv(c_matrix)
    terminal_mean = mean - a * covariance @ inverse @ mean
    terminal_covariance = (
        covariance
        - a**2
        * np.linalg.matrix_power(covariance, 3)
        @ np.linalg.matrix_power(inverse, 2)
    )
    slope = np.linalg.solve(
        terminal_covariance[:2, :2], terminal_covariance[:2, 2]
    )
    expected = terminal_mean[2] + (current - terminal_mean[:2]) @ slope
    prediction, _ = fit_diffusion_forecast(x, y, current, b=a * len(x))
    np.testing.assert_allclose(prediction[0], expected, atol=1e-12)


def test_single_asset_formula_and_long_only_clip():
    covariance = np.array([[0.0016]])
    np.testing.assert_allclose(
        allocate_exposure(np.array([0.006]), covariance, gamma=5), [0.75]
    )
    np.testing.assert_allclose(
        allocate_exposure(np.array([-0.001]), covariance, gamma=5), [0.0]
    )
    np.testing.assert_allclose(
        allocate_exposure(np.array([0.1]), covariance, gamma=5), [1.0]
    )


def test_multiasset_solution_obeys_budget_and_matches_interior_solution():
    forecast = np.array([0.001, 0.002])
    covariance = np.diag([0.002, 0.004])
    expected = np.linalg.solve(covariance, forecast) / 5
    np.testing.assert_allclose(
        allocate_exposure(forecast, covariance, gamma=5), expected, atol=1e-7
    )
    constrained = allocate_exposure(
        np.array([0.1, 0.1]), covariance, gamma=5, cap=0.4
    )
    assert np.all(constrained >= 0)
    assert np.all(constrained <= 0.4 + 1e-9)
    assert constrained.sum() <= 1 + 1e-9


def test_backtest_is_chronological_and_accounts_for_costs_and_cash():
    frame = demo_exposure_data(85)
    return_columns = ["ret_A", "ret_B"]
    settings = dict(window=65, b_grid=[0, 1], validation=4)
    results, forecasts = backtest_exposure(
        frame, return_columns, ["x_1", "x_2"], **settings
    )

    changed = frame.copy()
    changed.loc[72:, return_columns] += 0.2
    changed_results, changed_forecasts = backtest_exposure(
        changed, return_columns, ["x_1", "x_2"], **settings
    )
    cutoff = frame["date"].iloc[72]
    original_early = forecasts.loc[forecasts["date"] <= cutoff, "diffusion_ols"]
    changed_early = changed_forecasts.loc[
        changed_forecasts["date"] <= cutoff, "diffusion_ols"
    ]
    np.testing.assert_allclose(original_early, changed_early)

    weight_columns = ["weight_A", "weight_B", "cash_weight"]
    original_weights = results.loc[results["date"] <= cutoff, weight_columns]
    changed_weights = changed_results.loc[
        changed_results["date"] <= cutoff, weight_columns
    ]
    np.testing.assert_allclose(original_weights, changed_weights)
    np.testing.assert_allclose(results[weight_columns].sum(axis=1), 1.0)

    buy_and_hold = results.loc[results["method"] == "Buy and hold"]
    np.testing.assert_allclose(buy_and_hold["turnover"].iloc[1:], 0, atol=1e-12)
    first = buy_and_hold.iloc[0]
    assert first["net_return"] == pytest.approx(
        (1 - first["cost_fraction"]) * (1 + first["gross_return"]) - 1
    )


def test_latest_exposure_and_summary_have_expected_outputs():
    frame, return_columns, predictor_columns = prepare_exposure_data(
        demo_exposure_data(90), ["A"]
    )
    latest = latest_exposure(
        frame,
        return_columns,
        predictor_columns,
        frame[predictor_columns].iloc[-1].to_numpy(),
        window=65,
        b=1,
        gamma=5,
    )
    forecast = latest["forecasts"]["Diffusion OLS"][0]
    variance = latest["covariance"][0, 0]
    assert latest["weights"]["Diffusion OLS"][0] == pytest.approx(
        np.clip(forecast / (5 * variance), 0, 1)
    )

    results, forecasts = backtest_exposure(
        frame, return_columns, predictor_columns, window=65
    )
    summary = summarize_exposure(results, forecasts, periods_per_year=12, gamma=5)
    assert set(summary["Method"]) == {
        "Diffusion OLS",
        "OLS",
        "Historical mean",
        "Buy and hold",
        "Cash",
    }
    assert np.isfinite(summary.loc[summary["Method"] == "Diffusion OLS", "Forecast MSE"]).all()


def test_data_contract_rejects_reverse_dates_and_missing_values():
    frame = demo_exposure_data(20)
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_exposure_data(frame.iloc[::-1])
    frame.loc[3, "x_1"] = np.nan
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        prepare_exposure_data(frame)
