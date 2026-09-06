import numpy as np
import pandas as pd

from core.short_horizon_portfolio import (
    aggregate_nonoverlapping,
    drifted_weights,
    get_horizon_preset,
    project_long_only_capped,
)


def test_horizon_presets_include_short_horizons():
    assert get_horizon_preset("1 week")["lookback"] == 520
    assert get_horizon_preset("2 weeks")["lookback"] == 260
    assert get_horizon_preset("1 month")["lookback"] == 120
    assert get_horizon_preset("2 months")["lookback"] == 60
    assert get_horizon_preset("3 months")["lookback"] == 40


def test_aggregate_nonoverlapping_compounds_returns():
    idx = pd.date_range("2025-01-03", periods=4, freq="W-FRI")
    df = pd.DataFrame(
        {
            "A": [0.10, 0.20, -0.10, 0.05],
            "B": [0.00, 0.10, 0.02, 0.03],
        },
        index=idx,
    )
    out = aggregate_nonoverlapping(df, 2)
    assert len(out) == 2
    assert np.isclose(out.iloc[0]["A"], 1.10 * 1.20 - 1.0)
    assert np.isclose(out.iloc[1]["B"], 1.02 * 1.03 - 1.0)


def test_drifted_weights_sum_to_one():
    w = np.array([0.5, 0.3, 0.2])
    r = np.array([0.10, -0.05, 0.02])
    out = drifted_weights(w, r)
    assert np.isclose(out.sum(), 1.0)
    assert np.all(out >= 0)


def test_capped_projection_respects_constraints():
    raw = np.array([0.55, 0.25, 0.10, 0.05, 0.05])
    out = project_long_only_capped(raw, 0.40)
    assert np.isclose(out.sum(), 1.0, atol=1e-10)
    assert np.all(out >= -1e-12)
    assert np.max(out) <= 0.40 + 1e-9


def test_capped_projection_rejects_infeasible_cap():
    with np.testing.assert_raises(ValueError):
        project_long_only_capped(np.array([0.5, 0.5]), 0.40)
