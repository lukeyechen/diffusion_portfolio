import numpy as np
import pandas as pd

from core.method_comparison import _summary_from_returns


def test_method_summary_metrics():
    r = pd.Series([0.01, -0.005, 0.02, 0.0])
    turnover = pd.Series([np.nan, 0.1, 0.2, 0.1])

    out = _summary_from_returns(r, turnover, gamma=3.0)

    assert out["OOS periods"] == 4
    assert np.isfinite(out["Return / period"])
    assert np.isfinite(out["Volatility / period"])
    assert np.isfinite(out["CER / period"])
    assert np.isfinite(out["Average turnover"])
    assert out["Max drawdown"] <= 0.0
