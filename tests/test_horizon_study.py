import numpy as np
import pandas as pd

from core.horizon_study import return_horizon_study


def test_return_horizon_study_accepts_short_horizon_labels():
    rng = np.random.default_rng(123)
    columns = ["A", "B", "C"]
    horizons = {
        "1W": pd.DataFrame(rng.normal(0.001, 0.02, size=(80, 3)), columns=columns),
        "2W": pd.DataFrame(rng.normal(0.002, 0.03, size=(60, 3)), columns=columns),
        "1M": pd.DataFrame(rng.normal(0.004, 0.05, size=(50, 3)), columns=columns),
        "3M": pd.DataFrame(rng.normal(0.010, 0.08, size=(40, 3)), columns=columns),
    }

    out = return_horizon_study(horizons, beta=1.0)

    assert list(out["Horizon"]) == ["1W", "2W", "1M", "3M"]
    assert list(out["n"]) == [80, 60, 50, 40]
    assert (out["N"] == 3).all()
    assert {"b*", "b*/n", "T", "Theoretical feasible"}.issubset(out.columns)
