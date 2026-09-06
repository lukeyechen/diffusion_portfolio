from __future__ import annotations

import numpy as np
import pandas as pd

from .moments import sample_moments
from .tuning import theoretical_horizon


def return_horizon_study(
    horizon_returns: dict[str, pd.DataFrame],
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Summarize theoretical diffusion tuning for any ordered set of return horizons.

    The dictionary insertion order is preserved, so callers can study legacy
    monthly horizons such as 1M/3M/6M/12M or short holding periods such as
    1W/2W/1M/3M with the same theoretical-horizon calculation.
    """
    if not horizon_returns:
        raise ValueError("horizon_returns cannot be empty.")

    rows = []

    for label, frame in horizon_returns.items():
        df = frame.dropna()
        if df.empty:
            rows.append(
                {
                    "Horizon": str(label),
                    "n": 0,
                    "N": int(frame.shape[1]),
                    "N/n": np.nan,
                    "b*": np.nan,
                    "b*/n": np.nan,
                    "T": np.nan,
                    "Theoretical feasible": False,
                }
            )
            continue

        x = df.to_numpy(dtype=float)
        n = len(x)
        N = x.shape[1]

        mu, sigma = sample_moments(x, mle=True)

        try:
            T, b_star, signal = theoretical_horizon(
                mu,
                sigma,
                n_obs=n,
                beta=float(beta),
            )
            ratio = float(b_star / n)
        except Exception:
            T = np.nan
            b_star = np.nan
            signal = np.nan
            ratio = np.nan

        rows.append(
            {
                "Horizon": str(label),
                "n": int(n),
                "N": int(N),
                "N/n": float(N / n) if n > 0 else np.nan,
                "b*": float(b_star) if np.isfinite(b_star) else np.nan,
                "b*/n": ratio,
                "T": float(T) if np.isfinite(T) else np.nan,
                "Theoretical feasible": bool(np.isfinite(ratio) and ratio < 1.0),
            }
        )

    return pd.DataFrame(rows)
