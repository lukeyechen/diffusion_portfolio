from __future__ import annotations

import pandas as pd

from .monthly_oos import monthly_rebalance_oos_comparison


def multi_horizon_monthly_oos_study(
    horizon_data: dict[str, pd.DataFrame],
    monthly_realized_returns: pd.DataFrame,
    lookback_map: dict[str, int],
    test_periods: int,
    gamma: float,
    rule: str,
    m: int,
    beta: float,
    n_steps: int,
    constraint_mode: str,
    max_long_weight: float,
    max_short_weight: float,
    max_gross_exposure: float,
    neural_epochs: int,
    neural_batch: int,
    neural_learning_rate: float,
    neural_hidden_dim: int,
    neural_validation_fraction: float,
    neural_patience: int,
    neural_min_delta: float,
    seed: int = 42,
):
    """
    Run the same monthly OOS framework for 3M, 6M, and 12M model horizons.
    """
    all_rows = []
    diag_rows = []

    for k, horizon in enumerate(["3M", "6M", "12M"]):
        summary, detail, wealth, diagnostics, split_summary, regime_summary, regime_detail = monthly_rebalance_oos_comparison(
            model_returns=horizon_data[horizon],
            monthly_realized_returns=monthly_realized_returns,
            lookback=int(lookback_map[horizon]),
            test_periods=int(test_periods),
            gamma=float(gamma),
            rule=rule,
            m=int(m),
            beta=float(beta),
            n_steps=int(n_steps),
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
            neural_epochs=int(neural_epochs),
            neural_batch=int(neural_batch),
            neural_learning_rate=float(neural_learning_rate),
            neural_hidden_dim=int(neural_hidden_dim),
            neural_validation_fraction=float(neural_validation_fraction),
            neural_patience=int(neural_patience),
            neural_min_delta=float(neural_min_delta),
            seed=int(seed + 10000 * k),
        )

        summary = summary.copy()
        summary.insert(0, "Model horizon", horizon)
        all_rows.append(summary)

        diag_rows.append(
            {
                "Model horizon": horizon,
                **diagnostics,
            }
        )

    results = pd.concat(all_rows, ignore_index=True)
    diagnostics_df = pd.DataFrame(diag_rows)
    return results, diagnostics_df
