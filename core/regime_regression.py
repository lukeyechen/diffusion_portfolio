from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _ols_with_hc1(
    y: np.ndarray,
    X: np.ndarray,
    names: list[str],
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """
    OLS with HC1 heteroskedasticity-robust standard errors.

    Returns
    -------
    table : coefficient table
    diagnostics : n, k, R^2
    beta : coefficient vector
    cov : HC1 covariance matrix
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    meat = np.zeros((k, k), dtype=float)
    for i in range(n):
        xi = X[i:i + 1].T
        meat += (resid[i] ** 2) * (xi @ xi.T)

    scale = n / max(n - k, 1)
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    tstat = np.divide(
        beta,
        se,
        out=np.full_like(beta, np.nan),
        where=se > 0,
    )
    dof = max(n - k, 1)
    pval = 2.0 * (1.0 - stats.t.cdf(np.abs(tstat), df=dof))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    table = pd.DataFrame(
        {
            "Variable": names,
            "Coefficient": beta,
            "Robust SE": se,
            "t-stat": tstat,
            "p-value": pval,
        }
    )
    diagnostics = {
        "n": int(n),
        "k": int(k),
        "R²": float(r2),
    }
    return table, diagnostics, beta, cov


def build_relative_performance_dataset(
    monthly_detail: pd.DataFrame,
    regime_detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per OOS month with relative portfolio performance and regime controls.
    """
    perf = monthly_detail.pivot(
        index="Realized date",
        columns="Method",
        values="Realized monthly return",
    ).reset_index()

    required = {"Historical", "Gaussian Diffusion", "Neural Diffusion"}
    missing = required.difference(perf.columns)
    if missing:
        raise ValueError(f"Missing methods in monthly detail: {sorted(missing)}")

    regime = regime_detail.copy()
    cols = [
        "Realized date",
        "T source",
        "b*/n",
        "Market return / month",
        "Cross-sectional dispersion",
        "Average asset volatility",
        "Covariance trace",
        "Covariance condition number",
    ]
    regime = regime[cols].drop_duplicates(subset=["Realized date"])

    df = perf.merge(regime, on="Realized date", how="inner")

    df["Theoretical feasible"] = (df["T source"] == "Theoretical").astype(int)
    df["Neural - Historical"] = df["Neural Diffusion"] - df["Historical"]
    df["Gaussian - Historical"] = df["Gaussian Diffusion"] - df["Historical"]

    # Piecewise variables centered at the theoretical boundary x=1.
    x = df["b*/n"].astype(float)
    df["b*/n centered at 1"] = x - 1.0
    df["Below-boundary indicator"] = (x < 1.0).astype(int)
    df["Below-boundary slope interaction"] = (
        (x - 1.0) * (x < 1.0).astype(int)
    )

    return df.sort_values("Realized date").reset_index(drop=True)


def _prepare_base_controls(df: pd.DataFrame):
    return [
        df["Market return / month"].to_numpy(dtype=float),
        df["Cross-sectional dispersion"].to_numpy(dtype=float),
        df["Covariance trace"].to_numpy(dtype=float),
    ]


def _run_single_spec(
    df: pd.DataFrame,
    outcome: str,
    specification: str,
) -> tuple[pd.DataFrame, dict]:
    base_controls = _prepare_base_controls(df)

    if specification == "Binary feasibility":
        X = np.column_stack(
            [
                np.ones(len(df)),
                df["Theoretical feasible"].to_numpy(dtype=float),
                *base_controls,
            ]
        )
        names = [
            "Intercept",
            "Theoretical feasible",
            "Market return / month",
            "Cross-sectional dispersion",
            "Covariance trace",
        ]

    elif specification == "Continuous b*/n":
        X = np.column_stack(
            [
                np.ones(len(df)),
                df["b*/n"].to_numpy(dtype=float),
                *base_controls,
            ]
        )
        names = [
            "Intercept",
            "b*/n",
            "Market return / month",
            "Cross-sectional dispersion",
            "Covariance trace",
        ]

    elif specification == "Piecewise at 1":
        X = np.column_stack(
            [
                np.ones(len(df)),
                df["b*/n centered at 1"].to_numpy(dtype=float),
                df["Below-boundary indicator"].to_numpy(dtype=float),
                df["Below-boundary slope interaction"].to_numpy(dtype=float),
                *base_controls,
            ]
        )
        names = [
            "Intercept",
            "(b*/n - 1)",
            "I(b*/n < 1)",
            "(b*/n - 1) × I(b*/n < 1)",
            "Market return / month",
            "Cross-sectional dispersion",
            "Covariance trace",
        ]
    else:
        raise ValueError(f"Unknown specification: {specification}")

    table, diagnostics, beta, cov = _ols_with_hc1(
        df[outcome].to_numpy(dtype=float),
        X,
        names,
    )
    table.insert(0, "Specification", specification)
    table.insert(0, "Relative strategy", outcome)

    diagnostics = {
        "Relative strategy": outcome,
        "Specification": specification,
        **diagnostics,
    }
    return table, diagnostics



def _linear_combo_test(
    beta: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    dof: int,
) -> dict:
    """
    Wald/t test for c' beta with HC1 covariance.
    """
    weights = np.asarray(weights, dtype=float)
    est = float(weights @ beta)
    var = float(weights @ cov @ weights)
    se = float(np.sqrt(max(var, 0.0)))

    if se > 0:
        tstat = est / se
        pval = 2.0 * (1.0 - stats.t.cdf(abs(tstat), df=max(dof, 1)))
    else:
        tstat = np.nan
        pval = np.nan

    crit = stats.t.ppf(0.975, df=max(dof, 1))
    ci_low = est - crit * se
    ci_high = est + crit * se

    return {
        "Estimate": est,
        "Robust SE": se,
        "t-stat": tstat,
        "p-value": pval,
        "95% CI low": ci_low,
        "95% CI high": ci_high,
    }


def run_piecewise_slope_tests(relative_df: pd.DataFrame) -> pd.DataFrame:
    """
    Directly test:
      - right-side slope for b*/n >= 1,
      - left-side slope for b*/n < 1,
      - slope difference,
      - level jump at the boundary.

    Piecewise model:
        y = a
            + b1 (x - 1)
            + b2 I(x < 1)
            + b3 (x - 1) I(x < 1)
            + controls
            + e

    Hence:
      right slope = b1
      left slope = b1 + b3
      slope difference (left - right) = b3
      level jump below boundary = b2
    """
    required = [
        "b*/n centered at 1",
        "Below-boundary indicator",
        "Below-boundary slope interaction",
        "Market return / month",
        "Cross-sectional dispersion",
        "Covariance trace",
        "Neural - Historical",
        "Gaussian - Historical",
    ]
    df = relative_df.dropna(subset=required).copy()
    if len(df) < 20:
        raise ValueError("At least 20 OOS months are required for slope tests.")

    base_controls = _prepare_base_controls(df)
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["b*/n centered at 1"].to_numpy(dtype=float),
            df["Below-boundary indicator"].to_numpy(dtype=float),
            df["Below-boundary slope interaction"].to_numpy(dtype=float),
            *base_controls,
        ]
    )
    names = [
        "Intercept",
        "(b*/n - 1)",
        "I(b*/n < 1)",
        "(b*/n - 1) × I(b*/n < 1)",
        "Market return / month",
        "Cross-sectional dispersion",
        "Covariance trace",
    ]

    rows = []
    for outcome in ["Neural - Historical", "Gaussian - Historical"]:
        _, diagnostics, beta, cov = _ols_with_hc1(
            df[outcome].to_numpy(dtype=float),
            X,
            names,
        )
        dof = diagnostics["n"] - diagnostics["k"]

        tests = [
            (
                "Right-side slope (b*/n ≥ 1)",
                np.array([0, 1, 0, 0, 0, 0, 0], dtype=float),
            ),
            (
                "Left-side slope (b*/n < 1)",
                np.array([0, 1, 0, 1, 0, 0, 0], dtype=float),
            ),
            (
                "Slope difference (left - right)",
                np.array([0, 0, 0, 1, 0, 0, 0], dtype=float),
            ),
            (
                "Level jump at boundary",
                np.array([0, 0, 1, 0, 0, 0, 0], dtype=float),
            ),
        ]

        for test_name, weights in tests:
            result = _linear_combo_test(beta, cov, weights, dof)
            rows.append(
                {
                    "Relative strategy": outcome,
                    "Test": test_name,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def run_regime_regressions(
    relative_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run three regression specifications for both Neural-Historical and
    Gaussian-Historical relative returns.

    Specifications:
      1. Binary theoretical feasibility.
      2. Continuous b*/n.
      3. Piecewise linear regression around b*/n = 1.
    """
    required = [
        "Theoretical feasible",
        "b*/n",
        "b*/n centered at 1",
        "Below-boundary indicator",
        "Below-boundary slope interaction",
        "Market return / month",
        "Cross-sectional dispersion",
        "Covariance trace",
        "Neural - Historical",
        "Gaussian - Historical",
    ]

    df = relative_df.dropna(subset=required).copy()
    if len(df) < 20:
        raise ValueError("At least 20 OOS months are required for these regressions.")

    specs = [
        "Binary feasibility",
        "Continuous b*/n",
        "Piecewise at 1",
    ]
    outcomes = [
        "Neural - Historical",
        "Gaussian - Historical",
    ]

    tables = []
    diagnostics = []

    for outcome in outcomes:
        for spec in specs:
            table, diag = _run_single_spec(df, outcome, spec)
            tables.append(table)
            diagnostics.append(diag)

    return (
        pd.concat(tables, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def make_threshold_scatter_data(relative_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Realized date",
        "b*/n",
        "T source",
        "Neural - Historical",
        "Gaussian - Historical",
    ]
    return relative_df[cols].dropna().copy()
