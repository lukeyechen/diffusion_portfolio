from __future__ import annotations

import numpy as np
import pandas as pd

from .moments import sample_moments
from .portfolio_rules import compute_weights
from .turnover_upgrade import (
    exact_diffusion_augmented_moments,
    nested_exact_horizon,
    partial_rebalance,
    portfolio_turnover,
    solve_mv_turnover_aware,
)


CANDIDATE_T = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

# About ten years of estimation history with one-year inner validation blocks.
HORIZON_PRESETS: dict[str, dict[str, int | str]] = {
    "1 week": {
        "source": "weekly",
        "block_size": 1,
        "periods_per_year": 52,
        "lookback": 520,
        "inner_folds": 4,
        "validation_size": 52,
        "min_train_size": 200,
    },
    "2 weeks": {
        "source": "weekly",
        "block_size": 2,
        "periods_per_year": 26,
        "lookback": 260,
        "inner_folds": 4,
        "validation_size": 26,
        "min_train_size": 100,
    },
    "1 month": {
        "source": "monthly",
        "block_size": 1,
        "periods_per_year": 12,
        "lookback": 120,
        "inner_folds": 4,
        "validation_size": 12,
        "min_train_size": 50,
    },
    "2 months": {
        "source": "monthly",
        "block_size": 2,
        "periods_per_year": 6,
        "lookback": 60,
        "inner_folds": 4,
        "validation_size": 6,
        "min_train_size": 30,
    },
    "3 months": {
        "source": "monthly",
        "block_size": 3,
        "periods_per_year": 4,
        "lookback": 40,
        "inner_folds": 4,
        "validation_size": 4,
        "min_train_size": 20,
    },
}


def get_horizon_preset(label: str) -> dict[str, int | str]:
    if label not in HORIZON_PRESETS:
        raise ValueError(f"Unsupported holding period: {label}")
    return dict(HORIZON_PRESETS[label])


def aggregate_nonoverlapping(base_returns: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """Compound complete non-overlapping return blocks."""
    if not isinstance(base_returns, pd.DataFrame) or base_returns.empty:
        raise ValueError("base_returns must be a non-empty DataFrame.")
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if block_size == 1:
        return base_returns.copy()

    n_complete = (len(base_returns) // block_size) * block_size
    if n_complete < 2 * block_size:
        raise ValueError("Not enough observations for the requested holding period.")

    base = base_returns.iloc[:n_complete].copy()
    values = base.to_numpy(dtype=float).reshape(
        -1, block_size, base_returns.shape[1]
    )
    compounded = np.prod(1.0 + values, axis=1) - 1.0
    idx = base.index[block_size - 1 :: block_size]
    return pd.DataFrame(compounded, index=idx, columns=base_returns.columns)


def drifted_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    asset_returns = np.asarray(asset_returns, dtype=float).reshape(-1)
    gross = 1.0 + float(weights @ asset_returns)
    if gross <= 0 or not np.isfinite(gross):
        return weights.copy()
    out = weights * (1.0 + asset_returns) / gross
    return out / float(out.sum())


def project_long_only_capped(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Euclidean projection to {w: sum w=1, 0<=w_i<=max_weight}."""
    v = np.asarray(weights, dtype=float).reshape(-1)
    cap = float(max_weight)
    n = v.size
    if n == 0:
        raise ValueError("weights cannot be empty.")
    if cap <= 0 or cap * n < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for a fully invested portfolio.")

    # Projection has form clip(v - tau, 0, cap). Find tau by bisection.
    lo = float(np.min(v - cap)) - 1.0
    hi = float(np.max(v)) + 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        s = float(np.clip(v - mid, 0.0, cap).sum())
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - 0.5 * (lo + hi), 0.0, cap)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Projection failed.")
    # The bisection already enforces sum=1 to numerical precision.
    w = w / total
    if np.max(w) > cap + 1e-9:
        # One more projection avoids a normalization-induced cap violation.
        lo = float(np.min(w - cap)) - 1.0
        hi = float(np.max(w)) + 1.0
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            s = float(np.clip(w - mid, 0.0, cap).sum())
            if s > 1.0:
                lo = mid
            else:
                hi = mid
        w = np.clip(w - 0.5 * (lo + hi), 0.0, cap)
    return w


def _mv_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    returns: np.ndarray,
    *,
    gamma: float,
    max_long_weight: float,
) -> np.ndarray:
    return np.asarray(
        compute_weights(
            "Mean-Variance",
            mu,
            sigma,
            gamma=float(gamma),
            returns=returns,
            constraint_mode="Long-only",
            max_long_weight=float(max_long_weight),
            max_short_weight=0.20,
            max_gross_exposure=1.50,
        ),
        dtype=float,
    ).reshape(-1)


def _lw_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    returns: np.ndarray,
    *,
    gamma: float,
    max_long_weight: float,
) -> np.ndarray:
    return np.asarray(
        compute_weights(
            "Ledoit-Wolf Mean-Variance",
            mu,
            sigma,
            gamma=float(gamma),
            returns=returns,
            constraint_mode="Long-only",
            max_long_weight=float(max_long_weight),
            max_short_weight=0.20,
            max_gross_exposure=1.50,
        ),
        dtype=float,
    ).reshape(-1)


def select_best_nested_t(
    x: np.ndarray,
    cfg: dict,
    *,
    gamma: float = 3.0,
    m: int = 500,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_t: list[float] | None = None,
    max_long_weight: float = 0.40,
) -> tuple[float, list[dict]]:
    grid = CANDIDATE_T if candidate_t is None else [float(t) for t in candidate_t]
    _, results = nested_exact_horizon(
        np.asarray(x, dtype=float),
        gamma=float(gamma),
        m=int(m),
        beta=float(beta),
        n_steps=int(n_steps),
        candidate_T=grid,
        inner_folds=int(cfg["inner_folds"]),
        validation_size=int(cfg["validation_size"]),
        min_train_size=int(cfg["min_train_size"]),
        selection_rule="best",
        constraint_mode="Long-only",
        max_long_weight=float(max_long_weight),
    )
    best = max(results, key=lambda r: (r["mean_validation_CER"], -r["T"]))
    return float(best["T"]), results


def turnover_controlled_target(
    x: np.ndarray,
    previous_drifted: np.ndarray,
    cfg: dict,
    *,
    gamma: float = 3.0,
    m: int = 500,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_t: list[float] | None = None,
    turnover_penalty: float = 0.0025,
    rebalance_alpha: float = 1.0,
    max_long_weight: float = 0.40,
) -> dict:
    """Best-T exact diffusion target with turnover penalty and optional smoothing."""
    t_best, t_results = select_best_nested_t(
        x,
        cfg,
        gamma=gamma,
        m=m,
        beta=beta,
        n_steps=n_steps,
        candidate_t=candidate_t,
        max_long_weight=max_long_weight,
    )
    mu_d, sigma_d, mu_fake, sigma_fake = exact_diffusion_augmented_moments(
        np.asarray(x, dtype=float),
        m=int(m),
        horizon=float(t_best),
        beta=float(beta),
        n_steps=int(n_steps),
    )
    raw = solve_mv_turnover_aware(
        mu_d,
        sigma_d,
        previous_weights=np.asarray(previous_drifted, dtype=float),
        turnover_penalty=float(turnover_penalty),
        gamma=float(gamma),
        mode="Long-only",
        max_long_weight=float(max_long_weight),
    )
    alpha = float(rebalance_alpha)
    final = (
        partial_rebalance(raw, previous_drifted, alpha=alpha)
        if alpha < 1.0 - 1e-15
        else np.asarray(raw, dtype=float).copy()
    )
    final = project_long_only_capped(final, float(max_long_weight))
    return {
        "T": t_best,
        "t_results": t_results,
        "mu": mu_d,
        "sigma": sigma_d,
        "mu_fake": mu_fake,
        "sigma_fake": sigma_fake,
        "raw_target": np.asarray(raw, dtype=float),
        "weights": np.asarray(final, dtype=float),
        "turnover": portfolio_turnover(final, previous_drifted),
    }


def replay_latest_recommendation(
    returns: pd.DataFrame,
    cfg: dict,
    *,
    gamma: float = 3.0,
    m: int = 500,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_t: list[float] | None = None,
    turnover_penalty: float = 0.0025,
    rebalance_alpha: float = 1.0,
    max_long_weight: float = 0.40,
    replay_start: str | pd.Timestamp | None = "2019-01-01",
) -> dict:
    """Replay the strategy state, then compute the recommendation for the next period."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    lookback = int(cfg["lookback"])
    if len(returns) <= lookback:
        raise ValueError(
            f"Need more than {lookback} observations for this holding period; got {len(returns)}."
        )

    n_assets = returns.shape[1]
    previous_drifted = np.ones(n_assets, dtype=float) / n_assets
    eligible = list(range(lookback, len(returns)))
    if replay_start is not None:
        start_ts = pd.Timestamp(replay_start).tz_localize(None)
        filtered = [
            i
            for i in eligible
            if pd.Timestamp(returns.index[i]).tz_localize(None) >= start_ts
        ]
        if filtered:
            eligible = filtered

    last_t = np.nan
    last_target = None
    for i in eligible:
        x = returns.iloc[i - lookback : i].to_numpy(dtype=float)
        realized = returns.iloc[i].to_numpy(dtype=float)
        rec = turnover_controlled_target(
            x,
            previous_drifted,
            cfg,
            gamma=gamma,
            m=m,
            beta=beta,
            n_steps=n_steps,
            candidate_t=candidate_t,
            turnover_penalty=turnover_penalty,
            rebalance_alpha=rebalance_alpha,
            max_long_weight=max_long_weight,
        )
        previous_drifted = drifted_weights(rec["weights"], realized)
        last_t = float(rec["T"])
        last_target = np.asarray(rec["weights"], dtype=float).copy()

    x_next = returns.iloc[-lookback:].to_numpy(dtype=float)
    next_rec = turnover_controlled_target(
        x_next,
        previous_drifted,
        cfg,
        gamma=gamma,
        m=m,
        beta=beta,
        n_steps=n_steps,
        candidate_t=candidate_t,
        turnover_penalty=turnover_penalty,
        rebalance_alpha=rebalance_alpha,
        max_long_weight=max_long_weight,
    )
    next_rec.update(
        {
            "previous_drifted": previous_drifted,
            "previous_T": last_t,
            "last_target": last_target,
            "data_through": pd.Timestamp(returns.index[-1]),
            "lookback": lookback,
        }
    )
    return next_rec


def performance_metrics(r: np.ndarray, *, gamma: float, periods_per_year: int) -> dict:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        raise ValueError("No returns supplied.")
    wealth = np.cumprod(1.0 + r)
    total_return = float(wealth[-1] - 1.0)
    cagr = float(wealth[-1] ** (float(periods_per_year) / len(r)) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan
    mean_ann = float(np.mean(r) * periods_per_year)
    sharpe = mean_ann / vol if np.isfinite(vol) and vol > 0 else np.nan
    peak = np.maximum.accumulate(wealth)
    max_dd = float(np.min(wealth / peak - 1.0))
    var_ann = float(np.var(r, ddof=1) * periods_per_year) if len(r) > 1 else np.nan
    cer = mean_ann - 0.5 * float(gamma) * var_ann if np.isfinite(var_ann) else np.nan
    return {
        "Total return": total_return,
        "CAGR": cagr,
        "Annualized vol": vol,
        "Sharpe": sharpe,
        "Realized CER": cer,
        "Max drawdown": max_dd,
        "Positive periods": float(np.mean(r > 0)),
        "Final $10,000": float(10000.0 * wealth[-1]),
    }


def run_oos_comparison(
    returns: pd.DataFrame,
    cfg: dict,
    *,
    gamma: float = 3.0,
    m: int = 500,
    beta: float = 1.0,
    n_steps: int = 100,
    candidate_t: list[float] | None = None,
    turnover_penalty: float = 0.0025,
    rebalance_alpha: float = 1.0,
    max_long_weight: float = 0.40,
    oos_start: str | pd.Timestamp = "2019-01-01",
    costs: tuple[float, ...] = (0.0010, 0.0025),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling OOS comparison used by the upgraded Method Comparison UI."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    lookback = int(cfg["lookback"])
    ppy = int(cfg["periods_per_year"])
    if len(returns) <= lookback:
        raise ValueError(f"Need more than {lookback} observations; got {len(returns)}.")

    method_tc = "Turnover-Controlled Exact Diffusion"
    methods = [
        "Equal Weight",
        "Classical MV",
        "Classical MV + LW",
        "Exact Diffusion (Best-T)",
        "50% Exact Diff + 50% EW",
        method_tc,
    ]
    prev_post: dict[str, np.ndarray | None] = {name: None for name in methods}
    rows: list[dict] = []
    t_values: list[float] = []
    latest_weights: dict[str, np.ndarray] = {}
    start_ts = pd.Timestamp(oos_start).tz_localize(None)
    eligible = [
        i
        for i in range(lookback, len(returns))
        if pd.Timestamp(returns.index[i]).tz_localize(None) >= start_ts
    ]
    if not eligible:
        raise ValueError("No OOS periods remain after the lookback and OOS-start filters.")

    n_assets = returns.shape[1]
    ew = np.ones(n_assets, dtype=float) / n_assets

    for i in eligible:
        x = returns.iloc[i - lookback : i].to_numpy(dtype=float)
        realized = returns.iloc[i].to_numpy(dtype=float)
        mu_h, sigma_h = sample_moments(x, mle=True)
        w_classical = _mv_weights(
            mu_h, sigma_h, x, gamma=gamma, max_long_weight=max_long_weight
        )
        w_lw = _lw_weights(
            mu_h, sigma_h, x, gamma=gamma, max_long_weight=max_long_weight
        )

        t_best, _ = select_best_nested_t(
            x,
            cfg,
            gamma=gamma,
            m=m,
            beta=beta,
            n_steps=n_steps,
            candidate_t=candidate_t,
            max_long_weight=max_long_weight,
        )
        t_values.append(t_best)
        mu_d, sigma_d, _, _ = exact_diffusion_augmented_moments(
            x, m=int(m), horizon=t_best, beta=float(beta), n_steps=int(n_steps)
        )
        w_diff = _mv_weights(
            mu_d, sigma_d, x, gamma=gamma, max_long_weight=max_long_weight
        )
        w_blend = 0.5 * w_diff + 0.5 * ew
        w_blend = project_long_only_capped(w_blend, max_long_weight)

        prev_tc = prev_post[method_tc]
        if prev_tc is None:
            prev_tc = ew.copy()
        raw_tc = solve_mv_turnover_aware(
            mu_d,
            sigma_d,
            previous_weights=prev_tc,
            turnover_penalty=float(turnover_penalty),
            gamma=float(gamma),
            mode="Long-only",
            max_long_weight=float(max_long_weight),
        )
        w_tc = (
            partial_rebalance(raw_tc, prev_tc, alpha=float(rebalance_alpha))
            if float(rebalance_alpha) < 1.0 - 1e-15
            else np.asarray(raw_tc, dtype=float)
        )
        w_tc = project_long_only_capped(w_tc, max_long_weight)

        weights = {
            "Equal Weight": ew,
            "Classical MV": w_classical,
            "Classical MV + LW": w_lw,
            "Exact Diffusion (Best-T)": w_diff,
            "50% Exact Diff + 50% EW": w_blend,
            method_tc: w_tc,
        }

        row = {
            "Date": pd.Timestamp(returns.index[i]),
            "T": float(t_best),
        }
        for name, w in weights.items():
            old = prev_post[name]
            turnover = 0.0 if old is None else portfolio_turnover(w, old)
            row[f"return__{name}"] = float(w @ realized)
            row[f"turnover__{name}"] = float(turnover)
            prev_post[name] = drifted_weights(w, realized)
            latest_weights[name] = np.asarray(w, dtype=float).copy()
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for name in methods:
        gross = detail[f"return__{name}"].to_numpy(dtype=float)
        turnover = detail[f"turnover__{name}"].to_numpy(dtype=float)
        out = {
            "Method": name,
            "OOS periods": len(detail),
            "OOS start": detail["Date"].iloc[0],
            "OOS end": detail["Date"].iloc[-1],
            "Average turnover": float(turnover.mean()),
            **performance_metrics(gross, gamma=gamma, periods_per_year=ppy),
        }
        for cost in costs:
            bps = int(round(cost * 10000))
            net = performance_metrics(
                gross - float(cost) * turnover,
                gamma=gamma,
                periods_per_year=ppy,
            )
            out[f"Net CAGR {bps}bps"] = net["CAGR"]
            out[f"Net Sharpe {bps}bps"] = net["Sharpe"]
            out[f"Net Final $10,000 {bps}bps"] = net["Final $10,000"]
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    latest = pd.DataFrame(
        [
            {"Method": name, **{asset: float(wj) for asset, wj in zip(returns.columns, w)}}
            for name, w in latest_weights.items()
        ]
    )
    t_arr = np.asarray(t_values, dtype=float)
    tdiag = pd.DataFrame(
        [
            {
                "OOS periods": len(detail),
                "T>0 fraction": float(np.mean(t_arr > 0)),
                "Median T": float(np.median(t_arr)),
                "Mean T": float(np.mean(t_arr)),
            }
        ]
    )
    return summary, detail, latest, tdiag
