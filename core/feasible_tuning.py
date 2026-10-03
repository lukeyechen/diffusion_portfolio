"""Exact continuous-time portfolio proxy and feasible b tuning.

This is the unconditional Gaussian portfolio experiment, not diffusion OLS.
H is the centered MLE covariance; direct tuning uses exact continuous moments.
The optional old-strategy comparator preserves its original finite-step model.
The theorem concerns unconstrained, frictionless expected utility.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .diffusion_exposure import allocate_exposure
from .short_horizon_portfolio import (CANDIDATE_T, project_long_only_capped,
                                     turnover_controlled_target)
from .turnover_upgrade import partial_rebalance, solve_mv_turnover_aware
from .portfolio_rules import compute_weights
from .predictive_means import MEAN_MODELS
from .short_horizon_portfolio import performance_metrics

TRACE_CALIBRATION_VERSION = 2

OLD_METHOD = "Old Portfolio (Best-T + turnover control)"
CLASSICAL_MV_RULE = "allocation-aware-classical-v2"
COMPARISON_VERSION = 4


@dataclass(frozen=True)
class TradingComparison:
    """Common practical allocation settings, outside the theorem's guarantee."""
    inner_folds: int = 4
    validation_size: int = 52
    min_train_size: int = 200
    turnover_penalty: float = 0.0025
    rebalance_alpha: float = 0.5
    cap: float = 0.4
    m: int = 500
    beta: float = 1.0
    n_steps: int = 100
    candidate_t: tuple = tuple(CANDIDATE_T)
    mean_model: str = "Original diffusion mean"

    def validate(self, n, assets):
        if self.mean_model not in MEAN_MODELS:
            raise ValueError("Unknown old Portfolio expected-return model.")
        if self.inner_folds < 2 or self.validation_size < 2 or self.min_train_size < 2:
            raise ValueError("Old-method validation requires at least two folds and observations per block.")
        if n < self.min_train_size + self.inner_folds * self.validation_size:
            raise ValueError("Increase the common window to fit the old method's nested validation blocks.")
        if not 0 < self.cap <= 1 or self.cap * assets < 1 - 1e-12:
            raise ValueError("The common fully invested allocation needs cap × asset count ≥ 1.")
        if (not np.isfinite([self.turnover_penalty, self.rebalance_alpha, self.beta]).all()
                or self.turnover_penalty < 0 or not 0 < self.rebalance_alpha <= 1
                or self.beta <= 0 or self.m < 0 or self.n_steps < 1):
            raise ValueError("Invalid old-method trading or diffusion settings.")
        if not self.candidate_t or any(not np.isfinite(t) or t < 0 for t in self.candidate_t):
            raise ValueError("Old-method candidate T values must be finite and nonnegative.")

    def validation_config(self, n):
        return dict(lookback=n, inner_folds=self.inner_folds,
                    validation_size=self.validation_size, min_train_size=self.min_train_size)


@dataclass(frozen=True)
class TuningSettings:
    gamma: float = 3.0
    epsilon: float = 0.25
    c: float = 4.0
    a_max: float = 0.95
    fixed_b: float = 7.0
    b_min: float = 1.0
    b_max: float = 20.0

    def validate(self):
        values = np.array(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All tuning settings must be finite.")
        if self.gamma <= 0 or self.epsilon <= 0 or not 0 < self.c <= 4:
            raise ValueError("Require gamma > 0, epsilon > 0, and 0 < c <= 4.")
        if not 0 < self.a_max < 1 or self.fixed_b <= 0:
            raise ValueError("Require 0 < a_max < 1 and fixed b > 0.")
        if not 0 < self.b_min < self.b_max:
            raise ValueError("Require 0 < ratio lower bound < upper bound.")


def moments(sample):
    x = np.asarray(sample, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[0] <= x.shape[1]:
        raise ValueError("Need finite observations with sample size greater than asset count.")
    u = x.mean(axis=0)
    centered = x - u
    h = centered.T @ centered / len(x)
    values = np.linalg.eigvalsh(h)
    if values[0] <= 0 or values[0] <= values[-1] * 1e-12:
        raise ValueError("Sample covariance must be positive definite and well conditioned; increase the window or remove redundant assets.")
    return u, h


def endpoint_moments(u, h, a):
    """m=(1-a)D^-1 u; S=H-a^2 H^3 D^-2, D=(1-a)I+aH."""
    if not 0 <= a < 1:
        raise ValueError("Endpoint requires 0 <= a < 1.")
    values, vectors = np.linalg.eigh(h)
    if values[0] <= 0:
        raise ValueError("Endpoint covariance input must be positive definite.")
    denominator = 1 - a + a * values
    mean = vectors @ ((1 - a) / denominator * (vectors.T @ u))
    endpoint_values = values * (1 - a) * (1 - a + 2 * a * values) / denominator**2
    covariance = (vectors * endpoint_values) @ vectors.T
    return mean, covariance


def ratio_b(u, h, settings):
    a = (len(u) + 2) * float(u @ u) + float(np.trace(h))
    q = float(u @ h @ u)
    return settings.b_max if q <= 0 else float(np.clip(a / q, settings.b_min, settings.b_max))


def fit_portfolios(sample, settings, *, pilot=None, cap=None, trading=None, previous=None,
                   return_raw=False, methods=None):
    settings.validate()
    u, h = moments(sample)
    n = len(sample)
    previous = {} if previous is None else previous
    raw_portfolios = {}
    if trading is not None:
        trading.validate(n, len(u))
    choices = {
        "Trace tuning": settings.c / (settings.epsilon + float(np.trace(h))),
    }
    if pilot is not None:
        pilot_u, pilot_h = moments(pilot)
        if pilot_u.shape != u.shape:
            raise ValueError("Pilot and main sample must have the same assets.")
        choices["Pilot ratio"] = ratio_b(pilot_u, pilot_h, settings)

    def weights(mean, covariance, name):
        if trading is not None:
            if return_raw:
                raw_portfolios[name] = compute_weights(
                    "Mean-Variance", mean, covariance, gamma=settings.gamma,
                    constraint_mode="Long-only", max_long_weight=trading.cap)
            prev = previous.get(name, np.full(len(u), 1/len(u)))
            raw = solve_mv_turnover_aware(mean, covariance, prev,
                                         trading.turnover_penalty, gamma=settings.gamma,
                                         max_long_weight=trading.cap)
            final = partial_rebalance(raw, prev, trading.rebalance_alpha)
            return project_long_only_capped(final, trading.cap)
        if cap is None:
            w = np.linalg.solve(covariance, mean) / settings.gamma
        else:
            w = allocate_exposure(mean, covariance, gamma=settings.gamma, cap=cap)
        raw_portfolios[name] = w.copy()
        return w

    def classical_weights(mean, covariance):
        # Respect allocation constraints without turnover penalties or smoothing.
        if trading is not None:
            return compute_weights(
                "Mean-Variance", mean, covariance, gamma=settings.gamma,
                constraint_mode="Long-only", max_long_weight=trading.cap)
        if cap is not None:
            return allocate_exposure(mean, covariance, gamma=settings.gamma, cap=cap)
        return np.linalg.solve(covariance, mean) / settings.gamma

    classical = classical_weights(u, h)
    portfolios = {"Classical (main sample)": classical}
    raw_portfolios["Classical (main sample)"] = classical.copy()
    diagnostics = []
    for name, b in choices.items():
        if methods is not None and name not in methods:
            continue
        a = min(settings.a_max, b / n)
        mean, covariance = endpoint_moments(u, h, a)
        portfolios[name] = weights(mean, covariance, name)
        diagnostics.append({
            "Method": name, "b": b, "a": a,
            "Effective b = n a": n * a,
            "T": -float(np.log(a)) / (trading.beta if trading else 1.0),
            "Horizon rule": "Direct b rule",
        })
    if pilot is not None:
        full_u, full_h = moments(np.concatenate([pilot, sample]))
        full_classical = classical_weights(full_u, full_h)
        portfolios["Classical (main + pilot)"] = full_classical
        raw_portfolios["Classical (main + pilot)"] = full_classical.copy()
    if trading is not None and (methods is None or OLD_METHOD in methods):
        old = turnover_controlled_target(
            sample, previous.get(OLD_METHOD, np.full(len(u), 1/len(u))),
            trading.validation_config(n), gamma=settings.gamma, m=trading.m,
            beta=trading.beta, n_steps=trading.n_steps,
            candidate_t=list(trading.candidate_t), turnover_penalty=trading.turnover_penalty,
            rebalance_alpha=trading.rebalance_alpha, max_long_weight=trading.cap,
            mean_model=trading.mean_model,
        )
        portfolios[OLD_METHOD] = old["weights"]
        if return_raw:
            raw_portfolios[OLD_METHOD] = compute_weights(
                "Mean-Variance", old["mu"], old["sigma"], gamma=settings.gamma,
                constraint_mode="Long-only", max_long_weight=trading.cap)
        diagnostics.append({"Method": OLD_METHOD, "b": np.nan, "a": np.nan,
                            "Effective b = n a": np.nan,
                            "T": old["T"], "Horizon rule": "Nested candidate grid"})
    if return_raw:
        return portfolios, pd.DataFrame(diagnostics), raw_portfolios
    return portfolios, pd.DataFrame(diagnostics)


def coefficients(mu, sigma, settings):
    """Population asymptotic coefficients; ratio formula is local/interior."""
    settings.validate()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n_assets = len(mu)
    norm = float(mu @ mu)
    a = (n_assets + 2) * norm + float(np.trace(sigma))
    q = float(mu @ sigma @ mu)
    d = settings.epsilon + float(np.trace(sigma))
    k = lambda b: (b * a - b * b * q / 2) / settings.gamma
    rows = [{"Method": "Fixed b", "Limit b": settings.fixed_b,
             "K2": k(settings.fixed_b), "Formula applies": True}]
    b0 = settings.b_max if q == 0 else float(np.clip(a / q, settings.b_min, settings.b_max))
    # Away from clipping boundaries the bounded rule is either locally constant
    # or equal to the smooth A/Q ratio. At a boundary this proposition does not apply.
    boundary = q > 0 and (np.isclose(a / q, settings.b_min) or np.isclose(a / q, settings.b_max))
    interior = q > 0 and settings.b_min < a / q < settings.b_max
    xi = (2 * (n_assets + 1) + 2 * a * norm / q
          - 2 * a * float(mu @ sigma @ sigma @ mu) / q**2) if interior else 0.0
    rows.extend([
        {"Method": "Same-sample ratio", "Limit b": b0,
         "K2": np.nan if boundary else k(b0) + xi / settings.gamma,
         "Formula applies": not boundary},
        {"Method": "Pilot ratio", "Limit b": b0,
         "K2": k(b0), "Formula applies": True},
        {"Method": "Trace tuning", "Limit b": settings.c / d,
         "K2": (settings.c * a / d + (2 * settings.c - settings.c**2 / 2) * q / d**2) / settings.gamma,
         "Formula applies": True},
    ])
    return pd.DataFrame(rows)


def gaussian_experiment(mu, sigma, settings, *, n=500, repetitions=1000, seed=42,
                        include_pilot=False):
    if n <= len(mu) + 4 or repetitions < 2:
        raise ValueError("Increase Gaussian sample size or repetitions.")
    rng = np.random.default_rng(seed)
    gains = {}
    for _ in range(repetitions):
        main = rng.multivariate_normal(mu, sigma, n)
        pilot = rng.multivariate_normal(mu, sigma, n) if include_pilot else None
        portfolios, _ = fit_portfolios(main, settings, pilot=pilot)
        utility = lambda w: float(w @ mu - settings.gamma / 2 * w @ sigma @ w)
        baseline = utility(portfolios["Classical (main sample)"])
        for method, w in portfolios.items():
            if method != "Classical (main sample)":
                gains.setdefault(method, []).append(utility(w) - baseline)
    predicted = coefficients(mu, sigma, settings).set_index("Method")
    rows = []
    for method, values in gains.items():
        v = np.asarray(values)
        se = float(v.std(ddof=1) / np.sqrt(repetitions))
        rows.append({"Method": method, "Mean utility gain": v.mean(),
                     "Monte Carlo SE": se, "n² mean gain": n*n*v.mean(),
                     "n² MC SE": n*n*se,
                     "Asymptotic K2": predicted.loc[method, "K2"] if method in predicted.index else np.nan})
    return pd.DataFrame(rows)


def historical_backtest(returns, settings, *, window=120, pilot_size=0,
                        oos_start="2019-01-01", rf=0.0, cost_bps=0.0, cap=None,
                        trading=None, methods=None):
    if window <= returns.shape[1] or (pilot_size and pilot_size <= returns.shape[1]):
        raise ValueError("Main and pilot windows must exceed the asset count.")
    if not 0 <= cost_bps < 10000 or rf <= -1 or not np.isfinite(rf):
        raise ValueError("Invalid trading cost or risk-free return.")
    dates = pd.to_datetime(returns.index)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and increasing.")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Stock returns must be finite and greater than -100%.")
    previous = {}
    rows = []
    for t in range(window + pilot_size, len(values)):
        if dates[t] < pd.Timestamp(oos_start):
            continue
        sample = values[t-window:t] - rf
        pilot = values[t-window-pilot_size:t-window] - rf if pilot_size else None
        portfolios, diagnostics = fit_portfolios(
            sample, settings, pilot=pilot, cap=cap, trading=trading,
            previous={name: w[:-1] for name, w in previous.items()},
            methods=methods,
        )
        diag = diagnostics.set_index("Method")
        for method, w in portfolios.items():
            target = np.r_[w, 1 - w.sum()]
            initial = (np.r_[np.full(len(w), 1/len(w)), 0.0] if trading
                       else np.r_[np.zeros(len(w)), 1.0])
            old = previous.get(method, initial)
            turnover = float(np.abs(target-old).sum() / 2)
            gross = float(rf + w @ (values[t] - rf))
            net = gross - cost_bps * 1e-4 * turnover
            if gross <= -1 or net <= -1:
                if method == "Classical (main sample)":
                    raise ValueError(f"Classical MV exhausts its capital on {dates[t].date()}. Its unconstrained weights cannot continue after a loss of 100% or more; increase risk aversion γ to reduce exposure.")
                raise ValueError(f"{method} reaches a nonpositive wealth factor on {dates[t].date()}; reduce leverage or use the constrained comparison.")
            previous[method] = target * np.r_[1 + values[t], 1 + rf] / (1 + gross)
            row = {"Date": dates[t], "Method": method, "Gross return": gross,
                   "Net return": net, "Turnover": turnover,
                   "Cash weight": target[-1]}
            row.update({f"Weight {asset}": w[j] for j, asset in enumerate(returns.columns)})
            if method in diag.index:
                row.update(diag.loc[method].to_dict())
            rows.append(row)
    if not rows:
        raise ValueError("No evaluated periods remain; download more history or adjust the start and windows.")
    return pd.DataFrame(rows)


def latest_portfolios(returns, settings, history, *, window, rf=0.0, cap=None, trading=None,
                      return_raw=False):
    """Next target from each method's own last holdings, with no extra replay."""
    if len(returns) < window:
        raise ValueError("Not enough observations for the common estimation window.")
    previous = {}
    if trading is not None:
        for method, group in history.groupby("Method", sort=False):
            last = group.iloc[-1]
            if pd.Timestamp(last["Date"]) != pd.Timestamp(returns.index[-1]):
                raise ValueError("The replay must include the latest data period before calculating next weights.")
            w = np.array([last[f"Weight {asset}"] for asset in returns.columns])
            previous[method] = w * (1 + returns.iloc[-1].to_numpy()) / (1 + last["Gross return"])
    return fit_portfolios(returns.iloc[-window:].to_numpy()-rf, settings,
                          cap=cap, trading=trading, previous=previous, return_raw=return_raw)


def trace_calibration(returns, settings, epsilons, *, cs=None, window, calibration_start,
                        evaluation_start, periods_per_year, rf=0.0, cost_bps=0.0,
                        cap=None, trading=None, end_date=None):
    """Choose c and epsilon on past calibration only, then freeze for a held-out replay.

    No old nested-T calculation is needed for this trace-versus-classical experiment.
    The final evaluation starts both strategies from the same initial holdings.
    """
    grid = tuple(dict.fromkeys(float(e) for e in epsilons))
    if not grid or len(grid) > 12 or not np.isfinite(grid).all() or min(grid) <= 0:
        raise ValueError("Enter 1–12 finite, positive epsilon candidates.")
    c_grid = (settings.c,) if cs is None else tuple(dict.fromkeys(float(c) for c in cs))
    if not c_grid or len(c_grid) > 12 or not np.isfinite(c_grid).all() or min(c_grid) <= 0 or max(c_grid) > 4:
        raise ValueError("Enter 1–12 finite c candidates with 0 < c ≤ 4.")
    if end_date is not None:
        end = pd.Timestamp(end_date).normalize()
        if end < pd.Timestamp(evaluation_start):
            raise ValueError("End date must be on or after final evaluation starts.")
        returns = returns.loc[returns.index < end + pd.Timedelta(days=1)]
    cutoff = pd.Timestamp(evaluation_start)
    if pd.Timestamp(calibration_start) >= cutoff:
        raise ValueError("Calibration must start before the final evaluation.")
    calibration_data = returns.loc[returns.index < cutoff]
    methods = ("Classical (main sample)", "Trace tuning")

    def summarize(group):
        net = group["Net return"].to_numpy()
        if len(net) < 2:
            raise ValueError("Both calibration and final evaluation need at least two evaluated periods.")
        metrics = performance_metrics(net, gamma=settings.gamma, periods_per_year=periods_per_year)
        excess = net-rf
        return {"Periods": len(net), "Total return": metrics["Total return"],
                "CAGR": metrics["CAGR"], "Volatility": metrics["Annualized vol"],
                "Max drawdown": metrics["Max drawdown"],
                "Annualized MV excess": periods_per_year*(excess.mean()-settings.gamma/2*np.var(excess, ddof=1)),
                "Average turnover": group["Turnover"].mean()}

    rows = []
    for c in c_grid:
        for epsilon in grid:
            candidate = replace(settings, c=c, epsilon=epsilon)
            history = historical_backtest(calibration_data, candidate, window=window,
                                          oos_start=calibration_start, rf=rf, cost_bps=cost_bps,
                                          cap=cap, trading=trading, methods=methods)
            trace = history[history["Method"] == "Trace tuning"]
            rows.append({"c": c, "ε": epsilon, **summarize(trace)})
    calibration = pd.DataFrame(rows)
    winner = calibration.loc[calibration["Annualized MV excess"].idxmax()]
    selected = float(winner["ε"])
    selected_c = float(winner["c"])
    frozen = replace(settings, c=selected_c, epsilon=selected)
    evaluation = historical_backtest(returns, frozen, window=window,
                                     oos_start=evaluation_start, rf=rf, cost_bps=cost_bps,
                                     cap=cap, trading=trading, methods=methods)
    summary = pd.DataFrame([{"Method": method, **summarize(group)}
                            for method, group in evaluation.groupby("Method", sort=False)])
    return dict(calibration=calibration, selected_epsilon=selected, selected_c=selected_c,
                calibration_through=calibration_data.index[-1],
                evaluation_summary=summary, evaluation=evaluation)


def epsilon_sensitivity(returns, settings, epsilons, **kwargs):
    """Calibrate epsilon while keeping c fixed (existing experiment)."""
    result = trace_calibration(returns, settings, epsilons, **kwargs)
    result["calibration"] = result["calibration"].drop(columns="c")
    return result
