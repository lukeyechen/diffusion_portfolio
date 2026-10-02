"""Exact continuous-time portfolio proxy and feasible b tuning.

This is the unconditional Gaussian portfolio experiment, not diffusion OLS.
H is the centered MLE covariance. No finite synthetic sample or Euler stepping
is used. The theorem concerns unconstrained, frictionless expected utility.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diffusion_exposure import allocate_exposure


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


def fit_portfolios(sample, settings, *, pilot=None, cap=None):
    settings.validate()
    u, h = moments(sample)
    n = len(sample)
    choices = {
        "Fixed b": settings.fixed_b,
        "Same-sample ratio": ratio_b(u, h, settings),
        "Trace tuning": settings.c / (settings.epsilon + float(np.trace(h))),
    }
    if pilot is not None:
        pilot_u, pilot_h = moments(pilot)
        if pilot_u.shape != u.shape:
            raise ValueError("Pilot and main sample must have the same assets.")
        choices["Pilot ratio"] = ratio_b(pilot_u, pilot_h, settings)

    def weights(mean, covariance):
        if cap is None:
            return np.linalg.solve(covariance, mean) / settings.gamma
        return allocate_exposure(mean, covariance, gamma=settings.gamma, cap=cap)

    portfolios = {"Classical (main sample)": weights(u, h)}
    diagnostics = []
    for name, b in choices.items():
        a = min(settings.a_max, b / n)
        mean, covariance = endpoint_moments(u, h, a)
        portfolios[name] = weights(mean, covariance)
        diagnostics.append({
            "Method": name, "b": b, "a": a,
            "Effective b = n a": n * a,
            "a cap active": b / n > settings.a_max,
            "T (beta=1)": -float(np.log(a)),
        })
    if pilot is not None:
        full_u, full_h = moments(np.concatenate([pilot, sample]))
        portfolios["Classical (main + pilot)"] = weights(full_u, full_h)
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


def gaussian_experiment(mu, sigma, settings, *, n=500, repetitions=1000, seed=42):
    if n <= len(mu) + 4 or repetitions < 2:
        raise ValueError("Increase Gaussian sample size or repetitions.")
    rng = np.random.default_rng(seed)
    gains = {}
    for _ in range(repetitions):
        main = rng.multivariate_normal(mu, sigma, n)
        pilot = rng.multivariate_normal(mu, sigma, n)
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
                        oos_start="2019-01-01", rf=0.0, cost_bps=0.0, cap=None):
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
        portfolios, diagnostics = fit_portfolios(sample, settings, pilot=pilot, cap=cap)
        diag = diagnostics.set_index("Method")
        for method, w in portfolios.items():
            target = np.r_[w, 1 - w.sum()]
            old = previous.get(method, np.r_[np.zeros(len(w)), 1.0])
            turnover = float(np.abs(target-old).sum() / 2)
            gross = float(rf + w @ (values[t] - rf))
            net = gross - cost_bps * 1e-4 * turnover
            if gross <= -1 or net <= -1:
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
