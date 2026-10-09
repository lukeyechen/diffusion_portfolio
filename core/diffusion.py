from __future__ import annotations
import numpy as np
from .moments import sample_moments
from .covariance_solver import factor_covariance


def vp_signal(t: float, beta: float = 1.0) -> float:
    return float(np.exp(-0.5 * beta * t))


def forward_moments(mu_hat, sigma_hat, t: float, beta: float = 1.0):
    s = vp_signal(t, beta)
    N = len(mu_hat)
    sigma_t = s * s * sigma_hat + (1.0 - s * s) * np.eye(N)
    mu_t = s * mu_hat
    return mu_t, sigma_t


def generate_reverse_diffusion_samples(
    returns,
    m: int,
    horizon: float,
    beta: float = 1.0,
    n_steps: int = 200,
    seed: int | None = None,
):
    """
    Gaussian reverse-SDE research prototype using the linear score
    implied by sample moments and Euler-Maruyama discretization.
    """
    real = np.asarray(returns, dtype=float)
    if m <= 0:
        return np.empty((0, real.shape[1]))
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    mu_hat, sigma_hat = sample_moments(real, mle=True)
    N = real.shape[1]
    rng = np.random.default_rng(seed)

    if horizon == 0:
        return rng.multivariate_normal(mu_hat, factor_covariance(sigma_hat).matrix, size=m)

    prior_var = 1.0 - np.exp(-beta * horizon)
    X = rng.normal(size=(m, N)) * np.sqrt(max(prior_var, 1e-12))
    dt = horizon / n_steps

    for k in range(n_steps):
        t = k * dt
        u = max(horizon - t, 0.0)
        mu_u, sigma_u = forward_moments(mu_hat, sigma_hat, u, beta)
        solver = factor_covariance(sigma_u)
        # Evaluate C^-1(X-mu) directly; no explicit inverse is needed.
        precision_residual = solver.solve((X - mu_u).T).T
        drift = -beta * (precision_residual - 0.5 * X)
        X = X + drift * dt + np.sqrt(beta * dt) * rng.normal(size=X.shape)

    return X


def diffusion_augmented_moments(
    returns,
    m: int,
    horizon: float,
    beta: float = 1.0,
    n_steps: int = 200,
    seed: int | None = None,
):
    real = np.asarray(returns, dtype=float)
    fake = generate_reverse_diffusion_samples(
        real, m=m, horizon=horizon, beta=beta, n_steps=n_steps, seed=seed
    )
    combined = np.vstack([real, fake]) if len(fake) else real.copy()
    mu_aug, sigma_aug = sample_moments(combined, mle=True)
    return mu_aug, sigma_aug, fake, combined


def gaussian_score_diagnostics(mean, covariance, horizon, beta=1.0, n_steps=100):
    """Conditioning at the same Gaussian score times used by reverse integration."""
    if not np.isfinite(horizon) or horizon < 0 or not np.isfinite(beta) or beta <= 0 or n_steps < 1:
        raise ValueError('Invalid Gaussian score diagnostic horizon, beta or step count.')
    rows=[]
    times=[0.] if horizon==0 else [horizon-k*horizon/n_steps for k in range(n_steps)]
    for t in times:
        _,c=forward_moments(mean,covariance,t,beta)
        rows.append({'Forward time':t,**factor_covariance(c).diagnostics})
    return rows
