from __future__ import annotations
import numpy as np


def portfolio_mean(weights, mu):
    return float(np.asarray(weights) @ np.asarray(mu))


def portfolio_variance(weights, sigma):
    w = np.asarray(weights)
    return float(w @ np.asarray(sigma) @ w)


def portfolio_volatility(weights, sigma):
    return float(np.sqrt(max(portfolio_variance(weights, sigma), 0.0)))


def sharpe_ratio(weights, mu, sigma):
    vol = portfolio_volatility(weights, sigma)
    return np.nan if vol <= 0 else portfolio_mean(weights, mu) / vol


def certainty_equivalent(weights, mu, sigma, gamma: float = 3.0):
    return portfolio_mean(weights, mu) - 0.5 * gamma * portfolio_variance(weights, sigma)


def gross_exposure(weights):
    return float(np.abs(np.asarray(weights)).sum())


def net_exposure(weights):
    return float(np.asarray(weights).sum())


def turnover(weights_new, weights_old):
    return float(np.abs(np.asarray(weights_new) - np.asarray(weights_old)).sum())
