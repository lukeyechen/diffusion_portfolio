from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.data import download_yahoo_returns, load_returns_csv
from core.diffusion_exposure import completed_yahoo_returns
from core.feasible_tuning import TuningSettings, coefficients, fit_portfolios, gaussian_experiment, historical_backtest
from core.short_horizon_portfolio import HORIZON_PRESETS, aggregate_nonoverlapping, get_horizon_preset, performance_metrics


@st.cache_data(show_spinner=False, ttl=3600)
def _download(tickers, start, period):
    cfg = get_horizon_preset(period)
    base = download_yahoo_returns(tickers, start=start,
                                  interval="1wk" if cfg["source"] == "weekly" else "1mo")
    missing = set(tickers) - set(base.columns)
    if missing:
        raise ValueError("Missing Yahoo history: " + ", ".join(sorted(missing)))
    base = completed_yahoo_returns(base.loc[:, list(tickers)], source=cfg["source"])
    return aggregate_nonoverlapping(base, int(cfg["block_size"]))


@st.cache_data(show_spinner=False)
def _history(returns, settings, window, pilot_size, start, rf, cost, cap):
    return historical_backtest(returns, settings, window=window, pilot_size=pilot_size,
                               oos_start=start, rf=rf, cost_bps=cost, cap=cap)


@st.cache_data(show_spinner=False)
def _gaussian(mu, sigma, settings, n, repetitions, seed):
    return gaussian_experiment(mu, sigma, settings, n=n, repetitions=repetitions, seed=seed)


st.title("Feasible Tuning")
st.caption("Exact continuous-time Gaussian portfolio moments with fixed, pilot, same-sample ratio and trace tuning.")
experiment = st.radio("Experiment", ["Historical backtest", "Gaussian theorem check"], horizontal=True)
st.info(
    "The positive trace coefficient applies to asymptotic expected utility under the paper's Gaussian, "
    "unconstrained, frictionless assumptions. A stock backtest, weight constraints and trading costs "
    "are empirical comparisons. This is unconditional portfolio tuning, separate from Diffusion OLS."
)

s1, s2, s3 = st.columns(3)
gamma = s1.number_input("Risk aversion γ", min_value=0.01,
                        value=1.0 if experiment == "Gaussian theorem check" else 3.0,
                        key="feasible_gamma_" + experiment)
c = s2.number_input("Trace c (0 < c ≤ 4)", min_value=0.01, max_value=4.0, value=4.0)
epsilon = s3.number_input("Trace ε", min_value=0.000001, value=0.25, format="%.6f",
                          help="Positive variance-scale constant chosen before using the main sample.")
s4, s5, s6, s7 = st.columns(4)
a_max = s4.number_input("Maximum a", min_value=0.01, max_value=0.999, value=0.95)
fixed_b = s5.number_input("Fixed b comparator", min_value=0.001, value=7.0)
b_min = s6.number_input("Ratio b lower bound", min_value=0.001, value=1.0)
b_max = s7.number_input("Ratio b upper bound", min_value=0.002, value=20.0)
settings = TuningSettings(gamma=gamma, c=c, epsilon=epsilon, a_max=a_max,
                          fixed_b=fixed_b, b_min=b_min, b_max=b_max)
try:
    settings.validate()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Implemented rules and theorem scope"):
    st.latex(r"\hat b_{\rm tr}=c/(\varepsilon+\operatorname{tr}H),\qquad a_n=\min\{a_{\max},\hat b/n\}")
    st.latex(r"D=(1-a)I+aH,\quad m=(1-a)D^{-1}u,\quad S=H-a^2H^3D^{-2},\quad w=S^{-1}m/\gamma")
    st.latex(r"K_{2,\rm tr}=\gamma^{-1}\left[cA/D_\varepsilon+(2c-c^2/2)Q/D_\varepsilon^2\right]")
    st.write("Here A = (N+2)‖μ‖² + tr(Σ), Q = μᵀΣμ, and Dε = ε + tr(Σ). For c=4, the Q term cancels and the coefficient is positive, including μ=0.")
    st.write("The engine uses centered MLE covariance and exact proxy endpoint moments directly (the infinite-synthetic-sample limit). No Euler steps or real/synthetic mixture are applied.")
    st.write("Ratio rules are bounded A/Q plug-ins. Same-sample tuning includes the derivative interaction Ξg; consistency alone does not guarantee a gain. Independent pilot tuning supplies only b; classical and diffusion weights both use the same main block.")
    st.write("The pilot in the historical test is a preceding disjoint block. Such blocks are independent under i.i.d. sampling; real stock-return blocks need not be independent. The extra classical benchmark uses main plus pilot observations to show the cost of holding data out.")
    st.write("Choose c, ε and bounds before examining the main sample. The theorem is not a guarantee for parameters selected retrospectively to maximize this backtest. Returns are in decimal units and the identity reference makes coordinate units consequential.")

if experiment == "Gaussian theorem check":
    st.caption("Known population moments let us evaluate true utility, rather than treating in-sample fitted utility as truth. Defaults reproduce the one-asset negative-ratio example in your extension.")
    g1, g2, g3, g4 = st.columns(4)
    assets = g1.number_input("Number of assets", min_value=1, max_value=10, value=1)
    mu_scalar = g2.number_input("Population mean per asset", value=0.5, format="%.4f")
    variance = g3.number_input("Population variance per asset", min_value=0.000001, value=1.0)
    rho = g4.number_input("Common correlation", min_value=(-1/(assets-1)+0.001 if assets > 1 else -0.999),
                         max_value=0.999, value=0.0,
                         disabled=assets == 1)
    if assets == 1:
        rho = 0.0
    mu = np.full(assets, mu_scalar)
    sigma = variance * ((1-rho)*np.eye(assets) + rho*np.ones((assets, assets)))
    st.dataframe(coefficients(mu, sigma, settings), hide_index=True, use_container_width=True)
    g5, g6, g7 = st.columns(3)
    n = g5.number_input("Main sample size n", min_value=max(10, assets+5), max_value=5000, value=500)
    repetitions = g6.number_input("Monte Carlo repetitions", min_value=100, max_value=10000, value=1000)
    seed = g7.number_input("Random seed", min_value=0, value=42)
    st.caption("Each repetition uses an independent pilot of size n. Reported Monte Carlo standard errors measure simulation uncertainty; n² mean gain approaches K2 only asymptotically.")
    if st.button("Run Gaussian check", type="primary"):
        with st.spinner("Evaluating true Gaussian utility on paired samples..."):
            result = _gaussian(mu, sigma, settings, n, repetitions, seed)
        st.dataframe(result.style.format(precision=6), hide_index=True, use_container_width=True)
        st.caption("The main-plus-pilot classical row uses twice as much estimation data. No K2 from the pilot corollary is assigned to that different-data benchmark.")
    st.stop()

source = st.radio("Data source", ["Yahoo Finance", "Upload CSV"], horizontal=True)
d1, d2, d3 = st.columns([2, 1, 1])
tickers = tuple(dict.fromkeys(t.strip().upper() for t in d1.text_input("Tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN").split(",") if t.strip()))
start = d2.text_input("Data history starts", "2000-01-01")
period = d3.selectbox("Holding / rebalance interval", list(HORIZON_PRESETS))
returns = None
if source == "Yahoo Finance":
    signature = (tickers, start, period)
    if st.button("Download / refresh feasible-tuning data"):
        try:
            with st.spinner("Downloading completed return periods..."):
                downloaded = _download(tickers, start, period)
            st.session_state["feasible_data"] = (signature, downloaded)
        except Exception as exc:
            st.error(f"Data download failed: {exc}")
    saved = st.session_state.get("feasible_data")
    if saved is not None and saved[0] == signature:
        returns = saved[1]
else:
    upload = st.file_uploader("CSV of already sampled holding-period stock returns", type="csv")
    if upload is not None:
        returns = load_returns_csv(upload)
if returns is None:
    st.info("Download or upload return data to continue.")
    st.stop()

ppy = int(get_horizon_preset(period)["periods_per_year"])
h1, h2, h3, h4 = st.columns(4)
window = h1.number_input("Main estimation window", min_value=max(10, returns.shape[1]+1), value=120)
use_pilot = h2.checkbox("Compare pilot tuning", value=True)
pilot_size = h2.number_input("Pilot observations", min_value=max(10, returns.shape[1]+1), value=120, disabled=not use_pilot)
annual_rf = h3.number_input("Annual risk-free return (%)", min_value=-99.0, value=0.0)
cost = h4.number_input("Trading cost (bps per turnover)", min_value=0.0, max_value=500.0, value=0.0)
rf = (1+annual_rf/100)**(1/ppy)-1
oos_start = st.text_input("Backtest start date", "2019-01-01")
allocation = st.radio("Allocation", ["Theorem weights (unconstrained)", "Long-only with cash (empirical comparison)"], horizontal=True)
cap = None
if allocation.startswith("Long-only"):
    cap = st.number_input("Maximum weight per stock", min_value=0.01, max_value=1.0, value=0.4)
else:
    st.caption("Unconstrained weights can be negative or sum above 100%; cash can represent borrowing. They are not normalized to a fully invested long-only portfolio.")

signature = (returns.to_json(), settings, window, pilot_size if use_pilot else 0, oos_start, rf, cost, cap)
if st.button("Run feasible-tuning backtest", type="primary"):
    try:
        with st.spinner("Comparing exact proxy allocations through history..."):
            result = _history(returns, settings, window, pilot_size if use_pilot else 0,
                              oos_start, rf, cost, cap)
        st.session_state["feasible_result"] = (signature, result)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))
saved = st.session_state.get("feasible_result")
if saved is None or saved[0] != signature:
    st.stop()
result = saved[1]
st.subheader("Historical results")
st.caption(f"{result['Date'].min().date()} to {result['Date'].max().date()}; all methods use the same evaluated periods. Returns cover the complete test.")
summary, wealth = [], {}
for method, group in result.groupby("Method", sort=False):
    net = group["Net return"].to_numpy()
    metrics = performance_metrics(net, gamma=gamma, periods_per_year=ppy)
    excess = net-rf
    summary.append({"Method": method, "Total return": metrics["Total return"], "CAGR": metrics["CAGR"],
                    "Volatility": metrics["Annualized vol"], "Max drawdown": metrics["Max drawdown"],
                    "Annualized MV excess": ppy*(excess.mean()-gamma/2*np.var(excess, ddof=1)) if len(net)>1 else np.nan,
                    "Average turnover": group["Turnover"].mean()})
    wealth[method] = pd.Series(np.cumprod(1+net), index=group["Date"])
st.dataframe(pd.DataFrame(summary).style.format({k: "{:.2%}" for k in summary[0] if k != "Method"}), hide_index=True, use_container_width=True)
st.line_chart(pd.DataFrame(wealth), use_container_width=True)
st.caption("Wealth index starts at 1. Historical MV statistics do not estimate the theorem's population K2 directly.")
pilot = returns.iloc[-window-pilot_size:-window].to_numpy()-rf if use_pilot else None
try:
    latest, tuning = fit_portfolios(returns.iloc[-window:].to_numpy()-rf, settings, pilot=pilot, cap=cap)
    st.subheader("Next-period target weights")
    st.dataframe(pd.DataFrame([{"Method": name, **dict(zip(returns.columns, w)), "Cash": 1-w.sum()} for name,w in latest.items()]).style.format({k: "{:.2%}" for k in [*returns.columns,"Cash"]}), hide_index=True, use_container_width=True)
    st.subheader("Latest tuning values")
    st.dataframe(tuning, hide_index=True, use_container_width=True)
    st.caption("An active a cap changes the finite-sample rule. For sufficiently large n the cap becomes inactive under the bounded rules.")
except ValueError as exc:
    st.error(str(exc))
with st.expander("Rolling detail"):
    st.dataframe(result, hide_index=True, use_container_width=True)
st.download_button("Download feasible-tuning history", result.to_csv(index=False),
                   "feasible_tuning_history.csv", "text/csv")
