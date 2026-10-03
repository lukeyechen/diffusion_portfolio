from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.data import download_yahoo_returns, load_returns_csv
from core.diffusion_exposure import completed_yahoo_returns
from core.feasible_tuning import (OLD_METHOD, TradingComparison, TuningSettings, coefficients,
                                 gaussian_experiment, historical_backtest, latest_portfolios)
from core.short_horizon_portfolio import CANDIDATE_T, HORIZON_PRESETS, aggregate_nonoverlapping, get_horizon_preset, performance_metrics


def _remember_input(name, widget_key):
    st.session_state.setdefault("feasible_inputs", {})[name] = st.session_state[widget_key]


def _input(container, kind, label, *, key, **kwargs):
    """Keep values outside Streamlit's widget state, which is cleared off-page."""
    values = st.session_state.setdefault("feasible_inputs", {})
    widget_key = "_feasible_widget_" + key
    if widget_key not in st.session_state and key in values:
        value = values[key]
        if kind == "number_input":
            value = max(value, kwargs.get("min_value", value))
            value = min(value, kwargs.get("max_value", value))
        st.session_state[widget_key] = value
    value = getattr(container, kind)(label, key=widget_key, on_change=_remember_input,
                                     args=(key, widget_key), **kwargs)
    values[key] = value
    return value


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
def _history(returns, settings, window, start, rf, cost, cap, trading):
    return historical_backtest(returns, settings, window=window,
                               oos_start=start, rf=rf, cost_bps=cost, cap=cap, trading=trading)


@st.cache_data(show_spinner=False)
def _latest(returns, settings, history, window, rf, cap, trading):
    return latest_portfolios(returns, settings, history, window=window,
                             rf=rf, cap=cap, trading=trading)


@st.cache_data(show_spinner=False)
def _gaussian(mu, sigma, settings, n, repetitions, seed):
    return gaussian_experiment(mu, sigma, settings, n=n, repetitions=repetitions, seed=seed)


st.title("Feasible Tuning")
st.caption("Trace tuning, one same-sample ratio comparator, and a comparison with the old Portfolio method.")
experiment = _input(st, "radio", "Experiment", options=["Historical backtest", "Gaussian theorem check"], horizontal=True, key="experiment")
st.info(
    "The positive trace coefficient applies to asymptotic expected utility under the paper's Gaussian, "
    "unconstrained, frictionless assumptions. A stock backtest, weight constraints and trading costs "
    "are empirical comparisons. This is unconditional portfolio tuning, separate from Diffusion OLS."
)

shared = st.session_state.get("shared_current_window", {})
has_shared = (isinstance(shared.get("full_returns"), pd.DataFrame)
              and shared.get("holding_period") in HORIZON_PRESETS)
use_shared = False
if experiment == "Historical backtest" and has_shared:
    use_shared = _input(st, "checkbox", "Use Portfolio tab's saved data and settings", value=True, key="use_shared")
elif experiment == "Historical backtest":
    st.session_state.setdefault("feasible_inputs", {}).setdefault("use_shared", False)
defaults = shared if use_shared else {}

s1, s2, s3 = st.columns(3)
gamma = _input(s1, "number_input", "Risk aversion γ", min_value=0.01,
                        value=1.0 if experiment == "Gaussian theorem check" else float(defaults.get("gamma", 3.0)),
                        key="feasible_gamma_" + experiment + str(use_shared))
c = _input(s2, "number_input", "Trace c (0 < c ≤ 4)", min_value=0.01, max_value=4.0, value=4.0, key="trace_c")
epsilon = _input(s3, "number_input", "Trace ε", min_value=0.000001, value=0.25, format="%.6f", key="trace_epsilon",
                          help="Positive variance-scale constant chosen before using the main sample.")
s4, s5, s6, s7 = st.columns(4)
a_max = _input(s4, "number_input", "Maximum a", min_value=0.01, max_value=0.999, value=0.95, key="a_max")
fixed_b = _input(s5, "number_input", "Fixed b comparator", min_value=0.001, value=7.0, key="fixed_b")
b_min = _input(s6, "number_input", "Ratio b lower bound", min_value=0.001, value=1.0, key="b_min")
b_max = _input(s7, "number_input", "Ratio b upper bound", min_value=0.002, value=20.0, key="b_max")
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
    st.write("The direct tuning rules use centered MLE covariance and exact continuous-time endpoint moments, without Euler steps or a real/synthetic mixture. The old comparator retains its original finite-step moments and mixture.")
    st.write("The retained ratio method is the bounded same-sample A/Q plug-in. Its derivative interaction Ξg means consistency alone does not guarantee a gain. All displayed methods use the same main estimation block; no pilot block is held out.")
    st.write("Choose c, ε and bounds before examining the main sample. The theorem is not a guarantee for parameters selected retrospectively to maximize this backtest. Returns are in decimal units and the identity reference makes coordinate units consequential.")

if experiment == "Gaussian theorem check":
    st.caption("Known population moments let us evaluate true utility, rather than treating in-sample fitted utility as truth. Defaults reproduce the one-asset negative-ratio example in your extension.")
    g1, g2, g3, g4 = st.columns(4)
    assets = _input(g1, "number_input", "Number of assets", min_value=1, max_value=10, value=1, key="gaussian_assets")
    mu_scalar = _input(g2, "number_input", "Population mean per asset", value=0.5, format="%.4f", key="gaussian_mean")
    variance = _input(g3, "number_input", "Population variance per asset", min_value=0.000001, value=1.0, key="gaussian_variance")
    rho = _input(g4, "number_input", "Common correlation", min_value=(-1/(assets-1)+0.001 if assets > 1 else -0.999), key=f"gaussian_rho_{assets}",
                         max_value=0.999, value=0.0,
                         disabled=assets == 1)
    if assets == 1:
        rho = 0.0
    mu = np.full(assets, mu_scalar)
    sigma = variance * ((1-rho)*np.eye(assets) + rho*np.ones((assets, assets)))
    population = coefficients(mu, sigma, settings)
    st.dataframe(population[population["Method"] != "Pilot ratio"], hide_index=True, use_container_width=True)
    g5, g6, g7 = st.columns(3)
    n = _input(g5, "number_input", "Main sample size n", min_value=max(10, assets+5), max_value=5000, value=500, key=f"gaussian_n_{assets}")
    repetitions = _input(g6, "number_input", "Monte Carlo repetitions", min_value=100, max_value=10000, value=1000, key="gaussian_repetitions")
    seed = _input(g7, "number_input", "Random seed", min_value=0, value=42, key="gaussian_seed")
    st.caption("Each repetition uses one main sample of size n. Reported Monte Carlo standard errors measure simulation uncertainty; n² mean gain approaches K2 only asymptotically. The old constrained trading strategy is compared in the historical mode.")
    gaussian_signature = (tuple(mu), tuple(map(tuple, sigma)), settings, n, repetitions, seed)
    if st.button("Run Gaussian check", type="primary"):
        with st.spinner("Evaluating true Gaussian utility on paired samples..."):
            result = _gaussian(mu, sigma, settings, n, repetitions, seed)
        st.session_state["feasible_gaussian_result"] = (gaussian_signature, result)
    saved_gaussian = st.session_state.get("feasible_gaussian_result")
    if saved_gaussian is not None and saved_gaussian[0] == gaussian_signature:
        st.dataframe(saved_gaussian[1].style.format(precision=6), hide_index=True, use_container_width=True)
    st.stop()

if use_shared:
    returns = shared["full_returns"].copy()
    period = shared["holding_period"]
    st.caption(f"Using the Portfolio tab's saved data: {', '.join(returns.columns)}; {period}; data through {pd.Timestamp(returns.index[-1]).date()}. Settings below can be adjusted before running.")
    if shared.get("mean_model", "Original diffusion mean") != "Original diffusion mean":
        st.warning("The old-method comparator uses Original diffusion mean. Your saved Portfolio used a different mean model, so this row will not reproduce that saved forecast-based recommendation.")
else:
    source = _input(st, "radio", "Data source", options=["Yahoo Finance", "Upload CSV"], horizontal=True, key="source")
    d1, d2, d3 = st.columns([2, 1, 1])
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in _input(d1, "text_input", "Tickers", value="AAPL,MSFT,NVDA,GOOGL,AMZN", key="tickers").split(",") if t.strip()))
    start = _input(d2, "text_input", "Data history starts", value="2000-01-01", key="history_start")
    period = _input(d3, "selectbox", "Holding / rebalance interval", options=list(HORIZON_PRESETS), key="period")
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
            st.session_state["feasible_uploaded_returns"] = (upload.name, returns)
        elif "feasible_uploaded_returns" in st.session_state:
            filename, returns = st.session_state["feasible_uploaded_returns"]
            st.caption(f"Using saved uploaded returns: {filename}.")
if returns is None:
    st.info("Download or upload return data to continue.")
    st.stop()

cfg = get_horizon_preset(period)
ppy = int(cfg["periods_per_year"])
allocation = _input(st, "radio", "Allocation", options=["Turnover-controlled comparison with old Portfolio",
                                     "Theorem weights (unconstrained)",
                                     "Long-only with cash (empirical comparison)"], horizontal=True, key="allocation")
practical = allocation.startswith("Turnover-controlled")
required = int(cfg["min_train_size"]) + int(cfg["inner_folds"])*int(cfg["validation_size"])
minimum = max(10, returns.shape[1]+1, required if practical else 0)
h1, h2, h3 = st.columns(3)
window = _input(h1, "number_input", "Main estimation window", min_value=minimum,
                         value=max(minimum, int(defaults.get("lookback", cfg["lookback"]))),
                         key=f"feasible_window_{period}_{use_shared}_{practical}",
                         help="Every method uses this same trailing block. Weekly Portfolio preset: 520 observations.")
annual_rf = _input(h2, "number_input", "Annual risk-free return (%)", min_value=-99.0, value=0.0, key=f"riskfree_{practical}",
                            disabled=practical)
cost = _input(h3, "number_input", "Trading cost (bps per turnover)", min_value=0.0, max_value=500.0,
                       value=25.0 if practical else 0.0, key=f"feasible_cost_{practical}")
if practical:
    annual_rf = 0.0
rf = (1+annual_rf/100)**(1/ppy)-1
oos_start = _input(st, "text_input", "Backtest / strategy replay start date", value=str(defaults.get("replay_start", "2019-01-01")), key=f"replay_start_{use_shared}")
cap = None
trading = None
if practical:
    st.info("All methods use the same window, risk aversion, fully invested weight cap, turnover penalty and partial rebalance step. Each method replays its own holdings from equal weights on the same start date. Trace, fixed b and ratio use direct endpoint moments; the old method keeps its original Best-T grid, finite-step moments and real/synthetic mixture. This trading comparison is empirical.")
    t1, t2, t3 = st.columns(3)
    cap = _input(t1, "number_input", "Maximum weight per stock", min_value=1.0/returns.shape[1], max_value=1.0,
                          value=max(1.0/returns.shape[1], float(defaults.get("max_long_weight", 0.4))),
                          key=f"feasible_trading_cap_{use_shared}")
    penalty = _input(t2, "number_input", "Turnover penalty (bps)", min_value=0.0, max_value=100.0,
                              value=10000*float(defaults.get("turnover_penalty", 0.0025)),
                              key=f"feasible_penalty_{use_shared}")
    alpha = _input(t3, "number_input", "Rebalance step (%)", min_value=1.0, max_value=100.0,
                            value=100*float(defaults.get("rebalance_alpha", 0.5 if period == "1 week" else 1.0)),
                            key=f"feasible_alpha_{period}_{use_shared}")
    with st.expander("Old method settings"):
        t4, t5, t6 = st.columns(3)
        m = _input(t4, "number_input", "Old-method synthetic-equivalent M", min_value=0, value=int(defaults.get("m", 500)), key=f"old_m_{use_shared}")
        beta = _input(t5, "number_input", "Constant β", min_value=0.01, value=float(defaults.get("beta", 1.0)), key=f"old_beta_{use_shared}")
        steps = _input(t6, "number_input", "Old-method reverse SDE steps", min_value=10, value=int(defaults.get("n_steps", 100)), key=f"old_steps_{use_shared}")
        st.caption(f"Old method candidate T grid: {CANDIDATE_T}. Trace tuning calculates T directly.")
    trading = TradingComparison(inner_folds=int(cfg["inner_folds"]),
                                validation_size=int(cfg["validation_size"]),
                                min_train_size=int(cfg["min_train_size"]),
                                turnover_penalty=penalty/10000, rebalance_alpha=alpha/100,
                                cap=cap, m=m, beta=beta, n_steps=steps)
if allocation.startswith("Long-only"):
    cap = _input(st, "number_input", "Maximum weight per stock", min_value=0.01, max_value=1.0, value=0.4, key="cash_cap")
elif not practical:
    st.caption("Unconstrained weights can be negative or sum above 100%; cash can represent borrowing. They are not normalized to a fully invested long-only portfolio.")

signature = (returns.to_json(), settings, window, oos_start, rf, cost, cap, trading)
if st.button("Run feasible-tuning backtest", type="primary"):
    try:
        with st.spinner("Replaying the common comparison; the old method runs nested T validation each period..." if practical else "Comparing direct allocations through history..."):
            result = _history(returns, settings, window, oos_start, rf, cost, cap, trading)
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
if practical:
    table = pd.DataFrame(summary).set_index("Method")
    st.subheader("Trace tuning versus old Portfolio")
    compare = table.loc[[OLD_METHOD, "Trace tuning"]].reset_index()
    st.dataframe(compare.style.format({k: "{:.2%}" for k in compare.columns if k != "Method"}), hide_index=True, use_container_width=True)
    st.caption(f"Trace − old total return: {100*(table.loc['Trace tuning', 'Total return']-table.loc[OLD_METHOD, 'Total return']):+.2f} percentage points. Trace − old annualized MV excess: {100*(table.loc['Trace tuning', 'Annualized MV excess']-table.loc[OLD_METHOD, 'Annualized MV excess']):+.2f} percentage points. This is the complete historical test.")
try:
    latest, tuning = _latest(returns, settings, result, window, rf, cap, trading)
    st.subheader("Next-period target weights")
    st.caption(f"Common estimation block: {pd.Timestamp(returns.index[-window]).date()} to {pd.Timestamp(returns.index[-1]).date()}, n={window}, γ={gamma:g}. " + (f"Turnover penalty {penalty:g} bps, rebalance step {alpha:g}%, cap {cap:.0%}." if practical else "No turnover penalty or partial-rebalance adjustment to the targets."))
    st.dataframe(pd.DataFrame([{"Method": name, **dict(zip(returns.columns, w)), "Cash": 1-w.sum()} for name,w in latest.items()]).style.format({k: "{:.2%}" for k in [*returns.columns,"Cash"]}), hide_index=True, use_container_width=True)
    st.subheader("Latest tuning values")
    st.dataframe(tuning, hide_index=True, use_container_width=True)
    st.caption("An active a cap changes the finite-sample rule. For sufficiently large n the cap becomes inactive under the bounded rules.")
except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
    st.error(str(exc))
with st.expander("Rolling detail"):
    st.dataframe(result, hide_index=True, use_container_width=True)
st.download_button("Download feasible-tuning history", result.to_csv(index=False),
                   "feasible_tuning_history.csv", "text/csv")
