from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import importlib
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from core.data import download_yahoo_returns, load_returns_csv
from core.diffusion_exposure import completed_yahoo_returns
# The manual exec-based router can retain the previous module during a live
# deployment. Refresh it only when the new export is missing, preserving normal
# rerun class identities and saved result signatures.
import core.feasible_tuning as _feasible_module
if (not hasattr(_feasible_module, "trace_calibration")
        or not hasattr(_feasible_module, "epsilon_sensitivity")
        or getattr(_feasible_module, "TRACE_CALIBRATION_VERSION", None) != 2
        or getattr(_feasible_module, "COMPARISON_VERSION", None) != 4
        or getattr(_feasible_module, "CLASSICAL_MV_RULE", None) != "allocation-aware-classical-v2"):
    importlib.reload(_feasible_module)
from core.feasible_tuning import (OLD_METHOD, TradingComparison, TuningSettings, coefficients,
                                 epsilon_sensitivity, trace_calibration, gaussian_experiment, historical_backtest, latest_portfolios)
from core.short_horizon_portfolio import CANDIDATE_T, HORIZON_PRESETS, aggregate_nonoverlapping, get_horizon_preset, performance_metrics


def _remember_input(name, widget_key):
    st.session_state.setdefault("feasible_inputs", {})[name] = st.session_state[widget_key]


def _input(container, kind, label, *, key, sync=False, **kwargs):
    """Keep values outside Streamlit's widget state, which is cleared off-page."""
    values = st.session_state.setdefault("feasible_inputs", {})
    widget_key = "_feasible_widget_" + key
    if sync:
        st.session_state[widget_key] = kwargs["value"]
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
    # Comparison v4: Trace tuning only; allocation-aware Classical MV: invalidate pre-fix cached results.
    return historical_backtest(returns, settings, window=window,
                               oos_start=start, rf=rf, cost_bps=cost, cap=cap, trading=trading)


@st.cache_data(show_spinner=False)
def _latest(returns, settings, history, window, rf, cap, trading):
    # Comparison v4: Trace tuning only; allocation-aware Classical MV.
    return latest_portfolios(returns, settings, history, window=window,
                             rf=rf, cap=cap, trading=trading, return_raw=True)


@st.cache_data(show_spinner=False)
def _epsilon(returns, settings, candidates, window, start, evaluation_start, ppy, rf, cost, cap, trading, end_date):
    # Comparison v4: Trace tuning only; allocation-aware Classical MV.
    return epsilon_sensitivity(returns, settings, candidates, window=window,
                               calibration_start=start, evaluation_start=evaluation_start,
                               periods_per_year=ppy, rf=rf, cost_bps=cost, cap=cap, trading=trading, end_date=end_date)


@st.cache_data(show_spinner=False)
def _trace_grid(returns, settings, epsilons, cs, window, start, evaluation_start, ppy, rf, cost, cap, trading, end_date):
    return trace_calibration(returns, settings, epsilons, cs=cs, window=window,
                             calibration_start=start, evaluation_start=evaluation_start,
                             periods_per_year=ppy, rf=rf, cost_bps=cost, cap=cap, trading=trading, end_date=end_date)


def _label(name, practical):
    return "Classical MV" if name == "Classical (main sample)" else name


def _weight_table(portfolios, assets, practical):
    return pd.DataFrame([{"Method": _label(name, practical), **dict(zip(assets, w)),
                          "Cash": 1-w.sum()} for name, w in portfolios.items()])


@st.cache_data(show_spinner=False)
def _gaussian(mu, sigma, settings, n, repetitions, seed):
    # Comparison v4 excludes Fixed b and Same-sample ratio.
    return gaussian_experiment(mu, sigma, settings, n=n, repetitions=repetitions, seed=seed)


st.title("Feasible Tuning")
st.caption("Trace tuning, compared with Classical MV and the old Portfolio method.")
experiment = _input(st, "radio", "Experiment", options=["Historical backtest", "Gaussian theorem check"], horizontal=True, key="experiment")
st.info(
    "The positive trace coefficient applies to asymptotic expected utility under the paper's Gaussian, "
    "unconstrained, frictionless assumptions. A stock backtest, weight constraints and trading costs "
    "are empirical comparisons. This is unconditional portfolio tuning, separate from Diffusion OLS."
)
st.caption("Classical MV uses sample moments and respects the selected allocation constraints. It has no tuning, turnover penalty or partial rebalance. Theorem mode uses the unconstrained formula. Realized trading costs are deducted separately in the backtest.")

shared = st.session_state.get("shared_current_window", {})
has_shared = (isinstance(shared.get("full_returns"), pd.DataFrame)
              and shared.get("holding_period") in HORIZON_PRESETS)
use_shared = False
if experiment == "Historical backtest" and has_shared:
    use_shared = _input(st, "checkbox", "Use Portfolio tab's saved data and settings", value=True, key="use_shared")
elif experiment == "Historical backtest":
    st.session_state.setdefault("feasible_inputs", {}).setdefault("use_shared", False)
defaults = shared if use_shared else {}

gamma = _input(st, "number_input", "Risk aversion γ", min_value=0.01,
                        value=1.0 if experiment == "Gaussian theorem check" else float(defaults.get("gamma", 3.0)),
                        sync=use_shared, disabled=use_shared,
                        key="feasible_gamma_" + experiment + str(use_shared))
# Historical c and epsilon are selected jointly below; a_max is an internal safeguard.
settings = TuningSettings(gamma=gamma, c=4.0, epsilon=0.25, a_max=0.95)
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
    st.write("All displayed methods use the same main estimation block.")
    st.write("Choose c and ε before examining the main sample. The theorem is not a guarantee for parameters selected retrospectively to maximize this backtest. Returns are in decimal units and the identity reference makes coordinate units consequential.")

if experiment == "Gaussian theorem check":
    st.caption("The Gaussian theorem check uses c = 4, ε = 0.25 and maximum a = 0.95. Automatic historical calibration is available in Historical backtest mode.")
    st.caption("Known population moments let us evaluate true utility, rather than treating in-sample fitted utility as truth. Compare Trace tuning with the classical estimator.")
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
    st.dataframe(population[~population["Method"].isin(["Fixed b", "Pilot ratio", "Same-sample ratio"])], hide_index=True, use_container_width=True)
    g5, g6, g7 = st.columns(3)
    n = _input(g5, "number_input", "Main sample size n", min_value=max(10, assets+5), max_value=5000, value=500, key=f"gaussian_n_{assets}")
    repetitions = _input(g6, "number_input", "Monte Carlo repetitions", min_value=100, max_value=10000, value=1000, key="gaussian_repetitions")
    seed = _input(g7, "number_input", "Random seed", min_value=0, value=42, key="gaussian_seed")
    st.caption("Each repetition uses one main sample of size n. Reported Monte Carlo standard errors measure simulation uncertainty; n² mean gain approaches K2 only asymptotically. The old constrained trading strategy is compared in the historical mode.")
    gaussian_signature = (4, tuple(mu), tuple(map(tuple, sigma)), settings, n, repetitions, seed)
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
    st.caption(f"Using the Portfolio tab's saved data: {', '.join(returns.columns)}; {period}; data through {pd.Timestamp(returns.index[-1]).date()}. Common settings are locked to the saved recommendation; turn off the shared-data option to edit them independently.")
    if not shared.get("replay_start"):
        st.warning("This older Portfolio result did not save its replay date. Rerun Portfolio to publish a complete comparison snapshot. The fallback replay date is six calendar months before today.")
    if shared.get("mean_model", "Original diffusion mean") != "Original diffusion mean":
        st.info("The old Portfolio comparator uses your saved forecast model. The classical and direct tuning methods estimate unconditional sample means, as specified by the feasible-tuning experiment.")
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
if use_shared:
    cfg.update(shared.get("validation_config", {}))
ppy = int(cfg["periods_per_year"])
if use_shared:
    st.session_state["_feasible_widget_allocation"] = "Turnover-controlled comparison with old Portfolio"
allocation = _input(st, "radio", "Allocation", options=["Turnover-controlled comparison with old Portfolio",
                                     "Theorem weights (unconstrained)",
                                     "Long-only with cash (empirical comparison)"], horizontal=True, key="allocation", disabled=use_shared)
practical = allocation.startswith("Turnover-controlled")
required = int(cfg["min_train_size"]) + int(cfg["inner_folds"])*int(cfg["validation_size"])
minimum = max(10, returns.shape[1]+1, required if practical else 0)
h1, h2, h3 = st.columns(3)
window = _input(h1, "number_input", "Main estimation window", min_value=minimum,
                         value=max(minimum, int(defaults.get("lookback", cfg["lookback"]))),
                         sync=use_shared, disabled=use_shared,
                         key=f"feasible_window_{period}_{use_shared}_{practical}",
                         help="Every method uses this same trailing block. Weekly Portfolio preset: 520 observations.")
annual_rf = _input(h2, "number_input", "Annual risk-free return (%)", min_value=-99.0, value=0.0, key=f"riskfree_{practical}",
                            disabled=practical)
cost = _input(h3, "number_input", "Trading cost (bps per turnover)", min_value=0.0, max_value=500.0,
                       value=25.0 if practical else 0.0, key=f"feasible_cost_{practical}")
if practical:
    annual_rf = 0.0
rf = (1+annual_rf/100)**(1/ppy)-1
replay_default = (pd.Timestamp(datetime.now(ZoneInfo("America/Havana")).date()) - pd.DateOffset(months=6)).date().isoformat()
oos_start = _input(st, "text_input", "Backtest / strategy replay start date", value=str(defaults.get("replay_start") or replay_default), key=f"replay_start_{use_shared}", sync=use_shared, disabled=use_shared)
cap = None
trading = None
if practical:
    st.info("All methods use the same data, window, risk aversion and replay start. Classical MV uses the same fully invested long-only weight cap, without turnover penalties or partial rebalancing. The other strategies share the displayed cap, turnover penalty and partial rebalance step. Each method replays its own holdings from equal weights. This comparison includes differences in trading controls as well as estimators.")
    t1, t2, t3 = st.columns(3)
    cap = _input(t1, "number_input", "Maximum weight per stock", min_value=1.0/returns.shape[1], max_value=1.0,
                          value=max(1.0/returns.shape[1], float(defaults.get("max_long_weight", 0.4))),
                          sync=use_shared, disabled=use_shared,
                          key=f"feasible_trading_cap_{use_shared}")
    penalty = _input(t2, "number_input", "Turnover penalty (bps)", min_value=0.0, max_value=100.0,
                              value=10000*float(defaults.get("turnover_penalty", 0.0025)),
                              sync=use_shared, disabled=use_shared,
                              key=f"feasible_penalty_{use_shared}")
    alpha = _input(t3, "number_input", "Rebalance step (%)", min_value=1.0, max_value=100.0,
                            value=100*float(defaults.get("rebalance_alpha", 0.5 if period == "1 week" else 1.0)),
                            sync=use_shared, disabled=use_shared,
                            key=f"feasible_alpha_{period}_{use_shared}")
    with st.expander("Old method settings"):
        t4, t5, t6 = st.columns(3)
        m = _input(t4, "number_input", "Old-method synthetic-equivalent M", min_value=0, value=int(defaults.get("m", 500)), key=f"old_m_{use_shared}", sync=use_shared, disabled=use_shared)
        beta = _input(t5, "number_input", "Constant β", min_value=0.01, value=float(defaults.get("beta", 1.0)), key=f"old_beta_{use_shared}", sync=use_shared, disabled=use_shared)
        steps = _input(t6, "number_input", "Old-method reverse SDE steps", min_value=10, value=int(defaults.get("n_steps", 100)), key=f"old_steps_{use_shared}", sync=use_shared, disabled=use_shared)
        st.caption(f"Old method candidate T grid: {CANDIDATE_T}. Trace tuning calculates T directly.")
    trading = TradingComparison(inner_folds=int(cfg["inner_folds"]),
                                validation_size=int(cfg["validation_size"]),
                                min_train_size=int(cfg["min_train_size"]),
                                turnover_penalty=penalty/10000, rebalance_alpha=alpha/100,
                                cap=cap, m=m, beta=beta, n_steps=steps,
                                mean_model=defaults.get("mean_model", "Original diffusion mean"))
    st.caption(f"Old Portfolio expected-return model: {trading.mean_model}. Replay start: {oos_start}.")
if allocation.startswith("Long-only"):
    cap = _input(st, "number_input", "Maximum weight per stock", min_value=0.01, max_value=1.0, value=0.4, key="cash_cap")
elif not practical:
    st.caption("Unconstrained weights can be negative or sum above 100%; cash can represent borrowing. They are not normalized to a fully invested long-only portfolio.")

signature = (_feasible_module.COMPARISON_VERSION, _feasible_module.CLASSICAL_MV_RULE, returns.to_json(), settings, window, oos_start, rf, cost, cap, trading)
history_default = str(pd.Timestamp(returns.index[0]).date()) if use_shared else start
today_default = datetime.now(ZoneInfo("America/Havana")).date().isoformat()

with st.expander("Automatic Trace calibration"):
    st.caption("Run automatically selects c and ε jointly using data before the evaluation split, then freezes them for the main backtest. You can adjust the candidate grids and dates here.")
    cs_text = _input(st, "text_input", "c candidates (0 < c ≤ 4; maximum 12)", value="0.25,0.5,1,2,4", key="trace_cs")
    eps_text = _input(st, "text_input", "Joint ε candidates (maximum 12)", value="0.001,0.01,0.05,0.10,0.25", key="trace_eps")
    grid_start = _input(st, "text_input", "c calibration starts", value=history_default, key="trace_start")
    grid_final = oos_start
    st.caption(f"Calibration ends before the main backtest starts: {grid_final}.")
    grid_end = _input(st, "text_input", "c calibration / evaluation end date", value=today_default, key="trace_end")
    grid_signature = (signature, cs_text, eps_text, grid_start, grid_final, ppy, grid_end)
    if st.button("Run c calibration and final evaluation"):
        try:
            cs = tuple(float(v.strip()) for v in cs_text.split(",") if v.strip())
            eps = tuple(float(v.strip()) for v in eps_text.split(",") if v.strip())
            with st.spinner("Calibrating trace settings on past data, then evaluating the frozen pair..."):
                grid_result = _trace_grid(returns, settings, eps, cs, window, grid_start, grid_final, ppy, rf, cost, cap, trading, grid_end)
            st.session_state["feasible_trace_grid_result"] = (grid_signature, grid_result)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            st.error(str(exc))
    saved_grid = st.session_state.get("feasible_trace_grid_result")
    if saved_grid is not None and saved_grid[0] == grid_signature:
        grid_result = saved_grid[1]
        st.write(f"Frozen c: {grid_result['selected_c']:g}; frozen ε: {grid_result['selected_epsilon']:g}. Calibration through {pd.Timestamp(grid_result['calibration_through']).date()}.")
        st.write("c calibration results")
        grid_formats = {k: "{:.3%}" for k in grid_result["calibration"].columns if k not in ("c", "ε", "Periods")}
        st.dataframe(grid_result["calibration"].style.format(grid_formats), hide_index=True, use_container_width=True)
        st.write("Final evaluation: frozen c–ε pair versus Classical MV")
        st.success(f"Selected calibration settings used below: c = {grid_result['selected_c']:g}, ε = {grid_result['selected_epsilon']:g}.")
        grid_table = grid_result["evaluation_summary"].copy()
        grid_table.insert(1, "Selected c", np.where(grid_table["Method"] == "Trace tuning", grid_result["selected_c"], np.nan))
        grid_table.insert(2, "Selected ε", np.where(grid_table["Method"] == "Trace tuning", grid_result["selected_epsilon"], np.nan))
        grid_formats.update({"Selected c": "{:.6g}", "Selected ε": "{:.6g}"})
        grid_table["Method"] = grid_table["Method"].map(lambda name: _label(name, practical))
        for column in ("Selected c", "Selected ε"):
            grid_table[column] = grid_table[column].map(lambda value: "—" if pd.isna(value) else f"{value:.6g}")
        grid_formats = {key: value for key, value in grid_formats.items() if key not in ("Selected c", "Selected ε")}
        st.dataframe(grid_table.style.format(grid_formats, na_rep="—"), hide_index=True, use_container_width=True)
        grid_history = grid_result["evaluation"]
        st.line_chart(pd.DataFrame({_label(name, practical): pd.Series(np.cumprod(1+group["Net return"].to_numpy()), index=group["Date"]) for name, group in grid_history.groupby("Method", sort=False)}))
        st.caption("Choose grids and dates before examining final results. Repeated selection using the final period makes it exploratory. Both strategies start from the same initial allocation at the evaluation boundary.")

st.caption("Run automatically calibrates c and ε before replaying the strategies.")
if st.button("Run feasible-tuning backtest", type="primary"):
    try:
        cs = tuple(float(v.strip()) for v in cs_text.split(",") if v.strip())
        eps = tuple(float(v.strip()) for v in eps_text.split(",") if v.strip())
        with st.spinner("Selecting c and ε from past calibration data..."):
            grid_result = _trace_grid(returns, settings, eps, cs, window, grid_start, grid_final, ppy, rf, cost, cap, trading, grid_end)
        selected_settings = replace(settings, c=grid_result["selected_c"], epsilon=grid_result["selected_epsilon"])
        with st.spinner("Replaying the strategies using the selected c and ε..."):
            result = _history(returns, selected_settings, window, oos_start, rf, cost, cap, trading)
        st.session_state["feasible_trace_grid_result"] = (grid_signature, grid_result)
        st.session_state["feasible_result"] = (grid_signature, result, selected_settings)
        st.rerun()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))

saved = st.session_state.get("feasible_result")
if saved is None or saved[0] != grid_signature:
    st.stop()
result, settings = saved[1], saved[2]
st.success(f"Automatically selected Trace settings: c = {settings.c:g}, ε = {settings.epsilon:g}.")
st.subheader("Historical results")
st.caption(f"{result['Date'].min().date()} to {result['Date'].max().date()}; all methods use the same evaluated periods. Returns cover the complete test.")
summary, wealth = [], {}
for method, group in result.groupby("Method", sort=False):
    net = group["Net return"].to_numpy()
    metrics = performance_metrics(net, gamma=gamma, periods_per_year=ppy)
    excess = net-rf
    summary.append({"Method": _label(method, practical), "Total return": metrics["Total return"], "CAGR": metrics["CAGR"],
                    "Volatility": metrics["Annualized vol"], "Max drawdown": metrics["Max drawdown"],
                    "Annualized MV excess": ppy*(excess.mean()-gamma/2*np.var(excess, ddof=1)) if len(net)>1 else np.nan,
                    "Average turnover": group["Turnover"].mean()})
    wealth[_label(method, practical)] = pd.Series(np.cumprod(1+net), index=group["Date"])
st.dataframe(pd.DataFrame(summary).style.format({k: "{:.2%}" for k in summary[0] if k != "Method"}), hide_index=True, use_container_width=True)
st.line_chart(pd.DataFrame(wealth), use_container_width=True)
st.caption("Wealth index starts at 1. Historical MV statistics do not estimate the theorem's population K2 directly.")
try:
    latest, tuning, raw_targets = _latest(returns, settings, result, window, rf, cap, trading)
    st.subheader("Raw optimal targets — before turnover control")
    st.caption("Classical MV respects the selected allocation constraints in both tables. The other raw targets retain their weight constraints but have no turnover penalty or partial rebalance. Classical MV matches Portfolio's Classical MV when the shared settings are enabled.")
    raw_table = _weight_table(raw_targets, returns.columns, False)
    raw_table["Method"] = raw_table["Method"].replace({"Classical (main sample)": "Classical MV"})
    st.dataframe(raw_table.style.format({k: "{:.4%}" for k in [*returns.columns, "Cash"]}), hide_index=True, use_container_width=True)
    st.subheader("Final allocations — Classical MV stays unadjusted")
    st.caption(f"Common estimation block: {pd.Timestamp(returns.index[-window]).date()} to {pd.Timestamp(returns.index[-1]).date()}, n={window}, γ={gamma:g}. " + (f"Turnover penalty {penalty:g} bps, rebalance step {alpha:g}%, cap {cap:.0%}." if practical else "No turnover penalty or partial-rebalance adjustment to the targets."))
    st.dataframe(_weight_table(latest, returns.columns, practical).style.format({k: "{:.4%}" for k in [*returns.columns,"Cash"]}), hide_index=True, use_container_width=True)
    if use_shared and "weights" in shared and "classical_weights" in shared:
        st.subheader("Check against the saved Portfolio result")
        audit = pd.DataFrame([
            {"Comparison": "Raw Classical MV", "Maximum absolute difference (percentage points)": 100*np.max(np.abs(raw_targets["Classical (main sample)"]-np.asarray(shared["classical_weights"])))},
            {"Comparison": "Final old Portfolio", "Maximum absolute difference (percentage points)": 100*np.max(np.abs(latest[OLD_METHOD]-np.asarray(shared["weights"])))},
        ])
        st.dataframe(audit, hide_index=True, use_container_width=True)
        if audit.iloc[:, 1].max() <= 1e-5:
            st.success("Raw Classical MV and final old Portfolio both match the saved Portfolio recommendation.")
        else:
            st.warning("The saved Portfolio comparison does not match. Rerun Portfolio and this comparison with the complete saved settings before interpreting the differences.")
    st.subheader("Latest tuning values")
    tuning_display = tuning.drop(columns=["a", "Effective b = n a"], errors="ignore").copy()
    trace_rows = tuning_display["Method"] == "Trace tuning"
    tuning_display.insert(1, "Trace c", np.where(trace_rows, settings.c, np.nan))
    tuning_display.insert(2, "Trace ε", np.where(trace_rows, settings.epsilon, np.nan))
    tuning_display = tuning_display.rename(columns={"b": "b (noise level)"})
    st.caption(f"Trace tuning uses c = {settings.c:.6g}, ε = {settings.epsilon:.6g}, and b = c / (ε + tr(Σ̂)).")
    # Explicit text survives Streamlit serialization; Styler na_rep alone does not.
    for column in tuning_display.columns:
        if column in ("Trace c", "Trace ε", "b (noise level)") or tuning_display[column].isna().any():
            tuning_display[column] = tuning_display[column].map(
                lambda value: "—" if pd.isna(value) else
                (f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)))
    st.dataframe(tuning_display, hide_index=True, use_container_width=True)
    st.caption("— means the parameter is not used by that method. c and ε apply only to Trace tuning.")
    st.caption("An active a cap changes the finite-sample rule. For sufficiently large n the cap becomes inactive under the bounded rules.")
except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
    st.error(str(exc))
with st.expander("Rolling detail"):
    st.dataframe(result, hide_index=True, use_container_width=True)
st.download_button("Download feasible-tuning history", result.to_csv(index=False),
                   "feasible_tuning_history.csv", "text/csv")
