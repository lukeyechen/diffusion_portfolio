from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_M, DEFAULT_MAX_LONG_WEIGHT
from core.data import download_yahoo_returns, load_returns_csv
from core.metrics import (
    certainty_equivalent,
    portfolio_mean,
    portfolio_volatility,
    sharpe_ratio,
)
from core.moments import covariance_condition_number, sample_moments
from core.portfolio_rules import compute_weights
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    HORIZON_PRESETS,
    aggregate_nonoverlapping,
    get_horizon_preset,
    replay_latest_recommendation,
)


@st.cache_data(show_spinner=False)
def _download_holding_returns(tickers, start, holding_period):
    cfg = get_horizon_preset(holding_period)
    interval = "1wk" if cfg["source"] == "weekly" else "1mo"
    base = download_yahoo_returns(
        tickers,
        start=start,
        end=None,
        interval=interval,
    )
    return aggregate_nonoverlapping(base, int(cfg["block_size"]))


@st.cache_data(show_spinner=False)
def _cached_latest_recommendation(
    returns,
    cfg,
    gamma,
    m,
    beta,
    n_steps,
    turnover_penalty,
    rebalance_alpha,
    max_long_weight,
    replay_start,
):
    return replay_latest_recommendation(
        returns,
        cfg,
        gamma=float(gamma),
        m=int(m),
        beta=float(beta),
        n_steps=int(n_steps),
        candidate_t=CANDIDATE_T,
        turnover_penalty=float(turnover_penalty),
        rebalance_alpha=float(rebalance_alpha),
        max_long_weight=float(max_long_weight),
        replay_start=replay_start,
    )


st.title("Portfolio Builder")
st.caption(
    "Upgraded current-window portfolio: deterministic Gaussian diffusion moments, "
    "nested rolling Best-T selection, and turnover-aware rebalancing."
)

st.info(
    "This is the upgraded engine used in the recent 1-week / 2-week backtests. "
    "The old Monte-Carlo Gaussian and Learned Neural interfaces remain available under "
    "**Portfolio (Legacy)** in the sidebar."
)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
st.subheader("1. Data and Holding Period")
source = st.radio("Data source", ["Yahoo Finance", "Upload CSV"], horizontal=True)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    tickers_text = st.text_input(
        "Tickers",
        st.session_state.get("up_tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN"),
    )
with c2:
    start_date = st.text_input(
        "Start date",
        st.session_state.get("up_start", "2000-01-01"),
    )
with c3:
    holding_period = st.selectbox(
        "Holding / rebalance period",
        list(HORIZON_PRESETS.keys()),
        index=0,
        help="1 week and 2 weeks have validated turnover-control presets. Longer horizons are exploratory.",
    )

returns = None
if source == "Yahoo Finance":
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    if st.button("Download / refresh data", type="primary", use_container_width=True):
        if not tickers:
            st.error("Enter at least one ticker.")
            st.stop()
        with st.spinner(f"Downloading {holding_period} returns..."):
            returns = _download_holding_returns(tuple(tickers), start_date, holding_period)
        st.session_state["up_returns"] = returns
        st.session_state["up_signature"] = (tuple(tickers), start_date, holding_period)
        st.session_state["up_tickers"] = tickers_text
        st.session_state["up_start"] = start_date
    else:
        sig = st.session_state.get("up_signature")
        if sig == (tuple(tickers), start_date, holding_period):
            returns = st.session_state.get("up_returns")
else:
    uploaded = st.file_uploader("Upload return CSV", type=["csv"])
    if uploaded is not None:
        returns = load_returns_csv(uploaded)
        st.caption(
            "Uploaded CSV is treated as already sampled at the selected holding period; "
            "the app does not re-aggregate uploaded returns."
        )

if returns is None:
    st.info("Download data or upload a return CSV to continue.")
    st.stop()

returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
cfg = get_horizon_preset(holding_period)
ppy = int(cfg["periods_per_year"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Observations", len(returns))
m2.metric("Assets", returns.shape[1])
m3.metric("Periods / year", ppy)
m4.metric("Latest return date", str(pd.Timestamp(returns.index[-1]).date()))

with st.expander("Show latest returns"):
    st.dataframe(returns.tail(10), use_container_width=True)
    st.download_button(
        "Download holding-period returns (.csv)",
        returns.to_csv(index=True, index_label="Date").encode("utf-8"),
        file_name=f"returns_{holding_period.replace(' ', '_')}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
st.subheader("2. Upgraded Diffusion / Portfolio Settings")

required_min = int(cfg["min_train_size"]) + int(cfg["inner_folds"]) * int(cfg["validation_size"])
default_lookback = int(cfg["lookback"])
if len(returns) <= required_min:
    st.error(
        f"Not enough {holding_period} observations for nested validation. "
        f"Need more than {required_min}; found {len(returns)}."
    )
    st.stop()

lookback_max = len(returns) - 1
lookback_value = min(default_lookback, lookback_max)
lookback = st.number_input(
    "Estimation lookback observations",
    min_value=required_min,
    max_value=lookback_max,
    value=lookback_value,
    step=1,
    help=(
        f"Research preset for {holding_period}: {default_lookback} observations "
        f"(~{default_lookback / ppy:.1f} years)."
    ),
)
cfg["lookback"] = int(lookback)

r1, r2, r3, r4 = st.columns(4)
with r1:
    gamma = st.number_input("Risk aversion γ", min_value=0.1, value=float(DEFAULT_GAMMA), step=0.1)
with r2:
    max_long_weight = st.number_input(
        "Maximum weight per asset",
        min_value=max(1.0 / returns.shape[1], 0.05),
        max_value=1.0,
        value=max(float(DEFAULT_MAX_LONG_WEIGHT), 1.0 / returns.shape[1]),
        step=0.05,
    )
with r3:
    m = st.number_input(
        "Synthetic-equivalent M",
        min_value=0,
        value=int(DEFAULT_M),
        step=100,
        help="M is now a deterministic mixture weight; no random synthetic sample is drawn.",
    )
with r4:
    beta = st.number_input("Constant β", min_value=0.01, value=float(DEFAULT_BETA), step=0.1)

r5, r6, r7, r8 = st.columns(4)
with r5:
    n_steps = st.number_input("Reverse SDE steps", min_value=10, value=100, step=10)
with r6:
    preset_mode = st.selectbox(
        "Turnover preset",
        ["Validated preset", "Custom"],
        index=0,
    )
with r7:
    if preset_mode == "Validated preset":
        penalty_bps = 25.0
        st.number_input("Turnover penalty (bps)", value=penalty_bps, disabled=True)
    else:
        penalty_bps = st.number_input(
            "Turnover penalty (bps)", min_value=0.0, max_value=100.0, value=25.0, step=5.0
        )
with r8:
    if preset_mode == "Validated preset":
        preset_alpha = 0.50 if holding_period == "1 week" else 1.00
        rebalance_pct = 100.0 * preset_alpha
        st.number_input("Rebalance step (%)", value=rebalance_pct, disabled=True)
    else:
        rebalance_pct = st.number_input(
            "Rebalance step (%)", min_value=10.0, max_value=100.0, value=100.0, step=10.0
        )

turnover_penalty = float(penalty_bps) / 10000.0
rebalance_alpha = float(rebalance_pct) / 100.0
replay_start = st.text_input(
    "Historical strategy-state replay start",
    "2019-01-01",
    help=(
        "The turnover-aware target depends on the previous drifted portfolio. Replaying from "
        "2019 reproduces the state path used in the validated research backtest."
    ),
)

if preset_mode == "Validated preset" and holding_period not in ("1 week", "2 weeks"):
    st.warning(
        "The 25-bps turnover penalty is available here, but its best setting has only been "
        "validated in our 1-week and 2-week experiments. Treat longer-horizon use as exploratory."
    )

with st.expander("Nested Best-T details"):
    st.write(
        f"Candidate T grid: {CANDIDATE_T}. Each outer window uses "
        f"{cfg['inner_folds']} rolling validation folds of {cfg['validation_size']} observations."
    )
    st.latex(r"\hat T=\arg\max_T \frac{1}{K}\sum_{k=1}^K CER_k(T)")

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
run = st.button(
    "▶ Run upgraded portfolio",
    type="primary",
    use_container_width=True,
)

if not run and "upgraded_portfolio_result" not in st.session_state:
    st.info("Set the inputs above and click **Run upgraded portfolio**.")
    st.stop()

if run:
    with st.spinner(
        "Replaying historical turnover state, selecting nested Best-T, and computing exact diffusion moments..."
    ):
        rec = _cached_latest_recommendation(
            returns,
            cfg,
            gamma,
            m,
            beta,
            n_steps,
            turnover_penalty,
            rebalance_alpha,
            max_long_weight,
            replay_start,
        )

    x = returns.iloc[-int(lookback):].to_numpy(dtype=float)
    mu_h, sigma_h = sample_moments(x, mle=True)
    w_classical = compute_weights(
        "Mean-Variance",
        mu_h,
        sigma_h,
        gamma=float(gamma),
        returns=x,
        constraint_mode="Long-only",
        max_long_weight=float(max_long_weight),
    )
    w_lw = compute_weights(
        "Ledoit-Wolf Mean-Variance",
        mu_h,
        sigma_h,
        gamma=float(gamma),
        returns=x,
        constraint_mode="Long-only",
        max_long_weight=float(max_long_weight),
    )

    st.session_state["upgraded_portfolio_result"] = {
        "rec": rec,
        "holding_period": holding_period,
        "cfg": dict(cfg),
        "returns": returns.copy(),
        "gamma": float(gamma),
        "m": int(m),
        "beta": float(beta),
        "n_steps": int(n_steps),
        "turnover_penalty": float(turnover_penalty),
        "rebalance_alpha": float(rebalance_alpha),
        "max_long_weight": float(max_long_weight),
        "mu_hist": np.asarray(mu_h, dtype=float),
        "sigma_hist": np.asarray(sigma_h, dtype=float),
        "w_classical": np.asarray(w_classical, dtype=float),
        "w_lw": np.asarray(w_lw, dtype=float),
    }

res = st.session_state["upgraded_portfolio_result"]
rec = res["rec"]
assets = list(res["returns"].columns)
w = np.asarray(rec["weights"], dtype=float)
raw = np.asarray(rec["raw_target"], dtype=float)
prev = np.asarray(rec["previous_drifted"], dtype=float)
mu_used = np.asarray(rec["mu"], dtype=float)
sigma_used = np.asarray(rec["sigma"], dtype=float)

st.subheader("3. Recommended Weights")
weights_df = pd.DataFrame(
    {
        "Asset": assets,
        "Previous drifted": prev,
        "Raw TC target": raw,
        "Recommended weight": w,
    }
).sort_values("Recommended weight", ascending=False)
st.dataframe(
    weights_df.style.format(
        {
            "Previous drifted": "{:.2%}",
            "Raw TC target": "{:.2%}",
            "Recommended weight": "{:.2%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download recommended weights (.csv)",
    weights_df.to_csv(index=False).encode("utf-8"),
    file_name=f"recommended_weights_{res['holding_period'].replace(' ', '_')}.csv",
    mime="text/csv",
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Selected nested T", f"{float(rec['T']):.4f}")
k2.metric(
    "Previous selected T",
    f"{float(rec['previous_T']):.4f}" if np.isfinite(rec["previous_T"]) else "N/A",
)
k3.metric("Trade turnover", f"{float(rec['turnover']):.2%}")
k4.metric("Data through", str(pd.Timestamp(rec["data_through"]).date()))

if np.max(w) <= float(res["max_long_weight"]) + 1e-9:
    st.success(
        f"Final weights satisfy the hard long-only cap: max weight ≤ {res['max_long_weight']:.0%}."
    )
else:
    st.error("Final weights violate the requested cap. Do not trade these weights.")

st.subheader("4. Current-Window Diagnostics")
expected_return = portfolio_mean(w, mu_used)
volatility = portfolio_volatility(w, sigma_used)
sharpe = sharpe_ratio(w, mu_used, sigma_used)
cer = certainty_equivalent(w, mu_used, sigma_used, float(res["gamma"]))
condition = covariance_condition_number(sigma_used)

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Expected return / period", f"{expected_return:.3%}")
p2.metric("Volatility / period", f"{volatility:.3%}")
p3.metric("Sharpe / period", f"{sharpe:.3f}")
p4.metric("CER / period", f"{cer:.3%}")
p5.metric("κ(Σ)", f"{condition:,.1f}")

compare_df = pd.DataFrame(
    {
        "Asset": assets,
        "Upgraded Diffusion": w,
        "Classical MV": np.asarray(res["w_classical"], dtype=float),
        "Classical MV + LW": np.asarray(res["w_lw"], dtype=float),
        "Equal Weight": np.ones(len(assets)) / len(assets),
    }
)
st.markdown("**Current-window weight comparison**")
st.dataframe(
    compare_df.style.format(
        {c: "{:.2%}" for c in compare_df.columns if c != "Asset"}
    ),
    use_container_width=True,
    hide_index=True,
)

# Shared contract for the upgraded Method Comparison page.
portfolio_revision = int(st.session_state.get("portfolio_revision", 0)) + 1
st.session_state["portfolio_revision"] = portfolio_revision
st.session_state["shared_current_window"] = {
    "version": 3,
    "portfolio_revision": portfolio_revision,
    "upgraded": True,
    "estimator": "Diffusion",
    "score_model": "Exact Gaussian (deterministic)",
    "tuning_mode": "Nested Best-T",
    "holding_period": res["holding_period"],
    "interval_label": res["holding_period"],
    "asset_names": assets,
    "full_returns": res["returns"].copy(),
    "window_returns": res["returns"].iloc[-int(res["cfg"]["lookback"]):].copy(),
    "lookback": int(res["cfg"]["lookback"]),
    "periods_per_year": int(res["cfg"]["periods_per_year"]),
    "gamma": float(res["gamma"]),
    "portfolio_rule": "Mean-Variance",
    "constraint_mode": "Long-only",
    "max_long_weight": float(res["max_long_weight"]),
    "max_short_weight": 0.0,
    "max_gross_exposure": 1.0,
    "T": float(rec["T"]),
    "m": int(res["m"]),
    "beta": float(res["beta"]),
    "n_steps": int(res["n_steps"]),
    "turnover_penalty": float(res["turnover_penalty"]),
    "rebalance_alpha": float(res["rebalance_alpha"]),
    "previous_drifted": prev.copy(),
    "raw_target": raw.copy(),
    "mu_hist": np.asarray(res["mu_hist"], dtype=float).copy(),
    "sigma_hist": np.asarray(res["sigma_hist"], dtype=float).copy(),
    "mu_used": mu_used.copy(),
    "sigma_used": sigma_used.copy(),
    "weights": w.copy(),
    "weights_by_asset": {a: float(v) for a, v in zip(assets, w)},
}

st.caption(
    "The upgraded Method Comparison page reads this saved current-window state, so its "
    "synchronized portfolio row uses the same exact moments, T, turnover penalty, and weights."
)

if st.button("Open Method Comparison →", use_container_width=True):
    st.session_state["app_section"] = "Method Comparison"
    st.rerun()
