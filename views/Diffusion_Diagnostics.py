from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_M, DEFAULT_MAX_LONG_WEIGHT
from core.data import download_yahoo_returns, load_returns_csv
from core.metrics import certainty_equivalent, portfolio_mean, portfolio_volatility
from core.moments import covariance_condition_number, sample_moments
from core.portfolio_rules import compute_weights
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    HORIZON_PRESETS,
    aggregate_nonoverlapping,
    get_horizon_preset,
    replay_latest_recommendation,
)
from core.turnover_upgrade import portfolio_turnover


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
def _cached_diagnostics(
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


st.title("Diffusion Diagnostics")
st.caption(
    "Research/debugging view for the upgraded exact-moment, nested Best-T, turnover-controlled diffusion portfolio."
)
st.info(
    "Use this page to understand why the current diffusion portfolio chose its T and weights. "
    "It uses the same engine as the upgraded Portfolio page rather than the old Monte-Carlo diagnostic."
)

# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
st.subheader("1. Data and Holding Period")
source = st.radio(
    "Data source",
    ["Yahoo Finance", "Upload CSV"],
    horizontal=True,
    key="diag_up_source",
)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    tickers_text = st.text_input(
        "Tickers",
        st.session_state.get("diag_up_tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN"),
        key="diag_up_ticker_text",
    )
with c2:
    start_date = st.text_input(
        "Start date",
        st.session_state.get("diag_up_start", "2000-01-01"),
        key="diag_up_start_text",
    )
with c3:
    holding_period = st.selectbox(
        "Holding / rebalance period",
        list(HORIZON_PRESETS.keys()),
        index=0,
        key="diag_up_horizon",
    )

returns = None
if source == "Yahoo Finance":
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    if st.button("Download / refresh diagnostic data", type="primary", use_container_width=True):
        if not tickers:
            st.error("Enter at least one ticker.")
            st.stop()
        with st.spinner(f"Downloading {holding_period} returns..."):
            returns = _download_holding_returns(tuple(tickers), start_date, holding_period)
        st.session_state["diag_up_returns"] = returns
        st.session_state["diag_up_signature"] = (tuple(tickers), start_date, holding_period)
        st.session_state["diag_up_tickers"] = tickers_text
        st.session_state["diag_up_start"] = start_date
    elif st.session_state.get("diag_up_signature") == (tuple(tickers), start_date, holding_period):
        returns = st.session_state.get("diag_up_returns")
else:
    uploaded = st.file_uploader(
        "Upload holding-period return CSV",
        type=["csv"],
        key="diag_up_csv",
    )
    if uploaded is not None:
        returns = load_returns_csv(uploaded)
        st.caption(
            "Uploaded returns are treated as already sampled at the selected holding period; no additional aggregation is applied."
        )

if returns is None:
    st.info("Download data or upload a return CSV to continue.")
    st.stop()

returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
cfg = get_horizon_preset(holding_period)
ppy = int(cfg["periods_per_year"])
required_min = int(cfg["min_train_size"]) + int(cfg["inner_folds"]) * int(cfg["validation_size"])
if len(returns) <= required_min:
    st.error(
        f"Not enough {holding_period} observations for nested validation. Need more than {required_min}; found {len(returns)}."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Observations", len(returns))
m2.metric("Assets", returns.shape[1])
m3.metric("Periods / year", ppy)
m4.metric("Latest return date", str(pd.Timestamp(returns.index[-1]).date()))

# -----------------------------------------------------------------------------
# 2. Diagnostic settings
# -----------------------------------------------------------------------------
st.subheader("2. Diagnostic Settings")
lookback_max = len(returns) - 1
lookback_default = min(int(cfg["lookback"]), lookback_max)
lookback = st.number_input(
    "Estimation lookback observations",
    min_value=required_min,
    max_value=lookback_max,
    value=lookback_default,
    step=1,
    help=f"Research preset: {cfg['lookback']} observations (~{int(cfg['lookback']) / ppy:.1f} years).",
)
cfg["lookback"] = int(lookback)

s1, s2, s3, s4 = st.columns(4)
with s1:
    gamma = st.number_input(
        "Risk aversion γ",
        min_value=0.1,
        value=float(DEFAULT_GAMMA),
        step=0.1,
        key="diag_up_gamma",
    )
with s2:
    max_long_weight = st.number_input(
        "Maximum weight per asset",
        min_value=max(1.0 / returns.shape[1], 0.05),
        max_value=1.0,
        value=max(float(DEFAULT_MAX_LONG_WEIGHT), 1.0 / returns.shape[1]),
        step=0.05,
        key="diag_up_cap",
    )
with s3:
    m = st.number_input(
        "Synthetic-equivalent M",
        min_value=0,
        value=int(DEFAULT_M),
        step=100,
        key="diag_up_m",
        help="M is a deterministic mixture weight; no random synthetic sample is drawn.",
    )
with s4:
    beta = st.number_input(
        "Constant β",
        min_value=0.01,
        value=float(DEFAULT_BETA),
        step=0.1,
        key="diag_up_beta",
    )

s5, s6, s7, s8 = st.columns(4)
with s5:
    n_steps = st.number_input(
        "Reverse SDE steps",
        min_value=10,
        value=100,
        step=10,
        key="diag_up_steps",
    )
with s6:
    turnover_mode = st.selectbox(
        "Turnover-control settings",
        ["Validated preset", "Custom"],
        index=0,
        key="diag_up_tc_mode",
    )
with s7:
    if turnover_mode == "Validated preset":
        penalty_bps = 25.0
        st.number_input("TC optimizer penalty (bps)", value=penalty_bps, disabled=True, key="diag_up_penalty_disabled")
    else:
        penalty_bps = st.number_input(
            "TC optimizer penalty (bps)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
            key="diag_up_penalty",
        )
with s8:
    if turnover_mode == "Validated preset":
        rebalance_pct = 50.0 if holding_period == "1 week" else 100.0
        st.number_input("Rebalance step (%)", value=rebalance_pct, disabled=True, key="diag_up_alpha_disabled")
    else:
        rebalance_pct = st.number_input(
            "Rebalance step (%)",
            min_value=10.0,
            max_value=100.0,
            value=100.0,
            step=10.0,
            key="diag_up_alpha",
        )

turnover_penalty = float(penalty_bps) / 10000.0
rebalance_alpha = float(rebalance_pct) / 100.0
replay_start = st.text_input(
    "Historical strategy-state replay start",
    "2019-01-01",
    key="diag_up_replay_start",
    help="Needed because the TC target depends on the previous drifted portfolio. Using the same replay start reproduces the Portfolio state path.",
)

with st.expander("Diagnostic formulas"):
    st.write(f"Candidate T grid: {CANDIDATE_T}")
    st.latex(r"\hat T=\arg\max_T \frac{1}{K}\sum_{k=1}^{K} CER_k(T)")
    st.latex(r"\max_w\; w^\top\hat\mu_D-\frac{\gamma}{2}w^\top\hat\Sigma_Dw-\lambda\frac12\|w-w_{prev}\|_1")
    st.caption(
        "Exact moments mean exact first/second moments of the implemented finite-step Euler reverse process, not Monte-Carlo sampling."
    )

# -----------------------------------------------------------------------------
# 3. Run
# -----------------------------------------------------------------------------
if st.button("▶ Run diffusion diagnostics", type="primary", use_container_width=True, key="diag_up_run"):
    with st.spinner("Replaying strategy state and evaluating nested T candidates..."):
        rec = _cached_diagnostics(
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
    w_hist = compute_weights(
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
    w_diff_no_tc = compute_weights(
        "Mean-Variance",
        np.asarray(rec["mu"], dtype=float),
        np.asarray(rec["sigma"], dtype=float),
        gamma=float(gamma),
        returns=x,
        constraint_mode="Long-only",
        max_long_weight=float(max_long_weight),
    )

    st.session_state["diag_up_result"] = {
        "rec": rec,
        "mu_h": np.asarray(mu_h, dtype=float),
        "sigma_h": np.asarray(sigma_h, dtype=float),
        "w_hist": np.asarray(w_hist, dtype=float),
        "w_lw": np.asarray(w_lw, dtype=float),
        "w_diff_no_tc": np.asarray(w_diff_no_tc, dtype=float),
        "assets": list(returns.columns),
        "gamma": float(gamma),
        "max_long_weight": float(max_long_weight),
        "holding_period": holding_period,
        "turnover_penalty": float(turnover_penalty),
        "rebalance_alpha": float(rebalance_alpha),
    }

if "diag_up_result" not in st.session_state:
    st.info("Choose the settings and run diagnostics.")
    st.stop()

res = st.session_state["diag_up_result"]
rec = res["rec"]
assets = res["assets"]
mu_h = res["mu_h"]
sigma_h = res["sigma_h"]
mu_d = np.asarray(rec["mu"], dtype=float)
sigma_d = np.asarray(rec["sigma"], dtype=float)
mu_fake = np.asarray(rec["mu_fake"], dtype=float)
sigma_fake = np.asarray(rec["sigma_fake"], dtype=float)
prev = np.asarray(rec["previous_drifted"], dtype=float)
raw = np.asarray(rec["raw_target"], dtype=float)
final = np.asarray(rec["weights"], dtype=float)
w_diff_no_tc = np.asarray(res["w_diff_no_tc"], dtype=float)

# -----------------------------------------------------------------------------
# 4. T-selection diagnostics
# -----------------------------------------------------------------------------
st.subheader("3. Why This T Was Selected")
t_results = pd.DataFrame(rec["t_results"]).copy()
fold_count = int(t_results["fold_count"].max()) if not t_results.empty else 0
for k in range(fold_count):
    t_results[f"Fold {k + 1} CER"] = t_results["fold_CERs"].apply(
        lambda vals, kk=k: float(vals[kk]) if len(vals) > kk else np.nan
    )

best_row = t_results.loc[t_results["mean_validation_CER"].idxmax()]
selected_row = t_results[t_results["selected_by_rule"]].iloc[0]

t1, t2, t3, t4 = st.columns(4)
t1.metric("Selected nested T", f"{float(rec['T']):.3f}")
t2.metric("Best validation CER", f"{float(best_row['mean_validation_CER']):.4%}")
t3.metric("Selected-T CER", f"{float(selected_row['mean_validation_CER']):.4%}")
t4.metric("Data through", str(pd.Timestamp(rec["data_through"]).date()))

show_t_cols = [
    "T",
    "mean_validation_CER",
    "std_validation_CER",
    "standard_error_CER",
    "selected_by_rule",
] + [f"Fold {k + 1} CER" for k in range(fold_count)]
st.dataframe(
    t_results[show_t_cols].style.format(
        {
            "T": "{:.3f}",
            "mean_validation_CER": "{:.4%}",
            "std_validation_CER": "{:.4%}",
            "standard_error_CER": "{:.4%}",
            **{f"Fold {k + 1} CER": "{:.4%}" for k in range(fold_count)},
        }
    ),
    use_container_width=True,
    hide_index=True,
)

fig_t = px.line(
    t_results,
    x="T",
    y="mean_validation_CER",
    markers=True,
    title="Nested validation CER by diffusion horizon T",
)
fig_t.add_vline(x=float(rec["T"]), line_dash="dash", annotation_text="Selected T")
st.plotly_chart(fig_t, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Moment diagnostics
# -----------------------------------------------------------------------------
st.subheader("4. What Diffusion Changed in the Moments")
summary_diag = pd.DataFrame(
    {
        "Metric": [
            "Mean-vector norm",
            "Covariance trace",
            "Covariance condition number",
            "Average asset volatility",
        ],
        "Historical": [
            float(np.linalg.norm(mu_h)),
            float(np.trace(sigma_h)),
            float(covariance_condition_number(sigma_h)),
            float(np.mean(np.sqrt(np.clip(np.diag(sigma_h), 0.0, None)))),
        ],
        "Exact diffusion augmented": [
            float(np.linalg.norm(mu_d)),
            float(np.trace(sigma_d)),
            float(covariance_condition_number(sigma_d)),
            float(np.mean(np.sqrt(np.clip(np.diag(sigma_d), 0.0, None)))),
        ],
        "Reverse synthetic population": [
            float(np.linalg.norm(mu_fake)),
            float(np.trace(sigma_fake)),
            float(covariance_condition_number(sigma_fake)),
            float(np.mean(np.sqrt(np.clip(np.diag(sigma_fake), 0.0, None)))),
        ],
    }
)
st.dataframe(summary_diag, use_container_width=True, hide_index=True)

asset_moments = pd.DataFrame(
    {
        "Asset": assets,
        "Historical mean": mu_h,
        "Diffusion mean": mu_d,
        "Synthetic-population mean": mu_fake,
        "Historical volatility": np.sqrt(np.clip(np.diag(sigma_h), 0.0, None)),
        "Diffusion volatility": np.sqrt(np.clip(np.diag(sigma_d), 0.0, None)),
    }
)
st.dataframe(
    asset_moments.style.format(
        {
            "Historical mean": "{:.4%}",
            "Diffusion mean": "{:.4%}",
            "Synthetic-population mean": "{:.4%}",
            "Historical volatility": "{:.4%}",
            "Diffusion volatility": "{:.4%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# 6. Weight / turnover diagnostics
# -----------------------------------------------------------------------------
st.subheader("5. How Turnover Control Changed the Portfolio")
weight_df = pd.DataFrame(
    {
        "Asset": assets,
        "Previous drifted": prev,
        "Classical MV": res["w_hist"],
        "Classical MV + LW": res["w_lw"],
        "Exact Diffusion no TC": w_diff_no_tc,
        "Raw TC target": raw,
        "Final recommended": final,
    }
)
weight_df["Trade from previous"] = final - prev
st.dataframe(
    weight_df.style.format({c: "{:.2%}" for c in weight_df.columns if c != "Asset"}),
    use_container_width=True,
    hide_index=True,
)

no_tc_turnover = portfolio_turnover(w_diff_no_tc, prev)
raw_turnover = portfolio_turnover(raw, prev)
final_turnover = portfolio_turnover(final, prev)
turnover_reduction = 1.0 - final_turnover / no_tc_turnover if no_tc_turnover > 0 else np.nan

w1, w2, w3, w4 = st.columns(4)
w1.metric("No-TC desired turnover", f"{no_tc_turnover:.2%}")
w2.metric("Raw TC turnover", f"{raw_turnover:.2%}")
w3.metric("Final turnover", f"{final_turnover:.2%}")
w4.metric(
    "Turnover reduction vs no-TC",
    f"{turnover_reduction:.1%}" if np.isfinite(turnover_reduction) else "N/A",
)

# Model-implied utility comparison using the current exact diffusion moments.
utility_rows = []
for name, w in [
    ("Previous drifted", prev),
    ("Exact Diffusion no TC", w_diff_no_tc),
    ("Raw TC target", raw),
    ("Final recommended", final),
]:
    utility_rows.append(
        {
            "Portfolio": name,
            "Expected return / period": portfolio_mean(w, mu_d),
            "Volatility / period": portfolio_volatility(w, sigma_d),
            "CER / period": certainty_equivalent(w, mu_d, sigma_d, gamma=float(res["gamma"])),
            "Turnover from previous": portfolio_turnover(w, prev),
        }
    )
utility_df = pd.DataFrame(utility_rows)
st.markdown("**Current exact-diffusion model utility**")
st.dataframe(
    utility_df.style.format(
        {
            "Expected return / period": "{:.4%}",
            "Volatility / period": "{:.4%}",
            "CER / period": "{:.4%}",
            "Turnover from previous": "{:.2%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if np.max(final) <= float(res["max_long_weight"]) + 1e-9 and abs(float(final.sum()) - 1.0) < 1e-8:
    st.success(
        f"Final recommendation is feasible: fully invested, long-only, and each asset ≤ {res['max_long_weight']:.0%}."
    )
else:
    st.error("Final recommendation violates the requested portfolio constraints.")

st.download_button(
    "⬇ Download current diagnostics (.csv)",
    data=weight_df.to_csv(index=False).encode("utf-8"),
    file_name=f"diffusion_diagnostics_{res['holding_period'].replace(' ', '_')}.csv",
    mime="text/csv",
    use_container_width=True,
)
