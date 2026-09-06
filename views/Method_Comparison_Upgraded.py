from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_M, DEFAULT_MAX_LONG_WEIGHT
from core.data import download_yahoo_returns
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    HORIZON_PRESETS,
    aggregate_nonoverlapping,
    get_horizon_preset,
    run_oos_comparison,
)


@st.cache_data(show_spinner=False)
def _download_base_returns(tickers, start, source_name):
    interval = "1wk" if source_name == "weekly" else "1mo"
    return download_yahoo_returns(
        tickers,
        start=start,
        end=None,
        interval=interval,
    )


@st.cache_data(show_spinner=False)
def _cached_oos(
    returns,
    cfg,
    gamma,
    m,
    beta,
    n_steps,
    turnover_penalty,
    rebalance_alpha,
    max_long_weight,
    oos_start,
):
    return run_oos_comparison(
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
        oos_start=oos_start,
        costs=(0.0010, 0.0025),
    )


def _export_zip(result):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in result.items():
            if isinstance(df, pd.DataFrame):
                zf.writestr(f"{name}.csv", df.to_csv(index=False))
    return buffer.getvalue()


st.title("Method Comparison")
st.caption(
    "Upgraded short-horizon research page using deterministic exact diffusion moments, "
    "nested Best-T selection, and explicit turnover control."
)
st.info(
    "This page is designed to match the upgraded Portfolio engine. The original Neural / "
    "Gaussian research Sections A-E remain under **Method Comparison (Legacy)**."
)

# -----------------------------------------------------------------------------
# Current Portfolio synchronization
# -----------------------------------------------------------------------------
st.subheader("Current Portfolio Synchronization")
shared = st.session_state.get("shared_current_window")
if isinstance(shared, dict) and shared.get("upgraded"):
    sync_df = pd.DataFrame(
        {
            "Asset": shared.get("asset_names", []),
            "Current upgraded Portfolio weight": np.asarray(shared.get("weights", []), dtype=float),
        }
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Holding period", str(shared.get("holding_period", "N/A")))
    s2.metric("Selected T", f"{float(shared.get('T', np.nan)):.4f}")
    s3.metric("TC penalty", f"{10000 * float(shared.get('turnover_penalty', 0.0)):.0f} bps")
    s4.metric("Rebalance step", f"{100 * float(shared.get('rebalance_alpha', 1.0)):.0f}%")
    st.dataframe(
        sync_df.style.format({"Current upgraded Portfolio weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.success(
        "The current Portfolio state uses the same exact-moment / nested-T / turnover-control definitions as this page."
    )
else:
    st.warning(
        "No upgraded Portfolio result is saved in this session. You can still run the OOS comparison below."
    )

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
st.subheader("1. Comparison Settings")
c1, c2 = st.columns([2, 1])
with c1:
    tickers_text = st.text_input(
        "Tickers",
        st.session_state.get("mc_up_tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN"),
    )
with c2:
    start_date = st.text_input("Start date", st.session_state.get("mc_up_start", "2000-01-01"))

tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
selected_horizons = st.multiselect(
    "Holding periods to compare",
    list(HORIZON_PRESETS.keys()),
    default=["1 week", "2 weeks"],
    help="You can also include 1 month, 2 months, and 3 months. Those turnover settings are exploratory.",
)

r1, r2, r3, r4 = st.columns(4)
with r1:
    gamma = st.number_input("Risk aversion γ", min_value=0.1, value=float(DEFAULT_GAMMA), step=0.1)
with r2:
    max_long_weight = st.number_input(
        "Maximum weight per asset",
        min_value=max(1.0 / max(len(tickers), 1), 0.05),
        max_value=1.0,
        value=max(float(DEFAULT_MAX_LONG_WEIGHT), 1.0 / max(len(tickers), 1)),
        step=0.05,
    )
with r3:
    m = st.number_input(
        "Synthetic-equivalent M",
        min_value=0,
        value=int(DEFAULT_M),
        step=100,
        help="Exact moments are used; M controls the real/synthetic mixture weight rather than Monte-Carlo sample noise.",
    )
with r4:
    beta = st.number_input("Constant β", min_value=0.01, value=float(DEFAULT_BETA), step=0.1)

r5, r6, r7, r8 = st.columns(4)
with r5:
    n_steps = st.number_input("Reverse SDE steps", min_value=10, value=100, step=10)
with r6:
    turnover_bps = st.number_input(
        "Turnover penalty (bps)", min_value=0.0, max_value=100.0, value=25.0, step=5.0
    )
with r7:
    rebalance_mode = st.selectbox(
        "Rebalance rule",
        ["Validated per-horizon preset", "Full rebalance", "Custom"],
        index=0,
    )
with r8:
    if rebalance_mode == "Custom":
        custom_alpha_pct = st.number_input(
            "Custom rebalance step (%)", min_value=10.0, max_value=100.0, value=50.0, step=10.0
        )
    else:
        custom_alpha_pct = 100.0
        st.number_input("Custom rebalance step (%)", value=100.0, disabled=True)

oos_start = st.text_input(
    "True OOS evaluation start",
    "2019-01-01",
    help="Every tested period uses only data available before that realized holding-period return.",
)

with st.expander("Methods and nested-T definition"):
    st.markdown(
        """
        The upgraded table contains six methods:

        1. Equal Weight  
        2. Classical Mean-Variance  
        3. Classical Mean-Variance + Ledoit-Wolf  
        4. Exact Diffusion (Nested Best-T)  
        5. 50% Exact Diffusion + 50% Equal Weight  
        6. Turnover-Controlled Exact Diffusion
        """
    )
    st.write(f"Candidate T grid: {CANDIDATE_T}")
    st.latex(r"\hat T=\arg\max_T \frac{1}{K}\sum_{k=1}^{K} CER_k(T)")
    st.latex(
        r"\max_w\;w^\top\hat\mu_D-\frac{\gamma}{2}w^\top\hat\Sigma_Dw-\lambda\frac12\|w-w_{prev}\|_1"
    )

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if st.button("▶ Run upgraded method comparison", type="primary", use_container_width=True):
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()
    if not selected_horizons:
        st.error("Select at least one holding period.")
        st.stop()

    st.session_state["mc_up_tickers"] = tickers_text
    st.session_state["mc_up_start"] = start_date

    summaries = []
    details = []
    latest_tables = []
    tdiags = []
    progress = st.progress(0, text="Preparing data...")

    needed_sources = {get_horizon_preset(h)["source"] for h in selected_horizons}
    base_data = {}
    for j, src in enumerate(sorted(needed_sources)):
        progress.progress(
            int(10 * (j + 1) / max(len(needed_sources), 1)),
            text=f"Downloading {src} base returns...",
        )
        base_data[src] = _download_base_returns(tuple(tickers), start_date, src)[tickers].dropna()

    for idx, horizon in enumerate(selected_horizons):
        cfg = get_horizon_preset(horizon)
        holding_returns = aggregate_nonoverlapping(
            base_data[str(cfg["source"])], int(cfg["block_size"])
        )

        if rebalance_mode == "Validated per-horizon preset":
            alpha = 0.50 if horizon == "1 week" else 1.00
        elif rebalance_mode == "Full rebalance":
            alpha = 1.00
        else:
            alpha = float(custom_alpha_pct) / 100.0

        progress.progress(
            10 + int(85 * idx / max(len(selected_horizons), 1)),
            text=f"Running {horizon} rolling OOS comparison...",
        )
        summary, detail, latest, tdiag = _cached_oos(
            holding_returns,
            cfg,
            gamma,
            m,
            beta,
            n_steps,
            float(turnover_bps) / 10000.0,
            alpha,
            max_long_weight,
            oos_start,
        )
        summary.insert(0, "Horizon", horizon)
        summary.insert(1, "Rebalance step", alpha)
        detail.insert(0, "Horizon", horizon)
        latest.insert(0, "Horizon", horizon)
        tdiag.insert(0, "Horizon", horizon)
        tdiag["Rebalance step"] = alpha
        summaries.append(summary)
        details.append(detail)
        latest_tables.append(latest)
        tdiags.append(tdiag)

    result = {
        "summary": pd.concat(summaries, ignore_index=True),
        "detail": pd.concat(details, ignore_index=True, sort=False),
        "latest_weights": pd.concat(latest_tables, ignore_index=True, sort=False),
        "t_diagnostics": pd.concat(tdiags, ignore_index=True),
    }
    st.session_state["upgraded_method_comparison"] = result
    progress.progress(100, text="Comparison complete.")
    progress.empty()

if "upgraded_method_comparison" not in st.session_state:
    st.info("Choose the settings above and run the comparison.")
    st.stop()

res = st.session_state["upgraded_method_comparison"]
summary = res["summary"].copy()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
st.subheader("2. OOS Performance — All Methods")
show_cols = [
    "Horizon",
    "Method",
    "CAGR",
    "Sharpe",
    "Realized CER",
    "Max drawdown",
    "Average turnover",
    "Net CAGR 10bps",
    "Net CAGR 25bps",
    "Final $10,000",
]
st.dataframe(
    summary[show_cols].style.format(
        {
            "CAGR": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Realized CER": "{:.2%}",
            "Max drawdown": "{:.2%}",
            "Average turnover": "{:.2%}",
            "Net CAGR 10bps": "{:.2%}",
            "Net CAGR 25bps": "{:.2%}",
            "Final $10,000": "${:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

winners = (
    summary.loc[summary.groupby("Horizon")["CAGR"].idxmax(), ["Horizon", "Method", "CAGR"]]
    .sort_values("Horizon")
    .reset_index(drop=True)
)
st.markdown("**Gross CAGR winner by holding period**")
st.dataframe(
    winners.style.format({"CAGR": "{:.2%}"}),
    use_container_width=True,
    hide_index=True,
)

fig = px.bar(
    summary,
    x="Horizon",
    y="CAGR",
    color="Method",
    barmode="group",
    title="Gross OOS CAGR by holding period and method",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Diffusion / Turnover Diagnostics")
st.dataframe(
    res["t_diagnostics"].style.format(
        {
            "T>0 fraction": "{:.2%}",
            "Median T": "{:.3f}",
            "Mean T": "{:.3f}",
            "Rebalance step": "{:.0%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("4. Latest OOS Portfolio Weights")
latest = res["latest_weights"].copy()
asset_cols = [c for c in latest.columns if c not in ["Horizon", "Method"]]
st.dataframe(
    latest.style.format({c: "{:.2%}" for c in asset_cols}),
    use_container_width=True,
    hide_index=True,
)

focus = latest[latest["Method"] == "Turnover-Controlled Exact Diffusion"].copy()
if not focus.empty:
    st.markdown("**Turnover-Controlled Exact Diffusion — latest OOS weights**")
    st.dataframe(
        focus.style.format({c: "{:.2%}" for c in asset_cols}),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "⬇ Download upgraded comparison results (.zip)",
    data=_export_zip(res),
    file_name="upgraded_method_comparison.zip",
    mime="application/zip",
    use_container_width=True,
)

st.caption(
    "The rolling OOS table evaluates historical periods. For the recommendation for the NEXT "
    "1-week or 2-week holding period, use the Portfolio page; it replays the turnover state "
    "through the latest available return and then computes the next target."
)
