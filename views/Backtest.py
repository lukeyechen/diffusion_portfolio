from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_M, DEFAULT_MAX_LONG_WEIGHT
from core.data import download_yahoo_returns, load_returns_csv
from core.short_horizon_portfolio import (
    CANDIDATE_T,
    HORIZON_PRESETS,
    aggregate_nonoverlapping,
    get_horizon_preset,
    performance_metrics,
    run_oos_comparison,
)


METHODS = [
    "Equal Weight",
    "Classical MV",
    "Classical MV + LW",
    "Exact Diffusion (Best-T)",
    "50% Exact Diff + 50% EW",
    "Turnover-Controlled Exact Diffusion",
]


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
def _cached_backtest(
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


st.title("Strategy Backtest")
st.caption(
    "Investment-performance simulator for the same short-horizon engine used by Portfolio and Method Comparison."
)
st.info(
    "Use this page to answer: if I had followed one strategy through history, what would have happened to my money? "
    "Every OOS period is estimated using only information available before that realized holding-period return."
)

# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
st.subheader("1. Data and Holding Period")
source = st.radio(
    "Data source",
    ["Yahoo Finance", "Upload CSV"],
    horizontal=True,
    key="bt_source_upgraded",
)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    tickers_text = st.text_input(
        "Tickers",
        st.session_state.get("bt_up_tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN"),
        key="bt_up_ticker_text",
    )
with c2:
    start_date = st.text_input(
        "Start date",
        st.session_state.get("bt_up_start", "2000-01-01"),
        key="bt_up_start_text",
    )
with c3:
    holding_period = st.selectbox(
        "Holding / rebalance period",
        list(HORIZON_PRESETS.keys()),
        index=0,
        key="bt_up_horizon",
        help="1 week and 2 weeks have the strongest validated turnover-control results. Longer horizons remain exploratory.",
    )

returns = None
if source == "Yahoo Finance":
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    if st.button("Download / refresh backtest data", type="primary", use_container_width=True):
        if not tickers:
            st.error("Enter at least one ticker.")
            st.stop()
        with st.spinner(f"Downloading {holding_period} returns..."):
            returns = _download_holding_returns(tuple(tickers), start_date, holding_period)
        st.session_state["bt_up_returns"] = returns
        st.session_state["bt_up_signature"] = (tuple(tickers), start_date, holding_period)
        st.session_state["bt_up_tickers"] = tickers_text
        st.session_state["bt_up_start"] = start_date
    elif st.session_state.get("bt_up_signature") == (tuple(tickers), start_date, holding_period):
        returns = st.session_state.get("bt_up_returns")
else:
    uploaded = st.file_uploader(
        "Upload holding-period return CSV",
        type=["csv"],
        key="bt_up_csv",
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
# 2. Strategy settings
# -----------------------------------------------------------------------------
st.subheader("2. Backtest Settings")

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
    strategy = st.selectbox(
        "Strategy to evaluate",
        METHODS,
        index=5,
        key="bt_up_strategy",
    )
with s2:
    gamma = st.number_input(
        "Risk aversion γ",
        min_value=0.1,
        value=float(DEFAULT_GAMMA),
        step=0.1,
        key="bt_up_gamma",
    )
with s3:
    max_long_weight = st.number_input(
        "Maximum weight per asset",
        min_value=max(1.0 / returns.shape[1], 0.05),
        max_value=1.0,
        value=max(float(DEFAULT_MAX_LONG_WEIGHT), 1.0 / returns.shape[1]),
        step=0.05,
        key="bt_up_cap",
    )
with s4:
    initial_capital = st.number_input(
        "Initial capital ($)",
        min_value=100.0,
        value=10000.0,
        step=1000.0,
        key="bt_up_capital",
    )

s5, s6, s7, s8 = st.columns(4)
with s5:
    m = st.number_input(
        "Synthetic-equivalent M",
        min_value=0,
        value=int(DEFAULT_M),
        step=100,
        key="bt_up_m",
        help="Exact moments are used; M controls the real/synthetic mixture weight rather than random sample count.",
    )
with s6:
    beta = st.number_input(
        "Constant β",
        min_value=0.01,
        value=float(DEFAULT_BETA),
        step=0.1,
        key="bt_up_beta",
    )
with s7:
    n_steps = st.number_input(
        "Reverse SDE steps",
        min_value=10,
        value=100,
        step=10,
        key="bt_up_steps",
    )
with s8:
    transaction_cost_bps = st.number_input(
        "Realized trading cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=5.0,
        key="bt_up_cost",
        help="Applied after portfolio selection: net return = gross return - cost × turnover.",
    )

s9, s10, s11 = st.columns(3)
with s9:
    turnover_mode = st.selectbox(
        "Turnover-control settings",
        ["Validated preset", "Custom"],
        index=0,
        key="bt_up_tc_mode",
    )
with s10:
    if turnover_mode == "Validated preset":
        penalty_bps = 25.0
        st.number_input("TC optimizer penalty (bps)", value=penalty_bps, disabled=True, key="bt_up_penalty_disabled")
    else:
        penalty_bps = st.number_input(
            "TC optimizer penalty (bps)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
            key="bt_up_penalty",
        )
with s11:
    if turnover_mode == "Validated preset":
        rebalance_pct = 50.0 if holding_period == "1 week" else 100.0
        st.number_input("Rebalance step (%)", value=rebalance_pct, disabled=True, key="bt_up_alpha_disabled")
    else:
        rebalance_pct = st.number_input(
            "Rebalance step (%)",
            min_value=10.0,
            max_value=100.0,
            value=100.0,
            step=10.0,
            key="bt_up_alpha",
        )

turnover_penalty = float(penalty_bps) / 10000.0
rebalance_alpha = float(rebalance_pct) / 100.0

oos_start = st.text_input(
    "True OOS evaluation start",
    "2019-01-01",
    key="bt_up_oos_start",
    help="The first realized backtest return occurs on or after this date; all model inputs come strictly from earlier observations.",
)

if turnover_mode == "Validated preset" and holding_period not in ("1 week", "2 weeks"):
    st.warning(
        "TC25 is available for this horizon, but its best setting was validated only for the 1-week and 2-week experiments."
    )

with st.expander("What this backtest does"):
    st.write(f"Nested Best-T grid: {CANDIDATE_T}")
    st.latex(r"\hat T_t=\arg\max_T \frac{1}{K}\sum_{k=1}^{K} CER_k(T)")
    st.latex(r"r^{net}_t=r^{gross}_t-c\,TO_t")
    st.caption(
        "For the turnover-controlled method, the optimizer also includes an L1 turnover penalty before the realized trading cost is deducted."
    )

# -----------------------------------------------------------------------------
# 3. Run and results
# -----------------------------------------------------------------------------
if st.button("▶ Run strategy backtest", type="primary", use_container_width=True, key="bt_up_run"):
    with st.spinner("Running rolling OOS portfolio decisions and realized P&L..."):
        summary, detail, latest, tdiag = _cached_backtest(
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
        )
    st.session_state["bt_up_result"] = {
        "summary": summary,
        "detail": detail,
        "latest": latest,
        "tdiag": tdiag,
        "strategy": strategy,
        "holding_period": holding_period,
        "gamma": float(gamma),
        "periods_per_year": ppy,
        "transaction_cost": float(transaction_cost_bps) / 10000.0,
        "initial_capital": float(initial_capital),
    }

if "bt_up_result" not in st.session_state:
    st.info("Choose the settings and run the backtest.")
    st.stop()

res = st.session_state["bt_up_result"]
strategy = res["strategy"]
detail = res["detail"].copy()
ppy = int(res["periods_per_year"])
cost = float(res["transaction_cost"])
capital = float(res["initial_capital"])

gross = detail[f"return__{strategy}"].to_numpy(dtype=float)
turnover = detail[f"turnover__{strategy}"].to_numpy(dtype=float)
net = gross - cost * turnover
metrics_gross = performance_metrics(gross, gamma=res["gamma"], periods_per_year=ppy)
metrics_net = performance_metrics(net, gamma=res["gamma"], periods_per_year=ppy)

backtest_start = str(pd.Timestamp(detail["Date"].iloc[0]).date())
backtest_end = str(pd.Timestamp(detail["Date"].iloc[-1]).date())
holding_period = str(res["holding_period"])
period_count = len(detail)
gross_final = capital * float(np.prod(1.0 + gross))
net_final = capital * float(np.prod(1.0 + net))
net_total_return = net_final / capital - 1.0

st.subheader(f"3. Results — {strategy}")
st.caption(
    "These values cover the complete historical test, not one holding period."
)

st.markdown(f"**Backtest dates:** {backtest_start} to {backtest_end}")
d1, d2 = st.columns(2)
d1.metric("Holding / rebalance interval", holding_period)
d2.metric("Number of holding periods", f"{period_count:,}")

r1, r2, r3 = st.columns(3)
r1.metric("Starting value", f"${capital:,.2f}")
r2.metric("Final after costs", f"${net_final:,.2f}")
r3.metric("Final before costs", f"${gross_final:,.2f}")

r4, r5, r6 = st.columns(3)
r4.metric("Total return after costs", f"{net_total_return:.2%}")
r5.metric("Annualized return", f"{metrics_net['CAGR']:.2%}")
r6.metric("Annualized volatility", f"{metrics_net['Annualized vol']:.2%}")

s1, s2, s3 = st.columns(3)
s1.metric("Net Sharpe", f"{metrics_net['Sharpe']:.3f}")
s2.metric("Maximum drawdown", f"{metrics_net['Max drawdown']:.2%}")
s3.metric("Average turnover", f"{float(np.mean(turnover)):.2%}")

t_values = detail["T"].to_numpy(dtype=float)
t_counts = pd.Series(t_values).value_counts().sort_index()
most_frequent_t = float(t_counts[t_counts == t_counts.max()].index.min())
t1, t2, t3, t4 = st.columns(4)
t1.metric("Latest selected T", f"{t_values[-1]:.3f}")
t2.metric("Most frequent T", f"{most_frequent_t:.3f}")
t3.metric("Median T", f"{float(np.median(t_values)):.3f}")
t4.metric("Mean T", f"{float(np.mean(t_values)):.3f}")

t_frequency = pd.DataFrame(
    {
        "T": t_counts.index.to_numpy(dtype=float),
        "Selected periods": t_counts.to_numpy(dtype=int),
        "Selection share": t_counts.to_numpy(dtype=float) / len(t_values),
    }
)
with st.expander("Selected T frequency"):
    st.dataframe(
        t_frequency.style.format({"T": "{:.3f}", "Selection share": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

wealth_index = pd.to_datetime(detail["Date"])
wealth = pd.DataFrame(index=wealth_index)
wealth["Selected strategy — gross"] = capital * np.cumprod(1.0 + gross)
wealth["Selected strategy — net"] = capital * np.cumprod(1.0 + net)
if strategy != "Equal Weight":
    ew_gross = detail["return__Equal Weight"].to_numpy(dtype=float)
    ew_to = detail["turnover__Equal Weight"].to_numpy(dtype=float)
    wealth["Equal Weight — net"] = capital * np.cumprod(1.0 + ew_gross - cost * ew_to)

st.markdown("**Cumulative wealth**")
st.line_chart(wealth, use_container_width=True)

metric_table = pd.DataFrame(
    [
        {"Metric": "Total return", "Gross": f"{metrics_gross['Total return']:.3%}", "Net": f"{metrics_net['Total return']:.3%}"},
        {"Metric": "CAGR", "Gross": f"{metrics_gross['CAGR']:.3%}", "Net": f"{metrics_net['CAGR']:.3%}"},
        {"Metric": "Annualized volatility", "Gross": f"{metrics_gross['Annualized vol']:.3%}", "Net": f"{metrics_net['Annualized vol']:.3%}"},
        {"Metric": "Sharpe", "Gross": f"{metrics_gross['Sharpe']:.3f}", "Net": f"{metrics_net['Sharpe']:.3f}"},
        {"Metric": "Realized CER", "Gross": f"{metrics_gross['Realized CER']:.3%}", "Net": f"{metrics_net['Realized CER']:.3%}"},
        {"Metric": "Max drawdown", "Gross": f"{metrics_gross['Max drawdown']:.3%}", "Net": f"{metrics_net['Max drawdown']:.3%}"},
        {"Metric": "Positive periods", "Gross": f"{metrics_gross['Positive periods']:.3%}", "Net": f"{metrics_net['Positive periods']:.3%}"},
    ]
)
st.dataframe(
    metric_table,
    use_container_width=True,
    hide_index=True,
)

period_df = pd.DataFrame(
    {
        "Date": pd.to_datetime(detail["Date"]),
        "Gross return": gross,
        "Turnover": turnover,
        "Trading cost": cost * turnover,
        "Net return": net,
        "Selected T": detail["T"].to_numpy(dtype=float),
    }
)
period_df["Year"] = period_df["Date"].dt.year
annual = (
    period_df.groupby("Year")["Net return"]
    .apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))
    .reset_index(name="Net return")
)
st.markdown("**Calendar-year realized return**")
st.dataframe(
    annual.style.format({"Net return": "{:.2%}"}),
    use_container_width=True,
    hide_index=True,
)

st.markdown("**Latest backtested target weights**")
latest_row = res["latest"][res["latest"]["Method"] == strategy].copy()
asset_cols = [c for c in latest_row.columns if c != "Method"]
st.dataframe(
    latest_row.style.format({c: "{:.2%}" for c in asset_cols}),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Rolling OOS detail"):
    period_display = period_df.copy()
    period_display["Date"] = period_display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        period_display.style.format(
            {
                "Gross return": "{:.3%}",
                "Turnover": "{:.2%}",
                "Trading cost": "{:.3%}",
                "Net return": "{:.3%}",
                "Selected T": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "⬇ Download backtest history (.csv)",
    data=period_display.to_csv(index=False).encode("utf-8"),
    file_name=f"backtest_{holding_period.replace(' ', '_')}_{strategy.replace(' ', '_').replace('/', '_')}.csv",
    mime="text/csv",
    use_container_width=True,
)
