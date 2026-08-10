import pandas as pd
import streamlit as st

from config import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_M, SUPPORTED_RULES
from core.backtest import backtest_summary, rolling_backtest
from core.data import load_returns_csv

st.title("Rolling Backtest")

uploaded = st.file_uploader("Upload return CSV", type=["csv"], key="bt_csv")
if uploaded is None:
    st.stop()

returns = load_returns_csv(uploaded)
n_obs = len(returns)
if n_obs < 3:
    st.error("Not enough return observations for a rolling backtest.")
    st.stop()

min_lb = min(12, n_obs - 1)
max_lb = min(240, n_obs - 1)

if min_lb >= max_lb:
    lookback = max_lb
    st.info(f"Using lookback = {lookback}")
else:
    lookback = st.slider(
        "Lookback",
        min_value=min_lb,
        max_value=max_lb,
        value=min(120, max_lb),
        step=1,
    )
gamma = st.number_input("Risk aversion γ", min_value=0.1, value=DEFAULT_GAMMA, step=0.1)
rule = st.selectbox("Rule", SUPPORTED_RULES)
estimator = st.selectbox("Estimator", ["Historical", "Diffusion"])
m = st.number_input("M", min_value=0, value=DEFAULT_M, step=100)
beta = st.number_input("β", min_value=0.01, value=DEFAULT_BETA, step=0.1)
steps = st.number_input("Reverse steps", min_value=10, value=100, step=10)

if st.button("Run backtest", type="primary"):
    with st.spinner("Running rolling backtest..."):
        bt = rolling_backtest(
            returns, lookback=int(lookback), gamma=gamma, rule=rule,
            estimator=estimator, m=int(m), beta=beta, n_steps=int(steps)
        )
    st.session_state["bt"] = bt
    st.session_state["bt_summary"] = backtest_summary(bt, gamma=gamma)

bt = st.session_state.get("bt")
summary = st.session_state.get("bt_summary")

if bt is not None:
    st.subheader("Cumulative wealth")
    st.line_chart((1 + bt["portfolio_return"]).cumprod())
    if summary:
        st.dataframe(pd.DataFrame({"Metric": summary.keys(), "Value": summary.values()}),
                     hide_index=True, use_container_width=True)
    st.dataframe(bt.tail(20), use_container_width=True)
