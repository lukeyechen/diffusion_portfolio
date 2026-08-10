import numpy as np
import pandas as pd
import streamlit as st

from core.data import load_returns_csv
from core.diffusion import diffusion_augmented_moments
from core.moments import covariance_condition_number, sample_moments
from core.tuning import constant_beta_horizon, optimal_b_star

st.title("Diffusion Diagnostics")

uploaded = st.file_uploader("Upload return CSV", type=["csv"], key="diag_csv")
if uploaded is None:
    st.stop()

returns = load_returns_csv(uploaded)
n_obs = len(returns)
if n_obs < 2:
    st.error("Not enough return observations.")
    st.stop()

min_lb = min(12, n_obs)
max_lb = min(240, n_obs)

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
m = st.number_input("Synthetic samples M", min_value=10, value=500, step=100)
beta = st.number_input("β", min_value=0.01, value=1.0, step=0.1)
steps = st.number_input("Reverse steps", min_value=10, value=200, step=10)

x = returns.iloc[-lookback:].to_numpy()
mu_h, sig_h = sample_moments(x, mle=True)

try:
    b_star = optimal_b_star(mu_h, sig_h)
    T = constant_beta_horizon(b_star, len(x), beta=beta, clip=True)
except ValueError as exc:
    st.warning(str(exc))
    b_star, T = np.nan, 0.0

mu_d, sig_d, fake, _ = diffusion_augmented_moments(
    x, m=int(m), horizon=T, beta=beta, n_steps=int(steps), seed=42
)

c1, c2, c3 = st.columns(3)
c1.metric("b*", f"{b_star:.6g}")
c2.metric("T", f"{T:.6g}")
c3.metric("b*/n", f"{(b_star / len(x)) if np.isfinite(b_star) else np.nan:.6g}")

diag = pd.DataFrame({
    "Metric": ["||mu_hist||", "||mu_diff||", "trace(Sigma_hist)", "trace(Sigma_diff)",
               "cond(Sigma_hist)", "cond(Sigma_diff)"],
    "Value": [np.linalg.norm(mu_h), np.linalg.norm(mu_d), np.trace(sig_h), np.trace(sig_d),
              covariance_condition_number(sig_h), covariance_condition_number(sig_d)]
})
st.dataframe(diag, hide_index=True, use_container_width=True)

if len(fake):
    fake_df = pd.DataFrame(fake, columns=returns.columns)
    st.subheader("Generated-sample summary")
    st.dataframe(fake_df.describe().T, use_container_width=True)
