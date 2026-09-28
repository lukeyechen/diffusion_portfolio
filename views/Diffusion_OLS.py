from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.diffusion_exposure import (
    backtest_exposure,
    demo_exposure_data,
    latest_exposure,
    prepare_exposure_data,
    summarize_exposure,
)


@st.cache_data(show_spinner=False)
def _run_backtest(
    frame,
    return_columns,
    predictor_columns,
    window,
    fixed_b,
    b_grid,
    validation,
    response_scale,
    gamma,
    cap,
    cost_bps,
):
    return backtest_exposure(
        frame,
        list(return_columns),
        list(predictor_columns),
        window=int(window),
        b=float(fixed_b),
        b_grid=None if b_grid is None else tuple(b_grid),
        validation=int(validation),
        response_scale=float(response_scale),
        gamma=float(gamma),
        cap=float(cap),
        cost_bps=float(cost_bps),
    )


st.title("Diffusion OLS: Market Exposure")
st.caption(
    "Convert a next-period excess-return forecast into a long-only allocation "
    "between selected risky assets and cash. No trades are placed."
)
st.info(
    "For one risky asset, the tab implements w = clip(ŷ / (γv̂), 0, 1). "
    "Here v̂ is the estimated variance of the next-period excess return—not "
    "uncertainty about the forecast. With several assets, it uses the "
    "covariance-aware mean–variance extension and leaves unallocated wealth in cash."
)

with st.expander("Model and timing assumptions"):
    st.latex(r"R_{p,t+1}-R_{f,t+1}=w_t y_{t+1}")
    st.latex(
        r"w_t^*=\arg\max_{0\leq w\leq1} "
        r"\left\{w\widehat y_{\mathrm{diff}}(x_t)-"
        r"\frac{\gamma}{2}w^2\widehat v_t\right\}"
    )
    st.latex(
        r"w_t^{\mathrm{diff}}=\min\left\{1,\max\left\{0,"
        r"\frac{\widehat y_{\mathrm{diff}}(x_t)}{\gamma\widehat v_t}"
        r"\right\}\right\}"
    )
    st.markdown(
        "Every `x_` value on a row must have been observable **before** that "
        "row's return. The rolling test trains only on earlier rows. Returns and "
        "`rf` are simple per-period decimals (for example, `0.01` means 1%)."
    )

# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
st.subheader("1. Point-in-time data")
source = st.radio(
    "Data source",
    ["Synthetic demonstration", "Upload CSV"],
    horizontal=True,
    key="diff_ols_source",
)

if source == "Synthetic demonstration":
    raw_frame = demo_exposure_data()
    st.warning(
        "A, B and C are fictional assets. Demo results verify the workflow; "
        "they are not evidence of investment performance."
    )
    st.download_button(
        "Download example CSV schema",
        raw_frame.to_csv(index=False).encode("utf-8"),
        file_name="diffusion_ols_example.csv",
        mime="text/csv",
    )
else:
    uploaded = st.file_uploader(
        "CSV columns: date, rf, one or more ret_SYMBOL, and one or more x_NAME",
        type=["csv"],
        key="diff_ols_csv",
    )
    if uploaded is None:
        st.info("Upload a point-in-time CSV to continue, or select the synthetic demonstration.")
        st.stop()
    raw_frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))

available_assets = [
    column.removeprefix("ret_")
    for column in raw_frame.columns
    if column.startswith("ret_")
]
selected_assets = st.multiselect(
    "Risky assets",
    available_assets,
    default=available_assets[:1],
    key="diff_ols_assets",
    help="Choose one asset for the exact scalar formula above; choose several for its covariance-aware extension.",
)
if not selected_assets:
    st.error("Select at least one risky asset.")
    st.stop()

try:
    frame, return_columns, predictor_columns = prepare_exposure_data(
        raw_frame, selected_assets
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

d1, d2, d3, d4 = st.columns(4)
d1.metric("Observations", len(frame))
d2.metric("Risky assets", len(return_columns))
d3.metric("Predictors", len(predictor_columns))
d4.metric("Latest completed period", str(frame["date"].iloc[-1].date()))

with st.expander("Required CSV schema"):
    st.markdown(
        "- `date`: completed holding-period date, strictly increasing\n"
        "- `rf`: matching risk-free simple return\n"
        "- `ret_SYMBOL`: simple total return for each risky asset\n"
        "- `x_NAME`: predictor known before the return on the same row\n\n"
        "Do not backfill revised information. Lag end-of-period predictors before upload."
    )
    st.dataframe(frame.head(8), width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# 2. Settings
# -----------------------------------------------------------------------------
st.subheader("2. Forecast and exposure settings")
minimum_window = len(predictor_columns) + 3
if len(frame) <= minimum_window:
    st.error(
        f"Need more than {minimum_window} rows for {len(predictor_columns)} predictors."
    )
    st.stop()

s1, s2, s3, s4 = st.columns(4)
with s1:
    window = st.number_input(
        "Rolling estimation window",
        min_value=minimum_window,
        max_value=len(frame) - 1,
        value=min(120, len(frame) - 1),
        step=1,
        key="diff_ols_window",
    )
with s2:
    gamma = st.number_input(
        "Risk aversion γ",
        min_value=0.01,
        value=5.0,
        step=0.25,
        key="diff_ols_gamma",
        help="Larger γ reduces risky exposure for the same forecast and variance.",
    )
with s3:
    cap = st.number_input(
        "Maximum weight per risky asset",
        min_value=0.01,
        max_value=1.0,
        value=1.0 if len(return_columns) == 1 else max(0.4, 1 / len(return_columns)),
        step=0.05,
        key="diff_ols_cap",
    )
with s4:
    cost_bps = st.number_input(
        "Trading cost (bps per turnover)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=1.0,
        key="diff_ols_cost",
    )

s5, s6, s7 = st.columns(3)
with s5:
    periods_per_year = st.number_input(
        "Periods per year",
        min_value=1,
        max_value=365,
        value=12,
        step=1,
        key="diff_ols_ppy",
    )
with s6:
    response_scale = st.number_input(
        "Diffusion response scale",
        min_value=0.0001,
        value=1.0,
        step=1.0,
        format="%.4f",
        key="diff_ols_response_scale",
        help="1 uses decimal-return coordinates. This scale changes the diffusion estimator, so record it; OLS itself is invariant.",
    )
with s7:
    tuning_mode = st.selectbox(
        "Diffusion b selection",
        ["Fixed b", "Past-only validation"],
        key="diff_ols_tuning_mode",
    )

fixed_b = 1.0
b_grid = None
validation = min(24, max(1, int(window) - minimum_window))
if tuning_mode == "Fixed b":
    fixed_b = st.number_input(
        "Fixed b (a = b / window)",
        min_value=0.0,
        max_value=float(window) - 0.001,
        value=min(1.0, float(window) - 0.001),
        step=0.25,
        key="diff_ols_fixed_b",
    )
else:
    t1, t2 = st.columns([2, 1])
    with t1:
        grid_text = st.text_input(
            "Candidate b values",
            "0, 0.25, 1, 4, 16",
            key="diff_ols_grid",
        )
    with t2:
        validation = st.number_input(
            "Inner validation periods",
            min_value=1,
            max_value=max(1, int(window) - minimum_window),
            value=validation,
            step=1,
            key="diff_ols_validation",
        )
    try:
        b_grid = tuple(float(value.strip()) for value in grid_text.split(",") if value.strip())
    except ValueError:
        st.error("Candidate b values must be comma-separated numbers.")
        st.stop()
    smallest_inner = int(window) - int(validation)
    if not b_grid or any(value < 0 or value >= smallest_inner for value in b_grid):
        st.error(f"Each candidate b must satisfy 0 ≤ b < {smallest_inner}.")
        st.stop()

st.markdown("**Predictors available now for the next holding period**")
st.caption(
    "Defaults show the final historical row only for convenience. Replace them "
    "with the values actually observable at the decision time."
)
predictor_values = []
predictor_inputs = st.columns(min(4, len(predictor_columns)))
for index, column in enumerate(predictor_columns):
    with predictor_inputs[index % len(predictor_inputs)]:
        predictor_values.append(
            st.number_input(
                column,
                value=float(frame[column].iloc[-1]),
                format="%.6f",
                key=f"diff_ols_current_{column}",
            )
        )

current_signature = (
    tuple(return_columns),
    tuple(predictor_columns),
    int(pd.util.hash_pandas_object(frame, index=True).sum()),
    int(window),
    float(fixed_b),
    b_grid,
    int(validation),
    float(response_scale),
    float(gamma),
    float(cap),
    float(cost_bps),
    int(periods_per_year),
    tuple(float(value) for value in predictor_values),
)

run = st.button(
    "Run Diffusion OLS exposure analysis",
    type="primary",
    width="stretch",
)
if not run and "diff_ols_output" not in st.session_state:
    st.stop()

if run:
    try:
        with st.spinner("Running chronological forecasts and allocations..."):
            recommendation = latest_exposure(
                frame,
                return_columns,
                predictor_columns,
                np.asarray(predictor_values, dtype=float),
                window=int(window),
                b=float(fixed_b),
                b_grid=b_grid,
                validation=int(validation),
                response_scale=float(response_scale),
                gamma=float(gamma),
                cap=float(cap),
            )
            results, forecasts = _run_backtest(
                frame,
                tuple(return_columns),
                tuple(predictor_columns),
                int(window),
                float(fixed_b),
                b_grid,
                int(validation),
                float(response_scale),
                float(gamma),
                float(cap),
                float(cost_bps),
            )
        st.session_state["diff_ols_output"] = (
            recommendation,
            results,
            forecasts,
            current_signature,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        st.error(str(error))
        st.stop()

recommendation, results, forecasts, saved_signature = st.session_state[
    "diff_ols_output"
]
(
    saved_returns,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    saved_gamma,
    saved_cap,
    _,
    saved_ppy,
    _,
) = saved_signature
if saved_signature != current_signature:
    st.warning("Settings or current predictors changed. Run the analysis again to refresh the results.")

# -----------------------------------------------------------------------------
# 3. Recommendation
# -----------------------------------------------------------------------------
st.subheader("3. Next-period exposure")
asset_names = [column.removeprefix("ret_") for column in saved_returns]
forecast = recommendation["forecasts"]["Diffusion OLS"]
weights = recommendation["weights"]["Diffusion OLS"]
variance = np.diag(recommendation["covariance"])
recommendation_table = pd.DataFrame(
    {
        "Asset": asset_names,
        "Forecast excess return": forecast,
        "Estimated variance": variance,
        "Risky weight": weights,
    }
)
cash_weight = max(0.0, 1.0 - float(np.sum(weights)))

r1, r2, r3 = st.columns(3)
r1.metric("Selected b", f"{recommendation['selected_b']:.4g}")
r2.metric("a = b / window", f"{recommendation['a']:.6f}")
r3.metric("Cash weight", f"{cash_weight:.2%}")

st.dataframe(
    recommendation_table.style.format(
        {
            "Forecast excess return": "{:.4%}",
            "Estimated variance": "{:.8f}",
            "Risky weight": "{:.2%}",
        }
    ),
    width="stretch",
    hide_index=True,
)

if len(saved_returns) == 1:
    raw_weight = float(forecast[0] / (saved_gamma * variance[0]))
    st.success(
        f"Calculation: {forecast[0]:.6f} / ({saved_gamma:.4g} × "
        f"{variance[0]:.8f}) = {raw_weight:.4f}; clipped to "
        f"[0, {saved_cap:.2f}] gives {weights[0]:.2%} in {asset_names[0]} and "
        f"{cash_weight:.2%} in cash."
    )
else:
    st.caption(
        "For multiple assets, the displayed weights jointly account for their covariance, "
        "subject to long-only, per-asset caps and total risky weight ≤ 100%."
    )

# -----------------------------------------------------------------------------
# 4. Out-of-sample evidence
# -----------------------------------------------------------------------------
st.subheader("4. Chronological out-of-sample test")
summary = summarize_exposure(
    results,
    forecasts,
    periods_per_year=int(saved_ppy),
    gamma=float(saved_gamma),
)
percent_columns = [
    "Total return",
    "CAGR",
    "Annualized volatility",
    "Annualized MV excess",
    "Maximum drawdown",
    "Average turnover",
]
st.dataframe(
    summary.style.format(
        {
            **{column: "{:.2%}" for column in percent_columns},
            "Excess Sharpe": "{:.3f}",
            "Forecast MSE": "{:.8f}",
        },
        na_rep="—",
    ),
    width="stretch",
    hide_index=True,
)

wealth = results[["date", "method", "net_return"]].copy()
wealth["Wealth (start = 1)"] = wealth.groupby("method", sort=False)[
    "net_return"
].transform(lambda values: (1.0 + values).cumprod())
figure = px.line(
    wealth,
    x="date",
    y="Wealth (start = 1)",
    color="method",
    title="Net wealth after modeled trading costs",
)
st.plotly_chart(figure, width="stretch")

with st.expander("Rolling forecasts, selected b, weights and costs"):
    selected_method = st.selectbox(
        "Allocation detail",
        results["method"].drop_duplicates().tolist(),
        key="diff_ols_detail_method",
    )
    detail = results.loc[results["method"] == selected_method].copy()
    st.dataframe(detail, width="stretch", hide_index=True)
    st.markdown("**Forecast detail**")
    st.dataframe(forecasts, width="stretch", hide_index=True)

c1, c2, c3 = st.columns(3)
c1.download_button(
    "Download performance summary",
    summary.to_csv(index=False).encode("utf-8"),
    file_name="diffusion_ols_summary.csv",
    mime="text/csv",
)
c2.download_button(
    "Download backtest detail",
    results.to_csv(index=False).encode("utf-8"),
    file_name="diffusion_ols_backtest.csv",
    mime="text/csv",
)
c3.download_button(
    "Download forecast detail",
    forecasts.to_csv(index=False).encode("utf-8"),
    file_name="diffusion_ols_forecasts.csv",
    mime="text/csv",
)

st.warning(
    "This is a research backtest, not investment advice. It does not model taxes, "
    "market impact, data revisions, or all execution frictions. A strong historical "
    "result does not establish future performance."
)
