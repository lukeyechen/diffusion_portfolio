from __future__ import annotations

import io
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    CONSTRAINT_MODES,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_M,
    DEFAULT_SEED,
    DEFAULT_MAX_GROSS_EXPOSURE,
    DEFAULT_MAX_LONG_WEIGHT,
    DEFAULT_MAX_SHORT_WEIGHT,
    SUPPORTED_RULES,
)
from core.data import (
    download_yahoo_horizon_returns,
    download_yahoo_monthly_returns,
    load_returns_csv,
)
from core.diffusion import diffusion_augmented_moments
from core.horizon_study import return_horizon_study
from core.metrics import (
    certainty_equivalent,
    gross_exposure,
    portfolio_mean,
    portfolio_volatility,
    sharpe_ratio,
)
from core.moments import covariance_condition_number, sample_moments
from core.monthly_oos import monthly_rebalance_oos_comparison
from core.neural_score import neural_diffusion_augmented_moments
from core.portfolio_rules import compute_weights
from core.regime_regression import (
    build_relative_performance_dataset,
    make_threshold_scatter_data,
    run_piecewise_slope_tests,
    run_regime_regressions,
)
from core.tuning import theoretical_horizon



@st.cache_data(show_spinner=False)
def _cached_current_gaussian(
    x,
    m,
    horizon,
    beta,
    n_steps,
    seed,
):
    return diffusion_augmented_moments(
        x,
        m=int(m),
        horizon=float(horizon),
        beta=float(beta),
        n_steps=int(n_steps),
        seed=int(seed),
    )


@st.cache_data(show_spinner=False)
def _cached_current_neural(
    x,
    m,
    horizon,
    beta,
    n_steps,
    epochs,
    batch_size,
    learning_rate,
    hidden_dim,
    validation_fraction,
    patience,
    min_delta,
    seed,
):
    return neural_diffusion_augmented_moments(
        x,
        m=int(m),
        horizon=float(horizon),
        beta=float(beta),
        n_steps=int(n_steps),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        hidden_dim=int(hidden_dim),
        validation_fraction=float(validation_fraction),
        patience=int(patience),
        min_delta=float(min_delta),
        seed=int(seed),
    )





def _build_section_d_export_zip(snap):
    """Export only the Current-Window Estimator / Portfolio-Rule comparison."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        table_map = {
            "D1_estimator_based_metrics.csv": snap.get("snapshot"),
            "D2_common_historical_benchmark.csv": snap.get("benchmark"),
            "D3_portfolio_weights.csv": snap.get("weights"),
            "D4_moment_diagnostics.csv": snap.get("moments"),
            "D_neural_training_diagnostics.csv": snap.get("neural_training"),
            "D_portfolio_sync_audit.csv": snap.get("sync_audit"),
        }

        for name, df in table_map.items():
            if isinstance(df, pd.DataFrame):
                zf.writestr(name, df.to_csv(index=False))

        # D1 chart
        d1 = snap.get("snapshot")
        if isinstance(d1, pd.DataFrame) and not d1.empty:
            fig = px.bar(
                d1,
                x="Portfolio rule",
                y="CER",
                color="Estimator",
                barmode="group",
                title="D1 Estimator-Based Current-Window CER",
            )
            zf.writestr(
                "D1_estimator_based_CER.html",
                fig.to_html(full_html=True, include_plotlyjs="cdn"),
            )

        # D2 chart
        d2 = snap.get("benchmark")
        if isinstance(d2, pd.DataFrame) and not d2.empty:
            fig = px.bar(
                d2,
                x="Portfolio rule",
                y="Benchmark CER",
                color="Estimator",
                barmode="group",
                title="D2 Common Historical-Benchmark CER",
            )
            zf.writestr(
                "D2_common_benchmark_CER.html",
                fig.to_html(full_html=True, include_plotlyjs="cdn"),
            )

        # D3 chart
        d3 = snap.get("weights")
        if isinstance(d3, pd.DataFrame) and not d3.empty:
            fig = px.bar(
                d3,
                x="Asset",
                y="Weight",
                color="Estimator",
                barmode="group",
                facet_col="Portfolio rule",
                title="D3 Portfolio Weight Comparison",
            )
            zf.writestr(
                "D3_portfolio_weights.html",
                fig.to_html(full_html=True, include_plotlyjs="cdn"),
            )

        # Metadata
        metadata = pd.DataFrame(
            {
                "Field": [
                    "Portfolio revision",
                    "Used shared Portfolio",
                    "T",
                    "Theoretical T",
                    "T source",
                    "b*",
                    "b*/n signal",
                    "gamma",
                    "constraint mode",
                    "max long weight",
                    "max short weight",
                    "max gross exposure",
                ],
                "Value": [
                    snap.get("portfolio_revision"),
                    snap.get("used_shared_portfolio"),
                    snap.get("T"),
                    snap.get("T_theory"),
                    snap.get("T_source"),
                    snap.get("b_star"),
                    snap.get("signal"),
                    snap.get("effective_gamma"),
                    snap.get("effective_constraint_mode"),
                    snap.get("effective_max_long_weight"),
                    snap.get("effective_max_short_weight"),
                    snap.get("effective_max_gross_exposure"),
                ],
            }
        )
        zf.writestr("D_metadata.csv", metadata.to_csv(index=False))

        zf.writestr(
            "README.txt",
            "Section D saved-results package.\n"
            "Contains D1-D4 tables, diagnostics, synchronization audit, metadata, "
            "and interactive HTML charts.\n",
        )

    return buffer.getvalue()


def _build_method_comparison_export_zip(session_state):
    """Export every currently saved Method Comparison table plus chart-ready HTML."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if "mc_horizon_table" in session_state:
            df = session_state["mc_horizon_table"]
            zf.writestr("return_horizon_study.csv", df.to_csv(index=False))

            fig = px.bar(
                df,
                x="Horizon",
                y="b*/n",
                text_auto=".3f",
                title="Theoretical feasibility ratio by return horizon",
            )
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="b*/n = 1")
            zf.writestr(
                "return_horizon_study_chart.html",
                fig.to_html(full_html=True, include_plotlyjs="cdn"),
            )

        if "mc_selected_results" in session_state:
            res = session_state["mc_selected_results"]
            table_map = {
                "selected_oos_summary.csv": res.get("summary"),
                "selected_oos_detail.csv": res.get("detail"),
                "selected_oos_wealth.csv": res.get("wealth"),
                "selected_t_source_performance.csv": res.get("split_summary"),
                "selected_regime_summary.csv": res.get("regime_summary"),
                "selected_regime_detail.csv": res.get("regime_detail"),
                "selected_regression_table.csv": res.get("regression_table"),
                "selected_regression_diagnostics.csv": res.get("regression_diagnostics"),
                "selected_piecewise_slope_tests.csv": res.get("slope_tests"),
                "selected_relative_performance_data.csv": res.get("relative_df"),
            }
            for name, df in table_map.items():
                if isinstance(df, pd.DataFrame):
                    zf.writestr(name, df.to_csv(index=True if name == "selected_oos_wealth.csv" else False))

            wealth = res.get("wealth")
            if isinstance(wealth, pd.DataFrame) and not wealth.empty:
                wealth_plot = wealth.reset_index()
                xcol = wealth_plot.columns[0]
                fig = px.line(
                    wealth_plot,
                    x=xcol,
                    y=list(wealth.columns),
                    title=f"{res.get('horizon', '')} OOS cumulative wealth",
                )
                zf.writestr(
                    "selected_oos_cumulative_wealth_chart.html",
                    fig.to_html(full_html=True, include_plotlyjs="cdn"),
                )

            rel = res.get("relative_df")
            if isinstance(rel, pd.DataFrame) and not rel.empty:
                scatter_data = make_threshold_scatter_data(rel)
                scatter_long = pd.concat(
                    [
                        scatter_data[
                            ["Realized date", "b*/n", "T source", "Neural - Historical"]
                        ].rename(columns={"Neural - Historical": "Relative return"}).assign(
                            Strategy="Neural - Historical"
                        ),
                        scatter_data[
                            ["Realized date", "b*/n", "T source", "Gaussian - Historical"]
                        ].rename(columns={"Gaussian - Historical": "Relative return"}).assign(
                            Strategy="Gaussian - Historical"
                        ),
                    ],
                    ignore_index=True,
                )
                fig = px.scatter(
                    scatter_long,
                    x="b*/n",
                    y="Relative return",
                    color="Strategy",
                    symbol="T source",
                    hover_data=["Realized date"],
                    title="b*/n vs diffusion relative OOS performance",
                )
                fig.add_vline(x=1.0, line_dash="dash", annotation_text="b*/n = 1")
                fig.add_hline(y=0.0, line_dash="dot")
                zf.writestr(
                    "selected_bstar_ratio_scatter.html",
                    fig.to_html(full_html=True, include_plotlyjs="cdn"),
                )

        if "mc_multi_results" in session_state:
            multi = session_state["mc_multi_results"]
            for name, key in [
                ("multi_horizon_oos_performance.csv", "results"),
                ("multi_horizon_coverage.csv", "diagnostics"),
                ("multi_horizon_t_source_performance.csv", "split_results"),
                ("multi_horizon_regime_diagnostics.csv", "regime_results"),
            ]:
                df = multi.get(key)
                if isinstance(df, pd.DataFrame):
                    zf.writestr(name, df.to_csv(index=False))

            df = multi.get("results")
            if isinstance(df, pd.DataFrame) and not df.empty:
                fig = px.bar(
                    df,
                    x="Model horizon",
                    y="CER / month",
                    color="Method",
                    barmode="group",
                    title="Monthly OOS CER by model horizon and method",
                )
                zf.writestr(
                    "multi_horizon_cer_chart.html",
                    fig.to_html(full_html=True, include_plotlyjs="cdn"),
                )

        if "mc_snapshot" in session_state:
            snap = session_state["mc_snapshot"]
            for name, key in [
                ("current_window_estimator_based_metrics.csv", "snapshot"),
                ("current_window_common_benchmark_metrics.csv", "benchmark"),
                ("current_window_portfolio_weights.csv", "weights"),
                ("current_window_moment_comparison.csv", "moments"),
                ("current_window_neural_training_summary.csv", "neural_training"),
            ]:
                df = snap.get(key)
                if isinstance(df, pd.DataFrame):
                    zf.writestr(name, df.to_csv(index=False))

            df = snap.get("snapshot")
            if isinstance(df, pd.DataFrame) and not df.empty:
                fig = px.bar(
                    df,
                    x="Portfolio rule",
                    y="CER",
                    color="Estimator",
                    barmode="group",
                    title="Estimator-Based Current-Window CER",
                )
                zf.writestr(
                    "current_window_estimator_based_cer_chart.html",
                    fig.to_html(full_html=True, include_plotlyjs="cdn"),
                )

            bench = snap.get("benchmark")
            if isinstance(bench, pd.DataFrame) and not bench.empty:
                fig = px.bar(
                    bench,
                    x="Portfolio rule",
                    y="Benchmark CER",
                    color="Estimator",
                    barmode="group",
                    title="Common Historical-Benchmark CER",
                )
                zf.writestr(
                    "current_window_common_benchmark_cer_chart.html",
                    fig.to_html(full_html=True, include_plotlyjs="cdn"),
                )

        zf.writestr(
            "README.txt",
            "Method Comparison saved-results package.\n"
            "CSV files reproduce the saved tables. HTML files contain interactive charts.\n",
        )

    return buffer.getvalue()


st.title("Method Comparison")
st.caption(
    "Dedicated research page for estimator comparisons, return-horizon studies, "
    "monthly OOS tests, cumulative-wealth charts, regime diagnostics, and regressions."
)


mc_stop_col, mc_save_col, mc_download_col = st.columns([1, 1, 2])

with mc_stop_col:
    if st.button(
        "■ Stop",
        key="mc_global_stop",
        use_container_width=True,
        help="Requests a rerun and stops the active comparison workflow.",
    ):
        st.session_state["mc_stop_requested"] = True
        st.warning("Method Comparison stop requested.")
        st.stop()

with mc_save_col:
    if st.button(
        "💾 Save results",
        key="mc_save_results",
        use_container_width=True,
    ):
        try:
            st.session_state["mc_saved_zip"] = _build_method_comparison_export_zip(
                st.session_state
            )
            st.success("Comparison results package prepared.")
        except Exception as exc:
            st.error(f"Could not save comparison results: {exc}")

with mc_download_col:
    if "mc_saved_zip" in st.session_state:
        st.download_button(
            "⬇ Download saved comparison tables + charts (.zip)",
            data=st.session_state["mc_saved_zip"],
            file_name="method_comparison_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #28A745 !important;
        border-color: #28A745 !important;
        color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #218838 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Data source
# ------------------------------------------------------------
st.subheader("Comparison Data")

source = st.radio(
    "Data source",
    ["Yahoo Finance", "Upload CSV"],
    horizontal=True,
    key="mc_source",
)

ticker_list = None
start_date = None
uploaded_returns = None

if source == "Yahoo Finance":
    default_tickers = ",".join(
        st.session_state.get(
            "ticker_list",
            ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        )
    )
    default_start = st.session_state.get("start_date", "2000-01-01")

    ticker_text = st.text_input(
        "Tickers",
        default_tickers,
        key="mc_tickers",
    )
    start_date = st.text_input(
        "Start date",
        default_start,
        key="mc_start",
    )
    ticker_list = [x.strip().upper() for x in ticker_text.split(",") if x.strip()]
else:
    uploaded = st.file_uploader(
        "Upload return CSV",
        type=["csv"],
        key="mc_csv",
    )
    if uploaded is not None:
        uploaded_returns = load_returns_csv(uploaded)

# ------------------------------------------------------------
# Shared settings
# ------------------------------------------------------------
st.subheader("Shared Comparison Settings")

c1, c2, c3, c4 = st.columns(4)
with c1:
    model_horizon = st.selectbox(
        "Selected model return horizon",
        ["3M", "6M", "12M"],
        index=2,
        help="Used for the selected-horizon monthly OOS comparison.",
    )
with c2:
    monthly_oos_periods = st.slider(
        "Monthly OOS periods",
        min_value=12,
        max_value=120,
        value=120,
        step=12,
        help="60 months = 5 years; 120 months = 10 years.",
    )
with c3:
    gamma = st.number_input(
        "Risk aversion γ",
        min_value=0.1,
        value=float(DEFAULT_GAMMA),
        step=0.1,
    )
with c4:
    rule = st.selectbox(
        "Portfolio rule",
        SUPPORTED_RULES,
        index=1 if "Mean-Variance" in SUPPORTED_RULES else 0,
    )

c5, c6, c7, c8 = st.columns(4)
with c5:
    constraint_mode = st.selectbox(
        "Portfolio constraint",
        CONSTRAINT_MODES,
        index=0,
    )
with c6:
    max_long_weight = st.number_input(
        "Maximum long weight",
        min_value=0.05,
        max_value=1.0,
        value=float(DEFAULT_MAX_LONG_WEIGHT),
        step=0.05,
    )
with c7:
    max_short_weight = st.number_input(
        "Maximum short weight",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_MAX_SHORT_WEIGHT),
        step=0.05,
    )
with c8:
    max_gross_exposure = st.number_input(
        "Maximum gross exposure",
        min_value=1.0,
        max_value=5.0,
        value=float(DEFAULT_MAX_GROSS_EXPOSURE),
        step=0.1,
    )

st.markdown("**Diffusion / neural settings**")
n1, n2, n3, n4 = st.columns(4)
with n1:
    m = st.number_input(
        "Synthetic samples M",
        min_value=50,
        max_value=5000,
        value=int(DEFAULT_M),
        step=50,
    )
with n2:
    beta = st.number_input(
        "Constant β",
        min_value=0.01,
        value=float(DEFAULT_BETA),
        step=0.1,
    )
with n3:
    n_steps = st.number_input(
        "Reverse SDE steps",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
    )
with n4:
    lookback = st.number_input(
        "Rolling estimation lookback",
        min_value=20,
        max_value=240,
        value=120,
        step=1,
    )

n5, n6, n7, n8 = st.columns(4)
with n5:
    neural_epochs = st.number_input(
        "Maximum neural epochs",
        min_value=50,
        max_value=3000,
        value=300,
        step=50,
    )
with n6:
    neural_hidden = st.selectbox(
        "Hidden width",
        [32, 64, 128, 256],
        index=2,
    )
with n7:
    neural_lr = st.selectbox(
        "Learning rate",
        [1e-4, 3e-4, 1e-3, 3e-3],
        index=2,
        format_func=lambda x: f"{x:.0e}",
    )
with n8:
    neural_batch = st.selectbox(
        "Batch size",
        [16, 32, 64, 128],
        index=1,
    )

n9, n10, n11 = st.columns(3)
with n9:
    neural_val_fraction = st.slider(
        "Neural validation fraction",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )
with n10:
    neural_patience = st.number_input(
        "Early-stopping patience",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )
with n11:
    neural_min_delta = st.selectbox(
        "Minimum validation improvement",
        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        index=2,
        format_func=lambda x: f"{x:.0e}",
    )


@st.cache_data(show_spinner=False)
def _cached_monthly_oos_comparison(
    model_returns,
    monthly_realized_returns,
    lookback,
    test_periods,
    gamma,
    rule,
    m,
    beta,
    n_steps,
    constraint_mode,
    max_long_weight,
    max_short_weight,
    max_gross_exposure,
    neural_epochs,
    neural_batch,
    neural_learning_rate,
    neural_hidden_dim,
    neural_validation_fraction,
    neural_patience,
    neural_min_delta,
    seed,
):
    """Cache expensive rolling OOS comparisons by data + all model settings."""
    return monthly_rebalance_oos_comparison(
        model_returns=model_returns,
        monthly_realized_returns=monthly_realized_returns,
        lookback=int(lookback),
        test_periods=int(test_periods),
        gamma=float(gamma),
        rule=rule,
        m=int(m),
        beta=float(beta),
        n_steps=int(n_steps),
        constraint_mode=constraint_mode,
        max_long_weight=float(max_long_weight),
        max_short_weight=float(max_short_weight),
        max_gross_exposure=float(max_gross_exposure),
        neural_epochs=int(neural_epochs),
        neural_batch=int(neural_batch),
        neural_learning_rate=float(neural_learning_rate),
        neural_hidden_dim=int(neural_hidden_dim),
        neural_validation_fraction=float(neural_validation_fraction),
        neural_patience=int(neural_patience),
        neural_min_delta=float(neural_min_delta),
        seed=int(seed),
    )


# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------
def _get_horizon_data(months: int) -> pd.DataFrame:
    if source == "Yahoo Finance":
        return download_yahoo_horizon_returns(
            ticker_list,
            start_date,
            months,
        )

    # Uploaded CSV is treated as the selected horizon only.
    if uploaded_returns is None:
        raise ValueError("Upload a CSV first.")
    if months != {"3M": 3, "6M": 6, "12M": 12}[model_horizon]:
        raise ValueError(
            "The 3M/6M/12M automatic study requires Yahoo Finance because one uploaded "
            "CSV cannot represent all three return horizons."
        )
    return uploaded_returns.copy()


def _get_monthly_realized() -> pd.DataFrame:
    if source != "Yahoo Finance":
        raise ValueError(
            "Monthly OOS evaluation currently requires Yahoo Finance so the app can "
            "construct actual next-month realized returns."
        )
    return download_yahoo_monthly_returns(ticker_list, start_date)


def _format_performance(df: pd.DataFrame):
    return df.style.format(
        {
            "Return / month": "{:.3%}",
            "Volatility / month": "{:.3%}",
            "Sharpe / month": "{:.3f}",
            "CER / month": "{:.3%}",
            "Average turnover": "{:.3f}",
            "Max drawdown": "{:.3%}",
            "OOS months": "{:.0f}",
        }
    )


# ============================================================
# A. Return-horizon theoretical study
# ============================================================
st.divider()
st.header("A. Return-Horizon Study")
st.caption(
    "Compares theoretical diffusion feasibility across 1M, 3M, 6M, and 12M returns."
)

if st.button(
    "Run Return-Horizon Study",
    type="primary",
    key="run_return_horizon_study",
    use_container_width=True,
):
    st.session_state["mc_stop_requested"] = False
    if source != "Yahoo Finance":
        st.error("The automatic 1M/3M/6M/12M study requires Yahoo Finance.")
    else:
        try:
            with st.spinner("Computing 1M / 3M / 6M / 12M theoretical horizons..."):
                horizon_data = {
                    "1M": download_yahoo_horizon_returns(ticker_list, start_date, 1),
                    "3M": download_yahoo_horizon_returns(ticker_list, start_date, 3),
                    "6M": download_yahoo_horizon_returns(ticker_list, start_date, 6),
                    "12M": download_yahoo_horizon_returns(ticker_list, start_date, 12),
                }
                table = return_horizon_study(horizon_data, beta=float(beta))
            st.session_state["mc_horizon_table"] = table
        except Exception as exc:
            st.error(f"Return-horizon study failed: {exc}")

if "mc_horizon_table" in st.session_state:
    table = st.session_state["mc_horizon_table"]
    st.dataframe(
        table.style.format(
            {
                "N/n": "{:.4f}",
                "b*": "{:.4f}",
                "b*/n": "{:.4f}",
                "T": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = table.copy()
    fig = px.bar(
        chart_df,
        x="Horizon",
        y="b*/n",
        text_auto=".3f",
        title="Theoretical feasibility ratio by return horizon",
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        annotation_text="Boundary b*/n = 1",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# B. Selected-horizon monthly OOS comparison
# ============================================================
st.divider()
st.header("B. Selected-Horizon Monthly OOS Comparison")
st.caption(
    "Historical, Gaussian Diffusion, and Neural Diffusion are re-estimated through "
    "time and evaluated on the same next real monthly return."
)

_bmode1, _bmode2 = st.columns([1, 2])
with _bmode1:
    b_speed_mode = st.selectbox(
        "B speed mode",
        ["Turbo", "Fast", "Research"],
        index=0,
        key="mc_b_speed_mode",
    )

if b_speed_mode == "Turbo":
    b_test_periods = min(int(monthly_oos_periods), 36)
    b_epochs = min(int(neural_epochs), 50)
    b_m = min(int(m), 100)
    b_steps = min(int(n_steps), 25)
    b_patience = min(int(neural_patience), 15)
    b_hidden = min(int(neural_hidden), 64)
    b_batch = max(int(neural_batch), 64)
elif b_speed_mode == "Fast":
    b_test_periods = min(int(monthly_oos_periods), 60)
    b_epochs = min(int(neural_epochs), 100)
    b_m = min(int(m), 200)
    b_steps = min(int(n_steps), 50)
    b_patience = min(int(neural_patience), 30)
    b_hidden = min(int(neural_hidden), 128)
    b_batch = max(int(neural_batch), 64)
else:
    b_test_periods = int(monthly_oos_periods)
    b_epochs = int(neural_epochs)
    b_m = int(m)
    b_steps = int(n_steps)
    b_patience = int(neural_patience)
    b_hidden = int(neural_hidden)
    b_batch = int(neural_batch)

with _bmode2:
    st.info(
        f"{b_speed_mode}: OOS months≤{b_test_periods}, epochs≤{b_epochs}, "
        f"M={b_m}, reverse steps={b_steps}, hidden≤{b_hidden}, patience={b_patience}. "
        "Turbo/Fast are exploratory; use Research for the final 60–120 month study."
    )

selected_months = {"3M": 3, "6M": 6, "12M": 12}[model_horizon]

if st.button(
    f"Run {model_horizon} Monthly OOS Comparison",
    type="primary",
    key="run_selected_monthly_oos",
    use_container_width=True,
):
    st.session_state["mc_stop_requested"] = False
    try:
        progress = st.progress(0, text="Preparing comparison data...")
        progress.progress(5, text=f"Preparing overlapping {model_horizon} model returns...")
        model_returns = _get_horizon_data(selected_months)

        progress.progress(10, text="Preparing actual monthly OOS returns...")
        monthly_realized = _get_monthly_realized()

        feasible_lookback = min(
            int(lookback),
            max(20, len(model_returns) - 12),
        )

        progress.progress(
            15,
            text=(
                f"Running up to {int(b_test_periods)} rolling OOS months in "
                f"{b_speed_mode} mode. Neural models are retrained in rolling windows..."
            ),
        )

        (
            summary,
            detail,
            wealth,
            diagnostics,
            split_summary,
            regime_summary,
            regime_detail,
        ) = _cached_monthly_oos_comparison(
            model_returns=model_returns,
            monthly_realized_returns=monthly_realized,
            lookback=int(feasible_lookback),
            test_periods=int(b_test_periods),
            gamma=float(gamma),
            rule=rule,
            m=int(b_m),
            beta=float(beta),
            n_steps=int(b_steps),
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
            neural_epochs=int(b_epochs),
            neural_batch=int(b_batch),
            neural_learning_rate=float(neural_lr),
            neural_hidden_dim=int(b_hidden),
            neural_validation_fraction=float(neural_val_fraction),
            neural_patience=int(b_patience),
            neural_min_delta=float(neural_min_delta),
            seed=int(DEFAULT_SEED),
        )

        progress.progress(85, text="Building relative-performance regression dataset...")
        relative_df = build_relative_performance_dataset(detail, regime_detail)

        progress.progress(92, text="Estimating regression specifications...")
        regression_table, regression_diagnostics = run_regime_regressions(relative_df)

        progress.progress(97, text="Computing direct piecewise slope tests...")
        slope_tests = run_piecewise_slope_tests(relative_df)

        progress.progress(100, text="Selected-horizon comparison complete.")

        st.session_state["mc_selected_results"] = {
            "horizon": model_horizon,
            "speed_mode": b_speed_mode,
            "effective_oos_periods": b_test_periods,
            "summary": summary,
            "detail": detail,
            "wealth": wealth,
            "diagnostics": diagnostics,
            "split_summary": split_summary,
            "regime_summary": regime_summary,
            "regime_detail": regime_detail,
            "relative_df": relative_df,
            "regression_table": regression_table,
            "regression_diagnostics": regression_diagnostics,
            "slope_tests": slope_tests,
        }
        progress.empty()

    except Exception as exc:
        st.error(f"Selected-horizon monthly OOS comparison failed: {exc}")

if "mc_selected_results" in st.session_state:
    res = st.session_state["mc_selected_results"]
    st.success(
        f"Displaying saved B result: {res.get('speed_mode', 'legacy')} mode · "
        f"requested/effective OOS cap={res.get('effective_oos_periods', 'legacy')} months."
    )
    st.subheader(f"{res['horizon']} OOS Performance")
    st.dataframe(
        _format_performance(res["summary"]),
        use_container_width=True,
        hide_index=True,
    )

    d = res["diagnostics"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Requested OOS months", d["Requested OOS months"])
    d2.metric("Evaluated OOS months", d["Evaluated OOS months"])
    d3.metric("Theoretical T windows", d["Theoretical T windows"])
    d4.metric(
        "Fallback windows",
        d["Validation fallback windows"] + d["Safe fallback windows"],
    )

    st.subheader("Cumulative OOS Wealth")
    st.line_chart(res["wealth"])

    st.subheader("Performance by T Source")
    st.dataframe(
        _format_performance(res["split_summary"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Market / Estimation Regime Diagnostics")
    _regime_display = res["regime_summary"].rename(columns={
        "Mean-vector norm": "Mean-vector norm ∥μ^diff∥",
        "Covariance trace": "Covariance trace tr(Σ^diff)",
        "Covariance condition number": "Covariance condition number κ(Σ^diff)",
    })
    st.dataframe(
        _regime_display.style.format(
            {
                "Market return / month": "{:.3%}",
                "Average asset mean": "{:.3%}",
                "Cross-sectional dispersion": "{:.3%}",
                "Average asset volatility": "{:.3%}",
                "Mean-vector norm ∥μ^diff∥": "{:.4f}",
                "Covariance trace tr(Σ^diff)": "{:.4f}",
                "Covariance condition number κ(Σ^diff)": "{:.2f}",
                "b*": "{:.4f}",
                "b*/n": "{:.4f}",
                "T": "{:.4f}",
                "Windows": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show rolling OOS detail"):
        st.dataframe(
            res["detail"].style.format(
                {
                    "Realized monthly return": "{:.3%}",
                    "Turnover": "{:.3f}",
                    "T": "{:.4f}",
                    "b*": "{:.4f}",
                    "b*/n": "{:.4f}",
                    "s_T^2": "{:.4f}",
                    "Neural best validation DSM": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------
    # Regression views
    # ------------------------------------------------------------
    st.subheader("Regime-Controlled Relative Performance")

    st.dataframe(
        res["regression_diagnostics"].style.format(
            {"R²": "{:.3f}", "n": "{:.0f}", "k": "{:.0f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------
    # Regression view selector
    # ------------------------------------------------------------

    # Persistent selections.  These values control the table that is displayed.
    if "mc_reg_spec" not in st.session_state:
        st.session_state["mc_reg_spec"] = "Piecewise at 1"

    if "mc_reg_strategy" not in st.session_state:
        st.session_state["mc_reg_strategy"] = "Neural - Historical"

    with st.form("mc_regression_view_form", clear_on_submit=False):
        a, b = st.columns(2)

        with a:
            selected_spec_form = st.radio(
                "Regression specification",
                [
                    "Piecewise at 1",
                    "Continuous b*/n",
                    "Binary feasibility",
                ],
                index=[
                    "Piecewise at 1",
                    "Continuous b*/n",
                    "Binary feasibility",
                ].index(st.session_state["mc_reg_spec"]),
                key="mc_reg_spec_form",
            )

        with b:
            selected_strategy_form = st.radio(
                "Relative strategy",
                [
                    "Neural - Historical",
                    "Gaussian - Historical",
                ],
                index=[
                    "Neural - Historical",
                    "Gaussian - Historical",
                ].index(st.session_state["mc_reg_strategy"]),
                key="mc_reg_strategy_form",
            )

        apply_view = st.form_submit_button(
            "Apply regression view",
            use_container_width=True,
        )

    if apply_view:
        st.session_state["mc_reg_spec"] = selected_spec_form
        st.session_state["mc_reg_strategy"] = selected_strategy_form
        # Force one immediate rerun so the table below uses the newly applied
        # radio selections on the very next render (one click only).
        st.rerun()

    reg_spec = st.session_state["mc_reg_spec"]
    reg_strategy = st.session_state["mc_reg_strategy"]

    reg_view = res["regression_table"][
        (res["regression_table"]["Specification"] == reg_spec)
        & (res["regression_table"]["Relative strategy"] == reg_strategy)
    ].copy()

    st.markdown(f"**{reg_spec} — {reg_strategy}**")
    st.dataframe(
        reg_view[
            ["Variable", "Coefficient", "Robust SE", "t-stat", "p-value"]
        ].style.format(
            {
                "Coefficient": "{:.6f}",
                "Robust SE": "{:.6f}",
                "t-stat": "{:.3f}",
                "p-value": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    slope_view = res["slope_tests"][
        res["slope_tests"]["Relative strategy"] == reg_strategy
    ].copy()

    st.markdown("**Direct piecewise slope tests**")
    st.dataframe(
        slope_view.style.format(
            {
                "Estimate": "{:.6f}",
                "Robust SE": "{:.6f}",
                "t-stat": "{:.3f}",
                "p-value": "{:.4f}",
                "95% CI low": "{:.6f}",
                "95% CI high": "{:.6f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    scatter_data = make_threshold_scatter_data(res["relative_df"])
    scatter_long = pd.concat(
        [
            scatter_data[
                ["Realized date", "b*/n", "T source", "Neural - Historical"]
            ]
            .rename(columns={"Neural - Historical": "Relative return"})
            .assign(Strategy="Neural - Historical"),
            scatter_data[
                ["Realized date", "b*/n", "T source", "Gaussian - Historical"]
            ]
            .rename(columns={"Gaussian - Historical": "Relative return"})
            .assign(Strategy="Gaussian - Historical"),
        ],
        ignore_index=True,
    )

    fig = px.scatter(
        scatter_long,
        x="b*/n",
        y="Relative return",
        color="Strategy",
        symbol="T source",
        hover_data=["Realized date"],
        title="b*/n vs diffusion relative OOS performance",
    )
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        annotation_text="b*/n = 1",
    )
    fig.add_hline(y=0.0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# C. 3M / 6M / 12M Research Study
# ============================================================
st.divider()
st.header("C. 3M / 6M / 12M Research Study")
st.caption(
    "Runs the same monthly OOS framework for all three model horizons. "
    "This is intentionally separated from the Portfolio page because it is computationally expensive."
)

study_mode = st.radio(
    "Horizon-study speed",
    ["Turbo", "Fast", "Research"],
    horizontal=True,
    index=0,
    key="mc_study_mode",
)

if study_mode == "Turbo":
    study_epochs = min(40, int(neural_epochs))
    study_m = min(100, int(m))
    study_steps = min(25, int(n_steps))
    study_patience = min(12, int(neural_patience))
    study_hidden = min(64, int(neural_hidden))
    study_batch = max(64, int(neural_batch))
    study_oos_periods = min(24, int(monthly_oos_periods))
elif study_mode == "Fast":
    study_epochs = min(80, int(neural_epochs))
    study_m = min(150, int(m))
    study_steps = min(40, int(n_steps))
    study_patience = min(25, int(neural_patience))
    study_hidden = min(128, int(neural_hidden))
    study_batch = max(64, int(neural_batch))
    study_oos_periods = min(48, int(monthly_oos_periods))
else:
    study_epochs = int(neural_epochs)
    study_m = int(m)
    study_steps = int(n_steps)
    study_patience = int(neural_patience)
    study_hidden = int(neural_hidden)
    study_batch = int(neural_batch)
    study_oos_periods = int(monthly_oos_periods)

st.caption(
    f"{study_mode} mode: OOS months per horizon≤{study_oos_periods}, "
    f"epochs={study_epochs}, M={study_m}, reverse steps={study_steps}, "
    f"hidden={study_hidden}, batch={study_batch}, patience={study_patience}. "
    "Research preserves the full user-selected settings."
)

if st.button(
    "Run 3M / 6M / 12M Research Study",
    type="primary",
    key="run_multi_horizon_research",
    use_container_width=True,
):
    st.session_state["mc_stop_requested"] = False
    if source != "Yahoo Finance":
        st.error("The automatic 3M/6M/12M study requires Yahoo Finance.")
    else:
        try:
            progress = st.progress(0, text="Preparing multi-horizon research study...")
            monthly_realized = download_yahoo_monthly_returns(ticker_list, start_date)

            all_summary = []
            all_diag = []
            all_split = []
            all_regime = []

            for idx, (label, months) in enumerate([("3M", 3), ("6M", 6), ("12M", 12)]):
                if st.session_state.get("mc_stop_requested", False):
                    st.warning("Research study stopped before the next horizon.")
                    st.stop()
                base_pct = int(idx * 100 / 3)
                progress.progress(
                    base_pct,
                    text=f"Preparing {label} horizon ({idx + 1}/3)...",
                )
                model_returns = download_yahoo_horizon_returns(
                    ticker_list,
                    start_date,
                    months,
                )
                feasible_lookback = min(
                    int(lookback),
                    max(20, len(model_returns) - 12),
                )

                progress.progress(
                    min(base_pct + 5, 99),
                    text=f"Running {label} rolling OOS / neural retraining...",
                )

                (
                    summary_h,
                    detail_h,
                    wealth_h,
                    diagnostics_h,
                    split_h,
                    regime_h,
                    regime_detail_h,
                ) = _cached_monthly_oos_comparison(
                    model_returns=model_returns,
                    monthly_realized_returns=monthly_realized,
                    lookback=int(feasible_lookback),
                    test_periods=int(study_oos_periods),
                    gamma=float(gamma),
                    rule=rule,
                    m=int(study_m),
                    beta=float(beta),
                    n_steps=int(study_steps),
                    constraint_mode=constraint_mode,
                    max_long_weight=float(max_long_weight),
                    max_short_weight=float(max_short_weight),
                    max_gross_exposure=float(max_gross_exposure),
                    neural_epochs=int(study_epochs),
                    neural_batch=int(study_batch),
                    neural_learning_rate=float(neural_lr),
                    neural_hidden_dim=int(study_hidden),
                    neural_validation_fraction=float(neural_val_fraction),
                    neural_patience=int(study_patience),
                    neural_min_delta=float(neural_min_delta),
                    seed=int(DEFAULT_SEED + idx * 10000),
                )

                summary_h = summary_h.copy()
                summary_h.insert(0, "Model horizon", label)
                all_summary.append(summary_h)

                diag_row = {"Model horizon": label, **diagnostics_h}
                all_diag.append(diag_row)

                split_h = split_h.copy()
                split_h.insert(0, "Model horizon", label)
                all_split.append(split_h)

                regime_h = regime_h.copy()
                regime_h.insert(0, "Model horizon", label)
                all_regime.append(regime_h)

                progress.progress(
                    int((idx + 1) * 100 / 3),
                    text=f"Completed {label} horizon.",
                )

            results = pd.concat(all_summary, ignore_index=True)
            diagnostics = pd.DataFrame(all_diag)
            split_results = pd.concat(all_split, ignore_index=True)
            regime_results = pd.concat(all_regime, ignore_index=True)

            st.session_state["mc_multi_results"] = {
                "mode": study_mode,
                "effective_oos_periods": study_oos_periods,
                "results": results,
                "diagnostics": diagnostics,
                "split_results": split_results,
                "regime_results": regime_results,
            }
            progress.empty()

        except Exception as exc:
            st.error(f"3M / 6M / 12M research study failed: {exc}")

if "mc_multi_results" in st.session_state:
    multi = st.session_state["mc_multi_results"]
    st.success(f"Displaying saved {multi['mode']} mode research results.")

    st.subheader("3M / 6M / 12M OOS Performance")
    st.dataframe(
        _format_performance(multi["results"]),
        use_container_width=True,
        hide_index=True,
    )

    # Comparison chart: CER by horizon and estimator.
    chart_df = multi["results"].copy()
    fig = px.bar(
        chart_df,
        x="Model horizon",
        y="CER / month",
        color="Method",
        barmode="group",
        title="Monthly OOS CER by model horizon and method",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("OOS Coverage by Model Horizon")
    st.dataframe(
        multi["diagnostics"],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Performance by T Source")
    st.dataframe(
        _format_performance(multi["split_results"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Regime Diagnostics by T Source")
    _multi_regime_display = multi["regime_results"].rename(columns={
        "Mean-vector norm": "Mean-vector norm ∥μ^diff∥",
        "Covariance trace": "Covariance trace tr(Σ^diff)",
        "Covariance condition number": "Covariance condition number κ(Σ^diff)",
    })
    st.dataframe(
        _multi_regime_display.style.format(
            {
                "Market return / month": "{:.3%}",
                "Average asset mean": "{:.3%}",
                "Cross-sectional dispersion": "{:.3%}",
                "Average asset volatility": "{:.3%}",
                "Mean-vector norm ∥μ^diff∥": "{:.4f}",
                "Covariance trace tr(Σ^diff)": "{:.4f}",
                "Covariance condition number κ(Σ^diff)": "{:.2f}",
                "b*": "{:.4f}",
                "b*/n": "{:.4f}",
                "T": "{:.4f}",
                "Windows": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# D. Current-window estimator / portfolio-rule comparison
# ============================================================
st.divider()
st.header("D. Current-Window Estimator / Portfolio-Rule Comparison")

_dsave1, _dsave2, _dsave3 = st.columns([1, 1, 2])

with _dsave1:
    if st.button(
        "💾 Save Section D",
        key="save_section_d",
        use_container_width=True,
    ):
        if "mc_snapshot" not in st.session_state:
            st.warning("Run Section D first, then click Save Section D.")
        else:
            try:
                st.session_state["section_d_saved_zip"] = _build_section_d_export_zip(
                    st.session_state["mc_snapshot"]
                )
                st.success("Section D results package prepared.")
            except Exception as exc:
                st.error(f"Could not save Section D results: {exc}")

with _dsave2:
    if st.button(
        "Clear Section D",
        key="clear_section_d",
        use_container_width=True,
    ):
        st.session_state.pop("mc_snapshot", None)
        st.session_state.pop("section_d_saved_zip", None)
        st.rerun()

with _dsave3:
    if "section_d_saved_zip" in st.session_state:
        st.download_button(
            "⬇ Download Section D tables + charts (.zip)",
            data=st.session_state["section_d_saved_zip"],
            file_name="section_D_current_window_comparison.zip",
            mime="application/zip",
            use_container_width=True,
        )
st.caption(
    "Section D is tied to the latest completed Portfolio analysis. If a Portfolio result "
    "exists, D uses that exact estimation window, return interval, T, constraints, neural "
    "moments, synthetic returns, and recommended weights. The Method Comparison horizon "
    "selector above does NOT override the Portfolio window for Section D."
)

_shared_d = st.session_state.get("shared_current_window")
if isinstance(_shared_d, dict) and _shared_d:
    _d_interval = _shared_d.get("interval_label", "unknown")
    _d_lookback = _shared_d.get("lookback", "unknown")
    _d_rule = _shared_d.get("portfolio_rule", "unknown")
    _d_score = _shared_d.get("score_model", "unknown")
    st.success(
        f"Section D source: latest Portfolio run · interval={_d_interval} · "
        f"lookback={_d_lookback} · rule={_d_rule} · score={_d_score}."
    )
else:
    st.warning(
        "No completed Portfolio analysis is available in this session. Section D will "
        "run independently using the Method Comparison settings. For exact D3 ↔ Portfolio "
        "matching, run Portfolio first."
    )

st.markdown(
    """
    **Research logic**

    **A.** Is diffusion theoretically feasible? → Return-Horizon Study  
    **B.** What does diffusion do to μ and Σ? → Moment diagnostics  
    **C.** How does that change portfolio weights? → Historical vs Gaussian vs Neural × EW / MV / LW-MV  
    **D.** Does it improve future performance? → Rolling monthly OOS tests above  
    **E.** When does it outperform? → b*/n, regime, and piecewise regressions
    """
)

st.info(
    "The first table below uses each estimator's own μ and Σ to evaluate its portfolio. "
    "The second table fixes the evaluation moments to the same historical μ and Σ for "
    "every portfolio. The second comparison is therefore better for isolating the effect "
    "of the portfolio weights themselves."
)

_d_button_interval = (
    st.session_state.get("shared_current_window", {}).get("interval_label")
    if isinstance(st.session_state.get("shared_current_window"), dict)
    else None
)
_d_button_label = (
    f"Run current Portfolio-window Historical / Gaussian / Neural comparison "
    f"({_d_button_interval})"
    if _d_button_interval
    else f"Run current {model_horizon} Historical / Gaussian / Neural comparison"
)

if st.button(
    _d_button_label,
    key="run_method_snapshot",
    type="primary",
    use_container_width=True,
):
    try:
        progress = st.progress(0, text="Preparing current-window comparison...")

        shared = st.session_state.get("shared_current_window")

        # ------------------------------------------------------------
        # Preferred path: ALWAYS reuse the exact latest Portfolio Neural result.
        #
        # Previous bug:
        #   Section D required Portfolio horizon == the Method Comparison
        #   "Selected model return horizon". A 1M Portfolio result therefore
        #   silently failed synchronization when Method Comparison was set to
        #   12M, and D3 retrained a different Neural model.
        #
        # Section D now represents the CURRENT PORTFOLIO WINDOW, so the horizon
        # selector above is intentionally ignored here.
        # ------------------------------------------------------------
        use_shared = False
        if isinstance(shared, dict):
            shared_estimator = shared.get("estimator")
            shared_score = shared.get("score_model")
            shared_assets = shared.get("asset_names", [])
            required_keys = [
                "window_returns",
                "mu_hist",
                "sigma_hist",
                "mu_used",
                "sigma_used",
                "used_returns",
                "weights",
                "portfolio_rule",
                "gamma",
                "constraint_mode",
                "T",
            ]
            if (
                shared_estimator == "Diffusion"
                and shared_score == "Learned Neural Score"
                and len(shared_assets) > 0
                and all(key in shared for key in required_keys)
            ):
                use_shared = True

        if use_shared:
            progress.progress(
                10,
                text="Reusing the exact Portfolio current-window data and Neural Diffusion result...",
            )

            x_df = shared["window_returns"].copy()
            x = x_df.to_numpy(dtype=float)
            assets = list(x_df.columns)

            mu_h = np.asarray(shared["mu_hist"], dtype=float).copy()
            sig_h = np.asarray(shared["sigma_hist"], dtype=float).copy()

            # Neural moments / synthetic-return pool are taken directly from the
            # Portfolio run. There is NO second neural training here.
            mu_n = np.asarray(shared["mu_used"], dtype=float).copy()
            sig_n = np.asarray(shared["sigma_used"], dtype=float).copy()
            combined_n = np.asarray(shared["used_returns"], dtype=float).copy()
            neural_training_result = shared.get("neural_training_result")

            T_used = float(shared["T"])
            T_theory = float(shared["T"])
            b_star = float(shared.get("b_star", np.nan))
            signal = float(shared.get("signal_level", np.nan))
            T_source = "Exact Portfolio run"

            # For a fair D comparison, all portfolio rules use the exact same
            # Portfolio constraint/risk settings.
            d_gamma = float(shared["gamma"])
            d_constraint_mode = str(shared["constraint_mode"])
            d_max_long_weight = float(shared["max_long_weight"])
            d_max_short_weight = float(shared["max_short_weight"])
            d_max_gross_exposure = float(shared["max_gross_exposure"])
            d_m = int(shared["m"])
            d_beta = float(shared["beta"])
            d_n_steps = int(shared["n_steps"])
            shared_rule = str(shared["portfolio_rule"])
            shared_weights = np.asarray(shared["weights"], dtype=float).copy()

            progress.progress(
                30,
                text=f"Generating Gaussian Diffusion on the same window and T={T_used:.4f}...",
            )
            mu_g, sig_g, _, combined_g = _cached_current_gaussian(
                x,
                m=d_m,
                horizon=T_used,
                beta=d_beta,
                n_steps=d_n_steps,
                seed=int(DEFAULT_SEED),
            )

            data_source_note = (
                "Exact latest Portfolio window reused. Method Comparison's horizon selector "
                "was ignored for Section D. Neural Diffusion was NOT retrained."
            )

        # ------------------------------------------------------------
        # Fallback path: no compatible Portfolio result in session.
        # ------------------------------------------------------------
        else:
            progress.progress(
                10,
                text="No compatible Portfolio result found; building the comparison from Method Comparison settings...",
            )

            model_returns = _get_horizon_data(selected_months)
            lb = min(int(lookback), len(model_returns))
            x_df = model_returns.iloc[-lb:].copy()
            x = x_df.to_numpy(dtype=float)
            assets = list(x_df.columns)

            mu_h, sig_h = sample_moments(x, mle=True)

            T_theory, b_star, signal = theoretical_horizon(
                mu_h,
                sig_h,
                n_obs=len(x),
                beta=float(beta),
            )

            if T_theory > 0:
                T_used = float(T_theory)
                T_source = "Theoretical"
            else:
                T_used = 0.25
                T_source = "Safe fallback (theoretical T=0)"

            d_gamma = float(gamma)
            d_constraint_mode = constraint_mode
            d_max_long_weight = float(max_long_weight)
            d_max_short_weight = float(max_short_weight)
            d_max_gross_exposure = float(max_gross_exposure)
            d_m = int(m)
            d_beta = float(beta)
            d_n_steps = int(n_steps)
            shared_rule = None
            shared_weights = None

            progress.progress(
                25,
                text=f"Generating Gaussian Diffusion moments at T={T_used:.4f}...",
            )
            mu_g, sig_g, _, combined_g = _cached_current_gaussian(
                x,
                m=d_m,
                horizon=T_used,
                beta=d_beta,
                n_steps=d_n_steps,
                seed=int(DEFAULT_SEED),
            )

            progress.progress(
                45,
                text=(
                    f"Training Neural Diffusion score at T={T_used:.4f}. "
                    f"epochs≤{int(neural_epochs)}, hidden={int(neural_hidden)}..."
                ),
            )
            (
                mu_n,
                sig_n,
                _,
                combined_n,
                neural_training_result,
            ) = _cached_current_neural(
                x,
                m=d_m,
                horizon=T_used,
                beta=d_beta,
                n_steps=d_n_steps,
                epochs=int(neural_epochs),
                batch_size=int(neural_batch),
                learning_rate=float(neural_lr),
                hidden_dim=int(neural_hidden),
                validation_fraction=float(neural_val_fraction),
                patience=int(neural_patience),
                min_delta=float(neural_min_delta),
                seed=int(DEFAULT_SEED),
            )

            data_source_note = (
                "Independent Method Comparison run because no compatible completed Neural "
                "Diffusion Portfolio result exists. Run Portfolio first, then rerun Section D."
            )

        progress.progress(
            75,
            text="Constructing portfolios for all estimator/rule combinations...",
        )

        estimator_specs = [
            ("Historical", mu_h, sig_h, x),
            ("Gaussian Diffusion", mu_g, sig_g, combined_g),
            ("Neural Diffusion", mu_n, sig_n, combined_n),
        ]

        rows = []
        benchmark_rows = []
        weight_rows = []

        for estimator_name, mu, sig, estimator_returns in estimator_specs:
            for rule_name in SUPPORTED_RULES:
                # If this is the exact estimator + rule used on Portfolio, copy the
                # Portfolio weight vector verbatim. This guarantees D3 and Recommended
                # Weights are mathematically identical rather than merely close.
                if (
                    use_shared
                    and estimator_name == "Neural Diffusion"
                    and rule_name == shared_rule
                    and shared_weights is not None
                ):
                    w = shared_weights.copy()
                else:
                    w = compute_weights(
                        rule_name,
                        mu,
                        sig,
                        gamma=d_gamma,
                        returns=estimator_returns,
                        constraint_mode=d_constraint_mode,
                        max_long_weight=d_max_long_weight,
                        max_short_weight=d_max_short_weight,
                        max_gross_exposure=d_max_gross_exposure,
                    )
                    w = np.asarray(w, dtype=float).reshape(-1)

                rows.append(
                    {
                        "Estimator": estimator_name,
                        "Portfolio rule": rule_name,
                        "Expected return": portfolio_mean(w, mu),
                        "Volatility": portfolio_volatility(w, sig),
                        "Sharpe": sharpe_ratio(w, mu, sig),
                        "CER": certainty_equivalent(w, mu, sig, d_gamma),
                        "Gross exposure": gross_exposure(w),
                    }
                )

                benchmark_rows.append(
                    {
                        "Estimator": estimator_name,
                        "Portfolio rule": rule_name,
                        "Benchmark expected return": portfolio_mean(w, mu_h),
                        "Benchmark volatility": portfolio_volatility(w, sig_h),
                        "Benchmark Sharpe": sharpe_ratio(w, mu_h, sig_h),
                        "Benchmark CER": certainty_equivalent(w, mu_h, sig_h, d_gamma),
                        "Gross exposure": gross_exposure(w),
                    }
                )

                for asset, weight in zip(assets, w):
                    weight_rows.append(
                        {
                            "Estimator": estimator_name,
                            "Portfolio rule": rule_name,
                            "Asset": asset,
                            "Weight": float(weight),
                        }
                    )

        snapshot = pd.DataFrame(rows)
        benchmark = pd.DataFrame(benchmark_rows)
        weights_long = pd.DataFrame(weight_rows)

        moment_compare = pd.DataFrame(
            {
                "Metric": [
                    "Mean-vector norm",
                    "Covariance trace",
                    "Covariance condition number",
                ],
                "Historical": [
                    np.linalg.norm(mu_h),
                    np.trace(sig_h),
                    covariance_condition_number(sig_h),
                ],
                "Gaussian Diffusion": [
                    np.linalg.norm(mu_g),
                    np.trace(sig_g),
                    covariance_condition_number(sig_g),
                ],
                "Neural Diffusion": [
                    np.linalg.norm(mu_n),
                    np.trace(sig_n),
                    covariance_condition_number(sig_n),
                ],
            }
        )

        if neural_training_result is not None:
            actual_epochs = len(neural_training_result.train_losses)
            best_val = (
                min(neural_training_result.val_losses)
                if len(neural_training_result.val_losses)
                else np.nan
            )
            neural_training = pd.DataFrame(
                {
                    "Metric": [
                        "Epochs run",
                        "Best epoch",
                        "Best validation DSM",
                        "Validation zero-score baseline",
                        "Training device",
                    ],
                    "Value": [
                        actual_epochs,
                        neural_training_result.best_epoch,
                        best_val,
                        neural_training_result.zero_score_baseline_val,
                        neural_training_result.device,
                    ],
                }
            )
        else:
            neural_training = pd.DataFrame()

        # Exact-match audit for the Portfolio rule.
        sync_audit = None
        if use_shared and shared_weights is not None:
            d3_same_rule = (
                weights_long[
                    (weights_long["Estimator"] == "Neural Diffusion")
                    & (weights_long["Portfolio rule"] == shared_rule)
                ]
                .set_index("Asset")
                .reindex(assets)["Weight"]
                .to_numpy(dtype=float)
            )
            max_abs_diff = float(np.max(np.abs(d3_same_rule - shared_weights)))
            sync_audit = pd.DataFrame(
                {
                    "Check": [
                        "Portfolio rule reused",
                        "Maximum absolute weight difference",
                        "Exact synchronization",
                    ],
                    "Value": [
                        shared_rule,
                        max_abs_diff,
                        bool(max_abs_diff < 1e-12),
                    ],
                }
            )

        st.session_state.pop("section_d_saved_zip", None)
        st.session_state["mc_snapshot"] = {
            "portfolio_revision": (
                int(shared.get("portfolio_revision", 0))
                if use_shared and isinstance(shared, dict)
                else None
            ),
            "snapshot": snapshot,
            "benchmark": benchmark,
            "weights": weights_long,
            "moments": moment_compare,
            "neural_training": neural_training,
            "sync_audit": sync_audit,
            "data_source_note": data_source_note,
            "used_shared_portfolio": use_shared,
            "T": T_used,
            "T_theory": T_theory,
            "T_source": T_source,
            "b_star": b_star,
            "signal": signal,
            "effective_gamma": d_gamma,
            "effective_constraint_mode": d_constraint_mode,
            "effective_max_long_weight": d_max_long_weight,
            "effective_max_short_weight": d_max_short_weight,
            "effective_max_gross_exposure": d_max_gross_exposure,
        }

        progress.progress(100, text="Current-window comparison complete.")
        progress.empty()

    except Exception as exc:
        st.error(f"Current-window method comparison failed: {exc}")

if "mc_snapshot" in st.session_state:
    snap = st.session_state["mc_snapshot"]
    st.caption(
        "Saved Section D result is stored in session state and remains visible when you "
        "navigate away and return. It changes only when Section D is rerun or cleared."
    )

    _current_shared = st.session_state.get("shared_current_window")
    _current_revision = (
        int(_current_shared.get("portfolio_revision", 0))
        if isinstance(_current_shared, dict)
        else None
    )
    _snapshot_revision = snap.get("portfolio_revision")

    # Never show a Section-D table built from an older Portfolio run.
    if (
        snap.get("used_shared_portfolio", False)
        and _current_revision is not None
        and _snapshot_revision is not None
        and _snapshot_revision != _current_revision
    ):
        st.warning(
            "Section D was built from an older Portfolio revision. The saved D1-D4 "
            "tables remain visible for reference, but rerun Section D before interpreting "
            "them as the current Portfolio comparison."
        )

    if snap.get("used_shared_portfolio", False):
        st.success(
            "Synchronized with the latest Portfolio run. Neural Diffusion moments, "
            "synthetic-return pool, T, constraints, γ, and the Portfolio rule's exact "
            f"weight vector were reused. Portfolio revision={_snapshot_revision}."
        )
    else:
        st.warning(snap.get("data_source_note", "Independent comparison run."))

    if isinstance(snap.get("sync_audit"), pd.DataFrame):
        st.markdown("**Portfolio ↔ D3 synchronization audit**")
        st.dataframe(
            snap["sync_audit"],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("D1. Estimator-Based Portfolio Metrics")
    st.caption(
        "Each portfolio is evaluated under the same estimator that generated its μ and Σ. "
        "This shows the estimator's own implied risk/return picture."
    )
    st.dataframe(
        snap["snapshot"].style.format(
            {
                "Expected return": "{:.3%}",
                "Volatility": "{:.3%}",
                "Sharpe": "{:.3f}",
                "CER": "{:.3%}",
                "Gross exposure": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_own = px.bar(
        snap["snapshot"],
        x="Portfolio rule",
        y="CER",
        color="Estimator",
        barmode="group",
        title="Estimator-Based CER: Historical vs Gaussian vs Neural",
    )
    st.plotly_chart(fig_own, use_container_width=True)

    st.subheader("D2. Common Historical-Benchmark Evaluation")
    st.caption(
        "All portfolios keep their own weights, but expected return, volatility, Sharpe, "
        "and CER are recalculated using the same historical sample μ and Σ. "
        "This removes the problem of comparing portfolios under different estimated distributions."
    )
    st.dataframe(
        snap["benchmark"].style.format(
            {
                "Benchmark expected return": "{:.3%}",
                "Benchmark volatility": "{:.3%}",
                "Benchmark Sharpe": "{:.3f}",
                "Benchmark CER": "{:.3%}",
                "Gross exposure": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_bench = px.bar(
        snap["benchmark"],
        x="Portfolio rule",
        y="Benchmark CER",
        color="Estimator",
        barmode="group",
        title="Common Historical-Benchmark CER",
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    st.subheader("D3. Portfolio Weight Comparison")

    st.info(
        "Neural Diffusion in D3 uses the same estimator as Portfolio: the neural score "
        "is trained at the selected diffusion horizon T, reverse diffusion generates M "
        "synthetic returns, real and synthetic returns are combined, μ and Σ are estimated "
        "from that combined sample, and Mean-Variance weights are computed from those moments. "
        "When Section D is synchronized, the Neural-Diffusion row for the Portfolio-selected "
        "rule is copied directly from Portfolio Recommended Weights."
    )

    if snap.get("used_shared_portfolio", False):
        _audit = snap.get("sync_audit")
        if isinstance(_audit, pd.DataFrame) and not _audit.empty:
            _exact_row = _audit.loc[
                _audit["Check"] == "Exact synchronization", "Value"
            ]
            _exact = bool(_exact_row.iloc[0]) if len(_exact_row) else False
            if _exact:
                st.success(
                    "D3 Neural Diffusion for the Portfolio-selected rule is an exact copy "
                    "of Recommended Weights (maximum absolute difference < 1e-12)."
                )
            else:
                st.error(
                    "Synchronization audit failed. D3 and Portfolio weights differ; "
                    "do not interpret this table until the mismatch is resolved."
                )

    st.caption(
        "These are the actual weights produced by each estimator × portfolio-rule combination. "
        "When synchronized, the Neural Diffusion row for the rule selected on Portfolio is "
        "copied directly from Portfolio's Recommended Weights."
    )
    weights_pivot = snap["weights"].pivot_table(
        index=["Estimator", "Portfolio rule"],
        columns="Asset",
        values="Weight",
        aggfunc="first",
    ).reset_index()
    weight_format = {
        c: "{:.2%}"
        for c in weights_pivot.columns
        if c not in ["Estimator", "Portfolio rule"]
    }
    st.dataframe(
        weights_pivot.style.format(weight_format),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("D4. Historical vs Gaussian vs Neural Moment Diagnostics")

    _mom = snap["moments"].copy()

    # Preserve the numerical results internally, but display estimator-specific
    # mathematical notation so Historical, Gaussian, and Neural quantities are
    # not all mislabeled as "diff".
    _h_mean = float(_mom.loc[_mom["Metric"].str.contains("Mean-vector", regex=False), "Historical"].iloc[0])
    _g_mean = float(_mom.loc[_mom["Metric"].str.contains("Mean-vector", regex=False), "Gaussian Diffusion"].iloc[0])
    _n_mean = float(_mom.loc[_mom["Metric"].str.contains("Mean-vector", regex=False), "Neural Diffusion"].iloc[0])

    _h_trace = float(_mom.loc[_mom["Metric"].str.contains("Covariance trace", regex=False), "Historical"].iloc[0])
    _g_trace = float(_mom.loc[_mom["Metric"].str.contains("Covariance trace", regex=False), "Gaussian Diffusion"].iloc[0])
    _n_trace = float(_mom.loc[_mom["Metric"].str.contains("Covariance trace", regex=False), "Neural Diffusion"].iloc[0])

    _h_cond = float(_mom.loc[_mom["Metric"].str.contains("condition number", case=False, regex=False), "Historical"].iloc[0])
    _g_cond = float(_mom.loc[_mom["Metric"].str.contains("condition number", case=False, regex=False), "Gaussian Diffusion"].iloc[0])
    _n_cond = float(_mom.loc[_mom["Metric"].str.contains("condition number", case=False, regex=False), "Neural Diffusion"].iloc[0])

    d4_display = pd.DataFrame(
        {
            "Metric": [
                "Mean-vector norm",
                "Covariance trace",
                "Covariance condition number",
            ],
            "Historical": [
                f"‖μ̂ᴴ‖ = {_h_mean:.4f}",
                f"tr(Σ̂ᴴ) = {_h_trace:.4f}",
                f"κ(Σ̂ᴴ) = {_h_cond:.4f}",
            ],
            "Gaussian Diffusion": [
                f"‖μ̂ᴳ‖ = {_g_mean:.4f}",
                f"tr(Σ̂ᴳ) = {_g_trace:.4f}",
                f"κ(Σ̂ᴳ) = {_g_cond:.4f}",
            ],
            "Neural Diffusion": [
                f"‖μ̂ᴺ‖ = {_n_mean:.4f}",
                f"tr(Σ̂ᴺ) = {_n_trace:.4f}",
                f"κ(Σ̂ᴺ) = {_n_cond:.4f}",
            ],
        }
    )

    st.dataframe(
        d4_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "H = Historical, G = Gaussian Diffusion, N = Neural Diffusion. "
        "The underlying numerical moment table is unchanged; only the display notation is improved."
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("b*", f"{snap['b_star']:.4f}")
    s2.metric("b*/n signal", f"{snap['signal']:.4f}")
    s3.metric("Theoretical T", f"{snap['T_theory']:.4f}")
    s4.metric("T used", f"{snap['T']:.4f}")

    st.caption(f"T source: **{snap['T_source']}**")

    if isinstance(snap.get("neural_training"), pd.DataFrame) and not snap["neural_training"].empty:
        with st.expander("Neural Diffusion training diagnostics"):
            st.dataframe(
                snap["neural_training"],
                use_container_width=True,
                hide_index=True,
            )


st.divider()
st.caption(
    "Research/educational prototype. OOS results are more informative than same-window "
    "estimated metrics. Use identical constraints and OOS dates when comparing estimators."
)