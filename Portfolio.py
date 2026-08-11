from __future__ import annotations

import hashlib

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from config import (
    CONSTRAINT_MODES,
    DEFAULT_BETA,
    DEFAULT_CONSTRAINT_MODE,
    DEFAULT_GAMMA,
    DEFAULT_LOOKBACK,
    DEFAULT_M,
    DEFAULT_MAX_GROSS_EXPOSURE,
    DEFAULT_MAX_LONG_WEIGHT,
    DEFAULT_MAX_SHORT_WEIGHT,
    DEFAULT_SEED,
    SUPPORTED_RULES,
)
from core.data import (
    download_yahoo_returns,
    load_returns_csv,
)
from core.diffusion import diffusion_augmented_moments
from core.neural_score import neural_diffusion_augmented_moments
from core.regime_regression import (
    build_relative_performance_dataset,
    run_regime_regressions,
    run_piecewise_slope_tests,
    make_threshold_scatter_data,
)


@st.cache_data(show_spinner=False)
def _cached_single_horizon_oos(
    horizon_label, model_returns, monthly_realized_returns, lookback, test_periods,
    gamma, rule, m, beta, n_steps, constraint_mode, max_long_weight,
    max_short_weight, max_gross_exposure, neural_epochs, neural_batch,
    neural_learning_rate, neural_hidden_dim, neural_validation_fraction,
    neural_patience, neural_min_delta, seed,
):
    summary, detail, wealth, diagnostics, split_summary, regime_summary, regime_detail = monthly_rebalance_oos_comparison(
        model_returns=model_returns, monthly_realized_returns=monthly_realized_returns,
        lookback=int(lookback), test_periods=int(test_periods), gamma=float(gamma),
        rule=rule, m=int(m), beta=float(beta), n_steps=int(n_steps),
        constraint_mode=constraint_mode, max_long_weight=float(max_long_weight),
        max_short_weight=float(max_short_weight), max_gross_exposure=float(max_gross_exposure),
        neural_epochs=int(neural_epochs), neural_batch=int(neural_batch),
        neural_learning_rate=float(neural_learning_rate), neural_hidden_dim=int(neural_hidden_dim),
        neural_validation_fraction=float(neural_validation_fraction),
        neural_patience=int(neural_patience), neural_min_delta=float(neural_min_delta),
        seed=int(seed),
    )
    summary = summary.copy()
    summary.insert(0, "Model horizon", horizon_label)
    diagnostics = dict(diagnostics)
    diagnostics["Model horizon"] = horizon_label
    return summary, diagnostics, split_summary, regime_summary
from core.metrics import (
    certainty_equivalent,
    gross_exposure,
    net_exposure,
    portfolio_mean,
    portfolio_volatility,
    sharpe_ratio,
)
from core.moments import covariance_condition_number, sample_moments
from core.portfolio_rules import compute_weights
from core.tuning import (
    theoretical_horizon,
    validation_tuned_horizon,
)


@st.cache_data(show_spinner=False)
def _cached_gaussian_diffusion(
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
def _cached_neural_diffusion(
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


def _build_portfolio_export_zip(
    *,
    weights_df,
    portfolio_metrics_df,
    moment_df,
    tuning_df,
    validation_df=None,
    loss_df=None,
    baseline_df=None,
):
    """Build an in-memory ZIP containing all currently available tables/charts."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        tables = {
            "recommended_weights.csv": weights_df,
            "portfolio_metrics.csv": portfolio_metrics_df,
            "historical_vs_diffusion_moments.csv": moment_df,
            "diffusion_tuning.csv": tuning_df,
        }
        if validation_df is not None:
            tables["validation_horizon_results.csv"] = validation_df
        if loss_df is not None:
            tables["neural_training_losses.csv"] = loss_df
        if baseline_df is not None:
            tables["neural_score_baselines.csv"] = baseline_df

        for name, df in tables.items():
            if df is not None:
                zf.writestr(name, df.to_csv(index=False))

        # Save a self-contained HTML chart when neural loss history is available.
        if loss_df is not None and len(loss_df):
            fig = px.line(
                loss_df,
                x="epoch",
                y=["Training DSM", "Validation DSM"],
                title="Neural Score Training Loss",
            )
            zf.writestr(
                "neural_training_loss_chart.html",
                fig.to_html(full_html=True, include_plotlyjs="cdn"),
            )

        zf.writestr(
            "README.txt",
            "Portfolio App saved-results package.\n"
            "CSV files contain the displayed tables. "
            "HTML chart files can be opened in a web browser.\n",
        )

    return buffer.getvalue()


st.title("Portfolio Builder")
st.caption(
    "Research prototype for comparing historical and diffusion-augmented "
    "portfolio estimates. Practical constraints are enabled by default."
)

# ============================================================
# Data source
# ============================================================
# Streamlit removes widget state for widgets that are not rendered on a rerun.
# Because this app uses a manual section router, Portfolio widgets disappear while
# the user is on Method Comparison / Backtest / Diagnostics.  We therefore restore
# Portfolio controls from the last completed Portfolio analysis.
_saved_portfolio = st.session_state.get("shared_current_window")
if not isinstance(_saved_portfolio, dict):
    _saved_portfolio = {}
if _saved_portfolio.get("upload_fingerprint"):
    st.session_state.setdefault("portfolio_upload_fingerprint", _saved_portfolio["upload_fingerprint"])

_saved_signature = _saved_portfolio.get("returns_signature")
if not isinstance(_saved_signature, dict):
    _saved_signature = st.session_state.get("returns_signature")
if not isinstance(_saved_signature, dict):
    _saved_signature = {}

_saved_source = _saved_portfolio.get(
    "source",
    st.session_state.get("returns_source", "Upload CSV"),
)
_source_options = ["Upload CSV", "Yahoo Finance"]
_source_index = (
    _source_options.index(_saved_source)
    if _saved_source in _source_options
    else 0
)

source = st.radio(
    "Data source",
    _source_options,
    horizontal=True,
    index=_source_index,
)
returns = None

# If a completed Portfolio result exists, its exact cleaned return dataset is the
# source of truth when returning from another app section. This prevents a 120-month
# Portfolio run from being reconstructed from a stale/short 22-row dataset.
# A saved Portfolio snapshot is the durable source of truth.
# Do NOT depend on analysis_has_run here: that flag can be lost/reset during
# navigation, while the saved non-widget snapshot remains available.
_restore_completed_portfolio = bool(
    isinstance(_saved_portfolio, dict)
    and isinstance(_saved_portfolio.get("full_returns"), pd.DataFrame)
    and not _saved_portfolio.get("full_returns").empty
)

if _restore_completed_portfolio:
    # Re-arm the display gate when returning to Portfolio.
    st.session_state["analysis_has_run"] = True

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload return CSV", type=["csv"])
    if uploaded is not None:
        try:
            # Streamlit keeps file_uploader populated after navigation. Do not
            # mistake the same persistent file for a new upload.
            _uploaded_bytes = uploaded.getvalue()
            _upload_fingerprint = hashlib.sha256(_uploaded_bytes).hexdigest()
            _previous_upload_fingerprint = st.session_state.get("portfolio_upload_fingerprint")
            if _restore_completed_portfolio and _previous_upload_fingerprint == _upload_fingerprint:
                returns = _saved_portfolio["full_returns"].copy()
                st.session_state["returns"] = returns.copy()
                st.session_state["returns_source"] = "Upload CSV"
                st.session_state["analysis_has_run"] = True
            else:
                uploaded.seek(0)
                returns = load_returns_csv(uploaded)
                st.session_state["returns"] = returns
                st.session_state["returns_source"] = "Upload CSV"
                st.session_state.pop("returns_signature", None)
                st.session_state["portfolio_upload_fingerprint"] = _upload_fingerprint
                st.session_state["analysis_has_run"] = False
                st.session_state.pop("shared_current_window", None)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            st.stop()
else:
    _saved_tickers = _saved_signature.get(
        "tickers",
        tuple(st.session_state.get("ticker_list", ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])),
    )
    _saved_tickers_text = ",".join(_saved_tickers) if _saved_tickers else "AAPL,MSFT,NVDA,GOOGL,AMZN"
    _saved_start = str(
        _saved_signature.get(
            "start",
            st.session_state.get("start_date", "2021-01-01"),
        )
    )
    _interval_options = [
        "1 day",
        "1 week",
        "1 month",
        "3 months",
        "6 months",
        "1 year",
    ]
    _saved_interval = _saved_signature.get(
        "interval_label",
        _saved_portfolio.get("interval_label", "1 month"),
    )
    _interval_index = (
        _interval_options.index(_saved_interval)
        if _saved_interval in _interval_options
        else 2
    )

    tickers_text = st.text_input("Tickers", _saved_tickers_text)
    start = st.text_input("Start date", _saved_start)
    interval_label = st.selectbox(
        "Return interval",
        _interval_options,
        index=_interval_index,
        help=(
            "3-month, 6-month, and 1-year returns are built from monthly Yahoo prices "
            "using non-overlapping periods."
        ),
    )

    interval_map = {
        "1 day": ("1d", None),
        "1 week": ("1wk", None),
        "1 month": ("1mo", None),
        "3 months": ("1mo", 3),
        "6 months": ("1mo", 6),
        "1 year": ("1mo", 12),
    }

    # Approximate number of non-overlapping return observations per year.
    # We use these values to scale the lookback and diffusion minimum by
    # calendar history rather than imposing the same raw n at every interval.
    periods_per_year_map = {
        "1 day": 252,
        "1 week": 52,
        "1 month": 12,
        "3 months": 4,
        "6 months": 2,
        "1 year": 1,
    }

    yahoo_interval, horizon_months = interval_map[interval_label]
    periods_per_year = periods_per_year_map[interval_label]

    if st.button("Download data"):
        tickers = [x.strip() for x in tickers_text.split(",") if x.strip()]
        st.session_state["ticker_list"] = tickers
        st.session_state["start_date"] = start
        if not tickers:
            st.error("Please enter at least one ticker.")
            st.stop()

        try:
            with st.spinner("Downloading market data..."):
                returns = download_yahoo_returns(
                    tickers,
                    start=start,
                    interval=yahoo_interval,
                    return_horizon_months=horizon_months,
                )
            st.session_state["returns"] = returns
            st.session_state["returns_source"] = "Yahoo Finance"
            st.session_state["analysis_has_run"] = False
            st.session_state.pop("shared_current_window", None)
            st.session_state["returns_signature"] = {
                "tickers": tuple(t.upper() for t in tickers),
                "start": str(start),
                "interval_label": str(interval_label),
                "yahoo_interval": str(yahoo_interval),
                "horizon_months": horizon_months,
            }
        except Exception as exc:
            st.error(f"Could not download market data: {exc}")
            st.stop()

if returns is None:
    if _restore_completed_portfolio:
        # Exact dataset used by the last completed Portfolio analysis.
        returns = _saved_portfolio["full_returns"].copy()
        st.session_state["returns"] = returns.copy()
        st.session_state["returns_source"] = _saved_portfolio.get("source", source)
        if isinstance(_saved_portfolio.get("returns_signature"), dict):
            st.session_state["returns_signature"] = dict(
                _saved_portfolio["returns_signature"]
            )

        # Restore the exact data-source metadata too. The visible widgets above
        # are informational defaults on return; the completed analysis remains
        # tied to this saved dataset until the user explicitly downloads/uploads
        # new data, changes a setting and reruns, or presses Clear.
        source = _saved_portfolio.get("source", source)

        if source == "Yahoo Finance":
            interval_label = _saved_portfolio.get("interval_label", interval_label)
            periods_per_year = int(_saved_portfolio.get("periods_per_year", periods_per_year))
            horizon_months = _saved_portfolio.get("horizon_months", horizon_months)

        st.success(
            f"Restored completed Portfolio analysis: "
            f"{len(returns)} total {interval_label} return observations, "
            f"saved lookback={_saved_portfolio.get('lookback', 'unknown')}."
        )

    elif source == "Yahoo Finance":
        current_tickers = tuple(
            x.strip().upper() for x in tickers_text.split(",") if x.strip()
        )
        current_signature = {
            "tickers": current_tickers,
            "start": str(start),
            "interval_label": str(interval_label),
            "yahoo_interval": str(yahoo_interval),
            "horizon_months": horizon_months,
        }
        saved_signature = st.session_state.get("returns_signature")

        if (
            st.session_state.get("returns_source") == "Yahoo Finance"
            and saved_signature == current_signature
        ):
            returns = st.session_state.get("returns")
        elif st.session_state.get("returns") is not None:
            st.warning(
                "The data settings changed since the last download. "
                "The previous return dataset will not be reused because its interval/date/"
                "ticker settings do not match the current controls."
            )
            st.info(
                f"Click **Download data** to build fresh **{interval_label}** returns "
                "before running the portfolio."
            )
            st.stop()
    else:
        returns = st.session_state.get("returns")

if returns is None:
    st.info("Upload a return CSV or download market data to begin.")
    st.stop()

# For uploaded CSV files the app cannot infer the sampling interval reliably.
# Use monthly-style defaults unless the user specifies otherwise.
if source == "Upload CSV":
    _csv_interval_options = ["1 day", "1 week", "1 month", "3 months", "6 months", "1 year"]
    _saved_csv_interval = _saved_portfolio.get("interval_label", "1 month")
    _saved_csv_index = (
        _csv_interval_options.index(_saved_csv_interval)
        if _saved_csv_interval in _csv_interval_options
        else 2
    )
    csv_interval_label = st.selectbox(
        "CSV return interval",
        _csv_interval_options,
        index=_saved_csv_index,
    )
    periods_per_year_map = {
        "1 day": 252,
        "1 week": 52,
        "1 month": 12,
        "3 months": 4,
        "6 months": 2,
        "1 year": 1,
    }
    interval_label = csv_interval_label
    periods_per_year = periods_per_year_map[csv_interval_label]

if not isinstance(returns, pd.DataFrame) or returns.empty:
    st.error("The return dataset is empty or invalid.")
    st.stop()

returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

n_obs = len(returns)
n_assets = returns.shape[1]

if n_obs < 3:
    st.error(
        "Not enough return observations. Please use a longer date range "
        "or a higher-frequency interval."
    )
    st.stop()

# ============================================================
# Data preview and estimation-risk warning
# ============================================================
st.subheader("Return Data")
c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{n_obs}")
c2.metric("Assets", f"{n_assets}")
c3.metric("N / n", f"{n_assets / n_obs:.3f}")

st.dataframe(returns.tail(), use_container_width=True)

if source == "Yahoo Finance":
    st.caption(f"Selected return interval: {interval_label}")

# Scale sample guidance with the selected return interval.
# Five years is the baseline minimum calendar history and ten years is preferred.
calendar_min_obs = int(5 * periods_per_year)
calendar_preferred_obs = int(10 * periods_per_year)

# Also require enough observations relative to dimension.
min_diffusion_obs = max(calendar_min_obs, 3 * n_assets)
preferred_obs = max(calendar_preferred_obs, 10 * n_assets)

if n_obs < min_diffusion_obs:
    st.warning(
        f"Estimation risk is high for the selected {interval_label} return interval: "
        f"{n_obs} observations for {n_assets} assets. "
        f"Suggested diffusion minimum: {min_diffusion_obs} observations "
        f"(about 5 years of {interval_label} returns, subject to at least 3N observations)."
    )
else:
    st.success(
        f"Sample size meets the interval-adjusted diffusion minimum: "
        f"n={n_obs} ≥ {min_diffusion_obs}."
    )

st.caption(
    f"For {n_assets} assets at the {interval_label} interval: "
    f"minimum n ≈ {min_diffusion_obs}; preferred n ≈ {preferred_obs}."
)

# ============================================================
# Portfolio controls
# ============================================================
st.subheader("Portfolio Settings")

# Interval-adjusted lookback.
# IMPORTANT: each return interval has a separate widget key. This prevents a
# 1-year setting such as lookback=21 from being carried into the 1-month model.
basic_min_lookback = max(2, min(n_obs - 1, n_assets + 2))
max_lookback = n_obs - 1

if max_lookback < 2:
    st.error("There are not enough observations to estimate a portfolio.")
    st.stop()

# Preferred default: interval-adjusted preferred sample size, subject to data
# availability. For monthly data this targets 120 observations; if only e.g. 66
# are available, it uses 66 rather than carrying over a previous annual setting.
_preferred_default_lookback = min(
    max_lookback,
    max(basic_min_lookback, preferred_obs),
)

_saved_lookback = _saved_portfolio.get("lookback")
_saved_interval_label = _saved_portfolio.get("interval_label")
if (
    isinstance(_saved_lookback, (int, np.integer))
    and _saved_interval_label == interval_label
    and basic_min_lookback <= int(_saved_lookback) <= max_lookback
):
    default_lookback = int(_saved_lookback)
else:
    default_lookback = int(_preferred_default_lookback)

lookback_key = (
    f"portfolio_lookback__{interval_label.replace(' ', '_')}__N{n_assets}"
)

if (
    _restore_completed_portfolio
    and isinstance(_saved_lookback, (int, np.integer))
    and basic_min_lookback <= int(_saved_lookback) <= max_lookback
):
    # Exact completed-analysis lookback. We render it as disabled so navigating
    # away and back cannot silently mutate the saved analysis window.
    lookback = int(_saved_lookback)
    st.slider(
        "Lookback observations",
        min_value=int(basic_min_lookback),
        max_value=int(max_lookback),
        value=int(lookback),
        step=1,
        disabled=True,
        key=f"{lookback_key}__restored",
    )
    st.caption(
        "Restored lookback from the completed Portfolio run. "
        "Click Clear or download/upload new data to create a different run."
    )
elif basic_min_lookback >= max_lookback:
    lookback = max_lookback
    st.info(f"Using lookback = {lookback}.")
else:
    lookback = st.slider(
        "Lookback observations",
        min_value=int(basic_min_lookback),
        max_value=int(max_lookback),
        value=int(default_lookback),
        step=1,
        key=lookback_key,
    )

approx_years = lookback / float(periods_per_year)

l1, l2, l3 = st.columns(3)
l1.metric("Current lookback", int(lookback))
l2.metric("Diffusion minimum", int(min_diffusion_obs))
l3.metric("Preferred lookback", int(preferred_obs))

st.caption(
    f"Selected lookback: {lookback} observations ≈ {approx_years:.1f} years "
    f"at the {interval_label} interval. "
    f"Diffusion minimum = {min_diffusion_obs}; preferred = {preferred_obs}."
)

if lookback < min_diffusion_obs:
    st.warning(
        f"The current lookback ({lookback}) is below the diffusion minimum "
        f"({min_diffusion_obs}) for {interval_label} returns. "
        "Historical estimation can still be used; Diffusion will remain disabled."
    )
elif lookback < preferred_obs:
    st.info(
        f"The current lookback meets the diffusion minimum ({min_diffusion_obs}) "
        f"but is below the preferred sample size ({preferred_obs})."
    )
else:
    st.success(
        f"The current lookback meets the preferred interval-adjusted sample target "
        f"({preferred_obs} observations)."
    )


# ============================================================
# Integrated settings panel
# ============================================================
with st.form("portfolio_settings_form", clear_on_submit=False):
    st.markdown("### Portfolio / Diffusion Settings")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)

    with r1c1:
        gamma = st.number_input(
            "Risk aversion γ",
            min_value=0.1,
            value=float(_saved_portfolio.get("gamma", DEFAULT_GAMMA)),
            step=0.1,
        )

    with r1c2:
        _saved_rule = _saved_portfolio.get("portfolio_rule", "Mean-Variance")
        mean_variance_index = (
            SUPPORTED_RULES.index(_saved_rule)
            if _saved_rule in SUPPORTED_RULES
            else (
                SUPPORTED_RULES.index("Mean-Variance")
                if "Mean-Variance" in SUPPORTED_RULES
                else 0
            )
        )
        rule = st.selectbox(
            "Portfolio rule",
            SUPPORTED_RULES,
            index=mean_variance_index,
        )

    with r1c3:
        constraint_mode = st.selectbox(
            "Portfolio constraint",
            CONSTRAINT_MODES,
            index=(
                CONSTRAINT_MODES.index(_saved_portfolio.get("constraint_mode"))
                if _saved_portfolio.get("constraint_mode") in CONSTRAINT_MODES
                else CONSTRAINT_MODES.index(DEFAULT_CONSTRAINT_MODE)
            ),
            help=(
                "Long-only is recommended for practical use. "
                "Research / Unconstrained reproduces the paper-style optimizer."
            ),
        )

    with r1c4:
        _estimator_options = ["Historical", "Diffusion"]
        _saved_estimator = _saved_portfolio.get("estimator", "Diffusion")
        estimator = st.selectbox(
            "Estimator",
            _estimator_options,
            index=(
                _estimator_options.index(_saved_estimator)
                if _saved_estimator in _estimator_options
                else 1
            ),
            help="Diffusion is the default estimator.",
        )

    max_long_weight = float(DEFAULT_MAX_LONG_WEIGHT)
    max_short_weight = float(DEFAULT_MAX_SHORT_WEIGHT)
    max_gross_exposure = float(DEFAULT_MAX_GROSS_EXPOSURE)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r2c1:
        if constraint_mode == "Long-only":
            min_feasible_weight = 1.0 / n_assets
            default_cap = max(float(_saved_portfolio.get("max_long_weight", DEFAULT_MAX_LONG_WEIGHT)), min_feasible_weight)
            max_long_weight = st.slider(
                "Maximum weight per asset",
                min_value=float(min_feasible_weight),
                max_value=1.0,
                value=float(min(default_cap, 1.0)),
                step=0.05,
                format="%.2f",
            )
        elif constraint_mode == "Limited Long-Short":
            max_long_weight = st.slider(
                "Maximum long weight per asset",
                min_value=0.05,
                max_value=1.0,
                value=float(_saved_portfolio.get("max_long_weight", DEFAULT_MAX_LONG_WEIGHT)),
                step=0.05,
            )
        else:
            st.caption("No long-weight cap in unconstrained mode.")

    with r2c2:
        if constraint_mode == "Limited Long-Short":
            max_short_weight = st.slider(
                "Maximum short weight per asset",
                min_value=0.0,
                max_value=1.0,
                value=float(_saved_portfolio.get("max_short_weight", DEFAULT_MAX_SHORT_WEIGHT)),
                step=0.05,
            )
        else:
            st.number_input(
                "Maximum short weight",
                value=float(DEFAULT_MAX_SHORT_WEIGHT),
                disabled=True,
            )

    with r2c3:
        if constraint_mode == "Limited Long-Short":
            max_gross_exposure = st.slider(
                "Maximum gross exposure",
                min_value=1.0,
                max_value=3.0,
                value=float(_saved_portfolio.get("max_gross_exposure", DEFAULT_MAX_GROSS_EXPOSURE)),
                step=0.1,
            )
        else:
            st.number_input(
                "Maximum gross exposure",
                value=float(DEFAULT_MAX_GROSS_EXPOSURE),
                disabled=True,
            )

    with r2c4:
        if estimator == "Diffusion":
            _score_options = ["Analytical Gaussian", "Learned Neural Score"]
            _saved_score = _saved_portfolio.get("score_model", "Learned Neural Score")
            score_model = st.selectbox(
                "Diffusion score model",
                _score_options,
                index=(
                    _score_options.index(_saved_score)
                    if _saved_score in _score_options
                    else 1
                ),
                help=(
                    "Learned Neural Score is the default. Analytical Gaussian uses the "
                    "closed-form Gaussian score."
                ),
            )
        else:
            score_model = "None"
            st.selectbox(
                "Diffusion score model",
                ["Not used"],
                disabled=True,
            )

    if constraint_mode == "Research / Unconstrained":
        st.warning(
            "Research / Unconstrained can produce very large long and short positions."
        )

    if estimator == "Diffusion":
        st.markdown("**Diffusion / neural settings**")
        d1, d2, d3, d4 = st.columns(4)

        with d1:
            _tuning_options = ["Validation tuned", "Theoretical constrained"]
            _saved_tuning = _saved_portfolio.get("tuning_mode", "Theoretical constrained")
            tuning_mode = st.selectbox(
                "Diffusion horizon tuning",
                _tuning_options,
                index=(
                    _tuning_options.index(_saved_tuning)
                    if _saved_tuning in _tuning_options
                    else 1
                ),
                help=(
                    "Theoretical constrained is the default and uses "
                    "T=max(0, log(n/b*)/β)."
                ),
            )
        with d2:
            m = st.number_input(
                "Synthetic samples M",
                min_value=0,
                value=int(_saved_portfolio.get("m", DEFAULT_M)),
                step=100,
            )
        with d3:
            beta = st.number_input(
                "Constant β",
                min_value=0.01,
                value=float(_saved_portfolio.get("beta", DEFAULT_BETA)),
                step=0.1,
            )
        with d4:
            n_steps = st.number_input(
                "Reverse SDE steps",
                min_value=10,
                value=int(_saved_portfolio.get("n_steps", 100)),
                step=10,
            )

        if score_model == "Learned Neural Score":
            n1, n2, n3, n4 = st.columns(4)

            with n1:
                _neural_mode_options = ["Fast", "Standard", "Research"]
                _saved_neural_mode = _saved_portfolio.get("neural_training_mode", "Fast")
                neural_training_mode = st.selectbox(
                    "Neural training mode",
                    _neural_mode_options,
                    index=(
                        _neural_mode_options.index(_saved_neural_mode)
                        if _saved_neural_mode in _neural_mode_options
                        else 0
                    ),
                    help=(
                        "Fast: interactive use. Standard: stronger training. "
                        "Research: full manual controls."
                    ),
                )

            if neural_training_mode == "Fast":
                neural_epochs = 100
                neural_hidden = 64
                neural_lr = 1e-3
                neural_batch = 64
                neural_val_fraction = 0.20
                neural_patience = 30
                neural_min_delta = 1e-3

            elif neural_training_mode == "Standard":
                neural_epochs = 300
                neural_hidden = 128
                neural_lr = 1e-3
                neural_batch = 64
                neural_val_fraction = 0.20
                neural_patience = 75
                neural_min_delta = 1e-3

            else:
                neural_epochs = int(_saved_portfolio.get("neural_epochs", 1000))
                neural_hidden = int(_saved_portfolio.get("neural_hidden", 128))
                neural_lr = float(_saved_portfolio.get("neural_lr", 1e-3))
                neural_batch = int(_saved_portfolio.get("neural_batch", 32))
                neural_val_fraction = float(_saved_portfolio.get("neural_val_fraction", 0.20))
                neural_patience = int(_saved_portfolio.get("neural_patience", 100))
                neural_min_delta = float(_saved_portfolio.get("neural_min_delta", 1e-3))

            with n2:
                if neural_training_mode == "Research":
                    neural_epochs = st.number_input(
                        "Maximum neural epochs",
                        min_value=50,
                        max_value=5000,
                        value=int(neural_epochs),
                        step=50,
                    )
                else:
                    st.number_input(
                        "Maximum neural epochs",
                        value=int(neural_epochs),
                        disabled=True,
                    )

            with n3:
                if neural_training_mode == "Research":
                    _hidden_options = [32, 64, 128, 256]
                    neural_hidden = st.selectbox(
                        "Hidden width",
                        _hidden_options,
                        index=(
                            _hidden_options.index(int(neural_hidden))
                            if int(neural_hidden) in _hidden_options
                            else 2
                        ),
                    )
                else:
                    st.selectbox(
                        "Hidden width",
                        [int(neural_hidden)],
                        disabled=True,
                    )

            with n4:
                if neural_training_mode == "Research":
                    _batch_options = [16, 32, 64, 128]
                    neural_batch = st.selectbox(
                        "Batch size",
                        _batch_options,
                        index=(
                            _batch_options.index(int(neural_batch))
                            if int(neural_batch) in _batch_options
                            else 1
                        ),
                    )
                else:
                    st.selectbox(
                        "Batch size",
                        [int(neural_batch)],
                        disabled=True,
                    )

            n5, n6, n7 = st.columns(3)
            with n5:
                if neural_training_mode == "Research":
                    _lr_options = [1e-4, 3e-4, 1e-3, 3e-3]
                    neural_lr = st.selectbox(
                        "Learning rate",
                        _lr_options,
                        index=(
                            _lr_options.index(float(neural_lr))
                            if float(neural_lr) in _lr_options
                            else 2
                        ),
                        format_func=lambda x: f"{x:.0e}",
                    )
                else:
                    st.selectbox(
                        "Learning rate",
                        [float(neural_lr)],
                        disabled=True,
                        format_func=lambda x: f"{x:.0e}",
                    )

            with n6:
                if neural_training_mode == "Research":
                    neural_val_fraction = st.slider(
                        "Neural validation fraction",
                        min_value=0.10,
                        max_value=0.40,
                        value=float(neural_val_fraction),
                        step=0.05,
                    )
                else:
                    st.slider(
                        "Neural validation fraction",
                        min_value=0.10,
                        max_value=0.40,
                        value=float(neural_val_fraction),
                        step=0.05,
                        disabled=True,
                    )

            with n7:
                if neural_training_mode == "Research":
                    neural_patience = st.number_input(
                        "Early-stopping patience",
                        min_value=10,
                        max_value=500,
                        value=int(neural_patience),
                        step=10,
                    )
                    neural_min_delta = 1e-3
                else:
                    st.number_input(
                        "Early-stopping patience",
                        value=int(neural_patience),
                        disabled=True,
                    )

            st.caption(
                f"Effective configuration: **{neural_training_mode}** · "
                f"epochs={int(neural_epochs)} · hidden={int(neural_hidden)} · "
                f"batch={int(neural_batch)} · lr={neural_lr:.0e} · "
                f"validation={neural_val_fraction:.0%} · patience={int(neural_patience)}."
            )
        else:
            neural_training_mode = "Not used"
            neural_epochs = 100
            neural_hidden = 64
            neural_lr = 1e-3
            neural_batch = 64
            neural_val_fraction = 0.20
            neural_patience = 30
            neural_min_delta = 1e-3

        if tuning_mode == "Validation tuned":
            validation_fraction = st.slider(
                "Validation fraction",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
            )
        else:
            validation_fraction = 0.20
    else:
        score_model = "None"
        tuning_mode = "None"
        validation_fraction = 0.20
        m = 0
        beta = float(DEFAULT_BETA)
        n_steps = 100
        neural_training_mode = "Not used"
        neural_epochs = 100
        neural_hidden = 64
        neural_lr = 1e-3
        neural_batch = 64
        neural_val_fraction = 0.20
        neural_patience = 30
        neural_min_delta = 1e-3



    run_analysis = st.form_submit_button(
        "▶ Run portfolio analysis",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# Method-comparison studies moved to dedicated page
# ============================================================
st.info(
    "All cross-method research studies, OOS comparison tables, cumulative-wealth "
    "charts, horizon studies, regime diagnostics, and regression tests are now on "
    "the **Method Comparison** page."
)
if st.button(
    "Open Method Comparison →",
    key="open_method_comparison_page",
):
    st.session_state["app_section"] = "Method Comparison"
    st.rerun()

# ============================================================
# Completed-run integrity check
# ============================================================
if _restore_completed_portfolio:
    _saved_n = int(_saved_portfolio.get("n_obs", 0))
    _saved_lb = int(_saved_portfolio.get("lookback", 0))
    if len(returns) < _saved_lb:
        st.error(
            "Saved Portfolio state is inconsistent: the restored dataset is shorter "
            "than the saved lookback. Press Clear and rerun the Portfolio."
        )
        st.stop()

# ============================================================
# Explicit run gate
# ============================================================
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

stop_col, save_col, clear_col = st.columns([1, 1, 1])

with stop_col:
    stop_analysis = st.button(
        "■ Stop",
        use_container_width=True,
        help=(
            "Requests a Streamlit rerun and clears the active analysis state. "
            "This can stop the current workflow between computation stages."
        ),
    )

with save_col:
    save_analysis = st.button(
        "💾 Save",
        use_container_width=True,
        help=(
            "Prepare a ZIP containing the available result tables and charts. "
            "Run the portfolio analysis first."
        ),
    )

with clear_col:
    clear_analysis = st.button(
        "Clear",
        use_container_width=True,
    )

if stop_analysis:
    st.session_state["analysis_stop_requested"] = True
    st.session_state["analysis_has_run"] = False
    st.session_state.pop("portfolio_saved_zip", None)
    st.warning("Stop requested. The current analysis state has been cleared.")
    st.stop()

if run_analysis:
    st.session_state["analysis_stop_requested"] = False
    st.session_state["analysis_has_run"] = True

if clear_analysis:
    for key in [
        "analysis_has_run",
        "analysis_stop_requested",
        "main_analysis_cache_key",
        "main_analysis_results",
        "portfolio_saved_zip",
        "shared_current_window",
        "portfolio_upload_fingerprint",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

if save_analysis:
    if st.session_state.get("analysis_has_run", False):
        st.session_state["portfolio_save_requested"] = True
    else:
        st.warning("Run the portfolio analysis first, then click Save.")

if not st.session_state.get("analysis_has_run", False):
    st.info(
        "Adjust the settings above. Changes are staged inside the form and are only applied when you click **Run portfolio analysis**. "
        "Use **Stop** to cancel/reset the active workflow and **Save** after results exist."
    )
    st.stop()

# ============================================================
# Estimation window
# ============================================================
if _restore_completed_portfolio and isinstance(
    _saved_portfolio.get("window_returns"), pd.DataFrame
):
    # Use the exact estimation window from the completed run.
    window = _saved_portfolio["window_returns"].copy()
    lookback = int(_saved_portfolio.get("lookback", len(window)))
else:
    window = returns.iloc[-int(lookback):]

x = window.to_numpy(dtype=float)

mu_hist, sigma_hist = sample_moments(x, mle=True)
hist_cond = covariance_condition_number(sigma_hist)

st.subheader("Estimation Risk Diagnostics")
d1, d2, d3 = st.columns(3)
d1.metric("Lookback n", f"{len(x)}")
d2.metric("Assets N", f"{n_assets}")
d3.metric("Covariance condition number κ(Σ^diff)", f"{hist_cond:,.1f}")

if _restore_completed_portfolio:
    st.caption(
        f"Restored estimation window: n={len(x)} · "
        f"interval={_saved_portfolio.get('interval_label', interval_label)} · "
        f"saved lookback={_saved_portfolio.get('lookback', lookback)}."
    )

if len(x) <= n_assets:
    st.error(
        "The estimation window has no more observations than assets. "
        "The sample covariance is singular or extremely unstable. "
        "Increase the lookback or reduce the number of assets."
    )
    st.stop()
elif hist_cond < 50:
    st.success(f"Covariance stability: Good ({hist_cond:.1f})")
elif hist_cond < 200:
    st.warning(f"Covariance stability: Moderate ({hist_cond:.1f})")
else:
    st.error(
        f"Covariance stability: Poor ({hist_cond:.1f}). "
        "Consider more history, fewer assets, Ledoit-Wolf, or tighter constraints."
    )

# ============================================================
# Historical vs diffusion moments
# ============================================================
mu_used = mu_hist.copy()
sigma_used = sigma_hist.copy()
used_returns = x.copy()
T = 0.0
b_star = np.nan
fake_samples = None
neural_training_result = None

if st.session_state.get("analysis_stop_requested", False):
    st.warning("Analysis stopped before diffusion estimation.")
    st.stop()

if estimator == "Diffusion":
    # When restoring a completed run, the exact saved window is authoritative.
    # A completed diffusion result must not be invalidated merely because a
    # transient widget/global dataset was reconstructed differently after navigation.
    if _restore_completed_portfolio:
        estimator = str(_saved_portfolio.get("estimator", estimator))
        score_model = str(_saved_portfolio.get("score_model", score_model))
        tuning_mode = str(_saved_portfolio.get("tuning_mode", tuning_mode))

    # The minimum is interval-adjusted:
    # max(about 5 years of returns, 3N observations).
    min_diffusion_obs = max(int(5 * periods_per_year), 3 * n_assets)

    if len(x) < min_diffusion_obs:
        required_years = min_diffusion_obs / float(periods_per_year)
        st.error(
            f"Diffusion is disabled for this estimation window. "
            f"You have {len(x)} {interval_label} return observations for {n_assets} assets; "
            f"the interval-adjusted minimum is {min_diffusion_obs} "
            f"(approximately {required_years:.1f} years, and at least 3N observations). "
            "Increase the lookback or use Historical estimation."
        )
        st.stop()

    if score_model == "Learned Neural Score":
        neural_recommended = max(100, 10 * n_assets)
        if len(x) < neural_recommended:
            st.warning(
                f"Neural score training is statistically fragile with n={len(x)} and N={n_assets}. "
                f"A practical target is at least {neural_recommended} observations. "
                "The app will still run, but treat this as an experimental result."
            )

    validation_results = None
    signal_level = np.nan

    if tuning_mode == "Theoretical constrained":
        try:
            T, b_star, signal_level = theoretical_horizon(
                mu_hist,
                sigma_hist,
                len(x),
                beta=float(beta),
            )
        except ValueError as exc:
            st.error(f"Theoretical tuning failed: {exc}")
            st.stop()

        if T == 0.0:
            st.warning(
                "The theoretical interior solution is infeasible because b*/n >= 1, "
                "so the constrained theoretical optimum is T=0. "
                "For real-data use, consider Validation tuned."
            )

    else:
        try:
            with st.spinner("Selecting T using held-out validation CER..."):
                T, validation_results = validation_tuned_horizon(
                    x,
                    gamma=float(gamma),
                    rule=rule,
                    m=int(m),
                    beta=float(beta),
                    n_steps=int(n_steps),
                    validation_fraction=float(validation_fraction),
                    seed=int(DEFAULT_SEED),
                    constraint_mode=constraint_mode,
                    max_long_weight=float(max_long_weight),
                    max_short_weight=float(max_short_weight),
                    max_gross_exposure=float(max_gross_exposure),
                )
        except Exception as exc:
            st.error(f"Validation tuning failed: {exc}")
            st.stop()

    if score_model == "Learned Neural Score" and T <= 0:
        st.error(
            "The learned neural score requires a positive diffusion horizon T. "
            "Choose Validation tuned, or use a theoretical configuration with T > 0."
        )
        st.stop()

    try:
        if score_model == "Analytical Gaussian":
            with st.spinner("Generating Gaussian-score diffusion samples..."):
                (
                    mu_used,
                    sigma_used,
                    fake_samples,
                    combined,
                ) = _cached_gaussian_diffusion(
                    x,
                    m=int(m),
                    horizon=float(T),
                    beta=float(beta),
                    n_steps=int(n_steps),
                    seed=int(DEFAULT_SEED),
                )
            neural_training_result = None

        else:
            with st.spinner("Training neural score model and generating samples..."):
                (
                    mu_used,
                    sigma_used,
                    fake_samples,
                    combined,
                    neural_training_result,
                ) = _cached_neural_diffusion(
                    x,
                    m=int(m),
                    horizon=float(T),
                    beta=float(beta),
                    n_steps=int(n_steps),
                    epochs=int(neural_epochs),
                    batch_size=int(neural_batch),
                    learning_rate=float(neural_lr),
                    hidden_dim=int(neural_hidden),
                    validation_fraction=float(neural_val_fraction),
                    patience=int(neural_patience),
                    min_delta=float(neural_min_delta),
                    seed=int(DEFAULT_SEED),
                )

        used_returns = combined

    except Exception as exc:
        st.error(f"Diffusion augmentation failed: {exc}")
        st.stop()

if st.session_state.get("analysis_stop_requested", False):
    st.warning("Analysis stopped before portfolio construction.")
    st.stop()

# ============================================================
# Portfolio construction
# ============================================================
try:
    w = compute_weights(
        rule,
        mu_used,
        sigma_used,
        gamma=float(gamma),
        returns=used_returns,
        constraint_mode=constraint_mode,
        max_long_weight=float(max_long_weight),
        max_short_weight=float(max_short_weight),
        max_gross_exposure=float(max_gross_exposure),
    )
except Exception as exc:
    st.error(f"Portfolio construction failed: {exc}")
    st.stop()

w = np.asarray(w, dtype=float).reshape(-1)

if len(w) != n_assets or not np.all(np.isfinite(w)):
    st.error("The portfolio calculation produced an invalid weight vector.")
    st.stop()

# ============================================================
# Output
# ============================================================
weights_df = pd.DataFrame(
    {"Asset": returns.columns, "Weight": w}
).sort_values("Weight", ascending=False)

# ----------------------------------------------------------------
# Shared current-window contract
# ----------------------------------------------------------------
# Method Comparison D must use the exact same current-window Neural result
# instead of independently retraining / resampling a second neural model.
# This removes the previous bug where Recommended Weights and D3 Neural-MV
# could disagree even though the labels/settings looked identical.
shared_source_signature = st.session_state.get("returns_signature")
shared_horizon_months = (
    shared_source_signature.get("horizon_months")
    if isinstance(shared_source_signature, dict)
    else None
)
if shared_horizon_months is None:
    shared_horizon_months = {
        "1 month": 1,
        "3 months": 3,
        "6 months": 6,
        "1 year": 12,
    }.get(interval_label)

# Every completed Portfolio run gets a monotonically increasing revision.
# Method Comparison D stores the revision it was built from, so stale D1-D4
# results cannot survive a newer Portfolio run.
_portfolio_revision = int(st.session_state.get("portfolio_revision", 0)) + 1
st.session_state["portfolio_revision"] = _portfolio_revision

st.session_state["shared_current_window"] = {
    "version": 2,
    "portfolio_revision": _portfolio_revision,
    "source": source,
    "upload_fingerprint": st.session_state.get("portfolio_upload_fingerprint"),
    "returns_signature": shared_source_signature,
    "interval_label": interval_label,
    "horizon_months": shared_horizon_months,
    "asset_names": list(returns.columns),
    "full_returns": returns.copy(),
    "window_index": list(window.index),
    "window_returns": window.copy(),
    "lookback": int(lookback),
    "periods_per_year": int(periods_per_year),
    "n_obs": int(len(x)),
    "gamma": float(gamma),
    "portfolio_rule": str(rule),
    "constraint_mode": str(constraint_mode),
    "max_long_weight": float(max_long_weight),
    "max_short_weight": float(max_short_weight),
    "max_gross_exposure": float(max_gross_exposure),
    "estimator": str(estimator),
    "score_model": str(score_model),
    "tuning_mode": str(tuning_mode),
    "T": float(T),
    "b_star": float(b_star) if np.isfinite(b_star) else np.nan,
    "signal_level": float(signal_level) if "signal_level" in locals() and np.isfinite(signal_level) else np.nan,
    "m": int(m),
    "beta": float(beta),
    "n_steps": int(n_steps),
    "neural_training_mode": str(neural_training_mode),
    "neural_epochs": int(neural_epochs),
    "neural_hidden": int(neural_hidden),
    "neural_lr": float(neural_lr),
    "neural_batch": int(neural_batch),
    "neural_val_fraction": float(neural_val_fraction),
    "neural_patience": int(neural_patience),
    "neural_min_delta": float(neural_min_delta),
    "mu_hist": np.asarray(mu_hist, dtype=float).copy(),
    "sigma_hist": np.asarray(sigma_hist, dtype=float).copy(),
    "mu_used": np.asarray(mu_used, dtype=float).copy(),
    "sigma_used": np.asarray(sigma_used, dtype=float).copy(),
    "used_returns": np.asarray(used_returns, dtype=float).copy(),
    "weights": np.asarray(w, dtype=float).copy(),
    "weights_by_asset": {
        str(asset): float(weight)
        for asset, weight in zip(returns.columns, w)
    },
    "neural_training_result": neural_training_result,
}

st.subheader("Recommended Weights")
st.dataframe(
    weights_df.style.format({"Weight": "{:.2%}"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "These current-window moments and weights are saved in the app session and are "
    "reused by Method Comparison D, so the matching Neural Diffusion portfolio is "
    "not independently retrained."
)

expected_return = portfolio_mean(w, mu_used)
volatility = portfolio_volatility(w, sigma_used)
sharpe = sharpe_ratio(w, mu_used, sigma_used)
cer = certainty_equivalent(w, mu_used, sigma_used, float(gamma))
gross = gross_exposure(w)
net = net_exposure(w)
condition_number = covariance_condition_number(sigma_used)

st.subheader("Portfolio Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Expected return / period", f"{expected_return:.3%}")
m2.metric("Volatility / period", f"{volatility:.3%}")
m3.metric("Sharpe / period", f"{sharpe:.3f}")
m4.metric("CER / period", f"{cer:.3%}")

m5, m6, m7 = st.columns(3)
m5.metric("Gross exposure", f"{gross:.2f}")
m6.metric("Net exposure", f"{net:.2f}")
m7.metric("Covariance condition number", f"{condition_number:,.1f}")

if constraint_mode != "Research / Unconstrained":
    st.caption(
        f"Practical constraint mode: {constraint_mode}. "
        "Weights are fully invested with net exposure approximately 1.0."
    )

if estimator == "Diffusion":
    st.subheader("Diffusion Tuning")

    if tuning_mode == "Theoretical constrained":
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Estimated b*", f"{b_star:.6g}" if np.isfinite(b_star) else "N/A")
        t2.metric(
            "Constrained signal s_T²",
            f"{signal_level:.6g}" if np.isfinite(signal_level) else "N/A",
        )
        t3.metric("Chosen horizon T", f"{T:.6g}")
        t4.metric("Synthetic samples M", f"{int(m):,}")
        st.caption(
            "Theoretical constrained tuning uses "
            "T=max{0, log(n/b*)/β} and s_T²=min{1,b*/n}."
        )
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Chosen horizon T", f"{T:.3f}")
        t2.metric("Synthetic samples M", f"{int(m):,}")
        t3.metric("Tuning method", "Validation CER")

        import pandas as pd
        validation_df = pd.DataFrame(validation_results)
        st.dataframe(
            validation_df.style.format(
                {"T": "{:.2f}", "validation_CER": "{:.4%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Validation tuning chooses the candidate T with the highest held-out "
            "certainty-equivalent return. This is more robust for real data than "
            "directly plugging noisy sample moments into b*."
        )

    if score_model == "Learned Neural Score" and neural_training_result is not None:
        st.subheader("Neural Score Training")

        actual_epochs = len(neural_training_result.train_losses)
        best_epoch = neural_training_result.best_epoch
        best_val = min(neural_training_result.val_losses)
        final_train = neural_training_result.train_losses[-1]
        zero_val = neural_training_result.zero_score_baseline_val
        improvement_vs_zero = (
            100.0 * (zero_val - best_val) / zero_val if zero_val > 0 else np.nan
        )

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Epochs run", f"{actual_epochs}")
        n2.metric("Best epoch", f"{best_epoch}")
        n3.metric("Best validation DSM", f"{best_val:.6f}")
        n4.metric("Zero-score baseline", f"{zero_val:.6f}")

        n5, n6, n7, n8 = st.columns(4)
        n5.metric("Final training DSM", f"{final_train:.6f}")
        n6.metric(
            "Validation improvement vs zero",
            f"{improvement_vs_zero:.1f}%" if np.isfinite(improvement_vs_zero) else "N/A",
        )
        n7.metric(
            "Early stopping",
            "Yes" if neural_training_result.stopped_early else "No",
        )
        n8.metric("Training device", neural_training_result.device)

        loss_df = pd.DataFrame(
            {
                "epoch": np.arange(1, actual_epochs + 1),
                "Training DSM": neural_training_result.train_losses,
                "Validation DSM": neural_training_result.val_losses,
            }
        )
        loss_fig = px.line(
            loss_df,
            x="epoch",
            y=["Training DSM", "Validation DSM"],
            title="Neural Score Training",
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        baseline_df = pd.DataFrame(
            {
                "Metric": [
                    "Train zero-score baseline",
                    "Validation zero-score baseline",
                    "Best validation DSM",
                ],
                "Value": [
                    neural_training_result.zero_score_baseline_train,
                    neural_training_result.zero_score_baseline_val,
                    best_val,
                ],
            }
        )
        st.dataframe(
            baseline_df.style.format({"Value": "{:.6f}"}),
            use_container_width=True,
            hide_index=True,
        )

        if best_val < zero_val:
            st.success(
                f"The learned score beats the zero-score validation baseline by "
                f"{improvement_vs_zero:.1f}%."
            )
        else:
            st.warning(
                "The learned score does not beat the zero-score validation baseline. "
                "Treat the neural diffusion result as unreliable for this sample."
            )

        st.caption(
            "The model uses a chronological train/validation split. Standardization is "
            "fit on the training portion only. Early stopping restores the model from "
            "the epoch with the lowest validation DSM loss."
        )




# ============================================================
# Save current Portfolio results
# ============================================================
portfolio_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Expected return / period",
            "Volatility / period",
            "Sharpe / period",
            "CER / period",
            "Gross exposure",
            "Net exposure",
            "Covariance condition number",
        ],
        "Value": [
            expected_return,
            volatility,
            sharpe,
            cer,
            gross_exposure(w),
            net_exposure(w),
            covariance_condition_number(sigma_used),
        ],
    }
)

moment_df_export = pd.DataFrame(
    {
        "Metric": [
            "Historical mean-vector norm ∥μ^hist∥",
            "Used mean-vector norm ∥μ^diff∥",
            "Historical covariance trace tr(Σ^hist)",
            "Used covariance trace tr(Σ^diff)",
            "Historical covariance condition number κ(Σ^hist)",
            "Used covariance condition number κ(Σ^diff)",
        ],
        "Value": [
            float(np.linalg.norm(mu_hist)),
            float(np.linalg.norm(mu_used)),
            float(np.trace(sigma_hist)),
            float(np.trace(sigma_used)),
            float(covariance_condition_number(sigma_hist)),
            float(covariance_condition_number(sigma_used)),
        ],
    }
)

tuning_df_export = pd.DataFrame(
    {
        "Metric": [
            "Estimator",
            "Score model",
            "Tuning mode",
            "Neural training mode",
            "b*",
            "T",
            "Synthetic samples M",
            "beta",
            "Reverse SDE steps",
        ],
        "Value": [
            estimator,
            score_model,
            tuning_mode,
            neural_training_mode,
            b_star,
            T,
            m,
            beta,
            n_steps,
        ],
    }
)

if st.session_state.pop("portfolio_save_requested", False):
    try:
        saved_zip = _build_portfolio_export_zip(
            weights_df=weights_df,
            portfolio_metrics_df=portfolio_metrics_df,
            moment_df=moment_df_export,
            tuning_df=tuning_df_export,
            validation_df=validation_df if "validation_df" in locals() else None,
            loss_df=loss_df if "loss_df" in locals() else None,
            baseline_df=baseline_df if "baseline_df" in locals() else None,
        )
        st.session_state["portfolio_saved_zip"] = saved_zip
        st.success(
            "Results package prepared. It contains CSV tables and HTML chart files."
        )
    except Exception as exc:
        st.error(f"Could not prepare saved results: {exc}")

if "portfolio_saved_zip" in st.session_state:
    st.download_button(
        "⬇ Download saved Portfolio results (.zip)",
        data=st.session_state["portfolio_saved_zip"],
        file_name="portfolio_results.zip",
        mime="application/zip",
        use_container_width=True,
    )


st.divider()
st.caption(
    "Research/educational prototype only. Before real-money deployment, "
    "add transaction costs, tax effects, liquidity limits, rebalancing rules, "
    "data validation, authentication, and broker-side risk controls."
)
