from pathlib import Path

# Render the complete original legacy research page first. This preserves all
# existing A-E Neural/Gaussian research tools and appends the new short-horizon
# same-frequency OOS study below them.
_legacy_path = Path("views") / "Method_Comparison.py"
_legacy_code = compile(_legacy_path.read_text(encoding="utf-8"), str(_legacy_path), "exec")
exec(_legacy_code, globals(), globals())

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_M,
    DEFAULT_MAX_GROSS_EXPOSURE,
    DEFAULT_MAX_LONG_WEIGHT,
    DEFAULT_MAX_SHORT_WEIGHT,
    DEFAULT_SEED,
)
from core.data import download_yahoo_returns
from core.legacy_periodic_oos import (
    LEGACY_HORIZON_PRESETS,
    get_legacy_horizon_preset,
    legacy_periodic_oos_comparison,
)
from core.short_horizon_portfolio import aggregate_nonoverlapping


@st.cache_data(show_spinner=False)
def _legacy_short_download(tickers, start, source_name):
    interval = "1wk" if source_name == "weekly" else "1mo"
    return download_yahoo_returns(
        tickers,
        start=start,
        end=None,
        interval=interval,
    )


@st.cache_data(show_spinner=False)
def _legacy_short_cached_oos(
    returns,
    lookback,
    test_periods,
    periods_per_year,
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
    return legacy_periodic_oos_comparison(
        returns,
        lookback=int(lookback),
        test_periods=int(test_periods),
        periods_per_year=int(periods_per_year),
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


def _legacy_short_export(result):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in result.items():
            if isinstance(df, pd.DataFrame):
                zf.writestr(f"legacy_short_{name}.csv", df.to_csv(index=False))
        zf.writestr(
            "README.txt",
            "Legacy short-horizon OOS comparison: Historical, Gaussian Diffusion, and Learned Neural Diffusion.\n"
            "Holding periods: 1 week, 2 weeks, 1 month, and 3 months.\n"
            "Each horizon is evaluated on the next real return at that same holding period.\n",
        )
    return buffer.getvalue()


st.divider()
st.header("F. Short-Horizon Legacy OOS Comparison")
st.caption(
    "Adds 1-week, 2-week, 1-month, and 3-month holding periods to the Legacy research page. "
    "This section intentionally keeps the old Monte-Carlo Gaussian Diffusion and Learned Neural Score, "
    "but fixes the evaluation frequency: a 1-week estimator is tested on the next real 1-week return, "
    "a 2-week estimator on the next real 2-week return, and so on."
)

st.info(
    "This is different from the new non-Legacy Method Comparison. The new page uses exact moments, "
    "nested Best-T, and turnover control. This Legacy section preserves the older Gaussian/Neural "
    "estimators so you can study whether the Learned Neural Score adds value at short horizons."
)

# Reuse the values selected on the original Legacy page whenever available.
_default_tickers = globals().get("ticker_list")
if not isinstance(_default_tickers, list) or not _default_tickers:
    _default_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
_default_start = globals().get("start_date")
if not isinstance(_default_start, str) or not _default_start:
    _default_start = "2000-01-01"

s1, s2 = st.columns([2, 1])
with s1:
    legacy_short_tickers_text = st.text_input(
        "F tickers",
        ",".join(_default_tickers),
        key="legacy_short_tickers",
    )
with s2:
    legacy_short_start = st.text_input(
        "F start date",
        _default_start,
        key="legacy_short_start",
    )

legacy_short_tickers = [
    x.strip().upper() for x in legacy_short_tickers_text.split(",") if x.strip()
]
legacy_short_horizons = st.multiselect(
    "F holding periods",
    list(LEGACY_HORIZON_PRESETS.keys()),
    default=["1 week", "2 weeks", "1 month", "3 months"],
    key="legacy_short_horizons",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    legacy_short_speed = st.selectbox(
        "F speed mode",
        ["Turbo", "Fast", "Research"],
        index=0,
        key="legacy_short_speed",
        help=(
            "Neural Diffusion is retrained in every rolling OOS window. Turbo/Fast cap the number "
            "of OOS periods and neural training effort; Research uses the requested settings."
        ),
    )
with c2:
    legacy_short_requested = st.number_input(
        "F requested OOS periods / horizon",
        min_value=4,
        max_value=156,
        value=26,
        step=2,
        key="legacy_short_oos_periods",
    )
with c3:
    legacy_short_gamma = st.number_input(
        "F risk aversion γ",
        min_value=0.1,
        value=float(globals().get("gamma", DEFAULT_GAMMA)),
        step=0.1,
        key="legacy_short_gamma",
    )
with c4:
    _legacy_rule_default = str(globals().get("rule", "Mean-Variance"))
    legacy_short_rule = st.selectbox(
        "F portfolio rule",
        SUPPORTED_RULES,
        index=(SUPPORTED_RULES.index(_legacy_rule_default) if _legacy_rule_default in SUPPORTED_RULES else 0),
        key="legacy_short_rule",
    )

c5, c6, c7, c8 = st.columns(4)
with c5:
    _constraint_default = str(globals().get("constraint_mode", "Long-only"))
    legacy_short_constraint = st.selectbox(
        "F portfolio constraint",
        CONSTRAINT_MODES,
        index=(
            CONSTRAINT_MODES.index(_constraint_default)
            if _constraint_default in CONSTRAINT_MODES
            else 0
        ),
        key="legacy_short_constraint",
    )
with c6:
    legacy_short_max_long = st.number_input(
        "F max long weight",
        min_value=max(0.05, 1.0 / max(len(legacy_short_tickers), 1)),
        max_value=1.0,
        value=max(
            float(globals().get("max_long_weight", DEFAULT_MAX_LONG_WEIGHT)),
            1.0 / max(len(legacy_short_tickers), 1),
        ),
        step=0.05,
        key="legacy_short_max_long",
    )
with c7:
    legacy_short_max_short = st.number_input(
        "F max short weight",
        min_value=0.0,
        max_value=1.0,
        value=float(globals().get("max_short_weight", DEFAULT_MAX_SHORT_WEIGHT)),
        step=0.05,
        key="legacy_short_max_short",
    )
with c8:
    legacy_short_max_gross = st.number_input(
        "F max gross exposure",
        min_value=1.0,
        max_value=5.0,
        value=float(globals().get("max_gross_exposure", DEFAULT_MAX_GROSS_EXPOSURE)),
        step=0.1,
        key="legacy_short_max_gross",
    )

# Pull the old Neural/Gaussian controls from the Legacy page above.
base_m = int(globals().get("m", DEFAULT_M))
base_beta = float(globals().get("beta", DEFAULT_BETA))
base_steps = int(globals().get("n_steps", 100))
base_epochs = int(globals().get("neural_epochs", 300))
base_batch = int(globals().get("neural_batch", 64))
base_lr = float(globals().get("neural_lr", 1e-3))
base_hidden = int(globals().get("neural_hidden", 128))
base_val_fraction = float(globals().get("neural_val_fraction", 0.20))
base_patience = int(globals().get("neural_patience", 75))
base_min_delta = float(globals().get("neural_min_delta", 1e-3))

if legacy_short_speed == "Turbo":
    effective_oos = min(int(legacy_short_requested), 12)
    effective_m = min(base_m, 100)
    effective_steps = min(base_steps, 25)
    effective_epochs = min(base_epochs, 50)
    effective_patience = min(base_patience, 15)
    effective_hidden = min(base_hidden, 64)
    effective_batch = max(base_batch, 64)
elif legacy_short_speed == "Fast":
    effective_oos = min(int(legacy_short_requested), 26)
    effective_m = min(base_m, 200)
    effective_steps = min(base_steps, 50)
    effective_epochs = min(base_epochs, 100)
    effective_patience = min(base_patience, 30)
    effective_hidden = min(base_hidden, 128)
    effective_batch = max(base_batch, 64)
else:
    effective_oos = int(legacy_short_requested)
    effective_m = base_m
    effective_steps = base_steps
    effective_epochs = base_epochs
    effective_patience = base_patience
    effective_hidden = base_hidden
    effective_batch = base_batch

st.caption(
    f"Effective F configuration: mode={legacy_short_speed} · OOS periods≤{effective_oos} per horizon · "
    f"M={effective_m} · reverse steps={effective_steps} · neural epochs≤{effective_epochs} · "
    f"hidden≤{effective_hidden} · batch={effective_batch} · lr={base_lr:.0e}."
)

with st.expander("F methodology"):
    st.markdown(
        """
        **Methods**: Historical, Gaussian Diffusion, Learned Neural Diffusion.  
        **T selection**: the original Legacy hierarchy is preserved: positive theoretical T first; "
        "if theory gives T=0, use the old single validation split; if that cannot be used, apply the "
        "positive safe fallback T.  
        **OOS timing**: every portfolio is estimated only from prior observations and then held for exactly "
        "the selected next period. No future return is used to estimate that period's weights."
        """
    )

if st.button(
    "▶ Run F short-horizon Legacy comparison",
    type="primary",
    use_container_width=True,
    key="legacy_short_run",
):
    if not legacy_short_tickers:
        st.error("Enter at least one ticker for Section F.")
        st.stop()
    if not legacy_short_horizons:
        st.error("Select at least one F holding period.")
        st.stop()

    summaries = []
    details = []
    sources = []
    regimes = []
    wealth_tables = []
    coverage = []

    needed_sources = {
        str(get_legacy_horizon_preset(h)["source"]) for h in legacy_short_horizons
    }
    base_data = {}
    progress = st.progress(0, text="Downloading Legacy short-horizon data...")

    for j, src in enumerate(sorted(needed_sources)):
        base_data[src] = _legacy_short_download(
            tuple(legacy_short_tickers),
            legacy_short_start,
            src,
        )[legacy_short_tickers].dropna()
        progress.progress(
            10 + int(10 * (j + 1) / max(len(needed_sources), 1)),
            text=f"Prepared {src} base returns.",
        )

    for h_idx, horizon in enumerate(legacy_short_horizons):
        cfg = get_legacy_horizon_preset(horizon)
        holding_returns = aggregate_nonoverlapping(
            base_data[str(cfg["source"])],
            int(cfg["block_size"]),
        )
        default_lb = int(cfg["lookback"])
        lookback = min(default_lb, len(holding_returns) - 1)
        minimum_lb = max(30, 3 * len(legacy_short_tickers))
        if lookback < minimum_lb:
            st.warning(
                f"Skipping {horizon}: only {len(holding_returns)} observations are available, "
                f"which is insufficient for the Legacy diffusion comparison."
            )
            continue

        progress.progress(
            20 + int(75 * h_idx / max(len(legacy_short_horizons), 1)),
            text=(
                f"Running {horizon}: Historical / Gaussian / Neural over up to "
                f"{effective_oos} true OOS periods..."
            ),
        )

        summary, detail, wealth, diagnostics, source_counts, regime_detail = _legacy_short_cached_oos(
            holding_returns,
            lookback,
            effective_oos,
            int(cfg["periods_per_year"]),
            legacy_short_gamma,
            legacy_short_rule,
            effective_m,
            base_beta,
            effective_steps,
            legacy_short_constraint,
            legacy_short_max_long,
            legacy_short_max_short,
            legacy_short_max_gross,
            effective_epochs,
            effective_batch,
            base_lr,
            effective_hidden,
            base_val_fraction,
            effective_patience,
            base_min_delta,
            int(DEFAULT_SEED + 10000 * h_idx),
        )

        summary.insert(0, "Horizon", horizon)
        summary.insert(1, "Lookback", lookback)
        detail.insert(0, "Horizon", horizon)
        source_counts.insert(0, "Horizon", horizon)
        regime_detail.insert(0, "Horizon", horizon)

        wealth_out = wealth.reset_index().rename(columns={wealth.index.name or "index": "Realized date"})
        wealth_out.insert(0, "Horizon", horizon)

        summaries.append(summary)
        details.append(detail)
        sources.append(source_counts)
        regimes.append(regime_detail)
        wealth_tables.append(wealth_out)
        coverage.append({"Horizon": horizon, "Lookback": lookback, **diagnostics})

    if not summaries:
        progress.empty()
        st.error("No selected F horizon had enough data to run.")
        st.stop()

    result = {
        "summary": pd.concat(summaries, ignore_index=True),
        "detail": pd.concat(details, ignore_index=True, sort=False),
        "t_source_counts": pd.concat(sources, ignore_index=True),
        "regime_detail": pd.concat(regimes, ignore_index=True, sort=False),
        "wealth": pd.concat(wealth_tables, ignore_index=True, sort=False),
        "coverage": pd.DataFrame(coverage),
    }
    st.session_state["legacy_short_horizon_results"] = result
    progress.progress(100, text="Legacy short-horizon comparison complete.")
    progress.empty()

if "legacy_short_horizon_results" in st.session_state:
    f_res = st.session_state["legacy_short_horizon_results"]
    f_summary = f_res["summary"].copy()

    st.subheader("F1. OOS Performance")
    st.dataframe(
        f_summary.style.format(
            {
                "Return / period": "{:.3%}",
                "Volatility / period": "{:.3%}",
                "Sharpe / period": "{:.3f}",
                "CER / period": "{:.3%}",
                "CAGR": "{:.2%}",
                "Annualized volatility": "{:.2%}",
                "Annualized Sharpe": "{:.3f}",
                "Average turnover": "{:.2%}",
                "Max drawdown": "{:.2%}",
                "Net CAGR 10bps": "{:.2%}",
                "Net CAGR 25bps": "{:.2%}",
                "Final $10,000": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    winners = f_summary.loc[
        f_summary.groupby("Horizon")["CAGR"].idxmax(),
        ["Horizon", "Method", "CAGR"],
    ].reset_index(drop=True)
    st.markdown("**F gross-CAGR winner by holding period**")
    st.dataframe(
        winners.style.format({"CAGR": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        f_summary,
        x="Horizon",
        y="CAGR",
        color="Method",
        barmode="group",
        title="Legacy Historical vs Gaussian vs Neural — OOS CAGR",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("F2. T-Selection Coverage")
    st.dataframe(f_res["coverage"], use_container_width=True, hide_index=True)
    st.dataframe(f_res["t_source_counts"], use_container_width=True, hide_index=True)

    with st.expander("F3. Rolling OOS detail"):
        st.dataframe(
            f_res["detail"].style.format(
                {
                    "Realized return": "{:.3%}",
                    "Turnover": "{:.2%}",
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

    st.download_button(
        "⬇ Download F short-horizon Legacy results (.zip)",
        data=_legacy_short_export(f_res),
        file_name="legacy_short_horizon_method_comparison.zip",
        mime="application/zip",
        use_container_width=True,
        key="legacy_short_download_results",
    )
