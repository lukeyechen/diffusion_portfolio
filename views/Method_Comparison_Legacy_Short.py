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
    DEFAULT_MAX_GROSS_EXPOSURE,
    DEFAULT_MAX_LONG_WEIGHT,
    DEFAULT_MAX_SHORT_WEIGHT,
    DEFAULT_SEED,
    SUPPORTED_RULES,
)
from core.data import download_yahoo_returns, load_returns_csv
from core.diffusion import diffusion_augmented_moments
from core.horizon_study import return_horizon_study
from core.legacy_periodic_oos import (
    DEFAULT_SAFE_T,
    DEFAULT_T_GRID,
    LEGACY_HORIZON_PRESETS,
    get_legacy_horizon_preset,
    legacy_periodic_oos_comparison,
)
from core.moments import covariance_condition_number, sample_moments
from core.neural_score import neural_diffusion_augmented_moments
from core.portfolio_rules import compute_weights
from core.short_horizon_portfolio import aggregate_nonoverlapping
from core.tuning import theoretical_horizon, validation_tuned_horizon


HORIZON_LABELS = ["1 week", "2 weeks", "1 month", "3 months"]
DISPLAY_LABELS = {
    "1 week": "1W",
    "2 weeks": "2W",
    "1 month": "1M",
    "3 months": "3M",
}


@st.cache_data(show_spinner=False)
def _download_base(tickers, start, source_name):
    interval = "1wk" if source_name == "weekly" else "1mo"
    return download_yahoo_returns(
        tickers,
        start=start,
        end=None,
        interval=interval,
    )


def _holding_returns(base: pd.DataFrame, horizon: str) -> pd.DataFrame:
    cfg = get_legacy_horizon_preset(horizon)
    return aggregate_nonoverlapping(base, int(cfg["block_size"]))


def _effective_speed(mode, requested_oos, m, steps, epochs, hidden, batch, patience):
    if mode == "Turbo":
        return {
            "oos": min(int(requested_oos), 12),
            "m": min(int(m), 100),
            "steps": min(int(steps), 25),
            "epochs": min(int(epochs), 50),
            "hidden": min(int(hidden), 64),
            "batch": max(int(batch), 64),
            "patience": min(int(patience), 15),
        }
    if mode == "Fast":
        return {
            "oos": min(int(requested_oos), 26),
            "m": min(int(m), 200),
            "steps": min(int(steps), 50),
            "epochs": min(int(epochs), 100),
            "hidden": min(int(hidden), 128),
            "batch": max(int(batch), 64),
            "patience": min(int(patience), 30),
        }
    return {
        "oos": int(requested_oos),
        "m": int(m),
        "steps": int(steps),
        "epochs": int(epochs),
        "hidden": int(hidden),
        "batch": int(batch),
        "patience": int(patience),
    }


@st.cache_data(show_spinner=False)
def _cached_legacy_oos(
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


def _legacy_t_for_window(
    x,
    *,
    gamma,
    rule,
    m,
    beta,
    n_steps,
    constraint_mode,
    max_long_weight,
    max_short_weight,
    max_gross_exposure,
    seed,
):
    mu_h, sigma_h = sample_moments(x, mle=True)
    try:
        t_theory, b_star, signal = theoretical_horizon(
            mu_h,
            sigma_h,
            n_obs=len(x),
            beta=float(beta),
        )
    except Exception:
        t_theory, b_star, signal = 0.0, np.nan, np.nan

    if np.isfinite(t_theory) and t_theory > 0:
        return float(t_theory), "Theoretical", float(b_star), float(signal), mu_h, sigma_h

    vf = 0.10 if len(x) < 30 else (0.15 if len(x) < 50 else 0.20)
    n_val = max(10, int(round(len(x) * vf)))
    n_train = len(x) - n_val
    if n_train >= max(30, 3 * x.shape[1]):
        try:
            t_val, _ = validation_tuned_horizon(
                x,
                gamma=float(gamma),
                rule=rule,
                m=int(m),
                beta=float(beta),
                n_steps=int(n_steps),
                candidate_T=list(DEFAULT_T_GRID),
                validation_fraction=float(vf),
                seed=int(seed),
                constraint_mode=constraint_mode,
                max_long_weight=float(max_long_weight),
                max_short_weight=float(max_short_weight),
                max_gross_exposure=float(max_gross_exposure),
            )
            if np.isfinite(t_val) and t_val > 0:
                return float(t_val), "Validation fallback", float(b_star), float(signal), mu_h, sigma_h
        except Exception:
            pass

    return float(DEFAULT_SAFE_T), "Safe fallback", float(b_star), float(signal), mu_h, sigma_h


def _weights(rule, mu, sigma, returns, *, gamma, constraint_mode, max_long_weight, max_short_weight, max_gross_exposure):
    return np.asarray(
        compute_weights(
            rule,
            mu,
            sigma,
            gamma=float(gamma),
            returns=returns,
            constraint_mode=constraint_mode,
            max_long_weight=float(max_long_weight),
            max_short_weight=float(max_short_weight),
            max_gross_exposure=float(max_gross_exposure),
        ),
        dtype=float,
    ).reshape(-1)


@st.cache_data(show_spinner=False)
def _current_window_comparison(
    returns,
    lookback,
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
    data = returns.replace([np.inf, -np.inf], np.nan).dropna()
    x = data.iloc[-int(lookback):].to_numpy(dtype=float)
    assets = list(data.columns)

    T, t_source, b_star, theory_signal, mu_h, sigma_h = _legacy_t_for_window(
        x,
        gamma=gamma,
        rule=rule,
        m=m,
        beta=beta,
        n_steps=n_steps,
        constraint_mode=constraint_mode,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
        max_gross_exposure=max_gross_exposure,
        seed=int(seed + 71),
    )

    w_h = _weights(
        rule, mu_h, sigma_h, x,
        gamma=gamma,
        constraint_mode=constraint_mode,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
        max_gross_exposure=max_gross_exposure,
    )

    mu_g, sigma_g, _, combined_g = diffusion_augmented_moments(
        x,
        m=int(m),
        horizon=float(T),
        beta=float(beta),
        n_steps=int(n_steps),
        seed=int(seed + 101),
    )
    w_g = _weights(
        rule, mu_g, sigma_g, combined_g,
        gamma=gamma,
        constraint_mode=constraint_mode,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
        max_gross_exposure=max_gross_exposure,
    )

    mu_n, sigma_n, _, combined_n, neural_result = neural_diffusion_augmented_moments(
        x,
        m=int(m),
        horizon=float(T),
        beta=float(beta),
        n_steps=int(n_steps),
        epochs=int(neural_epochs),
        batch_size=int(neural_batch),
        learning_rate=float(neural_learning_rate),
        hidden_dim=int(neural_hidden_dim),
        validation_fraction=float(neural_validation_fraction),
        patience=int(neural_patience),
        min_delta=float(neural_min_delta),
        seed=int(seed + 202),
    )
    w_n = _weights(
        rule, mu_n, sigma_n, combined_n,
        gamma=gamma,
        constraint_mode=constraint_mode,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
        max_gross_exposure=max_gross_exposure,
    )

    methods = {
        "Historical": (mu_h, sigma_h, w_h),
        "Gaussian Diffusion": (mu_g, sigma_g, w_g),
        "Neural Diffusion": (mu_n, sigma_n, w_n),
    }

    metric_rows = []
    weight_rows = []
    moment_rows = []
    for method, (mu, sigma, w) in methods.items():
        er = float(w @ mu)
        var = float(w @ sigma @ w)
        vol = float(np.sqrt(max(var, 0.0)))
        cer = er - 0.5 * float(gamma) * var
        hist_er = float(w @ mu_h)
        hist_var = float(w @ sigma_h @ w)
        hist_cer = hist_er - 0.5 * float(gamma) * hist_var
        metric_rows.append(
            {
                "Estimator": method,
                "Expected return": er,
                "Volatility": vol,
                "CER": cer,
                "Historical benchmark CER": hist_cer,
            }
        )
        moment_rows.append(
            {
                "Estimator": method,
                "Mean norm": float(np.linalg.norm(mu)),
                "Covariance trace": float(np.trace(sigma)),
                "Covariance condition number": float(covariance_condition_number(sigma)),
            }
        )
        for asset, weight in zip(assets, w):
            weight_rows.append({"Estimator": method, "Asset": asset, "Weight": float(weight)})

    neural_best = (
        float(min(neural_result.val_losses)) if len(neural_result.val_losses) else np.nan
    )
    metadata = {
        "T": float(T),
        "T source": t_source,
        "b*": b_star,
        "b*/n": float(b_star / len(x)) if np.isfinite(b_star) else np.nan,
        "Theoretical signal": theory_signal,
        "s_T^2": float(np.exp(-float(beta) * float(T))),
        "Lookback": int(lookback),
        "Neural best validation DSM": neural_best,
    }
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(moment_rows),
        metadata,
    )


def _build_intensity(detail: pd.DataFrame, gamma: float) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    pieces = []
    for horizon, g in detail.groupby("Horizon", sort=False):
        ppy = int(get_legacy_horizon_preset(horizon)["periods_per_year"])
        pivot = g.pivot_table(
            index="Realized date",
            columns="Method",
            values="Realized return",
            aggfunc="first",
        ).sort_index()
        needed = {"Historical", "Gaussian Diffusion", "Neural Diffusion"}
        if not needed.issubset(pivot.columns):
            continue
        meta = (
            g[["Realized date", "T", "T source", "b*/n"]]
            .drop_duplicates("Realized date")
            .set_index("Realized date")
            .sort_index()
        )
        out = pivot.join(meta, how="left").reset_index()
        out.insert(0, "Horizon", horizon)
        out["Neural - Historical"] = out["Neural Diffusion"] - out["Historical"]
        out["Gaussian - Historical"] = out["Gaussian Diffusion"] - out["Historical"]

        window = max(4, ppy)
        for method in ["Historical", "Neural Diffusion"]:
            r = out[method]
            roll_mean = r.rolling(window, min_periods=window).mean()
            roll_var = r.rolling(window, min_periods=window).var(ddof=0)
            out[f"Rolling CER {method}"] = roll_mean - 0.5 * float(gamma) * roll_var
        out["Rolling CER Neural - Historical"] = (
            out["Rolling CER Neural Diffusion"] - out["Rolling CER Historical"]
        )
        out["CER window periods"] = window
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _results_zip(prefix: str, payload: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, obj in payload.items():
            if isinstance(obj, pd.DataFrame):
                zf.writestr(f"{prefix}_{name}.csv", obj.to_csv(index=False))
        zf.writestr(
            "README.txt",
            "Method Comparison (Legacy) short-horizon research output.\n"
            "Holding periods are 1W, 2W, 1M, and 3M and OOS evaluation uses the same realized frequency.\n",
        )
    return buffer.getvalue()


# =============================================================================
# Page header and shared controls
# =============================================================================
st.title("Method Comparison (Legacy) — Short-Horizon Research")
st.caption(
    "Legacy Historical / Monte-Carlo Gaussian Diffusion / Learned Neural Diffusion, "
    "upgraded consistently to 1W, 2W, 1M, and 3M holding periods."
)
st.info(
    "All sections A–E now use short holding periods. B and C test each model on the next real return "
    "at the same frequency; D is the current-window legacy estimator comparison; E studies diffusion intensity."
)

st.subheader("Comparison Data")
source = st.radio(
    "Data source",
    ["Yahoo Finance", "Upload CSV"],
    horizontal=True,
    key="legacy_full_source",
)

uploaded_returns = None
if source == "Yahoo Finance":
    d1, d2 = st.columns([2, 1])
    with d1:
        ticker_text = st.text_input(
            "Tickers",
            st.session_state.get("legacy_full_tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN"),
            key="legacy_full_ticker_text",
        )
    with d2:
        start_date = st.text_input(
            "Start date",
            st.session_state.get("legacy_full_start", "2000-01-01"),
            key="legacy_full_start_text",
        )
    ticker_list = [x.strip().upper() for x in ticker_text.split(",") if x.strip()]
else:
    uploaded = st.file_uploader(
        "Upload return CSV",
        type=["csv"],
        key="legacy_full_csv",
    )
    if uploaded is not None:
        uploaded_returns = load_returns_csv(uploaded)
    ticker_list = list(uploaded_returns.columns) if uploaded_returns is not None else ["Asset"]
    start_date = None

st.subheader("Shared Comparison Settings")
c1, c2, c3, c4 = st.columns(4)
with c1:
    selected_horizon = st.selectbox(
        "Selected holding / return horizon",
        HORIZON_LABELS,
        index=0,
        key="legacy_full_horizon",
        format_func=lambda x: DISPLAY_LABELS[x],
        help="Used by Sections B and D. Section C runs all four horizons.",
    )
with c2:
    requested_oos = st.slider(
        "OOS periods",
        min_value=4,
        max_value=156,
        value=52,
        step=4,
        key="legacy_full_oos",
        help="Number of true out-of-sample holding periods requested. Turbo/Fast may cap this for speed.",
    )
with c3:
    speed_mode = st.selectbox(
        "Research speed",
        ["Turbo", "Fast", "Research"],
        index=0,
        key="legacy_full_speed",
    )
with c4:
    gamma = st.number_input(
        "Risk aversion γ",
        min_value=0.1,
        value=float(DEFAULT_GAMMA),
        step=0.1,
        key="legacy_full_gamma",
    )

c5, c6, c7, c8 = st.columns(4)
with c5:
    rule = st.selectbox(
        "Portfolio rule",
        SUPPORTED_RULES,
        index=1 if "Mean-Variance" in SUPPORTED_RULES else 0,
        key="legacy_full_rule",
    )
with c6:
    constraint_mode = st.selectbox(
        "Portfolio constraint",
        CONSTRAINT_MODES,
        index=0,
        key="legacy_full_constraint",
    )
with c7:
    n_asset_hint = max(len(ticker_list), 1)
    max_long_weight = st.number_input(
        "Maximum long weight",
        min_value=min(1.0, max(0.05, 1.0 / n_asset_hint)),
        max_value=1.0,
        value=max(float(DEFAULT_MAX_LONG_WEIGHT), 1.0 / n_asset_hint),
        step=0.05,
        key="legacy_full_max_long",
    )
with c8:
    max_short_weight = st.number_input(
        "Maximum short weight",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_MAX_SHORT_WEIGHT),
        step=0.05,
        key="legacy_full_max_short",
    )

c9, c10 = st.columns(2)
with c9:
    max_gross_exposure = st.number_input(
        "Maximum gross exposure",
        min_value=1.0,
        max_value=5.0,
        value=float(DEFAULT_MAX_GROSS_EXPOSURE),
        step=0.1,
        key="legacy_full_gross",
    )
with c10:
    selected_default_lb = int(get_legacy_horizon_preset(selected_horizon)["lookback"])
    min_lookback = max(20, 3 * n_asset_hint)
    selected_lookback = st.number_input(
        "Selected-horizon rolling lookback",
        min_value=min_lookback,
        max_value=1040,
        value=max(min_lookback, selected_default_lb),
        step=1,
        key=f"legacy_full_lookback_{selected_horizon}",
        help=f"Preset for {DISPLAY_LABELS[selected_horizon]} is {selected_default_lb} observations (~10 years).",
    )

st.markdown("**Legacy diffusion / neural settings**")
n1, n2, n3, n4 = st.columns(4)
with n1:
    m = st.number_input(
        "Synthetic samples M",
        min_value=50,
        max_value=5000,
        value=int(DEFAULT_M),
        step=50,
        key="legacy_full_m",
    )
with n2:
    beta = st.number_input(
        "Constant β",
        min_value=0.01,
        value=float(DEFAULT_BETA),
        step=0.1,
        key="legacy_full_beta",
    )
with n3:
    n_steps = st.number_input(
        "Reverse SDE steps",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        key="legacy_full_steps",
    )
with n4:
    neural_epochs = st.number_input(
        "Maximum neural epochs",
        min_value=20,
        max_value=3000,
        value=300,
        step=20,
        key="legacy_full_epochs",
    )

n5, n6, n7, n8 = st.columns(4)
with n5:
    neural_hidden = st.selectbox(
        "Hidden width",
        [32, 64, 128, 256],
        index=2,
        key="legacy_full_hidden",
    )
with n6:
    neural_lr = st.selectbox(
        "Learning rate",
        [1e-4, 3e-4, 1e-3, 3e-3],
        index=2,
        format_func=lambda x: f"{x:.0e}",
        key="legacy_full_lr",
    )
with n7:
    neural_batch = st.selectbox(
        "Batch size",
        [16, 32, 64, 128],
        index=2,
        key="legacy_full_batch",
    )
with n8:
    neural_val_fraction = st.slider(
        "Neural validation fraction",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        key="legacy_full_val_fraction",
    )

n9, n10 = st.columns(2)
with n9:
    neural_patience = st.number_input(
        "Early-stopping patience",
        min_value=5,
        max_value=500,
        value=75,
        step=5,
        key="legacy_full_patience",
    )
with n10:
    neural_min_delta = st.selectbox(
        "Minimum validation improvement",
        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        index=2,
        format_func=lambda x: f"{x:.0e}",
        key="legacy_full_min_delta",
    )

speed = _effective_speed(
    speed_mode,
    requested_oos,
    m,
    n_steps,
    neural_epochs,
    neural_hidden,
    neural_batch,
    neural_patience,
)
st.caption(
    f"Effective {speed_mode} settings: OOS≤{speed['oos']} · M={speed['m']} · reverse steps={speed['steps']} · "
    f"neural epochs≤{speed['epochs']} · hidden≤{speed['hidden']} · batch={speed['batch']} · patience={speed['patience']}."
)


def _selected_returns() -> pd.DataFrame:
    if source == "Upload CSV":
        if uploaded_returns is None:
            raise ValueError("Upload a CSV first.")
        return uploaded_returns.replace([np.inf, -np.inf], np.nan).dropna()
    cfg = get_legacy_horizon_preset(selected_horizon)
    base = _download_base(tuple(ticker_list), start_date, str(cfg["source"]))
    return _holding_returns(base[ticker_list].dropna(), selected_horizon)


# =============================================================================
# A. Return-horizon study
# =============================================================================
st.divider()
st.header("A. Return-Horizon Study")
st.caption("Theoretical diffusion feasibility for 1W, 2W, 1M, and 3M returns.")

if st.button("Run A: 1W / 2W / 1M / 3M Horizon Study", type="primary", use_container_width=True):
    if source != "Yahoo Finance":
        st.error("Section A automatic multi-horizon construction requires Yahoo Finance.")
    else:
        try:
            with st.spinner("Building short-horizon returns and theoretical T values..."):
                weekly = _download_base(tuple(ticker_list), start_date, "weekly")[ticker_list].dropna()
                monthly = _download_base(tuple(ticker_list), start_date, "monthly")[ticker_list].dropna()
                horizon_data = {
                    "1W": weekly,
                    "2W": aggregate_nonoverlapping(weekly, 2),
                    "1M": monthly,
                    "3M": aggregate_nonoverlapping(monthly, 3),
                }
                a_table = return_horizon_study(horizon_data, beta=float(beta))
            st.session_state["legacy_full_A"] = a_table
        except Exception as exc:
            st.error(f"Section A failed: {exc}")

if "legacy_full_A" in st.session_state:
    a_table = st.session_state["legacy_full_A"]
    st.dataframe(
        a_table.style.format({"N/n": "{:.4f}", "b*": "{:.4f}", "b*/n": "{:.4f}", "T": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    fig_a = px.bar(a_table, x="Horizon", y="b*/n", text_auto=".3f", title="Theoretical feasibility ratio")
    fig_a.add_hline(y=1.0, line_dash="dash", annotation_text="b*/n = 1")
    st.plotly_chart(fig_a, use_container_width=True)


# =============================================================================
# B. Selected-horizon true OOS comparison
# =============================================================================
st.divider()
st.header(f"B. Selected-Horizon OOS Comparison — {DISPLAY_LABELS[selected_horizon]}")
st.caption(
    "Historical, Gaussian Diffusion, and Neural Diffusion are estimated from prior data and tested on the next real "
    f"{DISPLAY_LABELS[selected_horizon]} return. This is no longer a monthly-only evaluation."
)

if st.button(f"Run B: {DISPLAY_LABELS[selected_horizon]} OOS Comparison", type="primary", use_container_width=True):
    try:
        selected_returns = _selected_returns()
        lookback_b = min(int(selected_lookback), len(selected_returns) - 1)
        if lookback_b < max(30, 3 * selected_returns.shape[1]):
            raise ValueError("Not enough observations for the selected rolling lookback.")
        cfg_b = get_legacy_horizon_preset(selected_horizon)
        with st.spinner(f"Running {DISPLAY_LABELS[selected_horizon]} Legacy OOS comparison..."):
            summary, detail, wealth, diagnostics, source_counts, regime_detail = _cached_legacy_oos(
                selected_returns,
                lookback_b,
                speed["oos"],
                int(cfg_b["periods_per_year"]),
                gamma,
                rule,
                speed["m"],
                beta,
                speed["steps"],
                constraint_mode,
                max_long_weight,
                max_short_weight,
                max_gross_exposure,
                speed["epochs"],
                speed["batch"],
                neural_lr,
                speed["hidden"],
                neural_val_fraction,
                speed["patience"],
                neural_min_delta,
                DEFAULT_SEED,
            )
        detail.insert(0, "Horizon", selected_horizon)
        regime_detail.insert(0, "Horizon", selected_horizon)
        st.session_state["legacy_full_B"] = {
            "summary": summary,
            "detail": detail,
            "wealth": wealth,
            "diagnostics": pd.DataFrame([diagnostics]),
            "t_source_counts": source_counts,
            "regime_detail": regime_detail,
        }
    except Exception as exc:
        st.error(f"Section B failed: {exc}")

if "legacy_full_B" in st.session_state:
    b = st.session_state["legacy_full_B"]
    st.subheader("B1. Performance")
    st.dataframe(
        b["summary"].style.format({
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
        }),
        use_container_width=True,
        hide_index=True,
    )
    wealth_long = b["wealth"].reset_index().melt(id_vars="Realized date", var_name="Method", value_name="Wealth")
    st.plotly_chart(px.line(wealth_long, x="Realized date", y="Wealth", color="Method", title="B cumulative wealth"), use_container_width=True)
    st.subheader("B2. T-selection coverage")
    st.dataframe(b["diagnostics"], use_container_width=True, hide_index=True)
    st.dataframe(b["t_source_counts"], use_container_width=True, hide_index=True)
    with st.expander("B3. Rolling OOS detail"):
        st.dataframe(b["detail"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download B results (.zip)",
        data=_results_zip("B", b),
        file_name=f"legacy_B_{DISPLAY_LABELS[selected_horizon]}.zip",
        mime="application/zip",
        use_container_width=True,
    )


# =============================================================================
# C. Four-horizon research study
# =============================================================================
st.divider()
st.header("C. 1W / 2W / 1M / 3M Research Study")
st.caption(
    "Runs the same true same-frequency OOS framework across all four short holding periods. "
    "Each horizon uses its own approximately 10-year lookback preset."
)

if st.button("Run C: 1W / 2W / 1M / 3M Research Study", type="primary", use_container_width=True):
    if source != "Yahoo Finance":
        st.error("Section C automatic multi-horizon construction requires Yahoo Finance.")
    else:
        try:
            weekly = _download_base(tuple(ticker_list), start_date, "weekly")[ticker_list].dropna()
            monthly = _download_base(tuple(ticker_list), start_date, "monthly")[ticker_list].dropna()
            base_map = {"weekly": weekly, "monthly": monthly}
            summaries, details, coverage, sources = [], [], [], []
            progress = st.progress(0, text="Starting short-horizon Legacy study...")
            for idx, horizon in enumerate(HORIZON_LABELS):
                cfg_h = get_legacy_horizon_preset(horizon)
                hr = _holding_returns(base_map[str(cfg_h["source"])], horizon)
                lookback_h = min(int(cfg_h["lookback"]), len(hr) - 1)
                if lookback_h < max(30, 3 * hr.shape[1]):
                    continue
                progress.progress(int(5 + 90 * idx / len(HORIZON_LABELS)), text=f"Running {DISPLAY_LABELS[horizon]}...")
                summary, detail, wealth, diagnostics, source_counts, regime_detail = _cached_legacy_oos(
                    hr,
                    lookback_h,
                    speed["oos"],
                    int(cfg_h["periods_per_year"]),
                    gamma,
                    rule,
                    speed["m"],
                    beta,
                    speed["steps"],
                    constraint_mode,
                    max_long_weight,
                    max_short_weight,
                    max_gross_exposure,
                    speed["epochs"],
                    speed["batch"],
                    neural_lr,
                    speed["hidden"],
                    neural_val_fraction,
                    speed["patience"],
                    neural_min_delta,
                    int(DEFAULT_SEED + 10000 * idx),
                )
                summary.insert(0, "Horizon", horizon)
                detail.insert(0, "Horizon", horizon)
                source_counts.insert(0, "Horizon", horizon)
                summaries.append(summary)
                details.append(detail)
                sources.append(source_counts)
                coverage.append({"Horizon": horizon, "Lookback": lookback_h, **diagnostics})
            progress.progress(100, text="Short-horizon Legacy study complete.")
            progress.empty()
            if not summaries:
                raise ValueError("No horizon had sufficient data.")
            st.session_state["legacy_full_C"] = {
                "summary": pd.concat(summaries, ignore_index=True),
                "detail": pd.concat(details, ignore_index=True, sort=False),
                "coverage": pd.DataFrame(coverage),
                "t_source_counts": pd.concat(sources, ignore_index=True),
            }
        except Exception as exc:
            st.error(f"Section C failed: {exc}")

if "legacy_full_C" in st.session_state:
    c = st.session_state["legacy_full_C"]
    st.subheader("C1. OOS performance across all four horizons")
    st.dataframe(
        c["summary"].style.format({
            "CAGR": "{:.2%}",
            "Annualized volatility": "{:.2%}",
            "Annualized Sharpe": "{:.3f}",
            "CER / period": "{:.3%}",
            "Average turnover": "{:.2%}",
            "Max drawdown": "{:.2%}",
            "Net CAGR 10bps": "{:.2%}",
            "Net CAGR 25bps": "{:.2%}",
            "Final $10,000": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    winners = c["summary"].loc[c["summary"].groupby("Horizon")["CAGR"].idxmax(), ["Horizon", "Method", "CAGR"]]
    st.markdown("**Gross-CAGR winner by horizon**")
    st.dataframe(winners.style.format({"CAGR": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(c["summary"], x="Horizon", y="CAGR", color="Method", barmode="group", title="C Legacy OOS CAGR by holding period"), use_container_width=True)
    st.subheader("C2. Coverage and T sources")
    st.dataframe(c["coverage"], use_container_width=True, hide_index=True)
    st.dataframe(c["t_source_counts"], use_container_width=True, hide_index=True)
    with st.expander("C3. Rolling OOS detail"):
        st.dataframe(c["detail"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download C results (.zip)",
        data=_results_zip("C", c),
        file_name="legacy_C_1W_2W_1M_3M.zip",
        mime="application/zip",
        use_container_width=True,
    )


# =============================================================================
# D. Current-window estimator comparison
# =============================================================================
st.divider()
st.header(f"D. Current-Window Legacy Estimator Comparison — {DISPLAY_LABELS[selected_horizon]}")
st.caption(
    "Uses the latest selected-horizon estimation window to compare Historical, Monte-Carlo Gaussian Diffusion, "
    "and Learned Neural Diffusion weights and moment estimates."
)

if st.button(f"Run D: Current {DISPLAY_LABELS[selected_horizon]} Snapshot", type="primary", use_container_width=True):
    try:
        selected_returns = _selected_returns()
        lookback_d = min(int(selected_lookback), len(selected_returns))
        if lookback_d < max(30, 3 * selected_returns.shape[1]):
            raise ValueError("Not enough observations for current-window comparison.")
        with st.spinner("Training current-window Legacy estimators..."):
            metrics_d, weights_d, moments_d, metadata_d = _current_window_comparison(
                selected_returns,
                lookback_d,
                gamma,
                rule,
                speed["m"],
                beta,
                speed["steps"],
                constraint_mode,
                max_long_weight,
                max_short_weight,
                max_gross_exposure,
                speed["epochs"],
                speed["batch"],
                neural_lr,
                speed["hidden"],
                neural_val_fraction,
                speed["patience"],
                neural_min_delta,
                DEFAULT_SEED,
            )
        st.session_state["legacy_full_D"] = {
            "metrics": metrics_d,
            "weights": weights_d,
            "moments": moments_d,
            "metadata": pd.DataFrame([metadata_d]),
        }
    except Exception as exc:
        st.error(f"Section D failed: {exc}")

if "legacy_full_D" in st.session_state:
    d = st.session_state["legacy_full_D"]
    meta_row = d["metadata"].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected T", f"{float(meta_row['T']):.4f}")
    m2.metric("T source", str(meta_row["T source"]))
    m3.metric("b*/n", f"{float(meta_row['b*/n']):.4f}" if pd.notna(meta_row["b*/n"]) else "n/a")
    m4.metric("s_T²", f"{float(meta_row['s_T^2']):.4f}")
    st.subheader("D1. Estimator-based metrics")
    st.dataframe(d["metrics"].style.format({"Expected return": "{:.3%}", "Volatility": "{:.3%}", "CER": "{:.3%}", "Historical benchmark CER": "{:.3%}"}), use_container_width=True, hide_index=True)
    st.subheader("D2. Portfolio weights")
    st.plotly_chart(px.bar(d["weights"], x="Asset", y="Weight", color="Estimator", barmode="group", title="Current-window Legacy weights"), use_container_width=True)
    st.dataframe(d["weights"].style.format({"Weight": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.subheader("D3. Moment diagnostics")
    st.dataframe(d["moments"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download D results (.zip)",
        data=_results_zip("D", d),
        file_name=f"legacy_D_{DISPLAY_LABELS[selected_horizon]}.zip",
        mime="application/zip",
        use_container_width=True,
    )


# =============================================================================
# E. Diffusion-intensity research
# =============================================================================
st.divider()
st.header("E. Diffusion Intensity vs Neural OOS Value")
st.caption(
    "Short-horizon version of the intensity study. It relates T and b*/n to Neural-minus-Historical OOS value "
    "using a one-year rolling CER window appropriate to each holding frequency."
)

available_sources = []
if "legacy_full_C" in st.session_state:
    available_sources.append("Section C — all four horizons")
if "legacy_full_B" in st.session_state:
    available_sources.append("Section B — selected horizon")

if not available_sources:
    st.info("Run Section B or C first. Section E uses their true same-frequency OOS returns.")
else:
    e_source = st.selectbox("E data source", available_sources, key="legacy_full_E_source")
    if st.button("Run E: Diffusion-Intensity Study", type="primary", use_container_width=True):
        if e_source.startswith("Section C"):
            detail_e = st.session_state["legacy_full_C"]["detail"].copy()
        else:
            detail_e = st.session_state["legacy_full_B"]["detail"].copy()
        intensity = _build_intensity(detail_e, gamma=float(gamma))
        if intensity.empty:
            st.error("Could not build Section E intensity dataset.")
        else:
            summary_e = (
                intensity.groupby("Horizon", as_index=False)
                .agg(
                    Mean_T=("T", "mean"),
                    T_positive_fraction=("T", lambda s: float(np.mean(np.asarray(s) > 0))),
                    Mean_bstar_n=("b*/n", "mean"),
                    Mean_neural_excess=("Neural - Historical", "mean"),
                    Mean_rolling_CER_gain=("Rolling CER Neural - Historical", "mean"),
                )
            )
            valid = intensity[["T", "Rolling CER Neural - Historical"]].dropna()
            if len(valid) >= 6 and valid["T"].nunique() >= 3:
                X = np.column_stack([np.ones(len(valid)), valid["T"], valid["T"] ** 2])
                coef = np.linalg.lstsq(X, valid["Rolling CER Neural - Historical"], rcond=None)[0]
                t_opt = -coef[1] / (2.0 * coef[2]) if coef[2] < 0 else np.nan
                quad = pd.DataFrame({"Term": ["Intercept", "T", "T^2"], "Coefficient": coef})
                quad["Empirical T optimum"] = t_opt
            else:
                quad = pd.DataFrame()
            st.session_state["legacy_full_E"] = {
                "intensity": intensity,
                "horizon_summary": summary_e,
                "quadratic_fit": quad,
            }

if "legacy_full_E" in st.session_state:
    e = st.session_state["legacy_full_E"]
    st.subheader("E1. Horizon-level diffusion intensity and OOS value")
    st.dataframe(
        e["horizon_summary"].style.format({
            "Mean_T": "{:.4f}",
            "T_positive_fraction": "{:.1%}",
            "Mean_bstar_n": "{:.4f}",
            "Mean_neural_excess": "{:.3%}",
            "Mean_rolling_CER_gain": "{:.3%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    valid_plot = e["intensity"].dropna(subset=["Rolling CER Neural - Historical"])
    if not valid_plot.empty:
        fig_t = px.scatter(valid_plot, x="T", y="Rolling CER Neural - Historical", color="Horizon", hover_data=["Realized date", "b*/n", "T source"], title="E: T vs rolling Neural CER gain")
        fig_t.add_hline(y=0.0, line_dash="dash")
        st.plotly_chart(fig_t, use_container_width=True)
        fig_b = px.scatter(valid_plot, x="b*/n", y="Rolling CER Neural - Historical", color="Horizon", hover_data=["Realized date", "T", "T source"], title="E: b*/n vs rolling Neural CER gain")
        fig_b.add_hline(y=0.0, line_dash="dash")
        fig_b.add_vline(x=1.0, line_dash="dash")
        st.plotly_chart(fig_b, use_container_width=True)
    if not e["quadratic_fit"].empty:
        st.subheader("E2. Quadratic T fit")
        st.dataframe(e["quadratic_fit"], use_container_width=True, hide_index=True)
    with st.expander("E3. Rolling intensity data"):
        st.dataframe(e["intensity"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download E results (.zip)",
        data=_results_zip("E", e),
        file_name="legacy_E_short_horizon_intensity.zip",
        mime="application/zip",
        use_container_width=True,
    )
