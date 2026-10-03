from streamlit.testing.v1 import AppTest
from pathlib import Path
import numpy as np
import pandas as pd
from core.short_horizon_portfolio import get_horizon_preset, replay_latest_recommendation
from core.feasible_tuning import moments
from core.portfolio_rules import compute_weights

PAGE = Path(__file__).resolve().parents[1] / "views" / "Feasible_Tuning.py"
NAVIGATION_APP = f"""
from pathlib import Path
import streamlit as st
if st.sidebar.radio("Navigation", ["Feasible Tuning", "Other"], key="test_nav") == "Feasible Tuning":
    path = Path({str(PAGE)!r})
    exec(compile(path.read_text(), str(path), "exec"), globals(), globals())
else:
    st.write("Another tab")
"""


def _field(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def _returns():
    return pd.DataFrame(
        np.random.default_rng(42).normal(0.001, 0.02, (280, 5)),
        index=pd.date_range("2018-01-05", periods=280, freq="W-FRI"),
        columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    )


def _leave_and_return(app):
    app.sidebar.radio[0].set_value("Other").run()
    assert not app.exception
    assert not app.number_input
    app.sidebar.radio[0].set_value("Feasible Tuning").run(timeout=30)
    assert not app.exception


def test_feasible_tuning_page_and_gaussian_example():
    app = AppTest.from_file(PAGE).run()
    assert not app.exception
    app.radio[0].set_value("Gaussian theorem check").run()
    assert not app.exception
    assert app.dataframe[0].value.set_index("Method").loc["Same-sample ratio", "K2"] == -0.375
    assert "Pilot ratio" not in app.dataframe[0].value["Method"].values
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.dataframe) == 2


def test_historical_controls_and_results_without_live_download():
    app = AppTest.from_file(PAGE).run()
    tickers = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN")
    returns = pd.DataFrame(
        np.random.default_rng(42).normal(0.001, 0.02, (280, 5)),
        index=pd.date_range("2018-01-05", periods=280, freq="W-FRI"),
        columns=tickers,
    )
    app.session_state["feasible_data"] = ((tickers, "2000-01-01", "1 week"), returns)
    app.run()
    assert not app.exception
    allocation = next(item for item in app.radio if item.label == "Allocation")
    allocation.set_value("Long-only with cash (empirical comparison)").run()
    window = next(item for item in app.number_input if item.label == "Main estimation window")
    window.set_value(120).run()
    run = next(item for item in app.button if item.label == "Run feasible-tuning backtest")
    run.click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 5
    assert "Trace tuning" in app.dataframe[0].value["Method"].values
    assert list(app.dataframe[0].value["Method"]).count("Classical MV") == 1
    assert "Classical MV + turnover control" not in app.dataframe[0].value["Method"].values
    raw = app.dataframe[1].value.set_index("Method").loc["Classical MV"]
    final = app.dataframe[2].value.set_index("Method").loc["Classical MV"]
    np.testing.assert_allclose(raw.to_numpy(dtype=float), final.to_numpy(dtype=float))
    assert (final.to_numpy(dtype=float) >= 0).all()
    assert (final.to_numpy(dtype=float) <= .4 + 1e-7).all()
    assert "Backtest / strategy replay start date" in [item.label for item in app.text_input]
    assert "Pilot ratio" not in app.dataframe[0].value["Method"].values


def test_comparison_controls_reuse_saved_portfolio_settings():
    app = AppTest.from_file(PAGE)
    frame = pd.DataFrame(np.random.default_rng(3).normal(0.001, 0.02, (530, 5)),
                         index=pd.date_range("2015-01-02", periods=530, freq="W-FRI"),
                         columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    snapshot = dict(
        full_returns=frame, holding_period="1 week", lookback=520, gamma=4.0,
        max_long_weight=0.4, turnover_penalty=0.0015, rebalance_alpha=0.6,
        m=100, beta=1.5, n_steps=10, replay_start="2024-01-01",
    )
    cfg = get_horizon_preset("1 week")
    rec = replay_latest_recommendation(frame, cfg, gamma=4.0, m=100, beta=1.5,
                                       n_steps=10, turnover_penalty=.0015,
                                       rebalance_alpha=.6, max_long_weight=.4,
                                       replay_start="2024-01-01")
    u, h = moments(frame.iloc[-520:].to_numpy())
    snapshot.update(weights=rec["weights"], validation_config=cfg,
                    classical_weights=compute_weights(
                        "Mean-Variance", u, h, gamma=4, constraint_mode="Long-only", max_long_weight=.4),
                    classical_rule="allocation-aware-classical-v2")
    app.session_state["shared_current_window"] = snapshot
    app.run()
    assert not app.exception
    fields = {item.label: item.value for item in app.number_input}
    assert fields["Main estimation window"] == 520
    assert fields["Risk aversion γ"] == 4.0
    assert fields["Turnover penalty (bps)"] == 15.0
    assert fields["Rebalance step (%)"] == 60.0
    assert fields["Old-method synthetic-equivalent M"] == 100
    assert fields["Constant β"] == 1.5
    assert fields["Old-method reverse SDE steps"] == 10
    assert _field(app, "text_input", "Backtest / strategy replay start date").value == "2024-01-01"
    assert _field(app, "number_input", "Main estimation window").disabled
    assert not any("Pilot" in item.label for item in app.checkbox)
    run = next(item for item in app.button if item.label == "Run feasible-tuning backtest")
    run.click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 7
    assert any("both match the saved Portfolio" in item.value for item in app.success)
    assert "Old Portfolio (Best-T + turnover control)" in app.dataframe[2].value["Method"].values
    assert "Pilot ratio" not in app.dataframe[2].value["Method"].values


def test_historical_results_survive_navigation_and_new_portfolio_snapshot():
    app = AppTest.from_string(NAVIGATION_APP)
    returns = _returns()
    app.session_state["feasible_data"] = ((tuple(returns.columns), "2000-01-01", "1 week"), returns)
    app.run()
    _field(app, "radio", "Allocation").set_value("Long-only with cash (empirical comparison)").run()
    _field(app, "number_input", "Main estimation window").set_value(120).run()
    _field(app, "number_input", "Risk aversion γ").set_value(2.4).run()
    _field(app, "number_input", "Trace c (0 < c ≤ 4)").set_value(3.0).run()
    _field(app, "number_input", "Trading cost (bps per turnover)").set_value(10.0).run()
    _field(app, "button", "Run feasible-tuning backtest").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    original = [item.value.copy() for item in app.dataframe]
    assert len(original) == 5
    app.sidebar.radio[0].set_value("Other").run()
    assert not app.number_input
    # The Portfolio page can create a snapshot while Feasible Tuning is absent.
    app.session_state["shared_current_window"] = dict(
        full_returns=returns * 2, holding_period="1 week", lookback=520, gamma=4.0,
    )
    app.sidebar.radio[0].set_value("Feasible Tuning").run(timeout=30)
    assert not app.exception
    assert not app.error
    assert not _field(app, "checkbox", "Use Portfolio tab's saved data and settings").value
    assert _field(app, "number_input", "Main estimation window").value == 120
    assert _field(app, "number_input", "Risk aversion γ").value == 2.4
    assert _field(app, "number_input", "Trace c (0 < c ≤ 4)").value == 3.0
    assert _field(app, "number_input", "Trading cost (bps per turnover)").value == 10.0
    assert len(app.dataframe) == len(original)
    for before, after in zip(original, app.dataframe):
        pd.testing.assert_frame_equal(before, after.value)


def test_gaussian_results_survive_navigation_without_rerunning():
    app = AppTest.from_string(NAVIGATION_APP).run()
    _field(app, "radio", "Experiment").set_value("Gaussian theorem check").run()
    _field(app, "number_input", "Population mean per asset").set_value(0.3).run()
    _field(app, "number_input", "Monte Carlo repetitions").set_value(100).run()
    _field(app, "button", "Run Gaussian check").click().run(timeout=30)
    assert not app.exception
    original = app.dataframe[1].value.copy()
    _leave_and_return(app)
    assert _field(app, "radio", "Experiment").value == "Gaussian theorem check"
    assert _field(app, "number_input", "Population mean per asset").value == 0.3
    assert _field(app, "number_input", "Monte Carlo repetitions").value == 100
    assert len(app.dataframe) == 2
    pd.testing.assert_frame_equal(original, app.dataframe[1].value)
    # Changing a parameter still requires a fresh check rather than showing stale output.
    _field(app, "number_input", "Population mean per asset").set_value(0.4).run()
    assert len(app.dataframe) == 1


def test_uploaded_data_and_results_survive_navigation():
    app = AppTest.from_string(NAVIGATION_APP)
    app.session_state["feasible_uploaded_returns"] = ("returns.csv", _returns())
    app.run()
    _field(app, "radio", "Data source").set_value("Upload CSV").run()
    _field(app, "radio", "Allocation").set_value("Long-only with cash (empirical comparison)").run()
    _field(app, "number_input", "Main estimation window").set_value(120).run()
    _field(app, "button", "Run feasible-tuning backtest").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    original = app.dataframe[0].value.copy()
    _leave_and_return(app)
    assert _field(app, "radio", "Data source").value == "Upload CSV"
    assert any("returns.csv" in item.value for item in app.caption)
    assert len(app.dataframe) == 5
    pd.testing.assert_frame_equal(original, app.dataframe[0].value)


def test_shared_snapshot_refresh_overrides_stale_common_inputs():
    app = AppTest.from_string(NAVIGATION_APP)
    frame = pd.DataFrame(np.random.default_rng(3).normal(.001, .02, (530, 5)),
                         index=pd.date_range("2015-01-02", periods=530, freq="W-FRI"),
                         columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    snapshot = dict(full_returns=frame, holding_period="1 week", lookback=520,
                    gamma=4.0, max_long_weight=.4, n_steps=10, replay_start="2024-01-01")
    app.session_state["shared_current_window"] = snapshot
    app.run()
    app.sidebar.radio[0].set_value("Other").run()
    app.session_state["shared_current_window"] = dict(snapshot, gamma=2.0, replay_start="2023-01-01")
    app.sidebar.radio[0].set_value("Feasible Tuning").run()
    assert not app.exception
    assert _field(app, "number_input", "Risk aversion γ").value == 2.0
    assert _field(app, "text_input", "Backtest / strategy replay start date").value == "2023-01-01"
    assert _field(app, "radio", "Allocation").value == "Turnover-controlled comparison with old Portfolio"


def test_epsilon_experiment_survives_navigation_and_leaves_main_setting_unchanged():
    app = AppTest.from_string(NAVIGATION_APP)
    returns = _returns()
    app.session_state["feasible_data"] = ((tuple(returns.columns), "2000-01-01", "1 week"), returns)
    app.run()
    _field(app, "radio", "Allocation").set_value("Long-only with cash (empirical comparison)").run()
    _field(app, "number_input", "Main estimation window").set_value(120).run()
    _field(app, "text_input", "Positive ε candidates (maximum 12)").set_value("0.001,0.25").run()
    _field(app, "text_input", "Final evaluation starts").set_value(str(returns.index[220].date())).run()
    _field(app, "button", "Run ε calibration and final evaluation").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 2
    original = [item.value.copy() for item in app.dataframe]
    assert _field(app, "number_input", "Trace ε").value == .25
    _leave_and_return(app)
    assert len(app.dataframe) == 2
    for before, after in zip(original, app.dataframe):
        pd.testing.assert_frame_equal(before, after.value)


def test_live_upgrade_refreshes_stale_feasible_module_once():
    import core.feasible_tuning as module
    # Simulate the pre-upgrade module retained by the manual Streamlit router.
    original = module.epsilon_sensitivity
    del module.epsilon_sensitivity
    try:
        app = AppTest.from_file(PAGE).run()
        assert not app.exception
        assert callable(module.epsilon_sensitivity)
        refreshed_class = module.TuningSettings
        app.run()
        assert not app.exception
        assert module.TuningSettings is refreshed_class
    finally:
        if not hasattr(module, "epsilon_sensitivity"):
            module.epsilon_sensitivity = original


def test_c_calibration_and_joint_results_survive_navigation():
    app = AppTest.from_string(NAVIGATION_APP)
    returns = _returns()
    app.session_state["feasible_data"] = ((tuple(returns.columns), "2000-01-01", "1 week"), returns)
    app.run()
    _field(app, "radio", "Allocation").set_value("Long-only with cash (empirical comparison)").run()
    _field(app, "number_input", "Main estimation window").set_value(120).run()
    assert _field(app, "text_input", "c calibration starts").value == "2000-01-01"
    from datetime import datetime
    from zoneinfo import ZoneInfo
    assert _field(app, "text_input", "c calibration / evaluation end date").value == datetime.now(ZoneInfo("America/Havana")).date().isoformat()
    _field(app, "text_input", "c candidates (0 < c ≤ 4; maximum 12)").set_value("1,4").run()
    _field(app, "text_input", "c final evaluation starts").set_value(str(returns.index[220].date())).run()
    _field(app, "button", "Run c calibration and final evaluation").click().run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 2
    assert app.dataframe[0].value["ε"].eq(.25).all()
    original = app.dataframe[0].value.copy()
    _leave_and_return(app)
    pd.testing.assert_frame_equal(original, app.dataframe[0].value)
    assert _field(app, "number_input", "Trace c (0 < c ≤ 4)").value == 4
    _field(app, "radio", "Trace calibration mode").set_value("Joint c and ε").run()
    assert not app.dataframe
    _field(app, "text_input", "Joint ε candidates (maximum 12)").set_value(".01,.25").run()
    _field(app, "button", "Run c calibration and final evaluation").click().run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 4
    final = app.dataframe[1].value.set_index("Method")
    calibration = app.dataframe[0].value
    winner = calibration.loc[calibration["Annualized MV excess"].idxmax()]
    assert final.loc["Trace tuning", "Selected c"] == winner["c"]
    assert final.loc["Trace tuning", "Selected ε"] == winner["ε"]
    assert pd.isna(final.loc["Classical MV", "Selected c"])
    assert any("Selected calibration settings used below" in item.value for item in app.success)
    _field(app, "text_input", "c candidates (0 < c ≤ 4; maximum 12)").set_value("5").run()
    _field(app, "button", "Run c calibration and final evaluation").click().run(timeout=30)
    assert app.error and not app.exception
