from streamlit.testing.v1 import AppTest
from pathlib import Path
import numpy as np
import pandas as pd

PAGE = Path(__file__).resolve().parents[1] / "views" / "Feasible_Tuning.py"


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
    assert len(app.dataframe) == 4
    assert "Trace tuning" in app.dataframe[0].value["Method"].values
    assert "Backtest / strategy replay start date" in [item.label for item in app.text_input]
    assert "Pilot ratio" not in app.dataframe[0].value["Method"].values


def test_comparison_controls_reuse_saved_portfolio_settings():
    app = AppTest.from_file(PAGE)
    frame = pd.DataFrame(np.random.default_rng(3).normal(0.001, 0.02, (530, 5)),
                         index=pd.date_range("2015-01-02", periods=530, freq="W-FRI"),
                         columns=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    app.session_state["shared_current_window"] = dict(
        full_returns=frame, holding_period="1 week", lookback=520, gamma=4.0,
        max_long_weight=0.4, turnover_penalty=0.0015, rebalance_alpha=0.6,
        m=100, beta=1.5, n_steps=50,
    )
    app.run()
    assert not app.exception
    fields = {item.label: item.value for item in app.number_input}
    assert fields["Main estimation window"] == 520
    assert fields["Risk aversion γ"] == 4.0
    assert fields["Turnover penalty (bps)"] == 15.0
    assert fields["Rebalance step (%)"] == 60.0
    assert fields["Old-method synthetic-equivalent M"] == 100
    assert fields["Constant β"] == 1.5
    assert fields["Old-method reverse SDE steps"] == 50
    assert not any("Pilot" in item.label for item in app.checkbox)
    steps = next(item for item in app.number_input if item.label == "Old-method reverse SDE steps")
    steps.set_value(10).run()
    run = next(item for item in app.button if item.label == "Run feasible-tuning backtest")
    run.click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 5
    assert "Old Portfolio (Best-T + turnover control)" in app.dataframe[2].value["Method"].values
    assert "Pilot ratio" not in app.dataframe[2].value["Method"].values
