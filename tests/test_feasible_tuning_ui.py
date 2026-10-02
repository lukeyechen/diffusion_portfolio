from streamlit.testing.v1 import AppTest
import numpy as np
import pandas as pd


def test_feasible_tuning_page_and_gaussian_example():
    app = AppTest.from_file("views/Feasible_Tuning.py").run()
    assert not app.exception
    app.radio[0].set_value("Gaussian theorem check").run()
    assert not app.exception
    assert app.dataframe[0].value.set_index("Method").loc["Same-sample ratio", "K2"] == -0.375
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.dataframe) == 2


def test_historical_controls_and_results_without_live_download():
    app = AppTest.from_file("views/Feasible_Tuning.py").run()
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
    run = next(item for item in app.button if item.label == "Run feasible-tuning backtest")
    run.click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 4
    assert "Trace tuning" in app.dataframe[0].value["Method"].values
    assert "Backtest start date" in [item.label for item in app.text_input]
