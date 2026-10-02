from streamlit.testing.v1 import AppTest


def test_feasible_tuning_page_and_gaussian_example():
    app = AppTest.from_file("views/Feasible_Tuning.py").run()
    assert not app.exception
    app.radio[0].set_value("Gaussian theorem check").run()
    assert not app.exception
    assert app.dataframe[0].value.set_index("Method").loc["Same-sample ratio", "K2"] == -0.375
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.dataframe) == 2
