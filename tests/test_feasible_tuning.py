import unittest

import numpy as np
import pandas as pd

from core.feasible_tuning import (TuningSettings, coefficients, endpoint_moments,
                                 fit_portfolios, historical_backtest, moments)


class FeasibleTuningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.sample = rng.normal(0.02, 0.1, (40, 2))
        self.pilot = rng.normal(-0.01, 0.2, (30, 2))
        self.settings = TuningSettings()

    def test_negative_ratio_example(self):
        result = coefficients(np.array([0.5]), np.eye(1),
                              TuningSettings(gamma=1)).set_index("Method")
        self.assertAlmostEqual(result.loc["Fixed b", "K2"], 49/8)
        self.assertAlmostEqual(result.loc["Same-sample ratio", "K2"], -3/8)
        self.assertAlmostEqual(result.loc["Pilot ratio", "K2"], 49/8)
        self.assertAlmostEqual(result.loc["Trace tuning", "K2"], 28/5)

    def test_zero_mean_trace_is_positive(self):
        result = coefficients(np.zeros(2), np.eye(2), self.settings).set_index("Method")
        self.assertGreater(result.loc["Trace tuning", "K2"], 0)

    def test_isotropic_exact_endpoint_and_ols_limit(self):
        u = np.array([0.2, -0.1])
        mean, risk = endpoint_moments(u, np.eye(2), 0.2)
        np.testing.assert_allclose(mean, 0.8*u)
        np.testing.assert_allclose(risk, 0.96*np.eye(2))
        np.testing.assert_allclose(np.linalg.solve(risk, mean), u/1.2)
        mean0, risk0 = endpoint_moments(u, np.eye(2), 0)
        np.testing.assert_allclose(mean0, u)
        np.testing.assert_allclose(risk0, np.eye(2))

    def test_mle_covariance_and_singular_rejection(self):
        u, h = moments(self.sample)
        np.testing.assert_allclose(h, np.cov(self.sample, rowvar=False, bias=True))
        with self.assertRaises(ValueError):
            moments(np.ones((20, 2)))

    def test_pilot_does_not_change_main_estimators(self):
        base, _ = fit_portfolios(self.sample, self.settings)
        combined, _ = fit_portfolios(self.sample, self.settings, pilot=self.pilot)
        for method in base:
            np.testing.assert_allclose(base[method], combined[method])
        self.assertIn("Pilot ratio", combined)
        self.assertIn("Classical (main + pilot)", combined)

    def test_trace_cap_and_constrained_weights(self):
        settings = TuningSettings(epsilon=0.00001, a_max=0.3)
        portfolios, diagnostic = fit_portfolios(self.sample, settings, cap=0.4)
        trace = diagnostic.set_index("Method").loc["Trace tuning"]
        self.assertTrue(trace["a cap active"])
        self.assertAlmostEqual(trace["a"], 0.3)
        for w in portfolios.values():
            self.assertTrue(np.all(w >= -1e-10))
            self.assertTrue(np.all(w <= 0.4+1e-10))
            self.assertLessEqual(w.sum(), 1+1e-8)

    def test_historical_weights_are_past_only_and_costs_match(self):
        dates = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
        frame = pd.DataFrame(self.sample, index=dates, columns=["A", "B"])
        original = historical_backtest(frame, self.settings, window=15,
                                       pilot_size=10, cap=0.4, cost_bps=25)
        altered = frame.copy()
        altered.iloc[35:] = 0.5
        other = historical_backtest(altered, self.settings, window=15,
                                    pilot_size=10, cap=0.4, cost_bps=25)
        pd.testing.assert_frame_equal(original[original["Date"] < dates[35]],
                                      other[other["Date"] < dates[35]])
        first = original[original["Date"] == dates[35]]
        changed = other[other["Date"] == dates[35]]
        np.testing.assert_allclose(first[["Weight A", "Weight B"]],
                                   changed[["Weight A", "Weight B"]])
        np.testing.assert_allclose(original["Gross return"]-original["Net return"],
                                   0.0025*original["Turnover"])


if __name__ == "__main__":
    unittest.main()
