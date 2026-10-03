import unittest

import numpy as np
import pandas as pd

from core.feasible_tuning import (OLD_METHOD, TradingComparison, TuningSettings,
                                 coefficients, endpoint_moments, fit_portfolios,
                                 historical_backtest, latest_portfolios, moments)
from core.short_horizon_portfolio import replay_latest_recommendation


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


    def test_old_comparator_matches_existing_portfolio_replay(self):
        rng = np.random.default_rng(85)
        frame = pd.DataFrame(rng.normal(0.002, 0.02, (45, 2)),
                             index=pd.date_range("2020-01-03", periods=45, freq="W-FRI"),
                             columns=["A", "B"])
        trading = TradingComparison(inner_folds=2, validation_size=10,
                                    min_train_size=20, cap=0.6, m=10,
                                    n_steps=10, candidate_t=(0.0, 0.5))
        history = historical_backtest(frame, self.settings, window=40,
                                      oos_start="2020-01-01", cost_bps=25, trading=trading)
        latest, diagnostic = latest_portfolios(frame, self.settings, history,
                                               window=40, trading=trading)
        old = replay_latest_recommendation(
            frame, trading.validation_config(40), gamma=self.settings.gamma,
            m=trading.m, beta=trading.beta, n_steps=trading.n_steps,
            candidate_t=list(trading.candidate_t), turnover_penalty=trading.turnover_penalty,
            rebalance_alpha=trading.rebalance_alpha, max_long_weight=trading.cap,
            replay_start="2020-01-01",
        )
        np.testing.assert_allclose(latest[OLD_METHOD], old["weights"], atol=1e-8)
        self.assertAlmostEqual(diagnostic.set_index("Method").loc[OLD_METHOD, "T"], old["T"])
        self.assertNotIn("Pilot ratio", latest)
        for w in latest.values():
            self.assertAlmostEqual(w.sum(), 1)
            self.assertLessEqual(w.max(), trading.cap+1e-9)
        changed = frame.copy()
        changed.iloc[-1] = [0.8, -0.4]
        altered = historical_backtest(changed, self.settings, window=40,
                                      oos_start="2020-01-01", cost_bps=25, trading=trading)
        pd.testing.assert_frame_equal(history[history["Date"] < frame.index[-1]],
                                      altered[altered["Date"] < frame.index[-1]])
        np.testing.assert_allclose(history[history["Date"] == frame.index[-1]][["Weight A", "Weight B"]],
                                   altered[altered["Date"] == frame.index[-1]][["Weight A", "Weight B"]])

    def test_old_comparison_rejects_unmatched_or_infeasible_windows(self):
        with self.assertRaises(ValueError):
            fit_portfolios(self.sample, self.settings, trading=TradingComparison())
        with self.assertRaises(ValueError):
            fit_portfolios(self.sample, self.settings,
                           trading=TradingComparison(inner_folds=2, validation_size=5,
                                                      min_train_size=20, cap=0.4))


if __name__ == "__main__":
    unittest.main()
