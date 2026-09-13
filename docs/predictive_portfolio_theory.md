# Predictive means with diffusion portfolio risk

## What this change implements

The Streamlit Portfolio page now offers Original diffusion mean, OLS forecast,
and Diffusion forecast. Both forecasting options replace only the expected-return
vector in the existing turnover-aware, capped, fully invested optimizer. The
original return-based diffusion covariance and its selected portfolio T are shared.
The Method Comparison page optionally compares all three with identical controls
and independent historical holdings, reporting prediction MSE and net portfolio
metrics. Each outer forecast uses only its preceding lookback window.

Per asset, predictors at origin u are the last holding-period return, the trailing
three-period arithmetic mean and population standard deviation. The response is
the next non-overlapping holding-period return. All four variables are expressed
in fixed percentage-point units internally; no training mean is removed before
fitting the zero-mean diffusion model. Outputs return to decimal units. This
origin/scale convention is part of the model, not an invariant transformation.
Inputs are simple returns, matching the existing app, with risk-free rate assumed
zero. These are not historical excess returns against an observed risk-free series.
Historical price data and a user-selected universe can still have revision and
survivorship limitations; chronological fitting does not remove those limitations.

Joint moments use 1/n normalization. Exact affine terminal moments give the
regression coefficients. The stable eigenvalue formula for the terminal covariance
is lambda*(1-a)*(1-a+2*a*lambda)/(1-a+a*lambda)^2. a=0 is OLS. Forecast b is chosen
from {0, .25, 1, 4}, with a=b/n, by mean squared error across assets on up to 12
recent one-step validation origins, refitting at each origin. Ties favor lower b.
This grid is a practical choice, not an oracle guarantee. Forecast b is distinct
from the original portfolio T (whose implementation treats T=0 as a bypass).
At least 12 training pairs are required; without validation history the forecast
uses OLS. A singular or numerically rank-deficient joint covariance explicitly
falls back to least-squares OLS, with per-asset diagnostics.

The existing portfolio covariance propagates finite-step Euler moments and mixes
real/synthetic moments. It is not replaced by the regression joint covariance.

## Proposition: exact utility comparison for any selected portfolios

Condition on information F_t available when weights are selected. Let future
excess returns have true conditional mean mu_t and positive definite conditional
covariance C_t. Let gamma>0. Define

    U_t(w) = w' mu_t - (gamma/2) w' C_t w.

For any F_t-measurable original and combined weights w0 and w1, set d=w1-w0.
Then exactly

    U_t(w1)-U_t(w0)
      = d' (mu_t-gamma C_t w0) - (gamma/2) d' C_t d.       (1)

Proof: substitute w1=w0+d into U_t and expand the quadratic. Symmetry of C_t
combines the two cross terms. This identity holds for constrained weights and
partial rebalancing too; it does not assert that either weight is optimal.
If actual transaction costs are c0 and c1, subtract c1-c0 from (1). An optimizer
penalty is not itself proof of actual execution cost.

Thus improvement is equivalent to the first term exceeding the quadratic term
(and the additional actual cost, when included). mu_t and C_t are unknown, so
this identity is not a directly observable certificate from training estimates.

## Corollary: same estimated covariance, unconstrained portfolios

Let S_t be the common positive definite covariance used by the optimizer,
H_t=S_t^{-1}, and let q0 and q1 denote original and forecast mean inputs. Set
w_i=H_t q_i/gamma. Define

    L_t(q) = (H_t q-C_t^{-1}mu_t)' C_t (H_t q-C_t^{-1}mu_t).

Completing the square around w*=C_t^{-1}mu_t/gamma gives

    U_t(w*)-U_t(w_i) = L_t(q_i)/(2 gamma),
    U_t(w1)-U_t(w0) = [L_t(q0)-L_t(q1)]/(2 gamma).        (2)

If expectations are finite, the combined portfolio improves in expected utility
if and only if E[L_t(q1)] < E[L_t(q0)]. This remains valid when S_t is random and
estimated from the same data as the forecasts: all quantities are conditioned on
F_t first. Independence between covariance and mean estimates is not assumed.

## Oracle covariance special case

When S_t=C_t, equation (2) simplifies to

    E[U_t(w1)-U_t(w0)]
      = ( E[(q0-mu_t)' C_t^{-1}(q0-mu_t)]
         -E[(q1-mu_t)' C_t^{-1}(q1-mu_t)] )/(2 gamma).   (3)

This proves a sufficient and necessary improvement condition: lower expected
inverse-covariance-weighted mean error. Ordinary per-asset forecast MSE is a
different loss, and lacks the cross-asset error terms in (3).

Counterexample: take gamma=1, mu=(0,0), C=S=diag(.01,1), q0=(0,1),
and q1=(.2,0). Squared mean error improves from 1 to .04, but weighted error
increases from 1 to 4. Utility falls from -.5 to -2. Even oracle covariance
therefore cannot turn arbitrary ordinary-MSE improvement into portfolio dominance.
This counterexample concerns the unconstrained formula, not the app's capped rule.

## What the existing papers do and do not establish

The regression paper compares a scalar diffusion regression predictor with OLS
under its Gaussian i.i.d. model and an oracle asymptotic scaling. It does not
compare conditional forecasts with the portfolio paper's original diffusion mean.
The portfolio paper studies its own coupled mean/covariance estimator and utility
expansion. Substituting a new mean changes that estimator. Neither theorem proves
(2) or (3)'s required error ordering for this combined estimator. Also, validation
selection of b, time dependence, scale choices, fallback fits and portfolio
constraints are additional practical changes outside those oracle statements.

For the actual app, equation (1), including actual costs, is the relevant identity.
A dominance theorem would need additional assumptions and a proof of the resulting
expected inequality. This implementation instead provides a chronological
empirical comparison, without claiming guaranteed portfolio or profit improvement.

## Validation scope

Tests check exact regression against OLS and direct terminal-moment formulas,
point-in-time pairing and validation fits, preservation of the risk matrix,
chronological OOS invariance to future data, equal cap/rebalance rules, and the
utility identities/counterexample. These are correctness checks, not real-market
performance evidence. Production mobile defaults remain the original mean; the
new UI is introduced in Streamlit for evaluation before a mobile rollout.
