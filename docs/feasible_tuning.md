# Feasible portfolio tuning

Open **Feasible Tuning** in the Streamlit navigation. This implements the
unconditional portfolio experiment in the independent-pilot, same-sample and
trace-tuning extension. It does not replace the forecast model or validation
grid in **Diffusion OLS**.

## Estimator

For a main block of `n` excess-return observations, use its sample mean `u`
and centered maximum-likelihood covariance `H` (divisor `n`). The classical
benchmark is `solve(H, u) / gamma`. All diffusion comparisons use these same
main moments. A pilot supplies only the tuning constant.

The exact continuous-time proxy endpoint is computed directly:

```math
D=(1-a)I+aH,\qquad m=(1-a)D^{-1}u,
\qquad S=H-a^2H^3D^{-2},\qquad w=S^{-1}m/\gamma.
```

No Euler stepping, finite synthetic sample, covariance ridge or real/synthetic
mixture is used. Singular or poorly conditioned main covariances are rejected.

The compared tuning rules are:

| Rule | Tuning constant |
| --- | --- |
| Fixed b | User-specified fixed constant |
| Same-sample ratio | Bounded `((N+2) * u'u + trace(H)) / (u'Hu)` |
| Pilot ratio | The bounded ratio computed from pilot moments only |
| Trace tuning | `c / (epsilon + trace(H))` |

Each uses `a = min(a_max, b/n)`. The diagnostics display the effective `n*a`
and whether the cap is active. Choose `epsilon > 0`, `0 < c <= 4`,
`0 < a_max < 1` and fixed ratio bounds before examining the main sample.
Returns use decimal units, and rescaling coordinates changes the rule.

## Gaussian theorem check

Known population mean and covariance give true utility
`U(w) = w'mu - gamma/2 * w'Sigma*w`. Each Monte Carlo repetition draws an
independent main and pilot sample of size `n`. Gains are paired with the
classical estimator from the same main sample. The table reports Monte Carlo
standard errors, `n^2`-scaled gains and the population asymptotic coefficient.

With `A = (N+2)||mu||^2 + trace(Sigma)` and `Q = mu'Sigma*mu`, fixed and
independent-pilot coefficients are `(b*A - b^2*Q/2)/gamma`. The same-sample
ratio includes the derivative interaction `Xi/gamma`; at clipping boundaries
the smooth-rule coefficient is marked unavailable. Outside the bounds the
rule is locally constant. The optimal population ratio is undefined at zero
mean; the implemented bounded ratio saturates at its upper bound there.

For trace tuning, `D_epsilon = epsilon + trace(Sigma)` gives

```math
K_{2,\mathrm{tr}}=\frac{1}{\gamma}
\left[\frac{cA}{D_\varepsilon}
+\frac{(2c-c^2/2)Q}{D_\varepsilon^2}\right].
```

The defaults reproduce `N=1, mu=0.5, Sigma=1, gamma=1`: fixed `b=7`
and pilot coefficients `6.125`, same-sample ratio coefficient `-0.375`, and
trace `c=4, epsilon=0.25` coefficient `5.6`.

The main-plus-pilot classical benchmark uses twice as many observations.
The pilot corollary does not establish improvement against this benchmark,
so it has no corresponding `K2` prediction in the table.

## Historical comparison

Enter stock tickers and download completed Yahoo holding-period returns, or
upload a CSV of returns already sampled at the selected interval. Choose
the main window, optional preceding disjoint pilot window and explicit
backtest start. Each allocation uses only returns preceding the evaluated
period. All methods share the same evaluation dates. Real stock-return blocks
need not be independent or Gaussian.

The theorem allocation allows shorts and borrowing; cash is `1-sum(w)`.
The optional long-only comparison separately solves the mean-variance
objective subject to `0 <= w_i <= cap` and `sum(w) <= 1`, with cash as the
remainder. This constrained calculation is not covered by the theorem.

Turnover is half the absolute change across risky assets and cash, measured
against the previous period's drifted gross holdings. The modeled period
cost is `cost_bps * 1e-4 * turnover`; net return subtracts this amount from
gross return. This is a return-deduction cost model. Historical returns,
wealth and mean-variance statistics are empirical comparisons, not estimates
of the theorem's population coefficient. Nonpositive wealth factors are
rejected rather than reported as valid compounded growth.

The guaranteed positive trace coefficient is an asymptotic expected-utility
result under the paper's stated regime. It is not a finite-sample, constrained,
trading-cost or stock-performance guarantee. The trace result also does not
cover constants chosen retrospectively to maximize the backtest.

This change is available for Streamlit testing first. Mobile and PWA clients
continue to use their existing API contract until a separate synchronization.
