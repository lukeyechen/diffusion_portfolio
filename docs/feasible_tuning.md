# Feasible tuning and the old Portfolio comparison

Open **Feasible Tuning** in Streamlit. The displayed tuning rules are **Fixed b**, **Same-sample ratio**, and **Trace tuning**. Pilot ratio and its separate sample controls have been removed from the tab. The mathematical backend retains independent-pilot functions for theorem verification; the interface uses no pilot observations.

## Exact endpoint estimator

For the common main block of n excess-return observations, u is its sample mean and H is its centered maximum-likelihood covariance, with divisor n. The classical theorem estimator is solve(H, u)/gamma. Fixed b is user specified; the same-sample ratio is the bounded ((N+2)u'u + trace(H))/(u'Hu); trace tuning uses c/(epsilon+trace(H)).

Each direct rule uses a=min(a_max,b/n). With constant beta, T=-log(a)/beta; the theorem-only mode uses beta=1. The exact continuous-time proxy moments are

```math
D=(1-a)I+aH,\qquad m=(1-a)D^{-1}u,
\qquad S=H-a^2H^3D^{-2},\qquad w=S^{-1}m/\gamma.
```

The direct moment calculation uses no finite synthetic sample, Euler steps or mixture. Singular or poorly conditioned H is rejected. Set c, epsilon, a_max and ratio bounds before examining the sample. Decimal return units affect the identity-reference estimator.

## Practical comparison with the old Portfolio method

The default historical allocation mode is **Turnover-controlled comparison with old Portfolio**. Its rows are classical, fixed b, same-sample ratio, trace tuning, and **Old Portfolio (Best-T + turnover control)**.

Every row uses the same trailing estimation window, risk aversion, fully invested long-only cap, turnover penalty, partial rebalance step and realized trading-cost rate. The weekly window defaults to the Portfolio preset of 520 observations. The old method needs enough observations for its original nested validation blocks; these requirements are displayed and enforced rather than silently shortening its window.

Every method starts from equal weights on the selected replay start and evolves its own holdings. At each period it uses only the preceding estimation block. The common practical objective includes a penalty for changes from that method's drifted holdings. The common allocation wrapper uses the existing turnover-aware solver, partial rebalance and cap projection. This solver includes the existing numerical covariance ridge; practical allocations are not the theorem's unconstrained weights.

The old comparator calls the existing turnover_controlled_target engine with **Original diffusion mean**, the original Best-T candidate grid, finite-step reverse dynamics and real/synthetic moment mixture. The direct rules keep their continuous endpoint moments and calculate T directly. The old candidate search is repeated at each historical period and can take longer than the new direct rules.

When the Portfolio tab has a saved result, **Use Portfolio tab's saved data and settings** reuses its exact return table, interval, window, gamma, cap, turnover penalty, rebalance fraction, M, beta and step count. Match the replay start as well to reproduce its historical state. If the saved Portfolio used an OLS or diffusion forecast mean, the interface explains that the comparator uses the original unconditional diffusion mean instead.

The weights table displays all methods side by side. The summary and wealth curve share the same evaluated dates. **Trace tuning versus old Portfolio** shows their complete-test performance and differences in total return and annualized mean-variance excess return, in percentage points. Next-period weights continue each method's replayed holdings using the latest available estimation block.

Turnover is half the absolute change in risky assets and cash from the previous drifted gross holdings. Net return deducts cost_bps * 1e-4 * turnover. In the common fully invested comparison the risk-free rate is zero, matching the old engine. Nonpositive wealth factors are rejected.

## Other allocation modes

**Theorem weights (unconstrained)** uses exact inverse-covariance weights without a trading penalty or partial rebalance. Shorts and borrowing are allowed, and cash is 1-sum(w).

**Long-only with cash (empirical comparison)** maximizes the estimated mean-variance objective subject to 0<=w_i<=cap and sum(w)<=1. It includes no allocation turnover penalty or partial rebalancing. The old fully invested strategy is shown only in the matched turnover-controlled comparison.

## Gaussian theorem check

Known population moments let the app evaluate true utility U(w)=w'mu-gamma/2*w'Sigma*w. The displayed simulation uses a single main sample for classical, fixed b, same-sample ratio and trace tuning. The table reports paired utility gains, Monte Carlo standard errors, n^2-scaled gains and population asymptotic coefficients. The old constrained trading strategy is evaluated in historical mode.

Defaults reproduce the supplied one-asset example: N=1, mu=0.5, Sigma=1, gamma=1 gives K2=6.125 for fixed b=7, -0.375 for the bounded same-sample ratio, and 5.6 for trace c=4, epsilon=0.25. At ratio clipping boundaries the smooth-rule coefficient is unavailable.

The positive trace coefficient is an asymptotic expected-utility result under the paper's Gaussian regime. It does not guarantee improvement with constraints, turnover penalties, costs, finite historical samples or retrospectively selected constants. Those practical comparisons must be judged from their historical results.

This feature is tested on Streamlit first. PWA and Android clients retain their existing contract until separate synchronization.
