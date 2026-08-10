# Diffusion Portfolio Lab

A Streamlit research prototype for classical and diffusion-augmented portfolio estimation.

## Run

```bash
cd portfolio_app
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## CSV format

One row per date and one numeric column per asset. Returns should be decimals, so `0.01` means 1%.

## Important

`core/diffusion.py` is a transparent Gaussian reverse-SDE research prototype using the linear score implied by sample moments. Before using the app for real money, replace it with the exact validated diffusion implementation from the research code and add portfolio constraints, transaction costs, authentication, persistent storage, and broker-side safety controls.


## Deploy on Streamlit Community Cloud

1. Push the contents of this folder to a GitHub repository.
2. In Streamlit Community Cloud choose:
   - Repository: your GitHub repository
   - Branch: `main`
   - Main file path: `app.py`
3. If using the URL-style deployment form, use a URL ending in:
   `.../blob/main/app.py`

Version 2 includes safer slider handling for short datasets and clearer errors when Yahoo Finance returns too little history.


## Version 3 practical safeguards

The Portfolio page now includes:
- Long-only constraints by default.
- Limited long-short mode with a gross-exposure cap.
- Research/unconstrained mode for reproducing paper-style weights.
- A minimum sample-size requirement for diffusion: `max(60, 3*N)`.
- Estimation-risk warnings using `N/n`.
- Covariance condition-number diagnostics.
- A default Yahoo history of 2021 onward rather than a very short recent sample.

These safeguards are meant to prevent extremely leveraged portfolios from very small samples.


## Version 4: stabilized diffusion-horizon tuning

The Portfolio page now supports two diffusion-horizon methods:

1. **Validation tuned** (default for real-data use)
   - Chronologically splits the selected historical window into train and validation portions.
   - Tries candidate horizons `T = {0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00}`.
   - Generates diffusion augmentation from the training portion.
   - Builds the chosen constrained portfolio.
   - Selects the horizon with the highest held-out certainty-equivalent return.

2. **Theoretical constrained**
   - Computes `b* = A/Q`.
   - Uses `s_T^2 = min(1, b*/n)`.
   - For constant beta, uses `T = max(0, log(n/b*)/beta)`.
   - If `b*/n >= 1`, the constrained theoretical optimum is `T=0`.

For production research, nested rolling validation should eventually replace the single chronological train/validation split.


## Version 6: additional return intervals

The Portfolio page now supports:
- 1 day
- 1 week
- 1 month
- 3 months
- 6 months
- 1 year

For 3-month, 6-month, and 1-year choices, the app downloads monthly prices and constructs
non-overlapping multi-month returns. This avoids pretending Yahoo directly supports every
requested interval and keeps `n` interpretable as the number of return observations.


## Version 6.2: interval-adjusted lookback and diffusion minimum

The app no longer requires the same raw number of observations for every return interval.

It uses approximately five years of calendar history as the diffusion minimum and ten years
as the preferred lookback, with additional dimension safeguards:

- diffusion minimum: `max(5 * periods_per_year, 3 * N)`
- preferred lookback: `max(10 * periods_per_year, 10 * N)`

Approximate periods per year:
- daily: 252
- weekly: 52
- monthly: 12
- 3-month: 4
- 6-month: 2
- yearly: 1

Example for N=5:
- monthly minimum = max(60, 15) = 60
- 3-month minimum = max(20, 15) = 20
- 6-month minimum = max(10, 15) = 15
- yearly minimum = max(5, 15) = 15

Thus 26 six-month observations for 5 assets now satisfy the diffusion minimum.


## Version 7: learned neural score model

The Portfolio page now offers two diffusion score models:

### Analytical Gaussian
Uses the paper-style Gaussian affine score derived from sample moments:
`score_t(x) = -Sigma_t^{-1}(x - mu_t)`.

### Learned Neural Score
Uses a PyTorch MLP `s_theta(x,t)` trained by denoising score matching:

`x_t = s_t x_0 + sigma_t epsilon`

with score target

`-epsilon / sigma_t`.

The selected diffusion horizon `T` is used as the maximum noising/training time.
After training, the app integrates the reverse VP SDE from `T` to `0`, generates
synthetic return vectors, appends them to the historical sample, and recomputes
the portfolio moments.

Important:
- The current optimal-T theory is proved for the Gaussian/affine-score setting.
- Reusing that T for a nonlinear neural score model is an experimental extension,
  not a theorem-backed optimum.
- Neural training is fragile with very small n. The app warns when n < max(100, 10N).

## Version 8: neural validation and early stopping

The Learned Neural Score mode now includes:
- chronological train/validation split;
- training DSM loss;
- validation DSM loss;
- zero-score DSM baseline;
- percentage improvement over the validation zero-score baseline;
- early stopping with configurable patience and minimum improvement;
- automatic restoration of the best validation-loss model;
- smaller defaults for low-n portfolio problems: hidden width 32, maximum 1000 epochs, batch size 32.

If the learned validation loss does not beat the zero-score baseline, the app warns that the neural score result is unreliable.


## Version 9: explicit Run button + rolling OOS comparison

Nothing below the settings panel runs until the user clicks:

`Run portfolio analysis`

The app now performs a same-date rolling one-step-ahead comparison of:

1. Historical moments
2. Gaussian diffusion augmentation
3. Learned neural-score diffusion augmentation

For each method it reports:
- realized return per period
- volatility per period
- Sharpe ratio per period
- certainty-equivalent return (CER) per period
- average one-way turnover: `0.5 * sum(abs(w_t - w_{t-1}))`
- maximum drawdown
- number of OOS periods

Important implementation details:
- All methods are evaluated on exactly the same OOS dates.
- No future observation enters an estimation window.
- Theoretical constrained T is recomputed inside every rolling window.
- The neural score model is retrained inside every rolling window.
- The OOS-period control defaults to a small value because repeated neural training
  is computationally expensive on Streamlit Cloud.


## Version 10: theoretical T with validation fallback

The rolling OOS comparison no longer aborts when a rolling window has `b*/n >= 1`.

For each rolling estimation window:
1. Compute the constrained theoretical horizon.
2. If `b*/n < 1`, use the theoretical positive T.
3. If `b*/n >= 1`, choose a positive T by held-out validation CER from:
   `{0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00}`.
4. Evaluate Historical, Gaussian Diffusion, and Neural Diffusion on the same next-period return.

The dashboard reports:
- T
- T source (`Theoretical` or `Validation fallback`)
- b*
- b*/n
- s_T^2
- count of theoretical vs fallback windows

The realized OOS observation is never used to choose the fallback T.


## Version 11: three-level horizon fallback

Rolling OOS horizon selection now follows:

1. **Theoretical**
   - Use constrained theoretical T when `b*/n < 1`.

2. **Validation fallback**
   - If theoretical T=0 and the rolling sample is large enough,
     choose T by held-out validation CER.
   - Validation fraction adapts to sample size:
     - n < 30: 10%
     - 30 <= n < 50: 15%
     - n >= 50: 20%

3. **Safe fallback**
   - If the rolling sample is too short for valid T tuning,
     use `T = 0.25`.
   - This keeps all three methods on identical OOS dates instead of
     dropping windows or aborting the comparison.

The dashboard reports counts of Theoretical, Validation fallback,
and Safe fallback windows.


## Version 12: annual-return slider fix

Annual and other long-horizon return intervals can produce very small samples.
The app now avoids constructing Streamlit sliders with equal or invalid min/max values.

If there are too few observations:
- comparison lookback/OOS controls become fixed when possible;
- the rolling OOS comparison is skipped gracefully when fewer than 2 OOS periods remain;
- the main portfolio analysis can still run.


## Version 13: return-horizon study + monthly OOS evaluation

### Return-Horizon Study
The app can automatically download overlapping:
- 1M returns
- 3M returns
- 6M returns
- 12M returns

and report:
- n
- N
- N/n
- b*
- b*/n
- T
- theoretical feasibility

### Monthly OOS evaluation
The model may be estimated from a longer return horizon such as 12M, while the portfolio
is rebalanced monthly and evaluated on the next actual 1M return.

This separates:
- model return horizon
from
- OOS / rebalancing frequency

and produces many more genuine real-market OOS observations than annual-only evaluation.


## Version 14: longer monthly OOS + multi-horizon OOS matrix

Changes:
- Monthly OOS control now defaults to 60 months and supports up to 120 months.
- Monthly OOS no longer silently skips theoretical-T=0 windows:
  it uses the same Theoretical -> Validation fallback -> Safe fallback hierarchy.
- The UI reports requested vs evaluated OOS months and fallback counts.
- Added automatic 3M / 6M / 12M monthly OOS comparison with the same methods:
  Historical, Gaussian Diffusion, Neural Diffusion.
- Added per-horizon OOS coverage diagnostics.
- Run Portfolio Analysis button is styled green.


## Version 15: fast/research horizon study

- Separate green `Run 3M / 6M / 12M Study` button.
- Normal portfolio analysis no longer automatically launches the multi-horizon neural study.
- Fast mode: 100 neural epochs, up to M=200, up to 50 reverse steps, patience up to 30.
- Research mode: uses the full selected neural settings.
- Per-horizon `st.cache_data` caching avoids retraining unchanged 3M/6M/12M runs.
- Progress bar reports 3M, 6M, and 12M completion.


## Version 16: performance split by T source

The monthly OOS tables now separate:
- All windows
- Theoretical-T windows only
- Validation-fallback windows only
- Safe-fallback windows only

This is shown for both the selected horizon and the 3M/6M/12M study.

The old low-frequency rolling OOS comparison is collapsed as a secondary diagnostic
because long return horizons may leave only 2–5 OOS observations.


## Version 17: regime diagnostics for theoretical vs fallback T

The monthly OOS analysis now compares market and estimation conditions across:
- All windows
- Theoretical-T windows only
- Validation-fallback windows only
- Safe-fallback windows only

Reported diagnostics:
- equal-weight market proxy return for the next month
- average estimated asset mean
- cross-sectional dispersion of next-month asset returns
- average estimated asset volatility
- mean-vector norm
- covariance trace
- covariance condition number
- b*
- b*/n
- T

The same regime summary is also available in the 3M / 6M / 12M study.
This helps determine whether theoretical feasibility is associated with a distinct
market/data regime rather than assuming that T itself causes the performance difference.


## Version 18: regime-controlled relative-performance analysis

The selected-horizon monthly OOS section now estimates:

`Diffusion - Historical = alpha
 + beta1 * I(b*/n < 1)
 + beta2 * market return
 + beta3 * cross-sectional dispersion
 + beta4 * covariance trace
 + error`

for both:
- Neural Diffusion minus Historical
- Gaussian Diffusion minus Historical

The table reports OLS coefficients with HC1 heteroskedasticity-robust standard errors,
t-statistics, and p-values.

The app also plots:
- x-axis: `b*/n`
- y-axis: diffusion minus Historical realized monthly return
- vertical line: `b*/n = 1`
- horizontal line: zero relative performance

This helps distinguish a true theoretical-feasibility effect from a favorable market-regime effect.


## Version 19: three regression specifications around b*/n = 1

The regime-controlled OOS analysis now runs three models for both:
- Neural Diffusion minus Historical
- Gaussian Diffusion minus Historical

### 1. Binary feasibility
Uses `I(b*/n < 1)`.

### 2. Continuous b*/n
Uses the continuous ratio `b*/n`.

### 3. Piecewise at 1
Uses:
- `(b*/n - 1)`
- `I(b*/n < 1)`
- `(b*/n - 1) * I(b*/n < 1)`

This allows a possible level shift and different slopes on each side of the theoretical boundary.

All three specifications retain the controls:
- market return
- cross-sectional dispersion
- covariance trace

and report:
- HC1 robust standard errors
- t-statistics
- p-values
- R²
- number of observations


## Version 20: clearer regression-table controls

The regression section now always shows two radio controls:

- Regression specification
  - Piecewise at 1
  - Continuous b*/n
  - Binary feasibility

- Relative strategy
  - Neural - Historical
  - Gaussian - Historical

The default is:
- Piecewise at 1
- Neural - Historical

The selected table is displayed with only:
Variable, Coefficient, Robust SE, t-stat, p-value.

For the piecewise model, the three theoretically important coefficients are also
shown in a separate highlighted table:
- (b*/n - 1)
- I(b*/n < 1)
- (b*/n - 1) × I(b*/n < 1)


## Version 21: direct piecewise slope tests and confidence intervals

The piecewise-at-1 regression now directly reports four HC1-robust linear-combination tests:

- Right-side slope for `b*/n >= 1`
- Left-side slope for `b*/n < 1`
- Slope difference `(left - right)`
- Level jump at `b*/n = 1`

Each row reports:
- estimate
- HC1 robust standard error
- t-statistic
- p-value
- 95% confidence interval

For the piecewise model

`y = a + b1(x-1) + b2 I(x<1) + b3(x-1)I(x<1) + controls + e`

the derived quantities are:
- right slope = `b1`
- left slope = `b1 + b3`
- slope difference = `b3`
- level jump = `b2`


## Version 22: 120-month OOS default + persistent regression controls

### 120-month research default
The Monthly OOS periods control now defaults to 120 months (10 years).
The UI reports whether:
- 120+ months: preferred research target reached
- 60-119 months: acceptable 5-year minimum
- <60 months: sample remains small

### Regression bullets no longer erase the page
The regression specification and relative-strategy controls are now inside a Streamlit form.

Changing a radio button does not immediately rerun the application. After selecting:
- Piecewise at 1 / Continuous b*/n / Binary feasibility
- Neural-Historical / Gaussian-Historical

click `Apply regression view`.

The chosen view is stored in `st.session_state`.

### Main run persistence
After `Run portfolio analysis` is clicked, `analysis_has_run=True` is stored in session state.
Normal widget reruns no longer send the user back to the pre-run screen.
A `Clear results` button explicitly resets the run state.


## Version 23: staged progress for slow monthly OOS / regression analysis

The selected-horizon monthly OOS analysis now shows a progress bar:
- 5% prepare model-horizon returns
- 10% prepare realized monthly returns
- 15% begin rolling OOS / neural retraining
- 85% OOS comparison complete
- 90% build regression dataset
- 94% estimate binary / continuous / piecewise regressions
- 97% piecewise slope tests
- 100% complete

The long middle stage is still the expensive part because neural score models are retrained
inside rolling windows. The percentage is stage-based, not an exact epoch counter.


## Version 24: dedicated Method Comparison page

The Portfolio page is now focused on:
- data and estimation settings,
- estimation-risk diagnostics,
- current recommended portfolio weights,
- portfolio metrics,
- diffusion tuning,
- neural-score training diagnostics.

All cross-method comparison work has been moved to `Method Comparison`:
- 1M / 3M / 6M / 12M Return-Horizon Study
- selected-horizon monthly OOS comparison
- Historical vs Gaussian vs Neural OOS performance
- cumulative OOS wealth
- performance by T source
- market / estimation regime diagnostics
- binary / continuous / piecewise regressions
- direct piecewise slope tests
- b*/n threshold scatter
- 3M / 6M / 12M Research Study
- OOS coverage tables
- multi-horizon CER comparison chart
- current-window estimator / portfolio-rule comparison
- Historical vs Gaussian-Diffusion moments

This prevents the Portfolio page from becoming overloaded and keeps expensive research
comparisons in a separate tab/page.


## Version 25: interval-aware lookback and stale-data protection

Fixed the issue where switching from a long return interval (for example 1 year with
lookback 21) to 1 month could incorrectly preserve the old lookback and/or reuse the
old return dataset.

Changes:
- Yahoo return data now carry a settings signature: tickers, start date, return interval,
  Yahoo interval, and return-horizon months.
- If any of those controls change, stale returns are NOT reused. The app asks the user
  to click Download data to rebuild the correct return series.
- Lookback sliders now have interval-specific Streamlit keys. A 1-year lookback no longer
  carries over to 1-month, 3-month, 6-month, etc.
- Default lookback is interval-adjusted and targets the preferred sample when available:
  monthly -> about 120 observations, 3M -> about 40, 6M -> about 20, annual -> at least
  the dimension-based preferred target subject to available history.
- The Portfolio page displays Current lookback, Diffusion minimum, and Preferred lookback.
- Diffusion still enforces max(5 years of interval observations, 3N); the guard was not weakened.


## Version 26: Neural training modes

The Portfolio page now offers three neural-score training modes.

### Fast
Designed for normal interactive web use:
- max epochs: 100
- hidden width: 64
- learning rate: 1e-3
- batch size: 64
- validation fraction: 20%
- early-stopping patience: 30
- min validation improvement: 1e-3

### Standard
More training/capacity while remaining practical:
- max epochs: 300
- hidden width: 128
- learning rate: 1e-3
- batch size: 64
- validation fraction: 20%
- early-stopping patience: 75
- min validation improvement: 1e-3

### Research
Exposes manual controls for:
- epochs
- hidden width
- learning rate
- batch size
- validation fraction
- patience
- minimum validation improvement

The interval-aware lookback from v25 remains unchanged. Fast mode speeds the neural
training configuration rather than reverting to an undersized monthly sample.


## Version 27: integrated panel, defaults, stop and save

### Portfolio page defaults
- Portfolio rule: Mean-Variance
- Estimator: Diffusion
- Diffusion score model: Learned Neural Score
- Diffusion horizon tuning: Theoretical constrained
- Neural training mode: Fast

### Integrated settings panel
The Portfolio settings are grouped into aligned multi-column rows similar to the
Method Comparison settings panel.

### Stop
Both Portfolio and Method Comparison include Stop controls. The Stop button requests
a Streamlit rerun, clears/cancels the active workflow state, and cooperative checkpoints
stop the workflow between expensive stages/horizons. A single blocking PyTorch operation
cannot be interrupted mid-kernel, but the workflow stops at the next Streamlit stage.

### Save results
Portfolio can prepare a ZIP containing:
- recommended weights
- portfolio metrics
- moment diagnostics
- diffusion tuning
- validation horizon table when available
- neural training loss table
- zero-score baseline table
- interactive HTML neural-loss chart

Method Comparison can prepare a ZIP containing every currently saved comparison table
and the corresponding interactive HTML charts, including horizon, OOS wealth, CER,
and b*/n relative-performance charts.

### Caching
Gaussian and neural diffusion calculations on the Portfolio page are cached by data and
model settings. Clicking Save or changing a display-only control no longer retrains the
same neural model from scratch.


## Version 28: Section D — full Historical / Gaussian / Neural comparison

Section D now compares:
- Historical
- Gaussian Diffusion
- Neural Diffusion

across all available portfolio rules, including:
- Equal Weight
- Mean-Variance
- Ledoit-Wolf Mean-Variance

It now contains four linked views:

### D1. Estimator-Based Portfolio Metrics
Each estimator's portfolio is evaluated using that estimator's own mean/covariance.

### D2. Common Historical-Benchmark Evaluation
The weights from every estimator/rule combination are held fixed, and all portfolios are
evaluated using the exact same historical sample mean and covariance. This isolates differences
caused by the portfolio weights instead of mixing them with differences in estimated moments.

### D3. Portfolio Weight Comparison
Shows asset-by-asset weights for Historical / Gaussian / Neural × portfolio rule.

### D4. Historical vs Gaussian vs Neural Moments
Compares:
- Mean-vector norm ∥μ^diff∥
- Covariance trace tr(Σ^diff)
- Covariance condition number κ(Σ^diff)

The section also reports b*, b*/n, theoretical T, T actually used, T source, and neural
training diagnostics.

Gaussian and Neural current-window calculations are cached to reduce repeated training.


## Version 29: Method Comparison moved to second page + D4 notation fix

### Sidebar/page order
The Streamlit multipage order is now:
1. Portfolio
2. Method Comparison
3. Backtest
4. Diffusion Diagnostics

This is implemented by renaming the page files to:
- `1_Portfolio.py`
- `2_Method_Comparison.py`
- `3_Backtest.py`
- `4_Diffusion_Diagnostics.py`

The Portfolio page link to Method Comparison was also updated.

### D4 moment diagnostics
The D4 table now displays estimator-specific notation:

- Historical:
  - ‖μ̂ᴴ‖
  - tr(Σ̂ᴴ)
  - κ(Σ̂ᴴ)

- Gaussian Diffusion:
  - ‖μ̂ᴳ‖
  - tr(Σ̂ᴳ)
  - κ(Σ̂ᴳ)

- Neural Diffusion:
  - ‖μ̂ᴺ‖
  - tr(Σ̂ᴺ)
  - κ(Σ̂ᴺ)

The underlying numerical calculations are unchanged; this is a display/notation improvement.


## Version 29.3: collision-proof explicit router

This version removes the `pages/` directory from the distributed application.

Page scripts now live in `views/`, and `app.py` registers them explicitly with
`st.Page` and `st.navigation`.

Unique URLs are declared explicitly:
- Portfolio: `/portfolio`
- Method Comparison: `/method-comparison`
- Backtest: `/backtest`
- Diffusion Diagnostics: `/diffusion-diagnostics`

This avoids the Streamlit 1.44+ automatic-router restriction that rejects duplicate
inferred page URL pathnames.

`requirements.txt` now requires `streamlit>=1.45,<2`.

When deploying on Streamlit Cloud, make sure the Main file path is `app.py`.


## Version 29.4: remove Streamlit multipage routing entirely

The deployment log showed:

`StreamlitAPIException: Multiple Pages specified with URL pathname Backtest.`

To eliminate this class of error, v29.4 does not use Streamlit's multipage router at all.

`app.py` provides a sidebar radio with this order:
1. Portfolio
2. Method Comparison
3. Backtest
4. Diffusion Diagnostics

It executes the selected script from `views/`.

There are:
- no `st.Page(...)` calls
- no `st.navigation(...)` calls
- no `pages/` directory in this package
- no `st.switch_page(...)` calls

This makes duplicate page URL-path errors impossible from the packaged application.

Streamlit is pinned to `1.61.1`, matching the deployment environment shown in the user log.


## Version 30: Current-window synchronization bug fix

Fixed the bug where `Portfolio -> Recommended Weights` could differ from
`Method Comparison -> D3 -> Neural Diffusion -> Mean-Variance`.

### Root causes fixed
1. Method Comparison previously trained/generated a second Neural Diffusion model.
2. The Neural comparison used a different random seed (`DEFAULT_SEED + 7001`).
3. Method Comparison had its own lookback, risk, constraint, and neural settings.
4. Therefore the two screens could appear to describe the same model while actually
   using different moments and synthetic samples.

### New shared-current-window contract
After a Portfolio analysis completes, Portfolio stores the exact:
- estimation window
- asset ordering
- historical mean/covariance
- diffusion mean/covariance
- synthetic/combined return sample
- T and b*
- γ
- constraints
- diffusion settings
- neural training settings
- exact recommended weight vector

When Method Comparison D is run for the same model return horizon, it reuses that exact
Portfolio result instead of retraining Neural Diffusion.

For the portfolio rule used on Portfolio (normally Mean-Variance), D3 copies the exact
Portfolio weight vector. Therefore the two displays must match to machine precision.

D also displays a synchronization audit with the maximum absolute weight difference.

If no compatible Portfolio run exists, Method Comparison can still run independently,
but it clearly warns the user and uses the same `DEFAULT_SEED`.


## Version 31: Portfolio results persist across navigation

Fixed the bug where leaving Portfolio for another section and returning caused the
Portfolio controls/results to disappear or revert.

### Why it happened
The app's manual router executes only the selected view. Streamlit cleans up widget state
for widgets that are not rendered during a rerun. Therefore Portfolio's unkeyed controls
could revert to defaults after visiting Method Comparison / Backtest / Diagnostics.

The worst case was Data Source:
- user had Yahoo Finance results
- user navigated away
- Portfolio's source widget was absent
- returning to Portfolio reset it to Upload CSV
- the page no longer reconstructed the same analysis state

### Fix
The last completed Portfolio analysis (`shared_current_window`) is now the permanent
state source for rebuilding Portfolio controls. Returning to Portfolio restores:
- data source
- tickers
- start date
- return interval
- lookback
- γ
- portfolio rule
- constraints / max weights
- estimator
- score model
- T tuning mode
- M, β, reverse SDE steps
- neural training mode
- Research-mode neural parameters

`analysis_has_run` remains in session state, so the results render again when the user
returns. Cached diffusion functions avoid unnecessary retraining when the restored inputs
are unchanged.

The Clear button now also removes `shared_current_window`, so Clear still performs a true reset.


## Version 32: D3 exact Portfolio synchronization — horizon mismatch bug fixed

The remaining D3 mismatch was traced to Section D's synchronization condition.

### Root cause
Section D previously reused Portfolio's exact Neural result only when:

`Portfolio horizon == Method Comparison "Selected model return horizon"`

This was wrong for the purpose of Section D.

Example:
- Portfolio was run on 1-month returns.
- Method Comparison's shared setting remained 12M.
- Section D rejected the saved Portfolio result.
- D3 silently trained/generated an independent Neural Diffusion result.
- Therefore Neural Diffusion × Mean-Variance weights differed from Portfolio Recommended Weights.

### Correct behavior in v32
Section D is now defined as a **current Portfolio-window diagnostic**.

If the latest Portfolio analysis used:
- Estimator = Diffusion
- Score model = Learned Neural Score

then Section D ALWAYS reuses the exact Portfolio:
- return interval
- lookback window
- asset ordering
- historical moments
- neural moments
- synthetic/combined return pool
- T
- gamma
- constraints
- portfolio rule
- exact Recommended Weights

The Method Comparison horizon selector does not override Section D.

### D3 guarantee
For the portfolio rule selected on Portfolio, normally Mean-Variance:

`D3 Neural Diffusion weights = Portfolio Recommended Weights`

The weight vector is copied directly, not recomputed.

A synchronization audit reports:
- maximum absolute weight difference
- exact synchronization True/False

The expected maximum difference is 0 (up to machine precision).


## Version 33: exact Portfolio dataset persists across navigation

Fixed the remaining navigation bug where a completed monthly Portfolio analysis could
return as:

`You have 21 1 month return observations ... minimum is 60`

### Root cause
The app was preserving analysis flags and some parameters, but it was still reconstructing
the Portfolio from transient/global `returns` state after navigating away. That allowed
a completed 60/120-month analysis to come back with a different short dataset while
retaining the label `1 month`.

### Fix
A completed Portfolio run now stores the exact full cleaned return DataFrame in
`shared_current_window["full_returns"]`, together with:
- interval label
- periods per year
- lookback
- estimation window
- moments
- model settings
- weights

When returning to Portfolio after visiting another section:
1. the exact saved full dataset is restored;
2. the exact saved lookback is restored;
3. the lookback control is temporarily disabled for that completed run;
4. no diffusion minimum check is performed on a newly reconstructed/stale dataset;
5. cached results are reused.

Downloading a new Yahoo dataset or uploading a new CSV invalidates the old completed run.
Pressing Clear also resets the completed run.

This prevents a 120-month Portfolio result from silently turning into a 21-observation
monthly window merely because the user navigated to another section.


## Version 34: Portfolio restore no longer depends on `analysis_has_run`

The remaining navigation bug was caused by v33 requiring BOTH:
- `analysis_has_run == True`
- a valid saved `shared_current_window`

to restore the completed Portfolio.

The boolean gate can be reset/lost during navigation even while the durable Portfolio snapshot
survives. That caused the page to ignore the saved 60/120-observation window and fall back to a
short dataset such as 21 monthly observations.

### v34 behavior
If `shared_current_window` contains a valid saved full DataFrame, it alone is sufficient to restore
the completed Portfolio.

On return to Portfolio, v34:
1. restores the exact full return dataset;
2. restores the exact estimation-window DataFrame;
3. restores the exact saved lookback;
4. re-arms `analysis_has_run=True`;
5. refreshes the global `returns`, `returns_source`, and `returns_signature`;
6. displays the restored total observation count and restored estimation-window n.

Therefore a completed monthly Diffusion analysis can no longer be rejected as `n=21 < 60`
because of a lost display flag.


## Version 35: D3 stale-snapshot bug fixed

### What D3 Neural Diffusion actually uses
D3 and Portfolio use the same core estimator:

1. train the learned neural score on the current real return window;
2. use the selected diffusion horizon `T`;
3. run the neural reverse diffusion sampler;
4. generate `M` synthetic returns;
5. combine real + synthetic returns;
6. estimate `mu_aug` and `Sigma_aug` from the combined sample;
7. apply the selected portfolio rule (Mean-Variance, LW-MV, etc.).

For Mean-Variance this is the same `compute_weights(...)` path used on Portfolio.

### Remaining mismatch root cause
The D3 table could persist in `st.session_state["mc_snapshot"]` after Portfolio was rerun.
So D3 could show a valid but OLD Neural-MV portfolio, even though the Portfolio page showed
new Recommended Weights.

### Fix
Every completed Portfolio analysis now increments `portfolio_revision`.
The current-window Method Comparison snapshot records the exact revision it was built from.

When Portfolio is rerun:
- old `mc_snapshot` is deleted immediately;
- Section D cannot display old D1-D4 results;
- the user must rerun Section D, which then copies the exact current Portfolio weight vector
  for the matching Neural-Diffusion portfolio rule.

This makes Portfolio Recommended Weights the authoritative current-window portfolio.


## v36 — Portfolio navigation persistence root-cause fix
The CSV uploader remains populated across Streamlit reruns and page navigation. The old
code interpreted `uploaded is not None` as a new upload every time Portfolio was revisited,
then deleted `shared_current_window` and set `analysis_has_run=False`. This is why the
completed tables disappeared and the page fell back to n=21.

v36 stores a SHA-256 fingerprint of the uploaded CSV. The completed Portfolio snapshot is
invalidated only when the uploaded file actually changes. Returning with the same file
restores the saved full dataset, lookback, analysis flag, and tables.


## v37 — Section D save and persistence

Fixed the issue where `D. Current-Window Estimator / Portfolio-Rule Comparison`
could not be saved independently.

Section D now has:
- `Save Section D`
- `Clear Section D`
- `Download Section D tables + charts (.zip)`

The dedicated ZIP contains:
- D1 estimator-based metrics
- D2 common historical-benchmark metrics
- D3 portfolio weights
- D4 moment diagnostics
- Neural training diagnostics
- Portfolio ↔ D3 synchronization audit
- Section-D metadata
- interactive HTML charts for D1, D2, and D3

The Section-D result remains in `st.session_state["mc_snapshot"]` when navigating
away and back. A stale Portfolio revision now triggers a warning instead of silently
deleting the D tables.


## v38 — Faster B/C rolling OOS studies

The expensive rolling OOS studies now have explicit speed profiles and caching.

### B. Selected-Horizon Monthly OOS Comparison
- Turbo (default): max 36 OOS months, 50 neural epochs, M=100, 25 reverse steps,
  hidden width capped at 64, patience 15.
- Fast: max 60 OOS months, 100 epochs, M=200, 50 reverse steps, hidden width capped
  at 128, patience 30.
- Research: uses the full selected 12–120 OOS months and all user neural settings.

### C. 3M / 6M / 12M Research Study
- Turbo (default): max 24 OOS months per horizon, 40 epochs, M=100, 25 reverse steps,
  hidden width capped at 64.
- Fast: max 48 OOS months per horizon, 80 epochs, M=150, 40 reverse steps.
- Research: full selected OOS period and settings.

Turbo/Fast remain monthly rolling OOS experiments; they simply use fewer OOS months and
lighter neural/sampling settings. They are intended for exploratory work. Use Research
for final 60–120 month tables.

The underlying `monthly_rebalance_oos_comparison` call is now wrapped in `st.cache_data`.
Re-running an identical B or C study in the same Streamlit cache no longer repeats all
rolling neural retraining.
