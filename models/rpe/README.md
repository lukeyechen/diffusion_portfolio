# Frozen RPE research model

The JSON checkpoint contains only float32 arrays encoded as base64 and normalization statistics; loading does not deserialize Python objects. It was exported from the fresh prototype trained for the user's October 5, 2026 comparison. It is not the PDF's original checkpoint or RPE–Jorion model.

Training used 16,000 independent synthetic Gaussian tasks with five assets and 120 monthly observations. Mean prior: common N(.005,.015²) plus independent N(0,.012²) asset offsets. Monthly volatilities: log-uniform .03–.15. Correlations: two Gaussian factors (loading SD .5) plus independent diagonal variances uniform .4–1, then standardized. No historical market data or evaluation returns entered training.

Two hidden 128-unit SiLU layers; time features sin/cos at frequencies 1,2,4,8; 50 DDPM steps with linear beta .0001–.2; AdamW learning rate .001; 4,000 updates, batch 128, seed 731; block-balanced mean/covariance denoising loss. Target coordinates encode standardized sample-mean residuals and relative Cholesky factors, log diagonals. Both target and conditioning normalization are derived only from training tasks.

Independent prior simulation (200 tasks): marginal 95% mean coverage 93.1%, variance coverage 94.5%, normalized mean MSE ratio .8676. This is not a joint calibration guarantee or evidence of real-market superiority. RPE with TC did not show a reliable CER gain over trace tuning. The RPE tab supports five assets with contiguous monthly histories and fixes estimation windows at 120; unsupported inputs are rejected.

The original reproducible training code and simulation report are in the separately saved rpe_prototype_comparison.zip deliverable. This deployment includes inference only; it never retrains on backtest outcomes. Use the app's data upload or saved Portfolio data when Yahoo is unavailable.
