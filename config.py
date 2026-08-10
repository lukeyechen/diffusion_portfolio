APP_TITLE = "Diffusion Portfolio Lab"

DEFAULT_GAMMA = 3.0
DEFAULT_LOOKBACK = 120
DEFAULT_M = 500
DEFAULT_BETA = 1.0
DEFAULT_SEED = 42

# Practical safeguards for the user-facing app.
DEFAULT_CONSTRAINT_MODE = "Long-only"
DEFAULT_MAX_LONG_WEIGHT = 0.40
DEFAULT_MAX_SHORT_WEIGHT = 0.20
DEFAULT_MAX_GROSS_EXPOSURE = 1.50

SUPPORTED_RULES = [
    "Equal Weight",
    "Mean-Variance",
    "Ledoit-Wolf Mean-Variance",
]

CONSTRAINT_MODES = [
    "Long-only",
    "Limited Long-Short",
    "Research / Unconstrained",
]
