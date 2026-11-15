"""Forecasting models. Every model outputs a forecast of the *average daily
variance* over the next `h` trading days."""

from .garch import garch_forecasts
from .gbm import GBM
from .har import HAR

# key -> display name, in the order used by tables and charts.
MODEL_LABELS: dict[str, str] = {
    "rw": "Random walk",
    "ewma": "EWMA (RiskMetrics)",
    "garch": "GARCH(1,1)",
    "gjr": "GJR-GARCH",
    "har": "HAR",
    "har_log": "log-HAR",
    "lhar": "log-HAR + leverage",
    "gbm": "LightGBM",
    "ens": "Ensemble",
}
MODEL_FAMILY: dict[str, str] = {
    "rw": "Naive", "ewma": "Naive",
    "garch": "GARCH", "gjr": "GARCH",
    "har": "HAR", "har_log": "HAR", "lhar": "HAR",
    "gbm": "ML", "ens": "ML",
}
# The ensemble is fixed a priori: one model from each family, equal weights.
ENSEMBLE_MEMBERS = ("gjr", "lhar", "gbm")

__all__ = ["GBM", "HAR", "garch_forecasts", "MODEL_LABELS", "MODEL_FAMILY", "ENSEMBLE_MEMBERS"]
