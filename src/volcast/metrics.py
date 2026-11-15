"""Forecast loss functions.

QLIKE and MSE are the two losses that rank volatility forecasts consistently
even though the true variance is never observed and a noisy proxy is used in
its place (Patton, 2011). QLIKE is preferred: it depends only on the ratio
forecast/realised, so high-variance assets and crisis days do not dominate,
and it penalises under-prediction more than over-prediction - which is the
costly direction for risk management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def qlike(actual, forecast):
    """QLIKE loss  y/f - log(y/f) - 1  (>= 0, equals 0 iff forecast == actual)."""
    ratio = np.asarray(actual, dtype=float) / np.asarray(forecast, dtype=float)
    return ratio - np.log(ratio) - 1.0


def mse(actual, forecast):
    """Squared error on the variance scale."""
    return (np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)) ** 2


LOSSES = {"qlike": qlike, "mse": mse}


def loss_frame(forecasts: pd.DataFrame, models: list[str], loss: str = "qlike") -> pd.DataFrame:
    """Per-observation losses: same index as `forecasts`, one column per model."""
    fn = LOSSES[loss]
    target = forecasts["target"].to_numpy(dtype=float)
    return pd.DataFrame({m: fn(target, forecasts[m].to_numpy(dtype=float)) for m in models}, index=forecasts.index)


def mincer_zarnowitz_r2(actual, forecast) -> float:
    """R^2 of regressing realised log-variance on forecast log-variance."""
    a, f = np.log(np.asarray(actual, dtype=float)), np.log(np.asarray(forecast, dtype=float))
    return float(np.corrcoef(a, f)[0, 1] ** 2)
