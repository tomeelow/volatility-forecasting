"""Daily variance estimators built from OHLC bars.

Without intraday data the "realized variance" has to be proxied from daily bars.
Squared close-to-close returns are unbiased but extremely noisy; range-based
estimators use the intraday high/low and are 5-8x more efficient.

All functions take aligned arrays/Series of prices and return *daily variance*
(not annualised, not volatility).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_GK_C = 2.0 * np.log(2.0) - 1.0  # ~0.386


def squared_return(close: pd.Series) -> pd.Series:
    """Close-to-close squared log return: unbiased, very noisy."""
    return np.log(close / close.shift(1)) ** 2


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980): open-to-close variance from the high-low range."""
    return np.log(high / low) ** 2 / (4.0 * np.log(2.0))


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass (1980): open-to-close variance, ~7.4x as efficient as r^2."""
    return 0.5 * np.log(high / low) ** 2 - _GK_C * np.log(close / open_) ** 2


def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Rogers-Satchell (1991): open-to-close variance, robust to drift."""
    return np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)


def overnight_sq(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Squared overnight (previous close -> open) log return."""
    return np.log(open_ / close.shift(1)) ** 2


def gk_overnight(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass plus the squared overnight return.

    Estimates the full close-to-close variance (the quantity a portfolio is
    actually exposed to) while keeping most of the efficiency of the range.
    This is the realized-variance proxy used throughout the project.
    """
    return overnight_sq(open_, close) + garman_klass(open_, high, low, close)


def floor_variance(rv: pd.Series, window: int = 252, frac: float = 0.01) -> pd.Series:
    """Floor a variance series at `frac` x its trailing median.

    Stale or glitchy bars (high == low) produce a variance of ~0, which makes
    log-based losses explode. The floor only uses past data, so it is safe to
    apply before building features and targets.
    """
    med = rv.rolling(window, min_periods=20).median().shift(1)
    floor = (frac * med).fillna(frac * rv.expanding(min_periods=1).median())
    return np.maximum(rv, floor)
