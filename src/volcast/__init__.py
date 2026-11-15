"""volcast: walk-forward volatility forecasting and volatility targeting.

Nine models (random walk, EWMA, GARCH, GJR-GARCH, three HAR variants, a pooled
LightGBM and an ensemble) forecast the variance of 63 assets out of sample; the
forecasts are scored with QLIKE and Diebold-Mariano tests and used to run a
volatility-targeted portfolio.

Source: https://github.com/tomeelow/volatility-forecasting
"""

__version__ = "0.1.0"
