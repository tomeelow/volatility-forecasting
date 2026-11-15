"""Build the modelling panel: realized-variance proxy, targets and predictors.

Timing convention (the one thing that must be right): a row with date `t`
contains only information available at the close of day `t`. The target for
horizon `h` is the average daily variance over days t+1 .. t+h, and `tend_h`
is the date on which that target becomes fully observed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import estimators as est
from .config import ARTIFACTS_DIR, SETTINGS, UNIVERSE

log = logging.getLogger(__name__)

PANEL_PATH = ARTIFACTS_DIR / "panel.parquet"

EWMA_LAMBDA = 0.94  # RiskMetrics decay

HAR_FEATURES = ["rv_d", "rv_w", "rv_m"]
LEVERAGE_FEATURES = ["neg_d", "neg_w", "neg_m"]

# Scale-free predictors for the pooled gradient-boosting model. Everything is
# expressed relative to the asset's own monthly variance (`rv_m`) so that a
# single model can be trained across bonds, equities and commodities.
GBM_FEATURES = [
    "f_d", "f_d1", "f_d2", "f_w", "f_q", "f_y",        # own variance term structure
    "lvl",                                             # log annualised monthly variance
    "z_d", "z_w", "z_m",                               # standardised past returns (leverage)
    "on_share", "vov", "volu",                         # overnight share, vol-of-vol, volume surprise
    "mkt_d", "mkt_w", "mkt_y", "idio_d",               # cross-sectional (market-wide) variance state
    "spy_d", "spy_w", "vix", "vrp",                    # equity-market and implied-vol state
    "dow", "asset_class",
]


def _wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=col).sort_index()


def build_panel(prices: pd.DataFrame, aux: pd.DataFrame, horizons=SETTINGS.horizons) -> pd.DataFrame:
    """Long panel [date, ticker] with targets, naive forecasts and model features."""
    o, h, l, c, v = (_wide(prices, k) for k in ["open", "high", "low", "close", "volume"])

    interior_gaps = int((c.isna() & c.ffill().notna() & c.bfill().notna()).sum().sum())
    if interior_gaps:
        log.warning("%d missing bars inside asset histories", interior_gaps)

    ret = np.log(c).diff()
    rv_raw = est.gk_overnight(o, h, l, c)
    rv = rv_raw.apply(est.floor_variance)
    rv = rv.where(rv_raw.notna())
    overnight = est.overnight_sq(o, c)

    def mean(x: pd.DataFrame, n: int, min_periods: int | None = None) -> pd.DataFrame:
        return x.rolling(n, min_periods=min_periods or n).mean()

    w: dict[str, pd.DataFrame] = {"ret": ret, "r2": ret**2, "rv": rv}

    # --- HAR components (Corsi 2009): daily, weekly, monthly averages ----------
    w["rv_d"] = rv
    w["rv_w"] = mean(rv, 5)
    w["rv_m"] = mean(rv, 22)
    rv_q = mean(rv, 66)
    rv_y = mean(rv, 252, min_periods=126)

    # --- leverage terms: negative part of past returns -------------------------
    w["neg_d"] = ret.clip(upper=0)
    w["neg_w"] = mean(ret, 5).clip(upper=0)
    w["neg_m"] = mean(ret, 22).clip(upper=0)

    # --- model-free forecasts ----------------------------------------------------
    w["ewma"] = rv.ewm(alpha=1 - EWMA_LAMBDA, adjust=False, min_periods=22, ignore_na=True).mean()
    w["ewma"] = w["ewma"].where(rv.notna())

    # --- targets -------------------------------------------------------------------
    dates = pd.Series(rv.index, index=rv.index)
    for hz in horizons:
        w[f"target_{hz}"] = mean(rv, hz).shift(-hz)

    # --- scale-free features for boosting -----------------------------------------
    base = w["rv_m"]
    w["f_d"] = np.log(rv / base)
    w["f_d1"] = np.log(rv.shift(1) / base)
    w["f_d2"] = np.log(rv.shift(2) / base)
    w["f_w"] = np.log(w["rv_w"] / base)
    w["f_q"] = np.log(rv_q / base)
    w["f_y"] = np.log(rv_y / base)
    w["lvl"] = np.log(base * 252)
    w["z_d"] = ret / np.sqrt(base)
    w["z_w"] = ret.rolling(5).sum() / np.sqrt(5 * base)
    w["z_m"] = ret.rolling(22).sum() / np.sqrt(22 * base)
    w["on_share"] = mean(overnight, 22) / base
    w["vov"] = np.log(rv).rolling(22).std()
    w["volu"] = np.log(v.replace(0, np.nan) / mean(v.replace(0, np.nan), 22)).fillna(0.0)

    # Market-wide state: cross-sectional medians use same-day data only.
    def xs_median(x: pd.DataFrame) -> pd.DataFrame:
        med = x.median(axis=1)
        return pd.DataFrame(np.repeat(med.values[:, None], x.shape[1], axis=1), index=x.index, columns=x.columns)

    w["mkt_d"] = xs_median(w["f_d"])
    w["mkt_w"] = xs_median(w["f_w"])
    w["mkt_y"] = xs_median(w["f_y"])
    w["idio_d"] = w["f_d"] - w["mkt_d"]

    def broadcast(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(s.values[:, None], rv.shape[1], axis=1), index=rv.index, columns=rv.columns)

    vix = aux["vix"].reindex(rv.index).ffill()
    w["spy_d"] = broadcast(w["f_d"]["SPY"])
    w["spy_w"] = broadcast(w["f_w"]["SPY"])
    w["vix"] = broadcast(np.log(vix))
    # Variance risk premium proxy: implied (VIX^2) over realized SPY variance.
    w["vrp"] = broadcast(np.log((vix / 100) ** 2 / 252 / w["rv_m"]["SPY"]))

    panel = pd.concat({k: x.stack(future_stack=True) for k, x in w.items()}, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()

    panel["dow"] = panel["date"].dt.day_of_week.astype("int8")
    classes = sorted(set(UNIVERSE.values()))
    panel["asset_class"] = panel["ticker"].map(UNIVERSE).map({k: i for i, k in enumerate(classes)}).astype("int8")
    for hz in horizons:
        panel[f"tend_{hz}"] = panel["date"].map(dates.shift(-hz))

    # Keep rows where every predictor exists; targets may be NaN at the sample end
    # (those rows are still forecast - they are the "live" predictions).
    needed = ["rv", "ewma", *HAR_FEATURES, *LEVERAGE_FEATURES, *GBM_FEATURES]
    panel = panel.dropna(subset=needed).sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def build_and_save(prices: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    panel = build_panel(prices, aux)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    log.info(
        "Panel: %d rows, %d assets, %s -> %s",
        len(panel), panel["ticker"].nunique(), panel["date"].min().date(), panel["date"].max().date(),
    )
    return panel


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)
