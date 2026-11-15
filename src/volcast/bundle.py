"""Compact data bundle for the hosted dashboard.

The pipeline's working files in `artifacts/` and `data/` are large and are not
committed. A hosted dashboard cannot run the pipeline, so `volcast bundle`
exports just what the dashboard reads into `app/data/` (about 30 MB), which is
committed. Locally, fresh pipeline output is preferred when it exists.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from . import data, features, voltarget, walkforward
from .config import ROOT

log = logging.getLogger(__name__)

BUNDLE_DIR = ROOT / "app" / "data"
FORECASTS = BUNDLE_DIR / "forecasts.parquet"
IMPORTANCE = BUNDLE_DIR / "gbm_importance.parquet"
RETURNS = BUNDLE_DIR / "returns.parquet"
CALIBRATION = BUNDLE_DIR / "calibration.parquet"
TRAILING = BUNDLE_DIR / "trailing.parquet"

_KEYS = ["date", "ticker", "horizon"]
_RF = "__rf__"


def source() -> str | None:
    """Where the dashboard reads from: "local" pipeline output, the "bundle", or None.

    Set VOLCAST_DATA=bundle to force the committed bundle even when local
    pipeline output exists (used to test what a hosted deployment sees).
    """
    forced = os.environ.get("VOLCAST_DATA", "").lower()
    if forced != "bundle" and walkforward.FORECASTS_PATH.exists():
        return "local"
    return "bundle" if FORECASTS.exists() else None


def _local_market():
    returns = data.simple_returns(data.load_prices())
    rf = data.load_aux()["rf"]
    panel = pd.read_parquet(features.PANEL_PATH, columns=["date", "ticker", "r2", "rv", "rv_m"])
    calib = voltarget.calibration(panel)
    trailing = panel.pivot(index="date", columns="ticker", values="rv_m").iloc[-1]
    return returns, rf, calib, trailing


def export() -> None:
    """Write the dashboard bundle from the current pipeline output."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    write = {"index": False, "compression": "zstd", "compression_level": 19}

    forecasts = walkforward.load_forecasts().sort_values(["horizon", "ticker", "date"])
    values = [c for c in forecasts.columns if c not in _KEYS]
    forecasts[values] = forecasts[values].astype("float32")  # 7 significant digits is plenty
    forecasts["horizon"] = forecasts["horizon"].astype("int8")
    forecasts.to_parquet(FORECASTS, use_byte_stream_split=values, use_dictionary=["ticker"], **write)
    walkforward.load_importance().reset_index().to_parquet(IMPORTANCE, **write)

    returns, rf, calib, trailing = _local_market()
    oos = returns.index >= forecasts["date"].min()
    returns = returns.loc[oos].astype("float32")
    returns[_RF] = rf.reindex(returns.index)
    returns.reset_index().to_parquet(RETURNS, **write)
    calib.loc[calib.index >= forecasts["date"].min()].astype("float32").reset_index().to_parquet(CALIBRATION, **write)
    trailing.rename("rv_m").rename_axis("ticker").reset_index().to_parquet(TRAILING, **write)

    size = sum(p.stat().st_size for p in BUNDLE_DIR.glob("*.parquet")) / 1e6
    log.info("Wrote dashboard bundle to %s (%.1f MB, data to %s)", BUNDLE_DIR, size, forecasts["date"].max().date())


def forecasts() -> pd.DataFrame:
    if source() == "local":
        return walkforward.load_forecasts()
    fc = pd.read_parquet(FORECASTS)
    values = [c for c in fc.columns if c not in _KEYS]
    fc[values] = fc[values].astype("float64")
    fc["ticker"] = fc["ticker"].astype(str)
    fc["horizon"] = fc["horizon"].astype("int64")
    return fc


def importance() -> pd.DataFrame:
    if source() == "local":
        return walkforward.load_importance()
    return pd.read_parquet(IMPORTANCE).set_index("feature")


def market() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Returns (date x ticker), daily risk-free rate, proxy calibration, trailing monthly variance."""
    if source() == "local":
        return _local_market()
    returns = pd.read_parquet(RETURNS).set_index("date").astype("float64")
    rf = returns.pop(_RF).rename("rf")
    calib = pd.read_parquet(CALIBRATION).set_index("date").astype("float64")
    trailing = pd.read_parquet(TRAILING).set_index("ticker")["rv_m"]
    return returns, rf, calib, trailing
