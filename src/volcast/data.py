"""Download and clean daily OHLC bars (Yahoo Finance) with on-disk caching."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .config import DATA_DIR, RF_TICKER, SETTINGS, TRADING_DAYS, UNIVERSE, VIX_TICKER

log = logging.getLogger(__name__)

PRICES_PATH = DATA_DIR / "prices.parquet"
AUX_PATH = DATA_DIR / "aux.parquet"

_COLS = ["open", "high", "low", "close", "volume"]


def _fetch(tickers: list[str], start: str) -> pd.DataFrame:
    """Raw yfinance download -> long frame [date, ticker, open, high, low, close, volume]."""
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (network issue or rate limit).")
    long = (
        raw.stack(level=0, future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .rename(columns=str.lower)
        .reset_index()
    )
    long.columns.name = None
    return long[["date", "ticker", *_COLS]]


def clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and repair inconsistent highs/lows.

    Yahoo bars occasionally have a high below the open/close (or a low above),
    which would produce negative range-based variances.
    """
    df = bars.dropna(subset=["open", "high", "low", "close"]).copy()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def download(start: str | None = None, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the asset universe plus VIX and the T-bill rate; cache as parquet."""
    if PRICES_PATH.exists() and AUX_PATH.exists() and not force:
        log.info("Using cached data in %s", DATA_DIR)
        return load_prices(), load_aux()

    start = start or SETTINGS.data_start
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = list(UNIVERSE)
    log.info("Downloading %d tickers from %s ...", len(tickers), start)
    prices = clean_bars(_fetch(tickers, start))
    # Yahoo sporadically drops a ticker from a batch request; retry those.
    for attempt in range(1, 4):
        missing = sorted(set(tickers) - set(prices["ticker"].unique()))
        if not missing:
            break
        log.warning("Retry %d for missing tickers: %s", attempt, missing)
        time.sleep(2 * attempt)
        try:
            prices = pd.concat([prices, clean_bars(_fetch(missing, start))], ignore_index=True)
        except RuntimeError:
            continue
    missing = sorted(set(tickers) - set(prices["ticker"].unique()))
    if missing:
        raise RuntimeError(f"No data returned for: {missing}")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    # The current day's bar is incomplete while the market is open.
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    prices = prices[prices["date"] < today].reset_index(drop=True)
    prices.to_parquet(PRICES_PATH, index=False)

    aux_raw = _fetch([VIX_TICKER, RF_TICKER], start)
    aux = aux_raw.pivot(index="date", columns="ticker", values="close").rename(
        columns={VIX_TICKER: "vix", RF_TICKER: "rf_yield"}
    )
    aux = aux[aux.index < today]
    aux.columns.name = None
    aux.reset_index().to_parquet(AUX_PATH, index=False)

    log.info(
        "Saved %d rows, %d tickers, %s -> %s",
        len(prices), prices["ticker"].nunique(),
        prices["date"].min().date(), prices["date"].max().date(),
    )
    return prices, load_aux()


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES_PATH)


def load_aux() -> pd.DataFrame:
    """VIX close and daily risk-free rate, indexed by date."""
    aux = pd.read_parquet(AUX_PATH).set_index("date").sort_index()
    aux["vix"] = aux["vix"].ffill()
    # ^IRX is a discount yield in % p.a.; convert to a simple daily rate.
    aux["rf"] = (aux["rf_yield"].ffill().clip(lower=0) / 100.0) / TRADING_DAYS
    return aux[["vix", "rf"]]


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide frame of close-to-close simple returns (date x ticker)."""
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    return close.pct_change(fill_method=None)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    return np.log(close).diff()
