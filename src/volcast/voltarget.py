"""Glue between the forecast files and the volatility-targeting backtest."""

from __future__ import annotations

import logging

import pandas as pd

from . import data, features, walkforward
from .config import REPORTS_DIR
from .models import MODEL_LABELS
from .portfolio import VolTargetConfig, backtest, performance, proxy_calibration

log = logging.getLogger(__name__)

ORACLE = "oracle"       # perfect foresight: the realised future variance itself
UNMANAGED = "unmanaged"  # constant leverage of 1
STRATEGY_LABELS = {UNMANAGED: "Unmanaged (no targeting)", **MODEL_LABELS, ORACLE: "Oracle (perfect foresight)"}


def calibration(panel: pd.DataFrame) -> pd.DataFrame:
    """Past-only scale from the range-based proxy to close-to-close variance."""
    r2 = panel.pivot(index="date", columns="ticker", values="r2")
    rv = panel.pivot(index="date", columns="ticker", values="rv")
    return proxy_calibration(r2, rv)


def variance_forecasts(forecasts: pd.DataFrame, model: str, horizon: int, calib: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wide (date x ticker) daily variance forecasts of one model."""
    fc = forecasts[forecasts["horizon"] == horizon]
    wide = fc.pivot(index="date", columns="ticker", values="target" if model == ORACLE else model)
    if calib is not None:
        wide = wide * calib.reindex(index=wide.index, columns=wide.columns)
    return wide


def compare(
    returns: pd.DataFrame,
    forecasts: pd.DataFrame,
    rf: pd.Series,
    calib: pd.DataFrame | None,
    models: list[str],
    horizon: int,
    cfg: VolTargetConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest the same portfolio under several forecast models.

    Returns the summary table (one row per strategy) and the daily results,
    all restricted to a common sample so the rows are comparable.
    """
    tickers = list(returns.columns)
    runs: dict[str, pd.DataFrame] = {}
    for model in models:
        if model == UNMANAGED:
            runs[model] = backtest(returns, None, rf, cfg)
        else:
            var = variance_forecasts(forecasts, model, horizon, calib).reindex(columns=tickers)
            runs[model] = backtest(returns, var, rf, cfg)
    common = None
    for res in runs.values():
        idx = res["ret"].dropna().index
        common = idx if common is None else common.intersection(idx)
    runs = {m: res.loc[common].assign(equity=lambda d: (1 + d["ret"]).cumprod()) for m, res in runs.items()}
    summary = pd.DataFrame(
        {m: performance(res, None if m == UNMANAGED else cfg.target_vol) for m, res in runs.items()}
    ).T
    summary.index.name = "strategy"
    return summary, runs


def run_report() -> dict[str, pd.DataFrame]:
    """Reference backtests reported in the README; saved to `reports/`."""
    prices, aux = data.load_prices(), data.load_aux()
    panel, forecasts = features.load_panel(), walkforward.load_forecasts()
    calib = calibration(panel)
    returns = data.simple_returns(prices)
    oos = forecasts["date"].min()
    returns = returns[returns.index >= oos]
    models = [UNMANAGED, *MODEL_LABELS, ORACLE]

    cases = {
        "voltarget_spy": (["SPY"], VolTargetConfig(target_vol=0.10)),
        "voltarget_multi_asset": (list(returns.columns), VolTargetConfig(target_vol=0.10)),
    }
    out = {}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (tickers, cfg) in cases.items():
        summary, _ = compare(returns[tickers], forecasts, aux["rf"], calib, models, horizon=1, cfg=cfg)
        summary.to_csv(REPORTS_DIR / f"{name}.csv", float_format="%.6g")
        out[name] = summary
        log.info("%s: %d assets, target %.0f%%", name, len(tickers), cfg.target_vol * 100)
    return out
