"""Volatility-targeting backtest: the practical use of the forecasts.

At each rebalance the portfolio is levered up or down so that its *forecast*
volatility equals a fixed target:

    leverage_t = min( max_leverage, target_vol / forecast_vol_t )

The portfolio variance forecast combines the per-asset variance forecasts of
a model with an EWMA correlation matrix of past returns:

    var_p = (w * sigma)' C (w * sigma)

A better volatility forecast shows up as realised volatility that stays
closer to the target, with fewer and smaller leverage changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


@dataclass(frozen=True)
class VolTargetConfig:
    target_vol: float = 0.10      # annualised volatility target
    max_leverage: float = 2.0     # cap on gross exposure
    rebalance_every: int = 1      # trading days between rebalances
    cost_bps: float = 2.0         # cost per unit of one-way turnover, in basis points
    weighting: str = "equal"      # "equal" or "inverse_vol" base weights
    corr_halflife: int = 63       # half-life (days) of the EWMA correlation matrix
    exec_lag: int = 0             # 0: trade at the close the forecast is made; 1: one day later


def proxy_calibration(r2: pd.DataFrame, rv: pd.DataFrame, min_periods: int = 252) -> pd.DataFrame:
    """Expanding ratio of mean squared return to mean range-based variance.

    The forecasts target a range-based variance proxy; a portfolio is exposed
    to close-to-close variance. The two differ in level by a few percent per
    asset, so forecasts are rescaled with this (past-only) ratio before use.
    """
    ratio = r2.expanding(min_periods).mean() / rv.expanding(min_periods).mean()
    return ratio.clip(0.5, 2.0).fillna(1.0)


def backtest(
    returns: pd.DataFrame,
    var_forecast: pd.DataFrame | None,
    rf: pd.Series,
    cfg: VolTargetConfig = VolTargetConfig(),
) -> pd.DataFrame:
    """Simulate a volatility-targeted portfolio.

    Parameters
    ----------
    returns : simple daily returns, dates x assets.
    var_forecast : daily variance forecasts made at the close of each date
        (same shape as `returns`). ``None`` runs the unmanaged portfolio
        (leverage 1) for comparison.
    rf : daily risk-free rate; earned on cash, paid on borrowed money.

    Returns a daily frame with the strategy return (dated on the day it is
    earned), leverage, forecast portfolio volatility, turnover and costs.
    """
    managed = var_forecast is not None
    dates = returns.index if not managed else returns.index.intersection(var_forecast.dropna(how="all").index)
    R = returns.reindex(dates).to_numpy(dtype=float)
    n_days, n_assets = R.shape
    rf_arr = rf.reindex(dates).fillna(0.0).to_numpy(dtype=float)
    if managed:
        F = var_forecast.reindex(index=dates, columns=returns.columns).to_numpy(dtype=float)
        if cfg.exec_lag:
            F = np.vstack([np.full((cfg.exec_lag, n_assets), np.nan), F[: -cfg.exec_lag]])
    else:
        F = np.where(np.isfinite(R), 1.0, np.nan)

    lam = 0.5 ** (1.0 / cfg.corr_halflife)
    cov = np.zeros((n_assets, n_assets))
    cov_weight = 0.0
    weights = np.zeros(n_assets)          # current holdings (fractions of equity)
    out = np.full((n_days, 5), np.nan)    # ret, leverage, vol_forecast, turnover, cost
    warmup = 21 if managed else 0

    for t in range(n_days):
        # --- return earned today on yesterday's closing holdings -------------
        r_t = np.nan_to_num(R[t])
        if t > 0:
            gross = weights @ r_t + (1.0 - weights.sum()) * rf_arr[t]
            out[t, 0] = gross - out[t - 1, 4]
            if weights.any():
                weights = weights * (1.0 + r_t) / (1.0 + gross)  # holdings drift with prices
        cov = lam * cov + (1.0 - lam) * np.outer(r_t, r_t)
        cov_weight = lam * cov_weight + (1.0 - lam)

        # --- decide tonight's holdings ----------------------------------------
        f_t = F[t]
        avail = np.isfinite(f_t) & (f_t > 0)
        turnover = 0.0
        vol_p = np.nan
        if avail.any() and t >= warmup:
            base = np.zeros(n_assets)
            sigma = np.zeros(n_assets)
            sigma[avail] = np.sqrt(f_t[avail])
            base[avail] = 1.0 / sigma[avail] if (cfg.weighting == "inverse_vol" and managed) else 1.0
            base /= base.sum()
            if managed:
                sd = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
                corr = cov / np.outer(sd, sd)
                np.fill_diagonal(corr, 1.0)
                x = base * sigma
                vol_p = np.sqrt(max(x @ corr @ x, 1e-16) * TRADING_DAYS)
                leverage = min(cfg.max_leverage, cfg.target_vol / vol_p)
            else:
                leverage = 1.0
            if (t - warmup) % cfg.rebalance_every == 0:
                target = leverage * base
                turnover = np.abs(target - weights).sum()
                weights = target
        out[t, 1] = weights.sum()
        out[t, 2] = vol_p
        out[t, 3] = turnover
        out[t, 4] = turnover * cfg.cost_bps / 1e4

    res = pd.DataFrame(out, index=dates, columns=["ret", "leverage", "vol_forecast", "turnover", "cost"])
    res["rf"] = rf_arr
    # Leverage / forecast in force while each day's return was earned.
    res["held_leverage"] = res["leverage"].shift(1)
    res["held_vol_forecast"] = res["vol_forecast"].shift(1)
    res = res.iloc[warmup + 1 :]
    res["equity"] = (1.0 + res["ret"]).cumprod()
    return res


def rolling_vol(ret: pd.Series, window: int = 63) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def performance(res: pd.DataFrame, target_vol: float | None = None) -> dict[str, float]:
    """Summary statistics of a backtest produced by :func:`backtest`."""
    r = res["ret"].dropna()
    excess = r - res["rf"].reindex(r.index)
    years = len(r) / TRADING_DAYS
    equity = (1.0 + r).cumprod()
    vol = r.std() * np.sqrt(TRADING_DAYS)
    rvol = rolling_vol(r).dropna()
    stats = {
        "Ann. return": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "Ann. vol": vol,
        "Sharpe": excess.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": (equity / equity.cummax() - 1.0).min(),
        "Vol of vol": rvol.std(),
        "Worst day": r.min(),
        "Avg leverage": res["held_leverage"].mean(),
        "Turnover / yr": res["turnover"].sum() / years,
        "Cost / yr": res["cost"].sum() / years,
    }
    if target_vol is not None:
        stats["Vol tracking error"] = (rvol - target_vol).abs().mean()
        # Returns scaled by the forecast in force should have unit variance.
        z = (r / (res["held_vol_forecast"].reindex(r.index) * res["held_leverage"].reindex(r.index) / np.sqrt(TRADING_DAYS))).dropna()
        stats["Std of scaled returns"] = z.std() if len(z) else np.nan
    return stats
