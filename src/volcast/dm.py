"""Diebold-Mariano tests of equal predictive accuracy.

For two forecasts with per-period losses L1, L2 the loss differential is
d_t = L1_t - L2_t and the null hypothesis is E[d_t] = 0. The statistic is

    DM = mean(d) / sqrt( LRV(d) / T )

where LRV is a HAC (Newey-West) estimate of the long-run variance of d,
needed because loss differentials are autocorrelated - mechanically so for
overlapping multi-day targets (an h-day target induces MA(h-1) errors).

A negative statistic means forecast 1 has the lower loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    stat: float
    pvalue: float      # two-sided
    mean_diff: float   # mean(L1 - L2)
    n: int
    lags: int


def newey_west_lrv(x: np.ndarray, lags: int) -> float:
    """Long-run variance of `x` with a Bartlett kernel (always >= 0)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    lrv = x @ x / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * (x[k:] @ x[:-k]) / n
    return float(lrv)


def default_lags(n: int, horizon: int = 1) -> int:
    """Newey-West rule-of-thumb bandwidth, widened to cover the h-1 overlap.

    For h > 1 the Bartlett kernel down-weights the mechanical MA(h-1)
    autocorrelation, so the bandwidth is set to 2(h-1) rather than h-1.
    """
    rule = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(rule, 2 * (horizon - 1))


def dm_test(loss1, loss2, horizon: int = 1, lags: int | None = None) -> DMResult:
    """Diebold-Mariano test on two aligned loss series (time-ordered)."""
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return DMResult(np.nan, np.nan, np.nan, n, 0)
    lags = default_lags(n, horizon) if lags is None else lags
    lrv = newey_west_lrv(d, lags)
    if lrv <= 0:
        return DMResult(np.nan, np.nan, float(d.mean()), n, lags)
    stat = d.mean() / np.sqrt(lrv / n)
    pvalue = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return DMResult(float(stat), float(pvalue), float(d.mean()), n, lags)


def panel_dm(losses: pd.DataFrame, dates: pd.Series, m1: str, m2: str, horizon: int = 1) -> DMResult:
    """DM test on the whole panel.

    Loss differentials are first averaged across assets for each date, giving
    one time series. Volatility shocks hit all assets at once, so treating
    asset-days as independent would wildly overstate significance; averaging
    cross-sectionally first makes the test robust to that dependence.
    """
    d = (losses[m1] - losses[m2]).groupby(dates.to_numpy()).mean().sort_index()
    return dm_test(d.to_numpy(), np.zeros(len(d)), horizon=horizon)


def per_asset_dm(
    losses: pd.DataFrame, tickers: pd.Series, m1: str, m2: str, horizon: int = 1
) -> pd.DataFrame:
    """DM test of m1 vs m2 separately for every asset (rows must be time-ordered)."""
    rows = {}
    for ticker, idx in tickers.groupby(tickers.to_numpy()).groups.items():
        r = dm_test(losses.loc[idx, m1].to_numpy(), losses.loc[idx, m2].to_numpy(), horizon=horizon)
        rows[ticker] = {"stat": r.stat, "pvalue": r.pvalue, "mean_diff": r.mean_diff, "n": r.n}
    return pd.DataFrame.from_dict(rows, orient="index")


def dm_matrix(losses: pd.DataFrame, dates: pd.Series, models: list[str], horizon: int = 1):
    """Pairwise panel DM statistics and p-values (row model vs column model)."""
    stat = pd.DataFrame(np.nan, index=models, columns=models)
    pval = pd.DataFrame(np.nan, index=models, columns=models)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            r = panel_dm(losses, dates, a, b, horizon)
            stat.loc[a, b], stat.loc[b, a] = r.stat, -r.stat
            pval.loc[a, b] = pval.loc[b, a] = r.pvalue
    return stat, pval
