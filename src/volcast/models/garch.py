"""GARCH-family forecasts from daily returns (via the `arch` package).

Parameters are re-estimated at the start of every walk-forward block on an
expanding window; inside the block the variance recursion is rolled forward
with fixed parameters, so each forecast uses returns up to its origin only.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

log = logging.getLogger(__name__)

MIN_OBS = 250


def _asset_forecasts(
    ticker: str,
    dates: np.ndarray,
    ret: np.ndarray,
    rv: np.ndarray,
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]],
    horizons: tuple[int, ...],
    asymmetric: bool,
) -> pd.DataFrame:
    from arch import arch_model

    y = 100.0 * ret  # percent returns keep the optimiser well scaled
    max_h = max(horizons)
    model = arch_model(y, mean="Zero", vol="GARCH", p=1, o=int(asymmetric), q=1, rescale=False)
    out = {h: np.full(len(y), np.nan) for h in horizons}
    params = None

    for start, end in blocks:
        s = int(np.searchsorted(dates, np.datetime64(start), side="left"))
        e = int(np.searchsorted(dates, np.datetime64(end), side="right")) - 1
        if s < MIN_OBS or e < s:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = model.fit(disp="off", last_obs=s + 1, starting_values=params, show_warning=False)
                if not np.all(np.isfinite(res.params.values)):
                    raise ValueError("non-finite parameters")
                params = res.params.values
            except Exception:  # noqa: BLE001 - fall back to the previous estimate
                if params is None:
                    continue
                res = model.fix(params, last_obs=s + 1)
            var = res.forecast(horizon=max_h, start=s, reindex=False).variance.to_numpy()

        # GARCH forecasts the variance of close-to-close returns, while the target
        # is a range-based proxy whose level differs by a few percent per asset.
        # Rescale using the training window only so the comparison is fair.
        scale = rv[: s + 1].mean() / (ret[: s + 1] ** 2).mean()
        n = e - s + 1
        for h in horizons:
            out[h][s : e + 1] = var[:n, :h].mean(axis=1) / 1e4 * scale

    frame = pd.DataFrame({"date": dates, "ticker": ticker})
    for h in horizons:
        frame[f"h{h}"] = out[h]
    return frame


def garch_forecasts(
    panel: pd.DataFrame,
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]],
    horizons: tuple[int, ...],
    asymmetric: bool = False,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Walk-forward GARCH(1,1) / GJR-GARCH(1,1) forecasts for every asset.

    Returns a long frame [date, ticker, h1, h5, ...] of average daily variance
    forecasts made at the close of `date`.
    """
    tasks = [
        delayed(_asset_forecasts)(
            ticker, g["date"].to_numpy(), g["ret"].to_numpy(), g["rv"].to_numpy(),
            blocks, horizons, asymmetric,
        )
        for ticker, g in panel.groupby("ticker", sort=False)
    ]
    frames = Parallel(n_jobs=n_jobs)(tasks)
    return pd.concat(frames, ignore_index=True)
