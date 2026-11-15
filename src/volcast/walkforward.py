"""Walk-forward (expanding-window) out-of-sample forecasting engine.

The out-of-sample period is cut into consecutive blocks of `refit_every`
trading days. For each block every model is re-estimated on all data whose
*target* was fully observed by the block's first day, then used to forecast
each day in the block. Nothing from the future ever reaches a model:

    train rows :  tend_h <= block_start        (target fully realised)
    test rows  :  block_start <= date <= block_end
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, SETTINGS, Settings
from .models import ENSEMBLE_MEMBERS, GBM, HAR, MODEL_LABELS, garch_forecasts

log = logging.getLogger(__name__)

FORECASTS_PATH = ARTIFACTS_DIR / "forecasts.parquet"
IMPORTANCE_PATH = ARTIFACTS_DIR / "gbm_importance.parquet"

RW_COLUMN = {1: "rv_d", 5: "rv_w", 22: "rv_m"}


def make_blocks(dates: pd.DatetimeIndex, oos_start: str, refit_every: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the out-of-sample dates into consecutive (start, end) refit blocks."""
    oos = dates[dates >= pd.Timestamp(oos_start)]
    return [(oos[i], oos[min(i + refit_every, len(oos)) - 1]) for i in range(0, len(oos), refit_every)]


def split(panel: pd.DataFrame, horizon: int, start: pd.Timestamp, end: pd.Timestamp, min_train_obs: int):
    """Train/test frames for one block, restricted to assets with enough history."""
    tend = panel[f"tend_{horizon}"]
    train = panel[tend.notna() & (tend <= start)]
    counts = train["ticker"].value_counts()
    eligible = counts.index[counts >= min_train_obs]
    train = train[train["ticker"].isin(eligible)]
    test = panel[(panel["date"] >= start) & (panel["date"] <= end) & panel["ticker"].isin(eligible)]
    return train, test


def _rw(panel_rows: pd.DataFrame, horizon: int) -> np.ndarray:
    """Random walk: the last observed h-day average variance."""
    col = RW_COLUMN.get(horizon)
    if col is None:
        raise ValueError(f"No random-walk feature for horizon {horizon}")
    return panel_rows[col].to_numpy(dtype=float)


def run_horizon(panel: pd.DataFrame, horizon: int, blocks, settings: Settings):
    """Walk-forward forecasts of the panel models (HAR family + boosting) for one horizon."""
    models = [
        HAR("har"),
        HAR("har_log", log=True),
        HAR("lhar", log=True, leverage=True),
        GBM("gbm", seed=settings.seed),
    ]
    parts = []
    for i, (start, end) in enumerate(blocks):
        train, test = split(panel, horizon, start, end, settings.min_train_obs)
        if test.empty:
            continue
        out = test[["date", "ticker"]].copy()
        out["horizon"] = horizon
        out["target"] = test[f"target_{horizon}"].to_numpy()
        out["rw"] = _rw(test, horizon)
        out["ewma"] = test["ewma"].to_numpy()
        for model in models:
            out[model.name] = model.fit_predict(train, test, horizon)
        parts.append(out)
        if i % 10 == 0:
            log.info("h=%d block %d/%d (%s): train=%d test=%d", horizon, i + 1, len(blocks), start.date(), len(train), len(test))
    importance = models[-1].mean_importance().rename(f"h{horizon}")
    return pd.concat(parts, ignore_index=True), importance


def run(panel: pd.DataFrame, settings: Settings = SETTINGS) -> pd.DataFrame:
    """Run every model for every horizon and save the out-of-sample forecasts."""
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    blocks = make_blocks(dates, settings.oos_start, settings.refit_every)
    log.info("Walk-forward: %d blocks of %d days, %s -> %s", len(blocks), settings.refit_every, blocks[0][0].date(), blocks[-1][1].date())

    t0 = time.time()
    garch = {
        "garch": garch_forecasts(panel, blocks, settings.horizons, asymmetric=False, n_jobs=settings.n_jobs),
        "gjr": garch_forecasts(panel, blocks, settings.horizons, asymmetric=True, n_jobs=settings.n_jobs),
    }
    log.info("GARCH models done in %.0fs", time.time() - t0)

    frames, importances = [], []
    for h in settings.horizons:
        t0 = time.time()
        fc, imp = run_horizon(panel, h, blocks, settings)
        for name, g in garch.items():
            fc = fc.merge(g[["date", "ticker", f"h{h}"]].rename(columns={f"h{h}": name}), on=["date", "ticker"], how="left")
        fc["ens"] = fc[list(ENSEMBLE_MEMBERS)].mean(axis=1, skipna=False)
        frames.append(fc)
        importances.append(imp)
        log.info("h=%d done in %.0fs (%d forecasts)", h, time.time() - t0, len(fc))

    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts[["date", "ticker", "horizon", "target", *MODEL_LABELS]]
    n_bad = int(forecasts[list(MODEL_LABELS)].isna().any(axis=1).sum())
    if n_bad:
        log.warning("Dropping %d rows with a missing forecast", n_bad)
        forecasts = forecasts.dropna(subset=list(MODEL_LABELS)).reset_index(drop=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECASTS_PATH, index=False)
    pd.concat(importances, axis=1).rename_axis("feature").reset_index().to_parquet(IMPORTANCE_PATH, index=False)
    log.info("Saved %d forecast rows to %s", len(forecasts), FORECASTS_PATH)
    return forecasts


def load_forecasts() -> pd.DataFrame:
    return pd.read_parquet(FORECASTS_PATH)


def load_importance() -> pd.DataFrame:
    return pd.read_parquet(IMPORTANCE_PATH).set_index("feature")
