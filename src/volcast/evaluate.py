"""Out-of-sample evaluation: loss tables, Diebold-Mariano tests, model confidence set."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import dm
from .config import REPORTS_DIR, SETTINGS, UNIVERSE, Settings
from .metrics import loss_frame, mincer_zarnowitz_r2
from .models import MODEL_LABELS

log = logging.getLogger(__name__)

MODELS = list(MODEL_LABELS)


def prepare(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows of one horizon with a realised target, time-ordered within asset."""
    fc = forecasts[(forecasts["horizon"] == horizon) & forecasts["target"].notna()]
    return fc.sort_values(["ticker", "date"]).reset_index(drop=True)


def model_confidence_set(daily_losses: pd.DataFrame, size: float = 0.10, reps: int = 1000, seed: int = 7) -> pd.Series:
    """MCS p-values (Hansen, Lunde & Nason 2011) on date-averaged losses.

    Models with p-value >= `size` cannot be distinguished from the best one.
    """
    from arch.bootstrap import MCS

    block = max(5, int(len(daily_losses) ** (1 / 3)))
    mcs = MCS(daily_losses, size=size, reps=reps, block_size=block, method="R", seed=seed)
    mcs.compute()
    return mcs.pvalues["Pvalue"].reindex(daily_losses.columns)


def leaderboard(fc: pd.DataFrame, horizon: int, benchmark: str = SETTINGS.benchmark, with_mcs: bool = True) -> pd.DataFrame:
    """One row per model: losses, relative losses and significance vs the benchmark.

    `fc` must hold a single horizon (see :func:`prepare`).
    """
    ql = loss_frame(fc, MODELS, "qlike")
    se = loss_frame(fc, MODELS, "mse")
    ql_asset = ql.groupby(fc["ticker"].to_numpy()).mean()
    se_asset = se.groupby(fc["ticker"].to_numpy()).mean()

    table = pd.DataFrame(index=MODELS)
    table["QLIKE"] = ql.mean()
    table["QLIKE vs bench"] = table["QLIKE"] / table.loc[benchmark, "QLIKE"]
    # MSE is averaged as per-asset ratios, otherwise the most volatile assets dominate.
    table["MSE vs bench"] = se_asset.div(se_asset[benchmark], axis=0).mean()
    table["R2 (log)"] = [
        np.mean([mincer_zarnowitz_r2(g["target"], g[m]) for _, g in fc.groupby("ticker")]) for m in MODELS
    ]
    table["Best in # assets"] = ql_asset.idxmin(axis=1).value_counts().reindex(MODELS).fillna(0).astype(int)

    dm_stat, dm_p, better, worse = [], [], [], []
    for m in MODELS:
        if m == benchmark:
            dm_stat.append(np.nan), dm_p.append(np.nan), better.append(0), worse.append(0)
            continue
        panel = dm.panel_dm(ql, fc["date"], m, benchmark, horizon)
        per_asset = dm.per_asset_dm(ql, fc["ticker"], m, benchmark, horizon)
        sig = per_asset["pvalue"] < 0.05
        dm_stat.append(panel.stat), dm_p.append(panel.pvalue)
        better.append(int((sig & (per_asset["stat"] < 0)).sum()))
        worse.append(int((sig & (per_asset["stat"] > 0)).sum()))
    table["DM stat"] = dm_stat
    table["DM p-value"] = dm_p
    table["Sig. better (assets)"] = better
    table["Sig. worse (assets)"] = worse

    if with_mcs:
        daily = ql.groupby(fc["date"].to_numpy()).mean().sort_index()
        table["MCS p-value"] = model_confidence_set(daily)
    table.index.name = "model"
    return table


def qlike_by(fc: pd.DataFrame, key: pd.Series, relative_to: str | None = None) -> pd.DataFrame:
    """Mean QLIKE per group (`key` aligned with `fc`), optionally relative to a model."""
    out = loss_frame(fc, MODELS, "qlike").groupby(key.to_numpy()).mean()
    return out.div(out[relative_to], axis=0) if relative_to else out


def run(forecasts: pd.DataFrame, settings: Settings = SETTINGS) -> dict[str, pd.DataFrame]:
    """Compute every report table and write them to `reports/` as CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    boards = []
    for h in settings.horizons:
        fc = prepare(forecasts, h)
        board = leaderboard(fc, h, settings.benchmark)
        boards.append(board.reset_index().assign(horizon=h))

        ql = loss_frame(fc, MODELS, "qlike")
        stat, pval = dm.dm_matrix(ql, fc["date"], MODELS, h)
        tables[f"dm_stat_h{h}"], tables[f"dm_pvalue_h{h}"] = stat, pval
        tables[f"qlike_by_class_h{h}"] = qlike_by(fc, fc["ticker"].map(UNIVERSE), settings.benchmark)
        tables[f"qlike_by_year_h{h}"] = qlike_by(fc, fc["date"].dt.year)
        tables[f"qlike_by_asset_h{h}"] = qlike_by(fc, fc["ticker"])
        log.info("h=%d: %d forecasts, %d assets, %s -> %s", h, len(fc), fc["ticker"].nunique(), fc["date"].min().date(), fc["date"].max().date())

    tables["leaderboard"] = pd.concat(boards, ignore_index=True).set_index(["horizon", "model"])
    for name, table in tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", float_format="%.6g")
    return tables
