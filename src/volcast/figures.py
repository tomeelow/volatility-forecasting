"""Static charts for the README (`volcast figures`). The dashboard has the interactive versions."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import data, dm, evaluate, features, voltarget, walkforward
from .config import REPORTS_DIR, SETTINGS, TRADING_DAYS
from .metrics import loss_frame
from .models import MODEL_LABELS
from .portfolio import VolTargetConfig, rolling_vol

log = logging.getLogger(__name__)

FIG_DIR = REPORTS_DIR / "figures"

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, YELLOW, VIOLET = "#2a78d6", "#eda100", "#6250d6"
DIVERGING = LinearSegmentedColormap.from_list("cool_warm", ["#184f95", "#f0efec", "#b32f2f"])
SHORT = {"rw": "RW", "ewma": "EWMA", "garch": "GARCH", "gjr": "GJR", "har": "HAR", "har_log": "logHAR",
         "lhar": "LHAR", "gbm": "LGBM", "ens": "ENS"}


def _setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK2,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", FIG_DIR / name)


def leaderboard(board: pd.DataFrame, clip: float = 30.0) -> None:
    """QLIKE change vs the HAR benchmark for every model, one panel per horizon."""
    horizons = list(board.index.get_level_values("horizon").unique())
    order = list(MODEL_LABELS)[::-1]
    fig, axes = plt.subplots(1, len(horizons), figsize=(11, 3.9), sharey=True)
    for ax, h in zip(axes, horizons, strict=True):
        b = board.loc[h].reindex(order)
        change = (b["QLIKE vs bench"] - 1.0) * 100.0
        colors = [MUTED if m == SETTINGS.benchmark else BLUE for m in order]
        ax.barh([MODEL_LABELS[m] for m in order], change.clip(upper=clip), color=colors, height=0.6)
        for i, (m, v) in enumerate(change.items()):
            star = "*" if b.loc[m, "DM p-value"] < 0.05 else ""
            value = "0%" if round(v) == 0 else f"{v:+.0f}%"
            text = "benchmark" if m == SETTINGS.benchmark else value + star + (" →" if v > clip else "")
            ax.text(min(v, clip) + (1.2 if v >= 0 else -1.2), i, text, va="center", ha="left" if v >= 0 else "right", color=INK2, fontsize=9)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlim(-34, clip + 16)
        ax.set_title(f"{h}-day horizon")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
    fig.supxlabel("Out-of-sample QLIKE vs HAR benchmark, %  (lower is better; * = significant at 5%, panel Diebold-Mariano)", fontsize=9, color=INK2)
    fig.tight_layout()
    _save(fig, "leaderboard.png")


def forecast_example(forecasts: pd.DataFrame, ticker: str = "SPY", horizon: int = 5) -> None:
    """Forecast vs realised volatility through the COVID crash."""
    fc = forecasts[(forecasts["ticker"] == ticker) & (forecasts["horizon"] == horizon)].set_index("date").sort_index()
    fc = fc.loc["2020-01-01":"2020-09-30"]
    vol = lambda v: np.sqrt(v * TRADING_DAYS) * 100.0  # noqa: E731
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(fc.index, vol(fc["target"]), color=MUTED, alpha=0.18, linewidth=0)
    ax.plot(fc.index, vol(fc["target"]), color=MUTED, linewidth=1, label=f"Realised over the next {horizon} days")
    ax.plot(fc.index, vol(fc["har"]), color=VIOLET, label="HAR forecast")
    ax.plot(fc.index, vol(fc["gbm"]), color=BLUE, label="LightGBM forecast")
    ax.set_ylabel("Annualised volatility, %")
    ax.set_ylim(0)
    ax.set_title(f"{ticker}: {horizon}-day volatility forecasts through the COVID crash (out-of-sample)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", visible=False)
    _save(fig, "forecast_example.png")


def dm_heatmap(forecasts: pd.DataFrame, horizon: int = 1) -> None:
    """Pairwise panel Diebold-Mariano statistics."""
    fc = evaluate.prepare(forecasts, horizon)
    models = list(MODEL_LABELS)
    stat, _ = dm.dm_matrix(loss_frame(fc, models, "qlike"), fc["date"], models, horizon)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    im = ax.imshow(stat.to_numpy(), cmap=DIVERGING, vmin=-6, vmax=6)
    for i in range(len(models)):
        for j in range(len(models)):
            v = stat.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 3.5 else INK)
    ax.set_xticks(range(len(models)), [SHORT[m] for m in models], fontsize=8)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=0.7, label="DM statistic (blue: row model better)").outline.set_visible(False)
    ax.set_title(f"Diebold-Mariano statistics, {horizon}-day horizon (row vs column)", pad=28)
    _save(fig, "dm_matrix.png")


def vol_targeting(target: float = 0.10) -> None:
    """Realised volatility and growth of $1 for SPY with and without volatility targeting."""
    prices, aux = data.load_prices(), data.load_aux()
    panel, forecasts = features.load_panel(), walkforward.load_forecasts()
    returns = data.simple_returns(prices)
    returns = returns.loc[returns.index >= forecasts["date"].min(), ["SPY"]]
    cfg = VolTargetConfig(target_vol=target)
    _, runs = voltarget.compare(returns, forecasts, aux["rf"], voltarget.calibration(panel), ["unmanaged", "ewma", "gbm"], 1, cfg)
    series = [("unmanaged", "SPY, unmanaged", MUTED), ("ewma", "Vol-targeted, EWMA forecast", YELLOW), ("gbm", "Vol-targeted, LightGBM forecast", BLUE)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    for key, label, color in series:
        a.plot(rolling_vol(runs[key]["ret"]) * 100, color=color, label=label, linewidth=1.3)
        b.plot(runs[key]["equity"], color=color, label=label, linewidth=1.3)
    a.axhline(target * 100, color=INK, linewidth=1, linestyle="--", label=f"Target {target * 100:.0f}%")
    a.set_title("Realised volatility, rolling 3 months (% p.a.)")
    a.set_ylim(0)
    a.legend(loc="upper left")
    b.set_yscale("log")
    b.set_yticks([1, 2, 5, 10], ["$1", "$2", "$5", "$10"])
    b.minorticks_off()
    b.set_title("Growth of $1 (log scale)")
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "vol_targeting.png")


def make_all() -> None:
    _setup()
    forecasts = walkforward.load_forecasts()
    board = pd.read_csv(REPORTS_DIR / "leaderboard.csv", index_col=["horizon", "model"])
    leaderboard(board)
    forecast_example(forecasts)
    dm_heatmap(forecasts)
    vol_targeting()
