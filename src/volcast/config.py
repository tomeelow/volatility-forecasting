"""Project configuration: asset universe, sample dates, walk-forward settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

TRADING_DAYS = 252

# ticker -> asset class. 63 liquid US-listed stocks and ETFs, all with history
# reaching back to (at least) 2007 so the first training window covers the GFC.
UNIVERSE: dict[str, str] = {
    # --- single stocks -----------------------------------------------------
    "AAPL": "Stock: Tech", "MSFT": "Stock: Tech", "NVDA": "Stock: Tech",
    "INTC": "Stock: Tech", "CSCO": "Stock: Tech", "ORCL": "Stock: Tech",
    "IBM": "Stock: Tech", "GOOGL": "Stock: Tech", "AMZN": "Stock: Consumer",
    "JPM": "Stock: Financials", "BAC": "Stock: Financials", "C": "Stock: Financials",
    "WFC": "Stock: Financials", "GS": "Stock: Financials",
    "XOM": "Stock: Energy", "CVX": "Stock: Energy", "COP": "Stock: Energy",
    "JNJ": "Stock: Health", "PFE": "Stock: Health", "MRK": "Stock: Health",
    "UNH": "Stock: Health", "AMGN": "Stock: Health",
    "PG": "Stock: Consumer", "KO": "Stock: Consumer", "PEP": "Stock: Consumer",
    "WMT": "Stock: Consumer", "HD": "Stock: Consumer", "MCD": "Stock: Consumer",
    "DIS": "Stock: Consumer", "NKE": "Stock: Consumer",
    "BA": "Stock: Industrials", "CAT": "Stock: Industrials", "GE": "Stock: Industrials",
    "MMM": "Stock: Industrials", "T": "Stock: Telecom", "VZ": "Stock: Telecom",
    # --- equity index ETFs -------------------------------------------------
    "SPY": "Equity index", "QQQ": "Equity index", "IWM": "Equity index",
    "DIA": "Equity index", "EFA": "Equity index", "EEM": "Equity index",
    "EWJ": "Equity index", "FXI": "Equity index",
    # --- sector ETFs -------------------------------------------------------
    "XLF": "Sector ETF", "XLE": "Sector ETF", "XLK": "Sector ETF",
    "XLV": "Sector ETF", "XLI": "Sector ETF", "XLP": "Sector ETF",
    "XLY": "Sector ETF", "XLU": "Sector ETF", "XLB": "Sector ETF",
    # --- bonds -------------------------------------------------------------
    "TLT": "Bonds", "IEF": "Bonds", "LQD": "Bonds", "HYG": "Bonds", "TIP": "Bonds",
    # --- real assets -------------------------------------------------------
    "GLD": "Real assets", "SLV": "Real assets", "USO": "Real assets",
    "DBC": "Real assets", "VNQ": "Real assets",
}

# Auxiliary series (not forecast targets).
VIX_TICKER = "^VIX"
RF_TICKER = "^IRX"  # 13-week T-bill yield, % annualised


@dataclass(frozen=True)
class Settings:
    """Knobs of the experiment. Defaults reproduce the results in the README."""

    data_start: str = "2005-01-01"
    oos_start: str = "2010-01-04"       # first out-of-sample forecast origin
    horizons: tuple[int, ...] = (1, 5, 22)
    refit_every: int = 63               # trading days between model refits (~1 quarter)
    min_train_obs: int = 500            # per-asset rows required before forecasting
    benchmark: str = "har"              # reference model for relative losses / DM tests
    n_jobs: int = -1
    seed: int = 7


SETTINGS = Settings()
