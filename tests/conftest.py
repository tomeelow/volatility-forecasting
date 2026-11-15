import numpy as np
import pandas as pd
import pytest


def simulate_bars(tickers, n_days=900, steps=78, seed=0, daily_vol=0.012) -> pd.DataFrame:
    """Synthetic OHLC bars from intraday random walks with slowly varying volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    frames = []
    for k, ticker in enumerate(tickers):
        log_vol = np.zeros(n_days)
        for t in range(1, n_days):
            log_vol[t] = 0.97 * log_vol[t - 1] + 0.15 * rng.standard_normal()
        sigma = daily_vol * (1 + 0.3 * k) * np.exp(log_vol - log_vol.mean())
        overnight = 0.4 * sigma * rng.standard_normal(n_days)
        intraday = (sigma[:, None] * np.sqrt(0.84 / steps)) * rng.standard_normal((n_days, steps))
        path = np.cumsum(intraday, axis=1)
        open_log = np.zeros(n_days)
        close_log = np.zeros(n_days)
        prev_close = np.log(100.0)
        for t in range(n_days):
            open_log[t] = prev_close + overnight[t]
            close_log[t] = open_log[t] + path[t, -1]
            prev_close = close_log[t]
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": np.exp(open_log),
                    "high": np.exp(open_log + np.maximum(path.max(axis=1), 0)),
                    "low": np.exp(open_log + np.minimum(path.min(axis=1), 0)),
                    "close": np.exp(close_log),
                    "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
                    "true_var": sigma**2,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    return simulate_bars(["SPY", "TLT", "AAPL"])


@pytest.fixture(scope="session")
def aux(bars) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(bars["date"].unique()))
    rng = np.random.default_rng(1)
    return pd.DataFrame({"vix": 18 + rng.standard_normal(len(dates)), "rf": 0.0001}, index=dates)


@pytest.fixture(scope="session")
def panel(bars, aux) -> pd.DataFrame:
    from volcast.features import build_panel

    return build_panel(bars.drop(columns="true_var"), aux)
