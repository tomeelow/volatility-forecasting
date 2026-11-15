import numpy as np
import pandas as pd
import pytest

from volcast import dm
from volcast import estimators as est
from volcast.metrics import mse, qlike


def test_range_estimators_recover_true_variance(bars):
    d = bars[bars["ticker"] == "SPY"].set_index("date")
    rv = est.gk_overnight(d["open"], d["high"], d["low"], d["close"]).dropna()
    true = d["true_var"].reindex(rv.index)
    # Discrete sampling of the high/low biases range estimators slightly downward.
    assert rv.mean() / true.mean() == pytest.approx(1.0, abs=0.15)
    assert (est.garman_klass(d["open"], d["high"], d["low"], d["close"]) >= 0).all()


def test_range_proxy_is_less_noisy_than_squared_returns(bars):
    d = bars[bars["ticker"] == "SPY"].set_index("date")
    rv = est.gk_overnight(d["open"], d["high"], d["low"], d["close"]).dropna()
    r2 = est.squared_return(d["close"]).dropna()
    true = d["true_var"].reindex(rv.index)
    assert ((rv - true) ** 2).mean() < 0.5 * ((r2 - true) ** 2).mean()


def test_floor_variance_only_lifts_tiny_values():
    rv = pd.Series(np.full(300, 1e-4))
    rv.iloc[150] = 0.0
    floored = est.floor_variance(rv)
    assert floored.iloc[150] == pytest.approx(1e-6)
    assert (floored.drop(150) == 1e-4).all()


def test_qlike_properties():
    assert qlike(2.0, 2.0) == pytest.approx(0.0)
    y = np.array([1.0, 1.0])
    f = np.array([0.5, 2.0])
    loss = qlike(y, f)
    assert (loss > 0).all()
    assert loss[0] > loss[1]  # under-prediction costs more than over-prediction
    # Scale invariance: only the ratio matters.
    assert qlike(3.0, 6.0) == pytest.approx(qlike(1.0, 2.0))
    assert mse(3.0, 1.0) == pytest.approx(4.0)


def test_qlike_is_minimised_by_the_conditional_mean():
    rng = np.random.default_rng(0)
    y = rng.chisquare(1, 200_000) * 2.0  # noisy proxy with mean 2
    grid = np.linspace(1.0, 3.0, 41)
    best = grid[np.argmin([qlike(y, f).mean() for f in grid])]
    assert best == pytest.approx(2.0, abs=0.06)


def test_newey_west_matches_variance_without_lags():
    x = np.random.default_rng(0).standard_normal(500)
    assert dm.newey_west_lrv(x, 0) == pytest.approx(x.var())


def test_dm_detects_a_better_forecast():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(1000) ** 2
    res = dm.dm_test(base, base + 0.2 + 0.5 * rng.standard_normal(1000))
    assert res.stat < -5 and res.pvalue < 1e-6
    assert res.mean_diff < 0


def test_dm_size_under_the_null_with_autocorrelated_losses():
    rng = np.random.default_rng(1)
    rejections, trials, n = 0, 400, 600
    for _ in range(trials):
        e = rng.standard_normal(n + 1)
        d = e[1:] + 0.6 * e[:-1]  # MA(1) loss differential with zero mean
        rejections += dm.dm_test(d, np.zeros(n)).pvalue < 0.05
    assert 0.02 < rejections / trials < 0.10


def test_panel_dm_averages_across_assets_first():
    rng = np.random.default_rng(2)
    dates = pd.Series(np.repeat(pd.bdate_range("2020-01-01", periods=400), 5))
    common = np.repeat(rng.standard_normal(400), 5)  # shock shared by all assets
    losses = pd.DataFrame({"a": common + 0.1 * rng.standard_normal(2000), "b": common})
    res = dm.panel_dm(losses, dates, "a", "b")
    assert res.n == 400
    assert abs(res.stat) < 3
