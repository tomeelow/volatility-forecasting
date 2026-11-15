import numpy as np
import pandas as pd
import pytest

from volcast import walkforward as wf
from volcast.config import Settings
from volcast.features import GBM_FEATURES, HAR_FEATURES, LEVERAGE_FEATURES, build_panel
from volcast.models import GBM, HAR
from volcast.portfolio import VolTargetConfig, backtest, performance

PREDICTORS = ["ewma", *HAR_FEATURES, *LEVERAGE_FEATURES, *GBM_FEATURES]


def test_targets_are_strictly_in_the_future(panel):
    g = panel[panel["ticker"] == "SPY"].reset_index(drop=True)
    i = 100
    assert g.loc[i, "target_1"] == pytest.approx(g.loc[i + 1, "rv"])
    assert g.loc[i, "target_5"] == pytest.approx(g.loc[i + 1 : i + 5, "rv"].mean())
    assert g.loc[i, "tend_5"] == g.loc[i + 5, "date"]
    assert g["target_22"].tail(22).isna().all()


def test_features_do_not_look_ahead(bars, aux, panel):
    """Changing prices after a cutoff must not change any predictor before it."""
    cutoff = pd.Timestamp(sorted(bars["date"].unique())[600])
    shocked = bars.drop(columns="true_var").copy()
    after = shocked["date"] > cutoff
    shocked.loc[after, ["open", "high", "low", "close"]] *= 1.5
    shocked.loc[after, "high"] *= 1.2
    aux2 = aux.copy()
    aux2.loc[aux2.index > cutoff, "vix"] *= 2
    panel2 = build_panel(shocked, aux2)

    a = panel[panel["date"] <= cutoff].set_index(["date", "ticker"])[PREDICTORS]
    b = panel2[panel2["date"] <= cutoff].set_index(["date", "ticker"])[PREDICTORS]
    pd.testing.assert_frame_equal(a, b)
    # ... while the future does change, so the test has teeth.
    later = panel["date"] > cutoff + pd.Timedelta(days=5)
    assert not np.allclose(panel.loc[later, "rv_w"].to_numpy(), panel2.loc[panel2["date"] > cutoff + pd.Timedelta(days=5), "rv_w"].to_numpy())


def test_blocks_partition_the_out_of_sample_period(panel):
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    blocks = wf.make_blocks(dates, "2017-01-02", 63)
    oos = dates[dates >= "2017-01-02"]
    assert blocks[0][0] == oos[0] and blocks[-1][1] == oos[-1]
    covered = sum(((dates >= s) & (dates <= e)).sum() for s, e in blocks)
    assert covered == len(oos)
    assert all(blocks[i][1] < blocks[i + 1][0] for i in range(len(blocks) - 1))


@pytest.mark.parametrize("horizon", [1, 5, 22])
def test_split_never_trains_on_unrealised_targets(panel, horizon):
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    start, end = wf.make_blocks(dates, "2017-01-02", 63)[1]
    train, test = wf.split(panel, horizon, start, end, min_train_obs=100)
    assert train[f"tend_{horizon}"].max() <= start
    assert train["date"].max() < start
    assert test["date"].min() == start and test["date"].max() == end
    assert set(test["ticker"]) <= set(train["ticker"])


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 3000
    X = np.exp(rng.standard_normal((n, 3)) * 0.3) * 1e-4
    beta = np.array([2e-5, 0.3, 0.4, 0.2])
    df = pd.DataFrame(X, columns=HAR_FEATURES)
    df["ticker"] = "SPY"
    df["target_1"] = beta[0] + X @ beta[1:] + 1e-7 * rng.standard_normal(n)
    model = HAR("har")
    pred = model.fit_predict(df.iloc[:2500], df.iloc[2500:], horizon=1)
    assert model.coefs["SPY"][1:] == pytest.approx(beta[1:], abs=0.01)
    assert np.corrcoef(pred, df["target_1"].iloc[2500:])[0, 1] > 0.99


def test_models_produce_positive_finite_forecasts(panel):
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    start, end = wf.make_blocks(dates, "2017-06-01", 63)[0]
    train, test = wf.split(panel, 5, start, end, min_train_obs=100)
    models = [HAR("har"), HAR("har_log", log=True), HAR("lhar", log=True, leverage=True), GBM(num_boost_round=30)]
    for model in models:
        pred = model.fit_predict(train, test, horizon=5)
        assert np.isfinite(pred).all() and (pred > 0).all(), model.name


def test_walkforward_forecasts_beat_the_random_walk(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "FORECASTS_PATH", tmp_path / "fc.parquet")
    monkeypatch.setattr(wf, "IMPORTANCE_PATH", tmp_path / "imp.parquet")
    monkeypatch.setattr(wf, "ARTIFACTS_DIR", tmp_path)
    settings = Settings(oos_start="2017-01-02", horizons=(1,), refit_every=126, min_train_obs=200, n_jobs=1)
    fc = wf.run(panel, settings).dropna(subset=["target"])
    assert fc["ticker"].nunique() == 3
    from volcast.metrics import qlike

    losses = {m: qlike(fc["target"], fc[m]).mean() for m in ["rw", "har_log", "garch", "gbm"]}
    assert losses["har_log"] < losses["rw"]
    assert all(np.isfinite(v) for v in losses.values())


def _iid_returns(n=3000, vol=0.02, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"A": vol * rng.standard_normal(n), "B": vol * rng.standard_normal(n)}, index=dates)


def test_vol_targeting_hits_the_target_with_a_perfect_forecast():
    returns = _iid_returns()
    var = pd.DataFrame(0.02**2, index=returns.index, columns=returns.columns)
    rf = pd.Series(0.0, index=returns.index)
    cfg = VolTargetConfig(target_vol=0.10, max_leverage=5.0, cost_bps=0.0)
    res = backtest(returns, var, rf, cfg)
    stats = performance(res, cfg.target_vol)
    assert stats["Ann. vol"] == pytest.approx(0.10, rel=0.08)
    assert stats["Std of scaled returns"] == pytest.approx(1.0, rel=0.08)


def test_unmanaged_portfolio_has_unit_leverage_and_costs_reduce_returns():
    returns = _iid_returns()
    rf = pd.Series(0.0, index=returns.index)
    res = backtest(returns, None, rf, VolTargetConfig(cost_bps=0.0))
    assert res["held_leverage"].dropna().eq(1.0).all()
    # Equal weights rebalanced daily: the portfolio return is the cross-sectional mean.
    day = res.index[5]
    assert res.loc[day, "ret"] == pytest.approx(returns.loc[day].mean())

    var = pd.DataFrame(0.02**2, index=returns.index, columns=returns.columns)
    free = backtest(returns, var, rf, VolTargetConfig(cost_bps=0.0))
    costly = backtest(returns, var, rf, VolTargetConfig(cost_bps=50.0))
    assert costly["ret"].sum() < free["ret"].sum()


def test_leverage_is_capped_and_uses_only_past_information():
    returns = _iid_returns()
    rf = pd.Series(0.0, index=returns.index)
    var = pd.DataFrame(0.001**2, index=returns.index, columns=returns.columns)  # absurdly low forecast
    res = backtest(returns, var, rf, VolTargetConfig(max_leverage=1.5, cost_bps=0.0))
    assert res["leverage"].max() <= 1.5 + 1e-9

    # Changing a future return must not change earlier strategy returns.
    var = pd.DataFrame(0.02**2, index=returns.index, columns=returns.columns)
    base = backtest(returns, var, rf, VolTargetConfig())
    shocked = returns.copy()
    shocked.iloc[2000:] *= 3
    alt = backtest(shocked, var, rf, VolTargetConfig())
    cutoff = returns.index[1999]
    pd.testing.assert_series_equal(base.loc[:cutoff, "ret"], alt.loc[:cutoff, "ret"])
