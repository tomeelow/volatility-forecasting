"""Streamlit dashboard: explore forecasts, compare models, play with vol targeting.

Run with:  uv run volcast app      (or: uv run streamlit run app/app.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme

from volcast import data, dm, evaluate, features, voltarget, walkforward
from volcast.config import SETTINGS, TRADING_DAYS, UNIVERSE
from volcast.metrics import loss_frame
from volcast.models import MODEL_LABELS
from volcast.portfolio import VolTargetConfig, rolling_vol

st.set_page_config(page_title="Volatility Forecasting Lab", page_icon=":material/show_chart:", layout="wide")

MODELS = list(MODEL_LABELS)
BENCH = SETTINGS.benchmark
LABELS = voltarget.STRATEGY_LABELS
SHORT = {"rw": "RW", "ewma": "EWMA", "garch": "GARCH", "gjr": "GJR", "har": "HAR", "har_log": "logHAR",
         "lhar": "LHAR", "gbm": "LGBM", "ens": "ENS"}
HORIZON_NAMES = {1: "1 day", 5: "5 days (1 week)", 22: "22 days (1 month)"}
CLASSES = sorted(set(UNIVERSE.values()))
PAGES = {
    "Overview": "overview", "Forecast explorer": "explorer", "Model comparison": "comparison",
    "Vol targeting": "voltarget", "Latest forecasts": "latest",
}


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading forecasts…")
def load_forecasts() -> pd.DataFrame:
    return walkforward.load_forecasts()


@st.cache_data(show_spinner="Loading market data…")
def load_market():
    returns = data.simple_returns(data.load_prices())
    rf = data.load_aux()["rf"]
    panel = pd.read_parquet(features.PANEL_PATH, columns=["date", "ticker", "r2", "rv", "rv_m"])
    calib = voltarget.calibration(panel)
    trailing = panel.pivot(index="date", columns="ticker", values="rv_m").iloc[-1]
    return returns, rf, calib, trailing


@st.cache_data(show_spinner=False)
def subset(horizon: int, classes: tuple[str, ...], years: tuple[int, int]) -> pd.DataFrame:
    fc = evaluate.prepare(load_forecasts(), horizon)
    keep = fc["ticker"].map(UNIVERSE).isin(classes) & fc["date"].dt.year.between(*years)
    return fc[keep].reset_index(drop=True)


@st.cache_data(show_spinner="Scoring models…")
def board(horizon: int, classes: tuple[str, ...], years: tuple[int, int], with_mcs: bool = False) -> pd.DataFrame:
    return evaluate.leaderboard(subset(horizon, classes, years), horizon, BENCH, with_mcs=with_mcs)


@st.cache_data(show_spinner="Running backtests…")
def run_voltarget(tickers: tuple[str, ...], models: tuple[str, ...], horizon: int, cfg: VolTargetConfig):
    returns, rf, calib, _ = load_market()
    forecasts = load_forecasts()
    returns = returns.loc[returns.index >= forecasts["date"].min(), list(tickers)]
    return voltarget.compare(returns, forecasts, rf, calib, list(models), horizon, cfg)


def ann_vol(variance) -> np.ndarray:
    """Daily variance -> annualised volatility in percent."""
    return np.sqrt(np.asarray(variance, dtype=float) * TRADING_DAYS) * 100.0


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


if not walkforward.FORECASTS_PATH.exists():
    st.title("Volatility Forecasting Lab")
    st.error("No forecasts found. Run the pipeline first, then reload this page.")
    st.code("uv run volcast all", language="bash")
    st.stop()

forecasts = load_forecasts()
YEAR_MIN, YEAR_MAX = int(forecasts["date"].dt.year.min()), int(forecasts["date"].dt.year.max())
ALL_YEARS = (YEAR_MIN, YEAR_MAX)

# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Volatility Forecasting Lab")
    # Deep links: ?view=voltarget&h=5 opens a given page / horizon.
    query = st.query_params
    page = st.radio(
        "View", list(PAGES), label_visibility="collapsed",
        index=list(PAGES.values()).index(query.get("view")) if query.get("view") in PAGES.values() else 0,
    )
    st.divider()
    horizons = list(HORIZON_NAMES)
    horizon = st.selectbox(
        "Forecast horizon", horizons, format_func=HORIZON_NAMES.get,
        index=horizons.index(int(query["h"])) if query.get("h", "").isdigit() and int(query["h"]) in horizons else 0,
        help="Every model forecasts the average daily variance over the next h trading days.",
    )
    st.caption(
        f"{forecasts['ticker'].nunique()} assets · out-of-sample "
        f"{forecasts['date'].min():%b %Y} – {forecasts['date'].max():%b %Y} · "
        f"models refit every {SETTINGS.refit_every} trading days on an expanding window."
    )


# ----------------------------------------------------------------------- charts
def leaderboard_chart(table: pd.DataFrame) -> go.Figure:
    t = theme.tokens()
    d = table.sort_values("QLIKE", ascending=False)
    change = (d["QLIKE vs bench"] - 1.0) * 100.0
    star = np.where(d["DM p-value"] < 0.05, " *", "")
    fig = go.Figure(
        go.Bar(
            x=change, y=[LABELS[m] for m in d.index], orientation="h", width=0.55,
            marker={"color": [t["muted"] if m == BENCH else t["blue"] for m in d.index], "cornerradius": 4},
            text=[f"{v:+.1f}%{s}" if m != BENCH else "benchmark" for m, v, s in zip(d.index, change, star, strict=True)],
            textposition="outside", textfont={"color": t["ink2"], "size": 12}, cliponaxis=False,
            customdata=d["QLIKE"], hovertemplate="%{y}<br>QLIKE %{customdata:.4f}<br>%{x:+.1f}% vs HAR<extra></extra>",
        )
    )
    theme.style(fig, height=390, legend=False, unified_hover=False)
    pad = max(8.0, 0.30 * (change.max() - change.min()))
    fig.update_xaxes(title_text="QLIKE change vs HAR benchmark, %  (lower is better)", showgrid=True,
                     zeroline=True, zerolinecolor=t["axis"], range=[change.min() - pad, change.max() + pad])
    fig.update_yaxes(showgrid=False)
    return fig


def heatmap(z: pd.DataFrame, zmax: float, hover: str, height: int = 360, decimals: int = 0) -> go.Figure:
    """Diverging heatmap centred on zero: cool = row model better, warm = worse."""
    values = z.to_numpy(dtype=float)
    shown = np.round(values, decimals) + 0.0  # "+ 0.0" turns -0.0 into 0.0
    text = [["" if np.isnan(v) else (f"{v:.{decimals}f}" if abs(v) < 10 else f"{v:.0f}") for v in row] for row in shown]
    fig = go.Figure(
        go.Heatmap(
            z=values, x=[str(c) for c in z.columns], y=list(z.index),
            colorscale=theme.diverging_scale(), zmid=0, zmin=-zmax, zmax=zmax,
            text=text, texttemplate="%{text}", textfont={"size": 11}, xgap=2, ygap=2,
            hovertemplate=hover, colorbar={"thickness": 10, "len": 0.8, "outlinewidth": 0},
        )
    )
    theme.style(fig, height=height, legend=False, unified_hover=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", type="category", tickangle=0)
    return fig


def format_board(table: pd.DataFrame):
    out = table.rename(index=LABELS).rename_axis("Model")
    # The benchmark against itself: no difference, by definition.
    out.loc[LABELS[BENCH], ["DM stat", "DM p-value"]] = [0.0, 1.0]
    fmt = {
        "QLIKE": "{:.4f}", "QLIKE vs bench": "{:.3f}", "MSE vs bench": "{:.3f}", "R2 (log)": "{:.3f}",
        "DM stat": "{:+.2f}", "DM p-value": "{:.3f}", "MCS p-value": "{:.3f}",
    }
    return out.style.format({k: v for k, v in fmt.items() if k in out.columns}, na_rep="–")


# ====================================================================== Overview
if page == "Overview":
    st.title("Can volatility be forecast? Yes — and here is by how much")
    st.markdown(
        "Unlike prices, volatility is persistent and therefore predictable. This lab compares nine "
        "forecasting models — from a random walk to GARCH, HAR and gradient boosting — on "
        f"**{forecasts['ticker'].nunique()} assets** in a strict walk-forward test: every forecast "
        "is made with data available at the time, and judged on what happened next."
    )
    table = board(horizon, tuple(CLASSES), ALL_YEARS, with_mcs=True)
    best = table["QLIKE"].idxmin()
    n_obs = len(subset(horizon, tuple(CLASSES), ALL_YEARS))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", forecasts["ticker"].nunique())
    c2.metric("Out-of-sample forecasts", f"{n_obs / 1e3:,.0f}K", help="Per model, at the selected horizon.")
    c3.metric("Best model", LABELS[best])
    c4.metric(
        "QLIKE vs HAR benchmark", f"{(table.loc[best, 'QLIKE vs bench'] - 1) * 100:+.1f}%",
        help="Relative reduction in out-of-sample QLIKE loss achieved by the best model.",
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(f"Forecast accuracy, {HORIZON_NAMES[horizon]} ahead")
        show(leaderboard_chart(table))
        st.caption("\\* significantly different from HAR at the 5% level (panel Diebold-Mariano test).")
    with right:
        st.subheader("How to read this")
        st.markdown(
            "- **QLIKE** is the loss: `RV/F − log(RV/F) − 1`. It is zero for a perfect forecast, "
            "depends only on the ratio realised/forecast, and punishes *under*-predicting risk more "
            "than over-predicting it.\n"
            "- **HAR** (Corsi, 2009) is the standard academic benchmark; bars show each model's loss "
            "relative to it.\n"
            "- **Diebold-Mariano** tests whether a difference in loss is statistically real, with "
            "standard errors robust to autocorrelation and to shocks that hit all assets at once.\n"
            "- **MCS p-value**: models above 0.10 cannot be told apart from the best one."
        )
    st.subheader("Leaderboard")
    st.dataframe(format_board(table), width="stretch")
    st.caption(
        "“Sig. better / worse” count the assets where the model beats / loses to HAR at the 5% level "
        "in per-asset Diebold-Mariano tests. Use the sidebar to switch horizon."
    )

# ============================================================== Forecast explorer
elif page == "Forecast explorer":
    st.title("Forecast explorer")
    periods = {
        "Full sample": (None, None),
        "Last 2 years": (forecasts["date"].max() - pd.DateOffset(years=2), None),
        "2011 euro crisis": ("2011-06-01", "2012-01-31"),
        "2015–16 China / oil": ("2015-07-01", "2016-04-30"),
        "2018 Volmageddon": ("2018-01-01", "2018-06-30"),
        "2020 COVID crash": ("2020-01-01", "2020-09-30"),
        "2022 bear market": ("2022-01-01", "2022-12-31"),
    }
    c1, c2, c3 = st.columns([2, 3, 2])
    ticker = c1.selectbox("Asset", sorted(UNIVERSE), index=sorted(UNIVERSE).index("SPY"), format_func=lambda t: f"{t} · {UNIVERSE[t]}")
    chosen = c2.multiselect("Models", MODELS, default=["har", "gbm"], format_func=LABELS.get, max_selections=5)
    period = c3.selectbox("Period", list(periods), index=5)
    log_scale = st.toggle("Log scale", value=False)

    fc = forecasts[(forecasts["ticker"] == ticker) & (forecasts["horizon"] == horizon)].sort_values("date")
    start, end = periods[period]
    view = fc[(fc["date"] >= pd.Timestamp(start or fc["date"].min())) & (fc["date"] <= pd.Timestamp(end or fc["date"].max()))]

    latest = fc.iloc[-1]
    if chosen:
        cols = st.columns(len(chosen))
        for col, m in zip(cols, chosen, strict=True):
            col.metric(f"{LABELS[m]} · next {HORIZON_NAMES[horizon]}", f"{ann_vol(latest[m]):.1f}%",
                       help=f"Annualised volatility forecast made at the close of {latest['date']:%d %b %Y}.")

    t = theme.tokens()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["date"], y=ann_vol(view["target"]), name=f"Realised (next {HORIZON_NAMES[horizon]})",
        mode="lines", line={"color": t["muted"], "width": 1}, fill="tozeroy" if not log_scale else None,
        fillcolor="rgba(137,135,129,0.14)", hovertemplate="%{y:.1f}%",
    ))
    for m in chosen:
        fig.add_trace(go.Scatter(x=view["date"], y=ann_vol(view[m]), name=LABELS[m], mode="lines",
                                 line=theme.line(m), hovertemplate="%{y:.1f}%"))
    theme.style(fig, height=440)
    fig.update_yaxes(title_text="Annualised volatility, %", type="log" if log_scale else "linear", rangemode="tozero")
    show(fig)
    st.caption(
        "Each point is dated at the forecast origin: the forecast made that evening, and the volatility "
        "that was then realised over the following days. Realised volatility is a range-based estimate "
        "from daily open/high/low/close bars."
    )

    scored = view.dropna(subset=["target"]).reset_index(drop=True)
    if len(scored) >= 30:
        st.subheader(f"Accuracy on {ticker}, {period.lower()}")
        ql = loss_frame(scored, MODELS, "qlike")
        rows = []
        for m in MODELS:
            test = dm.dm_test(ql[m], ql[BENCH], horizon) if m != BENCH else None
            rows.append({
                "Model": LABELS[m], "QLIKE": ql[m].mean(), "vs HAR": ql[m].mean() / ql[BENCH].mean(),
                "DM stat": test.stat if test else np.nan, "DM p-value": test.pvalue if test else np.nan,
            })
        acc = pd.DataFrame(rows).set_index("Model").sort_values("QLIKE")
        st.dataframe(
            acc.style.format({"QLIKE": "{:.4f}", "vs HAR": "{:.3f}", "DM stat": "{:+.2f}", "DM p-value": "{:.3f}"}, na_rep="–"),
            width="stretch",
        )
        st.caption(f"{len(scored):,} forecasts. Negative DM statistic = lower loss than HAR on this asset and period.")

# =============================================================== Model comparison
elif page == "Model comparison":
    st.title("Model comparison")
    c1, c2 = st.columns([3, 2], gap="large")
    classes = tuple(c1.multiselect("Asset classes", CLASSES, default=CLASSES) or CLASSES)
    years = c2.slider("Years", YEAR_MIN, YEAR_MAX, ALL_YEARS)
    with_mcs = st.toggle("Model confidence set (bootstrap, slower)", value=False)

    fc = subset(horizon, classes, years)
    if fc["date"].nunique() < 60:
        st.warning("Too few observations for this filter.")
        st.stop()
    table = board(horizon, classes, years, with_mcs)
    st.caption(f"{len(fc):,} forecasts per model · {fc['ticker'].nunique()} assets · {HORIZON_NAMES[horizon]} ahead")
    st.dataframe(format_board(table), width="stretch")

    ql = loss_frame(fc, MODELS, "qlike")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Diebold-Mariano matrix")
        stat, _ = dm.dm_matrix(ql, fc["date"], MODELS, horizon)
        stat = stat.rename(index=LABELS, columns=SHORT)
        fig = heatmap(stat, 6.0, "%{y} vs %{x}<br>DM stat %{z:.2f}<extra></extra>", height=420, decimals=1)
        fig.update_xaxes(tickangle=-90)
        show(fig)
        st.caption("Row model vs column model (same order). Blue (negative): the row model has the lower loss. |stat| > 1.96 is significant at 5%.")
    with right:
        st.subheader("Where does the gain come from?")
        by_class = (evaluate.qlike_by(fc, fc["ticker"].map(UNIVERSE), BENCH).T - 1.0) * 100.0
        by_class = by_class.drop(index=BENCH).rename(index=LABELS)
        by_class.columns = [c.replace("Stock: ", "") + (" stocks" if c.startswith("Stock") else "") for c in by_class.columns]
        fig = heatmap(by_class, 30.0, "%{y} · %{x}<br>%{z:+.1f}% QLIKE vs HAR<extra></extra>", height=420)
        fig.update_xaxes(tickangle=-90)
        show(fig)
        st.caption("QLIKE change vs HAR, %, by asset class. Blue: better than HAR.")

    st.subheader("Stability through time")
    by_year = (evaluate.qlike_by(fc, fc["date"].dt.year, BENCH).T - 1.0) * 100.0
    by_year = by_year.drop(index=BENCH).rename(index=LABELS)
    show(heatmap(by_year, 30.0, "%{y} · %{x}<br>%{z:+.1f}% QLIKE vs HAR<extra></extra>", height=360))
    st.caption("QLIKE change vs HAR, %, by calendar year of the forecast. A model that is blue in every column wins consistently, not just in one episode.")

    st.subheader("What the boosting model looks at")
    imp = walkforward.load_importance()[f"h{horizon}"].sort_values().tail(12) * 100.0
    fig = go.Figure(go.Bar(
        x=imp.to_numpy(), y=list(imp.index), orientation="h", width=0.55,
        marker={"color": theme.tokens()["blue"], "cornerradius": 4},
        text=[f"{v:.1f}%" for v in imp], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}% of total gain<extra></extra>",
    ))
    theme.style(fig, height=380, legend=False, unified_hover=False)
    fig.update_xaxes(title_text="Share of total split gain, % (average over all refits)", showgrid=True, range=[0, imp.max() * 1.15])
    fig.update_yaxes(showgrid=False)
    show(fig)
    st.caption(
        "f_d / f_w / f_q / f_y: daily, weekly, quarterly, yearly variance relative to the monthly level · "
        "mkt_*: cross-sectional median of the same · spy_*: the same for SPY · vrp: implied (VIX) vs realised variance · "
        "z_*: standardised past returns · vov: volatility of volatility · on_share: overnight share of variance."
    )

# ================================================================== Vol targeting
elif page == "Vol targeting":
    st.title("Volatility targeting")
    st.markdown(
        "Scale exposure every rebalance so that the portfolio's **forecast** volatility equals a target: "
        "`leverage = min(cap, target / forecast)`. Better forecasts keep realised risk closer to the target."
    )
    presets = {
        "SPY only": ["SPY"],
        "SPY + TLT": ["SPY", "TLT"],
        "Multi-asset ETFs (10)": ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"],
        f"All {len(UNIVERSE)} assets": list(UNIVERSE),
        "Custom…": None,
    }
    freq = {"Daily (1-day forecast)": (1, 1), "Weekly (5-day forecast)": (5, 5), "Monthly (22-day forecast)": (21, 22)}

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        preset = st.selectbox("Portfolio", list(presets))
        tickers = presets[preset] or st.multiselect("Assets", sorted(UNIVERSE), default=["SPY", "TLT", "GLD"])
        weighting = st.radio("Base weights", ["equal", "inverse_vol"], horizontal=True,
                             format_func={"equal": "Equal", "inverse_vol": "Inverse volatility"}.get)
    with c2:
        target = st.slider("Target volatility, % p.a.", 4, 25, 10) / 100.0
        max_lev = st.slider("Maximum leverage", 1.0, 4.0, 2.0, 0.25)
        cost = st.slider("Trading cost, bps of turnover", 0.0, 20.0, 2.0, 0.5)
    with c3:
        rebalance = st.selectbox("Rebalance", list(freq))
        lag = st.radio("Execution", [0, 1], horizontal=True, format_func={0: "Same close", 1: "Next close"}.get,
                       help="Same close: trade at the close the forecast is computed from. Next close: one day later (more conservative).")
        strategies = st.multiselect("Compare", list(LABELS), default=["unmanaged", "ewma", "gbm"], format_func=LABELS.get, max_selections=6)

    if not tickers or not strategies:
        st.info("Pick at least one asset and one strategy.")
        st.stop()
    every, fc_horizon = freq[rebalance]
    cfg = VolTargetConfig(target_vol=target, max_leverage=max_lev, rebalance_every=every, cost_bps=cost,
                          weighting=weighting, exec_lag=lag)
    summary, runs = run_voltarget(tuple(tickers), tuple(strategies), fc_horizon, cfg)

    managed = [s for s in strategies if s != "unmanaged"]
    if managed:
        focus, row = managed[-1], summary.loc[managed[-1]]
        base = summary.loc["unmanaged"] if "unmanaged" in summary.index else None
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Realised vol · {LABELS[focus]}", f"{row['Ann. vol'] * 100:.1f}%", f"target {target * 100:.0f}%", delta_color="off", delta_arrow="off")
        k2.metric("Sharpe ratio", f"{row['Sharpe']:.2f}", f"{row['Sharpe'] - base['Sharpe']:+.2f} vs unmanaged" if base is not None else None)
        k3.metric("Max drawdown", f"{row['Max drawdown'] * 100:.1f}%",
                  f"{(row['Max drawdown'] - base['Max drawdown']) * 100:+.1f} pts vs unmanaged" if base is not None else None)
        k4.metric("Vol tracking error", f"{row['Vol tracking error'] * 100:.2f} pts",
                  help="Average absolute gap between rolling 63-day realised volatility and the target.")

    t = theme.tokens()
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Growth of $1")
        fig = go.Figure()
        for s in strategies:
            fig.add_trace(go.Scatter(x=runs[s].index, y=runs[s]["equity"], name=LABELS[s], mode="lines",
                                     line=theme.line(s), hovertemplate="$%{y:.2f}"))
        theme.style(fig, height=380)
        fig.update_yaxes(type="log", dtick="D2", title_text="Portfolio value (log scale)")
        show(fig)
    with right:
        st.subheader("Realised volatility, rolling 3 months")
        fig = go.Figure()
        for s in strategies:
            fig.add_trace(go.Scatter(x=runs[s].index, y=rolling_vol(runs[s]["ret"]) * 100, name=LABELS[s], mode="lines",
                                     line=theme.line(s), hovertemplate="%{y:.1f}%"))
        span = runs[strategies[0]].index
        fig.add_trace(go.Scatter(x=[span[0], span[-1]], y=[target * 100] * 2, name=f"Target {target * 100:.0f}%", mode="lines",
                                 line={"color": t["ink"], "width": 1, "dash": "dash"}, hoverinfo="skip"))
        theme.style(fig, height=380)
        fig.update_yaxes(title_text="Annualised volatility, %", rangemode="tozero")
        show(fig)

    if managed:
        st.subheader("Leverage")
        fig = go.Figure()
        for s in managed:
            fig.add_trace(go.Scatter(x=runs[s].index, y=runs[s]["held_leverage"], name=LABELS[s], mode="lines",
                                     line=theme.line(s, 1.5), hovertemplate="%{y:.2f}x"))
        theme.style(fig, height=280)
        fig.update_yaxes(title_text="Gross exposure, x", rangemode="tozero")
        show(fig)

    st.subheader("Summary")
    pct = ["Ann. return", "Ann. vol", "Max drawdown", "Vol of vol", "Worst day", "Cost / yr", "Vol tracking error"]
    fmt = {**{c: "{:.2%}" for c in pct}, "Sharpe": "{:.2f}", "Avg leverage": "{:.2f}", "Turnover / yr": "{:.1f}", "Std of scaled returns": "{:.2f}"}
    shown = summary.rename(index=LABELS).rename_axis("Strategy")
    # A text column so the unmanaged row can show a dash instead of an empty cell.
    shown["Std of scaled returns"] = shown["Std of scaled returns"].map(lambda v: "–" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(shown.style.format({k: v for k, v in fmt.items() if k != "Std of scaled returns"}), width="stretch")
    st.caption(
        f"{len(tickers)} asset(s), {runs[strategies[0]].index[0]:%b %Y} – {runs[strategies[0]].index[-1]:%b %Y}. "
        "Returns include trading costs and financing at the T-bill rate. Portfolio variance = forecast volatilities × "
        "EWMA correlations. “Std of scaled returns” should be 1.0 if forecasts are well calibrated. "
        "The oracle uses the variance that was actually realised — an unattainable upper bound."
    )

# =============================================================== Latest forecasts
else:
    st.title("Latest forecasts")
    _, _, _, trailing = load_market()
    model = st.columns(3)[0].selectbox("Model", MODELS, index=MODELS.index("ens"), format_func=LABELS.get)
    last = forecasts[forecasts["date"] == forecasts["date"].max()]
    wide = last.pivot(index="ticker", columns="horizon", values=model)
    out = pd.DataFrame({
        "Asset class": pd.Series(UNIVERSE),
        "Realised, last 22 days": ann_vol(trailing.reindex(wide.index)),
        **{f"Forecast, next {h}d": ann_vol(wide[h]) for h in wide.columns},
    }, index=wide.index)
    out["Forecast vs realised"] = out[f"Forecast, next {horizon}d"] / out["Realised, last 22 days"] - 1.0
    out = out.sort_values(f"Forecast, next {horizon}d", ascending=False).rename_axis("Asset")
    vol_cols = [c for c in out.columns if c.startswith(("Realised", "Forecast, "))]
    st.caption(
        f"Annualised volatility, %, forecast at the close of {forecasts['date'].max():%d %b %Y}. "
        "Refresh with `uv run volcast all --force`."
    )
    st.dataframe(
        out.style.format({**{c: "{:.1f}" for c in vol_cols}, "Forecast vs realised": "{:+.0%}"}),
        width="stretch", height=720,
        column_config={c: st.column_config.Column(width="small") for c in out.columns},
    )
