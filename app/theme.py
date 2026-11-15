"""Colours and Plotly styling shared by every chart in the dashboard.

Colour follows the entity: each model keeps the same colour on every chart,
whatever else is selected. The palette is a colour-blind-validated categorical
set (8 hues); the ensemble and the oracle are drawn in ink instead of a 9th hue.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#f0efec",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "div_cool": "#184f95", "div_warm": "#b32f2f",
}
DARK = {
    "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "neutral": "#383835",
    "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500",
    "magenta": "#d55181", "green": "#0ca30c", "violet": "#9085e9", "red": "#e66767",
    "div_cool": "#6da7ec", "div_warm": "#e66767",
}

# model / strategy key -> palette role
SERIES_ROLE = {
    "gbm": "blue", "lhar": "orange", "gjr": "aqua", "ewma": "yellow",
    "har_log": "magenta", "garch": "green", "har": "violet", "rw": "red",
    "ens": "ink", "oracle": "ink", "unmanaged": "muted", "realized": "muted",
}
SERIES_DASH = {"oracle": "dot"}


def tokens() -> dict[str, str]:
    """Palette for the viewer's current Streamlit theme."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit versions
        dark = False
    return DARK if dark else LIGHT


def color(key: str) -> str:
    return tokens()[SERIES_ROLE.get(key, "muted")]


def line(key: str, width: float = 2.0) -> dict:
    return {"color": color(key), "width": width, "dash": SERIES_DASH.get(key, "solid")}


def diverging_scale() -> list[list]:
    """Two opposite hues around a neutral midpoint (cool = better, warm = worse)."""
    t = tokens()
    return [[0.0, t["div_cool"]], [0.5, t["neutral"]], [1.0, t["div_warm"]]]


def style(fig: go.Figure, height: int = 380, legend: bool = True, unified_hover: bool = True) -> go.Figure:
    """Apply the shared look: recessive hairline grid, text in ink tokens, legend on top."""
    t = tokens()
    fig.update_layout(
        height=height,
        margin={"l": 56, "r": 24, "t": 40 if legend else 16, "b": 44},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "size": 13, "color": t["ink2"]},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title_text": ""},
        hovermode="x unified" if unified_hover else "closest",
        hoverlabel={"bgcolor": t["surface"], "font": {"color": t["ink"], "size": 12}, "bordercolor": t["axis"]},
    )
    axis = {
        "gridcolor": t["grid"], "gridwidth": 1, "linecolor": t["axis"], "zeroline": False,
        "tickfont": {"color": t["muted"], "size": 12}, "title_font": {"color": t["ink2"], "size": 12},
    }
    fig.update_xaxes(**axis, showgrid=False, automargin=True)
    fig.update_yaxes(**axis, automargin=True)
    return fig
