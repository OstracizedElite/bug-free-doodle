"""Shared dashboard styling and widgets.

Colour roles follow a validated palette (see docs/dashboard.md):
* polarity (UP / DOWN) uses the blue <-> red diverging pair, WAIT is neutral grey, and the
  direction is always also carried by a marker shape (triangle up / down) and a label;
* categorical identity (e.g. deterministic vs LLM-assisted) uses fixed slots in order;
* status colours are reserved for health states and always come with an icon and label.
Charts use one y-axis, 2px lines, hairline grids, and every chart has a table view.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

UP_COLOR = "#2a78d6"
DOWN_COLOR = "#e34948"
WAIT_COLOR = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "✅", "warning": "⚠️", "serious": "🟠", "critical": "⛔", "unknown": "❔"}
DIRECTION_COLOR = {"UP": UP_COLOR, "DOWN": DOWN_COLOR, "WAIT": WAIT_COLOR, "NONE": WAIT_COLOR}
DIRECTION_SYMBOL = {"UP": "triangle-up", "DOWN": "triangle-down", "WAIT": "circle-open", "NONE": "circle-open"}
DIRECTION_LABEL = {"UP": "▲ UP", "DOWN": "▼ DOWN", "WAIT": "■ WAIT", "NONE": "–"}


def style_fig(fig: go.Figure, height: int = 380, y_title: str | None = None, x_title: str | None = None,
              legend: bool = True) -> go.Figure:
    has_title = bool(fig.layout.title and fig.layout.title.text)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=72 if has_title else 40, b=8),
        hovermode="x unified" if fig.data and fig.data[0].type in ("scatter",) else "closest",
        # Legend sits on its own row under the title so the two never collide.
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0) if legend else None,
        showlegend=legend,
    )
    if has_title:
        fig.update_layout(title=dict(y=0.98, yanchor="top", x=0, xanchor="left", font=dict(size=15)))
    fig.update_layout(barcornerradius=4)
    fig.update_xaxes(showgrid=True, gridwidth=1, title_text=x_title)
    fig.update_yaxes(showgrid=True, gridwidth=1, title_text=y_title, zeroline=False)
    return fig


def show_chart(fig: go.Figure, table: pd.DataFrame | None = None, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key)
    if table is not None and not table.empty:
        with st.expander("Table view"):
            st.dataframe(table, width="stretch", hide_index=True)


def data_mode_banner(modes: Iterable[str], context: str = "") -> None:
    """Prominent label of what kind of data the page is showing."""
    modes = sorted({m for m in modes if m})
    suffix = f" — {context}" if context else ""
    if not modes:
        st.info(f"No market data in scope yet{suffix}. Run `python -m pit ingest` or `python -m pit demo`.")
        return
    if "synthetic" in modes:
        st.error(
            f"🧪 **SYNTHETIC DATA**{suffix}. These are generated prices, not market prices. "
            "Nothing shown here is evidence of real trading performance.",
        )
    if "historical" in modes:
        st.info(f"🗄️ **HISTORICAL DATA (replay)**{suffix}. Backtest results, not live forward testing.")
    if "delayed" in modes:
        st.warning(f"⏱️ **DELAYED DATA**{suffix}. Prices lag the market.")
    if "live" in modes:
        st.success(f"📡 **LIVE DATA**{suffix}. Research and paper trading only; no real orders are placed.")


def fmt_price(x, decimals: int = 2) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.{decimals}f}"


def fmt_pct(x, decimals: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{100 * x:.{decimals}f}%"


def fmt_ci(ci) -> str:
    if not ci or ci[0] is None:
        return "CI –"
    return f"95% CI {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%"


def fmt_num(x, decimals: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.{decimals}f}"


def direction_badge(direction: str) -> str:
    return DIRECTION_LABEL.get(direction, direction)


def status_badge(status: str) -> str:
    mapping = {"ok": "good", "closed": "good", "stale": "warning", "down": "critical", "unknown": "unknown"}
    level = mapping.get(status, "unknown")
    return f"{STATUS_ICON[level]} {status}"


def heuristic_note() -> None:
    st.caption(
        "P-HIT is an **uncalibrated heuristic** (phit-heuristic-v1), not a statistically calibrated probability. "
        "See the Performance page for its measured calibration."
    )
