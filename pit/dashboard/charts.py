"""Plotly figure builders for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pit.config.settings import TIMEFRAME_MINUTES
from pit.dashboard.components import (
    DIRECTION_COLOR,
    DIRECTION_SYMBOL,
    DOWN_COLOR,
    SERIES,
    UP_COLOR,
    WAIT_COLOR,
    style_fig,
)
from pit.features.frames import MarketFrames


def _ts(minutes) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(minutes, dtype="int64") * 60, unit="s", utc=True)


def resample_bars(candles: pd.DataFrame, tf: str) -> pd.DataFrame:
    if candles.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    frames = MarketFrames(candles, [tf])
    b = frames[tf]
    return pd.DataFrame({"time": _ts(b.start), "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                         "spread": b.spread})


def price_chart(candles: pd.DataFrame, tf: str, forecasts: pd.DataFrame | None = None, show_wait: bool = False,
                show_targets: bool = True, height: int = 560, title: str = "XAU/USD (mid)") -> tuple[go.Figure, pd.DataFrame]:
    bars = resample_bars(candles, tf)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=bars["time"], open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            name=f"XAU/USD {tf}", increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR, line_width=1,
        )
    )
    table = bars
    if forecasts is not None and not forecasts.empty:
        f = forecasts.copy()
        lo, hi = (int(candles["ts_min"].min()), int(candles["ts_min"].max()) + 1) if not candles.empty else (0, 0)
        f = f[(f["start_min"] >= lo) & (f["start_min"] < hi)]
        if not show_wait:
            f = f[f["direction"] != "WAIT"]
        for direction in ("UP", "DOWN", "WAIT"):
            sub = f[f["direction"] == direction]
            if sub.empty:
                continue
            hover = [
                f"<b>{r.direction}</b> (candidate {r.candidate_direction})<br>slot {r.slot_ts}<br>"
                f"ref {r.reference_price:.2f} → target {r.target_price if pd.notna(r.target_price) else '–'}<br>"
                f"score {r.composite_score if pd.notna(r.composite_score) else '–'} · P-HIT {r.p_hit if pd.notna(r.p_hit) else '–'}"
                f"<br>outcome {r.outcome_status or 'pending'}"
                for r in sub.itertuples()
            ]
            fig.add_trace(
                go.Scatter(
                    x=_ts(sub["start_min"]), y=sub["reference_price"], mode="markers",
                    name={"UP": "▲ UP forecast", "DOWN": "▼ DOWN forecast", "WAIT": "WAIT"}[direction],
                    marker=dict(symbol=DIRECTION_SYMBOL[direction], size=12, color=DIRECTION_COLOR[direction],
                                line=dict(width=2, color="rgba(255,255,255,0.9)")),
                    hovertext=hover, hoverinfo="text",
                )
            )
            if show_targets and direction != "WAIT":
                xs, ys = [], []
                for r in sub.itertuples():
                    if pd.notna(r.target_price):
                        xs += [_ts([r.start_min])[0], _ts([r.expiry_min])[0], None]
                        ys += [r.target_price, r.target_price, None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{direction} target",
                                         line=dict(color=DIRECTION_COLOR[direction], width=2, dash="dot"),
                                         hoverinfo="skip"))
        hits = f[(f["outcome_status"] == "hit") & f["first_touch_ts"].notna()]
        if not hits.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(hits["first_touch_ts"], utc=True), y=hits["target_price"], mode="markers",
                    name="target touched", marker=dict(symbol="circle", size=9, color=SERIES[2],
                                                       line=dict(width=2, color="rgba(255,255,255,0.9)")),
                    hovertext=[f"target {r.target_price:.2f} touched {r.first_touch_ts}" for r in hits.itertuples()],
                    hoverinfo="text",
                )
            )
        table = f[["slot_ts", "direction", "candidate_direction", "reference_price", "target_price", "p_hit",
                   "confidence", "outcome_status"]]
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    style_fig(fig, height=height, y_title="USD/oz", legend=len(fig.data) > 1)
    fig.update_layout(hovermode="closest")
    return fig, table


def components_chart(components: dict, weights: dict) -> go.Figure:
    names = list(components)
    vals = [components[k] if components[k] is not None else 0.0 for k in names]
    colors = [UP_COLOR if v > 0 else (DOWN_COLOR if v < 0 else WAIT_COLOR) for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h", marker_color=colors, width=0.5,
            text=[f"{v:+.2f}" for v in vals], textposition="outside",
            hovertext=[f"{k}: {v:+.3f} × weight {weights.get(k, 0):.2f} = {v * weights.get(k, 0):+.3f}" for k, v in zip(names, vals)],
            hoverinfo="text",
        )
    )
    fig.update_xaxes(range=[-1.15, 1.15])
    return style_fig(fig, height=300, x_title="normalised component (−1 … +1)", legend=False)


def reliability_chart(table: pd.DataFrame) -> go.Figure:
    t = table[table["n"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=WAIT_COLOR, width=1), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=t["mean_p_hit"], y=t["observed_rate"], mode="lines+markers", name="observed hit rate",
            line=dict(color=SERIES[0], width=2), marker=dict(size=9, color=SERIES[0]),
            error_y=dict(type="data", symmetric=False, array=t["ci_high"] - t["observed_rate"],
                         arrayminus=t["observed_rate"] - t["ci_low"], thickness=1, width=4),
            hovertext=[f"bin {r.bin}: n={r.n}, mean P-HIT {r.mean_p_hit:.2f}, observed {r.observed_rate:.2f}" for r in t.itertuples()],
            hoverinfo="text",
        )
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return style_fig(fig, height=380, x_title="mean heuristic P-HIT in bin", y_title="observed target-hit rate")


def bin_counts_chart(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=table["bin"], y=table["n"], marker_color=SERIES[0], width=0.4, name="forecasts"))
    return style_fig(fig, height=200, y_title="forecasts", legend=False)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, pct: bool = True, color: str = SERIES[0],
              reference: float | None = None, reference_label: str = "") -> go.Figure:
    vals = df[y]
    # Cap bars at roughly 24px: in a ~600px half-width plot a slot is 600/n px wide.
    width = min(0.6, max(0.08, 0.04 * len(df)))
    fig = go.Figure(
        go.Bar(x=df[x].astype(str), y=vals, marker_color=color, width=width,
               text=[("–" if pd.isna(v) else (f"{100 * v:.0f}%" if pct else f"{v:.2f}")) for v in vals],
               textposition="outside", name=y)
    )
    if reference is not None:
        fig.add_hline(y=reference, line_color=WAIT_COLOR, line_width=1,
                      annotation_text=reference_label, annotation_position="top left")
    fig.update_layout(title=title)
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, height=320, legend=False)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, y_title: str, color: str = SERIES[0],
               fill: bool = False) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=df[x], y=df[y], mode="lines", name=y, line=dict(color=color, width=2),
                   fill="tozeroy" if fill else None,
                   fillcolor=_rgba(color, 0.10) if fill else None)
    )
    fig.update_layout(title=title)
    return style_fig(fig, height=300, y_title=y_title, legend=False)


def target_distance_chart(table: pd.DataFrame) -> go.Figure:
    t = table[table["label"] != "mirror"].sort_values("atr_multiple")
    fig = go.Figure(
        go.Scatter(x=t["atr_multiple"], y=t["hit_rate"], mode="lines+markers+text", name="hit rate",
                   text=t["label"], textposition="top center", line=dict(color=SERIES[0], width=2),
                   marker=dict(size=9))
    )
    mirror = table[table["label"] == "mirror"]
    if not mirror.empty:
        fig.add_hline(y=float(mirror["hit_rate"].iloc[0]), line_color=WAIT_COLOR, line_width=1,
                      annotation_text="mirror target (control)", annotation_position="bottom left")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_layout(title="Target-hit rate by target distance")
    return style_fig(fig, height=320, x_title="target distance (× ATR14 1m)", y_title="hit rate", legend=False)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def tf_minutes(tf: str) -> int:
    return TIMEFRAME_MINUTES[tf]
