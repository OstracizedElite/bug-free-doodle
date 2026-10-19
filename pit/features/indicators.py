"""Deterministic indicator primitives operating on NumPy arrays.

Every function uses only the values passed in (which callers restrict to completed bars),
returns ``None`` when there is insufficient data instead of a partial warm-up value, and
has an exact definition documented in ``docs/methodology.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter, lfiltic


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """TR_i = max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|) for i = 1..n-1 (length n-1)."""
    if close.size < 2:
        return np.zeros(0)
    prev = close[:-1]
    h, l = high[1:], low[1:]
    return np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float | None:
    """Simple average of the last ``period`` true ranges (needs ``period + 1`` bars)."""
    tr = true_range(high, low, close)
    if tr.size < period:
        return None
    return float(tr[-period:].mean())


def mean_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, bars: int) -> float | None:
    """Average TR over up to the last ``bars`` bars (baseline for volatility expansion)."""
    tr = true_range(high, low, close)
    if tr.size == 0:
        return None
    return float(tr[-bars:].mean())


def sma(values: np.ndarray, period: int) -> float | None:
    if values.size < period:
        return None
    return float(values[-period:].mean())


def ema_series(values: np.ndarray, period: int) -> np.ndarray | None:
    """EMA over the given window, seeded with the SMA of its first ``period`` values.

    Returns the EMA for indices ``period-1 .. n-1`` (length ``n - period + 1``).
    alpha = 2 / (period + 1); EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}.
    """
    n = values.size
    if n < period:
        return None
    alpha = 2.0 / (period + 1.0)
    seed = float(values[:period].mean())
    rest = values[period:]
    if rest.size == 0:
        return np.array([seed])
    b, a = [alpha], [1.0, alpha - 1.0]
    tail = lfilter(b, a, rest, zi=lfiltic(b, a, [seed]))[0]
    return np.concatenate([[seed], tail])


def log_returns(close: np.ndarray, start: np.ndarray | None = None, max_step: int | None = None) -> np.ndarray:
    """Close-to-close log returns. If ``start``/``max_step`` given, drop returns spanning a
    time gap larger than ``max_step`` minutes (e.g. across a market close)."""
    if close.size < 2:
        return np.zeros(0)
    r = np.diff(np.log(close))
    if start is not None and max_step is not None:
        r = r[np.diff(start) <= max_step]
    return r


def directional_consistency(close: np.ndarray, bars: int) -> float | None:
    """(count of up closes - count of down closes) / bars over the last ``bars`` changes."""
    if close.size < bars + 1:
        return None
    d = np.diff(close[-(bars + 1):])
    return float((np.sum(d > 0) - np.sum(d < 0)) / bars)


def up_fraction(close: np.ndarray, bars: int) -> float | None:
    if close.size < bars + 1:
        return None
    d = np.diff(close[-(bars + 1):])
    return float(np.sum(d > 0) / bars)


def efficiency_ratio(close: np.ndarray, bars: int) -> float | None:
    """Signed efficiency ratio: net change / sum of absolute changes, in [-1, 1].

    A volatility-adjusted measure of directional consistency: +1 means every bar moved up,
    0 means the path went nowhere relative to the distance travelled.
    """
    if close.size < bars + 1:
        return None
    window = close[-(bars + 1):]
    travelled = float(np.abs(np.diff(window)).sum())
    if travelled == 0:
        return 0.0
    return float((window[-1] - window[0]) / travelled)


def swing_points(high: np.ndarray, low: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of confirmed swing highs and lows (fractals).

    Bar i is a swing high when ``high[i]`` is strictly greater than the ``left`` preceding
    highs and greater than or equal to the ``right`` following highs; swing lows mirror this.
    Only bars with ``right`` completed bars after them can qualify, so no swing is
    confirmed using data that would not have existed at the time.
    """
    n = high.size
    w = left + right + 1
    if n < w:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    hw = sliding_window_view(high, w)
    lw = sliding_window_view(low, w)
    centre_h = hw[:, left]
    centre_l = lw[:, left]
    left_h = hw[:, :left].max(axis=1) if left else np.full(centre_h.shape, -np.inf)
    right_h = hw[:, left + 1:].max(axis=1) if right else np.full(centre_h.shape, -np.inf)
    left_l = lw[:, :left].min(axis=1) if left else np.full(centre_l.shape, np.inf)
    right_l = lw[:, left + 1:].min(axis=1) if right else np.full(centre_l.shape, np.inf)
    highs = np.flatnonzero((centre_h > left_h) & (centre_h >= right_h)) + left
    lows = np.flatnonzero((centre_l < left_l) & (centre_l <= right_l)) + left
    return highs, lows
