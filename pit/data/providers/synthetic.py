"""Reproducible synthetic market data for development and testing.

SYNTHETIC DATA IS NOT MARKET DATA. Every candle produced here is labelled
``data_mode='synthetic'`` and ``quote_convention='synthetic_mid'`` and the dashboard shows
a prominent banner whenever synthetic data is in use. Results obtained on synthetic data
must never be presented as trading performance.

Model (per instrument, per minute ``t``):

* log-volatility ``v_t``: AR(1), stationary std ``vol_of_vol``, half-life ``vol_persistence``;
* drift ``d_t``: AR(1), stationary std ``drift_strength * sigma_base``, half-life
  ``drift_persistence`` - this creates trending and ranging episodes;
* return ``r_t = d_t + sigma_base * exp(v_t - vol_of_vol^2/2) * seasonal(hour) * eps_t`` with
  Student-t(5) innovations (fat tails), applied on open-market minutes only;
* spreads widen with volatility, in quiet sessions and just after the weekly open.

Note that the drift process deliberately embeds short-term autocorrelation so the
forecasting pipeline has something to find; positive results on synthetic data say
nothing about real markets.

Determinism: each UTC day is generated from ``default_rng([seed, instrument_offset, day])``
starting from the previous day's terminal state, so any window is reproducible
regardless of the order or granularity in which it is requested.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfiltic

from pit.common.timeutils import ensure_utc, to_epoch_min, utcnow
from pit.config.settings import SyntheticProviderConfig
from pit.data.market_calendar import MarketCalendar
from pit.data.models import ProviderCapabilities, Quote, conform_candle_frame
from pit.data.providers.base import InstrumentNotSupportedError, MarketDataProvider

TRADING_MINUTES_PER_YEAR = 252 * 23 * 60
_T_DF = 5.0
_T_SCALE = np.sqrt((_T_DF - 2.0) / _T_DF)  # unit-variance Student-t


@dataclass(frozen=True)
class _Spec:
    start_price: float
    annual_vol: float
    base_spread: float
    min_spread: float
    decimals: int
    seed_offset: int
    drift_strength: float


@dataclass(frozen=True)
class _DayState:
    logp: float
    v: float
    d: float
    prev_open: bool
    since_open: int  # open minutes since the market last (re)opened


class SyntheticProvider(MarketDataProvider):
    name = "synthetic"
    data_mode = "synthetic"
    quote_convention = "synthetic_mid"

    def __init__(
        self,
        cfg: SyntheticProviderConfig,
        calendar: MarketCalendar,
        instrument: str = "XAU/USD",
        fx_symbol: str = "AUD/USD",
        clock: Callable[[], datetime] = utcnow,
        cache_days: int = 96,
    ):
        self.cfg = cfg
        self.calendar = calendar
        self.clock = clock
        anchor_min = to_epoch_min(cfg.anchor)
        self.anchor_min = anchor_min - (anchor_min % 1440)
        self._specs = {
            instrument: _Spec(
                start_price=cfg.start_price,
                annual_vol=cfg.annual_vol,
                base_spread=cfg.base_spread,
                min_spread=0.05,
                decimals=2,
                seed_offset=0,
                drift_strength=cfg.drift_strength,
            ),
            fx_symbol: _Spec(
                start_price=cfg.fx_start_rate,
                annual_vol=0.09,
                base_spread=0.00012,
                min_spread=0.00005,
                decimals=5,
                seed_offset=1,
                drift_strength=cfg.drift_strength * 0.2,
            ),
        }
        self._states: dict[str, list[_DayState]] = {}
        self._cache: OrderedDict[tuple[str, int], pd.DataFrame] = OrderedDict()
        self._cache_days = cache_days
        self._mask_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ interface
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            instruments=tuple(self._specs),
            historical_candles=True,
            live_quotes=True,
            bid_ask_candles=True,
            tick_quotes=False,
            streaming=False,
            requires_credentials=False,
            quote_convention=self.quote_convention,
            notes="Deterministic synthetic data for development only. Not market data.",
        )

    def fetch_candles(self, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        now_min = to_epoch_min(self.clock())
        end_min = min(to_epoch_min(ensure_utc(end)), now_min)  # completed candles only
        return self.generate(instrument, to_epoch_min(ensure_utc(start)), end_min)

    def fetch_latest_quote(self, instrument: str) -> Quote | None:
        now_min = to_epoch_min(self.clock())
        frame = self.generate(instrument, now_min - 3 * 1440, now_min)
        if frame.empty:
            return None
        last = frame.iloc[-1]
        return Quote(
            instrument=instrument,
            source=self.name,
            ts=pd.Timestamp((int(last["ts_min"]) + 1) * 60, unit="s", tz="UTC").to_pydatetime(),
            bid=float(last["bid_close"]),
            ask=float(last["ask_close"]),
            data_mode="synthetic",
        )

    # ------------------------------------------------------------------ generation
    def generate(self, instrument: str, start_min: int, end_min: int) -> pd.DataFrame:
        """Candles with ``ts_min`` in ``[start_min, end_min)`` ignoring the wall clock."""
        if instrument not in self._specs:
            raise InstrumentNotSupportedError(f"synthetic provider has no instrument {instrument!r}")
        start_min = max(start_min, self.anchor_min)
        if end_min <= start_min:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        first_day = (start_min - self.anchor_min) // 1440
        last_day = (end_min - 1 - self.anchor_min) // 1440
        frames = [self._day(instrument, d) for d in range(first_day, last_day + 1)]
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["ts_min"] >= start_min) & (df["ts_min"] < end_min)]
        return df.reset_index(drop=True)

    def _open_mask(self, day: int) -> np.ndarray:
        if day in self._mask_cache:
            return self._mask_cache[day]
        start = self.anchor_min + day * 1440
        mask = self.calendar.open_mask(np.arange(start, start + 1440, dtype="int64"))
        self._mask_cache[day] = mask
        if len(self._mask_cache) > 4096:
            self._mask_cache.popitem(last=False)
        return mask

    def _state(self, instrument: str, day: int) -> _DayState:
        states = self._states.setdefault(instrument, [])
        spec = self._specs[instrument]
        if not states:
            states.append(
                _DayState(logp=float(np.log(spec.start_price)), v=0.0, d=0.0, prev_open=False, since_open=0)
            )
        while len(states) <= day:
            d = len(states) - 1
            _, new_state = self._simulate(instrument, d, states[d], build_frame=False)
            states.append(new_state)
        return states[day]

    def _day(self, instrument: str, day: int) -> pd.DataFrame:
        key = (instrument, day)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        frame, _ = self._simulate(instrument, day, self._state(instrument, day))
        self._cache[key] = frame
        if len(self._cache) > self._cache_days:
            self._cache.popitem(last=False)
        return frame

    def _simulate(
        self, instrument: str, day: int, state: _DayState, build_frame: bool = True
    ) -> tuple[pd.DataFrame | None, _DayState]:
        cfg, spec = self.cfg, self._specs[instrument]
        n = 1440
        start = self.anchor_min + day * 1440
        minutes = np.arange(start, start + n, dtype="int64")
        open_mask = self._open_mask(day)
        rng = np.random.default_rng([int(cfg.seed), spec.seed_offset, int(day)])

        e_vol = rng.standard_normal(n)
        e_drift = rng.standard_normal(n)
        e_ret = rng.standard_t(_T_DF, n) * _T_SCALE
        e_gap = rng.standard_normal(n)
        e_hi = np.abs(rng.standard_normal(n))
        e_lo = np.abs(rng.standard_normal(n))
        e_vol_count = rng.random(n)
        drop = rng.random(n)

        sigma_base = spec.annual_vol / np.sqrt(TRADING_MINUTES_PER_YEAR)
        phi_v = 0.5 ** (1.0 / cfg.vol_persistence_minutes)
        phi_d = 0.5 ** (1.0 / cfg.drift_persistence_minutes)
        sd_v = cfg.vol_of_vol * np.sqrt(1 - phi_v**2)
        sd_d = spec.drift_strength * sigma_base * np.sqrt(1 - phi_d**2)

        a_v, a_d = [1.0, -phi_v], [1.0, -phi_d]
        v = lfilter([sd_v], a_v, e_vol, zi=lfiltic([sd_v], a_v, [state.v]))[0]
        d = lfilter([sd_d], a_d, e_drift, zi=lfiltic([sd_d], a_d, [state.d]))[0]

        hour = ((minutes % 1440) / 60.0).astype("float64")
        seasonal = 1.0 + 0.35 * np.cos(2 * np.pi * (hour - 14.0) / 24.0)
        sigma = sigma_base * np.exp(v - 0.5 * cfg.vol_of_vol**2) * seasonal

        r = d + sigma * e_ret
        prev_open = np.concatenate([[state.prev_open], open_mask[:-1]])
        first_open = open_mask & ~prev_open
        r = np.where(first_open, r + 4.0 * sigma * e_gap, r)
        r = np.where(open_mask, r, 0.0)

        logp = state.logp + np.cumsum(r)
        close = np.exp(logp)
        open_ = np.exp(logp - r)
        high = np.maximum(open_, close) * np.exp(0.6 * sigma * e_hi)
        low = np.minimum(open_, close) * np.exp(-0.6 * sigma * e_lo)

        # Spread: wider in quiet hours, in high volatility and just after a (re)open.
        since_open = np.empty(n)
        run = state.since_open
        for i in range(n):  # 1440 iterations per day; negligible
            if open_mask[i]:
                run = run + 1 if prev_open[i] else 0
            since_open[i] = run
        spread = spec.base_spread * np.exp(0.5 * v) * (2.0 - seasonal)
        spread = np.where(since_open < 15, spread * 2.5, spread)
        q = 10.0**spec.decimals
        spread = np.maximum(np.round(spread * q) / q, spec.min_spread)

        half = spread / 2.0
        rnd = lambda x: np.round(x * q) / q  # noqa: E731
        bid_o, ask_o = rnd(open_ - half), rnd(open_ + half)
        bid_h, ask_h = rnd(high - half), rnd(high + half)
        bid_l, ask_l = rnd(low - half), rnd(low + half)
        bid_c, ask_c = rnd(close - half), rnd(close + half)
        mid_o, mid_h, mid_l, mid_c = (bid_o + ask_o) / 2, (bid_h + ask_h) / 2, (bid_l + ask_l) / 2, (bid_c + ask_c) / 2
        mid_h = np.maximum.reduce([mid_h, mid_o, mid_c])
        mid_l = np.minimum.reduce([mid_l, mid_o, mid_c])
        bid_h, ask_h = np.maximum.reduce([bid_h, bid_o, bid_c]), np.maximum.reduce([ask_h, ask_o, ask_c])
        bid_l, ask_l = np.minimum.reduce([bid_l, bid_o, bid_c]), np.minimum.reduce([ask_l, ask_o, ask_c])
        new_state = _DayState(
            logp=float(logp[-1]),
            v=float(v[-1]),
            d=float(d[-1]),
            prev_open=bool(open_mask[-1]),
            since_open=int(since_open[-1]),
        )
        if not build_frame:
            return None, new_state

        volume = np.floor(5 + 60 * seasonal * e_vol_count)

        keep = open_mask & ~(drop < cfg.missing_rate)
        frame = pd.DataFrame(
            {
                "ts_min": minutes[keep],
                "open": mid_o[keep],
                "high": mid_h[keep],
                "low": mid_l[keep],
                "close": mid_c[keep],
                "bid_open": bid_o[keep],
                "bid_high": bid_h[keep],
                "bid_low": bid_l[keep],
                "bid_close": bid_c[keep],
                "ask_open": ask_o[keep],
                "ask_high": ask_h[keep],
                "ask_low": ask_l[keep],
                "ask_close": ask_c[keep],
                "spread": (ask_c - bid_c)[keep],
                "spread_source": "observed",
                "volume": volume[keep],
                "quote_convention": self.quote_convention,
                "data_mode": "synthetic",
                "quality": "ok",
                "quality_notes": None,
            }
        )
        return conform_candle_frame(frame), new_state
