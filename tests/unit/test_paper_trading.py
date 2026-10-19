import math

import pytest

from pit.config.settings import MarketConfig, PaperConfig
from pit.data.market_calendar import MarketCalendar
from pit.evaluation.evaluator import CandleWindow
from pit.paper_trading.engine import AccountState, PaperTradingEngine
from tests.helpers import MONDAY_10_UTC, make_candles, make_forecast

T0 = MONDAY_10_UTC
CAL = MarketCalendar(MarketConfig())


def cfg(**kw):
    base = dict(base_currency="USD", initial_balance=50_000.0, risk_per_trade=0.005, slippage=0.0,
                max_notional_multiple=100.0, leverage=20.0, units_step=1.0, min_units=1.0)
    base.update(kw)
    return PaperConfig(**base)


def win(closes, start=T0, spread=0.2, highs=None, lows=None, opens=None):
    return CandleWindow.from_frame(make_candles(closes, start_min=start, spread=spread, highs=highs, lows=lows, opens=opens))


def test_risk_based_sizing_and_take_profit_usd_account():
    f, _ = make_forecast("UP", reference=2000.0, distance=2.0, horizon=10)
    w = win([2000.0, 2001.0, 2002.5, 2002.4])
    eng = PaperTradingEngine(cfg(), CAL, "acct")
    ev = eng.process(w, [f])
    # entry = ask open 2000.1; stop = 1998.0; risk/unit = 2.1; risk = 250 USD -> 119 oz
    assert ev.opened[0]["units"] == 119
    assert ev.opened[0]["entry_price"] == pytest.approx(2000.1)
    (closed,) = ev.closed
    assert closed["exit_reason"] == "tp" and closed["exit_price"] == 2002.0
    assert closed["net_pnl_usd"] == pytest.approx(119 * (2002.0 - 2000.1))
    assert eng.state.balance == pytest.approx(50_000 + 119 * 1.9)
    assert not eng.state.open_positions


def test_aud_conversion_uses_observed_fx_and_never_invents_rates():
    f, _ = make_forecast("UP", reference=2000.0, distance=2.0, horizon=10)
    w = win([2000.0, 2001.0, 2002.5])
    eng = PaperTradingEngine(cfg(base_currency="AUD"), CAL, "acct", fx_lookup=lambda m: 0.64)
    ev = eng.process(w, [f])
    # equity 50,000 AUD = 32,000 USD; risk 160 USD / 2.1 -> 76 oz
    assert ev.opened[0]["units"] == 76
    net_usd = 76 * (2002.0 - 2000.1)
    assert ev.closed[0]["net_pnl_base"] == pytest.approx(net_usd / 0.64)
    assert eng.state.balance == pytest.approx(50_000 + net_usd / 0.64)

    # No FX at entry -> trade skipped.
    eng2 = PaperTradingEngine(cfg(base_currency="AUD"), CAL, "acct2", fx_lookup=lambda m: None)
    ev2 = eng2.process(w, [f])
    assert not ev2.opened
    assert any('"fx_unavailable"' in t["details_json"] for t in ev2.transactions)

    # FX disappears before exit -> USD P&L held unconverted, AUD balance untouched.
    rates = {T0: 0.64}
    eng3 = PaperTradingEngine(cfg(base_currency="AUD"), CAL, "acct3", fx_lookup=lambda m: rates.get(m))
    ev3 = eng3.process(w, [f])
    assert ev3.closed[0]["net_pnl_base"] is None
    assert eng3.state.unconverted_usd == pytest.approx(net_usd)
    assert eng3.state.balance == 50_000


def test_stop_loss_and_ambiguous_exit_conservative():
    f, _ = make_forecast("DOWN", reference=2000.0, distance=1.0, horizon=10)
    closes = [2000.0, 2000.0]
    w = win(closes, highs=[2000.1, 2003.0], lows=[1999.9, 1997.0])
    eng = PaperTradingEngine(cfg(), CAL, "acct")
    ev = eng.process(w, [f])
    (closed,) = ev.closed
    assert closed["exit_reason"] == "ambiguous_sl" and closed["ambiguous"] == 1
    assert closed["exit_price"] == 2001.0 and closed["net_pnl_usd"] < 0


def test_expiry_exit_and_max_concurrent_positions():
    f1, _ = make_forecast("UP", distance=5.0, horizon=3, forecast_id="a")
    f2, _ = make_forecast("UP", distance=5.0, horizon=3, start_min=T0 + 1, forecast_id="b")
    w = win([2000.0, 2000.2, 2000.4, 2000.6, 2000.8])
    eng = PaperTradingEngine(cfg(max_concurrent_positions=1), CAL, "acct")
    ev = eng.process(w, [f1, f2])
    assert [o["forecast_id"] for o in ev.opened] == ["a"]
    assert ev.closed[0]["exit_reason"] == "expiry" and ev.closed[0]["exit_ts"].endswith("10:02:00Z")
    assert any('"max_concurrent_positions"' in t["details_json"] for t in ev.transactions)


def test_margin_closeout_on_gap_through_stop():
    # The stop normally caps the loss; a large gap past it is what threatens margin.
    c = cfg(leverage=500.0, risk_per_trade=0.1, max_notional_multiple=1000.0, margin_closeout_level=0.5)
    f, _ = make_forecast("UP", reference=2000.0, distance=40.0, horizon=10)
    w = win([2000.0, 2000.0, 1500.0], opens=[2000.0, 2000.0, 1500.0],
            highs=[2000.1, 2000.1, 1500.1], lows=[1999.9, 1999.9, 1499.9])
    eng = PaperTradingEngine(c, CAL, "acct")
    ev = eng.process(w, [f])
    units = ev.opened[0]["units"]
    margin = units * 2000.1 / 500
    assert 50_000 + units * (1499.8 - 2000.1) < 0.5 * margin  # the scenario really breaches
    (closed,) = ev.closed
    assert closed["exit_reason"] == "closeout"
    assert closed["exit_price"] == pytest.approx(1499.8)
    assert any(t["txn_type"] == "closeout" for t in ev.transactions)


def test_drawdown_halt_blocks_new_trades():
    c = cfg(risk_per_trade=0.05, max_drawdown=0.01)  # one stopped trade loses ~1.1%
    f1, _ = make_forecast("UP", reference=2000.0, distance=1.0, horizon=5, forecast_id="a")
    f2, _ = make_forecast("UP", reference=1999.0, distance=1.0, horizon=5, start_min=T0 + 3, forecast_id="b")
    w = win([2000.0, 1998.0, 1998.5, 1998.6, 1998.7, 1998.8])
    eng = PaperTradingEngine(c, CAL, "acct")
    ev = eng.process(w, [f1, f2])
    assert eng.state.drawdown_halted
    assert [o["forecast_id"] for o in ev.opened] == ["a"]
    assert any(t["txn_type"] == "halt" for t in ev.transactions)
    assert any('"drawdown_halt"' in t["details_json"] for t in ev.transactions if t["txn_type"] == "skip")


def test_daily_loss_halt_resets_next_day():
    c = cfg(risk_per_trade=0.03, max_daily_loss=0.01, max_drawdown=0.5)
    f1, _ = make_forecast("UP", reference=2000.0, distance=1.0, horizon=5, forecast_id="a")
    f2, _ = make_forecast("UP", reference=2000.0, distance=1.0, horizon=5, start_min=T0 + 3, forecast_id="b")
    next_day = T0 + 24 * 60
    f3, _ = make_forecast("UP", reference=2000.0, distance=1.0, horizon=5, start_min=next_day, forecast_id="c")
    w1 = make_candles([2000.0, 1998.0, 1998.0, 1998.0, 1998.0], start_min=T0)
    w2 = make_candles([2000.0, 2000.2, 2001.5], start_min=next_day)
    import pandas as pd

    w = CandleWindow.from_frame(pd.concat([w1, w2], ignore_index=True))
    eng = PaperTradingEngine(c, CAL, "acct")
    ev = eng.process(w, [f1, f2, f3])
    assert [o["forecast_id"] for o in ev.opened] == ["a", "c"]
    kinds = [t["txn_type"] for t in ev.transactions]
    assert "halt" in kinds and "resume" in kinds
    assert any('"daily_loss_halt"' in t["details_json"] for t in ev.transactions if t["txn_type"] == "skip")


def test_financing_charged_across_rollover():
    # Wednesday 2026-01-07 21:58 UTC (16:58 NY); rollover at 17:00 NY = 22:00 UTC; market reopens 23:00 UTC.
    from pit.common.timeutils import to_epoch_min

    start = to_epoch_min("2026-01-07T21:58:00Z")
    reopen = to_epoch_min("2026-01-07T23:00:00Z")
    import pandas as pd

    df = pd.concat([make_candles([2000.0, 2000.1], start_min=start), make_candles([2000.2, 2000.3], start_min=reopen)])
    w = CandleWindow.from_frame(df)
    f, _ = make_forecast("UP", reference=2000.0, distance=10.0, horizon=120, start_min=start)
    c = cfg(financing={"long_annual_rate": 0.073, "short_annual_rate": 0.03, "rollover_time_ny": "17:00"})
    eng = PaperTradingEngine(c, CAL, "acct")
    ev = eng.process(w, [f])
    fin = [t for t in ev.transactions if t["txn_type"] == "financing"]
    assert len(fin) == 1
    units = ev.opened[0]["units"]
    assert fin[0]["amount_usd"] == pytest.approx(-units * 2000.1 * 0.073 / 365, rel=1e-6)


def test_incremental_processing_matches_single_pass():
    forecasts = [make_forecast("UP" if i % 2 else "DOWN", reference=2000.0, distance=1.0, horizon=4,
                               start_min=T0 + 6 * i, forecast_id=f"f{i}")[0] for i in range(6)]
    closes = [2000.0 + 0.6 * math.sin(i / 2.0) for i in range(40)]
    df = make_candles(closes, start_min=T0)
    one = PaperTradingEngine(cfg(), CAL, "acct")
    ev_one = one.process(CandleWindow.from_frame(df), forecasts)

    split = PaperTradingEngine(cfg(), CAL, "acct")
    first = split.process(CandleWindow.from_frame(df.iloc[:17]), forecasts)
    state = AccountState.from_json(split.state.to_json())  # persisted and reloaded
    resumed = PaperTradingEngine(cfg(), CAL, "acct", state=state)
    second = resumed.process(CandleWindow.from_frame(df), forecasts)
    assert resumed.state.balance == pytest.approx(one.state.balance)
    assert len(first.closed) + len(second.closed) == len(ev_one.closed)
    assert [t["seq"] for t in first.transactions + second.transactions] == [t["seq"] for t in ev_one.transactions]


def test_wait_and_low_confidence_not_traded():
    w = win([2000.0, 2000.5, 2001.0])
    f_wait, _ = make_forecast("WAIT", candidate="UP", forecast_id="w")
    f_low, _ = make_forecast("UP", confidence="low", forecast_id="l")
    eng = PaperTradingEngine(cfg(min_confidence="medium"), CAL, "acct")
    ev = eng.process(w, [f_wait, f_low])
    assert not ev.opened
    reasons = [t["details_json"] for t in ev.transactions]
    assert any("confidence_below_minimum" in r for r in reasons)
    assert not any(t["forecast_id"] == "w" for t in ev.transactions)  # WAITs are not signals
