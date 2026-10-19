import json

import numpy as np
import pandas as pd
import pytest

from pit.config.settings import EvaluationConfig
from pit.evaluation.evaluator import CandleWindow, evaluate_forecast, simulate_bracket
from tests.helpers import MONDAY_10_UTC, make_candles, make_forecast

CFG = EvaluationConfig(slippage=0.05, min_coverage=0.9)
T0 = MONDAY_10_UTC


def window(closes, highs=None, lows=None, spread=0.2, opens=None):
    return CandleWindow.from_frame(make_candles(closes, start_min=T0, spread=spread, highs=highs, lows=lows, opens=opens))


def test_target_hit_timing_and_excursions():
    f, targets = make_forecast("UP", reference=2000.0, distance=1.0, horizon=10)
    closes = [2000.2, 2000.0, 1999.6, 2000.5, 2001.2, 2000.8, 2000.9, 2000.4, 2000.6, 2000.7]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    lows[2] = 1999.3
    e, rows = evaluate_forecast(f, targets, window(closes, highs, lows), 10, CFG)
    assert e["status"] == "hit"
    assert e["minutes_to_touch"] == 5  # touched in the 5th candle (index 4: high 2001.3)
    assert e["target_first"] == "yes"
    assert e["mfe"] == pytest.approx(1.3)
    assert e["mae"] == pytest.approx(0.7)
    assert e["mae_before_hit"] == pytest.approx(0.7) and e["mae_before_hit_ambiguous"] == 0
    assert e["direction_correct"] == 1 and e["close_at_expiry"] == 2000.7
    by = {r["label"]: r for r in rows}
    assert by["primary"]["hit"] == 1 and by["mirror"]["hit"] == 0


def test_miss_requires_coverage_otherwise_indeterminate():
    f, targets = make_forecast("DOWN", reference=2000.0, distance=2.0, horizon=10)
    flat = [2000.0] * 10
    e, _ = evaluate_forecast(f, targets, window(flat), 10, CFG)
    assert e["status"] == "miss" and e["coverage"] == 1.0
    e2, rows = evaluate_forecast(f, targets, window(flat[:5]), 10, CFG)
    assert e2["status"] == "indeterminate" and e2["coverage"] == 0.5
    assert all(r["hit"] is None for r in rows)
    assert e2["trade_outcome"] == "indeterminate" and e2["trade_pnl_per_oz"] is None


def test_target_and_mirror_in_same_candle_is_ambiguous():
    f, targets = make_forecast("UP", reference=2000.0, distance=1.0, horizon=5)
    closes = [2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
    highs = [2000.1, 2001.5, 2000.1, 2000.1, 2000.1]
    lows = [1999.9, 1998.5, 1999.9, 1999.9, 1999.9]
    e, _ = evaluate_forecast(f, targets, window(closes, highs, lows), 5, CFG)
    assert e["target_first"] == "ambiguous"
    assert e["mae_before_hit_ambiguous"] == 1


def test_wait_without_candidate_not_applicable():
    f, targets = make_forecast("WAIT", candidate="NONE")
    e, rows = evaluate_forecast(f, targets, window([2000.0] * 60), 60, CFG)
    assert e["status"] == "not_applicable" and rows == []


# ----------------------------------------------------------------------------- executable bracket
def test_long_take_profit_uses_ask_entry_and_bid_exit():
    w = window([2000.0, 2000.5, 2001.4], spread=0.2)  # open 2000.0 -> ask 2000.1
    r = simulate_bracket(w, "UP", take_profit=2001.0, stop=1999.0, slippage=0.05)
    assert r.outcome == "tp"
    assert r.entry_price == pytest.approx(2000.15)
    # bid high of candle 2 = 2001.5 - 0.1 = 2001.4 >= TP; limit fills at TP
    assert r.exit_price == 2001.0
    assert r.pnl_per_unit == pytest.approx(2001.0 - 2000.15)


def test_short_stop_loss_with_slippage():
    w = window([2000.0, 2000.6, 2001.2], spread=0.2)
    r = simulate_bracket(w, "DOWN", take_profit=1999.0, stop=2001.0, slippage=0.05)
    assert r.outcome == "sl"
    assert r.entry_price == pytest.approx(1999.85)  # bid open 1999.9 - slippage
    assert r.exit_price == pytest.approx(2001.05)  # ask reaches stop; fill + slippage
    assert r.pnl_per_unit == pytest.approx(1999.85 - 2001.05)


def test_gap_through_stop_fills_at_open():
    closes = [2000.0, 1997.0]
    opens = [2000.0, 1997.5]
    w = window(closes, opens=opens, spread=0.2)
    r = simulate_bracket(w, "UP", take_profit=2002.0, stop=1999.0, slippage=0.05)
    assert r.outcome == "sl"
    assert r.exit_price == pytest.approx(1997.4 - 0.05)  # bid open 1997.4, worse than the stop


def test_same_candle_stop_and_target_is_ambiguous_and_conservative():
    closes = [2000.0, 2000.0]
    highs = [2000.1, 2003.0]
    lows = [1999.9, 1997.0]
    w = window(closes, highs, lows)
    r = simulate_bracket(w, "UP", take_profit=2001.0, stop=1999.0, slippage=0.05)
    assert r.outcome == "ambiguous" and r.ambiguous
    assert r.exit_price == pytest.approx(1999.0 - 0.05)  # P&L assumes the stop came first
    assert r.pnl_per_unit < 0


def test_ticks_resolve_intrabar_order():
    closes = [2000.0, 2000.0]
    w = window(closes, [2000.1, 2003.0], [1999.9, 1997.0])
    minute = T0 + 1
    ticks = pd.DataFrame({
        "ts_ms": [minute * 60_000 + 1000, minute * 60_000 + 2000, minute * 60_000 + 3000],
        "bid": [2000.5, 2001.2, 1998.0], "ask": [2000.7, 2001.4, 1998.2],
    })
    r = simulate_bracket(w, "UP", 2001.0, 1999.0, 0.05, ticks=ticks)
    assert r.outcome == "tp" and r.resolved_by_ticks


def test_expiry_exit_at_bid_close():
    w = window([2000.0, 2000.3, 2000.4], spread=0.2)
    r = simulate_bracket(w, "UP", 2005.0, 1995.0, 0.05)
    assert r.outcome == "expired"
    assert r.exit_price == pytest.approx(2000.4 - 0.1 - 0.05)


def test_evaluation_records_trade_fields():
    f, targets = make_forecast("UP", reference=2000.0, distance=1.0, horizon=3)
    e, _ = evaluate_forecast(f, targets, window([2000.0, 2000.5, 2001.4]), 3, CFG)
    assert e["trade_outcome"] == "tp" and e["trade_ambiguous"] == 0
    assert e["trade_pnl_per_oz"] == pytest.approx(2001.0 - 2000.15)
    assert json.loads(e["details_json"])["entry_delay_min"] == 0
