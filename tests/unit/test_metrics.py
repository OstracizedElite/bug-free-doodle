import numpy as np
import pandas as pd
import pytest

from pit.analytics import metrics


def test_brier_logloss_and_wilson():
    p = np.array([0.9, 0.2, 0.7, np.nan])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    assert metrics.brier_score(p, y) == pytest.approx(((0.1) ** 2 + 0.2**2 + 0.7**2) / 3)
    assert metrics.log_loss(np.array([0.5]), np.array([1.0])) == pytest.approx(np.log(2))
    lo, hi = metrics.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert metrics.wilson_interval(0, 0) == (None, None)


def test_calibration_table_bins():
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.95, 1.0])
    y = np.array([0, 1, 0, 1, 1, 0])
    t = metrics.calibration_table(p, y, bins=10)
    assert t["n"].sum() == 6
    row = t[t["bin"] == "0.1-0.2"].iloc[0]
    assert row["n"] == 2 and row["observed_rate"] == 0.5
    top = t[t["bin"] == "0.9-1.0"].iloc[0]
    assert top["n"] == 3 and top["observed_rate"] == pytest.approx(2 / 3)


def test_block_bootstrap_is_deterministic_and_covers_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 400)
    a = metrics.block_bootstrap_ci(x, 4)
    b = metrics.block_bootstrap_ci(x, 4)
    assert a == b
    assert a[0] < x.mean() < a[1]
    assert metrics.block_bootstrap_ci(x[:3], 4) == (None, None)


def test_non_overlapping_subsample():
    df = pd.DataFrame({"start_min": [0, 15, 30, 45, 60, 75, 120], "expiry_min": [60, 75, 90, 105, 120, 135, 180]})
    sub = metrics.non_overlapping(df)
    assert sub["start_min"].tolist() == [0, 60, 120]


def test_max_drawdown():
    assert metrics.max_drawdown(np.cumsum([1.0, -2.0, -1.0, 3.0, -0.5])) == pytest.approx(3.0)


def _frame(n=40, hit_rate=0.75, seed=0):
    rng = np.random.default_rng(seed)
    hits = rng.random(n) < hit_rate
    return pd.DataFrame({
        "forecast_id": [f"f{i}" for i in range(n)],
        "slot_min": np.arange(n) * 15,
        "start_min": np.arange(n) * 15,
        "expiry_min": np.arange(n) * 15 + 60,
        "horizon_minutes": 60,
        "direction": np.where(np.arange(n) % 5 == 0, "WAIT", "UP"),
        "candidate_direction": "UP",
        "primary_wait_reason": np.where(np.arange(n) % 5 == 0, "score_below_threshold", None),
        "outcome_status": np.where(hits, "hit", "miss"),
        "target_first": np.where(hits, "yes", "no"),
        "direction_correct": hits.astype(int),
        "p_hit": 0.7,
        "trade_outcome": np.where(hits, "tp", "sl"),
        "trade_pnl_per_oz": np.where(hits, 0.8, -1.2),
        "minutes_to_touch": np.where(hits, 12, np.nan),
        "target_distance": 1.0,
        "atr_1m": 1.33,
        "mfe": 1.2, "mae": 0.8, "mae_before_hit": 0.5,
        "model_type": "deterministic", "model_version": "v", "session": "london",
        "confidence": "medium", "regime": "uncertain", "vol_regime": "normal",
    })


def test_summarise_and_breakdowns():
    df = _frame()
    s = metrics.summarise(df)
    d = df[df["direction"] != "WAIT"]
    assert s["n_forecasts"] == 40 and s["n_directional"] == len(d) and s["n_wait"] == 8
    assert s["target_hit_rate"] == pytest.approx((d["outcome_status"] == "hit").mean())
    assert s["expectancy_per_oz"] == pytest.approx(d["trade_pnl_per_oz"].mean())
    assert s["bootstrap_block"] == 4
    assert s["independent"]["n"] < s["n_directional"]
    wins = d.loc[d["trade_pnl_per_oz"] > 0, "trade_pnl_per_oz"].sum()
    losses = -d.loc[d["trade_pnl_per_oz"] <= 0, "trade_pnl_per_oz"].sum()
    assert s["profit_factor"] == pytest.approx(wins / losses)
    b = metrics.breakdown(df, "session")
    assert b.iloc[0]["n"] == len(d)
    w = metrics.wait_filter_analysis(df)
    assert set(w["group"]) == {"DIRECTIONAL", "WAIT:score_below_threshold"}
    cmp = metrics.compare_models(df)
    assert cmp.iloc[0]["model_type"] == "deterministic"


def test_mirror_edge():
    t = pd.DataFrame({
        "forecast_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
        "label": ["primary", "mirror"] * 6,
        "hit": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        "direction": "UP",
        "slot_min": [0, 0, 15, 15, 30, 30, 45, 45, 60, 60, 75, 75],
        "horizon_minutes": 60,
    })
    e = metrics.mirror_edge(t)
    assert e["n_pairs"] == 6
    assert e["primary_hit_rate"] == pytest.approx(5 / 6) and e["mirror_hit_rate"] == pytest.approx(3 / 6)
    assert e["edge"] == pytest.approx(2 / 6)
