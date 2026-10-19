import numpy as np
import pandas as pd
import pytest

from pit.features import indicators as ind


def test_true_range_and_atr_known_values():
    high = np.array([10.0, 12.0, 11.0, 15.0])
    low = np.array([9.0, 10.0, 8.0, 13.0])
    close = np.array([9.5, 11.0, 9.0, 14.0])
    tr = ind.true_range(high, low, close)
    # i=1: max(2, |12-9.5|=2.5, |10-9.5|=.5)=2.5; i=2: max(3, 0, 3)=3; i=3: max(2, 6, 4)=6
    assert tr.tolist() == [2.5, 3.0, 6.0]
    assert ind.atr(high, low, close, 2) == pytest.approx(4.5)
    assert ind.atr(high, low, close, 4) is None  # never a partial warm-up value


def test_ema_matches_recursive_definition_seeded_with_sma():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    e = ind.ema_series(x, 3)
    alpha = 0.5
    expected = [2.0]
    for v in x[3:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    assert np.allclose(e, expected)
    assert ind.ema_series(x[:2], 3) is None


def test_ema_cross_check_with_pandas_on_random_data():
    rng = np.random.default_rng(1)
    x = 100 + np.cumsum(rng.standard_normal(500))
    period = 21
    ours = ind.ema_series(x, period)
    seeded = np.concatenate([[x[:period].mean()], x[period:]])
    ref = pd.Series(seeded).ewm(span=period, adjust=False).mean().to_numpy()
    assert np.allclose(ours, ref)


def test_sma_consistency_efficiency():
    close = np.array([1.0, 2.0, 1.5, 2.5, 3.0])
    assert ind.sma(close, 2) == pytest.approx(2.75)
    # changes: +1, -0.5, +1, +0.5 -> 3 up, 1 down
    assert ind.directional_consistency(close, 4) == pytest.approx(0.5)
    assert ind.up_fraction(close, 4) == pytest.approx(0.75)
    # net +2 over travelled 3
    assert ind.efficiency_ratio(close, 4) == pytest.approx(2 / 3)
    assert ind.efficiency_ratio(np.array([1.0, 1.0, 1.0]), 2) == 0.0
    assert ind.efficiency_ratio(close, 10) is None


def test_log_returns_skip_gaps():
    close = np.array([100.0, 101.0, 102.0, 103.0])
    start = np.array([0, 1, 5, 6])
    r = ind.log_returns(close, start, max_step=1)
    assert np.allclose(r, [np.log(101 / 100), np.log(103 / 102)])


def test_swing_points_require_confirmation_bars():
    high = np.array([1, 2, 5, 3, 2, 4, 6, 3, 2, 7.0])
    low = np.array([0.5, 1, 4, 1.5, 1, 3, 5, 2, 0.5, 6.0])
    sh, sl = ind.swing_points(high, low, left=2, right=2)
    assert sh.tolist() == [2, 6]  # bar 9 (7.0) lacks two confirming bars -> not a swing yet
    assert sl.tolist() == [4]  # bar 8 lacks two confirming bars -> not a swing yet
    # Appending future bars can confirm new swings but never changes earlier ones.
    sh2, _ = ind.swing_points(np.append(high, [5, 4]), np.append(low, [5, 4]), 2, 2)
    assert sh2.tolist() == [2, 6, 9]
