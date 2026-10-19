import sqlite3

import pytest


def _insert_minimal_forecast(conn, forecast_id="f1"):
    conn.execute(
        """INSERT INTO forecasts(forecast_id, origin, run_id, data_mode, instrument, provider, quote_convention,
               model_type, model_version, config_version, slot_ts, slot_min, as_of_ts, created_at, start_ts,
               start_min, expiry_ts, expiry_min, horizon_minutes, reference_price_type, direction,
               candidate_direction, confidence, content_hash)
           VALUES (?, 'live', '', 'synthetic', 'XAU/USD', 'synthetic', 'synthetic_mid', 'deterministic',
               'v', 'c', '2026-01-05T10:00:00Z', 1, '2026-01-05T10:00:00Z', '2026-01-05T10:00:00Z',
               '2026-01-05T10:00:00Z', 1, '2026-01-05T11:00:00Z', 61, 60, 'mid_close_1m', 'WAIT', 'NONE',
               'none', 'h')""",
        (forecast_id,),
    )


def test_schema_initialises_and_integrity_ok(ctx):
    info = ctx.db.integrity_check()
    assert info["integrity"] == "ok"
    assert info["counts"]["forecasts"] == 0
    version = ctx.db.scalar("SELECT value FROM schema_info WHERE key='schema_version'")
    assert version == "1"


def test_forecasts_are_immutable(ctx):
    with ctx.db.connect() as conn:
        _insert_minimal_forecast(conn)
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("UPDATE forecasts SET direction = 'UP' WHERE forecast_id = 'f1'")
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("DELETE FROM forecasts WHERE forecast_id = 'f1'")
    row = ctx.db.query_one("SELECT direction FROM forecasts WHERE forecast_id = 'f1'")
    assert row["direction"] == "WAIT"


def test_duplicate_forecast_slot_rejected(ctx):
    with ctx.db.connect() as conn:
        _insert_minimal_forecast(conn, "f1")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_minimal_forecast(conn, "f2")  # same natural key (slot, horizon, model, config)


def test_transaction_rolls_back_on_error(ctx):
    with pytest.raises(RuntimeError):
        with ctx.db.transaction() as conn:
            _insert_minimal_forecast(conn, "f9")
            raise RuntimeError("boom")
    assert ctx.db.scalar("SELECT COUNT(*) FROM forecasts") == 0


def test_config_versions_registered(ctx):
    row = ctx.ops.get_config(ctx.settings.forecast_config_version)
    assert row is not None and row["scope"] == "forecast"
    assert row["config"]["forecasting"]["thresholds"]["up"] == 0.35


def test_feed_status_upsert_preserves_history(ctx):
    ctx.ops.upsert_feed_status("XAU/USD", "synthetic", "ok", data_mode="synthetic", last_candle_ts="t1", success=True)
    ctx.ops.upsert_feed_status("XAU/USD", "synthetic", "stale", error="timeout")
    (row,) = ctx.ops.feed_status()
    assert row["status"] == "stale"
    assert row["last_candle_ts"] == "t1"
    assert row["last_success_at"] is not None
    assert row["last_error"] == "timeout"
