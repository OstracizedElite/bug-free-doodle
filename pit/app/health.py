"""System health aggregation shared by the CLI and the dashboard."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pit.app.context import AppContext
from pit.common.timeutils import iso, parse_ts, utcnow


def system_health(ctx: AppContext) -> dict[str, Any]:
    now = utcnow()
    db_info = ctx.db.integrity_check()
    heartbeats = ctx.ops.heartbeats()
    for hb in heartbeats:
        hb["age_seconds"] = (now - parse_ts(hb["last_beat"])).total_seconds()

    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage = ctx.ops.usage_since(iso(month_start))
    llm_cost = ctx.db.scalar(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE created_at >= ?", (iso(month_start),)
    )
    last_forecast = ctx.db.query_one(
        "SELECT forecast_id, created_at, slot_ts, direction, data_mode FROM forecasts "
        "WHERE origin = 'live' ORDER BY slot_min DESC LIMIT 1"
    )
    errors_24h = ctx.db.scalar(
        "SELECT COUNT(*) FROM api_errors WHERE created_at >= ?", (iso(now - timedelta(hours=24)),)
    )
    missed_7d = ctx.db.scalar(
        "SELECT COUNT(*) FROM missed_slots WHERE detected_at >= ?", (iso(now - timedelta(days=7)),)
    )
    gaps_7d = ctx.db.query_one(
        "SELECT COUNT(*) AS gaps, COALESCE(SUM(missing_minutes), 0) AS minutes FROM data_gaps WHERE detected_at >= ?",
        (iso(now - timedelta(days=7)),),
    )
    data_cost = ctx.settings.costs.data_subscription_usd_per_month
    return {
        "checked_at": iso(now),
        "data_provider": ctx.settings.data.provider,
        "forecast_config_version": ctx.settings.forecast_config_version,
        "database": db_info,
        "feeds": ctx.ops.feed_status(),
        "heartbeats": heartbeats,
        "last_live_forecast": last_forecast,
        "api_errors_24h": errors_24h,
        "missed_slots_7d": missed_7d,
        "data_gaps_7d": gaps_7d,
        "monthly_cost": {
            "llm_usd": round(float(llm_cost or 0), 4),
            "data_subscription_usd": data_cost,
            "total_usd": round(float(llm_cost or 0) + data_cost, 4),
            "budget_usd": ctx.settings.costs.monthly_budget_usd,
            "api_usage": usage,
        },
    }
