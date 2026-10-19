"""Additional CLI commands, registered on import by :mod:`pit.app.cli`."""

from __future__ import annotations

import argparse
from datetime import timedelta

from pit.app.cli import _print, command
from pit.app.context import AppContext
from pit.common.timeutils import parse_ts, utcnow


# ---------------------------------------------------------------------------- market data
def _ingest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", help="UTC start (ISO-8601). Default: now - data.history_days")
    p.add_argument("--end", help="UTC end (ISO-8601, exclusive). Default: now")
    p.add_argument("--days", type=float, help="Alternative to --start: number of days back from --end")
    p.add_argument("--instrument", help="Instrument (default: configured symbol; FX pair also loaded for AUD accounts)")


@command("ingest", "Backfill historical 1-minute candles from the configured provider", _ingest_args)
def cmd_ingest(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.data.ingestion import IngestionService

    end = parse_ts(args.end) if args.end else utcnow()
    if args.start:
        start = parse_ts(args.start)
    else:
        start = end - timedelta(days=args.days or ctx.settings.data.history_days)
    svc = IngestionService(ctx)
    instruments = [args.instrument] if args.instrument else svc.instruments()
    results = [svc.backfill(start, end, instrument=i).as_dict() for i in instruments]
    _print(results)
    return 0 if all(r["error"] is None for r in results) else 1


@command("sync", "Fetch candles since the last stored candle (one incremental cycle)")
def cmd_sync(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.data.ingestion import IngestionService

    results = [r.as_dict() for r in IngestionService(ctx).sync_all()]
    _print(results)
    return 0 if all(r["error"] is None for r in results) else 1


def _import_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="CSV file of 1-minute candles (see pit/data/providers/csv_file.py)")
    p.add_argument("--source", default="csv", help="Source name stored with the candles")
    p.add_argument("--instrument", help="Instrument symbol (default: configured symbol)")
    p.add_argument("--timezone", default="UTC", help="Timezone of naive timestamps in the file")
    p.add_argument("--quote-convention", default="provider_mid")
    p.add_argument("--data-mode", default="historical", choices=["historical", "delayed", "live"])


@command("import-csv", "Validate and import a CSV of historical candles", _import_args)
def cmd_import_csv(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.data.ingestion import IngestionService
    from pit.data.providers.csv_file import CsvProvider

    instrument = args.instrument or ctx.settings.instrument.symbol
    provider = CsvProvider(
        args.path,
        instrument=instrument,
        quote_convention=args.quote_convention,
        data_mode=args.data_mode,
        source_name=args.source,
        timezone=args.timezone,
    )
    bounds = provider.bounds()
    if bounds is None:
        _print({"error": "no rows"})
        return 1
    from pit.common.timeutils import from_epoch_min

    svc = IngestionService(ctx, provider=provider)
    result = svc.backfill(from_epoch_min(bounds[0]), from_epoch_min(bounds[1]), instrument=instrument)
    out = result.as_dict()
    out["misaligned_rows_rejected"] = provider.misaligned_rows
    _print(out)
    return 0 if result.error is None else 1


@command("data-sources", "List stored candle sources with coverage and quality counts")
def cmd_data_sources(args: argparse.Namespace, ctx: AppContext) -> int:
    _print({"sources": ctx.candles.sources(), "recent_gaps": ctx.candles.gaps(limit=20)})
    return 0


# ---------------------------------------------------------------------------- forecasting
def _forecast_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-sync", action="store_true", help="Do not fetch new candles first")
    p.add_argument("--slot", help="Slot timestamp (UTC, aligned to the interval). Default: current slot. "
                   "Must be within scheduler.grace_seconds of now")


@command("forecast", "Generate (and store) the forecast for the current slot", _forecast_args)
def cmd_forecast(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.common.timeutils import to_epoch_min
    from pit.data.ingestion import IngestionService
    from pit.forecasting.service import ForecastService

    now = utcnow()
    if not args.no_sync:
        IngestionService(ctx).sync_all(now)
    svc = ForecastService(ctx, llm_reviewer=_llm_reviewer(ctx))
    slot = to_epoch_min(parse_ts(args.slot)) if args.slot else svc.current_slot(now)
    # Manual runs obey the same grace window as the scheduler: a slot that has passed is
    # recorded as missed rather than forecast after the fact.
    out = svc.run_slot(slot, now)
    _print(out.as_dict())
    return 0


def _llm_reviewer(ctx: AppContext):
    if not ctx.settings.llm.enabled:
        return None
    from pit.forecasting.llm.reviewer import LLMReviewer

    return LLMReviewer.from_context(ctx)
