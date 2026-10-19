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


# ---------------------------------------------------------------------------- evaluation & research
@command("evaluate", "Evaluate live forecasts whose horizon has elapsed")
def cmd_evaluate(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.evaluation.service import EvaluationService

    _print({"evaluated": EvaluationService(ctx).evaluate_due()})
    return 0


def _backtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", required=True, help="UTC start of the replay window (ISO-8601)")
    p.add_argument("--end", required=True, help="UTC end of the replay window (exclusive)")
    p.add_argument("--name", default="backtest")
    p.add_argument("--validation", default="exploratory", choices=["exploratory", "out_of_sample"])
    p.add_argument("--declaration", help="Research declaration id (required for out_of_sample)")
    p.add_argument("--notes")
    p.add_argument("--no-paper", action="store_true", help="Skip the paper-trading simulation")
    p.add_argument("--ingest", action="store_true",
                   help="Backfill the needed history from the configured provider first")
    p.add_argument("--store-snapshots", action="store_true", help="Persist full feature snapshots")


@command("backtest", "Replay history: forecast, evaluate and paper-trade with no look-ahead", _backtest_args)
def cmd_backtest(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.backtesting.replay import BacktestRunner
    from pit.data.ingestion import IngestionService

    start, end = parse_ts(args.start), parse_ts(args.end)
    if args.ingest:
        svc = IngestionService(ctx)
        warm = timedelta(days=ctx.settings.data.history_days + 1)
        for inst in svc.instruments():
            svc.backfill(start - warm, end + timedelta(hours=3), instrument=inst)
    runner = BacktestRunner(ctx, store_snapshots=args.store_snapshots)
    result = runner.run(
        start, end, name=args.name, validation_type=args.validation, declaration_id=args.declaration,
        notes=args.notes, paper=not args.no_paper,
        progress=lambda k, n: print(f"  slot {k}/{n}", flush=True) if k and k % 2000 == 0 else None,
    )
    _print({"run_id": result.run_id, "forecasts": result.forecasts, "evaluated": result.evaluated,
            "seconds": result.seconds, "summary": result.summary})
    return 0


def _declare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--holdout-start", required=True)
    p.add_argument("--holdout-end", required=True)
    p.add_argument("--hypothesis", required=True, help="The prediction being tested, stated in advance")


@command("declare", "Pre-register the current config and an untouched holdout for one out-of-sample test",
         _declare_args)
def cmd_declare(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.common.timeutils import iso

    decl = ctx.research.declare(
        ctx.settings.forecast_config_version, iso(parse_ts(args.holdout_start)), iso(parse_ts(args.holdout_end)),
        args.hypothesis,
    )
    _print({"declaration_id": decl, "config_version": ctx.settings.forecast_config_version})
    return 0


@command("paper", "Process new candles and signals through the live paper account")
def cmd_paper(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.paper_trading.service import PaperTradingService

    _print(PaperTradingService(ctx).process())
    return 0


def _report_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", help="Backtest run id (default: live forecasts)")
    p.add_argument("--by", nargs="*", default=["direction", "session", "vol_regime", "confidence"],
                   help="Breakdown dimensions")


@command("report", "Performance analytics for live forecasts or a backtest run", _report_args)
def cmd_report(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.analytics import metrics

    origin, run_id = ("backtest", args.run) if args.run else ("live", "")
    df = ctx.forecasts.frame(origin=origin, run_id=run_id)
    interval = ctx.settings.forecasting.interval_minutes
    out = {"origin": origin, "run_id": run_id or None,
           "data_modes": sorted(df["data_mode"].unique().tolist()) if not df.empty else []}
    if "synthetic" in out["data_modes"]:
        out["warning"] = "SYNTHETIC DATA - not genuine trading performance"
    out["by_model"] = metrics.compare_models(df, interval).to_dict("records") if not df.empty else []
    det = df[df["model_type"] == "deterministic"] if not df.empty else df
    out["deterministic"] = metrics.summarise(det, interval)
    out["mirror_edge"] = metrics.mirror_edge(ctx.forecasts.target_evaluations(origin, run_id), interval)
    out["target_distances"] = metrics.target_distance_table(ctx.forecasts.target_evaluations(origin, run_id)).to_dict("records")
    out["breakdowns"] = {b: metrics.breakdown(det, b).to_dict("records") for b in args.by}
    out["wait_filter"] = metrics.wait_filter_analysis(det).to_dict("records")
    _print(out)
    return 0


def _demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--days", type=int, default=21, help="Replay window length in days")


@command("demo", "Load synthetic data and run a labelled synthetic backtest for exploring the dashboard", _demo_args)
def cmd_demo(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.backtesting.replay import BacktestRunner
    from pit.data.ingestion import IngestionService

    if ctx.settings.data.provider != "synthetic":
        print("demo uses synthetic data only; set data.provider: synthetic")
        return 2
    now = utcnow()
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    svc = IngestionService(ctx)
    for inst in svc.instruments():
        svc.backfill(start - timedelta(days=ctx.settings.data.history_days + 1), now, instrument=inst)
    result = BacktestRunner(ctx).run(start, end, name=f"synthetic demo ({args.days}d)",
                                     notes="SYNTHETIC demonstration run")
    _print({"run_id": result.run_id, "forecasts": result.forecasts, "seconds": result.seconds,
            "warning": "SYNTHETIC DATA - not genuine trading performance",
            "next": "streamlit run pit/dashboard/app.py"})
    return 0


@command("verify", "Verify forecast content hashes and database integrity")
def cmd_verify(args: argparse.Namespace, ctx: AppContext) -> int:
    out = {"database": ctx.db.integrity_check(), "forecast_hashes": ctx.forecasts.verify_hashes()}
    _print(out)
    return 0 if not out["forecast_hashes"]["mismatched"] and out["database"]["integrity"] == "ok" else 1
