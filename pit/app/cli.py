"""Command-line interface: ``python -m pit <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from pit import MODEL_VERSION, __version__
from pit.app.context import AppContext
from pit.common.logging_setup import configure_logging

Handler = Callable[[argparse.Namespace, AppContext], int]
_COMMANDS: dict[str, tuple[Handler, Callable[[argparse.ArgumentParser], None], str]] = {}


def command(name: str, help_text: str, configure: Callable[[argparse.ArgumentParser], None] | None = None):
    def deco(fn: Handler) -> Handler:
        _COMMANDS[name] = (fn, configure or (lambda p: None), help_text)
        return fn

    return deco


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------- foundation
@command("init-db", "Create or upgrade the database schema")
def cmd_init_db(args: argparse.Namespace, ctx: AppContext) -> int:
    _print(ctx.db.integrity_check())
    return 0


def _config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--section", help="Only print this top-level section")


@command("config", "Print the effective configuration and its version hashes", _config_args)
def cmd_config(args: argparse.Namespace, ctx: AppContext) -> int:
    data = ctx.settings.model_dump(mode="json")
    if args.section:
        data = {args.section: data.get(args.section)}
    data["_versions"] = {
        "forecast_config_version": ctx.settings.forecast_config_version,
        "paper_config_version": ctx.settings.paper_config_version,
        "model_version": MODEL_VERSION,
    }
    _print(data)
    return 0


@command("status", "Print system health summary")
def cmd_status(args: argparse.Namespace, ctx: AppContext) -> int:
    from pit.app.health import system_health

    _print(system_health(ctx))
    return 0


# ---------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    # Import modules that register further commands.
    from pit.app import commands  # noqa: F401

    parser = argparse.ArgumentParser(prog="pit", description="PIT — Predictive Intelligence Trading (research only)")
    parser.add_argument("--version", action="version", version=f"pit {__version__} ({MODEL_VERSION})")
    parser.add_argument("--config", help="Extra YAML config layered over the defaults")
    parser.add_argument("--db", help="Database path (overrides config)")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, configure, help_text) in sorted(_COMMANDS.items()):
        p = sub.add_parser(name, help=help_text, description=help_text)
        configure(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ctx = AppContext.create(db_path=args.db, config_path=args.config)
    configure_logging(args.log_level or ctx.settings.logging.level, ctx.settings.resolve_path(ctx.settings.logging.file))
    handler = _COMMANDS[args.command][0]
    try:
        return int(handler(args, ctx) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
