"""Every dashboard page renders without exceptions against a populated database."""

from datetime import datetime, timedelta, timezone

import pytest
from streamlit.testing.v1 import AppTest

from pit.backtesting.replay import BacktestRunner
from pit.data.ingestion import IngestionService
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.synthetic import SyntheticProvider

pytestmark = pytest.mark.integration

VIEWS = ["overview", "live_chart", "history", "performance", "paper", "backtests", "settings", "health"]
NOW = datetime(2026, 2, 13, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory):
    from pit.app.context import AppContext
    from pit.config import load_settings

    path = tmp_path_factory.mktemp("dash") / "dash.db"
    settings = load_settings(include_local=False, overrides={
        "providers": {"synthetic": {"anchor": "2025-12-15T00:00:00Z"}},
        "data": {"history_days": 30},
        "features": {"lookback_bars": {"4h": 60, "12h": 30}, "min_bars": 30},
    })
    ctx = AppContext.create(settings=settings, db_path=path)
    ctx.set_provider(SyntheticProvider(settings.providers.synthetic, MarketCalendar(settings.market), clock=lambda: NOW))
    svc = IngestionService(ctx)
    for inst in svc.instruments():
        svc.backfill(NOW - timedelta(days=40), NOW, instrument=inst, now=NOW)
    run = BacktestRunner(ctx).run("2026-02-09T00:00:00Z", "2026-02-11T00:00:00Z", name="dash")
    return str(path), run.run_id


@pytest.mark.parametrize("scope", ["live", "backtest"])
def test_all_views_render(populated_db, scope, monkeypatch):
    path, run_id = populated_db
    monkeypatch.setenv("PIT_DATABASE_PATH", path)
    for view in VIEWS:
        at = AppTest.from_string(
            f"import sys; sys.path.insert(0, '.')\nfrom pit.dashboard.views import {view}\n{view}.render()\n",
            default_timeout=120,
        )
        at.session_state["origin"] = scope
        at.session_state["run_id"] = run_id if scope == "backtest" else ""
        at.run()
        assert not at.exception, f"{view} ({scope}): {[e.value for e in at.exception]}"
        if scope == "backtest" and view in ("overview", "history", "performance", "live_chart"):
            # Synthetic data must be labelled on every page that shows it.
            assert any("SYNTHETIC" in e.value for e in at.error), f"{view} lacks the synthetic-data banner"
