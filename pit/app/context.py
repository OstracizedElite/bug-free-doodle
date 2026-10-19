"""Application context: wires configuration, database and repositories together."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from pit.config import PROJECT_ROOT, Settings, load_settings
from pit.database.db import Database
from pit.database.ops_repo import OpsRepository

if TYPE_CHECKING:  # pragma: no cover
    from pit.data.providers.base import MarketDataProvider


@dataclass
class AppContext:
    settings: Settings
    db: Database
    _provider: Any = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        db_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> "AppContext":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        settings = settings or load_settings(config_path)
        db = Database(db_path or settings.database_path)
        db.initialise()
        ctx = cls(settings=settings, db=db)
        ctx.ops.register_config(
            settings.forecast_config_version, "forecast", settings.forecast_config_dict()
        )
        ctx.ops.register_config(settings.paper_config_version, "paper", settings.paper_config_dict())
        return ctx

    # ------------------------------------------------------------------ repositories
    @cached_property
    def ops(self) -> OpsRepository:
        return OpsRepository(self.db)

    @cached_property
    def candles(self):
        from pit.database.candle_repo import CandleRepository

        return CandleRepository(self.db)

    @cached_property
    def forecasts(self):
        from pit.database.forecast_repo import ForecastRepository

        return ForecastRepository(self.db)

    @cached_property
    def research(self):
        from pit.database.research_repo import ResearchRepository

        return ResearchRepository(self.db)

    @cached_property
    def paper(self):
        from pit.database.paper_repo import PaperRepository

        return PaperRepository(self.db)

    @cached_property
    def llm_calls(self):
        from pit.database.llm_repo import LLMRepository

        return LLMRepository(self.db)

    # ------------------------------------------------------------------ provider
    @property
    def provider(self) -> "MarketDataProvider":
        if self._provider is None:
            from pit.data.providers.registry import create_provider

            self._provider = create_provider(self.settings, ops=self.ops)
        return self._provider

    def set_provider(self, provider: "MarketDataProvider") -> None:
        self._provider = provider
