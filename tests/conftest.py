"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pit.app.context import AppContext
from pit.config import load_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # Never let a developer's local overrides or credentials leak into tests.
    for var in (
        "PIT_CONFIG",
        "PIT_DATABASE_PATH",
        "TWELVEDATA_API_KEY",
        "OANDA_API_TOKEN",
        "OANDA_ACCOUNT_ID",
        "MASSIVE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def settings():
    return load_settings(include_local=False)


@pytest.fixture
def make_settings():
    def _make(overrides=None):
        return load_settings(overrides=overrides or {}, include_local=False)

    return _make


@pytest.fixture
def ctx(tmp_path: Path, settings) -> AppContext:
    return AppContext.create(settings=settings, db_path=tmp_path / "pit.db")


@pytest.fixture
def make_ctx(tmp_path: Path, make_settings):
    counter = {"n": 0}

    def _make(overrides=None) -> AppContext:
        counter["n"] += 1
        return AppContext.create(settings=make_settings(overrides), db_path=tmp_path / f"pit{counter['n']}.db")

    return _make
