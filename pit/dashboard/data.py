"""Cached data access for the dashboard (read-mostly; short TTLs keep it near-live)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from pit.app.context import AppContext


@st.cache_resource(show_spinner=False)
def get_context(db_path: str | None = None, config_version_key: str = "") -> AppContext:
    return AppContext.create(db_path=db_path)


def resolved_db_path() -> str | None:
    return st.session_state.get("db_path") or os.environ.get("PIT_DATABASE_PATH") or None


def ctx() -> AppContext:
    # Keyed on the resolved path and config so a change never reuses a stale context.
    return get_context(resolved_db_path(), st.session_state.get("config_key", ""))


def scope() -> tuple[str, str]:
    """(origin, run_id) selected in the sidebar."""
    return st.session_state.get("origin", "live"), st.session_state.get("run_id", "")


@st.cache_data(ttl=15, show_spinner=False)
def forecasts(_ctx_key: str, origin: str, run_id: str, limit: int | None = None) -> pd.DataFrame:
    return ctx().forecasts.frame(origin=origin, run_id=run_id, limit=limit)


@st.cache_data(ttl=15, show_spinner=False)
def target_evaluations(_ctx_key: str, origin: str, run_id: str) -> pd.DataFrame:
    return ctx().forecasts.target_evaluations(origin, run_id)


@st.cache_data(ttl=10, show_spinner=False)
def candles(_ctx_key: str, instrument: str, source: str, start_min: int | None, end_min: int | None) -> pd.DataFrame:
    return ctx().candles.load(instrument, source, start_min, end_min)


@st.cache_data(ttl=30, show_spinner=False)
def runs(_ctx_key: str) -> list[dict[str, Any]]:
    return ctx().research.runs()


@st.cache_data(ttl=10, show_spinner=False)
def sources(_ctx_key: str) -> list[dict[str, Any]]:
    return ctx().candles.sources()


def key() -> str:
    """Cache key component that changes when the database selection changes."""
    return f"{resolved_db_path()}|{st.session_state.get('config_key', '')}"


def primary_source() -> str | None:
    """Candle source for the instrument: the configured provider if it has data, else the largest source."""
    c = ctx()
    srcs = [s for s in sources(key()) if s["instrument"] == c.settings.instrument.symbol]
    if not srcs:
        return None
    configured = c.settings.data.provider if c.settings.data.provider != "csv" else c.settings.providers.csv.source_name
    for s in srcs:
        if s["source"] == configured:
            return s["source"]
    return max(srcs, key=lambda s: s["candles"])["source"]


def source_mode(source: str | None) -> str | None:
    if not source:
        return None
    for s in sources(key()):
        if s["source"] == source:
            return s["data_mode"]
    return None


def clear() -> None:
    st.cache_data.clear()
