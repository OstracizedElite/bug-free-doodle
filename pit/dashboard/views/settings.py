"""Edit configuration (written to config/local.yaml, validated before saving)."""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from pit.config import load_settings, save_local_overrides
from pit.config.settings import LOCAL_CONFIG_PATH, SUPPORTED_HORIZONS
from pit.dashboard import data


def render() -> None:
    ctx = data.ctx()
    s = ctx.settings
    st.title("Settings")
    st.caption(
        f"Current forecast config version `{s.forecast_config_version}` · paper config `{s.paper_config_version}`. "
        "Any change to a forecasting parameter creates a NEW config version; earlier forecasts keep the version they "
        "were made with and are never re-labelled. Changes are saved to `config/local.yaml` and take effect when the "
        "scheduler restarts (or on the next CLI run)."
    )
    st.warning("Thresholds and weights are research assumptions. Changing them after looking at results makes "
               "later backtests on the same data exploratory, not out-of-sample.")

    with st.form("settings"):
        st.subheader("Market data")
        c1, c2 = st.columns(2)
        providers = ["synthetic", "csv", "twelvedata", "oanda", "massive"]
        provider = c1.selectbox("Provider", providers, index=providers.index(s.data.provider),
                                help="Paid or account-based providers require approval and credentials in .env")
        stale = c2.number_input("Stale after (seconds)", 30, 3600, s.data.stale_after_seconds)

        st.subheader("Forecasting")
        c1, c2, c3 = st.columns(3)
        interval = c1.selectbox("Forecast interval (minutes)", [1, 5, 15, 30, 60],
                                index=[1, 5, 15, 30, 60].index(s.forecasting.interval_minutes)
                                if s.forecasting.interval_minutes in [1, 5, 15, 30, 60] else 2)
        horizons = c2.multiselect("Horizons (minutes)", list(SUPPORTED_HORIZONS), default=s.forecasting.horizons)
        atr_mult = c3.number_input("Target distance (× ATR14 1m)", 0.1, 5.0, s.forecasting.targets.atr_multiple, 0.05)
        c1, c2, c3 = st.columns(3)
        up = c1.number_input("UP threshold", 0.05, 1.0, s.forecasting.thresholds.up, 0.05)
        down = c2.number_input("DOWN threshold", -1.0, -0.05, s.forecasting.thresholds.down, 0.05)
        conflict = c3.number_input("Max conflicting weight", 0.0, 1.0, s.forecasting.wait.max_conflict, 0.05)

        st.subheader("Optional LLM reasoning")
        c1, c2, c3, c4 = st.columns(4)
        llm_on = c1.toggle("Enable LLM review", value=s.llm.enabled,
                           help="Requires an approved API key in .env. Incurs API charges.")
        llm_provider = c2.selectbox("LLM provider", ["openai", "anthropic"], index=["openai", "anthropic"].index(s.llm.provider))
        llm_model = c3.text_input("Model", value=s.llm.model)
        budget = c4.number_input("Monthly LLM budget (USD)", 0.0, 1000.0, s.llm.monthly_budget_usd, 5.0)
        every = st.number_input("Consult the LLM every N slots", 1, 96, s.llm.every_n_slots)

        st.subheader("Paper trading")
        c1, c2, c3, c4 = st.columns(4)
        paper_on = c1.toggle("Paper trading enabled", value=s.paper.enabled)
        risk = c2.number_input("Risk per trade (fraction)", 0.0005, 0.05, s.paper.risk_per_trade, 0.0005, format="%.4f")
        leverage = c3.number_input("Leverage", 1.0, 30.0, s.paper.leverage, 1.0)
        max_pos = c4.number_input("Max concurrent positions", 1, 10, s.paper.max_concurrent_positions)
        c1, c2, c3, c4 = st.columns(4)
        daily = c1.number_input("Max daily loss", 0.001, 0.5, s.paper.max_daily_loss, 0.005, format="%.3f")
        mdd = c2.number_input("Max drawdown", 0.01, 0.9, s.paper.max_drawdown, 0.01)
        slip = c3.number_input("Slippage (USD/oz)", 0.0, 2.0, s.paper.slippage, 0.01)
        currency = c4.selectbox("Base currency", ["AUD", "USD"], index=["AUD", "USD"].index(s.paper.base_currency))

        st.subheader("Costs")
        data_cost = st.number_input("Approved data subscription (USD/month, for reporting)", 0.0, 2000.0,
                                    s.costs.data_subscription_usd_per_month, 1.0)
        saved = st.form_submit_button("Validate and save")

    if saved:
        overrides = {
            "data": {"provider": provider, "stale_after_seconds": int(stale)},
            "forecasting": {
                "interval_minutes": int(interval), "horizons": horizons,
                "thresholds": {"up": float(up), "down": float(down)},
                "targets": {"atr_multiple": float(atr_mult)},
                "wait": {"max_conflict": float(conflict)},
            },
            "llm": {"enabled": bool(llm_on), "provider": llm_provider, "model": llm_model,
                    "monthly_budget_usd": float(budget), "every_n_slots": int(every)},
            "paper": {"enabled": bool(paper_on), "risk_per_trade": float(risk), "leverage": float(leverage),
                      "max_concurrent_positions": int(max_pos), "max_daily_loss": float(daily),
                      "max_drawdown": float(mdd), "slippage": float(slip), "base_currency": currency},
            "costs": {"data_subscription_usd_per_month": float(data_cost)},
        }
        try:
            path = save_local_overrides(overrides)
            new = load_settings()
        except (ValidationError, ValueError) as exc:
            st.error(f"Not saved, the configuration is invalid: {exc}")
        else:
            st.success(f"Saved to {path}. New forecast config version: `{new.forecast_config_version}`"
                       + (" (unchanged)" if new.forecast_config_version == s.forecast_config_version else ""))
            st.session_state["config_key"] = new.forecast_config_version + new.paper_config_version
            data.clear()
            if llm_on:
                st.info("LLM review is enabled. API calls cost money; usage is capped by the monthly budget.")

    with st.expander("Effective configuration (read-only)"):
        st.json(s.model_dump(mode="json"))
    if LOCAL_CONFIG_PATH.exists():
        with st.expander("config/local.yaml"):
            st.code(LOCAL_CONFIG_PATH.read_text(), language="yaml")
