"""Virtual paper-trading account."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from pit.dashboard import charts, data
from pit.dashboard.components import DOWN_COLOR, data_mode_banner, fmt_num, fmt_pct, show_chart


def render() -> None:
    ctx = data.ctx()
    origin, run_id = data.scope()
    st.title("Paper Trading")
    st.caption("Simulated account only. No brokerage integration exists and no real orders are ever placed.")
    accounts = ctx.paper.accounts()
    if origin == "backtest":
        accounts = [a for a in accounts if a["run_id"] == run_id]
    else:
        accounts = [a for a in accounts if a["origin"] == "live"]
    if not accounts:
        st.info("No paper account for this scope yet. Live accounts are created by the scheduler or "
                "`python -m pit paper`; backtests create their own.")
        return
    acct = st.selectbox("Account", accounts, format_func=lambda a: f"{a['name']} · {a['account_id']}")
    fc = data.forecasts(data.key(), origin, run_id)
    data_mode_banner(fc["data_mode"].unique() if not fc.empty else [])
    state = json.loads(acct["state_json"])
    cfg = json.loads(acct["config_json"])["paper"]
    cur = acct["base_currency"]
    equity = ctx.paper.equity(acct["account_id"])
    positions = ctx.paper.positions(acct["account_id"])
    closed = positions[positions["status"] == "closed"] if not positions.empty else positions
    open_pos = positions[positions["status"] == "open"] if not positions.empty else positions

    last_equity = float(equity["equity"].iloc[-1]) if not equity.empty else state["balance"]
    k = st.columns(5)
    k[0].metric(f"Balance ({cur})", fmt_num(state["balance"]),
                delta=fmt_num(state["balance"] - acct["initial_balance"]))
    k[1].metric(f"Equity ({cur})", fmt_num(last_equity))
    k[2].metric("Max drawdown", fmt_pct(float(equity["drawdown"].max()) if not equity.empty else 0.0),
                help=f"Halt at {fmt_pct(cfg['max_drawdown'])}")
    k[3].metric("Closed trades", len(closed), help=f"Open positions: {len(open_pos)}")
    net_usd = float(closed["net_pnl_usd"].sum()) if not closed.empty else 0.0
    k[4].metric("Net P&L (USD)", fmt_num(net_usd))
    flags = []
    if state.get("drawdown_halted"):
        flags.append("⛔ trading halted: maximum drawdown reached (manual review required)")
    if state.get("daily_halted"):
        flags.append("⚠️ daily loss limit reached: no new trades until the next UTC day")
    if state.get("unconverted_usd"):
        flags.append(f"⚠️ {fmt_num(state['unconverted_usd'])} USD of P&L not converted to {cur}: "
                     "no AUD/USD rate was available at exit (never estimated)")
    for fl in flags:
        st.warning(fl)
    st.caption(
        f"Settings: leverage {cfg['leverage']}:1 · risk {fmt_pct(cfg['risk_per_trade'], 2)}/trade · stop "
        f"{cfg['stop_multiple']}× target distance · max {cfg['max_concurrent_positions']} concurrent · daily loss "
        f"{fmt_pct(cfg['max_daily_loss'])} · slippage {cfg['slippage']} USD/oz · financing long "
        f"{fmt_pct(cfg['financing']['long_annual_rate'])} / short {fmt_pct(cfg['financing']['short_annual_rate'])} p.a."
    )

    if not equity.empty:
        eq = equity.copy()
        eq["time"] = pd.to_datetime(eq["ts_min"] * 60, unit="s", utc=True)
        show_chart(charts.line_chart(eq, "time", "equity", f"Equity ({cur})", cur), eq[["ts", "balance", "equity", "drawdown"]],
                   key="equity")
        eq["drawdown_pct"] = -eq["drawdown"]
        fig = charts.line_chart(eq, "time", "drawdown_pct", "Drawdown from peak", "drawdown", color=DOWN_COLOR, fill=True)
        fig.update_yaxes(tickformat=".1%")
        show_chart(fig, None, key="drawdown")

    if not closed.empty:
        pnl = closed["net_pnl_usd"].to_numpy(dtype=float)
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        s = st.columns(5)
        s[0].metric("Win rate", fmt_pct((pnl > 0).mean()))
        s[1].metric("Average win (USD)", fmt_num(wins.mean() if wins.size else None))
        s[2].metric("Average loss (USD)", fmt_num(losses.mean() if losses.size else None))
        s[3].metric("Profit factor", fmt_num(wins.sum() / -losses.sum() if losses.sum() < 0 else None))
        s[4].metric("Ambiguous exits", int(closed["ambiguous"].sum()), help="Same-candle stop and target; booked as stop")
        reasons = closed["exit_reason"].value_counts().rename_axis("exit_reason").reset_index(name="trades")
        st.dataframe(reasons, hide_index=True)

    st.subheader("Open positions")
    st.dataframe(open_pos, hide_index=True, width="stretch") if not open_pos.empty else st.caption("None")
    st.subheader("Trade history")
    if not closed.empty:
        st.dataframe(closed.sort_values("exit_ts", ascending=False), hide_index=True, width="stretch")
    st.subheader("Transaction log (append-only)")
    tx = ctx.paper.transactions(acct["account_id"], limit=500)
    if not tx.empty:
        tx["reason"] = tx["details_json"].map(lambda j: json.loads(j).get("reason", "") if j else "")
        st.dataframe(tx[["seq", "ts", "txn_type", "reason", "amount_usd", "amount_base", "balance_after", "forecast_id",
                         "position_id"]], hide_index=True, width="stretch")
