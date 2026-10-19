"""Performance analytics.

Input is a frame of forecasts joined with their latest evaluation (``v_forecast_status``).

Statistical caveats built into the outputs:

* Forecasts every 15 minutes with a 60-minute horizon OVERLAP; consecutive outcomes are
  dependent. Confidence intervals therefore use a moving-block bootstrap with block length
  ``ceil(horizon / interval)``, and every summary also reports metrics on a greedy
  non-overlapping subsample.
* Target touches on a random walk are frequent for small targets. Directional skill is
  measured against the equidistant *mirror* target (paired difference), not against 50%.
* P-HIT is an uncalibrated heuristic; calibration tables and the Brier score show how well
  it matches observed frequencies. Nothing here adjusts the formula.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

DIRECTIONAL = ("UP", "DOWN")


# --------------------------------------------------------------------------- statistics
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def block_bootstrap_ci(
    values: np.ndarray,
    block: int,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float | None, float | None]:
    """Moving-block bootstrap percentile interval (deterministic seed)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return None, None
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    stats = np.apply_along_axis(stat, 1, x[idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def brier_score(p: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return None
    return float(np.mean((p[mask] - y[mask]) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float | None:
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return None
    pc = np.clip(p[mask], eps, 1 - eps)
    yy = y[mask]
    return float(-np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc)))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n = int(sel.sum())
        hits = int(y[sel].sum()) if n else 0
        ci = wilson_interval(hits, n)
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "bin_low": lo,
                "bin_high": hi,
                "n": n,
                "mean_p_hit": float(p[sel].mean()) if n else None,
                "observed_rate": hits / n if n else None,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(cum: np.ndarray) -> float:
    if cum.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    return float(np.max(peaks[1:] - cum)) if cum.size else 0.0


def non_overlapping(df: pd.DataFrame) -> pd.DataFrame:
    """Greedy subsample of forecasts whose [start, expiry) windows do not overlap."""
    if df.empty:
        return df
    d = df.sort_values("start_min")
    keep, last_end = [], -1
    for idx, s, e in zip(d.index, d["start_min"].to_numpy(), d["expiry_min"].to_numpy()):
        if s >= last_end:
            keep.append(idx)
            last_end = e
    return d.loc[keep]


# --------------------------------------------------------------------------- summaries
def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used across analytics."""
    if df.empty:
        return df.copy()
    d = df.copy()
    d["is_directional"] = d["direction"].isin(DIRECTIONAL)
    d["hit"] = np.where(d["outcome_status"] == "hit", 1.0, np.where(d["outcome_status"] == "miss", 0.0, np.nan))
    d["first"] = np.where(d["target_first"] == "yes", 1.0, np.where(d["target_first"].isin(["no", "neither"]), 0.0, np.nan))
    d["hour_utc"] = (d["slot_min"] // 60) % 24
    d["pnl"] = _as_float(d["trade_pnl_per_oz"])
    d["trade_success"] = np.where(d["trade_outcome"] == "tp", 1.0,
                                  np.where(d["trade_outcome"].isin(["sl", "expired", "ambiguous"]), 0.0, np.nan))
    td = _as_float(d["target_distance"])
    atr = _as_float(d["atr_1m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        d["target_atr"] = td / atr
    d["target_bucket"] = pd.cut(d["target_distance"], bins=[0, 0.5, 1, 2, 4, 8, np.inf],
                                labels=["<0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"]).astype(str)
    return d


def summarise(df: pd.DataFrame, interval_minutes: int = 15) -> dict[str, Any]:
    """Headline metrics for a set of forecasts (one model/config/horizon recommended)."""
    d = prepare(df)
    out: dict[str, Any] = {"n_forecasts": int(len(d))}
    if d.empty:
        return out
    dirn = d[d["is_directional"]]
    out["n_directional"] = int(len(dirn))
    out["n_wait"] = int((d["direction"] == "WAIT").sum())
    out["directional_coverage"] = len(dirn) / len(d) if len(d) else None
    out["n_evaluated"] = int(d["outcome_status"].notna().sum())
    if dirn.empty:
        return out

    horizon = int(dirn["horizon_minutes"].mode().iloc[0])
    block = max(1, math.ceil(horizon / interval_minutes))
    out["bootstrap_block"] = block

    hits = dirn["hit"].dropna()
    out["n_determinate"] = int(hits.size)
    out["target_hit_rate"] = float(hits.mean()) if hits.size else None
    out["target_hit_rate_ci"] = block_bootstrap_ci(hits.to_numpy(), block)
    out["indeterminate"] = int(dirn["outcome_status"].eq("indeterminate").sum())
    out["mean_minutes_to_touch"] = float(_as_float(dirn["minutes_to_touch"])[np.isfinite(_as_float(dirn["minutes_to_touch"]))].mean()) if dirn["minutes_to_touch"].notna().any() else None
    out["mean_target_distance"] = float(np.nanmean(_as_float(dirn["target_distance"])))
    out["mean_mfe"] = float(np.nanmean(_as_float(dirn["mfe"]))) if dirn["mfe"].notna().any() else None
    out["mean_mae"] = float(np.nanmean(_as_float(dirn["mae"]))) if dirn["mae"].notna().any() else None
    out["mean_mae_before_hit"] = float(np.nanmean(_as_float(dirn["mae_before_hit"]))) if dirn["mae_before_hit"].notna().any() else None

    first = dirn["first"].dropna()
    out["target_before_mirror_rate"] = float(first.mean()) if first.size else None
    out["target_before_mirror_ci"] = block_bootstrap_ci(first.to_numpy(), block)
    dc = _as_float(dirn["direction_correct"])
    dc = dc[np.isfinite(dc)]
    out["directional_accuracy_at_expiry"] = float(dc.mean()) if dc.size else None
    out["directional_accuracy_ci"] = block_bootstrap_ci(dc, block)

    p = _as_float(dirn["p_hit"])
    y = dirn["hit"].to_numpy(dtype=float)
    out["brier_score"] = brier_score(p, y)
    out["log_loss"] = log_loss(p, y)
    out["mean_p_hit"] = float(np.nanmean(p)) if np.isfinite(p).any() else None
    base = (y[np.isfinite(y)].mean() if np.isfinite(y).any() else None)
    out["brier_reference_climatology"] = float(base * (1 - base)) if base is not None else None

    pnl = dirn["pnl"].to_numpy()
    pnl = pnl[np.isfinite(pnl)]
    out["n_trades"] = int(pnl.size)
    if pnl.size:
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        out["win_rate"] = float((pnl > 0).mean())
        out["avg_win_per_oz"] = float(wins.mean()) if wins.size else None
        out["avg_loss_per_oz"] = float(losses.mean()) if losses.size else None
        out["expectancy_per_oz"] = float(pnl.mean())
        out["expectancy_ci"] = block_bootstrap_ci(pnl, block)
        gross_loss = -losses.sum()
        out["profit_factor"] = float(wins.sum() / gross_loss) if gross_loss > 0 else None
        out["net_pnl_per_oz"] = float(pnl.sum())
        order = np.argsort(dirn.loc[np.isfinite(dirn["pnl"]), "start_min"].to_numpy())
        out["max_drawdown_per_oz"] = max_drawdown(np.cumsum(pnl[order]))
    ts = dirn["trade_success"].dropna()
    out["executable_success_rate"] = float(ts.mean()) if ts.size else None
    amb = dirn["trade_outcome"].eq("ambiguous")
    out["ambiguous_trade_share"] = float(amb.mean())

    # Independent (non-overlapping) subsample.
    indep = non_overlapping(dirn)
    ih = indep["hit"].dropna()
    ipnl = indep["pnl"].to_numpy()
    ipnl = ipnl[np.isfinite(ipnl)]
    out["independent"] = {
        "n": int(len(indep)),
        "target_hit_rate": float(ih.mean()) if ih.size else None,
        "target_hit_rate_wilson": wilson_interval(int(ih.sum()), int(ih.size)) if ih.size else (None, None),
        "expectancy_per_oz": float(ipnl.mean()) if ipnl.size else None,
    }
    return out


def mirror_edge(target_evals: pd.DataFrame, interval_minutes: int = 15) -> dict[str, Any]:
    """Paired difference between primary-target and mirror-target hit rates."""
    if target_evals.empty:
        return {}
    t = target_evals[target_evals["direction"].isin(DIRECTIONAL)]
    piv = t.pivot_table(index="forecast_id", columns="label", values="hit", aggfunc="first")
    if "primary" not in piv or "mirror" not in piv:
        return {}
    pair = piv[["primary", "mirror"]].dropna()
    if pair.empty:
        return {}
    order = t.drop_duplicates("forecast_id").set_index("forecast_id").loc[pair.index, "slot_min"].sort_values().index
    diff = (pair.loc[order, "primary"] - pair.loc[order, "mirror"]).to_numpy(dtype=float)
    horizon = int(t["horizon_minutes"].mode().iloc[0])
    return {
        "n_pairs": int(len(pair)),
        "primary_hit_rate": float(pair["primary"].mean()),
        "mirror_hit_rate": float(pair["mirror"].mean()),
        "edge": float(diff.mean()),
        "edge_ci": block_bootstrap_ci(diff, max(1, math.ceil(horizon / interval_minutes))),
    }


def target_distance_table(target_evals: pd.DataFrame) -> pd.DataFrame:
    """Hit rates for every evaluated target distance (primary, alternatives, mirror)."""
    if target_evals.empty:
        return pd.DataFrame()
    t = target_evals[target_evals["direction"].isin(DIRECTIONAL)].copy()
    t["hit"] = pd.to_numeric(t["hit"], errors="coerce")
    g = t.groupby("label").agg(
        n=("hit", "count"),
        hit_rate=("hit", "mean"),
        mean_distance=("distance", "mean"),
        atr_multiple=("atr_multiple", "first"),
        mean_minutes_to_touch=("minutes_to_touch", "mean"),
    )
    return g.reset_index().sort_values(["atr_multiple", "label"])


def breakdown(df: pd.DataFrame, by: str, directional_only: bool = True) -> pd.DataFrame:
    d = prepare(df)
    if d.empty or by not in d.columns:
        return pd.DataFrame()
    if directional_only:
        d = d[d["is_directional"]]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(by, dropna=False)
    out = g.agg(
        n=("forecast_id", "count"),
        hit_rate=("hit", "mean"),
        target_first_rate=("first", "mean"),
        mean_p_hit=("p_hit", "mean"),
        expectancy_per_oz=("pnl", "mean"),
        net_pnl_per_oz=("pnl", "sum"),
        executable_success=("trade_success", "mean"),
    ).reset_index()
    briers = []
    for key, grp in g:
        briers.append(brier_score(_as_float(grp["p_hit"]), grp["hit"].to_numpy(dtype=float)))
    out["brier"] = briers
    return out


def performance_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    d = prepare(df)
    d = d[d["is_directional"]] if not d.empty else d
    if d.empty:
        return pd.DataFrame()
    d["period"] = pd.to_datetime(d["slot_min"] * 60, unit="s", utc=True).dt.floor(freq)
    out = d.groupby("period").agg(n=("forecast_id", "count"), hit_rate=("hit", "mean"), pnl=("pnl", "sum")).reset_index()
    out["cum_pnl_per_oz"] = out["pnl"].fillna(0).cumsum()
    return out


def wait_filter_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Outcomes of WAIT candidates by primary reason vs directional forecasts.

    If WAIT candidates hit as often as directional forecasts, the filter adds no value.
    """
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    d = d[d["candidate_direction"].isin(DIRECTIONAL)].copy()
    d["group"] = np.where(d["is_directional"], "DIRECTIONAL", "WAIT:" + d["primary_wait_reason"].fillna("?"))
    return d.groupby("group").agg(
        n=("forecast_id", "count"), hit_rate=("hit", "mean"), target_first_rate=("first", "mean"),
        expectancy_per_oz=("pnl", "mean"),
    ).reset_index()


def compare_models(df: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
    """Side-by-side headline metrics per model_type (e.g. deterministic vs llm_assisted)."""
    rows = []
    for (mt, mv), grp in df.groupby(["model_type", "model_version"], dropna=False):
        s = summarise(grp, interval_minutes)
        rows.append({
            "model_type": mt, "model_version": mv, "n": s.get("n_forecasts"), "directional": s.get("n_directional"),
            "coverage": s.get("directional_coverage"), "hit_rate": s.get("target_hit_rate"),
            "target_before_mirror": s.get("target_before_mirror_rate"), "brier": s.get("brier_score"),
            "expectancy_per_oz": s.get("expectancy_per_oz"), "profit_factor": s.get("profit_factor"),
        })
    return pd.DataFrame(rows)


def compare_matched(df: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
    """Deterministic vs LLM-assisted on exactly the same slots (pairs linked by parent id).

    LLM-assisted records are missing where the LLM was skipped or failed, so comparing all
    records would mix different samples; this restricts both sides to matched pairs.
    """
    if df.empty or "llm_assisted" not in set(df["model_type"]):
        return pd.DataFrame()
    assisted = df[df["model_type"] == "llm_assisted"]
    parents = set(assisted["parent_forecast_id"].dropna())
    matched = pd.concat([df[df["forecast_id"].isin(parents)], assisted[assisted["parent_forecast_id"].isin(parents)]])
    return compare_models(matched, interval_minutes)


def llm_filter_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Outcomes of deterministic signals the LLM accepted vs rejected.

    If rejected signals perform as well as accepted ones, the LLM filter adds no value.
    """
    if df.empty or "llm_assisted" not in set(df["model_type"]):
        return pd.DataFrame()
    det = prepare(df[df["model_type"] == "deterministic"])
    assisted = df[(df["model_type"] == "llm_assisted") & df["parent_forecast_id"].notna()]
    link = assisted[["parent_forecast_id", "direction"]].rename(
        columns={"parent_forecast_id": "forecast_id", "direction": "assisted_direction"})
    j = det[det["is_directional"]].merge(link, on="forecast_id", how="inner")
    if j.empty:
        return pd.DataFrame()
    j["llm_decision"] = np.where(j["assisted_direction"] == "WAIT", "rejected", "accepted")
    return j.groupby("llm_decision").agg(
        n=("forecast_id", "count"), hit_rate=("hit", "mean"), target_first_rate=("first", "mean"),
        expectancy_per_oz=("pnl", "mean"), net_pnl_per_oz=("pnl", "sum"),
    ).reset_index()
