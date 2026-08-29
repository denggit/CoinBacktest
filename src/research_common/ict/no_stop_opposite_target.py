#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-stop opposite-liquidity counterfactual replay for SOXL ICT research.

The experiment changes exactly one post-fill rule from R16:
- entry/fill is frozen from the R16 lifecycle;
- no intraday stop is executed after fill;
- the opposite frozen external-liquidity boundary remains the only TP;
- if TP is not reached, the position exits at the final 1m close before 16:30 ET.

The old terminal-extreme stop is retained only as a diagnostic counterfactual so
we can measure how many eventual opposite-liquidity winners it would have washed
out.  This module does not generate entry signals and does not use future bars
for entry selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, NY_TZ, slice_ny_day


@dataclass(frozen=True)
class NoStopReplayConfig:
    trade_end: dtime = dtime(16, 30)
    round_trip_cost: float = 0.0011
    conservative_limit_same_bar_target: bool = True


def _as_ts(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize(NY_TZ)
    return t


def _safe_float(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _first_true(mask: np.ndarray) -> int | None:
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else None


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    pos = float(x.clip(lower=0).sum())
    neg = float(-x.clip(upper=0).sum())
    if neg > EPS:
        return pos / neg
    return np.inf if pos > 0 else np.nan


def _period_label(value) -> str:
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    if t < pd.Timestamp("2025-01-01"):
        return "discovery_2023h2_2024"
    if t < pd.Timestamp("2026-01-01"):
        return "forward_2025"
    return "late_2026"


def replay_no_stop_to_opposite_or_close(
    bars_ny: pd.DataFrame,
    frozen_fills: pd.DataFrame,
    *,
    config: NoStopReplayConfig = NoStopReplayConfig(),
) -> pd.DataFrame:
    """Replay already-filled R16 entries with no post-fill stop.

    Parameters
    ----------
    bars_ny:
        1m bars indexed in New York time.
    frozen_fills:
        R16 lifecycle rows.  Only rows with ``filled=True`` should normally be
        supplied.  ``fill_time`` and the original entry price are treated as
        frozen facts from R16 so this experiment does not change entry timing.

    Notes
    -----
    For limit fills, an opposite target touched on the same 1m bar as the fill
    is path-ambiguous.  By default it is *not* counted on that bar; target
    scanning starts on the next minute.  This is intentionally conservative.
    Market-next-open entries may count a target touched in the entry bar because
    entry occurs at the bar open.
    """
    if frozen_fills.empty:
        return pd.DataFrame()

    out: list[dict[str, object]] = []
    day_cache: dict[str, dict[str, object]] = {}

    for row in frozen_fills.to_dict("records"):
        rec = dict(row)
        if not bool(row.get("filled", False)):
            continue
        day_text = str(row.get("ny_date"))
        day_data = day_cache.get(day_text)
        if day_data is None:
            day = slice_ny_day(bars_ny, pd.Timestamp(day_text).date(), dtime(8, 30), config.trade_end)
            idx = pd.DatetimeIndex(day.index).as_unit("ns")
            day_data = {
                "idx": idx,
                "ns": idx.asi8,
                "open": pd.to_numeric(day["open"], errors="coerce").to_numpy(float),
                "high": pd.to_numeric(day["high"], errors="coerce").to_numpy(float),
                "low": pd.to_numeric(day["low"], errors="coerce").to_numpy(float),
                "close": pd.to_numeric(day["close"], errors="coerce").to_numpy(float),
            }
            day_cache[day_text] = day_data

        idx: pd.DatetimeIndex = day_data["idx"]
        ns: np.ndarray = day_data["ns"]
        hi: np.ndarray = day_data["high"]
        lo: np.ndarray = day_data["low"]
        cl: np.ndarray = day_data["close"]

        fill_time = _as_ts(row.get("fill_time"))
        fill_pos = int(np.searchsorted(ns, int(fill_time.value), side="left"))
        if fill_pos >= len(idx) or idx[fill_pos] != fill_time:
            rec.update({"no_stop_valid": False, "no_stop_invalid_reason": "fill_time_not_on_session_axis"})
            out.append(rec)
            continue

        is_long = str(row.get("trade_side", "")) == "LONG"
        entry = _safe_float(row.get("entry_price_replay"))
        if not np.isfinite(entry):
            entry = _safe_float(row.get("entry_price"))
        target = _safe_float(row.get("target_price"))
        old_stop = _safe_float(row.get("stop_price"))
        if not (np.isfinite(entry) and entry > EPS and np.isfinite(target)):
            rec.update({"no_stop_valid": False, "no_stop_invalid_reason": "invalid_entry_or_target"})
            out.append(rec)
            continue

        target_ahead = target > entry + EPS if is_long else target < entry - EPS
        if not target_ahead:
            rec.update({"no_stop_valid": False, "no_stop_invalid_reason": "opposite_target_not_ahead_at_fill"})
            out.append(rec)
            continue

        scan_start = fill_pos
        order_type = str(row.get("entry_order_type", "limit"))
        same_bar_target_ambiguous = False
        if config.conservative_limit_same_bar_target and order_type == "limit":
            target_on_fill_bar = bool(hi[fill_pos] >= target - EPS) if is_long else bool(lo[fill_pos] <= target + EPS)
            if target_on_fill_bar:
                same_bar_target_ambiguous = True
                scan_start = min(fill_pos + 1, len(idx))

        target_rel: int | None = None
        if scan_start < len(idx):
            target_rel = _first_true(hi[scan_start:] >= target - EPS) if is_long else _first_true(lo[scan_start:] <= target + EPS)
        target_pos = None if target_rel is None else scan_start + target_rel

        eod_pos = len(idx) - 1
        if target_pos is not None:
            exit_pos = target_pos
            exit_price = target
            exit_reason = "opposite_liquidity_tp"
            tp_hit = True
        else:
            exit_pos = eod_pos
            exit_price = float(cl[eod_pos])
            exit_reason = "session_close"
            tp_hit = False

        gross = (exit_price - entry) / entry if is_long else (entry - exit_price) / entry
        net = float(gross - float(config.round_trip_cost))

        hpost = hi[fill_pos : exit_pos + 1]
        lpost = lo[fill_pos : exit_pos + 1]
        max_fav_abs = float(np.nanmax(hpost) - entry) if is_long else float(entry - np.nanmin(lpost))
        max_adv_abs = float(entry - np.nanmin(lpost)) if is_long else float(np.nanmax(hpost) - entry)
        mfe_pct = max(0.0, max_fav_abs) / entry
        mae_pct = max(0.0, max_adv_abs) / entry

        old_risk_abs = entry - old_stop if is_long else old_stop - entry
        mae_old_r = float(max_adv_abs / old_risk_abs) if np.isfinite(old_risk_abs) and old_risk_abs > EPS else np.nan
        reward_old_r = float(abs(target - entry) / old_risk_abs) if np.isfinite(old_risk_abs) and old_risk_abs > EPS else np.nan

        old_stop_hit = bool(row.get("stop_hit", False))
        old_stop_time = row.get("stop_time", pd.NaT)
        rescued_after_old_stop = bool(old_stop_hit and tp_hit)
        old_stop_before_tp = False
        if old_stop_hit:
            try:
                st = _as_ts(old_stop_time)
                old_stop_before_tp = target_pos is None or st <= idx[target_pos]
            except Exception:
                old_stop_before_tp = True

        rec.update({
            "no_stop_valid": True,
            "no_stop_invalid_reason": "",
            "no_stop_entry_price": entry,
            "no_stop_target_price": target,
            "no_stop_tp_hit": bool(tp_hit),
            "no_stop_tp_time": idx[target_pos] if target_pos is not None else pd.NaT,
            "no_stop_exit_time": idx[exit_pos],
            "no_stop_exit_price": float(exit_price),
            "no_stop_exit_reason": exit_reason,
            "no_stop_gross_return": float(gross),
            "no_stop_net_return": net,
            "no_stop_is_profitable": bool(net > 0),
            "no_stop_minutes_held": float((idx[exit_pos] - idx[fill_pos]).total_seconds() / 60.0),
            "no_stop_mfe_pct": float(mfe_pct),
            "no_stop_mae_pct": float(-mae_pct),
            "no_stop_mae_old_r": float(-mae_old_r) if np.isfinite(mae_old_r) else np.nan,
            "reward_to_opposite_old_r": reward_old_r,
            "same_bar_fill_target_ambiguous": same_bar_target_ambiguous,
            "old_terminal_stop_hit": old_stop_hit,
            "old_terminal_stop_before_tp": bool(old_stop_before_tp),
            "rescued_after_old_terminal_stop": rescued_after_old_stop,
        })
        out.append(rec)

    return pd.DataFrame(out)


def summarize_no_stop(replayed: pd.DataFrame) -> pd.DataFrame:
    if replayed.empty:
        return pd.DataFrame()
    q = replayed.loc[replayed["no_stop_valid"].fillna(False).astype(bool)].copy()
    if q.empty:
        return pd.DataFrame()

    net = pd.to_numeric(q["no_stop_net_return"], errors="coerce")
    winners = net[net > 0]
    losers = net[net < 0]
    old_net = pd.to_numeric(q.get("net_return_exit_100"), errors="coerce")
    mean_win = float(winners.mean()) if len(winners) else np.nan
    mean_loss_abs = float(-losers.mean()) if len(losers) else np.nan

    row = {
        "trades": int(len(q)),
        "tp_hits": int(q["no_stop_tp_hit"].fillna(False).sum()),
        "tp_rate": float(q["no_stop_tp_hit"].fillna(False).mean()),
        "profitable_trades": int((net > 0).sum()),
        "win_rate": float((net > 0).mean()),
        "profit_factor": _profit_factor(net),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(net.median()),
        "mean_winner": mean_win,
        "mean_loser_abs": mean_loss_abs,
        "payoff_ratio": float(mean_win / mean_loss_abs) if np.isfinite(mean_win) and np.isfinite(mean_loss_abs) and mean_loss_abs > EPS else np.nan,
        "session_close_exits": int(q["no_stop_exit_reason"].eq("session_close").sum()),
        "session_close_profitable_rate": float(q.loc[q["no_stop_exit_reason"].eq("session_close"), "no_stop_is_profitable"].mean()) if bool(q["no_stop_exit_reason"].eq("session_close").any()) else np.nan,
        "median_mae_pct": float(pd.to_numeric(q["no_stop_mae_pct"], errors="coerce").median()),
        "p10_mae_pct": float(pd.to_numeric(q["no_stop_mae_pct"], errors="coerce").quantile(0.10)),
        "worst_mae_pct": float(pd.to_numeric(q["no_stop_mae_pct"], errors="coerce").min()),
        "median_mae_old_r": float(pd.to_numeric(q["no_stop_mae_old_r"], errors="coerce").median()),
        "p10_mae_old_r": float(pd.to_numeric(q["no_stop_mae_old_r"], errors="coerce").quantile(0.10)),
        "worst_mae_old_r": float(pd.to_numeric(q["no_stop_mae_old_r"], errors="coerce").min()),
        "old_terminal_stop_hits": int(q["old_terminal_stop_hit"].fillna(False).sum()),
        "rescued_after_old_terminal_stop": int(q["rescued_after_old_terminal_stop"].fillna(False).sum()),
        "rescued_share_of_old_stop_hits": float(q.loc[q["old_terminal_stop_hit"].fillna(False).astype(bool), "rescued_after_old_terminal_stop"].mean()) if bool(q["old_terminal_stop_hit"].fillna(False).any()) else np.nan,
        "same_bar_fill_target_ambiguous": int(q["same_bar_fill_target_ambiguous"].fillna(False).sum()),
        "old_stop_profit_factor": _profit_factor(old_net) if old_net is not None else np.nan,
        "old_stop_mean_net_return": float(old_net.mean()) if old_net is not None else np.nan,
        "old_stop_tp_before_stop_rate": float(q.get("milestone_100_before_stop", pd.Series(index=q.index, dtype=bool)).fillna(False).mean()),
    }
    return pd.DataFrame([row])


def summarize_no_stop_by_period(replayed: pd.DataFrame) -> pd.DataFrame:
    if replayed.empty:
        return pd.DataFrame()
    q = replayed.loc[replayed["no_stop_valid"].fillna(False).astype(bool)].copy()
    if q.empty:
        return pd.DataFrame()
    q["period"] = q["ny_date"].map(_period_label)
    rows: list[pd.DataFrame] = []
    for period, g in q.groupby("period", sort=True):
        s = summarize_no_stop(g)
        if not s.empty:
            s.insert(0, "period", period)
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_eod_failures(replayed: pd.DataFrame) -> pd.DataFrame:
    """Describe the no-TP session-close tail, which is the main no-stop risk."""
    if replayed.empty:
        return pd.DataFrame()
    q = replayed.loc[
        replayed["no_stop_valid"].fillna(False).astype(bool)
        & replayed["no_stop_exit_reason"].eq("session_close")
    ].copy()
    if q.empty:
        return pd.DataFrame()
    ret = pd.to_numeric(q["no_stop_net_return"], errors="coerce")
    mae = pd.to_numeric(q["no_stop_mae_pct"], errors="coerce")
    return pd.DataFrame([{
        "session_close_exits": len(q),
        "positive_at_close": int((ret > 0).sum()),
        "positive_at_close_rate": float((ret > 0).mean()),
        "mean_close_net_return": float(ret.mean()),
        "median_close_net_return": float(ret.median()),
        "p10_close_net_return": float(ret.quantile(0.10)),
        "p05_close_net_return": float(ret.quantile(0.05)),
        "worst_close_net_return": float(ret.min()),
        "median_mae_pct": float(mae.median()),
        "p10_mae_pct": float(mae.quantile(0.10)),
        "worst_mae_pct": float(mae.min()),
    }])
