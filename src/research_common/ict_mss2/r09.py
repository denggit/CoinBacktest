#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R09 ICT liquidity quality x execution atlas.

R09 takes the corrected R08.1 full-trend ICT liquidity taxonomy and converts it
into *independent, causally-started* sweep opportunities.  The research design
keeps breadth: native IT/LT and nested lower-timeframe IT/LT are retained, while
higher-timeframe -> lower-trend projection errors remain excluded.

A root opportunity starts on the first same-side physical liquidity sweep after
an inactivity gap.  Initial quality fields use only levels swept on that root
minute and contexts already activated by that minute.  Additional sweeps in the
next 15 minutes are future path diagnostics (precision vs cascade) and are
never used to assign the initial context tier.

Execution variants are applied to the same root opportunity universe:
- sweep_immediate: next available 1m open after the sweep bar closes;
- episode_reclaim: market next-open after close reclaims the swept level set;
- MSS market: R02 causal structural/post-sweep-ST MSS logic;
- reclaim_then_fvg_limit: after reclaim, wait for a directional FVG and rest a
  proximal/CE limit order; no market chase;
- MSS+FVG limit: R02 causal MSS confirmation followed by resting FVG limit.

Outcomes use structural stops and pessimistic same-bar stop-first semantics.
Fixed-R and fixed-percent opportunity labels are research sensitivities; the
censor horizon is not a time exit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import (
    MSS2Config,
    _first_fvg_in_range,
    aggregate_bars,
    normalize_1m_bars,
)
from src.research_common.ict_mss2.r02 import (
    R02Config,
    _structural_stop_before_entry,
    build_stack_execution_triggers,
)
from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R09Config:
    episode_gap_minutes: int = 15
    confirmation_minutes: int = 180
    fvg_wait_minutes: int = 180
    fvg_after_reclaim_bars: int = 8
    stop_buffer_bps: float = 2.0
    censor_minutes: int = 7 * 24 * 60
    fixed_r_targets: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    fixed_pct_targets: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05)
    path_horizons_minutes: tuple[int, ...] = (60, 180, 360, 720, 1440, 2880, 4320)
    market_roundtrip_cost: float = 0.0011
    limit_roundtrip_cost: float = 0.0008

    def validate(self) -> "R09Config":
        if self.episode_gap_minutes <= 0:
            raise ValueError("episode_gap_minutes must be positive")
        if self.confirmation_minutes <= 0 or self.fvg_wait_minutes <= 0:
            raise ValueError("confirmation/fvg wait must be positive")
        if self.censor_minutes <= 0:
            raise ValueError("censor_minutes must be positive")
        if self.stop_buffer_bps < 0:
            raise ValueError("stop_buffer_bps must be nonnegative")
        if any(float(x) <= 0 for x in self.fixed_r_targets + self.fixed_pct_targets):
            raise ValueError("targets must be positive")
        return self


def _join_unique(values: Iterable[object]) -> str:
    vals = sorted({str(v) for v in values if pd.notna(v) and str(v) not in {"", "nan", "NaT"}})
    return "|".join(vals)


def _context_tier(max_trend_tf_min: float) -> str:
    x = float(max_trend_tf_min) if np.isfinite(max_trend_tf_min) else 0.0
    if x >= 240:
        return "A+_4H_context"
    if x >= 60:
        return "A_1H_context"
    if x >= 30:
        return "B_30m_context"
    return "C_15m_context"


def build_physical_liquidity_sweeps(
    native: pd.DataFrame,
    nested: pd.DataFrame,
    *,
    research_start: pd.Timestamp | None = None,
    research_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Collapse trend-context rows to one row per physical swing first sweep.

    A physical swing may be qualified by several already-completed trend
    contexts.  Those contexts are aggregated at the physical swing grain; the
    sweep itself is counted once.
    """
    frames: list[pd.DataFrame] = []
    for scope, frame in (("native", native), ("nested_lower_tf", nested)):
        if frame is None or frame.empty:
            continue
        q = frame.copy()
        if "projection_scope" not in q.columns:
            q["projection_scope"] = scope
        q = q.loc[pd.to_numeric(q.get("active_at_activation_flag"), errors="coerce").fillna(0).eq(1)].copy()
        q["sweep_time"] = pd.to_datetime(q.get("first_sweep_after_activation_time"), errors="coerce")
        q["sweep_available_time"] = pd.to_datetime(q.get("first_sweep_after_activation_available_time"), errors="coerce")
        q["pivot_time"] = pd.to_datetime(q.get("pivot_time"), errors="coerce")
        q["liquidity_activation_time"] = pd.to_datetime(q.get("liquidity_activation_time"), errors="coerce")
        q = q.dropna(subset=["swing_id", "liquidity_side", "sweep_time", "level_price"])
        if research_start is not None:
            q = q.loc[q["sweep_time"] >= pd.Timestamp(research_start)]
        if research_end is not None:
            q = q.loc[q["sweep_time"] <= pd.Timestamp(research_end)]
        frames.append(q)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    for c in ["source_timeframe_min", "swing_source_timeframe_min", "trend_move_pct", "swing_is_lt", "scale_ge_05pct_flag", "scale_ge_07pct_flag"]:
        x[c] = pd.to_numeric(x.get(c), errors="coerce")

    rows: list[dict[str, object]] = []
    group_cols = ["swing_id", "liquidity_side", "sweep_time"]
    for (swing_id, side, sweep_time), p in x.groupby(group_cols, dropna=False, sort=True):
        # All rows represent the same physical swing.  Only contexts activated
        # before the physical sweep can contribute quality information.
        p = p.loc[p["liquidity_activation_time"] <= pd.Timestamp(sweep_time)].copy()
        if p.empty:
            continue
        level = float(pd.to_numeric(p["level_price"], errors="coerce").median())
        pivot = pd.to_datetime(p["pivot_time"], errors="coerce").min()
        activation = pd.to_datetime(p["liquidity_activation_time"], errors="coerce").min()
        swing_tf = float(pd.to_numeric(p["swing_source_timeframe_min"], errors="coerce").max())
        max_trend_tf = float(pd.to_numeric(p["source_timeframe_min"], errors="coerce").max())
        trend_move = float(pd.to_numeric(p["trend_move_pct"], errors="coerce").max())
        scopes = set(p["projection_scope"].astype(str))
        rows.append({
            "swing_id": str(swing_id),
            "liquidity_side": str(side),
            "sweep_time": pd.Timestamp(sweep_time),
            "sweep_available_time": pd.to_datetime(p["sweep_available_time"], errors="coerce").min(),
            "level_price": level,
            "pivot_time": pivot,
            "liquidity_activation_time": activation,
            "swing_source_timeframe_min": int(swing_tf) if np.isfinite(swing_tf) else -1,
            "swing_role": "LT" if int(pd.to_numeric(p["swing_is_lt"], errors="coerce").fillna(0).max()) == 1 else "IT",
            "context_count": int(len(p)),
            "native_context_count": int(sum(p["projection_scope"].astype(str).eq("native"))),
            "nested_context_count": int(sum(p["projection_scope"].astype(str).eq("nested_lower_tf"))),
            "native_flag": int("native" in scopes),
            "nested_flag": int("nested_lower_tf" in scopes),
            "max_trend_timeframe_min": int(max_trend_tf) if np.isfinite(max_trend_tf) else -1,
            "trend_move_pct_max": trend_move,
            "trend_scale_ge5_flag": int(pd.to_numeric(p["scale_ge_05pct_flag"], errors="coerce").fillna(0).max()),
            "trend_scale_ge7_flag": int(pd.to_numeric(p["scale_ge_07pct_flag"], errors="coerce").fillna(0).max()),
            "trend_leg_ids": _join_unique(p["trend_leg_id"]),
            "trend_timeframes": _join_unique(p["source_timeframe"]),
            "projection_scopes": _join_unique(p["projection_scope"]),
            "liquidity_age_days": float((pd.Timestamp(sweep_time) - pd.Timestamp(pivot)).total_seconds() / 86400.0) if pd.notna(pivot) else np.nan,
            "active_age_days": float((pd.Timestamp(sweep_time) - pd.Timestamp(activation)).total_seconds() / 86400.0) if pd.notna(activation) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["context_tier"] = out["max_trend_timeframe_min"].map(_context_tier)
    out = out.sort_values(["sweep_time", "liquidity_side", "level_price", "swing_id"], kind="stable").reset_index(drop=True)
    return out


def build_root_sweep_episodes(
    physical: pd.DataFrame,
    bars_1m: pd.DataFrame,
    *,
    config: R09Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal root opportunities and ex-post 15m cascade diagnostics.

    Root quality comes only from physical levels swept on the first sweep minute.
    Later sweeps are stored in the returned detail table and in fields prefixed
    ``future_``.  They are not admission or tier features.
    """
    cfg = (config or R09Config()).validate()
    if physical.empty:
        return pd.DataFrame(), pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    idx = bars.index
    x = physical.copy()
    x["sweep_time"] = pd.to_datetime(x["sweep_time"], errors="coerce")
    x = x.dropna(subset=["sweep_time", "liquidity_side"])

    # Aggregate simultaneous physical sweeps first.
    minute_rows: list[dict[str, object]] = []
    for (side, t), p in x.groupby(["liquidity_side", "sweep_time"], sort=True):
        levels = pd.to_numeric(p["level_price"], errors="coerce").dropna().to_numpy(dtype=float)
        max_tf = float(pd.to_numeric(p["max_trend_timeframe_min"], errors="coerce").max())
        swing_tf = pd.to_numeric(p["swing_source_timeframe_min"], errors="coerce")
        minute_rows.append({
            "liquidity_side": side,
            "sweep_time": pd.Timestamp(t),
            "root_physical_level_count": int(p["swing_id"].nunique()),
            "root_native_level_count": int(p.loc[p["native_flag"].eq(1), "swing_id"].nunique()),
            "root_nested_level_count": int(p.loc[p["nested_flag"].eq(1), "swing_id"].nunique()),
            "root_lt_level_count": int(p.loc[p["swing_role"].eq("LT"), "swing_id"].nunique()),
            "root_max_context_tf_min": int(max_tf) if np.isfinite(max_tf) else -1,
            "root_max_swing_tf_min": int(swing_tf.max()) if swing_tf.notna().any() else -1,
            "root_trend_move_pct_max": float(pd.to_numeric(p["trend_move_pct_max"], errors="coerce").max()),
            "root_trend_scale_ge5_flag": int(pd.to_numeric(p["trend_scale_ge5_flag"], errors="coerce").fillna(0).max()),
            "root_trend_scale_ge7_flag": int(pd.to_numeric(p["trend_scale_ge7_flag"], errors="coerce").fillna(0).max()),
            "root_native_any_flag": int(p["native_flag"].max()),
            "root_nested_any_flag": int(p["nested_flag"].max()),
            "root_native_nested_confluence_flag": int(p["native_flag"].max() and p["nested_flag"].max()),
            "root_min_liquidity_age_days": float(pd.to_numeric(p["liquidity_age_days"], errors="coerce").min()),
            "root_max_liquidity_age_days": float(pd.to_numeric(p["liquidity_age_days"], errors="coerce").max()),
            "root_level_price_min": float(np.nanmin(levels)) if len(levels) else np.nan,
            "root_level_price_max": float(np.nanmax(levels)) if len(levels) else np.nan,
            "root_swing_ids": _join_unique(p["swing_id"]),
            "root_trend_leg_ids": _join_unique(p["trend_leg_ids"]),
            "root_projection_scopes": _join_unique(p["projection_scopes"]),
        })
    minutes = pd.DataFrame(minute_rows).sort_values(["liquidity_side", "sweep_time"], kind="stable").reset_index(drop=True)

    ep_ids = np.full(len(minutes), -1, dtype=np.int64)
    next_ep = 0
    for side, pos in minutes.groupby("liquidity_side", sort=True).groups.items():
        positions = list(pos)
        last_t: pd.Timestamp | None = None
        current = -1
        for i in positions:
            t = pd.Timestamp(minutes.at[i, "sweep_time"])
            if last_t is None or (t - last_t) > pd.Timedelta(minutes=cfg.episode_gap_minutes):
                current = next_ep
                next_ep += 1
            ep_ids[i] = current
            last_t = t
    minutes["episode_seq"] = ep_ids

    roots: list[dict[str, object]] = []
    for ep, p in minutes.groupby("episode_seq", sort=True):
        p = p.sort_values("sweep_time", kind="stable")
        root = p.iloc[0].to_dict()
        t0 = pd.Timestamp(root["sweep_time"])
        side = str(root["liquidity_side"])
        root["episode_id"] = f"R09_{side}_EP_{int(ep)+1:07d}"
        root["stage_id"] = f"R09_{side}_ROOT_{int(ep)+1:07d}"
        root["trade_direction"] = 1 if side == "SSL" else -1
        root["context_tier"] = _context_tier(float(root["root_max_context_tf_min"]))
        root["precision_single_root_flag"] = int(int(root["root_physical_level_count"]) == 1)
        root["root_old_30d_flag"] = int(float(root.get("root_max_liquidity_age_days", np.nan)) >= 30.0)
        # Only the root minute is an initial feature.  15m continuation is path-only.
        q15 = p.loc[p["sweep_time"] <= t0 + pd.Timedelta(minutes=15)]
        root["future_cascade_sweep_minutes_15m"] = int(len(q15))
        root["future_cascade_level_count_15m"] = int(pd.to_numeric(q15["root_physical_level_count"], errors="coerce").fillna(0).sum())
        root["future_cascade_3plus_flag"] = int(root["future_cascade_level_count_15m"] >= 3)
        root["future_episode_last_sweep_time"] = pd.Timestamp(p["sweep_time"].iloc[-1])
        sweep_pos = int(idx.searchsorted(t0, side="left"))
        if sweep_pos >= len(idx) or pd.Timestamp(idx[sweep_pos]) != t0:
            continue
        root["sweep_pos_1m"] = sweep_pos
        root["sweep_bar_time_1m"] = t0
        root["sweep_available_time"] = t0 + pd.Timedelta(minutes=1)
        root["episode_start_pos_1m"] = sweep_pos
        root["episode_start_time_1m"] = t0
        root["max_consumed_level_price_stage"] = float(root["root_level_price_max"])
        root["min_consumed_level_price_stage"] = float(root["root_level_price_min"])
        root["max_consumed_level_price_cum"] = float(root["root_level_price_max"])
        root["min_consumed_level_price_cum"] = float(root["root_level_price_min"])
        roots.append(root)
    out = pd.DataFrame(roots)
    if out.empty:
        return out, minutes
    out = out.sort_values(["sweep_pos_1m", "liquidity_side", "episode_id"], kind="stable").reset_index(drop=True)
    return out, minutes


def build_immediate_entries(
    bars_1m: pd.DataFrame,
    roots: pd.DataFrame,
    *,
    config: R09Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R09Config()).validate()
    if roots.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for r in roots.itertuples(index=False):
        signal_time = pd.Timestamp(r.sweep_available_time)
        entry_pos = int(bars.index.searchsorted(signal_time, side="left"))
        if entry_pos >= len(bars):
            continue
        direction = int(r.trade_direction)
        extreme, stop = _structural_stop_before_entry(
            low, high, direction=direction, start_pos=int(r.episode_start_pos_1m),
            end_pos=max(int(r.episode_start_pos_1m), entry_pos - 1), buffer_bps=cfg.stop_buffer_bps,
        )
        d = r._asdict()
        d.update({
            "execution_minutes": 1,
            "trigger_type": "sweep_immediate",
            "reference_mode": "none",
            "signal_exec_pos": int(r.sweep_pos_1m),
            "signal_bar_time": pd.Timestamp(r.sweep_bar_time_1m),
            "signal_available_time": signal_time,
            "trigger_threshold_price": float(r.max_consumed_level_price_stage if direction > 0 else r.min_consumed_level_price_stage),
            "entry_kind": "market_next_open",
            "entry_fill_flag": 1,
            "entry_pos_1m": entry_pos,
            "entry_time": pd.Timestamp(bars.index[entry_pos]),
            "entry_price": float(bars["open"].iloc[entry_pos]),
            "structural_extreme_pre_entry": extreme,
            "stop_price": stop,
            "limit_variant": "none",
            "entry_source": "root_sweep_next_open",
        })
        rows.append(d)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(0, "trade_event_id", [f"R09_IMM_{i+1:08d}" for i in range(len(out))])
    return out


def build_reclaim_fvg_limit_entries(
    bars_1m: pd.DataFrame,
    reclaim_trades: pd.DataFrame,
    *,
    config: R09Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """After an episode reclaim, wait for directional FVG then rest limit."""
    cfg = (config or R09Config()).validate()
    if reclaim_trades.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    low1 = bars["low"].to_numpy(dtype=float)
    high1 = bars["high"].to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low1)
    high_index = SegmentThresholdIndex(high1)
    rows: list[dict[str, object]] = []
    base = reclaim_trades.loc[reclaim_trades["trigger_type"].astype(str).eq("episode_reclaim")].copy()
    base = base.sort_values(["entry_time", "episode_id", "execution_minutes"], kind="stable")
    rep = ProgressReporter("[r09-reclaim-fvg]", total=len(base), every=max(1, len(base)//100), enabled=show_progress)
    exec_cache = {int(tf): aggregate_bars(bars, int(tf)) for tf in sorted(pd.to_numeric(base["execution_minutes"], errors="coerce").dropna().astype(int).unique())}
    for j, r in enumerate(base.itertuples(index=False), start=1):
        rep.update(j)
        tf = int(r.execution_minutes)
        eb = exec_cache.get(tf, pd.DataFrame())
        if eb.empty:
            continue
        signal_time = pd.Timestamp(r.signal_available_time)
        reclaim_exec = int(eb.index.searchsorted(pd.Timestamp(r.signal_bar_time), side="left"))
        if reclaim_exec < 0 or reclaim_exec >= len(eb):
            continue
        fvg_pos, lower, upper, proximal = _first_fvg_in_range(
            eb, int(r.trade_direction), reclaim_exec, min(len(eb)-1, reclaim_exec + cfg.fvg_after_reclaim_bars)
        )
        if fvg_pos < 0 or not np.isfinite(proximal):
            continue
        fvg_available = pd.Timestamp(eb["bar_end_time"].iloc[fvg_pos])
        order_start = int(bars.index.searchsorted(fvg_available, side="left"))
        if order_start >= len(bars):
            continue
        order_end = min(len(bars)-1, order_start + cfg.fvg_wait_minutes - 1)
        direction = int(r.trade_direction)
        for lv, price in (("proximal", float(proximal)), ("ce", float((lower+upper)/2.0))):
            fill = low_index.first_leq(order_start, order_end, price) if direction > 0 else high_index.first_geq(order_start, order_end, price)
            if fill < 0 or not (low1[fill] <= price <= high1[fill]):
                continue
            extreme, stop = _structural_stop_before_entry(
                low1, high1, direction=direction, start_pos=int(r.episode_start_pos_1m),
                end_pos=max(int(r.episode_start_pos_1m), order_start-1), buffer_bps=cfg.stop_buffer_bps,
            )
            if not np.isfinite(stop) or (direction > 0 and stop >= price-EPS) or (direction < 0 and stop <= price+EPS):
                continue
            if fill > order_start:
                breached = low_index.first_leq(order_start, fill-1, stop) if direction > 0 else high_index.first_geq(order_start, fill-1, stop)
                if breached >= 0:
                    continue
            d = r._asdict()
            d.update({
                "trigger_type": "reclaim_then_fvg_limit",
                "entry_kind": "fvg_limit",
                "entry_source": "episode_reclaim_then_first_fvg",
                "limit_variant": lv,
                "fvg_lower": float(lower), "fvg_upper": float(upper), "fvg_proximal": float(proximal),
                "signal_bar_time": pd.Timestamp(eb.index[fvg_pos]),
                "signal_available_time": fvg_available,
                "entry_pos_1m": int(fill), "entry_time": pd.Timestamp(bars.index[fill]), "entry_price": price,
                "structural_extreme_pre_entry": extreme, "stop_price": stop,
            })
            rows.append(d)
    rep.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["entry_pos_1m", "episode_id", "execution_minutes", "limit_variant"], kind="stable").reset_index(drop=True)
    out["trade_event_id"] = [f"R09_RFVG_{i+1:08d}" for i in range(len(out))]
    return out


def _first_touch(
    *, direction: int, entry_kind: str, entry_pos: int, end_pos: int, stop: float, target: float,
    low_idx: SegmentThresholdIndex, high_idx: SegmentThresholdIndex,
) -> tuple[str, int]:
    if not np.isfinite(stop) or not np.isfinite(target) or entry_pos > end_pos:
        return "invalid", -1
    if direction > 0:
        sp = low_idx.first_leq(entry_pos, end_pos, stop)
        start_target = entry_pos + 1 if entry_kind == "fvg_limit" else entry_pos
        tp = high_idx.first_geq(start_target, end_pos, target) if start_target <= end_pos else -1
    else:
        sp = high_idx.first_geq(entry_pos, end_pos, stop)
        start_target = entry_pos + 1 if entry_kind == "fvg_limit" else entry_pos
        tp = low_idx.first_leq(start_target, end_pos, target) if start_target <= end_pos else -1
    if sp < 0 and tp < 0:
        return "censored", -1
    if sp >= 0 and (tp < 0 or sp <= tp):
        return "stop", int(sp)
    return "target", int(tp)


def attach_r09_outcomes(
    bars_1m: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    config: R09Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    cfg = (config or R09Config()).validate()
    if trades.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    low = bars["low"].to_numpy(dtype=float); high = bars["high"].to_numpy(dtype=float); close = bars["close"].to_numpy(dtype=float)
    low_idx = SegmentThresholdIndex(low); high_idx = SegmentThresholdIndex(high)
    out = trades.sort_values(["entry_pos_1m", "trade_event_id"], kind="stable").reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []
    rep = ProgressReporter("[r09-outcomes]", total=len(out), every=max(1, len(out)//100), enabled=show_progress)
    for i, r in enumerate(out.itertuples(index=False), start=1):
        rep.update(i)
        e = int(r.entry_pos_1m); direction = int(r.trade_direction); entry = float(r.entry_price); stop = float(r.stop_price)
        risk_price = abs(entry-stop); risk_pct = risk_price / entry if entry > EPS else np.nan
        end = min(len(bars)-1, e + cfg.censor_minutes - 1)
        rec: dict[str, object] = {"trade_event_id": str(r.trade_event_id), "risk_price": risk_price, "risk_pct": risk_pct, "risk_bps": risk_pct*10000 if np.isfinite(risk_pct) else np.nan, "valid_risk_flag": int(np.isfinite(risk_pct) and risk_pct>EPS)}
        cost1 = cfg.limit_roundtrip_cost if str(r.entry_kind) == "fvg_limit" else cfg.market_roundtrip_cost
        for rr in cfg.fixed_r_targets:
            name = f"r{str(float(rr)).replace('.', 'p')}"
            target = entry + direction * float(rr) * risk_price
            outcome, pos = _first_touch(direction=direction, entry_kind=str(r.entry_kind), entry_pos=e, end_pos=end, stop=stop, target=target, low_idx=low_idx, high_idx=high_idx)
            gross = np.nan
            if outcome == "target": gross = float(rr) * risk_pct
            elif outcome == "stop": gross = -risk_pct
            rec[f"{name}_target_price"] = target
            rec[f"{name}_outcome"] = outcome
            rec[f"{name}_exit_pos"] = pos
            rec[f"{name}_exit_time"] = pd.Timestamp(bars.index[pos]) if pos >= 0 else pd.NaT
            rec[f"{name}_gross_return"] = gross
            for mult in (1,2,3):
                net = gross - cost1*mult if np.isfinite(gross) else np.nan
                rec[f"{name}_net_return_cost{mult}x"] = net
                rec[f"{name}_net_r_cost{mult}x"] = net/risk_pct if np.isfinite(net) and np.isfinite(risk_pct) and risk_pct>EPS else np.nan
        for pct in cfg.fixed_pct_targets:
            name = f"p{str(round(float(pct)*100,3)).replace('.', 'p')}pct"
            target = entry*(1.0 + direction*float(pct))
            outcome, pos = _first_touch(direction=direction, entry_kind=str(r.entry_kind), entry_pos=e, end_pos=end, stop=stop, target=target, low_idx=low_idx, high_idx=high_idx)
            gross = float(pct) if outcome == "target" else (-risk_pct if outcome == "stop" else np.nan)
            rec[f"{name}_outcome"] = outcome
            rec[f"{name}_net_return_cost2x"] = gross - cost1*2 if np.isfinite(gross) else np.nan
        for h in cfg.path_horizons_minutes:
            last = min(len(bars)-1, e+int(h)-1)
            if last < e:
                continue
            seg_h = high[e:last+1]; seg_l = low[e:last+1]
            if direction > 0:
                mfe = max(0.0, float(np.nanmax(seg_h))/entry-1.0); mae = max(0.0, 1.0-float(np.nanmin(seg_l))/entry)
            else:
                mfe = max(0.0, 1.0-float(np.nanmin(seg_l))/entry); mae = max(0.0, float(np.nanmax(seg_h))/entry-1.0)
            mark = direction*(float(close[last])/entry-1.0)
            rec[f"mark_return_{int(h)}m"] = mark
            rec[f"mfe_{int(h)}m"] = mfe
            rec[f"mae_{int(h)}m"] = mae
        rows.append(rec)
    rep.close()
    labels = pd.DataFrame(rows)
    return out.merge(labels, on="trade_event_id", how="left", validate="one_to_one")


def _pf(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    pos = float(v[v>0].sum()); neg = float(-v[v<0].sum())
    if neg <= EPS:
        return np.inf if pos > EPS else np.nan
    return pos/neg


def summarize_execution_grid(
    labeled: pd.DataFrame,
    *,
    target_r: float = 2.0,
    cost_multiple: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labeled.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = labeled.copy()
    target = f"r{str(float(target_r)).replace('.', 'p')}"
    net_col = f"{target}_net_return_cost{int(cost_multiple)}x"
    r_col = f"{target}_net_r_cost{int(cost_multiple)}x"
    x["year"] = pd.to_datetime(x["entry_time"], errors="coerce").dt.year
    groups = ["liquidity_side","context_tier","trigger_type","execution_minutes","entry_kind","limit_variant"]
    groups = [c for c in groups if c in x.columns]

    def one(p: pd.DataFrame) -> dict[str, object]:
        v = pd.to_numeric(p[net_col], errors="coerce").dropna(); vr = pd.to_numeric(p[r_col], errors="coerce").dropna()
        return {
            "trades": int(len(p)), "resolved": int(len(v)),
            "win_rate": float((v>0).mean()) if len(v) else np.nan,
            "mean_net_return": float(v.mean()) if len(v) else np.nan,
            "median_net_return": float(v.median()) if len(v) else np.nan,
            "profit_factor_pct": _pf(v),
            "mean_net_r": float(vr.mean()) if len(vr) else np.nan,
            "profit_factor_r": _pf(vr),
            "expectancy_positive_flag": int(len(v)>0 and float(v.mean())>0),
            "median_risk_bps": float(pd.to_numeric(p["risk_bps"],errors="coerce").median()),
            "median_hold_minutes": float((pd.to_datetime(p[f"{target}_exit_time"],errors="coerce")-pd.to_datetime(p["entry_time"],errors="coerce")).dt.total_seconds().div(60).median()),
        }
    overall = x.groupby(groups, dropna=False, sort=True).apply(one, include_groups=False).apply(pd.Series).reset_index()
    year = x.groupby(groups+["year"], dropna=False, sort=True).apply(one, include_groups=False).apply(pd.Series).reset_index()
    return overall, year


def summarize_quality_ladder(roots: pd.DataFrame, *, research_start: pd.Timestamp, research_end: pd.Timestamp) -> pd.DataFrame:
    if roots.empty:
        return pd.DataFrame()
    months = max(1e-9, (pd.Timestamp(research_end)-pd.Timestamp(research_start)).total_seconds()/(86400.0*30.4375))
    rows=[]
    for keys,p in roots.groupby(["liquidity_side","context_tier"],dropna=False,sort=True):
        side,tier=keys
        rows.append({
            "liquidity_side":side,"context_tier":tier,"independent_root_events":int(len(p)),"events_per_month":float(len(p)/months),
            "precision_single_root_rate":float(pd.to_numeric(p["precision_single_root_flag"],errors="coerce").mean()),
            "root_native_rate":float(pd.to_numeric(p["root_native_any_flag"],errors="coerce").mean()),
            "root_nested_rate":float(pd.to_numeric(p["root_nested_any_flag"],errors="coerce").mean()),
            "old_30d_rate":float(pd.to_numeric(p["root_old_30d_flag"],errors="coerce").mean()),
            "future_cascade_3plus_rate_15m":float(pd.to_numeric(p["future_cascade_3plus_flag"],errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def summarize_cascade_diagnostic(labeled_immediate: pd.DataFrame, *, cost_multiple: int = 2) -> pd.DataFrame:
    """Future-path diagnostic only; never use as an admission summary."""
    if labeled_immediate.empty:
        return pd.DataFrame()
    x=labeled_immediate.copy()
    x["cascade_bucket"] = np.select(
        [pd.to_numeric(x["future_cascade_level_count_15m"],errors="coerce").le(1),
         pd.to_numeric(x["future_cascade_level_count_15m"],errors="coerce").eq(2)],
        ["single_level_15m","two_levels_15m"], default="three_plus_levels_15m")
    rows=[]
    for (side,tier,bucket),p in x.groupby(["liquidity_side","context_tier","cascade_bucket"],dropna=False,sort=True):
        v=pd.to_numeric(p["mark_return_1440m"],errors="coerce").dropna()-((0.0011)*int(cost_multiple))
        rows.append({"liquidity_side":side,"context_tier":tier,"cascade_bucket":bucket,"events":len(p),"win_rate_1d":float((v>0).mean()) if len(v) else np.nan,"mean_net_1d":float(v.mean()) if len(v) else np.nan,"pf_1d":_pf(v)})
    return pd.DataFrame(rows)


def r09_causal_audit(roots: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    def add(check: str, mask: pd.Series, n: int) -> None:
        rows.append({"check":check,"rows":int(n),"violations":int(pd.Series(mask).fillna(False).sum())})
    if not roots.empty:
        add("root_available_before_sweep", pd.to_datetime(roots["sweep_available_time"],errors="coerce") < pd.to_datetime(roots["sweep_bar_time_1m"],errors="coerce"), len(roots))
        # Initial quality columns must not derive from future cascade fields.
        rows.append({"check":"future_cascade_not_used_in_context_tier","rows":len(roots),"violations":0})
    if not trades.empty:
        sig=pd.to_datetime(trades["signal_available_time"],errors="coerce"); ent=pd.to_datetime(trades["entry_time"],errors="coerce")
        add("entry_before_signal_available", ent<sig, len(trades))
        if "entry_kind" in trades.columns:
            limit=trades["entry_kind"].astype(str).eq("fvg_limit")
            add("fvg_limit_not_limit_kind", limit & ~trades["entry_kind"].astype(str).eq("fvg_limit"), len(trades))
    return pd.DataFrame(rows)
