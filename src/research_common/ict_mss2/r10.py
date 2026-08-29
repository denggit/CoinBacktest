#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R10 unified ICT liquidity trading engine helpers.

R10 is a consolidation study, not another signal atlas.  It takes the broad
R09 SSL root universe and turns it into one long-only trade lifecycle:

    full-trend qualified SSL sweep -> 2m episode reclaim -> one base position
        -> optional later structural MSS state upgrade
        -> base profit realization + slow structural runner

MSS and FVG are no longer independent trades in this engine.  R10 keeps the
entry rule fixed and uses MSS only as a later causal state upgrade.  Add-ons are
disabled in v1.  The purpose is to test whether a single coherent lifecycle can
produce a smoother capital curve without hard-filtering down to only A+ setups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars
from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R10Config:
    execution_minutes: int = 2
    stop_buffer_bps: float = 2.0
    base_target_r: float = 2.0
    major_upgrade_r: float = 3.0
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    max_notional_multiple: float = 3.0
    # Fixed before R10 results are known; not a parameter search.
    risk_schedules: tuple[tuple[str, float, float, float, float], ...] = (
        ("equal_low", 0.0050, 0.0050, 0.0050, 0.0050),
        ("quality_scaled", 0.0035, 0.0010, 0.0075, 0.0075),
        ("quality_scaled_no_B", 0.0035, 0.0000, 0.0075, 0.0075),
    )
    lifecycle_variants: tuple[str, ...] = (
        "full_5m_ltl",
        "base75_2r_runner25",
        "base50_2r_runner50",
    )

    def validate(self) -> "R10Config":
        if int(self.execution_minutes) <= 0:
            raise ValueError("execution_minutes must be positive")
        if float(self.base_target_r) <= 0 or float(self.major_upgrade_r) <= float(self.base_target_r):
            raise ValueError("major_upgrade_r must be above base_target_r > 0")
        if float(self.market_roundtrip_cost) < 0:
            raise ValueError("market_roundtrip_cost cannot be negative")
        if float(self.max_notional_multiple) <= 0:
            raise ValueError("max_notional_multiple must be positive")
        allowed = {"full_5m_ltl", "base75_2r_runner25", "base50_2r_runner50"}
        if set(self.lifecycle_variants) - allowed:
            raise ValueError("unknown lifecycle variant")
        for schedule in self.risk_schedules:
            if len(schedule) != 5:
                raise ValueError("risk schedule must be (name,C,B,A,A+)")
            if any(float(v) < 0 or float(v) > 0.02 for v in schedule[1:]):
                raise ValueError("risk budget must be inside [0,2%]")
        return self


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").dropna()
    if x.empty:
        return np.nan
    gp = float(x.loc[x > 0].sum())
    gl = float(-x.loc[x < 0].sum())
    if gl <= EPS:
        return np.inf if gp > 0 else np.nan
    return gp / gl


def build_unified_reclaim_base(r09_outcomes: pd.DataFrame, *, execution_minutes: int = 2) -> pd.DataFrame:
    """One causal base entry per SSL root episode: 2m episode reclaim only."""
    if r09_outcomes.empty:
        return pd.DataFrame()
    x = r09_outcomes.copy()
    mask = (
        x["liquidity_side"].astype(str).eq("SSL")
        & x["trigger_type"].astype(str).eq("episode_reclaim")
        & pd.to_numeric(x["execution_minutes"], errors="coerce").eq(int(execution_minutes))
        & x["entry_kind"].astype(str).eq("market_next_open")
    )
    x = x.loc[mask].copy()
    if x.empty:
        return x
    x["entry_time"] = pd.to_datetime(x["entry_time"], errors="coerce")
    x["signal_available_time"] = pd.to_datetime(x["signal_available_time"], errors="coerce")
    x["sweep_bar_time_1m"] = pd.to_datetime(x["sweep_bar_time_1m"], errors="coerce")
    x = x.dropna(subset=["episode_id", "entry_time", "entry_price", "stop_price"])
    x = x.sort_values(["entry_time", "episode_id", "trade_event_id"], kind="stable")
    # R09 can contain only one episode-reclaim per tf, but enforce the contract.
    x = x.drop_duplicates("episode_id", keep="first").reset_index(drop=True)
    x["initial_stop_price"] = pd.to_numeric(x["stop_price"], errors="coerce")
    x["initial_risk_return"] = (pd.to_numeric(x["entry_price"], errors="coerce") - x["initial_stop_price"]) / pd.to_numeric(x["entry_price"], errors="coerce")
    x = x.loc[x["initial_risk_return"].gt(EPS)].copy()
    return x


def build_structural_mss_upgrade_map(
    base: pd.DataFrame,
    r09_outcomes: pd.DataFrame,
    *,
    execution_minutes: int = 2,
) -> pd.DataFrame:
    """Attach earliest later 2m structural MSS as a *state upgrade*, not entry."""
    if base.empty:
        return pd.DataFrame()
    m = r09_outcomes.loc[
        r09_outcomes["liquidity_side"].astype(str).eq("SSL")
        & r09_outcomes["trigger_type"].astype(str).eq("mss_structural_market")
        & pd.to_numeric(r09_outcomes["execution_minutes"], errors="coerce").eq(int(execution_minutes))
    ].copy()
    if m.empty:
        out = base[["episode_id"]].copy()
        out["mss_upgrade_time"] = pd.NaT
        out["mss_upgrade_price"] = np.nan
        return out
    m["mss_upgrade_time"] = pd.to_datetime(m["signal_available_time"], errors="coerce")
    m["mss_upgrade_price"] = pd.to_numeric(m["entry_price"], errors="coerce")
    m = m.sort_values(["episode_id", "mss_upgrade_time"], kind="stable")
    b = base[["episode_id", "entry_time"]].copy()
    b["entry_time"] = pd.to_datetime(b["entry_time"], errors="coerce")
    q = b.merge(m[["episode_id", "mss_upgrade_time", "mss_upgrade_price"]], on="episode_id", how="left")
    q = q.loc[q["mss_upgrade_time"].isna() | q["mss_upgrade_time"].ge(q["entry_time"])].copy()
    q = q.sort_values(["episode_id", "mss_upgrade_time"], kind="stable").drop_duplicates("episode_id", keep="first")
    return b[["episode_id"]].merge(q[["episode_id", "mss_upgrade_time", "mss_upgrade_price"]], on="episode_id", how="left")


def _prepare_ltl_arrays(trailing_events: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if trailing_events.empty:
        return {5: (np.array([], dtype=np.int64), np.array([], dtype=float)), 15: (np.array([], dtype=np.int64), np.array([], dtype=float))}
    e = trailing_events.copy()
    e["activation_time"] = pd.to_datetime(e["activation_time"], errors="coerce")
    e = e.loc[e["event_type"].astype(str).eq("ltl") & pd.to_numeric(e["trail_tf_min"], errors="coerce").isin([5, 15])].dropna(subset=["activation_time", "anchor_price"])
    e["pos"] = idx.searchsorted(pd.DatetimeIndex(e["activation_time"]), side="left")
    e = e.loc[pd.to_numeric(e["pos"], errors="coerce").between(0, len(idx) - 1)]
    for tf in (5, 15):
        p = e.loc[pd.to_numeric(e["trail_tf_min"], errors="coerce").eq(tf)].sort_values(["pos", "anchor_price"], kind="stable")
        out[tf] = (
            pd.to_numeric(p["pos"], errors="coerce").to_numpy(dtype=np.int64),
            pd.to_numeric(p["anchor_price"], errors="coerce").to_numpy(dtype=float),
        )
    return out


def _fraction_for_variant(variant: str) -> tuple[float, float]:
    if variant == "full_5m_ltl":
        return 0.0, 1.0
    if variant == "base75_2r_runner25":
        return 0.75, 0.25
    if variant == "base50_2r_runner50":
        return 0.50, 0.50
    raise ValueError(f"unknown lifecycle {variant}")


def simulate_unified_lifecycles(
    base: pd.DataFrame,
    primary_1m: pd.DataFrame,
    trailing_events: pd.DataFrame,
    mss_upgrades: pd.DataFrame,
    *,
    config: R10Config | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Simulate the consolidated one-position lifecycle with stop-first ties.

    Partial variants do **not** trail before the 2R base target.  This is an
    intentional response to the R05/R06 finding that fast early trailing kills
    large winners.  After 2R, the runner moves to break-even from the next 1m
    bar and then follows 5m LTL.  If a causal structural MSS has occurred and
    price reaches 3R, later runner management slows to newly-confirmed 15m LTL.
    """
    cfg = (config or R10Config()).validate()
    if base.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    idx = pd.DatetimeIndex(bars.index)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    high_index = SegmentThresholdIndex(high)
    ltl = _prepare_ltl_arrays(trailing_events, idx)
    mss = mss_upgrades.set_index("episode_id") if not mss_upgrades.empty else pd.DataFrame()
    buf = float(cfg.stop_buffer_bps) / 10_000.0

    rows: list[dict[str, object]] = []
    total = len(base) * len(cfg.lifecycle_variants)
    reporter = ProgressReporter("[r10-lifecycle]", total=total, every=max(1, total // 100), enabled=show_progress)
    done = 0
    for src in base.itertuples(index=False):
        entry_pos = int(src.entry_pos_1m)
        entry = float(src.entry_price)
        stop0 = float(src.initial_stop_price)
        if not (0 <= entry_pos < len(idx) and np.isfinite(entry) and np.isfinite(stop0) and 0 < stop0 < entry):
            continue
        risk_px = entry - stop0
        target2 = entry + float(cfg.base_target_r) * risk_px
        target3 = entry + float(cfg.major_upgrade_r) * risk_px
        mss_time = pd.NaT
        if not isinstance(mss, pd.DataFrame) or not mss.empty:
            if src.episode_id in mss.index:
                v = mss.loc[src.episode_id, "mss_upgrade_time"]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                mss_time = pd.to_datetime(v, errors="coerce")
        mss_pos = int(idx.searchsorted(mss_time, side="left")) if pd.notna(mss_time) else -1
        if mss_pos >= len(idx):
            mss_pos = -1
        for variant in cfg.lifecycle_variants:
            done += 1
            reporter.update(done)
            base_frac, runner_frac = _fraction_for_variant(variant)
            active_stop = stop0
            base_exit_pos = -1
            base_exit_price = np.nan
            runner_exit_pos = -1
            runner_exit_price = np.nan
            trail_updates_5m = 0
            trail_updates_15m = 0
            major_pos = -1
            stop_before_base_flag = 0

            if variant == "full_5m_ltl":
                positions, anchors = ltl[5]
                left = int(np.searchsorted(positions, entry_pos, side="right"))
                cursor = entry_pos
                for j in range(left, len(positions)):
                    pos = int(positions[j])
                    breach = low_index.first_leq(cursor, pos - 1, active_stop)
                    if breach >= 0:
                        runner_exit_pos = int(breach)
                        runner_exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else active_stop
                        break
                    cand = float(anchors[j]) * (1.0 - buf)
                    if np.isfinite(cand) and cand > active_stop:
                        active_stop = cand
                        trail_updates_5m += 1
                        if np.isfinite(open_[pos]) and open_[pos] <= active_stop:
                            runner_exit_pos = pos
                            runner_exit_price = float(open_[pos])
                            break
                    cursor = pos
                if runner_exit_pos < 0:
                    breach = low_index.first_leq(cursor, len(idx) - 1, active_stop)
                    if breach >= 0:
                        runner_exit_pos = int(breach)
                        runner_exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else active_stop
            else:
                # Same-bar TP+SL is pessimistically stop-first.
                stop_pos = low_index.first_leq(entry_pos, len(idx) - 1, stop0)
                tp2_pos = high_index.first_geq(entry_pos, len(idx) - 1, target2)
                if stop_pos >= 0 and (tp2_pos < 0 or stop_pos <= tp2_pos):
                    runner_exit_pos = int(stop_pos)
                    runner_exit_price = float(min(open_[stop_pos], stop0)) if np.isfinite(open_[stop_pos]) else stop0
                    stop_before_base_flag = 1
                elif tp2_pos >= 0:
                    base_exit_pos = int(tp2_pos)
                    base_exit_price = float(target2)
                    # BE becomes active only from the *next* 1m bar.
                    runner_start = min(len(idx) - 1, base_exit_pos + 1)
                    active_stop = max(stop0, entry)
                    # Major state starts only after both MSS and 3R have become known.
                    tp3_pos = high_index.first_geq(entry_pos, len(idx) - 1, target3)
                    if mss_pos >= 0 and tp3_pos >= 0:
                        major_pos = max(mss_pos, int(tp3_pos))
                    p5, a5 = ltl[5]
                    p15, a15 = ltl[15]
                    # Merge relevant event boundaries once for this runner.
                    i5 = int(np.searchsorted(p5, runner_start, side="left"))
                    i15 = int(np.searchsorted(p15, runner_start, side="left"))
                    cursor = runner_start
                    while i5 < len(p5) or i15 < len(p15):
                        next5 = int(p5[i5]) if i5 < len(p5) else len(idx) + 1
                        next15 = int(p15[i15]) if i15 < len(p15) else len(idx) + 1
                        use15 = major_pos >= 0 and min(next5, next15) >= major_pos
                        if use15:
                            pos = next15
                            anchor = float(a15[i15]) if i15 < len(p15) else np.nan
                            if i15 < len(p15):
                                i15 += 1
                            # Once major, ignore 5m events as the runner deliberately slows.
                            while i5 < len(p5) and int(p5[i5]) <= pos:
                                i5 += 1
                        else:
                            pos = next5
                            anchor = float(a5[i5]) if i5 < len(p5) else np.nan
                            if i5 < len(p5):
                                i5 += 1
                        if pos > len(idx) - 1:
                            break
                        if pos < runner_start:
                            continue
                        breach = low_index.first_leq(cursor, pos - 1, active_stop)
                        if breach >= 0:
                            runner_exit_pos = int(breach)
                            runner_exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else active_stop
                            break
                        cand = anchor * (1.0 - buf) if np.isfinite(anchor) else np.nan
                        if np.isfinite(cand) and cand > active_stop:
                            active_stop = cand
                            if use15:
                                trail_updates_15m += 1
                            else:
                                trail_updates_5m += 1
                            if np.isfinite(open_[pos]) and open_[pos] <= active_stop:
                                runner_exit_pos = pos
                                runner_exit_price = float(open_[pos])
                                break
                        cursor = pos
                    if runner_exit_pos < 0:
                        breach = low_index.first_leq(cursor, len(idx) - 1, active_stop)
                        if breach >= 0:
                            runner_exit_pos = int(breach)
                            runner_exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else active_stop

            final_pos = runner_exit_pos
            final_px = runner_exit_price
            if variant == "full_5m_ltl":
                gross = final_px / entry - 1.0 if final_pos >= 0 and np.isfinite(final_px) else np.nan
            elif stop_before_base_flag:
                gross = final_px / entry - 1.0 if final_pos >= 0 and np.isfinite(final_px) else np.nan
            elif base_exit_pos >= 0 and final_pos >= 0 and np.isfinite(final_px):
                gross = base_frac * (base_exit_price / entry - 1.0) + runner_frac * (final_px / entry - 1.0)
            else:
                gross = np.nan
            rows.append({
                "episode_id": src.episode_id,
                "trade_event_id": src.trade_event_id,
                "context_tier": src.context_tier,
                "entry_time": src.entry_time,
                "entry_pos_1m": entry_pos,
                "entry_price": entry,
                "initial_stop_price": stop0,
                "initial_risk_return": risk_px / entry,
                "lifecycle_variant": variant,
                "base_fraction": base_frac,
                "runner_fraction": runner_frac,
                "base_target_r": float(cfg.base_target_r),
                "base_target_price": target2,
                "base_target_hit_flag": int(base_exit_pos >= 0),
                "base_exit_time": idx[base_exit_pos] if base_exit_pos >= 0 else pd.NaT,
                "base_exit_price": base_exit_price,
                "stop_before_base_flag": stop_before_base_flag,
                "mss_upgrade_time": mss_time,
                "mss_upgrade_flag": int(pd.notna(mss_time)),
                "major_upgrade_time": idx[major_pos] if major_pos >= 0 else pd.NaT,
                "major_upgrade_flag": int(major_pos >= 0),
                "trail_updates_5m": trail_updates_5m,
                "trail_updates_15m": trail_updates_15m,
                "final_stop_price": active_stop,
                "exit_time": idx[final_pos] if final_pos >= 0 else pd.NaT,
                "exit_pos_1m": final_pos,
                "exit_price": final_px,
                "holding_minutes": int(final_pos - entry_pos) if final_pos >= 0 else np.nan,
                "right_edge_open_flag": int(final_pos < 0),
                "gross_return_unit_notional": gross,
                "root_physical_level_count": getattr(src, "root_physical_level_count", np.nan),
                "root_native_any_flag": getattr(src, "root_native_any_flag", np.nan),
                "root_nested_any_flag": getattr(src, "root_nested_any_flag", np.nan),
                "root_old_30d_flag": getattr(src, "root_old_30d_flag", np.nan),
                "root_max_context_tf_min": getattr(src, "root_max_context_tf_min", np.nan),
                "root_max_swing_tf_min": getattr(src, "root_max_swing_tf_min", np.nan),
                "root_swing_ids": getattr(src, "root_swing_ids", ""),
                "root_trend_leg_ids": getattr(src, "root_trend_leg_ids", ""),
            })
    return pd.DataFrame(rows)


def _budget(schedule: tuple[str, float, float, float, float], tier: str) -> float:
    mapping = {
        "C_15m_context": float(schedule[1]),
        "B_30m_context": float(schedule[2]),
        "A_1H_context": float(schedule[3]),
        "A+_4H_context": float(schedule[4]),
    }
    return mapping.get(str(tier), 0.0)


def attach_risk_sizing(paths: pd.DataFrame, *, config: R10Config | None = None) -> pd.DataFrame:
    cfg = (config or R10Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for p in paths.itertuples(index=False):
        unit_risk = float(p.initial_risk_return)
        if not np.isfinite(unit_risk) or unit_risk <= EPS:
            continue
        gross = float(p.gross_return_unit_notional) if np.isfinite(p.gross_return_unit_notional) else np.nan
        for schedule in cfg.risk_schedules:
            budget = _budget(schedule, str(p.context_tier))
            if budget <= EPS:
                continue
            notional = min(float(cfg.max_notional_multiple), budget / unit_risk)
            worst_case_risk = notional * unit_risk
            for cost_scale in cfg.cost_scales:
                eq_ret = notional * (gross - float(cfg.market_roundtrip_cost) * float(cost_scale)) if np.isfinite(gross) else np.nan
                rows.append({
                    **p._asdict(),
                    "risk_schedule": schedule[0],
                    "risk_budget_fraction": budget,
                    "cost_scale": float(cost_scale),
                    "notional_multiple": notional,
                    "base_notional_multiple": notional,
                    "addon_notional_multiple": 0.0,
                    "addon_used_flag": 0,
                    "worst_case_initial_risk": worst_case_risk,
                    "strategy_equity_return": eq_ret,
                })
    return pd.DataFrame(rows)


def select_single_position(sized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ETH net position per lifecycle/risk/cost scenario."""
    if sized.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = ["lifecycle_variant", "risk_schedule", "cost_scale"]
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for key, part in sized.groupby(groups, dropna=False, sort=True):
        part = part.sort_values(["entry_time", "episode_id"], kind="stable").copy()
        keep = np.zeros(len(part), dtype=bool)
        blocked_until = pd.Timestamp.min
        unresolved = False
        for i, row in enumerate(part.itertuples(index=False)):
            et = pd.Timestamp(row.entry_time)
            if unresolved or et < blocked_until:
                continue
            keep[i] = True
            if pd.isna(row.exit_time):
                unresolved = True
            else:
                blocked_until = pd.Timestamp(row.exit_time)
        ex = part.iloc[np.flatnonzero(keep)].copy()
        parts.append(ex)
        audit.append(dict(zip(groups, key)) | {
            "candidate_signals": int(len(part)),
            "executed_positions": int(len(ex)),
            "overlap_skipped": int(len(part) - len(ex)),
            "execution_rate": float(len(ex) / len(part)) if len(part) else np.nan,
            "right_edge_open_positions": int(ex["exit_time"].isna().sum()) if not ex.empty else 0,
        })
    return pd.concat(parts, ignore_index=True, sort=False), pd.DataFrame(audit)


def build_daily_partial_equity(
    executed: pd.DataFrame,
    primary_1m: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
    market_roundtrip_cost: float = 0.0011,
) -> pd.DataFrame:
    """Daily MTM equity supporting a realized base partial plus open runner."""
    if executed.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    daily = pd.to_numeric(bars["close"], errors="coerce").resample("1D").last().dropna()
    trades = executed.sort_values(["entry_time", "episode_id"], kind="stable").reset_index(drop=True).copy()
    for c in ("entry_time", "base_exit_time", "exit_time"):
        trades[c] = pd.to_datetime(trades[c], errors="coerce")
    first_day = pd.Timestamp(trades["entry_time"].min()).normalize()
    daily = daily.loc[daily.index >= first_day]
    realized = float(initial_equity)
    pointer = 0
    active = -1
    rows: list[dict[str, object]] = []
    n = len(trades)
    for d, px in daily.items():
        day_end = pd.Timestamp(d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if active >= 0:
            r = trades.iloc[active]
            if pd.notna(r["exit_time"]) and pd.Timestamp(r["exit_time"]) <= day_end:
                ret = float(r["strategy_equity_return"]) if np.isfinite(r["strategy_equity_return"]) else 0.0
                realized *= max(EPS, 1.0 + ret)
                pointer = active + 1
                active = -1
        while active < 0 and pointer < n:
            r = trades.iloc[pointer]
            if pd.Timestamp(r["entry_time"]) > day_end:
                break
            if pd.notna(r["exit_time"]) and pd.Timestamp(r["exit_time"]) <= day_end:
                ret = float(r["strategy_equity_return"]) if np.isfinite(r["strategy_equity_return"]) else 0.0
                realized *= max(EPS, 1.0 + ret)
                pointer += 1
                continue
            active = pointer
            break
        mtm = realized
        if active >= 0:
            r = trades.iloc[active]
            entry = float(r["entry_price"])
            nmult = float(r["notional_multiple"])
            bf = float(r["base_fraction"])
            rf = float(r["runner_fraction"])
            base_done = pd.notna(r["base_exit_time"]) and pd.Timestamp(r["base_exit_time"]) <= day_end
            if str(r["lifecycle_variant"]) == "full_5m_ltl":
                gross = float(px) / entry - 1.0
                paid_cost_fraction = 0.5
            elif base_done:
                gross = bf * (float(r["base_exit_price"]) / entry - 1.0) + rf * (float(px) / entry - 1.0)
                paid_cost_fraction = 0.5 + 0.5 * bf
            else:
                gross = float(px) / entry - 1.0
                paid_cost_fraction = 0.5
            open_eq_ret = nmult * (gross - float(market_roundtrip_cost) * float(r["cost_scale"]) * paid_cost_fraction)
            mtm = realized * max(EPS, 1.0 + open_eq_ret)
        rows.append({"date": pd.Timestamp(d), "equity": mtm})
    return pd.DataFrame(rows)


def summarize_scenario(executed: pd.DataFrame, equity: pd.DataFrame, *, months: float) -> dict[str, object]:
    if executed.empty or equity.empty:
        return {}
    x = pd.to_numeric(executed["strategy_equity_return"], errors="coerce").dropna()
    eq = equity.sort_values("date").copy()
    es = pd.to_numeric(eq["equity"], errors="coerce").dropna()
    peak = es.cummax(); dd = es / peak - 1.0
    in_dd = dd.lt(-EPS).to_numpy(dtype=bool); longest = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0; longest = max(longest, cur)
    month = eq.set_index("date")["equity"].resample("ME").last().dropna().pct_change().dropna()
    quarter = eq.set_index("date")["equity"].resample("QE").last().dropna().pct_change().dropna()
    roll90 = eq.set_index("date")["equity"].pct_change(90).dropna()
    logeq = np.log(es.clip(lower=EPS).to_numpy(dtype=float))
    r2 = np.nan
    if len(logeq) > 1:
        t = np.arange(len(logeq), dtype=float); co = np.polyfit(t, logeq, 1); pred = co[0] * t + co[1]
        ssr = float(np.sum((logeq - pred) ** 2)); sst = float(np.sum((logeq - logeq.mean()) ** 2)); r2 = 1.0 - ssr / sst if sst > EPS else np.nan
    ordered = x.sort_values(ascending=False)
    no5=x.copy(); no10=x.copy()
    if len(ordered):
        no5.loc[ordered.index[:min(5,len(ordered))]]=0.0; no10.loc[ordered.index[:min(10,len(ordered))]]=0.0
    entries=pd.to_datetime(executed["entry_time"],errors="coerce").dropna().sort_values(); gaps=entries.diff().dt.total_seconds().div(86400).dropna()
    return {
        "executed_trades": int(len(executed)),
        "trades_per_month": float(len(executed)/months) if months>0 else np.nan,
        "win_rate": float((x>0).mean()) if len(x) else np.nan,
        "trade_pf": _profit_factor(x),
        "mean_equity_return_per_trade": float(x.mean()) if len(x) else np.nan,
        "final_equity_multiple": float(es.iloc[-1]),
        "total_return": float(es.iloc[-1]-1.0),
        "max_drawdown_daily_mtm": float(dd.min()),
        "longest_drawdown_days": int(longest),
        "ulcer_index": float(np.sqrt(np.mean(np.square(dd.clip(upper=0.0))))),
        "log_equity_r2": r2,
        "positive_month_rate": float((month>0).mean()) if len(month) else np.nan,
        "median_month_return": float(month.median()) if len(month) else np.nan,
        "worst_month_return": float(month.min()) if len(month) else np.nan,
        "positive_quarter_rate": float((quarter>0).mean()) if len(quarter) else np.nan,
        "rolling90_positive_rate": float((roll90>0).mean()) if len(roll90) else np.nan,
        "median_holding_hours": float(pd.to_numeric(executed["holding_minutes"],errors="coerce").median()/60.0),
        "max_days_between_entries": float(gaps.max()) if len(gaps) else np.nan,
        "mss_upgrade_rate": float(pd.to_numeric(executed["mss_upgrade_flag"],errors="coerce").mean()),
        "major_upgrade_rate": float(pd.to_numeric(executed["major_upgrade_flag"],errors="coerce").mean()),
        "base_target_hit_rate": float(pd.to_numeric(executed["base_target_hit_flag"],errors="coerce").mean()),
        "final_equity_no_top5": float(np.prod(1.0+no5.to_numpy(dtype=float))),
        "final_equity_no_top10": float(np.prod(1.0+no10.to_numpy(dtype=float))),
    }


def summarize_years(executed: pd.DataFrame) -> pd.DataFrame:
    if executed.empty:
        return pd.DataFrame()
    x=executed.copy(); x["year"]=pd.to_datetime(x["entry_time"],errors="coerce").dt.year
    rows=[]
    groups=["lifecycle_variant","risk_schedule","cost_scale","year"]
    for key,p in x.groupby(groups,dropna=False,sort=True):
        v=pd.to_numeric(p["strategy_equity_return"],errors="coerce").dropna()
        rows.append(dict(zip(groups,key))|{
            "trades":len(p),"resolved":len(v),"win_rate":float((v>0).mean()) if len(v) else np.nan,
            "trade_pf":_profit_factor(v),"mean_equity_return_per_trade":float(v.mean()) if len(v) else np.nan,
            "compounded_return":float(np.prod(1.0+v.to_numpy(dtype=float))-1.0) if len(v) else np.nan,
        })
    return pd.DataFrame(rows)


def r10_causal_audit(base: pd.DataFrame, paths: pd.DataFrame, sized: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if not base.empty:
        rows.append({"check":"base_signal_available_not_after_entry","rows":len(base),"violations":int((pd.to_datetime(base["signal_available_time"],errors="coerce")>pd.to_datetime(base["entry_time"],errors="coerce")).sum())})
        rows.append({"check":"ssl_long_only","rows":len(base),"violations":int((base["liquidity_side"].astype(str)!="SSL").sum())})
    if not paths.empty:
        q=paths.loc[pd.to_numeric(paths["mss_upgrade_flag"],errors="coerce").eq(1)]
        rows.append({"check":"mss_upgrade_not_before_entry","rows":len(q),"violations":int((pd.to_datetime(q["mss_upgrade_time"],errors="coerce")<pd.to_datetime(q["entry_time"],errors="coerce")).sum())})
        q=paths.loc[pd.to_numeric(paths["major_upgrade_flag"],errors="coerce").eq(1)]
        rows.append({"check":"major_upgrade_not_before_entry","rows":len(q),"violations":int((pd.to_datetime(q["major_upgrade_time"],errors="coerce")<pd.to_datetime(q["entry_time"],errors="coerce")).sum())})
    if not sized.empty:
        rows.append({"check":"risk_budget_le_2pct","rows":len(sized),"violations":int(pd.to_numeric(sized["risk_budget_fraction"],errors="coerce").gt(0.02+EPS).sum())})
        rows.append({"check":"worst_case_initial_risk_within_budget","rows":len(sized),"violations":int((pd.to_numeric(sized["worst_case_initial_risk"],errors="coerce")>pd.to_numeric(sized["risk_budget_fraction"],errors="coerce")+1e-10).sum())})
        rows.append({"check":"notional_cap_le_3x","rows":len(sized),"violations":int(pd.to_numeric(sized["notional_multiple"],errors="coerce").gt(3.0+EPS).sum())})
    return pd.DataFrame(rows)
