#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Executable R14 policy primitives for SOXL ICT research.

R14 is deliberately much narrower than R13.  R13 is a semantic atlas; R14
turns a small set of predeclared interpretations into one causal setup per
physical liquidity sweep and then one account lifecycle at a time.

Important boundaries:
* no 5m execution entries (5m remains context only in R14);
* no Swing +/- absolute-dollar entry gate;
* no requirement that the opposite target be EQL/equal-like;
* target state routes execution, but source-liquidity state never gates entry;
* no later MSS/re-entry from the same physical sweep after a setup is chosen;
* cross-timeframe arbitration is earliest-causal-signal only.  It never chooses
  a later 1m/2m setup using future knowledge that the later setup will exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, compound_account, summarize_variant
from .semantic_consolidation import consolidate_fvg_entry_choices


VISIBLE_TIERS = frozenset({"visible_p50_p80", "strong_ge_p80"})

@dataclass(frozen=True)
class ExecutableLeg:
    execution_tf: str
    target_states: tuple[str, ...]
    entry_models: tuple[str, ...]


@dataclass(frozen=True)
class ExecutablePolicy:
    policy_id: str
    legs: tuple[ExecutableLeg, ...]


DEFAULT_POLICIES: tuple[ExecutablePolicy, ...] = (
    # The narrow R14 core is copied from mechanisms that were positive across
    # discovery/2025/2026 in R13.  This is a candidate freeze, not untouched
    # OOS proof.  Equal-like is NOT mandatory: the 2m leg explicitly trades
    # partial-consumed targets.
    ExecutablePolicy(
        "core_break_middle",
        (
            ExecutableLeg("1m", ("shallow_probe_equal_like",), ("break_middle_near",)),
            ExecutableLeg("2m", ("partial_consumed",), ("break_middle_ce",)),
        ),
    ),
    ExecutablePolicy(
        "core_first_train_1m",
        (
            ExecutableLeg("1m", ("shallow_probe_equal_like",), ("first_train_near",)),
            ExecutableLeg("2m", ("partial_consumed",), ("break_middle_ce",)),
        ),
    ),
    ExecutablePolicy(
        "shallow_1m_break_middle",
        (ExecutableLeg("1m", ("shallow_probe_equal_like",), ("break_middle_near",)),),
    ),
    ExecutablePolicy(
        "partial_2m_break_middle_ce",
        (ExecutableLeg("2m", ("partial_consumed",), ("break_middle_ce",)),),
    ),
)


def _narrative_key_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "event_id", "execution_tf", "break_available_time",
        "mss_reference_time", "mss_reference_price",
    ]
    return [c for c in candidates if c in frame.columns]


def build_fixed_fvg_entry_catalog(
    primary_narratives: pd.DataFrame,
    fvgs_with_state: pd.DataFrame,
) -> pd.DataFrame:
    """Return R13's predeclared FVG execution choices for R14 selection.

    R14 does not rank these by PnL.  Each policy leg names exactly which entry
    model it is allowed to use.
    """
    if primary_narratives.empty or fvgs_with_state.empty:
        return pd.DataFrame()
    out = consolidate_fvg_entry_choices(primary_narratives, fvgs_with_state)
    if out.empty:
        return out
    out = out.loc[out["execution_tf"].astype(str).isin({"1m", "2m"})].copy()
    out["entry_model_r14"] = out["entry_model_r13"].astype(str)
    return out.reset_index(drop=True)


def select_policy_leg_entries(catalog: pd.DataFrame, policy: ExecutablePolicy) -> pd.DataFrame:
    """Select causal state/TF/entry legs declared by one executable policy."""
    if catalog.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for leg_rank, leg in enumerate(policy.legs):
        q = catalog.loc[
            catalog["execution_tf"].astype(str).eq(leg.execution_tf)
            & catalog["target_liquidity_state"].astype(str).isin(leg.target_states)
            & catalog["entry_model_r13"].astype(str).isin(leg.entry_models)
            & catalog["structure_visibility_tier_r13"].astype(str).isin(VISIBLE_TIERS)
        ].copy()
        if q.empty:
            continue
        model_rank = {m: i for i, m in enumerate(leg.entry_models)}
        q["_leg_rank_r14"] = int(leg_rank)
        q["_entry_model_rank_r14"] = q["entry_model_r13"].astype(str).map(model_rank).fillna(999).astype(int)
        key = _narrative_key_columns(q)
        q = q.sort_values(key + ["_entry_model_rank_r14", "entry_available_time"], kind="mergesort")
        q = q.drop_duplicates(key, keep="first")
        parts.append(q)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.reset_index(drop=True)


def _visibility_rank(value: object) -> int:
    text = str(value)
    return {"strong_ge_p80": 0, "visible_p50_p80": 1}.get(text, 9)


def select_one_setup_per_sweep(
    routed_entries: pd.DataFrame,
    *,
    policy: ExecutablePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causally choose one setup per physical liquidity sweep.

    The first eligible policy-leg signal wins.  Ties prefer clearer structure,
    then the earlier-declared policy leg.  There is no re-entry from the same
    sweep in R14.
    """
    if routed_entries.empty:
        return routed_entries.copy(), pd.DataFrame()
    q = routed_entries.copy()
    q["_signal_r14"] = pd.to_datetime(q["entry_available_time"], errors="coerce", utc=True)
    q["_vis_rank_r14"] = q["structure_visibility_tier_r13"].map(_visibility_rank).astype(int)
    if "_leg_rank_r14" not in q:
        q["_leg_rank_r14"] = 0
    q = q.sort_values(
        ["event_id", "_signal_r14", "_vis_rank_r14", "_leg_rank_r14", "break_available_time", "mss_reference_price"],
        kind="mergesort",
    )
    chosen = q.drop_duplicates("event_id", keep="first").copy()
    chosen["policy_id_r14"] = policy.policy_id
    chosen["signal_time"] = chosen["entry_available_time"]
    chosen["attempt_id"] = [f"R14|{policy.policy_id}|{i:07d}" for i in range(len(chosen))]
    chosen["one_setup_per_physical_sweep_r14"] = True

    chosen_idx = set(chosen.index)
    rejected = q.loc[[i for i in q.index if i not in chosen_idx]].copy()
    if not rejected.empty:
        rejected["policy_id_r14"] = policy.policy_id
        rejected["r14_rejection_reason"] = "later_or_duplicate_setup_same_physical_sweep"
    drop = ["_signal_r14", "_vis_rank_r14", "_entry_model_rank_r14", "_leg_rank_r14"]
    return chosen.drop(columns=drop, errors="ignore").reset_index(drop=True), rejected.drop(columns=drop, errors="ignore").reset_index(drop=True)


def build_policy_attempts(
    primary_narratives: pd.DataFrame,
    fvgs_with_state: pd.DataFrame,
    *,
    policies: Sequence[ExecutablePolicy] = DEFAULT_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build narrow, state-specific executable attempts for R14.

    R14 intentionally does not route fresh/deep targets into this first profit
    core.  They remain valid ICT research contexts in R13 and are deferred to a
    later management/expansion study.  The first R14 goal is to test whether
    the already-positive shallow/partial mechanisms survive one-setup-per-sweep
    and one-account execution.
    """
    if primary_narratives.empty or fvgs_with_state.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    catalog = build_fixed_fvg_entry_catalog(primary_narratives, fvgs_with_state)
    all_attempts: list[pd.DataFrame] = []
    all_rejections: list[pd.DataFrame] = []
    for policy in policies:
        selected = select_policy_leg_entries(catalog, policy)
        if selected.empty:
            continue
        # Core legs always aim at the already-raided-but-not-deep-consumed
        # external liquidity.  This includes partial consumption and therefore
        # does not impose an EQL/equal-like requirement.
        selected = selected.copy()
        selected["entry_price"] = pd.to_numeric(selected["entry_price"], errors="coerce")
        selected["target_price_r14"] = pd.to_numeric(selected["target_price"], errors="coerce")
        selected["target_model_r14"] = "external_raided_not_fully_consumed"
        selected["target_router_r14"] = "profit_core_state_specific"
        selected["target_router_note_r14"] = ""
        is_long = selected["trade_side"].astype(str).eq("LONG")
        stop = pd.to_numeric(selected["stop_price"], errors="coerce")
        entry = pd.to_numeric(selected["entry_price"], errors="coerce")
        target = pd.to_numeric(selected["target_price_r14"], errors="coerce")
        risk = pd.Series(np.where(is_long, entry-stop, stop-entry), index=selected.index, dtype=float)
        reward = pd.Series(np.where(is_long, target-entry, entry-target), index=selected.index, dtype=float)
        valid = risk.gt(EPS) & reward.gt(EPS) & np.isfinite(entry.to_numpy(float, na_value=np.nan)) & np.isfinite(target.to_numpy(float, na_value=np.nan))
        invalid = selected.loc[~valid].copy()
        if not invalid.empty:
            invalid["policy_id_r14"] = policy.policy_id
            invalid["r14_rejection_reason"] = "invalid_risk_reward"
            all_rejections.append(invalid)
        selected = selected.loc[valid].copy()
        if selected.empty:
            continue
        selected["risk_abs_r14"] = risk.loc[selected.index].to_numpy(float)
        selected["planned_reward_abs_r14"] = reward.loc[selected.index].to_numpy(float)
        selected["planned_rr_r14"] = (reward.loc[selected.index] / risk.loc[selected.index]).to_numpy(float)
        selected["fvg_near_edge_entry"] = entry.loc[selected.index].to_numpy(float)
        selected["target_price"] = target.loc[selected.index].to_numpy(float)
        selected["planned_rr"] = selected["planned_rr_r14"].to_numpy(float)
        chosen, duplicate_rejects = select_one_setup_per_sweep(selected, policy=policy)
        if not chosen.empty:
            all_attempts.append(chosen)
        if not duplicate_rejects.empty:
            all_rejections.append(duplicate_rejects)
    attempts = pd.concat(all_attempts, ignore_index=True, sort=False) if all_attempts else pd.DataFrame()
    rejects = pd.concat(all_rejections, ignore_index=True, sort=False) if all_rejections else pd.DataFrame()
    return attempts, rejects, catalog


def period_label(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    if ts < pd.Timestamp("2025-01-01"):
        return "discovery_2023h2_2024"
    if ts < pd.Timestamp("2026-01-01"):
        return "forward_2025"
    return "late_2026"


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    gains = float(x[x > 0].sum()); losses = float(-x[x < 0].sum())
    return gains / losses if losses > EPS else (np.inf if gains > EPS else np.nan)


def summarize_lifecycle(lifecycle: pd.DataFrame, *, initial_capital: float = 10_000.0) -> dict[str, object]:
    """Project-level summary with trade expectancy and account metrics."""
    base = summarize_variant(lifecycle, skipped_overlap=0, initial_capital=initial_capital)
    if lifecycle.empty:
        return base
    filled = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].sort_values("fill_time", kind="mergesort")
    if filled.empty:
        return base
    x = pd.to_numeric(filled["net_return"], errors="coerce").dropna()
    wins = x[x > 0]; losses = x[x < 0]
    base.update({
        "expectancy_pct_per_trade": float(x.mean()),
        "avg_win_pct": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_pct": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio_net_return": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "median_mfe_r": float(pd.to_numeric(filled["mfe_r"], errors="coerce").median()),
        "median_mae_r": float(pd.to_numeric(filled["mae_r"], errors="coerce").median()),
    })
    return base


def account_curve(lifecycle: pd.DataFrame, *, initial_capital: float = 10_000.0) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    filled = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].copy()
    return compound_account(filled, initial_capital=initial_capital)


def monthly_table(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    f = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].copy()
    if f.empty:
        return pd.DataFrame()
    f["fill_time"] = pd.to_datetime(f["fill_time"])
    f["month"] = f["fill_time"].dt.strftime("%Y-%m")
    rows = []
    for month, g in f.groupby("month", sort=True):
        ar = pd.to_numeric(g["account_return"], errors="coerce").fillna(0.0)
        net = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        rows.append({
            "month": month,
            "trades": int(len(g)),
            "account_return": float(np.prod(1.0 + ar.to_numpy(float)) - 1.0),
            "profit_factor": _profit_factor(net),
            "win_rate": float((net > 0).mean()) if len(net) else np.nan,
            "mean_net_return": float(net.mean()) if len(net) else np.nan,
        })
    return pd.DataFrame(rows)


def annual_table(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    f = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].copy()
    if f.empty:
        return pd.DataFrame()
    f["fill_time"] = pd.to_datetime(f["fill_time"])
    f["year"] = f["fill_time"].dt.year
    rows = []
    for year, g in f.groupby("year", sort=True):
        ar = pd.to_numeric(g["account_return"], errors="coerce").fillna(0.0)
        net = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        rows.append({
            "year": int(year), "trades": int(len(g)),
            "account_return": float(np.prod(1.0 + ar.to_numpy(float)) - 1.0),
            "profit_factor": _profit_factor(net),
            "win_rate": float((net > 0).mean()) if len(net) else np.nan,
            "mean_net_return": float(net.mean()) if len(net) else np.nan,
        })
    return pd.DataFrame(rows)


def contribution_table(lifecycle: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    f = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].copy()
    if f.empty:
        return pd.DataFrame()
    rows = []
    for key, g in f.groupby(list(group_cols), dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        net = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        rows.append({
            **dict(zip(group_cols, key)),
            "trades": int(len(g)),
            "win_rate": float((net > 0).mean()) if len(net) else np.nan,
            "profit_factor": _profit_factor(net),
            "mean_net_return": float(net.mean()) if len(net) else np.nan,
            "median_mfe_r": float(pd.to_numeric(g["mfe_r"], errors="coerce").median()),
            "median_mae_r": float(pd.to_numeric(g["mae_r"], errors="coerce").median()),
        })
    return pd.DataFrame(rows)


def opportunity_metrics(
    attempts: pd.DataFrame,
    lifecycle: pd.DataFrame,
    valid_sessions: Sequence[object],
) -> dict[str, object]:
    n_sessions = max(1, len(valid_sessions))
    setup_days = set(attempts["ny_date"].astype(str)) if not attempts.empty else set()
    filled = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)] if not lifecycle.empty else pd.DataFrame()
    trade_days = set(filled["ny_date"].astype(str)) if not filled.empty else set()
    ordered = [str(pd.Timestamp(x).date()) for x in valid_sessions]

    def longest_gap(active: set[str]) -> int:
        best = cur = 0
        for d in ordered:
            if d in active:
                cur = 0
            else:
                cur += 1; best = max(best, cur)
        return best

    return {
        "valid_sessions": int(len(valid_sessions)),
        "selected_setups": int(len(attempts)),
        "setups_per_session": float(len(attempts) / n_sessions),
        "setup_active_days": int(len(setup_days)),
        "setup_active_day_rate": float(len(setup_days) / n_sessions),
        "filled_trades": int(len(filled)),
        "filled_trades_per_session": float(len(filled) / n_sessions),
        "trade_active_days": int(len(trade_days)),
        "trade_active_day_rate": float(len(trade_days) / n_sessions),
        "longest_no_setup_sessions": int(longest_gap(setup_days)),
        "longest_no_trade_sessions": int(longest_gap(trade_days)),
    }
