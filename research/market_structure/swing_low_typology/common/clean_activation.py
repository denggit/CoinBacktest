#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal clean-path targets, activation mapping, and TP1 execution variants.

Research 18 keeps the +1% close-only research target.  This module separates:

* outcome labels (future, model targets only),
* causal activation state (current/older closed bars only), and
* deterministic execution (next-open entry, future closed-bar decisions).

No function uses an eventual Swing Low type or a future region end as a feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from research.market_structure.swing_low_typology.common.broad_reversal_evaluation import (
    CloseTargetCostSpec,
)
from research.market_structure.swing_low_typology.common.deployable_first_sweep_backtest import (
    enforce_single_position,
)

EPS = 1e-12


@dataclass(frozen=True)
class ActivationSpec:
    maximum_wait_bars: int = 10
    minimum_bars_since_low: int = 1
    minimum_reclaim_bp: float = 10.0
    require_positive_close_return: bool = True
    require_close_above_previous: bool = True


@dataclass(frozen=True)
class ExitPolicySpec:
    policy_id: str
    horizon_bars: int
    use_region_low_invalidation: bool = False
    stale_check_bars: int | None = None


FROZEN_EXIT_POLICIES: tuple[ExitPolicySpec, ...] = (
    ExitPolicySpec("TP1_TIME30", 30),
    ExitPolicySpec("TP1_TIME60", 60),
    ExitPolicySpec("TP1_TIME180", 180),
    ExitPolicySpec("TP1_REGION_LOW_INVALIDATION_TIME60", 60, use_region_low_invalidation=True),
    ExitPolicySpec("TP1_STALE30_TIME60", 60, stale_check_bars=30),
    ExitPolicySpec(
        "TP1_REGION_LOW_INVALIDATION_STALE30_TIME60",
        60,
        use_region_low_invalidation=True,
        stale_check_bars=30,
    ),
)


def attach_clean_path_targets(
    frame: pd.DataFrame,
    *,
    horizon: int = 60,
    maximum_mae_before_tp_pct: float = 0.50,
    adverse_mae_pct: float = 0.75,
) -> pd.DataFrame:
    """Attach frozen future path targets; these columns are labels only."""

    out = frame.copy()
    required = {
        f"tp_1_h{int(horizon)}",
        f"time_to_tp_1_h{int(horizon)}",
        f"mae_before_tp_1_h{int(horizon)}_pct",
        f"mae_h{int(horizon)}_pct",
        f"permanent_failure_h{int(horizon)}",
    }
    missing = sorted(required.difference(out.columns))
    if missing:
        raise RuntimeError(f"clean-path targets missing label columns: {missing}")
    tp = out[f"tp_1_h{int(horizon)}"].fillna(False).astype(bool)
    time_to_tp = pd.to_numeric(out[f"time_to_tp_1_h{int(horizon)}"], errors="coerce")
    mae_before = pd.to_numeric(
        out[f"mae_before_tp_1_h{int(horizon)}_pct"], errors="coerce"
    )
    mae = pd.to_numeric(out[f"mae_h{int(horizon)}_pct"], errors="coerce")
    permanent = out[f"permanent_failure_h{int(horizon)}"].fillna(False).astype(bool)
    out["target_clean_tp60"] = (
        tp & mae_before.le(float(maximum_mae_before_tp_pct)).fillna(False)
    ).astype(bool)
    out["target_fast_clean_tp30"] = (
        out["target_clean_tp60"] & time_to_tp.le(30).fillna(False)
    ).astype(bool)
    out["target_adverse_path60"] = (
        mae.ge(float(adverse_mae_pct)).fillna(False) | permanent
    ).astype(bool)
    for column in ("target_clean_tp60", "target_fast_clean_tp30", "target_adverse_path60"):
        if out[column].isna().any() or not pd.api.types.is_bool_dtype(out[column].dtype):
            raise RuntimeError(f"path target {column} must be resolved bool without NA")
    return out


def activation_state_table(
    region_states: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    spec: ActivationSpec | None = None,
) -> pd.DataFrame:
    """Evaluate the frozen activation rule on every causal region snapshot."""

    spec = spec or ActivationSpec()
    required = {
        "event_id",
        "causal_region_id",
        "extreme_pos",
        "extreme_time",
        "causal_region_start_pos",
        "region_bars_since_low",
        "region_close_above_previous",
        "region_rebound_from_low",
    }
    missing = sorted(required.difference(region_states.columns))
    if missing:
        raise RuntimeError(f"activation states missing columns: {missing}")
    ordered = region_states.sort_values(
        ["causal_region_id", "extreme_pos", "event_id"], kind="mergesort"
    ).reset_index(drop=True).copy()
    pos = pd.to_numeric(ordered["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    if pos.size == 0:
        return ordered.assign(
            activation_current_return=pd.Series(dtype="float32"),
            activation_running_region_low=pd.Series(dtype="float64"),
            activation_rule_passed=pd.Series(dtype=bool),
            activation_rule_id=pd.Series(dtype="string"),
        )
    if int(pos.min()) < 1 or int(pos.max()) >= len(close):
        raise RuntimeError("activation state positions outside loaded bars or lack previous bar")
    current_return = close[pos] / np.maximum(close[pos - 1], EPS) - 1.0
    rebound = pd.to_numeric(ordered["region_rebound_from_low"], errors="coerce").to_numpy(dtype=float)
    running_low = np.divide(
        close[pos],
        1.0 + rebound,
        out=np.full(len(ordered), np.nan, dtype=float),
        where=np.isfinite(rebound) & ((1.0 + rebound) > EPS),
    )
    bars_since_low = pd.to_numeric(ordered["region_bars_since_low"], errors="coerce").to_numpy(dtype=float)
    above_previous = pd.to_numeric(
        ordered["region_close_above_previous"], errors="coerce"
    ).to_numpy(dtype=float)
    reclaim = rebound >= float(spec.minimum_reclaim_bp) / 10_000.0
    eligible = np.isfinite(running_low) & (bars_since_low >= int(spec.minimum_bars_since_low)) & reclaim
    if spec.require_positive_close_return:
        eligible &= current_return > 0.0
    if spec.require_close_above_previous:
        eligible &= above_previous >= 0.5
    ordered["activation_current_return"] = current_return.astype(np.float32)
    ordered["activation_running_region_low"] = running_low
    ordered["activation_rule_passed"] = eligible.astype(bool)
    ordered["activation_rule_id"] = "STOP_NEW_LOW_POSITIVE_RESPONSE_RECLAIM10BP"
    return ordered


def build_activation_map(
    armed_events: pd.DataFrame,
    states: pd.DataFrame,
    *,
    maximum_wait_bars: int = 10,
) -> pd.DataFrame:
    """Map each armed event to the first later causal activation in its region.

    The implementation is vectorized with a forward ``merge_asof``.  It is
    equivalent to a per-event search but avoids tens of thousands of Python
    row iterations on the full ETH history.
    """

    required_armed = {"event_id", "causal_region_id", "extreme_pos", "extreme_time"}
    required_states = {
        "event_id",
        "causal_region_id",
        "extreme_pos",
        "extreme_time",
        "activation_rule_passed",
        "activation_running_region_low",
        "activation_rule_id",
    }
    missing = sorted(required_armed.difference(armed_events.columns))
    if missing:
        raise RuntimeError(f"armed events missing columns: {missing}")
    missing = sorted(required_states.difference(states.columns))
    if missing:
        raise RuntimeError(f"activation map states missing columns: {missing}")
    if int(maximum_wait_bars) < 0:
        raise ValueError("maximum_wait_bars must be >= 0")

    left = armed_events[
        ["event_id", "causal_region_id", "extreme_pos", "extreme_time"]
    ].copy()
    left = left.rename(
        columns={
            "event_id": "armed_event_id",
            "extreme_pos": "armed_extreme_pos",
            "extreme_time": "armed_extreme_time",
        }
    )
    left["causal_region_id"] = left["causal_region_id"].astype(str)
    left["armed_event_id"] = left["armed_event_id"].astype(str)
    left["armed_extreme_pos"] = pd.to_numeric(
        left["armed_extreme_pos"], errors="raise"
    ).astype(np.int64)
    left["armed_extreme_time"] = pd.to_datetime(
        left["armed_extreme_time"], errors="raise"
    )
    if left["armed_event_id"].duplicated().any():
        raise RuntimeError("armed event_id must be unique")
    left["_original_order"] = np.arange(len(left), dtype=np.int64)

    right = states[states["activation_rule_passed"].fillna(False).astype(bool)][
        [
            "event_id",
            "causal_region_id",
            "extreme_pos",
            "extreme_time",
            "activation_running_region_low",
            "activation_rule_id",
        ]
    ].copy()
    right = right.rename(
        columns={
            "event_id": "activation_event_id",
            "extreme_pos": "activation_signal_pos",
            "extreme_time": "activation_signal_time",
            "activation_running_region_low": "activation_region_low",
        }
    )
    right["causal_region_id"] = right["causal_region_id"].astype(str)
    right["activation_signal_pos"] = pd.to_numeric(
        right["activation_signal_pos"], errors="raise"
    ).astype(np.int64)
    right["activation_signal_time"] = pd.to_datetime(
        right["activation_signal_time"], errors="raise"
    )
    # Deterministic tie-break for multiple snapshots at the same region/position.
    right = (
        right.sort_values(
            ["causal_region_id", "activation_signal_pos", "activation_event_id"],
            kind="mergesort",
        )
        .drop_duplicates(["causal_region_id", "activation_signal_pos"], keep="first")
    )

    if left.empty:
        result = left.drop(columns=["_original_order"]).copy()
        result["activation_found"] = pd.Series(dtype=bool)
        result["activation_wait_bars"] = pd.Series(dtype="float64")
        for column, dtype in (
            ("activation_event_id", "string"),
            ("activation_signal_pos", "float64"),
            ("activation_signal_time", "datetime64[ns]"),
            ("activation_region_low", "float64"),
            ("activation_rule_id", "string"),
        ):
            result[column] = pd.Series(dtype=dtype)
        return result

    if right.empty:
        merged = left.copy()
        merged["activation_event_id"] = ""
        merged["activation_signal_pos"] = np.nan
        merged["activation_signal_time"] = pd.NaT
        merged["activation_region_low"] = np.nan
        merged["activation_rule_id"] = "STOP_NEW_LOW_POSITIVE_RESPONSE_RECLAIM10BP"
    else:
        # merge_asof requires the distance key to be globally monotonic.  The
        # ``by`` column still prevents matches across causal regions.
        left_sorted = left.sort_values(
            ["armed_extreme_pos", "causal_region_id", "armed_event_id"],
            kind="mergesort",
        )
        right_sorted = right.sort_values(
            ["activation_signal_pos", "causal_region_id", "activation_event_id"],
            kind="mergesort",
        )
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on="armed_extreme_pos",
            right_on="activation_signal_pos",
            by="causal_region_id",
            direction="forward",
            allow_exact_matches=True,
        )
    wait = pd.to_numeric(merged["activation_signal_pos"], errors="coerce") - pd.to_numeric(
        merged["armed_extreme_pos"], errors="raise"
    )
    found = wait.between(0, int(maximum_wait_bars), inclusive="both")
    merged["activation_found"] = found.fillna(False).astype(bool)
    merged["activation_wait_bars"] = wait.where(found)
    for column, missing_value in (
        ("activation_event_id", ""),
        ("activation_signal_pos", np.nan),
        ("activation_signal_time", pd.NaT),
        ("activation_region_low", np.nan),
    ):
        merged.loc[~found, column] = missing_value
    merged.loc[~found, "activation_rule_id"] = (
        "STOP_NEW_LOW_POSITIVE_RESPONSE_RECLAIM10BP"
    )
    result = (
        merged.sort_values("_original_order", kind="mergesort")
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )
    if result["armed_event_id"].duplicated().any():
        raise RuntimeError("activation map produced duplicate armed_event_id")
    found = result["activation_found"].astype(bool)
    if found.any() and not (
        pd.to_datetime(result.loc[found, "activation_signal_time"], errors="raise")
        >= pd.to_datetime(result.loc[found, "armed_extreme_time"], errors="raise")
    ).all():
        raise RuntimeError("activation occurred before arm time")
    return result


def first_policy_event_per_region(selected: pd.DataFrame) -> pd.DataFrame:
    """Keep the first rank-qualified causal event per region."""

    if selected.empty:
        return selected.copy()
    required = {"causal_region_id", "extreme_pos", "event_id"}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise RuntimeError(f"region dedup missing columns: {missing}")
    return (
        selected.sort_values(
            ["causal_region_id", "extreme_pos", "event_id"], kind="mergesort"
        )
        .groupby("causal_region_id", sort=False)
        .head(1)
        .sort_values(["extreme_pos", "event_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def materialize_activation_events(
    selected_armed: pd.DataFrame,
    activation_map: pd.DataFrame,
) -> pd.DataFrame:
    """Convert found activations into normal event rows for label/execution code."""

    if selected_armed.empty:
        # Preserve a stable schema so downstream label builders and joins can
        # handle folds with no causal activation instead of failing on a
        # missing event_id column.
        empty = selected_armed.copy()
        for column, dtype in (
            ("base_event_id", "string"),
            ("signal_region_low", "float64"),
            ("entry_mode", "string"),
        ):
            if column not in empty.columns:
                empty[column] = pd.Series(index=empty.index, dtype=dtype)
        return empty
    merged = selected_armed.merge(
        activation_map,
        left_on="event_id",
        right_on="armed_event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_activation"),
    )
    merged = merged[merged["activation_found"].fillna(False).astype(bool)].copy()
    if merged.empty:
        return merged
    merged["base_event_id"] = merged["event_id"].astype(str)
    merged["event_id"] = (
        "ACT_" + merged["base_event_id"].astype(str) + "_" + merged["activation_signal_pos"].astype(int).astype(str)
    )
    merged["extreme_pos"] = pd.to_numeric(merged["activation_signal_pos"], errors="raise").astype(np.int64)
    merged["extreme_time"] = pd.to_datetime(merged["activation_signal_time"], errors="raise")
    merged["signal_region_low"] = pd.to_numeric(merged["activation_region_low"], errors="coerce")
    merged["entry_mode"] = "CAUSAL_ACTIVATION_WAIT10"
    if merged["event_id"].duplicated().any():
        raise RuntimeError("activation materialization produced duplicate event_id")
    return merged.reset_index(drop=True)


def attach_signal_region_low(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Attach the running region low available at each base signal."""

    out = events.copy()
    required = {"extreme_pos", "region_rebound_from_low"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise RuntimeError(f"base signal region low missing columns: {missing}")
    pos = pd.to_numeric(out["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    rebound = pd.to_numeric(out["region_rebound_from_low"], errors="coerce").to_numpy(dtype=float)
    low = np.divide(
        close[pos],
        1.0 + rebound,
        out=np.full(len(out), np.nan, dtype=float),
        where=np.isfinite(rebound) & ((1.0 + rebound) > EPS),
    )
    out["signal_region_low"] = low
    out["entry_mode"] = "BASE_NEXT_OPEN"
    return out


def replay_tp1_exit_policy(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    policy: ExitPolicySpec,
    costs: CloseTargetCostSpec | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    """Replay frozen TP1 exits using only future closed-bar decisions."""

    if events.empty:
        return pd.DataFrame()
    costs = costs or CloseTargetCostSpec()
    horizon = int(policy.horizon_bars)
    if horizon < 1:
        raise ValueError("exit horizon must be positive")
    frame = bars
    index = pd.DatetimeIndex(frame.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise RuntimeError(
            "TP1 exit replay requires unique, monotonic bars because event positions are positional"
        )
    open_ = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    positions = pd.to_numeric(events["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if np.any((positions < 0) | (positions >= len(frame))):
        raise RuntimeError("TP1 exit replay received event positions outside loaded bars")
    delay = int(entry_delay_bars)
    if delay < 0:
        raise ValueError("entry_delay_bars must be >= 0")
    entry_pos = positions + 1 + delay
    # Pure TP/time exits only need ``horizon`` future closes.  Policies with a
    # confirmed early decision additionally need the next bar open even when
    # the decision occurs on the final close in the horizon.
    needs_confirmed_next_open = bool(
        policy.use_region_low_invalidation or policy.stale_check_bars is not None
    )
    final_required_offset = horizon if needs_confirmed_next_open else horizon - 1
    valid = (entry_pos >= 0) & (entry_pos + final_required_offset < len(frame))
    source = events.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    positions = positions[valid]
    entry_pos = entry_pos[valid]
    if source.empty:
        return pd.DataFrame()
    raw_entry = open_[entry_pos]
    finite_entry = np.isfinite(raw_entry) & (raw_entry > EPS)
    source = source.iloc[np.flatnonzero(finite_entry)].reset_index(drop=True)
    positions = positions[finite_entry]
    entry_pos = entry_pos[finite_entry]
    raw_entry = raw_entry[finite_entry]
    if source.empty:
        return pd.DataFrame()

    windows = np.lib.stride_tricks.sliding_window_view(close, horizon)
    path = windows[entry_pos]
    finite_path = np.isfinite(path).all(axis=1)
    source = source.iloc[np.flatnonzero(finite_path)].reset_index(drop=True)
    positions = positions[finite_path]
    entry_pos = entry_pos[finite_path]
    raw_entry = raw_entry[finite_path]
    path = path[finite_path]
    if source.empty:
        return pd.DataFrame()

    n = len(source)
    tp_mask = path >= raw_entry[:, None] * 1.01
    tp_hit = tp_mask.any(axis=1)
    tp_first = np.where(tp_hit, np.argmax(tp_mask, axis=1), horizon + 1).astype(np.int64)

    invalid_first = np.full(n, horizon + 1, dtype=np.int64)
    if policy.use_region_low_invalidation:
        if "signal_region_low" not in source.columns:
            raise RuntimeError(
                f"exit policy {policy.policy_id} requires signal_region_low"
            )
        region_low = pd.to_numeric(
            source["signal_region_low"], errors="coerce"
        ).to_numpy(dtype=float)
        invalid_mask = np.isfinite(region_low)[:, None] & (path <= region_low[:, None])
        invalid_hit = invalid_mask.any(axis=1)
        invalid_first = np.where(invalid_hit, np.argmax(invalid_mask, axis=1), horizon + 1).astype(np.int64)

    stale_first = np.full(n, horizon + 1, dtype=np.int64)
    if policy.stale_check_bars is not None:
        check = int(policy.stale_check_bars)
        if not 1 <= check < horizon:
            raise ValueError("stale_check_bars must be within horizon")
        no_tp_by_check = tp_first >= check
        stale = no_tp_by_check & (path[:, check - 1] <= raw_entry)
        stale_first = np.where(stale, check - 1, horizon + 1).astype(np.int64)

    decision_first = np.minimum(invalid_first, stale_first)
    decision_hit = decision_first < horizon
    # Sentinels for "no TP" and "no early decision" are deliberately equal.
    # Never compare the sentinel positions without their corresponding hit
    # masks, otherwise a no-hit trade is incorrectly classified as a TP.
    tp_wins = tp_hit & ((~decision_hit) | (tp_first <= decision_first))
    early_decision = decision_hit & ((~tp_hit) | (decision_first < tp_first))
    time_exit = ~(tp_wins | early_decision)
    classification_count = (
        tp_wins.astype(np.int8)
        + early_decision.astype(np.int8)
        + time_exit.astype(np.int8)
    )
    if not np.all(classification_count == 1):
        raise RuntimeError("TP1 exit replay did not assign exactly one exit path")

    exit_pos = np.empty(n, dtype=np.int64)
    raw_exit = np.empty(n, dtype=float)
    reason = np.empty(n, dtype=object)
    # TP is executed at the frozen +1% target on the triggering closed bar.
    exit_pos[tp_wins] = entry_pos[tp_wins] + tp_first[tp_wins]
    raw_exit[tp_wins] = raw_entry[tp_wins] * 1.01
    reason[tp_wins] = "take_profit_on_closed_bar"
    # Confirmed invalidation/stale decisions execute at the next bar open.
    if early_decision.any():
        trigger = decision_first[early_decision]
        confirmed_exit_pos = entry_pos[early_decision] + trigger + 1
        if np.any((confirmed_exit_pos < 0) | (confirmed_exit_pos >= len(frame))):
            raise RuntimeError("confirmed TP1 early exit position is outside loaded bars")
        exit_pos[early_decision] = confirmed_exit_pos
        raw_exit[early_decision] = open_[confirmed_exit_pos]
        invalid_wins = invalid_first[early_decision] <= stale_first[early_decision]
        early_reason = np.full(
            int(early_decision.sum()),
            "region_low_invalidation_next_open",
            dtype=object,
        )
        stale_wins = ~invalid_wins
        if stale_wins.any():
            if policy.stale_check_bars is None:
                raise RuntimeError("stale exit selected without a configured stale check")
            early_reason[stale_wins] = (
                f"stale{int(policy.stale_check_bars)}_next_open"
            )
        reason[early_decision] = early_reason
    if time_exit.any():
        exit_pos[time_exit] = entry_pos[time_exit] + horizon - 1
        raw_exit[time_exit] = path[time_exit, horizon - 1]
        reason[time_exit] = f"time_h{horizon}"

    if np.any((exit_pos < entry_pos) | (exit_pos >= len(frame))):
        raise RuntimeError("TP1 exit replay produced an out-of-bounds exit position")
    if not np.isfinite(raw_exit).all():
        raise RuntimeError("TP1 exit replay produced a non-finite raw exit price")

    multiplier = float(costs.cost_multiplier)
    entry_exec = raw_entry * (1.0 + float(costs.entry_slippage_pct) * multiplier)
    exit_exec = raw_exit * (1.0 - float(costs.exit_slippage_pct) * multiplier)
    gross = raw_exit / raw_entry - 1.0
    net = exit_exec / entry_exec - 1.0 - (
        float(costs.entry_fee_rate) + float(costs.exit_fee_rate)
    ) * multiplier
    path_return = path / raw_entry[:, None] - 1.0
    observed_last = np.full(n, horizon - 1, dtype=np.int64)
    observed_last[tp_wins] = tp_first[tp_wins]
    observed_last[early_decision] = decision_first[early_decision]
    if np.any((observed_last < 0) | (observed_last >= horizon)):
        raise RuntimeError("TP1 exit replay produced an invalid observed path boundary")
    cumulative_mae = np.minimum.accumulate(path_return, axis=1)
    cumulative_mfe = np.maximum.accumulate(path_return, axis=1)
    row_index = np.arange(n, dtype=np.int64)
    realized_mae = cumulative_mae[row_index, observed_last]
    realized_mfe = cumulative_mfe[row_index, observed_last]
    # Confirmed exits execute on the next open, so include any opening gap in
    # the realized excursion metrics.  Post-exit closes are intentionally
    # excluded.
    raw_exit_return = raw_exit / raw_entry - 1.0
    realized_mae = np.minimum(realized_mae, raw_exit_return)
    realized_mfe = np.maximum(realized_mfe, raw_exit_return)
    score = pd.to_numeric(
        source.get("opportunity_score", pd.Series(np.nan, index=source.index)), errors="coerce"
    )
    output = pd.DataFrame(
        {
            "event_id": source["event_id"].astype(str).to_numpy(),
            "signal_time": index[positions],
            "entry_time": index[entry_pos],
            "exit_time": index[exit_pos],
            "signal_pos": positions,
            "entry_pos": entry_pos,
            "exit_pos": exit_pos,
            "entry_price": entry_exec,
            "raw_entry_price": raw_entry,
            "exit_price": exit_exec,
            "raw_exit_price": raw_exit,
            "exit_reason": reason,
            "bars_held": exit_pos - entry_pos + 1,
            "tp_hit": tp_wins,
            "stop_hit": np.zeros(n, dtype=bool),
            "same_bar_stop_tp_both_hit_flag": np.zeros(n, dtype=bool),
            "add_on_filled": np.zeros(n, dtype=bool),
            "gross_return": gross,
            "net_return": net,
            "mae": realized_mae,
            "mfe": realized_mfe,
            "opportunity_score": score.to_numpy(dtype=float),
            "exit_policy_id": policy.policy_id,
            "entry_delay_bars": int(entry_delay_bars),
            "cost_multiplier": multiplier,
        }
    )
    for column in (
        "fold",
        "expert_id",
        "policy_id",
        "top_pct",
        "entry_mode",
        "causal_region_id",
        "base_event_id",
        "signal_region_low",
    ):
        if column in source.columns:
            output[column] = source[column].to_numpy()
    if not (pd.to_datetime(output["entry_time"]) > pd.to_datetime(output["signal_time"])).all():
        raise RuntimeError("TP1 exit replay violated signal -> next open")
    confirmed = output["exit_reason"].astype(str).str.endswith("_next_open")
    if confirmed.any() and not (
        pd.to_datetime(output.loc[confirmed, "exit_time"])
        > pd.to_datetime(output.loc[confirmed, "entry_time"])
    ).all():
        raise RuntimeError("confirmed early exit did not occur after entry")
    return output


def executable_tp1_policy(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    policy: ExitPolicySpec,
    costs: CloseTargetCostSpec,
    entry_delay_bars: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    isolated = replay_tp1_exit_policy(
        bars,
        events,
        policy=policy,
        costs=costs,
        entry_delay_bars=entry_delay_bars,
    )
    portfolio, skipped = enforce_single_position(isolated)
    return portfolio, {
        "raw_signals": int(len(events)),
        "deduplicated_signals": 0,
        "skipped_overlap": int(skipped),
    }
