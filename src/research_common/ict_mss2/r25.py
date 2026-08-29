#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R25 fixed r0020 directional-run exhaustion helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars

EPS = 1e-12
RANGE_REQUIRED = (
    "bar_id", "start_ts", "end_ts", "duration_seconds", "open", "high",
    "low", "close", "direction",
)


@dataclass(frozen=True)
class R25Config:
    range_pct: float = 0.0020
    min_run_bars: int = 4
    entry_delays_minutes: tuple[int, ...] = (0, 1)
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01")

    def validate(self) -> "R25Config":
        if self.range_pct != 0.0020 or self.min_run_bars != 4:
            raise ValueError("R25 scale and run length are frozen")
        if self.entry_delays_minutes != (0, 1):
            raise ValueError("R25 allows only primary and one-minute delay stress")
        if self.market_roundtrip_cost <= 0 or self.cost_scales != (1.0, 2.0, 3.0):
            raise ValueError("invalid R25 cost contract")
        if not (self.discovery_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("invalid R25 research splits")
        return self


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def normalize_range_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize loader output while retaining explicit source-valid flags."""
    missing = sorted(set(RANGE_REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"range bars missing required columns: {missing}")
    out = frame.reset_index(drop=True).copy()
    out["start_ts"] = pd.to_datetime(out["start_ts"], errors="coerce")
    out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
    for name in ("duration_seconds", "open", "high", "low", "close", "direction"):
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce")
    out = out.sort_values(["end_ts", "bar_id"], kind="mergesort", na_position="last").reset_index(drop=True)
    prices = out[["open", "high", "low", "close"]].to_numpy(dtype=float)
    finite_prices = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    duplicate_id = out["bar_id"].duplicated(keep=False)
    out["source_valid"] = (
        out["bar_id"].notna()
        & out["start_ts"].notna()
        & out["end_ts"].notna()
        & out["start_ts"].le(out["end_ts"])
        & out["direction"].isin([-1.0, 1.0])
        & finite_prices
        & ~duplicate_id
    )
    return out


def range_source_audit(frame: pd.DataFrame, *, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    bars = normalize_range_bars(frame)
    cutoff = None if cutoff is None else pd.Timestamp(cutoff)
    checks = [
        ("rows", len(bars)),
        ("valid_rows", int(bars["source_valid"].sum())),
        ("invalid_rows", int((~bars["source_valid"]).sum())),
        ("duplicate_bar_ids", int(bars["bar_id"].duplicated(keep=False).sum())),
        ("exact_duplicate_rows", int(bars.duplicated(list(RANGE_REQUIRED), keep=False).sum())),
        ("required_null_rows", int(bars[list(RANGE_REQUIRED)].isna().any(axis=1).sum())),
        ("invalid_direction_rows", int((~bars["direction"].isin([-1.0, 1.0])).sum())),
        ("start_after_end", int(bars["start_ts"].gt(bars["end_ts"]).sum())),
        ("zero_duration", int(_num(bars, "duration_seconds").le(0).sum())),
        ("equal_end_timestamp_rows", int(bars["end_ts"].duplicated(keep=False).sum())),
    ]
    if cutoff is not None:
        checks.append(("end_at_or_after_cutoff", int(bars["end_ts"].ge(cutoff).sum())))
    return pd.DataFrame(checks, columns=["check", "value"])


def range_temporal_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly source-shape profile; Range Bars have no fixed expected count."""
    bars = normalize_range_bars(frame)
    bars = bars.loc[bars["end_ts"].notna()].copy()
    bars["month"] = bars["end_ts"].dt.to_period("M").astype(str)
    bars["zero_duration"] = _num(bars, "duration_seconds").le(0)
    bars["equal_end_timestamp"] = bars["end_ts"].duplicated(keep=False)
    rows = []
    for month, part in bars.groupby("month", sort=True):
        rows.append({
            "month": month,
            "rows": len(part),
            "valid_rows": int(part["source_valid"].sum()),
            "zero_duration_rate": float(part["zero_duration"].mean()),
            "equal_end_timestamp_rate": float(part["equal_end_timestamp"].mean()),
            "median_duration_seconds": float(_num(part, "duration_seconds").median()),
            "p90_duration_seconds": float(_num(part, "duration_seconds").quantile(0.90)),
        })
    return pd.DataFrame(rows)


def build_range_run_events(
    range_bars: pd.DataFrame,
    *,
    config: R25Config | None = None,
) -> pd.DataFrame:
    """Build maximal same-direction runs confirmed by the first opposite bar."""
    cfg = (config or R25Config()).validate()
    bars = normalize_range_bars(range_bars)
    if bars.empty:
        return pd.DataFrame()
    valid = bars["source_valid"].to_numpy(dtype=bool)
    direction = _num(bars, "direction").fillna(0).to_numpy(dtype=np.int8)
    start_ts = bars["start_ts"].to_numpy(dtype="datetime64[ns]")
    end_ts = bars["end_ts"].to_numpy(dtype="datetime64[ns]")
    open_ = _num(bars, "open").to_numpy(dtype=float)
    high = _num(bars, "high").to_numpy(dtype=float)
    low = _num(bars, "low").to_numpy(dtype=float)
    duration = _num(bars, "duration_seconds").to_numpy(dtype=float)
    bar_id = _num(bars, "bar_id").fillna(-1).to_numpy(dtype=np.int64)
    rows: list[dict[str, object]] = []
    run_start: int | None = None
    run_dir = 0
    event_no = 0

    for pos in range(len(bars)):
        if not valid[pos]:
            run_start = None
            run_dir = 0
            continue
        this_dir = int(direction[pos])
        if run_start is None:
            run_start = pos
            run_dir = this_dir
            continue
        if this_dir == run_dir:
            continue

        run_end = pos - 1
        run_count = run_end - run_start + 1
        if run_count >= cfg.min_run_bars:
            event_no += 1
            trade_dir = -run_dir
            run_slice = slice(run_start, run_end + 1)
            target = float(open_[run_start])
            if trade_dir > 0:
                sequence_extreme = float(min(np.nanmin(low[run_slice]), low[pos]))
                touched_target = bool(high[pos] >= target)
            else:
                sequence_extreme = float(max(np.nanmax(high[run_slice]), high[pos]))
                touched_target = bool(low[pos] <= target)
            first_start = pd.Timestamp(start_ts[run_start])
            last_end = pd.Timestamp(end_ts[run_end])
            span_seconds = float((last_end - first_start) / pd.Timedelta(seconds=1))
            eligible = bool(span_seconds > 0 and not touched_target)
            reason = "eligible" if eligible else ("nonpositive_run_span" if span_seconds <= 0 else "target_touched_on_confirmation")
            dur = duration[run_slice]
            prior = dur[:-1]
            terminal_ratio = float(dur[-1] / np.nanmedian(prior)) if len(prior) and np.nanmedian(prior) > EPS else np.nan
            rows.append(
                {
                    "event_id": f"R25_EVENT_{event_no:07d}",
                    "run_direction": int(run_dir),
                    "trade_direction": int(trade_dir),
                    "direction": "Long" if trade_dir > 0 else "Short",
                    "run_start_pos": int(run_start),
                    "run_end_pos": int(run_end),
                    "confirmation_pos": int(pos),
                    "run_start_bar_id": int(bar_id[run_start]),
                    "run_end_bar_id": int(bar_id[run_end]),
                    "confirmation_bar_id": int(bar_id[pos]),
                    "run_start_ts": first_start,
                    "run_end_ts": last_end,
                    "confirmation_start_ts": pd.Timestamp(start_ts[pos]),
                    "confirmation_end_ts": pd.Timestamp(end_ts[pos]),
                    "signal_time": pd.Timestamp(end_ts[pos]),
                    "run_bars": int(run_count),
                    "run_span_seconds": span_seconds,
                    "run_origin_price": target,
                    "run_extreme_price": sequence_extreme,
                    "confirmation_open": float(open_[pos]),
                    "confirmation_high": float(high[pos]),
                    "confirmation_low": float(low[pos]),
                    "stop_price": sequence_extreme,
                    "target_price": target,
                    "run_duration_mean_seconds": float(np.nanmean(dur)),
                    "run_duration_median_seconds": float(np.nanmedian(dur)),
                    "run_terminal_duration_ratio": terminal_ratio,
                    "confirmation_touched_target": touched_target,
                    "signal_eligible": eligible,
                    "exclusion_reason": reason,
                }
            )
        run_start = pos
        run_dir = this_dir
    return pd.DataFrame(rows)


def _passage(
    time_ns: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    start_pos: int,
    end_pos: int,
    direction: int,
    stop: float,
    target: float,
) -> tuple[str, int, float, float, float] | None:
    favorable = -np.inf
    adverse = np.inf
    entry = float(open_[start_pos])
    for pos in range(start_pos, end_pos):
        if direction > 0:
            favorable = max(favorable, float(high[pos] / entry - 1.0))
            adverse = min(adverse, float(low[pos] / entry - 1.0))
            stop_hit = low[pos] <= stop
            target_hit = high[pos] >= target
        else:
            favorable = max(favorable, float(1.0 - low[pos] / entry))
            adverse = min(adverse, float(1.0 - high[pos] / entry))
            stop_hit = high[pos] >= stop
            target_hit = low[pos] <= target
        if stop_hit:
            fill = min(float(open_[pos]), stop) if direction > 0 else max(float(open_[pos]), stop)
            return "STOP", pos, fill, favorable, adverse
        if target_hit:
            return "TARGET", pos, target, favorable, adverse
    return None


def simulate_range_run_reversal(
    bars_1m: pd.DataFrame,
    events: pd.DataFrame,
    *,
    direction: int,
    split: str,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    entry_delay_minutes: int = 0,
    config: R25Config | None = None,
) -> pd.DataFrame:
    """Simulate one non-overlapping direction sleeve with split-boundary censoring."""
    cfg = (config or R25Config()).validate()
    if direction not in (-1, 1) or entry_delay_minutes not in cfg.entry_delays_minutes:
        raise ValueError("invalid R25 simulation path")
    split_start, split_end = pd.Timestamp(split_start), pd.Timestamp(split_end)
    bars = normalize_1m_bars(bars_1m)
    bars = bars.loc[(bars.index >= split_start) & (bars.index < split_end)]
    if bars.empty:
        return pd.DataFrame()
    time_ns = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    open_ = _num(bars, "open").to_numpy(dtype=float)
    high = _num(bars, "high").to_numpy(dtype=float)
    low = _num(bars, "low").to_numpy(dtype=float)
    candidates = events.loc[
        events["trade_direction"].eq(direction)
        & pd.to_datetime(events["signal_time"]).ge(split_start)
        & pd.to_datetime(events["signal_time"]).lt(split_end)
    ].sort_values(["signal_time", "confirmation_bar_id"], kind="mergesort")
    rows: list[dict[str, object]] = []
    available_ns = int(split_start.value)
    ordinal = 0
    for event in candidates.itertuples(index=False):
        ordinal += 1
        base = {
            "trade_id": f"R25_D{entry_delay_minutes}_{split}_{'LONG' if direction > 0 else 'SHORT'}_{ordinal:05d}",
            "event_id": event.event_id,
            "research_split": split,
            "direction": "Long" if direction > 0 else "Short",
            "trade_direction": int(direction),
            "entry_delay_minutes": int(entry_delay_minutes),
            "run_start_bar_id": event.run_start_bar_id,
            "run_end_bar_id": event.run_end_bar_id,
            "confirmation_bar_id": event.confirmation_bar_id,
            "run_start_ts": event.run_start_ts,
            "run_end_ts": event.run_end_ts,
            "confirmation_end_ts": event.confirmation_end_ts,
            "signal_time": event.signal_time,
            "run_bars": event.run_bars,
            "run_span_seconds": event.run_span_seconds,
            "run_origin_price": event.run_origin_price,
            "run_terminal_duration_ratio": event.run_terminal_duration_ratio,
            "confirmation_touched_target": event.confirmation_touched_target,
            "stop_price": event.stop_price,
            "target_price": event.target_price,
        }
        if not bool(event.signal_eligible):
            rows.append({**base, "path_status": "pre_entry_ineligible", "exit_reason": event.exclusion_reason})
            continue
        signal_ns = int(pd.Timestamp(event.signal_time).value)
        primary_pos = int(np.searchsorted(time_ns, signal_ns, side="right"))
        if primary_pos >= len(time_ns):
            rows.append({**base, "path_status": "no_next_entry", "exit_reason": "NO_NEXT_OBSERVED_MINUTE"})
            continue
        primary_time = pd.Timestamp(time_ns[primary_pos])
        delayed_floor = int((primary_time + pd.Timedelta(minutes=entry_delay_minutes)).value)
        entry_pos = int(np.searchsorted(time_ns, delayed_floor, side="left"))
        if entry_pos >= len(time_ns):
            rows.append({**base, "primary_entry_time": primary_time, "path_status": "no_next_entry", "exit_reason": "NO_DELAYED_OBSERVED_MINUTE"})
            continue
        entry_time = pd.Timestamp(time_ns[entry_pos])
        entry_price = float(open_[entry_pos])
        base.update({"primary_entry_time": primary_time, "entry_time": entry_time, "entry_price": entry_price})
        if int(entry_time.value) < available_ns:
            rows.append({**base, "path_status": "overlap_ignored", "exit_reason": "POSITION_ALREADY_OPEN"})
            continue
        stop, target = float(event.stop_price), float(event.target_price)
        between = stop < entry_price < target if direction > 0 else target < entry_price < stop
        if not between:
            rows.append({**base, "path_status": "entry_geometry_invalid", "exit_reason": "ENTRY_NOT_BETWEEN_BARRIERS"})
            continue
        risk_pct = abs(entry_price - stop) / entry_price
        reward_pct = abs(target - entry_price) / entry_price
        hit = _passage(time_ns, open_, high, low, entry_pos, len(time_ns), direction, stop, target)
        if hit is None:
            available_ns = int(split_end.value)
            rows.append({
                **base, "path_status": "boundary_censored", "exit_reason": "SPLIT_BOUNDARY_CENSORED",
                "risk_distance_pct": risk_pct, "reward_distance_pct": reward_pct,
                "initial_reward_risk": reward_pct / risk_pct if risk_pct > EPS else np.nan,
            })
            continue
        reason, exit_pos, exit_price, mfe, mae = hit
        exit_time = pd.Timestamp(time_ns[exit_pos])
        available_ns = int((exit_time + pd.Timedelta(minutes=1)).value)
        gross = float(direction * (exit_price / entry_price - 1.0))
        row = {
            **base,
            "path_status": "included", "exit_reason": reason,
            "exit_time": exit_time, "exit_price": exit_price,
            "holding_minutes": float((exit_time - entry_time) / pd.Timedelta(minutes=1) + 1.0),
            "risk_distance_pct": risk_pct, "reward_distance_pct": reward_pct,
            "initial_reward_risk": reward_pct / risk_pct if risk_pct > EPS else np.nan,
            "mfe_pct": mfe, "mae_pct": mae, "gross_return": gross,
            "realized_r": gross / risk_pct if risk_pct > EPS else np.nan,
        }
        for scale in cfg.cost_scales:
            row[f"fee_cost{int(scale)}x"] = float(scale * cfg.market_roundtrip_cost)
            row[f"net_return_cost{int(scale)}x"] = gross - scale * cfg.market_roundtrip_cost
        rows.append(row)
    return pd.DataFrame(rows)


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > EPS else (np.inf if gains > EPS else np.nan)


def _max_underwater_days(daily_equity: pd.Series) -> int:
    peak = daily_equity.cummax()
    below = daily_equity.lt(peak)
    if not below.any():
        return 0
    groups = below.ne(below.shift(fill_value=False)).cumsum()
    return int(below.groupby(groups).sum().max())


def summarize_r25(trades: pd.DataFrame, *, config: R25Config | None = None) -> pd.DataFrame:
    cfg = (config or R25Config()).validate()
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    rows: list[dict[str, object]] = []
    for (delay, split, direction), part in closed.groupby(["entry_delay_minutes", "research_split", "direction"], sort=True):
        part = part.sort_values(["exit_time", "trade_id"], kind="mergesort")
        start = cfg.discovery_start if split == "discovery" else cfg.validation_start
        end = cfg.validation_start if split == "discovery" else cfg.embargo_start
        months_n = 24 if split == "discovery" else 6
        month_index = pd.period_range(start, end - pd.Timedelta(seconds=1), freq="M")
        quarter_index = pd.period_range(start, end - pd.Timedelta(seconds=1), freq="Q")
        net2 = _num(part, "net_return_cost2x")
        top = net2.sort_values(ascending=False)
        without5 = net2.drop(index=top.head(5).index)
        without10 = net2.drop(index=top.head(10).index)
        exit_period = pd.to_datetime(part["exit_time"])
        monthly = net2.groupby(exit_period.dt.to_period("M")).sum().reindex(month_index, fill_value=0.0)
        quarterly = net2.groupby(exit_period.dt.to_period("Q")).sum().reindex(quarter_index, fill_value=0.0)
        daily_index = pd.date_range(start.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D")
        daily = net2.groupby(exit_period.dt.normalize()).sum().reindex(daily_index, fill_value=0.0)
        daily_equity = (1.0 + daily).cumprod()
        daily_dd = daily_equity / daily_equity.cummax() - 1.0
        rolling = (1.0 + daily).rolling(90, min_periods=90).apply(np.prod, raw=True) - 1.0
        entries = pd.to_datetime(part.sort_values("entry_time")["entry_time"]).reset_index(drop=True)
        exits = pd.to_datetime(part.sort_values("entry_time")["exit_time"]).reset_index(drop=True)
        flat = (entries.iloc[1:].reset_index(drop=True) - exits.iloc[:-1].reset_index(drop=True)) / pd.Timedelta(days=1)
        realized_equity = (1.0 + net2).cumprod()
        realized_dd = realized_equity / realized_equity.cummax() - 1.0
        rows.append({
            "entry_delay_minutes": int(delay), "research_split": split, "direction": direction,
            "trades": len(part), "trades_per_month": len(part) / months_n,
            "win_rate": float(_num(part, "gross_return").gt(0).mean()),
            "gross_pf": _pf(part["gross_return"]), "mean_gross_return": float(_num(part, "gross_return").mean()),
            "r_pf": _pf(part["realized_r"]), "expectancy_r": float(_num(part, "realized_r").mean()),
            "net_pf_cost1x": _pf(part["net_return_cost1x"]), "mean_net_return_cost1x": float(_num(part, "net_return_cost1x").mean()),
            "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()),
            "net_pf_cost3x": _pf(part["net_return_cost3x"]), "mean_net_return_cost3x": float(_num(part, "net_return_cost3x").mean()),
            "net_sum_cost2x": float(net2.sum()), "compounded_return_cost2x": float(realized_equity.iloc[-1] - 1.0),
            "realized_max_drawdown_cost2x": float(realized_dd.min()), "daily_realized_max_drawdown_cost2x": float(daily_dd.min()),
            "positive_months": int(monthly.gt(0).sum()), "positive_month_rate_cost2x": float(monthly.gt(0).mean()),
            "positive_quarters": int(quarterly.gt(0).sum()), "positive_quarter_rate_cost2x": float(quarterly.gt(0).mean()),
            "rolling_90d_positive_rate_cost2x": float(rolling.dropna().gt(0).mean()) if rolling.notna().any() else np.nan,
            "longest_underwater_days_cost2x": _max_underwater_days(daily_equity),
            "longest_entry_gap_days": float(entries.diff().max() / pd.Timedelta(days=1)) if len(entries) > 1 else np.nan,
            "longest_flat_days": float(flat.clip(lower=0).max()) if len(flat) else np.nan,
            "median_flat_days": float(flat.clip(lower=0).median()) if len(flat) else np.nan,
            "p90_flat_days": float(flat.clip(lower=0).quantile(0.90)) if len(flat) else np.nan,
            "median_holding_minutes": float(_num(part, "holding_minutes").median()),
            "median_risk_distance_pct": float(_num(part, "risk_distance_pct").median()),
            "median_initial_reward_risk": float(_num(part, "initial_reward_risk").median()),
            "median_realized_r": float(_num(part, "realized_r").median()),
            "median_mfe_pct": float(_num(part, "mfe_pct").median()), "median_mae_pct": float(_num(part, "mae_pct").median()),
            "target_first_rate": float(part["exit_reason"].eq("TARGET").mean()), "stop_first_rate": float(part["exit_reason"].eq("STOP").mean()),
            "fees_cost1x": float(_num(part, "fee_cost1x").sum()), "fees_cost2x": float(_num(part, "fee_cost2x").sum()), "fees_cost3x": float(_num(part, "fee_cost3x").sum()),
            "net_pf_cost2x_top5_removed": _pf(without5), "net_sum_cost2x_top5_removed": float(without5.sum()),
            "net_pf_cost2x_top10_removed": _pf(without10), "net_sum_cost2x_top10_removed": float(without10.sum()),
        })
    return pd.DataFrame(rows)


def summarize_r25_periods(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    if closed.empty:
        return pd.DataFrame(), pd.DataFrame()
    closed["year"] = pd.to_datetime(closed["exit_time"]).dt.year
    closed["quarter"] = pd.to_datetime(closed["exit_time"]).dt.to_period("Q").astype(str)
    outputs: list[pd.DataFrame] = []
    for period in ("year", "quarter"):
        rows = []
        for (delay, direction, value), part in closed.groupby(["entry_delay_minutes", "direction", period], sort=True):
            net2 = _num(part, "net_return_cost2x")
            rows.append({
                "entry_delay_minutes": int(delay), "direction": direction, period: value,
                "trades": len(part), "net_pf_cost2x": _pf(net2),
                "mean_net_return_cost2x": float(net2.mean()), "net_sum_cost2x": float(net2.sum()),
            })
        outputs.append(pd.DataFrame(rows))
    return outputs[0], outputs[1]


def build_r25_gate(score: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction in ("Long", "Short"):
        reasons: list[str] = []
        primary = score.loc[score["entry_delay_minutes"].eq(0) & score["direction"].eq(direction)].set_index("research_split")
        for split, minimum in (("discovery", 100), ("validation", 20)):
            if split not in primary.index:
                reasons.append(f"missing_{split}")
                continue
            row = primary.loc[split]
            if int(row["trades"]) < minimum: reasons.append(f"{split}_sample")
            if float(row["net_pf_cost2x"]) < 1.4: reasons.append(f"{split}_pf2x")
            if float(row["mean_net_return_cost2x"]) <= 0: reasons.append(f"{split}_expectancy")
            if float(row["positive_month_rate_cost2x"]) < 0.80: reasons.append(f"{split}_positive_months")
            if float(row["median_realized_r"]) <= 0: reasons.append(f"{split}_median_r")
        if "discovery" in primary.index and float(primary.loc["discovery", "net_sum_cost2x_top10_removed"]) <= 0:
            reasons.append("discovery_top10")
        year_rows = years.loc[years["entry_delay_minutes"].eq(0) & years["direction"].eq(direction)].set_index("year")
        for year in (2023, 2024, 2025):
            if year not in year_rows.index or float(year_rows.loc[year, "net_sum_cost2x"]) <= 0:
                reasons.append(f"year_{year}")
        delayed = score.loc[score["entry_delay_minutes"].eq(1) & score["direction"].eq(direction)].set_index("research_split")
        for split in ("discovery", "validation"):
            if split not in delayed.index or float(delayed.loc[split, "mean_net_return_cost2x"]) <= 0:
                reasons.append(f"delay1_{split}")
        rows.append({"direction": direction, "research_candidate": int(not reasons), "reason": "PASS" if not reasons else "FAIL_" + ",".join(reasons)})
    return pd.DataFrame(rows)


def r25_causal_audit(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: R25Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R25Config()).validate()
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    entered = trades.loc[trades["path_status"].isin(["included", "boundary_censored"])].copy()
    rows = [
        {"check": "unique_trade_id", "violations": int(trades["trade_id"].duplicated().sum())},
        {"check": "unique_event_ids", "violations": int(events["event_id"].duplicated().sum())},
        {"check": "fixed_min_run", "violations": int(_num(events, "run_bars").lt(cfg.min_run_bars).sum())},
        {"check": "opposite_confirmation", "violations": int((_num(events, "run_direction") == _num(events, "trade_direction")).sum())},
        {"check": "signal_at_completed_end", "violations": int((pd.to_datetime(events["signal_time"]) != pd.to_datetime(events["confirmation_end_ts"])).sum())},
        {"check": "entry_strictly_after_signal", "violations": int((pd.to_datetime(entered["entry_time"]) <= pd.to_datetime(entered["signal_time"])).sum())},
        {"check": "delay_not_before_primary", "violations": int((pd.to_datetime(entered["entry_time"]) < pd.to_datetime(entered["primary_entry_time"]) + pd.to_timedelta(_num(entered, "entry_delay_minutes"), unit="m")).sum())},
        {"check": "entry_between_barriers", "violations": int((((_num(entered, "trade_direction") > 0) & ~((_num(entered, "stop_price") < _num(entered, "entry_price")) & (_num(entered, "entry_price") < _num(entered, "target_price")))) | ((_num(entered, "trade_direction") < 0) & ~((_num(entered, "target_price") < _num(entered, "entry_price")) & (_num(entered, "entry_price") < _num(entered, "stop_price"))))).sum())},
        {"check": "exit_after_entry", "violations": int((pd.to_datetime(closed["exit_time"]) < pd.to_datetime(closed["entry_time"])).sum())},
        {"check": "discovery_boundary", "violations": int((closed["research_split"].eq("discovery") & pd.to_datetime(closed["exit_time"]).ge(cfg.validation_start)).sum())},
        {"check": "validation_boundary", "violations": int((closed["research_split"].eq("validation") & pd.to_datetime(closed["exit_time"]).ge(cfg.embargo_start)).sum())},
        {"check": "holdout_absent", "violations": int(trades["research_split"].isin(["embargo", "holdout"]).sum())},
        {"check": "ineligible_never_entered", "violations": int(trades.loc[trades["path_status"].isin(["included", "boundary_censored"]), "confirmation_touched_target"].fillna(False).sum())},
    ]
    for scale in cfg.cost_scales:
        expected = _num(closed, "gross_return") - scale * cfg.market_roundtrip_cost
        actual = _num(closed, f"net_return_cost{int(scale)}x")
        rows.append({"check": f"cost{int(scale)}x_formula", "violations": int((actual - expected).abs().gt(1e-12).sum())})
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["violations"].eq(0), "PASS", "FAIL")
    return out
