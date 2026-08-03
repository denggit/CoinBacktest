#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Target-centric R03.1 trade replay.

There is deliberately no minimum holding time and no 15m model/trend exit. A
position leaves only through target, causal stop/profit protection, or the
research horizon safety cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.backtest import MarketPath
from src.ai_research.swing_baseline.config import SwingTargetSpec
from src.ai_research.swing_baseline.modeling import PeriodData

from .config import ExitPolicySpec, SwingEntryMvpConfig


@dataclass(frozen=True)
class EntryTradeRecord:
    fold_id: str
    architecture: str
    target_id: str
    direction: str
    exit_policy: str
    quantile: float
    delay_minutes: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    initial_stop_price: float
    final_stop_price: float
    target_price: float
    exit_reason: str
    hold_hours: float
    gross_return: float
    mfe: float
    mae: float
    target_hit: bool
    protection_activated: bool
    score_direction: float
    score_opposite: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("signal_time", "entry_time", "exit_time"):
            payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class EntrySimulationResult:
    records: tuple[EntryTradeRecord, ...]
    rejected_structural_risk: int


def score_thresholds(scores: np.ndarray, quantiles: Iterable[float]) -> dict[float, float]:
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        raise RuntimeError("R03.1 calibration scores are empty")
    return {float(q): float(np.quantile(valid, q)) for q in quantiles}


def _context_map(data: PeriodData) -> dict[str, np.ndarray]:
    return {name: data.context[:, index] for index, name in enumerate(data.context_columns)}


def _stop_for_entry(
    *,
    direction: int,
    entry_price: float,
    signal_position: int,
    target: SwingTargetSpec,
    policy: ExitPolicySpec,
    context: dict[str, np.ndarray],
    config: SwingEntryMvpConfig,
) -> float | None:
    if not policy.use_structural_stop:
        risk = float(target.max_adverse_move)
    else:
        atr_abs = float(context["ctx_atr_abs_4h"][signal_position])
        if direction > 0:
            structural = float(context["ctx_recent_low_4h"][signal_position]) - config.base.structural_buffer_atr * atr_abs
            risk = (entry_price - structural) / entry_price
        else:
            structural = float(context["ctx_recent_high_4h"][signal_position]) + config.base.structural_buffer_atr * atr_abs
            risk = (structural - entry_price) / entry_price
        if not np.isfinite(risk) or risk <= 0 or risk > target.max_adverse_move:
            return None
        risk = max(config.base.min_initial_stop_pct, float(risk))
    if direction > 0:
        return entry_price * (1.0 - risk)
    return entry_price * (1.0 + risk)


def simulate_entry_portfolio(
    *,
    fold_id: str,
    architecture: str,
    target: SwingTargetSpec,
    direction_name: str,
    policy: ExitPolicySpec,
    quantile: float,
    delay_minutes: int,
    period: PeriodData,
    score_long: np.ndarray,
    score_short: np.ndarray,
    threshold: float,
    market_path: MarketPath,
    config: SwingEntryMvpConfig,
) -> EntrySimulationResult:
    if direction_name not in {"long", "short"}:
        raise ValueError(f"unsupported direction: {direction_name}")
    direction = 1 if direction_name == "long" else -1
    score_direction = score_long if direction > 0 else score_short
    score_opposite = score_short if direction > 0 else score_long
    signals = (
        np.isfinite(score_direction)
        & np.isfinite(score_opposite)
        & (score_direction >= threshold)
        & ((score_direction - score_opposite) >= config.score_margin)
    )
    context = _context_map(period)
    path_times = market_path.times_ns
    delay_ns = int(pd.Timedelta(minutes=delay_minutes).value)
    horizon_ns = int(pd.Timedelta(hours=target.horizon_hours).value)
    cooldown_ns = int(pd.Timedelta(minutes=config.cooldown_minutes).value)
    next_signal_ns = np.int64(-2**63)
    records: list[EntryTradeRecord] = []
    rejected = 0

    for signal_position in np.flatnonzero(signals):
        signal_ns = int(period.timestamps_ns[signal_position])
        if signal_ns < next_signal_ns:
            continue
        requested_entry_ns = signal_ns + delay_ns
        entry_pos = int(np.searchsorted(path_times, requested_entry_ns, side="left"))
        if entry_pos >= len(path_times):
            break
        entry_ns = int(path_times[entry_pos])
        entry_price = float(market_path.open[entry_pos])
        initial_stop = _stop_for_entry(
            direction=direction,
            entry_price=entry_price,
            signal_position=signal_position,
            target=target,
            policy=policy,
            context=context,
            config=config,
        )
        if initial_stop is None:
            rejected += 1
            continue
        target_price = entry_price * (1.0 + target.target_move) if direction > 0 else entry_price * (1.0 - target.target_move)
        active_stop = float(initial_stop)
        protection_activated = False
        peak = entry_price
        trough = entry_price
        max_mfe = 0.0
        max_mae = 0.0
        end_ns = min(entry_ns + horizon_ns, int(path_times[-1]))
        end_pos = int(np.searchsorted(path_times, end_ns, side="right")) - 1
        exit_ns = int(path_times[end_pos])
        exit_price = float(market_path.close[end_pos])
        exit_reason = "horizon_cap" if end_ns < int(path_times[-1]) else "data_end"
        target_hit = False

        for minute_pos in range(entry_pos, end_pos + 1):
            minute_ns = int(path_times[minute_pos])
            minute_high = float(market_path.high[minute_pos])
            minute_low = float(market_path.low[minute_pos])

            # Conservative same-minute path rule: the active adverse boundary wins.
            if direction > 0 and minute_low <= active_stop:
                max_mae = max(max_mae, max(0.0, 1.0 - active_stop / entry_price))
                exit_ns = minute_ns
                exit_price = active_stop
                exit_reason = "protected_stop" if protection_activated else "initial_stop"
                break
            if direction < 0 and minute_high >= active_stop:
                max_mae = max(max_mae, max(0.0, active_stop / entry_price - 1.0))
                exit_ns = minute_ns
                exit_price = active_stop
                exit_reason = "protected_stop" if protection_activated else "initial_stop"
                break

            if direction > 0 and minute_high >= target_price:
                exit_ns = minute_ns
                exit_price = target_price
                exit_reason = "target_hit"
                target_hit = True
                peak = max(peak, target_price)
                max_mfe = max(max_mfe, target.target_move)
                break
            if direction < 0 and minute_low <= target_price:
                exit_ns = minute_ns
                exit_price = target_price
                exit_reason = "target_hit"
                target_hit = True
                trough = min(trough, target_price)
                max_mfe = max(max_mfe, target.target_move)
                break

            peak = max(peak, minute_high)
            trough = min(trough, minute_low)
            if direction > 0:
                max_mfe = max(max_mfe, peak / entry_price - 1.0)
                max_mae = max(max_mae, 1.0 - trough / entry_price)
            else:
                max_mfe = max(max_mfe, 1.0 - trough / entry_price)
                max_mae = max(max_mae, peak / entry_price - 1.0)

            # Protection is updated after the minute path, becoming active next minute.
            if policy.enable_profit_protection and max_mfe >= target.target_move * config.protection_trigger_fraction:
                locked_return = target.target_move * config.locked_profit_fraction
                if direction > 0:
                    active_stop = max(active_stop, entry_price * (1.0 + locked_return))
                else:
                    active_stop = min(active_stop, entry_price * (1.0 - locked_return))
                protection_activated = True

        gross = (
            (exit_price - entry_price) / entry_price
            if direction > 0
            else (entry_price - exit_price) / entry_price
        )
        hold_hours = (exit_ns - entry_ns) / float(pd.Timedelta(hours=1).value)
        records.append(
            EntryTradeRecord(
                fold_id=fold_id,
                architecture=architecture,
                target_id=target.target_id,
                direction=direction_name,
                exit_policy=policy.policy_id,
                quantile=float(quantile),
                delay_minutes=int(delay_minutes),
                signal_time=pd.Timestamp(signal_ns),
                entry_time=pd.Timestamp(entry_ns),
                exit_time=pd.Timestamp(exit_ns),
                entry_price=entry_price,
                exit_price=float(exit_price),
                initial_stop_price=float(initial_stop),
                final_stop_price=float(active_stop),
                target_price=float(target_price),
                exit_reason=exit_reason,
                hold_hours=float(hold_hours),
                gross_return=float(gross),
                mfe=float(max_mfe),
                mae=float(max_mae),
                target_hit=bool(target_hit),
                protection_activated=bool(protection_activated),
                score_direction=float(score_direction[signal_position]),
                score_opposite=float(score_opposite[signal_position]),
            )
        )
        next_signal_ns = np.int64(exit_ns + cooldown_ns)
    return EntrySimulationResult(tuple(records), rejected)


def summarize_records(
    result: EntrySimulationResult,
    *,
    fold_id: str,
    architecture: str,
    target_id: str,
    direction: str,
    exit_policy: str,
    quantile: float,
    delay_minutes: int,
    cost_multiplier: float,
    config: SwingEntryMvpConfig,
) -> dict[str, object]:
    records = result.records
    base = {
        "fold_id": fold_id,
        "architecture": architecture,
        "target_id": target_id,
        "direction": direction,
        "exit_policy": exit_policy,
        "quantile": float(quantile),
        "delay_minutes": int(delay_minutes),
        "cost_multiplier": float(cost_multiplier),
        "rejected_structural_risk": int(result.rejected_structural_risk),
    }
    if not records:
        return {
            **base,
            "trades": 0,
            "win_rate": 0.0,
            "target_hit_rate": 0.0,
            "mean_gross_return": 0.0,
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "positive_month_ratio": 0.0,
            "positive_quarter_ratio": 0.0,
            "top3_removed_total_return": 0.0,
            "average_hold_hours": 0.0,
            "median_hold_hours": 0.0,
            "average_mfe": 0.0,
            "average_mae": 0.0,
            "mfe_capture_ratio": 0.0,
        }
    gross = np.asarray([record.gross_return for record in records], dtype=float)
    net = gross - config.base.base_round_trip_cost * float(cost_multiplier)
    curve = np.cumprod(1.0 + net)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    drawdown = curve / peaks - 1.0
    wins = net[net > 0]
    losses = net[net <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("inf")
    exits = pd.DatetimeIndex([record.exit_time for record in records])
    series = pd.Series(net, index=exits)
    monthly = series.groupby(exits.to_period("M")).sum()
    quarterly = series.groupby(exits.to_period("Q")).sum()
    removed = net.copy()
    remove_count = min(3, len(removed))
    if remove_count:
        top = np.argpartition(removed, -remove_count)[-remove_count:]
        removed[top] = 0.0
    mfe = np.asarray([record.mfe for record in records], dtype=float)
    positive_mfe = mfe > 1e-12
    capture = np.zeros(len(records), dtype=float)
    capture[positive_mfe] = np.maximum(gross[positive_mfe], 0.0) / mfe[positive_mfe]
    holds = np.asarray([record.hold_hours for record in records], dtype=float)
    return {
        **base,
        "trades": int(len(records)),
        "win_rate": float(np.mean(net > 0)),
        "target_hit_rate": float(np.mean([record.target_hit for record in records])),
        "mean_gross_return": float(np.mean(gross)),
        "mean_net_return": float(np.mean(net)),
        "median_net_return": float(np.median(net)),
        "profit_factor": pf,
        "total_return": float(curve[-1] - 1.0),
        "max_drawdown": float(np.min(drawdown)),
        "positive_month_ratio": float(np.mean(monthly > 0)) if len(monthly) else 0.0,
        "positive_quarter_ratio": float(np.mean(quarterly > 0)) if len(quarterly) else 0.0,
        "top3_removed_total_return": float(np.prod(1.0 + removed) - 1.0),
        "average_hold_hours": float(np.mean(holds)),
        "median_hold_hours": float(np.median(holds)),
        "average_mfe": float(np.mean(mfe)),
        "average_mae": float(np.mean([record.mae for record in records])),
        "mfe_capture_ratio": float(np.mean(capture[positive_mfe])) if positive_mfe.any() else 0.0,
    }


def evaluate_scenarios(
    *,
    fold_id: str,
    architecture: str,
    target: SwingTargetSpec,
    period: PeriodData,
    score_long: np.ndarray,
    score_short: np.ndarray,
    thresholds: dict[str, dict[float, float]],
    market_path: MarketPath,
    config: SwingEntryMvpConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for direction in config.direction_modes:
        for policy in config.exit_policies:
            for quantile, threshold in thresholds[direction].items():
                for delay in config.base.delay_scenarios_minutes:
                    simulation = simulate_entry_portfolio(
                        fold_id=fold_id,
                        architecture=architecture,
                        target=target,
                        direction_name=direction,
                        policy=policy,
                        quantile=quantile,
                        delay_minutes=delay,
                        period=period,
                        score_long=score_long,
                        score_short=score_short,
                        threshold=threshold,
                        market_path=market_path,
                        config=config,
                    )
                    for cost_multiplier in config.base.cost_stress_multipliers:
                        row = summarize_records(
                            simulation,
                            fold_id=fold_id,
                            architecture=architecture,
                            target_id=target.target_id,
                            direction=direction,
                            exit_policy=policy.policy_id,
                            quantile=quantile,
                            delay_minutes=delay,
                            cost_multiplier=cost_multiplier,
                            config=config,
                        )
                        row["score_threshold"] = float(threshold)
                        summaries.append(row)
                    if delay == config.base.execution_delay_minutes:
                        trade_rows.extend(record.to_dict() for record in simulation.records)
    return pd.DataFrame(summaries), pd.DataFrame(trade_rows)
