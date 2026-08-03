#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative structural-exit backtest for R03 swing predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import SwingBaselineConfig, SwingTargetSpec
from .dataset import load_year_shard
from .modeling import PeriodData


@dataclass(frozen=True)
class MarketPath:
    times_ns: np.ndarray
    ohlc: np.ndarray

    @property
    def open(self) -> np.ndarray:
        return self.ohlc[:, 0]

    @property
    def high(self) -> np.ndarray:
        return self.ohlc[:, 1]

    @property
    def low(self) -> np.ndarray:
        return self.ohlc[:, 2]

    @property
    def close(self) -> np.ndarray:
        return self.ohlc[:, 3]


def build_market_path(paths: Iterable[Path], start: pd.Timestamp, end: pd.Timestamp) -> MarketPath:
    time_parts: list[np.ndarray] = []
    ohlc_parts: list[np.ndarray] = []
    start_ns = int(pd.Timestamp(start).value)
    end_ns = int(pd.Timestamp(end).value)
    for path in paths:
        shard = load_year_shard(path)
        left = int(np.searchsorted(shard.minute_times_ns, start_ns, side="left"))
        right = int(np.searchsorted(shard.minute_times_ns, end_ns, side="right"))
        if right > left:
            time_parts.append(np.asarray(shard.minute_times_ns[left:right], dtype=np.int64))
            ohlc_parts.append(np.asarray(shard.minute_ohlc[left:right], dtype=np.float64))
    if not time_parts:
        raise RuntimeError(f"no R03 minute path for {start} -> {end}")
    times = np.concatenate(time_parts)
    ohlc = np.concatenate(ohlc_parts)
    order = np.argsort(times, kind="stable")
    times = times[order]
    ohlc = ohlc[order]
    unique = np.ones(len(times), dtype=bool)
    unique[:-1] = times[:-1] != times[1:]
    return MarketPath(times_ns=times[unique], ohlc=ohlc[unique])


@dataclass(frozen=True)
class TradeRecord:
    fold_id: str
    architecture: str
    target_id: str
    quantile: float
    delay_minutes: int
    direction: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    initial_stop_price: float
    exit_reason: str
    hold_hours: float
    gross_return: float
    mfe: float
    mae: float
    score_long: float
    score_short: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("signal_time", "entry_time", "exit_time"):
            payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class SimulationResult:
    records: tuple[TradeRecord, ...]
    rejected_structural_risk: int


def score_thresholds(
    score_long: np.ndarray,
    score_short: np.ndarray,
    quantiles: Iterable[float],
) -> dict[float, tuple[float, float]]:
    valid_long = score_long[np.isfinite(score_long)]
    valid_short = score_short[np.isfinite(score_short)]
    if len(valid_long) == 0 or len(valid_short) == 0:
        raise RuntimeError("R03 calibration scores are empty")
    return {
        float(quantile): (
            float(np.quantile(valid_long, quantile)),
            float(np.quantile(valid_short, quantile)),
        )
        for quantile in quantiles
    }


def _signal_directions(
    score_long: np.ndarray,
    score_short: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    margin: float,
) -> np.ndarray:
    directions = np.zeros(len(score_long), dtype=np.int8)
    long_signal = (score_long >= long_threshold) & ((score_long - score_short) >= margin)
    short_signal = (score_short >= short_threshold) & ((score_short - score_long) >= margin)
    directions[long_signal] = 1
    directions[short_signal] = -1
    return directions


def _context_map(data: PeriodData) -> dict[str, np.ndarray]:
    return {
        name: data.context[:, index]
        for index, name in enumerate(data.context_columns)
    }


def _feature_index(feature_columns: tuple[str, ...], name: str) -> int:
    try:
        return feature_columns.index(name)
    except ValueError as exc:
        raise RuntimeError(f"R03 missing required dynamic-exit feature: {name}") from exc


def _initial_stop(
    direction: int,
    entry_price: float,
    context: dict[str, np.ndarray],
    signal_position: int,
    config: SwingBaselineConfig,
) -> tuple[float, float] | None:
    atr_abs = float(context["ctx_atr_abs_4h"][signal_position])
    if direction > 0:
        structural = float(context["ctx_recent_low_4h"][signal_position]) - config.structural_buffer_atr * atr_abs
        risk = (entry_price - structural) / entry_price
        if not np.isfinite(risk):
            return None
        if risk > config.max_initial_stop_pct:
            return None
        risk = max(config.min_initial_stop_pct, risk)
        return entry_price * (1.0 - risk), risk
    structural = float(context["ctx_recent_high_4h"][signal_position]) + config.structural_buffer_atr * atr_abs
    risk = (structural - entry_price) / entry_price
    if not np.isfinite(risk):
        return None
    if risk > config.max_initial_stop_pct:
        return None
    risk = max(config.min_initial_stop_pct, risk)
    return entry_price * (1.0 + risk), risk


def _future_exit_reason(
    direction: int,
    score_long: float,
    score_short: float,
    close_rel_ema20_1h: float,
    ema20_slope3_4h: float,
    long_threshold: float,
    short_threshold: float,
    margin: float,
) -> str | None:
    if direction > 0:
        if close_rel_ema20_1h < -0.003 and ema20_slope3_4h <= 0.0:
            return "trend_invalidation"
        if score_short >= short_threshold and (score_short - score_long) >= margin:
            return "model_reversal"
    else:
        if close_rel_ema20_1h > 0.003 and ema20_slope3_4h >= 0.0:
            return "trend_invalidation"
        if score_long >= long_threshold and (score_long - score_short) >= margin:
            return "model_reversal"
    return None


def simulate_structural_portfolio(
    *,
    fold_id: str,
    architecture: str,
    target: SwingTargetSpec,
    quantile: float,
    delay_minutes: int,
    period: PeriodData,
    feature_columns: tuple[str, ...],
    score_long: np.ndarray,
    score_short: np.ndarray,
    thresholds: tuple[float, float],
    market_path: MarketPath,
    config: SwingBaselineConfig,
) -> SimulationResult:
    """Simulate one position at a time with no fixed-time primary exit.

    A maximum holding time remains only as a safety cap.  Stops are checked
    before the current minute's high/low can update a trailing stop, preventing
    same-bar lookahead optimism.
    """
    directions = _signal_directions(
        score_long,
        score_short,
        thresholds[0],
        thresholds[1],
        config.probability_margin,
    )
    context = _context_map(period)
    rel_ema_idx = _feature_index(feature_columns, "tf1h_close_rel_ema20")
    slope_idx = _feature_index(feature_columns, "tf4h_ema20_slope3")
    decision_times = period.timestamps_ns
    path_times = market_path.times_ns
    records: list[TradeRecord] = []
    rejected = 0
    next_available_signal_ns = np.int64(-2**63)
    max_hold_ns = int(pd.Timedelta(hours=config.max_hold_hours).value)
    delay_ns = int(pd.Timedelta(minutes=delay_minutes).value)

    for signal_position in np.flatnonzero(directions):
        signal_ns = int(decision_times[signal_position])
        if signal_ns < next_available_signal_ns:
            continue
        direction = int(directions[signal_position])
        entry_ns = signal_ns + delay_ns
        entry_pos = int(np.searchsorted(path_times, entry_ns, side="left"))
        if entry_pos >= len(path_times):
            break
        entry_ns = int(path_times[entry_pos])
        entry_price = float(market_path.open[entry_pos])
        stop_info = _initial_stop(direction, entry_price, context, signal_position, config)
        if stop_info is None:
            rejected += 1
            continue
        active_stop, _ = stop_info
        initial_stop = active_stop
        stop_reason = "hard_or_structural_stop"
        peak = entry_price
        trough = entry_price
        max_mfe = 0.0
        max_mae = 0.0
        end_ns = min(entry_ns + max_hold_ns, int(path_times[-1]))
        end_pos = int(np.searchsorted(path_times, end_ns, side="right")) - 1
        decision_pointer = int(np.searchsorted(decision_times, signal_ns, side="right"))
        scheduled_exit_ns: int | None = None
        scheduled_reason: str | None = None
        exit_price = float(market_path.close[end_pos])
        exit_ns = int(path_times[end_pos])
        exit_reason = "max_hold" if end_ns < int(path_times[-1]) else "data_end"

        for minute_pos in range(entry_pos, end_pos + 1):
            minute_ns = int(path_times[minute_pos])
            while decision_pointer < len(decision_times) and int(decision_times[decision_pointer]) + delay_ns <= minute_ns:
                reason = _future_exit_reason(
                    direction,
                    float(score_long[decision_pointer]),
                    float(score_short[decision_pointer]),
                    float(period.full_x[decision_pointer, rel_ema_idx]),
                    float(period.full_x[decision_pointer, slope_idx]),
                    float(thresholds[0]),
                    float(thresholds[1]),
                    float(config.probability_margin),
                )
                if reason is not None and scheduled_exit_ns is None:
                    scheduled_exit_ns = int(decision_times[decision_pointer]) + delay_ns
                    scheduled_reason = reason
                decision_pointer += 1
            if scheduled_exit_ns is not None and minute_ns >= scheduled_exit_ns:
                exit_price = float(market_path.open[minute_pos])
                exit_ns = minute_ns
                exit_reason = str(scheduled_reason)
                break

            minute_high = float(market_path.high[minute_pos])
            minute_low = float(market_path.low[minute_pos])
            if direction > 0 and minute_low <= active_stop:
                exit_price = active_stop
                exit_ns = minute_ns
                exit_reason = stop_reason
                break
            if direction < 0 and minute_high >= active_stop:
                exit_price = active_stop
                exit_ns = minute_ns
                exit_reason = stop_reason
                break

            peak = max(peak, minute_high)
            trough = min(trough, minute_low)
            if direction > 0:
                max_mfe = max(max_mfe, peak / entry_price - 1.0)
                max_mae = max(max_mae, 1.0 - trough / entry_price)
            else:
                max_mfe = max(max_mfe, 1.0 - trough / entry_price)
                max_mae = max(max_mae, peak / entry_price - 1.0)

            decision_for_minute = min(
                len(decision_times) - 1,
                max(signal_position, decision_pointer - 1),
            )
            atr15 = float(context["ctx_atr_pct_15m"][decision_for_minute])
            atr4h = float(context["ctx_atr_pct_4h"][decision_for_minute])
            if max_mfe >= config.breakeven_trigger_pct:
                if direction > 0:
                    active_stop = max(active_stop, entry_price * (1.0 + config.base_round_trip_cost))
                else:
                    active_stop = min(active_stop, entry_price * (1.0 - config.base_round_trip_cost))
                stop_reason = "breakeven_stop"
            if max_mfe >= config.trailing_trigger_pct:
                trail_distance = float(
                    np.clip(
                        max(config.min_trailing_distance_pct, 1.5 * atr15),
                        config.min_trailing_distance_pct,
                        config.max_trailing_distance_pct,
                    )
                )
                if max_mfe >= config.strong_trend_trigger_pct:
                    trail_distance = float(
                        np.clip(
                            max(trail_distance, 0.8 * atr4h),
                            config.min_trailing_distance_pct,
                            config.max_trailing_distance_pct,
                        )
                    )
                if direction > 0:
                    active_stop = max(active_stop, peak * (1.0 - trail_distance))
                else:
                    active_stop = min(active_stop, trough * (1.0 + trail_distance))
                stop_reason = "trailing_stop"

        gross = exit_price / entry_price - 1.0 if direction > 0 else entry_price / exit_price - 1.0
        hold_hours = (exit_ns - entry_ns) / float(pd.Timedelta(hours=1).value)
        records.append(
            TradeRecord(
                fold_id=fold_id,
                architecture=architecture,
                target_id=target.target_id,
                quantile=float(quantile),
                delay_minutes=int(delay_minutes),
                direction="long" if direction > 0 else "short",
                signal_time=pd.Timestamp(signal_ns),
                entry_time=pd.Timestamp(entry_ns),
                exit_time=pd.Timestamp(exit_ns),
                entry_price=entry_price,
                exit_price=float(exit_price),
                initial_stop_price=float(initial_stop),
                exit_reason=exit_reason,
                hold_hours=float(hold_hours),
                gross_return=float(gross),
                mfe=float(max_mfe),
                mae=float(max_mae),
                score_long=float(score_long[signal_position]),
                score_short=float(score_short[signal_position]),
            )
        )
        next_available_signal_ns = np.int64(exit_ns + pd.Timedelta(minutes=config.decision_interval_minutes).value)
    return SimulationResult(records=tuple(records), rejected_structural_risk=rejected)


def summarize_records(
    result: SimulationResult,
    *,
    fold_id: str,
    architecture: str,
    target_id: str,
    quantile: float,
    delay_minutes: int,
    cost_multiplier: float,
    config: SwingBaselineConfig,
) -> dict[str, object]:
    records = result.records
    if not records:
        return {
            "fold_id": fold_id,
            "architecture": architecture,
            "target_id": target_id,
            "quantile": quantile,
            "delay_minutes": delay_minutes,
            "cost_multiplier": cost_multiplier,
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "win_rate": 0.0,
            "mean_gross_return": 0.0,
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "positive_month_ratio": 0.0,
            "positive_quarter_ratio": 0.0,
            "top5_removed_total_return": 0.0,
            "average_hold_hours": 0.0,
            "average_mfe": 0.0,
            "average_mae": 0.0,
            "rejected_structural_risk": result.rejected_structural_risk,
        }
    gross = np.asarray([record.gross_return for record in records], dtype=float)
    cost = config.base_round_trip_cost * float(cost_multiplier)
    net = gross - cost
    capital_curve = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.concatenate([[1.0], capital_curve]))[1:]
    drawdown = capital_curve / peak - 1.0
    wins = net[net > 0]
    losses = net[net <= 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("inf")
    exit_times = pd.DatetimeIndex([record.exit_time for record in records])
    series = pd.Series(net, index=exit_times)
    monthly = series.groupby(exit_times.to_period("M")).sum()
    quarterly = series.groupby(exit_times.to_period("Q")).sum()
    removed = net.copy()
    remove_count = min(5, len(removed))
    if remove_count:
        top = np.argpartition(removed, -remove_count)[-remove_count:]
        removed[top] = 0.0
    return {
        "fold_id": fold_id,
        "architecture": architecture,
        "target_id": target_id,
        "quantile": float(quantile),
        "delay_minutes": int(delay_minutes),
        "cost_multiplier": float(cost_multiplier),
        "trades": int(len(records)),
        "long_trades": int(sum(record.direction == "long" for record in records)),
        "short_trades": int(sum(record.direction == "short" for record in records)),
        "win_rate": float(np.mean(net > 0)),
        "mean_gross_return": float(np.mean(gross)),
        "mean_net_return": float(np.mean(net)),
        "median_net_return": float(np.median(net)),
        "profit_factor": profit_factor,
        "total_return": float(capital_curve[-1] - 1.0),
        "max_drawdown": float(np.min(drawdown)),
        "positive_month_ratio": float(np.mean(monthly > 0)) if len(monthly) else 0.0,
        "positive_quarter_ratio": float(np.mean(quarterly > 0)) if len(quarterly) else 0.0,
        "top5_removed_total_return": float(np.prod(1.0 + removed) - 1.0),
        "average_hold_hours": float(np.mean([record.hold_hours for record in records])),
        "average_mfe": float(np.mean([record.mfe for record in records])),
        "average_mae": float(np.mean([record.mae for record in records])),
        "rejected_structural_risk": int(result.rejected_structural_risk),
    }


def evaluate_prediction_scenarios(
    *,
    fold_id: str,
    architecture: str,
    target: SwingTargetSpec,
    period: PeriodData,
    feature_columns: tuple[str, ...],
    score_long: np.ndarray,
    score_short: np.ndarray,
    thresholds_by_quantile: dict[float, tuple[float, float]],
    market_path: MarketPath,
    config: SwingBaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    base_trade_rows: list[dict[str, object]] = []
    for quantile, thresholds in thresholds_by_quantile.items():
        for delay in config.delay_scenarios_minutes:
            simulation = simulate_structural_portfolio(
                fold_id=fold_id,
                architecture=architecture,
                target=target,
                quantile=quantile,
                delay_minutes=delay,
                period=period,
                feature_columns=feature_columns,
                score_long=score_long,
                score_short=score_short,
                thresholds=thresholds,
                market_path=market_path,
                config=config,
            )
            for cost_multiplier in config.cost_stress_multipliers:
                summaries.append(
                    summarize_records(
                        simulation,
                        fold_id=fold_id,
                        architecture=architecture,
                        target_id=target.target_id,
                        quantile=quantile,
                        delay_minutes=delay,
                        cost_multiplier=cost_multiplier,
                        config=config,
                    )
                )
            if delay == config.execution_delay_minutes:
                base_trade_rows.extend(record.to_dict() for record in simulation.records)
    return pd.DataFrame(summaries), pd.DataFrame(base_trade_rows)
