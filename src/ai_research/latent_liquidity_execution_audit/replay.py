#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal 1-second replay for fixed post-release confirmation rules."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from types import SimpleNamespace

import numpy as np
import pandas as pd

from .config import StablePathExecutionAuditConfig
from src.ai_research.latent_liquidity_path_atlas.candidates import normalize_second_bars
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter


@dataclass
class ReplayResult:
    aligned_price_path: pd.DataFrame
    aligned_flow_path: pd.DataFrame
    confirmation_events: pd.DataFrame
    replay_quality: pd.DataFrame


def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).astype("datetime64[ns]")
    out = out.loc[~out.index.isna()].sort_index(kind="mergesort")
    out = out.loc[~out.index.duplicated(keep="last")]
    return out


def _confirmation_no_new_extreme(path: pd.DataFrame, side: str, seconds: int, max_seconds: int) -> int | None:
    lows = path["low"].to_numpy(dtype=float)
    highs = path["high"].to_numpy(dtype=float)
    last_extreme = 0
    best = lows[0] if side == "DOWN" else highs[0]
    stop = min(len(path) - 1, int(max_seconds))
    for i in range(1, stop + 1):
        value = lows[i] if side == "DOWN" else highs[i]
        improved = value < best if side == "DOWN" else value > best
        if improved:
            best = value
            last_extreme = i
        if i - last_extreme >= int(seconds):
            return i
    return None


def _confirmation_reclaim(path: pd.DataFrame, side: str, threshold_bp: float, max_seconds: int) -> int | None:
    lows = path["low"].to_numpy(dtype=float)
    highs = path["high"].to_numpy(dtype=float)
    close = path["close"].to_numpy(dtype=float)
    stop = min(len(path) - 1, int(max_seconds))
    extreme = lows[0] if side == "DOWN" else highs[0]
    for i in range(1, stop + 1):
        if side == "DOWN":
            extreme = min(extreme, lows[i])
            if close[i] >= extreme * (1.0 + threshold_bp / 1e4):
                return i
        else:
            extreme = max(extreme, highs[i])
            if close[i] <= extreme * (1.0 - threshold_bp / 1e4):
                return i
    return None


def _confirmation_second_push_failure(path: pd.DataFrame, side: str, config: StablePathExecutionAuditConfig) -> int | None:
    lows = path["low"].to_numpy(dtype=float)
    highs = path["high"].to_numpy(dtype=float)
    close = path["close"].to_numpy(dtype=float)
    stop = min(len(path) - 1, int(config.max_confirmation_seconds))
    extreme = lows[0] if side == "DOWN" else highs[0]
    state = "SEEK_REBOUND"
    retest_extreme = extreme
    for i in range(1, stop + 1):
        if side == "DOWN":
            if lows[i] < extreme:
                extreme = lows[i]
                state = "SEEK_REBOUND"
            rebound_bp = (close[i] / extreme - 1.0) * 1e4
            if state == "SEEK_REBOUND" and rebound_bp >= config.second_push_rebound_bp:
                state = "SEEK_RETEST"
            elif state == "SEEK_RETEST":
                distance_bp = (lows[i] / extreme - 1.0) * 1e4
                if distance_bp <= config.second_push_retest_tolerance_bp:
                    if lows[i] < extreme * (1.0 - config.second_push_new_extreme_tolerance_bp / 1e4):
                        extreme = lows[i]
                        state = "SEEK_REBOUND"
                    else:
                        retest_extreme = min(extreme, lows[i])
                        state = "SEEK_SECOND_REBOUND"
            elif state == "SEEK_SECOND_REBOUND":
                if lows[i] < retest_extreme * (1.0 - config.second_push_new_extreme_tolerance_bp / 1e4):
                    extreme = lows[i]
                    state = "SEEK_REBOUND"
                elif (close[i] / retest_extreme - 1.0) * 1e4 >= config.second_push_rebound_bp:
                    return i
        else:
            if highs[i] > extreme:
                extreme = highs[i]
                state = "SEEK_REBOUND"
            rebound_bp = (extreme / close[i] - 1.0) * 1e4
            if state == "SEEK_REBOUND" and rebound_bp >= config.second_push_rebound_bp:
                state = "SEEK_RETEST"
            elif state == "SEEK_RETEST":
                distance_bp = (extreme / highs[i] - 1.0) * 1e4
                if distance_bp <= config.second_push_retest_tolerance_bp:
                    if highs[i] > extreme * (1.0 + config.second_push_new_extreme_tolerance_bp / 1e4):
                        extreme = highs[i]
                        state = "SEEK_REBOUND"
                    else:
                        retest_extreme = max(extreme, highs[i])
                        state = "SEEK_SECOND_REBOUND"
            elif state == "SEEK_SECOND_REBOUND":
                if highs[i] > retest_extreme * (1.0 + config.second_push_new_extreme_tolerance_bp / 1e4):
                    extreme = highs[i]
                    state = "SEEK_REBOUND"
                elif (retest_extreme / close[i] - 1.0) * 1e4 >= config.second_push_rebound_bp:
                    return i
    return None


def confirmation_offsets(path_after_event: pd.DataFrame, side: str, config: StablePathExecutionAuditConfig) -> dict[str, int | None]:
    rules: dict[str, int | None] = {
        f"NO_NEW_EXTREME_{config.stabilization_seconds}S": _confirmation_no_new_extreme(
            path_after_event, side, config.stabilization_seconds, config.max_confirmation_seconds
        ),
        "SECOND_PUSH_FAILURE": _confirmation_second_push_failure(path_after_event, side, config),
    }
    for threshold in config.reclaim_thresholds_bp:
        key = int(threshold) if float(threshold).is_integer() else str(threshold).replace(".", "P")
        rules[f"RECLAIM_{key}BP"] = _confirmation_reclaim(
            path_after_event, side, threshold, config.max_confirmation_seconds
        )
    return rules


def _running_extreme_until(path_after_event: pd.DataFrame, side: str, position: int) -> float:
    subset = path_after_event.iloc[: position + 1]
    return float(subset["low"].min()) if side == "DOWN" else float(subset["high"].max())


def _first_barrier_result(
    future: pd.DataFrame,
    side: str,
    stop_price: float,
    target_price: float,
) -> str:
    for row in future.itertuples(index=False):
        if side == "DOWN":  # LONG after a down release
            stop_hit = float(row.low) <= stop_price
            target_hit = float(row.high) >= target_price
        else:  # SHORT after an up release
            stop_hit = float(row.high) >= stop_price
            target_hit = float(row.low) <= target_price
        if stop_hit:
            return "STOP"  # conservative if both occur in the same 1s bar
        if target_hit:
            return "TARGET"
    return "NONE"


def _simulate_confirmation(
    path_after_event: pd.DataFrame,
    side: str,
    confirmation_pos: int,
    delay_seconds: int,
    cost_bp: float,
    horizon_seconds: int,
    config: StablePathExecutionAuditConfig,
) -> dict[str, object] | None:
    entry_pos = confirmation_pos + int(delay_seconds)
    if entry_pos >= len(path_after_event):
        return None
    entry_price = float(path_after_event.iloc[entry_pos]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    extreme = _running_extreme_until(path_after_event, side, confirmation_pos)
    if side == "DOWN":
        stop_price = extreme * (1.0 - config.structural_stop_buffer_bp / 1e4)
        risk_bp = (entry_price / stop_price - 1.0) * 1e4
        if risk_bp <= 0:
            return None
        target_1r = entry_price * (1.0 + risk_bp / 1e4)
        target_2r = entry_price * (1.0 + 2.0 * risk_bp / 1e4)
    else:
        stop_price = extreme * (1.0 + config.structural_stop_buffer_bp / 1e4)
        risk_bp = (stop_price / entry_price - 1.0) * 1e4
        if risk_bp <= 0:
            return None
        target_1r = entry_price * (1.0 - risk_bp / 1e4)
        target_2r = entry_price * (1.0 - 2.0 * risk_bp / 1e4)
    end_pos = min(len(path_after_event), entry_pos + int(horizon_seconds))
    if end_pos <= entry_pos + 1:
        return None
    future = path_after_event.iloc[entry_pos:end_pos]
    if side == "DOWN":
        mfe_bp = (float(future["high"].max()) / entry_price - 1.0) * 1e4
        mae_bp = (1.0 - float(future["low"].min()) / entry_price) * 1e4
        terminal_gross_bp = (float(future.iloc[-1]["close"]) / entry_price - 1.0) * 1e4
        stop_return_bp = (stop_price / entry_price - 1.0) * 1e4
    else:
        mfe_bp = (1.0 - float(future["low"].min()) / entry_price) * 1e4
        mae_bp = (float(future["high"].max()) / entry_price - 1.0) * 1e4
        terminal_gross_bp = (1.0 - float(future.iloc[-1]["close"]) / entry_price) * 1e4
        stop_return_bp = (1.0 - stop_price / entry_price) * 1e4
    stop_vs_1r = _first_barrier_result(future, side, stop_price, target_1r)
    stop_vs_2r = _first_barrier_result(future, side, stop_price, target_2r)
    if side == "DOWN":
        stopped = bool(future["low"].le(stop_price).any())
    else:
        stopped = bool(future["high"].ge(stop_price).any())
    realized_gross_bp = stop_return_bp if stopped else terminal_gross_bp
    return {
        "entry_pos_seconds": int(entry_pos + 1),
        "entry_price": entry_price,
        "structural_extreme": extreme,
        "stop_price": stop_price,
        "stop_distance_bp": float(risk_bp),
        "mfe_bp": float(mfe_bp),
        "mae_bp": float(mae_bp),
        "mfe_r": float(mfe_bp / risk_bp),
        "mae_r": float(mae_bp / risk_bp),
        "one_r_before_stop": stop_vs_1r == "TARGET",
        "two_r_before_stop": stop_vs_2r == "TARGET",
        "stopped_before_horizon": stopped,
        "terminal_gross_bp": float(terminal_gross_bp),
        "realized_gross_bp": float(realized_gross_bp),
        "net_return_bp": float(realized_gross_bp - cost_bp),
    }


def _safe_columnwise_nanquantile(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Columnwise nanquantile without emitting all-NaN slice warnings.

    All-NaN offsets are legitimate for derived metrics such as one-second impact
    at the first replay point.  Preserve them as NaN and only call NumPy on
    columns that contain at least one finite value.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a 2D replay matrix")
    result = np.full((len(quantiles), array.shape[1]), np.nan, dtype=float)
    valid_columns = np.isfinite(array).any(axis=0)
    if valid_columns.any():
        result[:, valid_columns] = np.nanquantile(array[:, valid_columns], quantiles, axis=0)
    return result


def _aggregate_quantile_paths(path_store: dict[tuple[object, ...], list[dict[str, np.ndarray]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    for key, records in path_store.items():
        cluster, side, period = key
        if not records:
            continue
        names = records[0].keys()
        stacked = {name: np.vstack([record[name] for record in records]) for name in names}
        offsets = stacked["offset_seconds"][0].astype(int)
        price_metrics = ("release_direction_close_bp", "release_direction_extreme_bp")
        flow_metrics = ("notional_intensity", "trades_intensity", "release_aligned_delta_share", "impact_bp_per_million")
        for metric in price_metrics:
            values = stacked[metric]
            q_values = _safe_columnwise_nanquantile(values, quantiles)
            for pos, offset in enumerate(offsets):
                row = {"path_cluster": cluster, "event_side": side, "period": period, "offset_seconds": int(offset), "metric": metric, "episodes": len(records)}
                for q, q_array in zip(quantiles, q_values):
                    row[f"q{int(q * 100):02d}"] = float(q_array[pos])
                price_rows.append(row)
        for metric in flow_metrics:
            values = stacked[metric]
            q_values = _safe_columnwise_nanquantile(values, quantiles)
            for pos, offset in enumerate(offsets):
                row = {"path_cluster": cluster, "event_side": side, "period": period, "offset_seconds": int(offset), "metric": metric, "episodes": len(records)}
                for q, q_array in zip(quantiles, q_values):
                    row[f"q{int(q * 100):02d}"] = float(q_array[pos])
                flow_rows.append(row)
    return pd.DataFrame(price_rows), pd.DataFrame(flow_rows)


def replay_samples(
    samples: pd.DataFrame,
    config: StablePathExecutionAuditConfig,
    *,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    progress: bool = True,
) -> ReplayResult:
    if samples.empty:
        return ReplayResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1s", data_dir=data_dir, db_name=db_name)
    work = samples.copy()
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    work = work.loc[work["event_time"].notna()].drop_duplicates("event_id")
    work["event_day"] = work["event_time"].dt.floor("D")
    day_groups = list(work.groupby("event_day", sort=True))
    reporter = ProgressReporter(
        label="[latent-liquidity-r01.2] causal 1s replay days",
        total=len(day_groups),
        every=1,
        enabled=progress,
    )
    path_store: dict[tuple[object, ...], list[dict[str, np.ndarray]]] = {}
    confirmation_rows: list[dict[str, object]] = []
    requested = len(work)
    complete = 0
    missing = 0
    requested_by_stratum = Counter(
        (int(row.path_cluster), str(row.event_side), str(row.period)) for row in work.itertuples(index=False)
    )
    complete_by_stratum: Counter[tuple[int, str, str]] = Counter()
    missing_by_stratum: Counter[tuple[int, str, str]] = Counter()
    for day_number, (_, day_events) in enumerate(day_groups, start=1):
        fill_pad = int(config.replay_max_fill_gap_seconds)
        load_start = day_events["event_time"].min() - pd.Timedelta(seconds=config.pre_replay_seconds + fill_pad)
        load_end = day_events["event_time"].max() + pd.Timedelta(
            seconds=config.post_replay_seconds + max(config.entry_delay_seconds) + fill_pad
        )
        bars = loader.fetch_data_by_date_range(load_start, load_end, build_missing=False, force_rebuild=False, cvd_mode="range")
        bars = _normalize_bars(bars)
        if not bars.empty:
            bars = normalize_second_bars(
                bars,
                SimpleNamespace(max_fill_gap_seconds=config.replay_max_fill_gap_seconds),
            )
        for event in day_events.itertuples(index=False):
            event_time = pd.Timestamp(event.event_time)
            expected = pd.date_range(
                event_time - pd.Timedelta(seconds=config.pre_replay_seconds),
                event_time + pd.Timedelta(seconds=config.post_replay_seconds),
                freq="1s",
            )
            path = bars.reindex(expected)
            required = ["open", "high", "low", "close", "notional", "trades_count", "delta_notional"]
            has_required_gap = path[required].isna().any().any()
            crosses_unsafe_gap = bool(path.get("unsafe_gap", pd.Series(1, index=path.index)).fillna(1).astype(bool).any())
            if has_required_gap or crosses_unsafe_gap:
                missing += 1
                missing_by_stratum[(int(event.path_cluster), str(event.event_side), str(event.period))] += 1
                continue
            complete += 1
            complete_by_stratum[(int(event.path_cluster), str(event.event_side), str(event.period))] += 1
            event_pos = config.pre_replay_seconds
            event_ref = float(event.event_reference_price)
            side = str(event.event_side)
            close = path["close"].to_numpy(dtype=float)
            low = path["low"].to_numpy(dtype=float)
            high = path["high"].to_numpy(dtype=float)
            if side == "DOWN":
                signed_close = (event_ref - close) / event_ref * 1e4
                signed_extreme = (event_ref - low) / event_ref * 1e4
                aligned_delta = -path["delta_notional"].to_numpy(dtype=float)
            else:
                signed_close = (close - event_ref) / event_ref * 1e4
                signed_extreme = (high - event_ref) / event_ref * 1e4
                aligned_delta = path["delta_notional"].to_numpy(dtype=float)
            pre_slice = slice(0, event_pos)
            notional = path["notional"].to_numpy(dtype=float)
            trades = path["trades_count"].to_numpy(dtype=float)
            base_notional = max(float(np.nanmedian(notional[pre_slice])), 1e-9)
            base_trades = max(float(np.nanmedian(trades[pre_slice])), 1e-9)
            notional_million = np.maximum(notional / 1_000_000.0, 1e-6)
            one_second_move_bp = np.r_[np.nan, np.abs(np.diff(close) / close[:-1]) * 1e4]
            key = (int(event.path_cluster), side, str(event.period))
            path_store.setdefault(key, []).append(
                {
                    "offset_seconds": np.arange(-config.pre_replay_seconds, config.post_replay_seconds + 1, dtype=np.int16),
                    "release_direction_close_bp": signed_close.astype(np.float32),
                    "release_direction_extreme_bp": signed_extreme.astype(np.float32),
                    "notional_intensity": (notional / base_notional).astype(np.float32),
                    "trades_intensity": (trades / base_trades).astype(np.float32),
                    "release_aligned_delta_share": (aligned_delta / np.maximum(notional, 1e-9)).astype(np.float32),
                    "impact_bp_per_million": (one_second_move_bp / notional_million).astype(np.float32),
                }
            )
            after = path.iloc[event_pos + 1 :].reset_index(drop=True)
            rules = confirmation_offsets(after, side, config)
            for rule_name, confirmation_pos in rules.items():
                if confirmation_pos is None:
                    confirmation_rows.append(
                        {
                            "event_id": event.event_id,
                            "path_cluster": int(event.path_cluster),
                            "event_side": side,
                            "period": str(event.period),
                            "rule": rule_name,
                            "detected": False,
                        }
                    )
                    continue
                for delay in config.entry_delay_seconds:
                    for cost_multiple in config.cost_multipliers:
                        cost_bp = config.roundtrip_cost_bp * cost_multiple
                        for horizon in config.terminal_horizons_seconds:
                            result = _simulate_confirmation(
                                after,
                                side,
                                int(confirmation_pos),
                                int(delay),
                                float(cost_bp),
                                int(horizon),
                                config,
                            )
                            if result is None:
                                continue
                            confirmation_rows.append(
                                {
                                    "event_id": event.event_id,
                                    "path_cluster": int(event.path_cluster),
                                    "event_side": side,
                                    "period": str(event.period),
                                    "rule": rule_name,
                                    "detected": True,
                                    "confirmation_seconds": int(confirmation_pos + 1),
                                    "entry_delay_seconds": int(delay),
                                    "cost_multiple": float(cost_multiple),
                                    "roundtrip_cost_bp": float(cost_bp),
                                    "horizon_seconds": int(horizon),
                                    **result,
                                }
                            )
        reporter.update(day_number)
    reporter.close()
    price_path, flow_path = _aggregate_quantile_paths(path_store)
    confirmation = pd.DataFrame(confirmation_rows)
    quality_rows: list[dict[str, object]] = [
        {"check": "requested_replay_episodes", "value": requested, "status": "INFO"},
        {"check": "complete_replay_episodes", "value": complete, "status": "PASS" if complete > 0 else "FAIL"},
        {"check": "missing_replay_episodes", "value": missing, "status": "WARN" if missing else "PASS"},
        {
            "check": "replay_completion_rate",
            "value": complete / requested if requested else 0.0,
            "status": "PASS" if requested and complete / requested >= 0.95 else "WARN",
        },
    ]
    for key in sorted(requested_by_stratum):
        cluster, side, period = key
        stratum_requested = int(requested_by_stratum[key])
        stratum_complete = int(complete_by_stratum[key])
        rate = stratum_complete / stratum_requested if stratum_requested else 0.0
        quality_rows.append(
            {
                "check": f"completion_rate_cluster_{cluster}_{side}_{period}",
                "value": rate,
                "status": "PASS" if rate >= 0.95 else "WARN",
            }
        )
    quality = pd.DataFrame(quality_rows)
    return ReplayResult(price_path, flow_path, confirmation, quality)
