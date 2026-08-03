#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal feature modules for R13 supervised meta-labeling.

All joins are backward-looking at ``decision_time``.  The module builders are
chunked so multi-year 1-second and footprint data are never loaded in one shot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.post_sweep_footprint_books.config import PostSweepFootprintBooksConfig
from src.research_common.post_sweep_footprint_books.footprint import build_footprint_context
from src.research_common.post_sweep_micro.universe import load_binance_oi_context
from src.research_common.progress import ProgressReporter

from .config import PostSweepSupervisedConfig

EPS = 1e-12
TRADE_FLOW_COLUMNS: tuple[str, ...] = (
    "notional", "buy_notional", "sell_notional", "delta_notional", "trades_count",
    "buy_trades_count", "sell_trades_count", "large_buy_notional", "large_sell_notional",
    "large_delta_notional", "large_buy_trades_count", "large_sell_trades_count",
    "large_trades_count",
)


@dataclass(frozen=True)
class FeatureModuleResult:
    name: str
    features: pd.DataFrame
    audit: pd.DataFrame


def checkpoint_index(datasets: dict[int, pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "checkpoint_id", "zone_event_id", "checkpoint_minutes", "decision_time",
        "event_available_time", "event_bar_time", "period", "split",
    ]
    parts = []
    for frame in datasets.values():
        available = [name for name in columns if name in frame.columns]
        part = frame.loc[:, available].copy()
        for name in columns:
            if name not in part.columns:
                part[name] = pd.NaT if name.endswith("_time") else np.nan
        parts.append(part.loc[:, columns])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    out["checkpoint_id"] = out["checkpoint_id"].astype(str)
    for name in ("decision_time", "event_available_time", "event_bar_time"):
        out[name] = pd.to_datetime(out[name], errors="coerce")
    if out["checkpoint_id"].duplicated().any():
        raise RuntimeError("checkpoint_index contains duplicate checkpoint_id")
    return out.sort_values(["decision_time", "checkpoint_id"], kind="mergesort").reset_index(drop=True)


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cursor = start.normalize()
    final = end.normalize() + pd.Timedelta(days=1)
    delta = pd.Timedelta(days=max(1, int(days)))
    while cursor < final:
        nxt = min(cursor + delta, final)
        yield cursor, nxt
        cursor = nxt


def _prefix(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values) + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(np.nan_to_num(values, nan=0.0), out=out[1:])
    return out


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= EPS:
        return np.nan
    return float(a / b)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric Series aligned to ``frame`` even when a column is absent.

    ``DataFrame.get`` returns ``None`` for a missing column and
    ``pd.to_numeric(None)`` becomes a scalar ``numpy.float64``.  Downstream
    vector operations such as ``.notna()``/``.gt()`` then fail.  R13 feature
    modules deliberately tolerate partially populated report schemas, so every
    optional numeric input must retain Series semantics.
    """
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float, name=name)
    values = pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(values.to_numpy(dtype=float, copy=False), index=frame.index, name=name)


def _trade_interval_stats(
    *,
    prefix: str,
    start_ns: int,
    end_ns: int,
    time_ns: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    prefixes: dict[str, np.ndarray],
    max_trade: np.ndarray,
) -> dict[str, object]:
    left = int(np.searchsorted(time_ns, start_ns, side="left"))
    right = int(np.searchsorted(time_ns, end_ns, side="left"))
    result: dict[str, object] = {
        f"{prefix}_present": False,
        f"{prefix}_observed_seconds": 0,
        f"{prefix}_return_bp": np.nan,
        f"{prefix}_range_bp": np.nan,
        f"{prefix}_close_off_low_bp": np.nan,
        f"{prefix}_delta_ratio": np.nan,
        f"{prefix}_sell_share": np.nan,
        f"{prefix}_large_sell_share": np.nan,
        f"{prefix}_downside_bp_per_sell_million": np.nan,
        f"{prefix}_upside_bp_per_buy_million": np.nan,
        f"{prefix}_notional": np.nan,
        f"{prefix}_trades_count": np.nan,
        f"{prefix}_max_trade_notional": np.nan,
    }
    if right <= left:
        return result
    entry = float(open_[left])
    terminal = float(close[right - 1])
    hi = float(np.nanmax(high[left:right]))
    lo = float(np.nanmin(low[left:right]))
    notional = float(prefixes["notional"][right] - prefixes["notional"][left])
    buy = float(prefixes["buy_notional"][right] - prefixes["buy_notional"][left])
    sell = float(prefixes["sell_notional"][right] - prefixes["sell_notional"][left])
    delta = float(prefixes["delta_notional"][right] - prefixes["delta_notional"][left])
    large_sell = float(prefixes["large_sell_notional"][right] - prefixes["large_sell_notional"][left])
    trades = float(prefixes["trades_count"][right] - prefixes["trades_count"][left])
    return_bp = (terminal / entry - 1.0) * 10_000.0 if entry > 0 else np.nan
    range_bp = (hi / lo - 1.0) * 10_000.0 if lo > 0 else np.nan
    close_off_low = (terminal / lo - 1.0) * 10_000.0 if lo > 0 else np.nan
    downside = max(0.0, -return_bp) if np.isfinite(return_bp) else np.nan
    upside = max(0.0, return_bp) if np.isfinite(return_bp) else np.nan
    result.update({
        f"{prefix}_present": True,
        f"{prefix}_observed_seconds": int(right - left),
        f"{prefix}_return_bp": return_bp,
        f"{prefix}_range_bp": range_bp,
        f"{prefix}_close_off_low_bp": close_off_low,
        f"{prefix}_delta_ratio": _safe_ratio(delta, notional),
        f"{prefix}_sell_share": _safe_ratio(sell, buy + sell),
        f"{prefix}_large_sell_share": _safe_ratio(large_sell, sell),
        f"{prefix}_downside_bp_per_sell_million": _safe_ratio(downside, sell / 1_000_000.0),
        f"{prefix}_upside_bp_per_buy_million": _safe_ratio(upside, buy / 1_000_000.0),
        f"{prefix}_notional": notional,
        f"{prefix}_trades_count": trades,
        f"{prefix}_max_trade_notional": float(np.nanmax(max_trade[left:right])) if right > left else np.nan,
    })
    return result


def build_trade_1s_features(
    checkpoints: pd.DataFrame,
    *,
    symbol: str,
    data_dir: str | Path | None,
    db_name: str,
    config: PostSweepSupervisedConfig,
    progress: bool = True,
) -> FeatureModuleResult:
    cfg = config.validate()
    if checkpoints.empty:
        return FeatureModuleResult("trade_1s", pd.DataFrame(), pd.DataFrame())
    source = checkpoints.copy()
    source["decision_time"] = pd.to_datetime(source["decision_time"], errors="coerce")
    source["event_available_time"] = pd.to_datetime(source["event_available_time"], errors="coerce")
    source = source.dropna(subset=["checkpoint_id", "decision_time", "event_available_time"])
    chunks = list(_date_chunks(source["decision_time"].min(), source["decision_time"].max(), cfg.trade_chunk_days))
    loader = OKXTradeBarLoader(symbol=symbol, timeframe="1s", data_dir=data_dir, db_name=db_name)
    reporter = ProgressReporter("[r13] 1s trade features", len(chunks), every=1, enabled=progress)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    recent_windows = (5, 15, 30, 60)
    phases = (
        ("trade_pre_sweep_60s", -60, 0),
        ("trade_post_sweep_0_60s", 0, 60),
        ("trade_post_sweep_60_180s", 60, 180),
        ("trade_post_sweep_180_300s", 180, 300),
        ("trade_post_sweep_300_600s", 300, 600),
    )
    for chunk_no, (core_start, core_end) in enumerate(chunks, start=1):
        subset = source.loc[source["decision_time"].between(core_start, core_end, inclusive="left")].copy()
        if subset.empty:
            reporter.update(chunk_no)
            continue
        query_start = min(subset["event_available_time"].min() - pd.Timedelta(seconds=60), core_start - pd.Timedelta(seconds=60))
        query_end = subset["decision_time"].max() - pd.Timedelta(microseconds=1)
        bars = loader.load_local_data(query_start, query_end)
        rows: list[dict[str, object]] = []
        if bars.empty:
            rows = [{"checkpoint_id": value, "trade1s_causal_valid": False} for value in subset["checkpoint_id"].astype(str)]
            audits.append({"core_start": core_start, "core_end": core_end, "events": len(subset), "rows_1s": 0, "coverage": 0.0, "status": "missing_cache"})
        else:
            bars = bars.sort_index(kind="mergesort")
            time_ns = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            arrays = {
                name: pd.to_numeric(bars.get(name), errors="coerce").to_numpy(dtype=float)
                for name in ("open", "high", "low", "close", *TRADE_FLOW_COLUMNS, "max_trade_notional")
            }
            prefixes = {name: _prefix(arrays[name]) for name in TRADE_FLOW_COLUMNS}
            for item in subset.itertuples(index=False):
                record: dict[str, object] = {"checkpoint_id": str(item.checkpoint_id)}
                decision = pd.Timestamp(item.decision_time)
                event_available = pd.Timestamp(item.event_available_time)
                decision_ns = int(decision.value)
                for seconds in recent_windows:
                    record.update(_trade_interval_stats(
                        prefix=f"trade_recent_{seconds}s",
                        start_ns=int((decision - pd.Timedelta(seconds=seconds)).value),
                        end_ns=decision_ns,
                        time_ns=time_ns,
                        open_=arrays["open"], high=arrays["high"], low=arrays["low"], close=arrays["close"],
                        prefixes=prefixes, max_trade=arrays["max_trade_notional"],
                    ))
                for prefix, start_offset, end_offset in phases:
                    phase_end = event_available + pd.Timedelta(seconds=end_offset)
                    if phase_end <= decision:
                        record.update(_trade_interval_stats(
                            prefix=prefix,
                            start_ns=int((event_available + pd.Timedelta(seconds=start_offset)).value),
                            end_ns=int(phase_end.value),
                            time_ns=time_ns,
                            open_=arrays["open"], high=arrays["high"], low=arrays["low"], close=arrays["close"],
                            prefixes=prefixes, max_trade=arrays["max_trade_notional"],
                        ))
                    else:
                        # Keep a stable schema without exposing a partially observed phase.
                        record.update(_trade_interval_stats(
                            prefix=prefix,
                            start_ns=0, end_ns=0, time_ns=np.empty(0, dtype=np.int64),
                            open_=np.empty(0), high=np.empty(0), low=np.empty(0), close=np.empty(0),
                            prefixes={name: np.zeros(1) for name in TRADE_FLOW_COLUMNS},
                            max_trade=np.empty(0),
                        ))
                latest = int(np.searchsorted(time_ns, decision_ns, side="left") - 1)
                latest_time = pd.Timestamp(time_ns[latest]) if latest >= 0 else pd.NaT
                record["trade1s_latest_bar_time"] = latest_time
                record["trade1s_causal_valid"] = bool(latest >= 0 and latest_time < decision)
                record["trade1s_recent60_observed_share"] = float(record.get("trade_recent_60s_observed_seconds", 0)) / 60.0
                rows.append(record)
            part = pd.DataFrame(rows)
            coverage = float(part["trade1s_causal_valid"].fillna(False).mean()) if len(part) else 0.0
            audits.append({"core_start": core_start, "core_end": core_end, "events": len(subset), "rows_1s": len(bars), "coverage": coverage, "status": "complete"})
        outputs.append(pd.DataFrame(rows))
        reporter.update(chunk_no)
    reporter.close()
    features = pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()
    if not features.empty and features["checkpoint_id"].duplicated().any():
        raise RuntimeError("duplicate checkpoint_id in 1s trade features")
    return FeatureModuleResult("trade_1s", features, pd.DataFrame(audits))


def _range_summary(prefix: str, local: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        f"{prefix}_count": int(len(local)),
        f"{prefix}_direction_sum": np.nan,
        f"{prefix}_down_share": np.nan,
        f"{prefix}_duration_mean": np.nan,
        f"{prefix}_delta_ratio": np.nan,
        f"{prefix}_sell_share": np.nan,
        f"{prefix}_notional": np.nan,
        f"{prefix}_max_trade_notional": np.nan,
    }
    if local.empty:
        return result
    direction = pd.to_numeric(local["direction"], errors="coerce")
    notional = pd.to_numeric(local["notional"], errors="coerce").sum()
    sell = pd.to_numeric(local["sell_notional"], errors="coerce").sum()
    buy = pd.to_numeric(local["buy_notional"], errors="coerce").sum()
    delta = pd.to_numeric(local["delta_notional"], errors="coerce").sum()
    result.update({
        f"{prefix}_direction_sum": float(direction.sum()),
        f"{prefix}_down_share": float((direction < 0).mean()),
        f"{prefix}_duration_mean": float(pd.to_numeric(local["duration_seconds"], errors="coerce").mean()),
        f"{prefix}_delta_ratio": _safe_ratio(float(delta), float(notional)),
        f"{prefix}_sell_share": _safe_ratio(float(sell), float(buy + sell)),
        f"{prefix}_notional": float(notional),
        f"{prefix}_max_trade_notional": float(pd.to_numeric(local["max_trade_notional"], errors="coerce").max()),
    })
    return result


def build_range_features(
    checkpoints: pd.DataFrame,
    *,
    symbol: str,
    data_dir: str | Path | None,
    db_name: str,
    config: PostSweepSupervisedConfig,
    progress: bool = True,
) -> FeatureModuleResult:
    cfg = config.validate()
    source = checkpoints.dropna(subset=["checkpoint_id", "decision_time", "event_available_time"]).copy()
    if source.empty:
        return FeatureModuleResult("range_r0020", pd.DataFrame(), pd.DataFrame())
    chunks = list(_date_chunks(source["decision_time"].min(), source["decision_time"].max(), cfg.range_chunk_days))
    loader = OKXRangeBarLoader(symbol=symbol, range_pct=0.0020, data_dir=data_dir, db_name=db_name, initialize_db=False)
    reporter = ProgressReporter("[r13] r0020 range features", len(chunks), every=1, enabled=progress)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    columns = (
        "bar_id", "end_ts", "direction", "duration_seconds", "open", "close",
        "notional", "buy_notional", "sell_notional", "delta_notional", "max_trade_notional",
    )
    for chunk_no, (core_start, core_end) in enumerate(chunks, start=1):
        subset = source.loc[source["decision_time"].between(core_start, core_end, inclusive="left")].copy()
        if subset.empty:
            reporter.update(chunk_no)
            continue
        bars = loader.load_local_data(core_start - pd.Timedelta(days=2), core_end, columns=columns)
        rows: list[dict[str, object]] = []
        if bars.empty:
            rows = [{"checkpoint_id": str(value), "range_causal_valid": False} for value in subset["checkpoint_id"]]
            audits.append({"core_start": core_start, "core_end": core_end, "events": len(subset), "range_rows": 0, "coverage": 0.0, "status": "missing_cache"})
        else:
            local = bars.reset_index(drop=True).sort_values(["end_ts", "bar_id"], kind="mergesort")
            end_ns = pd.to_datetime(local["end_ts"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64)
            direction_values = pd.to_numeric(local["direction"], errors="coerce").to_numpy(dtype=float)
            down_positions = np.flatnonzero(direction_values < 0)
            open_values = pd.to_numeric(local["open"], errors="coerce").to_numpy(dtype=float)
            close_values = pd.to_numeric(local["close"], errors="coerce").to_numpy(dtype=float)
            sell_values = pd.to_numeric(local["sell_notional"], errors="coerce").to_numpy(dtype=float)
            duration_values = pd.to_numeric(local["duration_seconds"], errors="coerce").to_numpy(dtype=float)
            downside_bp = np.full(len(local), np.nan, dtype=float)
            valid_price = np.isfinite(open_values) & np.isfinite(close_values) & (open_values > 0)
            downside_bp[valid_price] = np.maximum(
                0.0,
                -(close_values[valid_price] / open_values[valid_price] - 1.0) * 10_000.0,
            )
            down_impact = np.full(len(local), np.nan, dtype=float)
            valid_impact = np.isfinite(downside_bp) & np.isfinite(sell_values) & (np.abs(sell_values) > EPS)
            down_impact[valid_impact] = downside_bp[valid_impact] / (sell_values[valid_impact] / 1_000_000.0)
            for item in subset.itertuples(index=False):
                decision = pd.Timestamp(item.decision_time)
                event_available = pd.Timestamp(item.event_available_time)
                right = int(np.searchsorted(end_ns, int(decision.value), side="right"))
                event_left = int(np.searchsorted(end_ns, int(event_available.value), side="right"))
                record: dict[str, object] = {"checkpoint_id": str(item.checkpoint_id)}
                record["range_causal_valid"] = bool(right > 0 and pd.Timestamp(end_ns[right - 1]) <= decision)
                record["range_last_end_time"] = pd.Timestamp(end_ns[right - 1]) if right > 0 else pd.NaT
                record["range_completed_since_sweep"] = int(max(0, right - event_left))
                for n_bars in (1, 3, 5):
                    window = local.iloc[max(0, right - n_bars):right]
                    record.update(_range_summary(f"range_last{n_bars}", window))
                completed_down_count = int(np.searchsorted(down_positions, right, side="left"))
                if completed_down_count >= 2:
                    current_pos = int(down_positions[completed_down_count - 1])
                    previous_pos = int(down_positions[completed_down_count - 2])
                    record["range_last_down_impact_ratio_vs_previous"] = _safe_ratio(
                        float(down_impact[current_pos]), float(down_impact[previous_pos])
                    )
                    record["range_last_down_duration_ratio_vs_previous"] = _safe_ratio(
                        float(duration_values[current_pos]), float(duration_values[previous_pos])
                    )
                else:
                    record["range_last_down_impact_ratio_vs_previous"] = np.nan
                    record["range_last_down_duration_ratio_vs_previous"] = np.nan
                rows.append(record)
            part = pd.DataFrame(rows)
            audits.append({
                "core_start": core_start, "core_end": core_end, "events": len(subset), "range_rows": len(local),
                "coverage": float(part["range_causal_valid"].fillna(False).mean()) if len(part) else 0.0,
                "status": "complete",
            })
        outputs.append(pd.DataFrame(rows))
        reporter.update(chunk_no)
    reporter.close()
    features = pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()
    if not features.empty and features["checkpoint_id"].duplicated().any():
        raise RuntimeError("duplicate checkpoint_id in range features")
    return FeatureModuleResult("range_r0020", features, pd.DataFrame(audits))


def build_footprint_features(
    checkpoints: pd.DataFrame,
    *,
    symbol: str,
    data_dir: str | Path | None,
    range_db_name: str,
    footprint_db_name: str,
    config: PostSweepSupervisedConfig,
    progress: bool = True,
) -> FeatureModuleResult:
    cfg = config.validate()
    source = checkpoints[["checkpoint_id", "decision_time"]].copy()
    source = source.rename(columns={"decision_time": "checkpoint_available_time"})
    source = source.dropna(subset=["checkpoint_id", "checkpoint_available_time"])
    if source.empty:
        return FeatureModuleResult("footprint", pd.DataFrame(), pd.DataFrame())
    range_loader = OKXRangeBarLoader(
        symbol=symbol, range_pct=0.0020, data_dir=data_dir, db_name=range_db_name, initialize_db=False,
    )
    footprint_loader = OKXRangeFootprintLoader(
        symbol=symbol, range_pct=0.0020, price_step=1.0, data_dir=data_dir, db_name=footprint_db_name,
    )
    fp_cfg = PostSweepFootprintBooksConfig(
        range_pct=0.0020,
        footprint_price_step=1.0,
        footprint_chunk_days=cfg.footprint_chunk_days,
        footprint_lag_bars=3,
    ).validate()
    try:
        result = build_footprint_context(
            source,
            range_loader=range_loader,
            footprint_loader=footprint_loader,
            config=fp_cfg,
            progress=progress,
        )
    except Exception as exc:
        missing = pd.DataFrame({"checkpoint_id": source["checkpoint_id"].astype(str), "fp_causal_valid": False})
        audit = pd.DataFrame([{"events": len(source), "coverage": 0.0, "status": "error", "error": str(exc)}])
        return FeatureModuleResult("footprint", missing, audit)
    context = result.context.copy()
    keep = [name for name in context.columns if name == "checkpoint_id" or name.startswith("fp_")]
    features = context.loc[:, keep].copy() if keep else pd.DataFrame({"checkpoint_id": source["checkpoint_id"]})
    audit = result.audit.copy()
    return FeatureModuleResult("footprint", features, audit)


def build_oi_features(
    checkpoints: pd.DataFrame,
    dataset_lookup: pd.DataFrame,
    *,
    symbol: str,
    data_dir: str | Path | None,
    db_name: str,
) -> FeatureModuleResult:
    source = checkpoints[["checkpoint_id", "decision_time"]].rename(columns={"decision_time": "checkpoint_available_time"}).copy()
    context_cols = [
        "checkpoint_id", "checkpoint_close", "sweep_bar_close", "post_delta_notional_sum",
        "post_notional_sum", "pre_return_5m", "current_delta_ratio",
    ]
    available = [name for name in context_cols if name in dataset_lookup.columns]
    source = source.merge(dataset_lookup[available].drop_duplicates("checkpoint_id"), on="checkpoint_id", how="left", validate="one_to_one")
    close = _numeric_column(source, "checkpoint_close")
    sweep_close = _numeric_column(source, "sweep_bar_close")
    fallback_return = _numeric_column(source, "pre_return_5m") * 10_000.0
    price_change = fallback_return.copy()
    valid_price = close.notna() & sweep_close.gt(0)
    price_change.loc[valid_price] = (close.loc[valid_price] / sweep_close.loc[valid_price] - 1.0) * 10_000.0
    source["price_change_5m_bp"] = price_change

    post_delta = _numeric_column(source, "post_delta_notional_sum")
    post_notional = _numeric_column(source, "post_notional_sum")
    fallback_delta = _numeric_column(source, "current_delta_ratio")
    delta_ratio = fallback_delta.copy()
    valid_notional = post_notional.abs().gt(EPS)
    delta_ratio.loc[valid_notional] = post_delta.loc[valid_notional] / post_notional.loc[valid_notional]
    source["delta_ratio_5m"] = delta_ratio
    try:
        features = load_binance_oi_context(
            source[["checkpoint_id", "checkpoint_available_time", "price_change_5m_bp", "delta_ratio_5m"]],
            symbol=symbol,
            data_dir=data_dir,
            db_name=db_name,
            publication_lag="1min",
        )
    except Exception as exc:
        missing = pd.DataFrame({"checkpoint_id": source["checkpoint_id"].astype(str), "oi_context_present": False})
        return FeatureModuleResult("oi", missing, pd.DataFrame([{"events": len(source), "coverage": 0.0, "status": "error", "error": str(exc)}]))
    keep = [name for name in features.columns if name == "checkpoint_id" or name.startswith("oi_") or name in {
        "position_flow_state_5m", "delta_oi_state_5m", "down_oi_up_flag", "down_oi_down_flag",
        "negative_delta_oi_up_flag", "negative_delta_oi_down_flag", "taker_volume_imbalance",
        "top_trader_account_long_share", "top_trader_position_long_share", "global_account_long_share",
    }]
    features = features.loc[:, keep].copy()
    coverage = float(_coerce_bool(features.get("oi_context_present", pd.Series(False, index=features.index))).mean()) if len(features) else 0.0
    audit = pd.DataFrame([{"events": len(source), "attached": len(features), "coverage": coverage, "status": "complete" if len(features) else "missing_cache"}])
    return FeatureModuleResult("oi", features, audit)


def cache_module(result: FeatureModuleResult, cache_dir: str | Path) -> tuple[Path, Path]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    feature_path = root / f"{result.name}_features.csv.gz"
    audit_path = root / f"{result.name}_audit.csv"
    result.features.to_csv(feature_path, index=False, encoding="utf-8-sig", compression="gzip")
    result.audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return feature_path, audit_path


def load_cached_module(name: str, cache_dir: str | Path) -> FeatureModuleResult | None:
    root = Path(cache_dir)
    feature_path = root / f"{name}_features.csv.gz"
    audit_path = root / f"{name}_audit.csv"
    if not feature_path.exists():
        return None
    features = pd.read_csv(feature_path, low_memory=False)
    for column in features.columns:
        if column.endswith("_time") or column.endswith("_ts"):
            features[column] = pd.to_datetime(features[column], errors="coerce")
    audit = pd.read_csv(audit_path, low_memory=False) if audit_path.exists() else pd.DataFrame()
    return FeatureModuleResult(name, features, audit)


def module_coverage(
    checkpoints: pd.DataFrame,
    module: FeatureModuleResult,
    present_column: str,
) -> pd.DataFrame:
    base = checkpoints[["checkpoint_id", "checkpoint_minutes", "split"]].copy()
    available = module.features[["checkpoint_id", present_column]].copy() if present_column in module.features.columns else pd.DataFrame({"checkpoint_id": module.features.get("checkpoint_id", pd.Series(dtype=str)), present_column: False})
    merged = base.merge(available, on="checkpoint_id", how="left")
    merged[present_column] = _coerce_bool(merged[present_column])
    rows = []
    for (minutes, split), group in merged.groupby(["checkpoint_minutes", "split"], dropna=False, sort=True):
        rows.append({
            "module": module.name,
            "checkpoint_minutes": int(minutes),
            "split": split,
            "events": len(group),
            "present": int(group[present_column].sum()),
            "coverage": float(group[present_column].mean()) if len(group) else np.nan,
        })
    return pd.DataFrame(rows)
