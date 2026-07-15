#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal closed-bar multi-timeframe context for first-sweep research 13.

Only fully closed higher-timeframe bars are exposed to a 1m decision.  Each
aggregated bar is indexed by ``available_time = bar_start + timeframe`` and is
joined backward to the event's 1m ``feature_available_time``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-12
TF15_GROUP = "T1_closed_15m"
TF60_GROUP = "T2_closed_60m"


@dataclass(frozen=True)
class MultiFrameContextResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    group_membership: pd.DataFrame
    alignment_audit: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index, name=column)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype(float).where(denominator.abs() > EPS)
    return numerator.astype(float) / den


def _aggregate_closed_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{int(minutes)}min"
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "notional": "sum",
        "trades_count": "sum",
        "delta_notional": "sum",
        "large_delta_notional": "sum",
    }
    available = {column: method for column, method in aggregations.items() if column in bars.columns}
    out = bars[list(available)].resample(rule, label="left", closed="left").agg(available)
    out["source_bar_count"] = bars["close"].resample(rule, label="left", closed="left").count()
    out = out.dropna(subset=["open", "high", "low", "close"])
    # The source is a 1m axis.  Missing source minutes make an HTF bar
    # incomplete and therefore not deployable as a closed-bar context.
    out = out[out["source_bar_count"] >= int(minutes)].copy()
    out.index.name = "bar_start_time"
    out["available_time"] = out.index + pd.Timedelta(minutes=int(minutes))
    return out


def _build_features(htf: pd.DataFrame, minutes: int) -> pd.DataFrame:
    prefix = f"mtf_{int(minutes)}m"
    open_ = _numeric(htf, "open")
    high = _numeric(htf, "high")
    low = _numeric(htf, "low")
    close = _numeric(htf, "close")
    notional = _numeric(htf, "notional")
    trades = _numeric(htf, "trades_count")
    delta = _numeric(htf, "delta_notional")
    large_delta = _numeric(htf, "large_delta_notional")
    ret1 = close.pct_change(fill_method=None)

    output = pd.DataFrame(index=htf.index)
    output[f"{prefix}_bar_return"] = _safe_ratio(close, open_) - 1.0
    output[f"{prefix}_bar_range_pct"] = _safe_ratio(high - low, close)
    output[f"{prefix}_bar_close_location"] = _safe_ratio(close - low, high - low)
    output[f"{prefix}_bar_delta_ratio"] = _safe_ratio(delta, notional)
    output[f"{prefix}_bar_large_delta_ratio"] = _safe_ratio(large_delta, notional)
    output[f"{prefix}_return_3"] = _safe_ratio(close, close.shift(3)) - 1.0
    output[f"{prefix}_return_6"] = _safe_ratio(close, close.shift(6)) - 1.0
    rolling_low = low.rolling(6, min_periods=3).min()
    rolling_high = high.rolling(6, min_periods=3).max()
    output[f"{prefix}_range_position_6"] = _safe_ratio(close - rolling_low, rolling_high - rolling_low)
    output[f"{prefix}_realized_vol_6"] = ret1.rolling(6, min_periods=3).std()
    output[f"{prefix}_down_bar_share_6"] = (close < open_).astype(float).rolling(6, min_periods=3).mean()
    output[f"{prefix}_delta_ratio_3"] = _safe_ratio(
        delta.rolling(3, min_periods=2).sum(), notional.rolling(3, min_periods=2).sum()
    )
    output[f"{prefix}_delta_ratio_6"] = _safe_ratio(
        delta.rolling(6, min_periods=3).sum(), notional.rolling(6, min_periods=3).sum()
    )
    output[f"{prefix}_large_delta_ratio_6"] = _safe_ratio(
        large_delta.rolling(6, min_periods=3).sum(), notional.rolling(6, min_periods=3).sum()
    )
    output[f"{prefix}_notional_intensity"] = _safe_ratio(
        notional, notional.shift(1).rolling(12, min_periods=4).mean()
    )
    output[f"{prefix}_trades_intensity"] = _safe_ratio(
        trades, trades.shift(1).rolling(12, min_periods=4).mean()
    )
    negative_delta = (-output[f"{prefix}_delta_ratio_3"]).clip(lower=0.0)
    downside_return = (-output[f"{prefix}_return_3"]).clip(lower=0.0)
    output[f"{prefix}_sell_impact_efficiency_3"] = downside_return / (negative_delta + 1e-4)
    output[f"{prefix}_return_acceleration"] = output[f"{prefix}_return_3"] - 0.5 * output[f"{prefix}_return_6"]
    output[f"{prefix}_available_time"] = htf["available_time"].to_numpy()
    return output.reset_index(drop=True)


def _dictionary_rows(minutes: int, feature_columns: Sequence[str]) -> list[dict[str, object]]:
    group = TF15_GROUP if int(minutes) == 15 else TF60_GROUP
    return [
        {
            "feature": column,
            "feature_group": group,
            "source": f"fully closed {int(minutes)}m aggregate from 1m trade bars",
            "description": column.replace("_", " "),
            "causal_rule": f"mtf_{int(minutes)}m_available_time <= 1m feature_available_time",
        }
        for column in feature_columns
    ]


def attach_closed_multiframe_context(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    timeframes_minutes: Sequence[int] = (15, 60),
) -> MultiFrameContextResult:
    if decisions.empty:
        return MultiFrameContextResult(
            frame=decisions.copy(), dictionary=pd.DataFrame(), group_membership=pd.DataFrame(), alignment_audit=pd.DataFrame()
        )
    if "event_id" not in decisions or "feature_available_time" not in decisions:
        raise RuntimeError("multiframe context requires event_id and feature_available_time")

    out = decisions.reset_index(drop=True).copy()
    out["feature_available_time"] = pd.to_datetime(out["feature_available_time"], errors="coerce")
    if out["feature_available_time"].isna().any():
        raise RuntimeError("multiframe context contains invalid feature_available_time")
    out["event_id"] = out["event_id"].astype(str)
    out["_row_order"] = np.arange(len(out), dtype=np.int64)
    dictionary: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for minutes in tuple(int(value) for value in timeframes_minutes):
        if minutes not in (15, 60):
            raise ValueError("research 13 supports only predeclared 15m and 60m contexts")
        htf = _aggregate_closed_bars(bars, minutes)
        features = _build_features(htf, minutes)
        available_column = f"mtf_{minutes}m_available_time"
        left = out[["_row_order", "event_id", "feature_available_time"]].sort_values("feature_available_time")
        right = features.sort_values(available_column)
        merged = pd.merge_asof(
            left,
            right,
            left_on="feature_available_time",
            right_on=available_column,
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row_order")
        feature_columns = [
            column for column in merged.columns
            if column.startswith(f"mtf_{minutes}m_") and column != available_column
        ]
        phase_column = f"mtf_{minutes}m_cycle_phase"
        event_time = pd.to_datetime(out["feature_available_time"], errors="coerce")
        phase = ((event_time.dt.minute % minutes) + event_time.dt.second / 60.0) / float(minutes)
        out[phase_column] = phase.to_numpy(dtype=np.float32)
        out[available_column] = pd.to_datetime(merged[available_column], errors="coerce").to_numpy()
        for column in feature_columns:
            out[column] = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=np.float32)
        used = pd.to_datetime(out[available_column], errors="coerce")
        decision_time = pd.to_datetime(out["feature_available_time"], errors="coerce")
        violations = int((used > decision_time).fillna(False).sum())
        non_null = int(used.notna().sum())
        maximum_lag = float((decision_time - used).dt.total_seconds().max()) if non_null else np.nan
        audit_rows.append({
            "timeframe_minutes": minutes,
            "event_count": len(out),
            "context_non_null_count": non_null,
            "context_coverage": float(non_null / max(1, len(out))),
            "available_time_violations": violations,
            "maximum_available_lag_seconds": maximum_lag,
            "complete_htf_bars": len(htf),
            "passed": bool(violations == 0 and non_null > 0),
        })
        group = TF15_GROUP if minutes == 15 else TF60_GROUP
        all_features = [*feature_columns, phase_column]
        dictionary.extend(_dictionary_rows(minutes, all_features))
        membership.extend(
            {"feature_group": group, "feature": column, "feature_count": len(all_features)}
            for column in all_features
        )

    out = out.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
    return MultiFrameContextResult(
        frame=out,
        dictionary=pd.DataFrame(dictionary).drop_duplicates("feature").reset_index(drop=True),
        group_membership=pd.DataFrame(membership),
        alignment_audit=pd.DataFrame(audit_rows),
    )
