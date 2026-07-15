#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal range-bar / range-footprint feature research helpers.

The helpers in this module never build missing range data.  They read the local
SQLite caches produced by ``tools/prebuild_okx_range_all.py`` and align only
range bars whose ``end_ts`` is no later than the current 1m feature available
 time.  Footprint rows are aggregated inside SQLite by closed range bar before
being returned to Python, keeping the multi-year price-bucket dataset bounded.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader, range_code
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.research_common.progress import ProgressReporter

EPS = 1e-12


@dataclass(frozen=True)
class EmpiricalRankReference:
    """Frozen empirical CDF used for deployable raw-score percentiles."""

    sorted_values: np.ndarray

    @classmethod
    def fit(cls, values: Sequence[float]) -> "EmpiricalRankReference":
        array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        array = np.sort(array[np.isfinite(array)])
        if array.size == 0:
            array = np.asarray([0.0], dtype=float)
        return cls(sorted_values=array)

    def transform(self, values: Sequence[float]) -> np.ndarray:
        array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        out = np.full(array.shape, np.nan, dtype=float)
        finite = np.isfinite(array)
        if finite.any():
            # right insertion keeps equal raw scores in the same percentile
            # band.  Unlike isotonic probabilities, the underlying raw scores
            # remain continuous in normal operation.
            out[finite] = np.searchsorted(self.sorted_values, array[finite], side="right") / float(len(self.sorted_values))
        return np.clip(out, 0.0, 1.0)


def deployable_policy_specs() -> pd.DataFrame:
    """Small predeclared policy family; no automatic test-set winner search."""

    rows: list[dict[str, object]] = []
    for tp_fraction in (0.02, 0.05, 0.10, 0.20):
        rows.extend(
            [
                {
                    "policy_id": f"TP{int(tp_fraction*100):02d}_ONLY",
                    "tp_top_fraction": tp_fraction,
                    "fast30_min_percentile": np.nan,
                    "clean50_min_percentile": np.nan,
                    "risk_max_percentile": np.nan,
                },
                {
                    "policy_id": f"TP{int(tp_fraction*100):02d}_FAST50",
                    "tp_top_fraction": tp_fraction,
                    "fast30_min_percentile": 0.50,
                    "clean50_min_percentile": np.nan,
                    "risk_max_percentile": np.nan,
                },
                {
                    "policy_id": f"TP{int(tp_fraction*100):02d}_CLEAN50",
                    "tp_top_fraction": tp_fraction,
                    "fast30_min_percentile": np.nan,
                    "clean50_min_percentile": 0.50,
                    "risk_max_percentile": np.nan,
                },
                {
                    "policy_id": f"TP{int(tp_fraction*100):02d}_FAST50_CLEAN50",
                    "tp_top_fraction": tp_fraction,
                    "fast30_min_percentile": 0.50,
                    "clean50_min_percentile": 0.50,
                    "risk_max_percentile": np.nan,
                },
                {
                    "policy_id": f"TP{int(tp_fraction*100):02d}_CLEAN50_RISK75",
                    "tp_top_fraction": tp_fraction,
                    "fast30_min_percentile": np.nan,
                    "clean50_min_percentile": 0.50,
                    "risk_max_percentile": 0.75,
                },
            ]
        )
    return pd.DataFrame(rows)


def select_ranked_events(frame: pd.DataFrame, spec: pd.Series, *, cooldown_bars: int) -> pd.DataFrame:
    tp_min = 1.0 - float(spec["tp_top_fraction"])
    eligible = frame[pd.to_numeric(frame["p_tp60_rank"], errors="coerce") >= tp_min].copy()
    if pd.notna(spec["fast30_min_percentile"]):
        eligible = eligible[pd.to_numeric(eligible["p_fast30_rank"], errors="coerce") >= float(spec["fast30_min_percentile"])]
    if pd.notna(spec["clean50_min_percentile"]):
        eligible = eligible[pd.to_numeric(eligible["p_clean50_rank"], errors="coerce") >= float(spec["clean50_min_percentile"])]
    if pd.notna(spec["risk_max_percentile"]):
        eligible = eligible[pd.to_numeric(eligible["mae_horizon_risk_rank"], errors="coerce") <= float(spec["risk_max_percentile"])]
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["extreme_pos", "event_id"]).drop_duplicates("causal_region_id", keep="first")
    if int(cooldown_bars) > 0:
        chosen: list[int] = []
        last_position = -10**18
        for row_index, position in zip(eligible.index, pd.to_numeric(eligible["extreme_pos"], errors="raise").astype(int)):
            if int(position) - last_position < int(cooldown_bars):
                continue
            chosen.append(int(row_index))
            last_position = int(position)
        eligible = eligible.loc[chosen]
    return eligible.reset_index(drop=True)


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.isfinite(b) & (np.abs(b) > EPS))


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=1).sum()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=1).mean()


def _derive_range_features(
    range_bars: pd.DataFrame,
    footprint_aggregates: pd.DataFrame | None,
    *,
    code: str,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], pd.DataFrame]:
    if range_bars.empty:
        raise RuntimeError(f"range bars {code} are empty")
    bars = range_bars.reset_index(drop=True).copy().sort_values(["end_ts", "bar_id"]).reset_index(drop=True)
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], errors="coerce")
    bars = bars.dropna(subset=["end_ts"]).drop_duplicates("bar_id", keep="last")
    if footprint_aggregates is not None and not footprint_aggregates.empty:
        fp = footprint_aggregates.copy()
        fp["bar_id"] = pd.to_numeric(fp["bar_id"], errors="coerce").fillna(0).astype("int64")
        bars = bars.merge(fp.drop(columns=["end_ts"], errors="ignore"), on="bar_id", how="left", validate="one_to_one")

    def n(column: str, default: float = 0.0) -> pd.Series:
        if column not in bars.columns:
            return pd.Series(default, index=bars.index, dtype=float)
        return pd.to_numeric(bars[column], errors="coerce").fillna(default)

    width = (n("high") - n("low")).replace(0.0, np.nan)
    rb_return = _safe_divide(n("close") - n("open"), n("open"))
    delta_ratio = _safe_divide(n("delta_notional"), n("notional"))
    large_delta_ratio = _safe_divide(n("large_delta_notional"), n("notional"))
    close_location = _safe_divide(n("close") - n("low"), width)
    duration = n("duration_seconds").clip(lower=0.0)
    duration_safe = duration.clip(lower=1.0)

    feature_data: dict[str, object] = {
        "bar_id": bars["bar_id"].astype("int64").to_numpy(),
        "range_end_ts": bars["end_ts"].to_numpy(),
    }
    rb: dict[str, np.ndarray | pd.Series] = {
        f"{code}_rb_return": rb_return,
        f"{code}_rb_direction": n("direction"),
        f"{code}_rb_duration_log": np.log1p(duration),
        f"{code}_rb_speed_per_min": np.abs(rb_return) * 60.0 / duration_safe,
        f"{code}_rb_delta_ratio": delta_ratio,
        f"{code}_rb_large_delta_ratio": large_delta_ratio,
        f"{code}_rb_taker_buy_ratio": n("taker_buy_ratio"),
        f"{code}_rb_close_location": close_location,
        f"{code}_rb_notional_per_second_log": np.log1p(n("notional") / duration_safe),
        f"{code}_rb_trades_per_second_log": np.log1p(n("trades_count") / duration_safe),
        f"{code}_rb_sell_absorption": np.maximum(-delta_ratio, 0.0) * np.clip(close_location, 0.0, 1.0),
        f"{code}_rb_buy_failure": np.maximum(delta_ratio, 0.0) * np.clip(1.0 - close_location, 0.0, 1.0),
        f"{code}_rb_price_response_efficiency": np.clip(rb_return / (np.abs(delta_ratio) + 1e-4), -20.0, 20.0),
    }
    duration_median20 = duration.rolling(20, min_periods=3).median()
    notional_median20 = n("notional").rolling(20, min_periods=3).median()
    trades_median20 = n("trades_count").rolling(20, min_periods=3).median()
    rb[f"{code}_rb_duration_rel20"] = _safe_divide(duration, duration_median20)
    rb[f"{code}_rb_notional_rel20"] = _safe_divide(n("notional"), notional_median20)
    rb[f"{code}_rb_trades_rel20"] = _safe_divide(n("trades_count"), trades_median20)

    for window in (3, 6, 12):
        sum_notional = _rolling_sum(n("notional"), window)
        rb[f"{code}_rb_direction_mean_{window}"] = _rolling_mean(n("direction"), window)
        rb[f"{code}_rb_delta_ratio_{window}"] = _safe_divide(_rolling_sum(n("delta_notional"), window), sum_notional)
        rb[f"{code}_rb_large_delta_ratio_{window}"] = _safe_divide(_rolling_sum(n("large_delta_notional"), window), sum_notional)
        rb[f"{code}_rb_duration_mean_log_{window}"] = np.log1p(_rolling_mean(duration, window))
        rb[f"{code}_rb_sell_absorption_mean_{window}"] = _rolling_mean(pd.Series(rb[f"{code}_rb_sell_absorption"]), window)
        rb[f"{code}_rb_down_share_{window}"] = _rolling_mean((n("direction") < 0).astype(float), window)
        rb[f"{code}_rb_notional_rel20_mean_{window}"] = _rolling_mean(pd.Series(rb[f"{code}_rb_notional_rel20"]), window)
        if window >= 6:
            half = window // 2
            recent_sell = _rolling_mean(pd.Series(np.maximum(-delta_ratio, 0.0)), half)
            prior_sell = recent_sell.shift(half)
            rb[f"{code}_rb_sell_pressure_decay_{window}"] = _safe_divide(prior_sell - recent_sell, np.abs(prior_sell) + 1e-4)
            recent_duration = _rolling_mean(duration, half)
            prior_duration = recent_duration.shift(half)
            rb[f"{code}_rb_duration_change_{window}"] = _safe_divide(recent_duration - prior_duration, prior_duration + 1.0)

    for name, values in rb.items():
        feature_data[name] = np.asarray(values, dtype=np.float32)
    rb_columns = tuple(rb)

    fp_columns: list[str] = []
    if footprint_aggregates is not None and not footprint_aggregates.empty:
        level_count = n("fp_level_count").clip(lower=1.0)
        abs_delta = n("fp_abs_delta_notional")
        positive_delta = n("fp_positive_delta_notional")
        negative_delta = n("fp_negative_delta_notional")
        bucket_width = (n("fp_high_bucket") - n("fp_low_bucket")).replace(0.0, np.nan)
        fp: dict[str, np.ndarray | pd.Series] = {
            f"{code}_fp_level_count_log": np.log1p(level_count),
            f"{code}_fp_delta_ratio": _safe_divide(n("fp_delta_notional"), n("fp_notional")),
            f"{code}_fp_abs_delta_ratio": _safe_divide(abs_delta, n("fp_notional")),
            f"{code}_fp_delta_concentration": _safe_divide(n("fp_max_abs_delta_notional"), abs_delta),
            f"{code}_fp_large_delta_concentration": _safe_divide(n("fp_max_abs_large_delta_notional"), n("fp_abs_large_delta_notional")),
            f"{code}_fp_positive_level_share": _safe_divide(n("fp_positive_levels"), level_count),
            f"{code}_fp_negative_level_share": _safe_divide(n("fp_negative_levels"), level_count),
            f"{code}_fp_level_imbalance": _safe_divide(n("fp_positive_levels") - n("fp_negative_levels"), level_count),
            f"{code}_fp_notional_centroid_norm": _safe_divide(n("fp_notional_centroid") - n("fp_low_bucket"), bucket_width),
            f"{code}_fp_positive_centroid_norm": _safe_divide(n("fp_positive_delta_centroid") - n("fp_low_bucket"), bucket_width),
            f"{code}_fp_negative_centroid_norm": _safe_divide(n("fp_negative_delta_centroid") - n("fp_low_bucket"), bucket_width),
            f"{code}_fp_centroid_gap_norm": _safe_divide(n("fp_positive_delta_centroid") - n("fp_negative_delta_centroid"), bucket_width),
            f"{code}_fp_lower_half_negative_share": _safe_divide(n("fp_lower_half_negative_delta"), negative_delta),
            f"{code}_fp_upper_half_positive_share": _safe_divide(n("fp_upper_half_positive_delta"), positive_delta),
            f"{code}_fp_max_trade_share": _safe_divide(n("fp_max_trade_notional"), n("fp_notional")),
        }
        for window in (3, 6):
            for base_name in (
                "fp_delta_ratio",
                "fp_abs_delta_ratio",
                "fp_delta_concentration",
                "fp_level_imbalance",
                "fp_centroid_gap_norm",
                "fp_lower_half_negative_share",
                "fp_upper_half_positive_share",
            ):
                key = f"{code}_{base_name}"
                fp[f"{key}_mean_{window}"] = _rolling_mean(pd.Series(fp[key]), window)
        for name, values in fp.items():
            feature_data[name] = np.asarray(values, dtype=np.float32)
        fp_columns = list(fp)

    features = pd.DataFrame(feature_data)
    dictionary_rows = [
        {
            "feature": column,
            "source": "range_bar" if column in rb_columns else "range_footprint",
            "range_code": code,
            "causal_rule": "range_end_ts <= current_1m_feature_available_time",
        }
        for column in (*rb_columns, *fp_columns)
    ]
    return features, rb_columns, tuple(fp_columns), pd.DataFrame(dictionary_rows)


def _month_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first = pd.Timestamp(start).to_period("M").start_time
    stop = pd.Timestamp(end).to_period("M").start_time + pd.offsets.MonthBegin(1)
    starts = pd.date_range(first, stop, freq="MS")
    return [(starts[i], min(starts[i + 1], pd.Timestamp(end) + pd.Timedelta(microseconds=1))) for i in range(len(starts) - 1)]


def load_footprint_aggregates_local(
    *,
    symbol: str,
    range_pct: float,
    price_step: float,
    data_dir: str | Path | None,
    db_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Aggregate price buckets by closed range bar inside SQLite, month by month."""

    loader = OKXRangeFootprintLoader(
        symbol=symbol,
        range_pct=float(range_pct),
        price_step=float(price_step),
        data_dir=data_dir,
        db_name=db_name,
    )
    db_path = Path(loader.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"range footprint DB not found: {db_path}")
    periods = _month_boundaries(pd.Timestamp(start_date), pd.Timestamp(end_date))
    reporter = ProgressReporter(f"[range-fp] {range_code(range_pct)} monthly SQL", total=len(periods), every=1, enabled=show_progress)
    parts: list[pd.DataFrame] = []
    query = f"""
    WITH filtered AS (
        SELECT bar_id, end_ts, price_bucket, notional, trades_count,
               delta_notional, large_delta_notional, max_trade_notional
        FROM {loader.table_name}
        WHERE end_ts >= ? AND end_ts < ?
    ), bounds AS (
        SELECT bar_id, MIN(price_bucket) AS low_bucket, MAX(price_bucket) AS high_bucket
        FROM filtered GROUP BY bar_id
    )
    SELECT
        f.bar_id AS bar_id,
        MIN(f.end_ts) AS end_ts,
        COUNT(*) AS fp_level_count,
        SUM(f.notional) AS fp_notional,
        SUM(f.trades_count) AS fp_trades_count,
        SUM(f.delta_notional) AS fp_delta_notional,
        SUM(ABS(f.delta_notional)) AS fp_abs_delta_notional,
        SUM(CASE WHEN f.delta_notional > 0 THEN f.delta_notional ELSE 0 END) AS fp_positive_delta_notional,
        SUM(CASE WHEN f.delta_notional < 0 THEN -f.delta_notional ELSE 0 END) AS fp_negative_delta_notional,
        MAX(ABS(f.delta_notional)) AS fp_max_abs_delta_notional,
        SUM(f.large_delta_notional) AS fp_large_delta_notional,
        SUM(ABS(f.large_delta_notional)) AS fp_abs_large_delta_notional,
        MAX(ABS(f.large_delta_notional)) AS fp_max_abs_large_delta_notional,
        SUM(CASE WHEN f.delta_notional > 0 THEN 1 ELSE 0 END) AS fp_positive_levels,
        SUM(CASE WHEN f.delta_notional < 0 THEN 1 ELSE 0 END) AS fp_negative_levels,
        SUM(f.price_bucket * f.notional) / NULLIF(SUM(f.notional), 0) AS fp_notional_centroid,
        SUM(CASE WHEN f.delta_notional > 0 THEN f.price_bucket * f.delta_notional ELSE 0 END)
            / NULLIF(SUM(CASE WHEN f.delta_notional > 0 THEN f.delta_notional ELSE 0 END), 0) AS fp_positive_delta_centroid,
        SUM(CASE WHEN f.delta_notional < 0 THEN f.price_bucket * (-f.delta_notional) ELSE 0 END)
            / NULLIF(SUM(CASE WHEN f.delta_notional < 0 THEN -f.delta_notional ELSE 0 END), 0) AS fp_negative_delta_centroid,
        SUM(CASE WHEN f.price_bucket <= (b.low_bucket + b.high_bucket) / 2.0 AND f.delta_notional < 0 THEN -f.delta_notional ELSE 0 END)
            AS fp_lower_half_negative_delta,
        SUM(CASE WHEN f.price_bucket >= (b.low_bucket + b.high_bucket) / 2.0 AND f.delta_notional > 0 THEN f.delta_notional ELSE 0 END)
            AS fp_upper_half_positive_delta,
        MAX(f.max_trade_notional) AS fp_max_trade_notional,
        MIN(b.low_bucket) AS fp_low_bucket,
        MAX(b.high_bucket) AS fp_high_bucket
    FROM filtered f
    JOIN bounds b ON b.bar_id = f.bar_id
    GROUP BY f.bar_id
    ORDER BY end_ts, f.bar_id
    """
    with sqlite3.connect(db_path) as conn:
        for i, (month_start, month_end) in enumerate(periods, start=1):
            part = pd.read_sql_query(
                query,
                conn,
                params=(month_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], month_end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]),
                parse_dates=["end_ts"],
            )
            if not part.empty:
                parts.append(part)
            reporter.update(i)
    reporter.close()
    if not parts:
        raise RuntimeError(f"no local range footprint rows for {range_code(range_pct)} in {start_date}->{end_date}; table={loader.table_name}")
    out = pd.concat(parts, ignore_index=True)
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").fillna(0).astype("int64")
    return out.sort_values(["end_ts", "bar_id"]).drop_duplicates("bar_id", keep="last").reset_index(drop=True)


def load_and_align_range_features(
    candidates: pd.DataFrame,
    *,
    symbol: str,
    range_pct: float,
    price_step: float,
    data_dir: str | Path | None,
    range_bar_db_name: str,
    footprint_db_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    include_footprint: bool,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Load local closed range data, derive rolling features and causal-align."""

    code = range_code(float(range_pct))
    loader = OKXRangeBarLoader(
        symbol=symbol,
        range_pct=float(range_pct),
        data_dir=data_dir,
        db_name=range_bar_db_name,
    )
    range_bars = loader.load_local_data(start_date=start_date, end_date=end_date)
    if range_bars.empty:
        raise RuntimeError(f"no local range bars for {code} in {start_date}->{end_date}; db={loader.db_path} table={loader.table_name}")
    footprint = None
    if include_footprint:
        footprint = load_footprint_aggregates_local(
            symbol=symbol,
            range_pct=float(range_pct),
            price_step=float(price_step),
            data_dir=data_dir,
            db_name=footprint_db_name,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
        )
    range_feature_table, rb_columns, fp_columns, dictionary = _derive_range_features(
        range_bars,
        footprint,
        code=code,
    )

    left = candidates[["event_id", "feature_available_time"]].copy()
    left["feature_available_time"] = pd.to_datetime(left["feature_available_time"], errors="coerce")
    left["__order"] = np.arange(len(left), dtype=np.int64)
    right = range_feature_table.sort_values(["range_end_ts", "bar_id"]).copy()
    merged = pd.merge_asof(
        left.sort_values("feature_available_time"),
        right,
        left_on="feature_available_time",
        right_on="range_end_ts",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("__order")
    merged[f"{code}_range_age_seconds"] = (
        pd.to_datetime(merged["feature_available_time"]) - pd.to_datetime(merged["range_end_ts"])
    ).dt.total_seconds()
    aligned_columns = (*rb_columns, *fp_columns, f"{code}_range_age_seconds")
    out = merged[["event_id", "range_end_ts", *aligned_columns]].reset_index(drop=True)
    violations = (
        pd.to_datetime(out["range_end_ts"]) > pd.to_datetime(candidates["feature_available_time"]).reset_index(drop=True)
    ).fillna(False)
    diagnostics = pd.DataFrame(
        [
            {
                "range_code": code,
                "range_pct": float(range_pct),
                "range_bar_rows": int(len(range_bars)),
                "footprint_bar_rows": int(0 if footprint is None else len(footprint)),
                "footprint_to_range_bar_coverage": float(0.0 if footprint is None else min(1.0, len(footprint) / max(1, len(range_bars)))),
                "candidate_rows": int(len(candidates)),
                "aligned_non_null_rows": int(out["range_end_ts"].notna().sum()),
                "aligned_coverage": float(out["range_end_ts"].notna().mean()),
                "available_time_violations": int(violations.sum()),
                "minimum_range_age_seconds": float(pd.to_numeric(out[f"{code}_range_age_seconds"], errors="coerce").min()),
                "maximum_range_age_seconds": float(pd.to_numeric(out[f"{code}_range_age_seconds"], errors="coerce").max()),
                "range_bar_feature_count": int(len(rb_columns) + 1),
                "footprint_feature_count": int(len(fp_columns)),
            }
        ]
    )
    dictionary = pd.concat(
        [
            dictionary,
            pd.DataFrame(
                [
                    {
                        "feature": f"{code}_range_age_seconds",
                        "source": "range_bar",
                        "range_code": code,
                        "causal_rule": "current_1m_feature_available_time - last_closed_range_end_ts",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return out, (*rb_columns, f"{code}_range_age_seconds"), fp_columns, dictionary, diagnostics


def range_future_perturbation_audit(
    candidates: pd.DataFrame,
    range_feature_table: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    sample_size: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Perturb future range rows and verify current as-of features are unchanged."""

    if candidates.empty or range_feature_table.empty:
        return pd.DataFrame(columns=["event_id", "passed", "max_abs_diff"])
    rng = np.random.default_rng(random_state)
    positions = np.linspace(0, len(candidates) - 1, min(sample_size, len(candidates)), dtype=int)
    rows: list[dict[str, object]] = []
    source = range_feature_table.sort_values("range_end_ts").copy()
    numeric_columns = [column for column in feature_columns if column in source.columns]
    for i, position in enumerate(positions):
        candidate = candidates.iloc[int(position)]
        available = pd.Timestamp(candidate["feature_available_time"])
        before = source[source["range_end_ts"] <= available].tail(1)
        changed = source.copy()
        future_mask = pd.to_datetime(changed["range_end_ts"]) > available
        for column in numeric_columns:
            values = pd.to_numeric(changed[column], errors="coerce").to_numpy(dtype=float, copy=True)
            if future_mask.any():
                scale = rng.uniform(0.2, 4.0, int(future_mask.sum()))
                values[future_mask.to_numpy()] = np.where(np.isfinite(values[future_mask.to_numpy()]), values[future_mask.to_numpy()] * scale, values[future_mask.to_numpy()])
            changed[column] = values
        after = changed[changed["range_end_ts"] <= available].tail(1)
        if before.empty and after.empty:
            max_diff = 0.0
        elif before.empty or after.empty:
            max_diff = float("inf")
        else:
            left = before[numeric_columns].to_numpy(dtype=float)
            right = after[numeric_columns].to_numpy(dtype=float)
            max_diff = float(np.nanmax(np.abs(left - right))) if numeric_columns else 0.0
        rows.append({"event_id": candidate["event_id"], "available_time": available, "passed": bool(max_diff <= 1e-12), "max_abs_diff": max_diff})
    return pd.DataFrame(rows)
