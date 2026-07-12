#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: basic impulse event study (round 01).

Research-only scope
-------------------
This script answers one question only: after a short-window abnormal directional
price impulse, does ETH tend to continue, reverse, or show no stable displacement?
It is not a strategy backtest and contains no TP/SL, trend, volume, order-flow,
regime, portfolio, or position-sizing filters.

Causal policy
-------------
- Source OHLCV bars are local 1m OKX trade bars, UTC+8, and left-labeled by bar start time.
- The loader is forced to local-cache-only mode (build_missing=False); this script never downloads ordinary K-lines or missing trade files.
- Every impulse feature at bar t uses only bar t and older closed bars.
- The historical volatility baseline is shifted by the entire impulse window, so
  the impulse itself is excluded from its normalization denominator.
- signal_bar_end = signal_bar_start + 1 minute.
- The event is confirmed when the signal bar closes.
- OKXTradeBarLoader intentionally omits no-trade/missing minutes. The script
  regularizes the execution axis to calendar 1m bars using the same flat-bar
  convention as the project's shared trade aggregation helpers, while retaining
  a source-bar observation mask.
- Synthetic gap rows are never eligible signal, entry, impulse-window, or
  forward-path inputs. They exist only to keep calendar-minute shifts exact.
- entry_time is the next calendar-minute bar open, which has the same timestamp
  as signal_bar_end on the regularized left-labeled 1m axis.
- Forward returns and MFE/MAE use only fully observed bars after the signal bar.

Output notes
------------
05_events.csv intentionally contains raw events with a per-threshold dedup flag.
It can be large, but it contains event rows only, never a full 1m-bar audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "01_basic_impulse_event_study"
SCRIPT_VERSION = "1.0.2"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R01"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Basic Impulse Event Study"
DEFAULT_OUT_DIR = "data/reports/research/momentum/directional_impulse_continuation/01_basic_impulse_event_study"
BAR_DELTA = pd.Timedelta(minutes=1)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_HORIZONS = (1, 3, 5, 10, 15, 30, 60, 120, 240)

FEATURE_COLUMNS = (
    "impulse_return",
    "normalized_impulse",
    "directional_efficiency",
    "window_range_bps",
    "window_realized_vol",
    "up_bar_ratio",
    "down_bar_ratio",
    "largest_bar_contribution",
    "close_location_in_window",
    "pre_impulse_volatility",
    "pre_impulse_return",
)


@dataclass(frozen=True)
class FeatureArrays:
    source_window_valid: np.ndarray
    impulse_return: np.ndarray
    normalized_impulse: np.ndarray
    directional_efficiency: np.ndarray
    window_range_bps: np.ndarray
    window_realized_vol: np.ndarray
    up_bar_ratio: np.ndarray
    down_bar_ratio: np.ndarray
    largest_bar_contribution: np.ndarray
    close_location_in_window: np.ndarray
    pre_impulse_volatility: np.ndarray
    pre_impulse_return: np.ndarray


@dataclass(frozen=True)
class PathArrays:
    future_high: np.ndarray
    future_low: np.ndarray


@dataclass(frozen=True)
class BucketSpec:
    feature: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]


BUCKET_SPECS = (
    BucketSpec("directional_efficiency", (-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf), ("<=0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">0.80")),
    BucketSpec("window_range_bps", (-np.inf, 10, 20, 40, 80, 160, np.inf), ("<=10", "10-20", "20-40", "40-80", "80-160", ">160")),
    BucketSpec("window_realized_vol", (-np.inf, 0.001, 0.002, 0.004, 0.008, 0.016, np.inf), ("<=0.10%", "0.10-0.20%", "0.20-0.40%", "0.40-0.80%", "0.80-1.60%", ">1.60%")),
    BucketSpec("up_bar_ratio", (-np.inf, 0.25, 0.50, 0.75, np.inf), ("<=0.25", "0.25-0.50", "0.50-0.75", ">0.75")),
    BucketSpec("largest_bar_contribution", (-np.inf, 0.25, 0.50, 0.75, np.inf), ("<=0.25", "0.25-0.50", "0.50-0.75", ">0.75")),
    BucketSpec("close_location_in_window", (-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf), ("<=0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">0.80")),
    BucketSpec("pre_impulse_volatility", (-np.inf, 0.001, 0.002, 0.004, 0.008, 0.016, np.inf), ("<=0.10%", "0.10-0.20%", "0.20-0.40%", "0.40-0.80%", "0.80-1.60%", ">1.60%")),
    BucketSpec("pre_impulse_return", (-np.inf, -0.005, -0.002, -0.0005, 0.0005, 0.002, 0.005, np.inf), ("<=-0.50%", "-0.50--0.20%", "-0.20--0.05%", "-0.05-0.05%", "0.05-0.20%", "0.20-0.50%", ">0.50%")),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ETH 1m directional impulse continuation event study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None, help="Optional project data directory containing okx_trade_bars.db")
    p.add_argument("--trade-bar-db-name", default="okx_trade_bars.db", help="Local trade-bar SQLite filename under --data-dir")
    p.add_argument("--impulse-windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    p.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage", type=float, default=0.00020)
    p.add_argument("--exit-slippage", type=float, default=0.00020)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--skip-events-csv", action="store_true", help="Development-only. Default writes required 05_events.csv and 06_signal_audit.csv.")
    p.add_argument("--self-test", action="store_true", help="Run a small deterministic synthetic end-to-end test instead of loading OKX data.")
    return p.parse_args(argv)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip()))
    if not values or any(v <= 0 for v in values):
        raise ValueError("integer list must contain positive values")
    return tuple(sorted(values))


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(x.strip()) for x in str(raw).split(",") if x.strip()))
    if not values or any((not math.isfinite(v)) or v <= 0 for v in values):
        raise ValueError("float list must contain positive finite values")
    return tuple(sorted(values))


def _date_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("Project date arguments must be timezone-naive UTC+8 timestamps")
    if end == end.normalize() and len(str(end_date).strip()) <= 10:
        end_exclusive = end + pd.Timedelta(days=1)
    else:
        end_exclusive = end + BAR_DELTA
    if end_exclusive <= start:
        raise ValueError("end-date must be after start-date")
    return start, end_exclusive


def _inclusive_loader_end(end_date: str) -> pd.Timestamp:
    end = pd.Timestamp(end_date)
    if end == end.normalize() and len(str(end_date).strip()) <= 10:
        return end + pd.Timedelta(days=1) - BAR_DELTA
    return end


def _trade_bar_db_path(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    return data_dir / str(args.trade_bar_db_name)


def _max_true_run(mask: np.ndarray) -> int:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return 0
    split_at = np.flatnonzero(np.diff(positions) > 1) + 1
    return int(max(len(chunk) for chunk in np.split(positions, split_at)))


def _regularize_trade_bar_axis(bars: pd.DataFrame) -> pd.DataFrame:
    """Build an exact calendar-minute axis without treating gap rows as real data.

    ``OKXTradeBarLoader`` drops empty resample buckets by design. The shared
    ``src.backtest_common.data.merge_second_bars`` helper regularizes trade bars
    by carrying the last close through empty intervals and using zero activity.
    We apply the same price/activity convention here, but additionally preserve
    ``source_bar_observed_flag`` so no event, entry, or forward path may depend on
    a synthetic row.
    """
    raw = bars.sort_index().copy()
    if raw.empty:
        return raw
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("Trade-bar index must be DatetimeIndex")
    if raw.index.has_duplicates:
        raise RuntimeError(f"Trade-bar index contains {int(raw.index.duplicated().sum())} duplicate timestamps")

    full_index = pd.date_range(raw.index[0], raw.index[-1], freq="1min")
    observed = full_index.isin(raw.index)
    out = raw.reindex(full_index)
    carried_close = pd.to_numeric(out["close"], errors="coerce").ffill()
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(carried_close)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out["source_bar_observed_flag"] = observed.astype(bool)
    out["synthetic_gap_bar_flag"] = (~observed).astype(bool)
    out.index.name = raw.index.name or "timestamp"

    missing_mask = ~observed
    missing_positions = np.flatnonzero(missing_mask)
    gap_segments = int(0 if missing_positions.size == 0 else 1 + np.sum(np.diff(missing_positions) > 1))
    out.attrs.update(
        {
            "source_rows": int(len(raw)),
            "regularized_rows": int(len(out)),
            "synthetic_gap_bar_count": int(missing_mask.sum()),
            "gap_segment_count": gap_segments,
            "max_gap_minutes": _max_true_run(missing_mask),
            "gap_policy": (
                "calendar-minute axis regularized with previous close and zero volume; "
                "synthetic rows excluded from signal, entry, impulse, and forward paths"
            ),
        }
    )
    return out


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    loader_end = _inclusive_loader_end(args.end_date)
    db_path = _trade_bar_db_path(args)
    print(
        f"[load] OKXTradeBarLoader local-cache-only {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{loader_end}",
        flush=True,
    )
    print(f"       db={db_path}", flush=True)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Local trade-bar DB not found: {db_path}. "
            "This research script does not download ordinary K-lines or build missing trade bars."
        )
    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        db_name=args.trade_bar_db_name,
        align_with_okx_loader_timezone=True,
    )
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        loader_end,
        cvd_mode="range",
        build_missing=False,
        force_rebuild=False,
    )
    if bars.empty:
        raise RuntimeError(
            f"No local trade bars loaded from {db_path}. "
            "build_missing=False was enforced, so no download was attempted."
        )
    raw = bars.sort_index().copy()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Trade-bar OHLCV data missing columns: {missing}")
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=required)
    out = _regularize_trade_bar_axis(raw)
    print(
        f"       source_rows={out.attrs['source_rows']:,} regularized_rows={len(out):,} "
        f"range={out.index[0]} -> {out.index[-1]}",
        flush=True,
    )
    print(
        f"       gap_segments={out.attrs['gap_segment_count']:,} "
        f"synthetic_gap_bars={out.attrs['synthetic_gap_bar_count']:,} "
        f"max_gap={out.attrs['max_gap_minutes']}m; gap-dependent events are excluded",
        flush=True,
    )
    return out


def validate_bars(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    print("[validate] checking UTC+8 left-label assumptions, regularized continuity and OHLC validity", flush=True)
    if args.timeframe != "1m":
        raise ValueError("Round 01 is fixed to timeframe=1m")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be DatetimeIndex")
    if bars.index.tz is not None:
        raise RuntimeError("Expected timezone-naive UTC+8 project timestamps; got timezone-aware index. Hold and inspect data semantics.")
    if not bars.index.is_monotonic_increasing:
        raise RuntimeError("OHLCV index is not monotonic increasing")
    if bars.index.has_duplicates:
        duplicates = int(bars.index.duplicated().sum())
        raise RuntimeError(f"OHLCV index contains {duplicates} duplicate timestamps")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("OHLCV contains non-positive prices")
    invalid_ohlc = (
        (bars["high"] < bars[["open", "close", "low"]].max(axis=1))
        | (bars["low"] > bars[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise RuntimeError(f"OHLCV contains {int(invalid_ohlc.sum())} invalid high/low rows")

    if "source_bar_observed_flag" not in bars.columns:
        raise RuntimeError("Regularized trade-bar axis is missing source_bar_observed_flag")
    observed = bars["source_bar_observed_flag"].astype(bool)
    diffs = bars.index.to_series().diff().dropna()
    post_regularization_gap_count = int((diffs != BAR_DELTA).sum())
    if post_regularization_gap_count:
        raise RuntimeError(f"Regularized 1m axis still contains {post_regularization_gap_count} non-1m gaps")
    synthetic = ~observed
    if synthetic.any():
        synthetic_ohlc = bars.loc[synthetic, ["open", "high", "low", "close"]]
        flat_ok = synthetic_ohlc.nunique(axis=1, dropna=False).eq(1)
        if not bool(flat_ok.all()):
            raise RuntimeError("Synthetic gap bars are not flat previous-close bars")
        if not bool((pd.to_numeric(bars.loc[synthetic, "volume"], errors="coerce") == 0.0).all()):
            raise RuntimeError("Synthetic gap bars must have zero volume")

    start, end_exclusive = _date_bounds(args.start_date, args.end_date)
    if bars.index.min() > pd.Timestamp(args.warmup_start_date):
        raise RuntimeError("Local trade-bar data starts after warmup_start_date; no download was attempted")
    expected_last_bar = _inclusive_loader_end(args.end_date)
    if bars.index.max() < expected_last_bar:
        raise RuntimeError(
            f"Local trade-bar data ends at {bars.index.max()}, before required {expected_last_bar}. "
            "No download was attempted; complete the local trade-bar cache first."
        )
    research_rows = int(((bars.index >= start) & (bars.index < end_exclusive)).sum())
    if research_rows <= 0:
        raise RuntimeError("No rows inside the research date range")
    return {
        "index_timezone_semantics": "timezone-naive timestamps interpreted by project convention as UTC+8",
        "bar_label_semantics": "left-labeled bar start",
        "post_regularization_gap_count": post_regularization_gap_count,
        "source_gap_segment_count": int(bars.attrs.get("gap_segment_count", 0)),
        "synthetic_gap_bar_count": int(synthetic.sum()),
        "max_gap_minutes": int(bars.attrs.get("max_gap_minutes", 0)),
        "gap_policy": str(bars.attrs.get("gap_policy", "")),
        "research_axis_rows": research_rows,
        "research_observed_rows": int((((bars.index >= start) & (bars.index < end_exclusive)) & observed.to_numpy()).sum()),
        "data_start": str(bars.index.min()),
        "data_end": str(bars.index.max()),
    }


def _future_window_extreme(values: pd.Series, horizon: int, op: str) -> np.ndarray:
    """Extreme over bars t+1..t+horizon with a full-window requirement."""
    shifted = pd.to_numeric(values, errors="coerce").shift(-1)
    rev = shifted.iloc[::-1]
    if op == "max":
        out = rev.rolling(int(horizon), min_periods=int(horizon)).max().iloc[::-1]
    elif op == "min":
        out = rev.rolling(int(horizon), min_periods=int(horizon)).min().iloc[::-1]
    else:
        raise ValueError("op must be max or min")
    return out.to_numpy(dtype=float)


def build_path_cache(bars: pd.DataFrame, horizons: tuple[int, ...], *, progress_enabled: bool) -> dict[int, PathArrays]:
    print("[forward paths] precomputing vectorized future high/low windows once", flush=True)
    cache: dict[int, PathArrays] = {}
    with ProgressReporter(
        label="[forward paths] horizons",
        total=len(horizons),
        every=1,
        enabled=progress_enabled,
    ) as progress:
        for done, horizon in enumerate(horizons, start=1):
            cache[int(horizon)] = PathArrays(
                future_high=_future_window_extreme(bars["high"], int(horizon), "max"),
                future_low=_future_window_extreme(bars["low"], int(horizon), "min"),
            )
            progress.update(done)
    return cache


def build_base_volatility(bars: pd.DataFrame, lookback: int, min_periods: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = pd.to_numeric(bars["close"], errors="coerce")
    observed = bars["source_bar_observed_flag"].astype(bool)
    valid_one_minute_return = observed & observed.shift(1, fill_value=False)
    log_return = np.log(close).diff().where(valid_one_minute_return)
    abs_price_change = close.diff().abs().where(valid_one_minute_return)
    historical_1m_vol = log_return.rolling(int(lookback), min_periods=int(min_periods)).std(ddof=0)
    return log_return, abs_price_change, historical_1m_vol


def build_window_features(
    bars: pd.DataFrame,
    window: int,
    log_return: pd.Series,
    abs_price_change: pd.Series,
    historical_1m_vol: pd.Series,
) -> FeatureArrays:
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    observed = bars["source_bar_observed_flag"].astype(bool)
    w = int(window)
    source_window_valid = observed.rolling(w + 1, min_periods=w + 1).sum().eq(w + 1)
    pre_source_window_valid = source_window_valid.shift(w, fill_value=False)

    impulse_return = (close / close.shift(w) - 1.0).where(source_window_valid)
    expected_window_vol = historical_1m_vol.shift(w) * math.sqrt(w)
    normalized_impulse = (impulse_return / expected_window_vol.replace(0.0, np.nan)).where(source_window_valid)

    path_abs_change = abs_price_change.rolling(w, min_periods=w).sum()
    directional_efficiency = ((close - close.shift(w)).abs() / path_abs_change.replace(0.0, np.nan)).where(source_window_valid)

    rolling_high = high.rolling(w, min_periods=w).max()
    rolling_low = low.rolling(w, min_periods=w).min()
    window_range_bps = ((rolling_high / rolling_low.replace(0.0, np.nan) - 1.0) * 10_000.0).where(source_window_valid)
    window_realized_vol = np.sqrt(log_return.pow(2).rolling(w, min_periods=w).sum()).where(source_window_valid)

    up_bar_ratio = (log_return > 0).astype(float).where(log_return.notna()).rolling(w, min_periods=w).mean().where(source_window_valid)
    down_bar_ratio = (log_return < 0).astype(float).where(log_return.notna()).rolling(w, min_periods=w).mean().where(source_window_valid)
    abs_log_return = log_return.abs()
    largest_bar_contribution = (abs_log_return.rolling(w, min_periods=w).max() / abs_log_return.rolling(w, min_periods=w).sum().replace(0.0, np.nan)).where(source_window_valid)
    close_location = ((close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)).where(source_window_valid)

    pre_impulse_volatility = np.sqrt(log_return.pow(2).rolling(w, min_periods=w).sum()).shift(w).where(pre_source_window_valid)
    pre_impulse_return = (close.shift(w) / close.shift(2 * w) - 1.0).where(pre_source_window_valid)

    return FeatureArrays(
        source_window_valid=source_window_valid.to_numpy(dtype=bool),
        impulse_return=impulse_return.to_numpy(dtype=float),
        normalized_impulse=normalized_impulse.to_numpy(dtype=float),
        directional_efficiency=directional_efficiency.to_numpy(dtype=float),
        window_range_bps=window_range_bps.to_numpy(dtype=float),
        window_realized_vol=window_realized_vol.to_numpy(dtype=float),
        up_bar_ratio=up_bar_ratio.to_numpy(dtype=float),
        down_bar_ratio=down_bar_ratio.to_numpy(dtype=float),
        largest_bar_contribution=largest_bar_contribution.to_numpy(dtype=float),
        close_location_in_window=close_location.to_numpy(dtype=float),
        pre_impulse_volatility=pre_impulse_volatility.to_numpy(dtype=float),
        pre_impulse_return=pre_impulse_return.to_numpy(dtype=float),
    )


def _deduplicate_positions(positions: np.ndarray, cooldown_bars: int) -> np.ndarray:
    """Keep the first same-direction event until cooldown from the last kept event."""
    keep = np.zeros(len(positions), dtype=bool)
    last_kept = -10**18
    cooldown = int(cooldown_bars)
    for i, pos in enumerate(positions):
        pos_i = int(pos)
        if pos_i - last_kept >= cooldown:
            keep[i] = True
            last_kept = pos_i
    return keep


def _profit_factor(values: np.ndarray) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return float("nan")
    gross_profit = float(x[x > 0].sum())
    gross_loss = float(-x[x < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def _top_positive_share(values: np.ndarray, top_n: int) -> float:
    x = values[np.isfinite(values) & (values > 0)]
    if x.size == 0:
        return float("nan")
    total = float(x.sum())
    if total <= 0:
        return float("nan")
    if x.size <= top_n:
        return 1.0
    top = np.partition(x, x.size - top_n)[-top_n:]
    return float(top.sum() / total)


def _tail_loss(values: np.ndarray, tail_fraction: float = 0.05) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return float("nan")
    n = max(1, int(math.ceil(x.size * float(tail_fraction))))
    worst = np.partition(x, n - 1)[:n]
    return float(worst.mean())


def _stats(
    gross: np.ndarray,
    fee_only: np.ndarray,
    normal: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(gross) & np.isfinite(fee_only) & np.isfinite(normal)
    g = gross[valid]
    f = fee_only[valid]
    n = normal[valid]
    mf = mfe[valid] if len(mfe) == len(valid) else np.array([], dtype=float)
    ma = mae[valid] if len(mae) == len(valid) else np.array([], dtype=float)
    if n.size == 0:
        return {
            "events": 0,
            "mean_gross": np.nan,
            "mean_fee_only_net": np.nan,
            "mean_net": np.nan,
            "median_gross": np.nan,
            "median_fee_only_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "standard_deviation": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "tail_loss_bottom_5pct": np.nan,
            "mean_mfe": np.nan,
            "mean_mae": np.nan,
            "top_1_event_contribution": np.nan,
            "top_5_event_contribution": np.nan,
        }
    return {
        "events": int(n.size),
        "mean_gross": float(np.mean(g)),
        "mean_fee_only_net": float(np.mean(f)),
        "mean_net": float(np.mean(n)),
        "median_gross": float(np.median(g)),
        "median_fee_only_net": float(np.median(f)),
        "median_net": float(np.median(n)),
        "win_rate": float(np.mean(n > 0)),
        "profit_factor": _profit_factor(n),
        "standard_deviation": float(np.std(n, ddof=1)) if n.size > 1 else 0.0,
        "p01": float(np.quantile(n, 0.01)),
        "p05": float(np.quantile(n, 0.05)),
        "p25": float(np.quantile(n, 0.25)),
        "p50": float(np.quantile(n, 0.50)),
        "p75": float(np.quantile(n, 0.75)),
        "p95": float(np.quantile(n, 0.95)),
        "p99": float(np.quantile(n, 0.99)),
        "tail_loss_bottom_5pct": _tail_loss(n),
        "mean_mfe": float(np.nanmean(mf)) if np.isfinite(mf).any() else np.nan,
        "mean_mae": float(np.nanmean(ma)) if np.isfinite(ma).any() else np.nan,
        "top_1_event_contribution": _top_positive_share(n, 1),
        "top_5_event_contribution": _top_positive_share(n, 5),
    }


def _build_event_frame(
    *,
    bars: pd.DataFrame,
    positions: np.ndarray,
    dedup_flags: np.ndarray,
    direction: str,
    side: int,
    window: int,
    threshold: float,
    features: FeatureArrays,
    path_cache: dict[int, PathArrays],
    full_forward_observed_mask: np.ndarray,
    horizons: tuple[int, ...],
    fee_cost: float,
    normal_cost: float,
    event_id_start: int,
) -> pd.DataFrame:
    index = bars.index
    open_arr = bars["open"].to_numpy(dtype=float)
    close_arr = bars["close"].to_numpy(dtype=float)
    entry_pos = positions + 1
    signal_start = index[positions]
    signal_end = signal_start + BAR_DELTA
    entry_time = index[entry_pos]
    entry_price = open_arr[entry_pos]
    expected_entry_price = pd.to_numeric(bars["open"], errors="coerce").reindex(signal_end).to_numpy(dtype=float)
    entry_price_mismatch = ~np.isclose(entry_price, expected_entry_price, rtol=0.0, atol=1e-12, equal_nan=True)
    signal_source_observed = bars["source_bar_observed_flag"].to_numpy(dtype=bool)[positions]
    entry_source_observed = bars["source_bar_observed_flag"].to_numpy(dtype=bool)[entry_pos]
    impulse_source_observed = features.source_window_valid[positions]
    forward_source_observed = full_forward_observed_mask[positions]
    synthetic_dependency = ~(
        signal_source_observed & entry_source_observed & impulse_source_observed & forward_source_observed
    )

    data: dict[str, Any] = {
            "event_id": np.arange(event_id_start, event_id_start + len(positions), dtype=np.int64),
            "direction": direction,
            "side": int(side),
            "impulse_window": int(window),
            "impulse_window_label": f"{int(window)}m",
            "threshold": float(threshold),
            "signal_time": signal_end,
            "signal_bar_start": signal_start,
            "signal_bar_end": signal_end,
            "entry_time": entry_time,
            "expected_entry_time": signal_end,
            "entry_price": entry_price,
            "expected_entry_price": expected_entry_price,
            "entry_not_next_open_flag": entry_time != signal_end,
            "entry_price_mismatch_flag": entry_price_mismatch,
            "signal_source_bar_observed_flag": signal_source_observed,
            "entry_source_bar_observed_flag": entry_source_observed,
            "impulse_source_window_observed_flag": impulse_source_observed,
            "full_forward_observed_flag": forward_source_observed,
            "synthetic_bar_dependency_flag": synthetic_dependency,
            "raw_event_flag": np.ones(len(positions), dtype=bool),
            "deduplicated_event_flag": dedup_flags.astype(bool),
            "full_forward_window_flag": np.ones(len(positions), dtype=bool),
        }
    for name in FEATURE_COLUMNS:
        data[name] = getattr(features, name)[positions]

    for horizon in horizons:
        h = int(horizon)
        exit_price = close_arr[positions + h]
        gross = (exit_price / entry_price - 1.0) * float(side)
        data[f"forward_return_{h}m"] = gross
        data[f"fee_only_net_return_{h}m"] = gross - float(fee_cost)
        data[f"normal_net_return_{h}m"] = gross - float(normal_cost)

        future_high = path_cache[h].future_high[positions]
        future_low = path_cache[h].future_low[positions]
        if side == 1:
            mfe = future_high / entry_price - 1.0
            mae = future_low / entry_price - 1.0
        else:
            mfe = 1.0 - future_low / entry_price
            mae = 1.0 - future_high / entry_price
        data[f"mfe_{h}m"] = mfe
        data[f"mae_{h}m"] = mae
    return pd.DataFrame(data)


def _group_indices(labels: np.ndarray, mask: np.ndarray) -> dict[Any, np.ndarray]:
    base = np.flatnonzero(mask)
    if base.size == 0:
        return {}
    selected_labels = labels[base]
    order = np.argsort(selected_labels, kind="stable")
    sorted_indices = base[order]
    sorted_labels = selected_labels[order]
    unique, starts = np.unique(sorted_labels, return_index=True)
    ends = np.r_[starts[1:], len(sorted_indices)]
    return {label.item() if hasattr(label, "item") else label: sorted_indices[a:b] for label, a, b in zip(unique, starts, ends, strict=False)}


def _summary_rows_for_combo(
    events: pd.DataFrame,
    *,
    direction: str,
    window: int,
    threshold: float,
    horizons: tuple[int, ...],
    study_months: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    years = pd.to_datetime(events["signal_bar_start"]).dt.year.to_numpy()
    months = pd.to_datetime(events["signal_bar_start"]).dt.to_period("M").astype(str).to_numpy()
    dedup = events["deduplicated_event_flag"].to_numpy(dtype=bool)
    arrays = {
        int(h): (
            events[f"forward_return_{int(h)}m"].to_numpy(dtype=float),
            events[f"fee_only_net_return_{int(h)}m"].to_numpy(dtype=float),
            events[f"normal_net_return_{int(h)}m"].to_numpy(dtype=float),
            events[f"mfe_{int(h)}m"].to_numpy(dtype=float),
            events[f"mae_{int(h)}m"].to_numpy(dtype=float),
        )
        for h in horizons
    }

    for event_set, base_mask in (("raw", np.ones(len(events), dtype=bool)), ("deduplicated", dedup)):
        year_groups = _group_indices(years, base_mask)
        month_groups = _group_indices(months, base_mask)
        base_indices = np.flatnonzero(base_mask)
        for horizon in horizons:
            h = int(horizon)
            gross_all, fee_all, net_all, mfe_all, mae_all = arrays[h]
            stat = _stats(
                gross_all[base_indices],
                fee_all[base_indices],
                net_all[base_indices],
                mfe_all[base_indices],
                mae_all[base_indices],
            )

            year_means: dict[int, float] = {}
            for year, idx in year_groups.items():
                period_stat = _stats(gross_all[idx], fee_all[idx], net_all[idx], mfe_all[idx], mae_all[idx])
                yearly_rows.append(
                    {
                        "direction": direction,
                        "impulse_window": int(window),
                        "threshold": float(threshold),
                        "horizon": h,
                        "event_set": event_set,
                        "year": int(year),
                        **period_stat,
                    }
                )
                year_means[int(year)] = float(period_stat["mean_net"])

            for month, idx in month_groups.items():
                period_stat = _stats(gross_all[idx], fee_all[idx], net_all[idx], mfe_all[idx], mae_all[idx])
                monthly_rows.append(
                    {
                        "direction": direction,
                        "impulse_window": int(window),
                        "threshold": float(threshold),
                        "horizon": h,
                        "event_set": event_set,
                        "month": str(month),
                        **period_stat,
                    }
                )

            finite_years = {y: v for y, v in year_means.items() if math.isfinite(v)}
            if finite_years:
                worst_year = min(finite_years, key=finite_years.get)
                positive_years = sum(v > 0 for v in finite_years.values())
            else:
                worst_year = None
                positive_years = 0
            summary_rows.append(
                {
                    "direction": direction,
                    "impulse_window": int(window),
                    "threshold": float(threshold),
                    "horizon": h,
                    "event_set": event_set,
                    "events_per_month": float(stat["events"] / max(1, study_months)),
                    **stat,
                    "positive_year_count": int(positive_years),
                    "total_year_count": int(len(finite_years)),
                    "worst_year": worst_year,
                    "worst_year_mean_net": finite_years.get(worst_year, np.nan) if worst_year is not None else np.nan,
                }
            )

            candidate_indices = base_indices[np.isfinite(net_all[base_indices]) & (net_all[base_indices] > 0)]
            if candidate_indices.size:
                order = candidate_indices[np.argsort(net_all[candidate_indices])[::-1][:5]]
                total_positive = float(net_all[candidate_indices].sum())
                cumulative = 0.0
                for rank, row_idx in enumerate(order, start=1):
                    contribution = float(net_all[row_idx] / total_positive) if total_positive > 0 else np.nan
                    cumulative += contribution if math.isfinite(contribution) else 0.0
                    top_rows.append(
                        {
                            "direction": direction,
                            "impulse_window": int(window),
                            "threshold": float(threshold),
                            "horizon": h,
                            "event_set": event_set,
                            "rank": rank,
                            "event_id": int(events.iloc[row_idx]["event_id"]),
                            "signal_time": events.iloc[row_idx]["signal_time"],
                            "normal_net_return": float(net_all[row_idx]),
                            "contribution_to_positive_return": contribution,
                            "cumulative_top_contribution": cumulative,
                        }
                    )
    return summary_rows, yearly_rows, monthly_rows, top_rows

def _feature_bucket_rows(
    events: pd.DataFrame,
    *,
    direction: str,
    window: int,
    threshold: float,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dedup = events["deduplicated_event_flag"].to_numpy(dtype=bool)
    if not dedup.any():
        return rows
    arrays = {
        int(h): (
            events[f"forward_return_{int(h)}m"].to_numpy(dtype=float),
            events[f"fee_only_net_return_{int(h)}m"].to_numpy(dtype=float),
            events[f"normal_net_return_{int(h)}m"].to_numpy(dtype=float),
            events[f"mfe_{int(h)}m"].to_numpy(dtype=float),
            events[f"mae_{int(h)}m"].to_numpy(dtype=float),
        )
        for h in horizons
    }
    for spec in BUCKET_SPECS:
        values = pd.to_numeric(events[spec.feature], errors="coerce").to_numpy(dtype=float)
        bucket_index = np.digitize(values, np.asarray(spec.edges[1:-1], dtype=float), right=True)
        groups = _group_indices(bucket_index, dedup & np.isfinite(values))
        for horizon in horizons:
            h = int(horizon)
            gross_all, fee_all, net_all, mfe_all, mae_all = arrays[h]
            for bucket_no, label in enumerate(spec.labels):
                idx = groups.get(bucket_no, np.array([], dtype=int))
                stat = _stats(gross_all[idx], fee_all[idx], net_all[idx], mfe_all[idx], mae_all[idx])
                rows.append(
                    {
                        "direction": direction,
                        "impulse_window": int(window),
                        "threshold": float(threshold),
                        "horizon": h,
                        "event_set": "deduplicated",
                        "feature": spec.feature,
                        "bucket": label,
                        **stat,
                    }
                )
    return rows

def _build_plateau(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "horizon", "event_set"]
    for key, part in summary.groupby(keys, dropna=False, observed=False):
        ordered = part.sort_values("threshold")
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "threshold": float(row["threshold"]),
                    "events": int(row["events"]),
                    "events_per_month": float(row["events_per_month"]),
                    "mean_net": float(row["mean_net"]),
                    "median_net": float(row["median_net"]),
                    "profit_factor": row["profit_factor"],
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                }
            )
            if previous is None:
                item.update(
                    {
                        "previous_threshold": np.nan,
                        "event_retention_vs_previous": np.nan,
                        "mean_net_change_vs_previous": np.nan,
                        "median_net_change_vs_previous": np.nan,
                        "profit_factor_change_vs_previous": np.nan,
                    }
                )
            else:
                prev_events = float(previous["events"])
                item.update(
                    {
                        "previous_threshold": float(previous["threshold"]),
                        "event_retention_vs_previous": float(row["events"] / prev_events) if prev_events > 0 else np.nan,
                        "mean_net_change_vs_previous": float(row["mean_net"] - previous["mean_net"]),
                        "median_net_change_vs_previous": float(row["median_net"] - previous["median_net"]),
                        "profit_factor_change_vs_previous": (
                            float(row["profit_factor"] - previous["profit_factor"])
                            if isinstance(row["profit_factor"], (int, float, np.floating))
                            and isinstance(previous["profit_factor"], (int, float, np.floating))
                            and math.isfinite(float(row["profit_factor"]))
                            and math.isfinite(float(previous["profit_factor"]))
                            else np.nan
                        ),
                    }
                )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _event_count_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    all_threshold_positions: np.ndarray,
    research_window_mask: np.ndarray,
    eligible_mask: np.ndarray,
    dedup_all: np.ndarray,
    study_months: int,
) -> list[dict[str, Any]]:
    research_count = int(research_window_mask.sum())
    eligible_count = int(eligible_mask.sum())
    dedup_eligible_count = int((eligible_mask & dedup_all).sum())
    overlap_ratio = 1.0 - (dedup_eligible_count / eligible_count) if eligible_count else np.nan
    return [
        {
            "direction": direction,
            "impulse_window": int(window),
            "threshold": float(threshold),
            "event_set": "raw",
            "raw_detected_count_full_loaded_axis": int(len(all_threshold_positions)),
            "raw_detected_count_research_window_before_full_path_check": research_count,
            "events": eligible_count,
            "events_per_month": float(eligible_count / max(1, study_months)),
            "overlap_ratio": overlap_ratio,
            "cooldown_bars": int(window),
        },
        {
            "direction": direction,
            "impulse_window": int(window),
            "threshold": float(threshold),
            "event_set": "deduplicated",
            "raw_detected_count_full_loaded_axis": int(len(all_threshold_positions)),
            "raw_detected_count_research_window_before_full_path_check": research_count,
            "events": dedup_eligible_count,
            "events_per_month": float(dedup_eligible_count / max(1, study_months)),
            "overlap_ratio": overlap_ratio,
            "cooldown_bars": int(window),
        },
    ]

def _write_stream_csv(frame: pd.DataFrame, path: Path, *, first_write: bool) -> None:
    frame.to_csv(
        path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )


def _build_signal_audit(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_id",
        "direction",
        "impulse_window",
        "threshold",
        "signal_time",
        "signal_bar_start",
        "signal_bar_end",
        "entry_time",
        "expected_entry_time",
        "entry_price",
        "expected_entry_price",
        "entry_not_next_open_flag",
        "entry_price_mismatch_flag",
        "signal_source_bar_observed_flag",
        "entry_source_bar_observed_flag",
        "impulse_source_window_observed_flag",
        "full_forward_observed_flag",
        "synthetic_bar_dependency_flag",
        "full_forward_window_flag",
    ]
    out = events[cols].copy()
    out["signal_not_at_bar_end_flag"] = pd.to_datetime(out["signal_time"]) != pd.to_datetime(out["signal_bar_end"])
    out["entry_before_signal_available_flag"] = pd.to_datetime(out["entry_time"]) < pd.to_datetime(out["signal_time"])
    out["lookahead_flag"] = out[
        [
            "entry_not_next_open_flag",
            "entry_price_mismatch_flag",
            "signal_not_at_bar_end_flag",
            "entry_before_signal_available_flag",
            "synthetic_bar_dependency_flag",
        ]
    ].any(axis=1)
    return out


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _classification_for_direction(summary: pd.DataFrame, direction: str) -> tuple[str, pd.DataFrame]:
    part = summary[(summary["direction"] == direction) & (summary["event_set"] == "deduplicated")].copy()
    if part.empty:
        return "no eligible events", part
    robust = part[
        (part["events"] >= 300)
        & (part["mean_net"] > 0)
        & (part["median_net"] > 0)
        & (pd.to_numeric(part["profit_factor"], errors="coerce") > 1.0)
        & (part["positive_year_count"] >= np.maximum(2, part["total_year_count"] - 1))
        & (part["top_5_event_contribution"] < 0.25)
    ]
    if not robust.empty:
        return "cost-after continuation evidence exists in multiple rows; still research-only", robust
    gross_only = part[(part["mean_gross"] > 0) & (part["mean_net"] <= 0)]
    if not gross_only.empty:
        return "some gross continuation exists but normal execution cost removes it", gross_only
    reversal = part[(part["mean_gross"] < 0) & (part["median_gross"] < 0)]
    if len(reversal) >= max(3, len(part) // 3):
        return "descriptive results lean toward reversal/no continuation; reversal must be researched in another Edge", reversal
    return "no stable cost-after continuation evidence in the basic event universe", part


def build_research_brief(
    summary: pd.DataFrame,
    event_counts: pd.DataFrame,
    plateau: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    lines = [
        f"# {TITLE}",
        "",
        "> Automated descriptive brief. This is not an accepted Edge decision and not a strategy backtest.",
        "",
        "## Research question",
        "",
        "After a short-window abnormal directional ETH price impulse, is the forward displacement continuation, reversal, or statistically unstructured after normal costs?",
        "",
        "## Causal and cost assumptions",
        "",
        "- 1m UTC+8 left-labeled bars.",
        "- Closed signal bar; next bar open entry.",
        "- Historical 1m volatility denominator excludes the complete impulse window.",
        f"- Fee-only round trip: {meta['fee_only_cost']:.4%}.",
        f"- Normal fee + slippage round trip: {meta['normal_execution_cost']:.4%}.",
        "- Raw and same-direction cooldown-deduplicated event sets are both reported.",
        "",
        "## Direction-level descriptive result",
        "",
    ]
    for direction in ("LONG", "SHORT"):
        classification, evidence = _classification_for_direction(summary, direction)
        lines.append(f"### {direction}")
        lines.append("")
        lines.append(f"- Classification: **{classification}**.")
        if not evidence.empty:
            display = evidence.sort_values(["mean_net", "events"], ascending=[False, False]).head(8)
            lines.append("- Representative rows (not parameter selection):")
            lines.append("")
            lines.append("| window | threshold | horizon | events | mean net | median net | win rate | PF | positive years | top5 share |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for _, row in display.iterrows():
                pf = pd.to_numeric(pd.Series([row["profit_factor"]]), errors="coerce").iloc[0]
                lines.append(
                    f"| {int(row['impulse_window'])}m | {float(row['threshold']):.2f} | {int(row['horizon'])}m | "
                    f"{int(row['events'])} | {float(row['mean_net']):.4%} | {float(row['median_net']):.4%} | "
                    f"{float(row['win_rate']):.2%} | {float(pf):.3f} | {int(row['positive_year_count'])}/{int(row['total_year_count'])} | "
                    f"{float(row['top_5_event_contribution']):.2%} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Frequency and overlap",
            "",
        ]
    )
    base_counts = event_counts[(event_counts["threshold"] == event_counts["threshold"].min())].copy()
    if not base_counts.empty:
        lines.append("| direction | window | set | events | events/month | overlap ratio |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for _, row in base_counts.sort_values(["direction", "impulse_window", "event_set"]).iterrows():
            lines.append(
                f"| {row['direction']} | {int(row['impulse_window'])}m | {row['event_set']} | {int(row['events'])} | "
                f"{float(row['events_per_month']):.2f} | {float(row['overlap_ratio']):.2%} |"
            )
    lines.extend(
        [
            "",
            "## Threshold plateau check",
            "",
            "`08_threshold_plateau.csv` retains every threshold and reports event retention plus adjacent changes in mean, median, and PF. A valid continuation mechanism should not exist at one isolated threshold only.",
            "",
            "## Interpretation boundaries",
            "",
            "- Positive gross return alone is not an Edge.",
            "- A negative continuation result does not authorize turning this directory into a reversal strategy.",
            "- Feature buckets are descriptive only; none were used to filter events in round 01.",
            "- The next numbered study should be chosen only after reviewing long/short asymmetry, cost survival, yearly consistency, threshold continuity, and top-trade dependency.",
            "",
        ]
    )
    return "\n".join(lines)


def update_research_log(
    log_path: Path,
    summary: pd.DataFrame,
    event_counts: pd.DataFrame,
    brief: str,
    meta: dict[str, Any],
) -> None:
    start_marker = "<!-- AUTO:ROUND_01_RESULTS_START -->"
    end_marker = "<!-- AUTO:ROUND_01_RESULTS_END -->"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation Research Log\n"
    base_counts = event_counts[event_counts["threshold"] == event_counts["threshold"].min()].copy()
    representative = summary[
        (summary["event_set"] == "deduplicated")
        & (summary["threshold"] == summary["threshold"].min())
    ].copy()
    if not representative.empty:
        representative = (
            representative.sort_values(["direction", "impulse_window", "mean_net"], ascending=[True, True, False])
            .groupby(["direction", "impulse_window"], as_index=False, observed=False)
            .head(1)
        )

    generated = [
        start_marker,
        "## Round 01 - Basic impulse event study (auto-updated after local run)",
        "",
        f"- Run completed: `{meta['created_at']}`",
        f"- Data: `{meta['symbol']} {meta['timeframe']}`, `{meta['warmup_start_date']}` warmup, `{meta['research_start']}` to `{meta['research_end']}` research.",
        f"- Cost: fee-only `{meta['fee_only_cost']:.4%}`, normal `{meta['normal_execution_cost']:.4%}`.",
        "- Research question: continuation, reversal, or no stable displacement after abnormal short-window directional movement.",
        "- Change versus previous round: first round; no previous strategy logic or parameter inheritance.",
        "",
        "### Base-threshold frequency",
        "",
        "| direction | window | set | events | monthly frequency | overlap ratio |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for _, row in base_counts.sort_values(["direction", "impulse_window", "event_set"]).iterrows():
        generated.append(
            f"| {row['direction']} | {int(row['impulse_window'])}m | {row['event_set']} | {int(row['events'])} | "
            f"{float(row['events_per_month']):.2f} | {float(row['overlap_ratio']):.2%} |"
        )
    generated.extend(
        [
            "",
            "### Representative base-threshold rows",
            "",
            "The table uses the best normal-cost horizon only to compress the log; all horizons and thresholds remain in the CSV reports and no winner is selected here.",
            "",
            "| direction | window | horizon | events | mean net | median net | win rate | PF | yearly consistency |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in representative.iterrows():
        pf = pd.to_numeric(pd.Series([row["profit_factor"]]), errors="coerce").iloc[0]
        generated.append(
            f"| {row['direction']} | {int(row['impulse_window'])}m | {int(row['horizon'])}m | {int(row['events'])} | "
            f"{float(row['mean_net']):.4%} | {float(row['median_net']):.4%} | {float(row['win_rate']):.2%} | "
            f"{float(pf):.3f} | {int(row['positive_year_count'])}/{int(row['total_year_count'])} |"
        )
    generated.extend(
        [
            "",
            "### Result interpretation",
            "",
            "See `11_research_brief.md` and the complete summary tables. No accepted/rejected decision is written automatically; the report must be reviewed before choosing round 02.",
            "",
            "### Failed branches",
            "",
            "None declared automatically. Round 01 intentionally contains no filters and does not optimize parameters.",
            "",
            "### Next-round reason",
            "",
            "Pending evidence review. If continuation survives normal cost with median/year/plateau support, round 02 should isolate one mechanism such as directional efficiency. If results lean toward reversal, open a different Edge rather than changing this directory's hypothesis.",
            end_marker,
        ]
    )
    generated_text = "\n".join(generated)
    if start_marker in existing and end_marker in existing:
        before = existing.split(start_marker, 1)[0].rstrip()
        after = existing.split(end_marker, 1)[1].lstrip()
        updated = before + "\n\n" + generated_text + ("\n\n" + after if after else "\n")
    else:
        updated = existing.rstrip() + "\n\n" + generated_text + "\n"
    updated = updated.replace("### Current status\n\n`pending_local_run`", "### Current status\n\n`round_01_completed_pending_review`")
    log_path.write_text(updated, encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = _parse_int_csv(args.impulse_windows)
    thresholds = _parse_float_csv(args.thresholds)
    horizons = _parse_int_csv(args.horizons)
    preferred_bucket_horizons = (15, 60, 240)
    bucket_horizons = tuple(h for h in preferred_bucket_horizons if h in horizons) or horizons
    if windows != DEFAULT_WINDOWS:
        print(f"[warning] non-default impulse windows supplied: {windows}", flush=True)
    if thresholds != DEFAULT_THRESHOLDS:
        print(f"[warning] non-default thresholds supplied: {thresholds}", flush=True)
    if horizons != DEFAULT_HORIZONS:
        print(f"[warning] non-default horizons supplied: {horizons}", flush=True)
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "05_events.csv"
    audit_path = out_dir / "06_signal_audit.csv"
    for path in (events_path, audit_path):
        if path.exists():
            path.unlink()

    validation = validate_bars(bars, args)
    start, end_exclusive = _date_bounds(args.start_date, args.end_date)
    max_horizon = max(horizons)
    n = len(bars)
    research_mask = (bars.index >= start) & (bars.index < end_exclusive)
    observed_arr = bars["source_bar_observed_flag"].to_numpy(dtype=bool)
    full_path_mask = np.arange(n, dtype=int) + max_horizon < n
    forward_observed_count = (
        pd.Series(observed_arr, index=bars.index)
        .shift(-1)
        .iloc[::-1]
        .rolling(max_horizon, min_periods=max_horizon)
        .sum()
        .iloc[::-1]
        .to_numpy(dtype=float)
    )
    full_forward_observed_mask = forward_observed_count == float(max_horizon)
    next_bar_mask = np.arange(n, dtype=int) + 1 < n
    expected_next_time = bars.index.to_numpy(dtype="datetime64[ns]") + np.timedelta64(1, "m")
    actual_next_time = np.empty(n, dtype="datetime64[ns]")
    actual_next_time[:] = np.datetime64("NaT")
    actual_next_time[:-1] = bars.index.to_numpy(dtype="datetime64[ns]")[1:]
    next_source_observed = np.zeros(n, dtype=bool)
    next_source_observed[:-1] = observed_arr[1:]
    causal_next_bar_mask = next_bar_mask & (actual_next_time == expected_next_time) & next_source_observed
    eligible_signal_mask = research_mask & observed_arr & full_path_mask & full_forward_observed_mask & causal_next_bar_mask
    study_months = len(pd.period_range(start=start.to_period("M"), end=(end_exclusive - BAR_DELTA).to_period("M"), freq="M"))

    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    if not math.isclose(fee_only_cost, 0.0011, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] fee-only cost differs from project standard 0.11%: {fee_only_cost:.6%}", flush=True)
    if not math.isclose(normal_cost, 0.0015, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] normal cost differs from round-01 standard 0.15%: {normal_cost:.6%}", flush=True)

    path_cache = build_path_cache(bars, horizons, progress_enabled=not args.no_progress)
    log_return, abs_price_change, historical_1m_vol = build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )

    event_count_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    total_combos = len(windows) * len(thresholds) * 2

    print("[feature build] building one impulse window at a time and reusing the historical volatility baseline", flush=True)
    combo_progress = ProgressReporter(
        label="[event detection + summaries] combos",
        total=total_combos,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    )
    combo_done = 0
    minimum_threshold = min(thresholds)
    for window in windows:
        features = build_window_features(bars, window, log_return, abs_price_change, historical_1m_vol)
        norm = features.normalized_impulse
        long_pool = np.flatnonzero(np.isfinite(norm) & (norm >= minimum_threshold))
        short_pool = np.flatnonzero(np.isfinite(norm) & (norm <= -minimum_threshold))

        for direction, side, pool in (("LONG", 1, long_pool), ("SHORT", -1, short_pool)):
            for threshold in thresholds:
                if side == 1:
                    all_positions = pool[norm[pool] >= float(threshold)]
                else:
                    all_positions = pool[norm[pool] <= -float(threshold)]
                dedup_all = _deduplicate_positions(all_positions, int(window))
                research_candidates = research_mask[all_positions] & causal_next_bar_mask[all_positions]
                eligible = eligible_signal_mask[all_positions]
                event_count_rows.extend(
                    _event_count_rows(
                        direction=direction,
                        window=window,
                        threshold=threshold,
                        all_threshold_positions=all_positions,
                        research_window_mask=research_candidates,
                        eligible_mask=eligible,
                        dedup_all=dedup_all,
                        study_months=study_months,
                    )
                )
                positions = all_positions[eligible]
                dedup_flags = dedup_all[eligible]
                if positions.size:
                    event_frame = _build_event_frame(
                        bars=bars,
                        positions=positions,
                        dedup_flags=dedup_flags,
                        direction=direction,
                        side=side,
                        window=window,
                        threshold=threshold,
                        features=features,
                        path_cache=path_cache,
                        full_forward_observed_mask=full_forward_observed_mask,
                        horizons=horizons,
                        fee_cost=fee_only_cost,
                        normal_cost=normal_cost,
                        event_id_start=event_id_cursor,
                    )
                    event_id_cursor += len(event_frame)
                    sr, yr, mr, tr = _summary_rows_for_combo(
                        event_frame,
                        direction=direction,
                        window=window,
                        threshold=threshold,
                        horizons=horizons,
                        study_months=study_months,
                    )
                    summary_rows.extend(sr)
                    yearly_rows.extend(yr)
                    monthly_rows.extend(mr)
                    top_rows.extend(tr)
                    if math.isclose(float(threshold), float(minimum_threshold), rel_tol=0.0, abs_tol=1e-12):
                        bucket_rows.extend(
                            _feature_bucket_rows(
                                event_frame,
                                direction=direction,
                                window=window,
                                threshold=threshold,
                                horizons=bucket_horizons,
                            )
                        )
                    if not args.skip_events_csv:
                        _write_stream_csv(event_frame, events_path, first_write=first_event_write)
                        first_event_write = False
                        audit = _build_signal_audit(event_frame)
                        _write_stream_csv(audit, audit_path, first_write=first_audit_write)
                        first_audit_write = False
                    del event_frame
                combo_done += 1
                combo_progress.update(combo_done)
        del features
    combo_progress.close()

    print("[summaries] assembling event, horizon, yearly, monthly, feature and dependency tables", flush=True)
    event_counts = pd.DataFrame(event_count_rows)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    feature_buckets = pd.DataFrame(bucket_rows)
    top_dependency = pd.DataFrame(top_rows)
    plateau = _build_plateau(summary)

    if first_event_write and not args.skip_events_csv:
        empty_event_cols = [
            "event_id", "direction", "impulse_window", "threshold", "signal_time", "signal_bar_start", "signal_bar_end",
            "entry_time", "expected_entry_time", "entry_price", "expected_entry_price", "entry_not_next_open_flag",
            "entry_price_mismatch_flag", "signal_source_bar_observed_flag", "entry_source_bar_observed_flag",
            "impulse_source_window_observed_flag", "full_forward_observed_flag", "synthetic_bar_dependency_flag",
            *FEATURE_COLUMNS, "raw_event_flag", "deduplicated_event_flag",
        ]
        pd.DataFrame(columns=empty_event_cols).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    if args.skip_events_csv:
        empty_event_cols = [
            "event_id", "direction", "impulse_window", "threshold", "signal_time", "signal_bar_start", "signal_bar_end",
            "entry_time", "expected_entry_time", "entry_price", "expected_entry_price", "entry_not_next_open_flag",
            "entry_price_mismatch_flag", "signal_source_bar_observed_flag", "entry_source_bar_observed_flag",
            "impulse_source_window_observed_flag", "full_forward_observed_flag", "synthetic_bar_dependency_flag",
            *FEATURE_COLUMNS, "raw_event_flag", "deduplicated_event_flag",
        ]
        pd.DataFrame(columns=empty_event_cols).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    meta = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "portfolio_plan": "ETH_NOVA_PORTFOLIO",
        "title": TITLE,
        "status": "research_only_not_tradable",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": "src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader",
        "trade_bar_db_path": str(_trade_bar_db_path(args)),
        "local_cache_only": True,
        "build_missing": False,
        "ordinary_kline_download_enabled": False,
        "timezone_convention": "UTC+8 project convention; timestamps remain timezone-naive",
        "warmup_start_date": args.warmup_start_date,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "effective_latest_signal_bar_start_for_full_path": str(bars.index[np.flatnonzero(eligible_signal_mask)[-1]]) if eligible_signal_mask.any() else None,
        "impulse_windows": list(windows),
        "thresholds": list(thresholds),
        "horizons": list(horizons),
        "feature_bucket_scope": {
            "threshold": float(minimum_threshold),
            "event_set": "deduplicated",
            "horizons": list(bucket_horizons),
            "purpose": "descriptive only; never used as round-01 event filters",
        },
        "normalization": {
            "formula": "impulse_return / (historical rolling std of 1m log returns shifted by impulse_window * sqrt(impulse_window))",
            "vol_lookback_bars": int(args.vol_lookback_bars),
            "vol_min_periods": int(args.vol_min_periods),
            "impulse_excluded_from_baseline": True,
        },
        "deduplication": "within each direction/window/threshold keep first event until impulse_window bars from last kept event",
        "entry_policy": "closed signal bar; next 1m bar open",
        "short_return_formula": "1 - future_price / entry_price",
        "fee_only_cost": fee_only_cost,
        "normal_execution_cost": normal_cost,
        "cost_components": {
            "entry_fee": float(args.entry_fee_rate),
            "exit_fee": float(args.exit_fee_rate),
            "entry_slippage": float(args.entry_slippage),
            "exit_slippage": float(args.exit_slippage),
        },
        "research_month_count": int(study_months),
        "input_rows": int(len(bars)),
        "source_observed_rows": int(bars["source_bar_observed_flag"].sum()),
        "synthetic_gap_bar_count": int((~bars["source_bar_observed_flag"].astype(bool)).sum()),
        "source_gap_segment_count": int(bars.attrs.get("gap_segment_count", 0)),
        "max_gap_minutes": int(bars.attrs.get("max_gap_minutes", 0)),
        "gap_handling": str(bars.attrs.get("gap_policy", "")),
        "full_forward_observed_required_minutes": int(max_horizon),
        "eligible_signal_rows": int(eligible_signal_mask.sum()),
        "total_event_rows_written": int(event_id_cursor - 1),
        "events_csv_skipped_for_development": bool(args.skip_events_csv),
        "validation": validation,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "research_boundary": "standalone price-impulse study; no existing Portfolio inputs or conclusions used",
    }

    brief = build_research_brief(summary, event_counts, plateau, meta)
    artifact_frames = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (summary, out_dir / "02_horizon_summary.csv"),
        (yearly, out_dir / "03_yearly.csv"),
        (monthly, out_dir / "04_monthly.csv"),
        (feature_buckets, out_dir / "07_feature_bucket_summary.csv"),
        (plateau, out_dir / "08_threshold_plateau.csv"),
        (top_dependency, out_dir / "09_top_trade_dependency.csv"),
    ]
    print("[artifacts] writing required report files", flush=True)
    with ProgressReporter(
        label="[artifacts] tables",
        total=len(artifact_frames) + 3,
        every=1,
        enabled=not args.no_progress,
    ) as progress:
        done = 0
        for frame, path in artifact_frames:
            frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
            done += 1
            progress.update(done)
        _write_json(meta, out_dir / "10_run_meta.json")
        done += 1
        progress.update(done)
        (out_dir / "11_research_brief.md").write_text(brief, encoding="utf-8")
        done += 1
        progress.update(done)
        log_path = Path(__file__).resolve().parent / "00_research_log.md"
        update_research_log(log_path, summary, event_counts, brief, meta)
        done += 1
        progress.update(done)

    if not args.skip_review_pack:
        print("[review pack] packaging summary artifacts; oversized event CSV files are skipped by project limits", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)

    print(f"[done] report_dir={out_dir}", flush=True)
    return {
        "report_dir": out_dir,
        "events": events_path,
        "audit": audit_path,
        "review_pack": out_dir / "gpt_review_pack.zip",
    }


def _synthetic_bars() -> pd.DataFrame:
    rng = np.random.default_rng(20260711)
    n = 8_000
    index = pd.date_range("2022-12-20 00:00:00", periods=n, freq="1min")
    log_ret = rng.normal(0.0, 0.00055, size=n)
    for pos, shock in ((1800, 0.010), (2600, -0.012), (4100, 0.015), (6200, -0.014)):
        log_ret[pos : pos + 5] += shock / 5.0
        log_ret[pos + 5 : pos + 20] += np.sign(shock) * 0.00035
    close = 1200.0 * np.exp(np.cumsum(log_ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.00045, 0.00015, size=n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=5.0, sigma=0.6, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] running deterministic synthetic end-to-end research", flush=True)
    raw_bars = _synthetic_bars()
    raw_bars = raw_bars.drop(raw_bars.index[3700:3707])
    bars = _regularize_trade_bar_axis(raw_bars)
    log_path = Path(__file__).resolve().parent / "00_research_log.md"
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r01_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.skip_review_pack = True
            args.no_progress = True
            result = run_research(bars, args)
            required = [
            "01_event_counts.csv",
            "02_horizon_summary.csv",
            "03_yearly.csv",
            "04_monthly.csv",
            "05_events.csv",
            "06_signal_audit.csv",
            "07_feature_bucket_summary.csv",
            "08_threshold_plateau.csv",
            "09_top_trade_dependency.csv",
            "10_run_meta.json",
            "11_research_brief.md",
        ]
            missing = [name for name in required if not (result["report_dir"] / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            audit = pd.read_csv(result["report_dir"] / "06_signal_audit.csv")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains lookahead/data-integrity flags")
            meta = json.loads((result["report_dir"] / "10_run_meta.json").read_text(encoding="utf-8"))
            if int(meta.get("synthetic_gap_bar_count", 0)) != 7:
                raise AssertionError("self-test did not preserve the expected seven-minute source gap")
            if not audit.empty and audit["synthetic_bar_dependency_flag"].astype(bool).any():
                raise AssertionError("self-test retained an event that depends on a synthetic gap bar")
            summary = pd.read_csv(result["report_dir"] / "02_horizon_summary.csv")
            if summary.empty:
                raise AssertionError("self-test summary is empty")
    finally:
        if original_log is None:
            log_path.unlink(missing_ok=True)
        else:
            log_path.write_text(original_log, encoding="utf-8")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    bars = load_bars(args)
    run_research(bars, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
