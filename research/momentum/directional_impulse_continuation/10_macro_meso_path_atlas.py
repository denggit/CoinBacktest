#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: macro/meso path atlas (round 10).

Research question
-----------------
Which *pre-existing macro environment* and *impulse-formation structure* change
what happens after a directional impulse?

This round deliberately does not create a trading rule.  Future path labels are
used only as dependent variables for anatomy:

- immediate_runner
- pullback_runner
- directional_failure
- one_sided_continuation
- two_sided_expansion
- muted

All explanatory features are available no later than the impulse signal close.
The study evaluates one mechanism at a time.  It does not combine filters, search
for the best bucket, optimize TP/SL, or promote a strategy.

Data layers
-----------
1. Local OKX 1m trade bars: price, volume, total/large trade flow.
2. Rolling macro context ending *before the impulse starts*: 30m/60m/240m.
3. Local OKX range bars: r0015/r0020/r0025 impulse-formation speed and flow.
4. Ex-post 60m path labels from the existing causal path-anatomy implementation.

Causality
---------
- Signal bar p is left-labelled and becomes available at p + 1 minute.
- Macro context ends at close(p - impulse_window), before the impulse window.
- Meso time/trade features use only closed bars through p.
- Range bars are included only when range_bar.end_ts <= signal_time.
- Reference entry remains p+1 open for path anatomy.
- No future path label is used to construct any feature or event.

Performance
-----------
- One 1m load; local-cache-only.
- Range-bar SQLite reads are local-only via load_local_data; no auto-build.
- Prefix sums provide O(1) event-window flow/range aggregations.
- One bounded-memory path build per direction/window; all thresholds and features
  reuse it.
- No iterrows over market data and no per-variant full-history rescans.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader, range_code  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402


def _load_round08_module():
    path = Path(__file__).resolve().with_name("08_cvd_path_regime_anatomy_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round08_for_r10", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-08 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r08 = _load_round08_module()
r07 = r08.r07
r02 = r08.r02
r01 = r08.r01

SCRIPT_NAME = "10_macro_meso_path_atlas"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R10"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Macro Meso Path Atlas"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "10_macro_meso_path_atlas"
)

DEFAULT_WINDOWS = (5, 10, 15)
DEFAULT_THRESHOLDS = (1.5, 2.0, 2.5)
DEFAULT_MACRO_WINDOWS = (30, 60, 240)
DEFAULT_RANGE_PCTS = (0.0015, 0.0020, 0.0025)
DEFAULT_MAX_PATH = 60
PATH_LABELS = r08.PATH_LABELS


@dataclass(frozen=True)
class BucketSpec:
    mechanism: str
    feature_name: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]


@dataclass
class RangeCache:
    range_pct: float
    tag: str
    end_ns: np.ndarray
    direction: np.ndarray
    duration: np.ndarray
    notional: np.ndarray
    delta: np.ndarray
    large_delta: np.ndarray
    trades: np.ndarray
    prefix_direction: np.ndarray
    prefix_duration: np.ndarray
    prefix_notional: np.ndarray
    prefix_delta: np.ndarray
    prefix_large_delta: np.ndarray
    prefix_trades: np.ndarray
    data_start: pd.Timestamp
    data_end: pd.Timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal macro/meso path atlas for ETH directional impulses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--trade-bar-db-name", default="okx_trade_bars.db")
    p.add_argument("--range-bar-db-name", default="okx_range_bars.db")
    p.add_argument("--impulse-windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    p.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    p.add_argument("--macro-windows", default=",".join(map(str, DEFAULT_MACRO_WINDOWS)))
    p.add_argument("--range-pcts", default=",".join(map(str, DEFAULT_RANGE_PCTS)))
    p.add_argument("--max-path-minutes", type=int, default=DEFAULT_MAX_PATH)
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--path-chunk-size", type=int, default=5000)
    p.add_argument("--min-bucket-events", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--skip-events-csv", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _parse_positive_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(sorted(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_positive_floats(raw: str, *, name: str) -> tuple[float, ...]:
    values = tuple(sorted(dict.fromkeys(float(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any((not math.isfinite(v)) or v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive finite values")
    return values


def _prefix(values: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))


def _range_sum(prefix: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    s = np.asarray(start, dtype=np.int64)
    e = np.asarray(end, dtype=np.int64)
    out = np.full(len(s), np.nan, dtype=np.float64)
    valid = (s >= 0) & (e >= s) & (e + 1 < len(prefix))
    out[valid] = prefix[e[valid] + 1] - prefix[s[valid]]
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast_shapes(n.shape, d.shape), np.nan, dtype=np.float64)
    np.divide(n, d, out=out, where=np.isfinite(n) & np.isfinite(d) & (np.abs(d) > 1e-15))
    return out


def _date_ns(values: Iterable[Any]) -> np.ndarray:
    return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _range_db_path(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    return data_dir / str(args.range_bar_db_name)


def _normalize_loaded_range_bars(rb: pd.DataFrame, *, range_tag: str) -> pd.DataFrame:
    """Normalize ``OKXRangeBarLoader.load_local_data`` output without reinterpreting it.

    The shared loader intentionally returns ``end_ts`` both as the named index and
    as a retained column (``set_index(..., drop=False)``).  Reset the index before
    any column-based operations so pandas does not treat ``end_ts`` as an
    ambiguous label.  The loader already orders rows by ``end_ts, bar_id``; a
    vectorized order check avoids an unnecessary full sort on multi-year caches.
    """
    required = {
        "bar_id",
        "end_ts",
        "direction",
        "duration_seconds",
        "notional",
        "delta_notional",
        "large_delta_notional",
        "trades_count",
    }
    missing = sorted(required.difference(rb.columns))
    if missing:
        raise RuntimeError(f"Range-bar frame {range_tag} missing required columns: {missing}")

    out = rb.reset_index(drop=True)
    out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce")
    out = out.dropna(subset=["end_ts", "bar_id"]).copy()
    if out.empty:
        return out
    out["bar_id"] = out["bar_id"].astype("int64")

    end_ns = out["end_ts"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    bar_ids = out["bar_id"].to_numpy(dtype=np.int64)
    if len(out) > 1:
        ordered = (end_ns[1:] > end_ns[:-1]) | (
            (end_ns[1:] == end_ns[:-1]) & (bar_ids[1:] >= bar_ids[:-1])
        )
        if not bool(np.all(ordered)):
            order = np.lexsort((bar_ids, end_ns))
            out = out.iloc[order].reset_index(drop=True)
    return out


def _load_range_caches(args: argparse.Namespace, range_pcts: tuple[float, ...]) -> tuple[dict[float, RangeCache], list[dict[str, Any]]]:
    db_path = _range_db_path(args)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Local range-bar DB not found: {db_path}. Round 10 never downloads or builds missing range bars."
        )
    caches: dict[float, RangeCache] = {}
    audit: list[dict[str, Any]] = []
    loader_end = r01._inclusive_loader_end(args.end_date)
    print(f"[load] local range bars db={db_path}", flush=True)
    for rp in range_pcts:
        tag = range_code(float(rp))
        loader = OKXRangeBarLoader(
            symbol=args.symbol,
            range_pct=float(rp),
            data_dir=args.data_dir,
            db_name=args.range_bar_db_name,
            align_with_okx_loader_timezone=True,
        )
        # load_local_data is intentionally used instead of fetch_data_by_date_range:
        # no ensure_cached_range call, no download, no implicit build.
        rb = loader.load_local_data(
            start_date=args.warmup_start_date,
            end_date=loader_end,
        )
        if rb.empty:
            audit.append({"range_pct": float(rp), "range_code": tag, "status": "missing_or_empty", "rows": 0})
            print(f"       {tag}: empty/missing table; skipped", flush=True)
            continue
        rb = _normalize_loaded_range_bars(rb, range_tag=tag)
        if rb.empty:
            audit.append({"range_pct": float(rp), "range_code": tag, "status": "invalid_or_empty", "rows": 0})
            print(f"       {tag}: no valid local rows after normalization; skipped", flush=True)
            continue
        ends = _date_ns(rb["end_ts"])
        direction = pd.to_numeric(rb["direction"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        duration = pd.to_numeric(rb["duration_seconds"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        notional = pd.to_numeric(rb["notional"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        delta = pd.to_numeric(rb["delta_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        large_delta = pd.to_numeric(rb["large_delta_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        trades = pd.to_numeric(rb["trades_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        cache = RangeCache(
            range_pct=float(rp), tag=tag, end_ns=ends, direction=direction, duration=duration,
            notional=notional, delta=delta, large_delta=large_delta, trades=trades,
            prefix_direction=_prefix(direction), prefix_duration=_prefix(duration),
            prefix_notional=_prefix(notional), prefix_delta=_prefix(delta),
            prefix_large_delta=_prefix(large_delta), prefix_trades=_prefix(trades),
            data_start=pd.Timestamp(rb["end_ts"].min()), data_end=pd.Timestamp(rb["end_ts"].max()),
        )
        required_start = pd.Timestamp(args.start_date)
        required_end = r01._inclusive_loader_end(args.end_date)
        coverage_ok = (cache.data_start <= required_start) and (cache.data_end >= required_end - pd.Timedelta(days=1))
        if not coverage_ok:
            audit.append({
                "range_pct": float(rp), "range_code": tag, "status": "partial_coverage_skipped",
                "rows": int(len(rb)), "data_start": str(cache.data_start), "data_end": str(cache.data_end),
                "required_start": str(required_start), "required_end": str(required_end),
                "local_only": True, "auto_build_disabled": True,
            })
            print(f"       {tag}: partial coverage {cache.data_start}->{cache.data_end}; skipped", flush=True)
            continue
        caches[float(rp)] = cache
        audit.append({
            "range_pct": float(rp), "range_code": tag, "status": "loaded", "rows": int(len(rb)),
            "data_start": str(cache.data_start), "data_end": str(cache.data_end),
            "required_start": str(required_start), "required_end": str(required_end),
            "local_only": True, "auto_build_disabled": True,
        })
        print(f"       {tag}: rows={len(rb):,} range={cache.data_start}->{cache.data_end}", flush=True)
    if not caches:
        raise RuntimeError(
            f"No requested local range-bar scale was available in {db_path}. "
            "Round 10 is HOLD until range bars are prebuilt."
        )
    return caches, audit


def _trade_feature_arrays(bars: pd.DataFrame, log_return: pd.Series, abs_change: pd.Series) -> dict[str, np.ndarray]:
    flow = r08._flow_arrays(bars)
    observed = bars["source_bar_observed_flag"].astype(bool).to_numpy()
    lr = pd.to_numeric(log_return, errors="coerce").to_numpy(dtype=np.float64)
    ac = pd.to_numeric(abs_change, errors="coerce").to_numpy(dtype=np.float64)
    out = dict(flow)
    out["log_return"] = lr
    out["abs_change"] = ac
    out["log_return_sq_prefix"] = _prefix(np.square(np.nan_to_num(lr, nan=0.0)))
    out["abs_change_prefix"] = _prefix(ac)
    out["observed_prefix"] = _prefix(observed.astype(np.float64))
    return out


def _window_observed(arr: dict[str, np.ndarray], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    count = _range_sum(arr["observed_prefix"], start, end)
    required = end - start + 1
    return (start >= 0) & np.isfinite(count) & (count == required.astype(float))


def _build_macro_rolling_cache(
    bars: pd.DataFrame,
    log_return: pd.Series,
    macro_windows: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    lr2 = pd.to_numeric(log_return, errors="coerce").pow(2)
    out: dict[int, dict[str, np.ndarray]] = {}
    print("[feature build] macro rolling extremes/realized volatility", flush=True)
    with ProgressReporter(label="[macro cache] windows", total=len(macro_windows), every=1, enabled=True) as progress:
        for done, m in enumerate(macro_windows, start=1):
            out[int(m)] = {
                "high": high.rolling(int(m), min_periods=int(m)).max().to_numpy(dtype=float),
                "low": low.rolling(int(m), min_periods=int(m)).min().to_numpy(dtype=float),
                "rv": np.sqrt(lr2.rolling(int(m), min_periods=int(m)).sum()).to_numpy(dtype=float),
            }
            progress.update(done)
    return out


def _macro_features(
    bars: pd.DataFrame,
    arr: dict[str, np.ndarray],
    rolling: dict[int, dict[str, np.ndarray]],
    historical_1m_vol: np.ndarray,
    positions: np.ndarray,
    *,
    side: int,
    impulse_window: int,
    macro_windows: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    p = np.asarray(positions, dtype=np.int64)
    context_end = p - int(impulse_window)
    close = arr["close"]
    features: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    for m in macro_windows:
        m = int(m)
        start = context_end - m
        valid = _window_observed(arr, start + 1, context_end) & (start >= 0)
        ret = np.full(len(p), np.nan, dtype=float)
        ret[valid] = float(side) * (close[context_end[valid]] / close[start[valid]] - 1.0)
        baseline_pos = context_end - m
        baseline = np.full(len(p), np.nan, dtype=float)
        base_valid = valid & (baseline_pos >= 0)
        baseline[base_valid] = historical_1m_vol[baseline_pos[base_valid]] * math.sqrt(m)
        dir_return_z = _safe_div(ret, baseline)
        abs_path = _range_sum(arr["abs_change_prefix"], start + 1, context_end)
        net_change = np.abs(close[context_end] - close[start])
        efficiency = _safe_div(net_change, abs_path)
        rv = rolling[m]["rv"][context_end]
        compression_score = _safe_div(baseline, rv)  # >1 means compressed versus prior history.
        hi = rolling[m]["high"][context_end]
        lo = rolling[m]["low"][context_end]
        loc = _safe_div(close[context_end] - lo, hi - lo)
        dir_loc = loc if int(side) == 1 else 1.0 - loc
        notional = _range_sum(arr["notional_prefix"], start + 1, context_end)
        delta = _range_sum(arr["delta_notional_prefix"], start + 1, context_end)
        dir_pressure = float(side) * _safe_div(delta, notional)
        prefix = f"macro_{m}m"
        features[f"{prefix}_dir_return_z"] = np.where(valid, dir_return_z, np.nan)
        features[f"{prefix}_directional_efficiency"] = np.where(valid, efficiency, np.nan)
        features[f"{prefix}_vol_compression_score"] = np.where(valid, compression_score, np.nan)
        features[f"{prefix}_directional_range_location"] = np.where(valid, dir_loc, np.nan)
        features[f"{prefix}_dir_delta_pressure"] = np.where(valid, dir_pressure, np.nan)
        validity[prefix] = valid
    context_end_time = pd.to_datetime(bars.index[context_end]) + pd.Timedelta(minutes=1)
    features["macro_context_end_time_ns"] = _date_ns(context_end_time)
    return features, validity


def _meso_features(
    bars: pd.DataFrame,
    arr: dict[str, np.ndarray],
    positions: np.ndarray,
    *,
    side: int,
    impulse_window: int,
    price_features: Any,
) -> dict[str, np.ndarray]:
    p = np.asarray(positions, dtype=np.int64)
    w = int(impulse_window)
    start = p - w + 1
    pre_start = p - 2 * w + 1
    pre_end = p - w
    half = max(1, w // 2)
    late_start = p - half + 1
    early_end = late_start - 1
    early_start = p - w + 1

    notional = _range_sum(arr["notional_prefix"], start, p)
    delta = _range_sum(arr["delta_notional_prefix"], start, p)
    large_total = _range_sum(arr["large_notional_prefix"], start, p)
    large_delta = _range_sum(arr["large_delta_notional_prefix"], start, p)
    trades = _range_sum(arr["trades_count_prefix"], start, p)
    pre_notional = _range_sum(arr["notional_prefix"], pre_start, pre_end)
    pre_trades = _range_sum(arr["trades_count_prefix"], pre_start, pre_end)
    early_notional = _range_sum(arr["notional_prefix"], early_start, early_end)
    early_delta = _range_sum(arr["delta_notional_prefix"], early_start, early_end)
    late_notional = _range_sum(arr["notional_prefix"], late_start, p)
    late_delta = _range_sum(arr["delta_notional_prefix"], late_start, p)

    mid = p - half
    close = arr["close"]
    first_ret = float(side) * (close[mid] / close[p - w] - 1.0)
    second_ret = float(side) * (close[p] / close[mid] - 1.0)
    accel_bps = (second_ret - first_ret) * 10_000.0
    dir_pressure = float(side) * _safe_div(delta, notional)
    dir_large_pressure = float(side) * _safe_div(large_delta, large_total)
    delta_accel = float(side) * (_safe_div(late_delta, late_notional) - _safe_div(early_delta, early_notional))
    close_loc = np.asarray(price_features.close_location_in_window[p], dtype=float)
    if int(side) == -1:
        close_loc = 1.0 - close_loc

    return {
        "meso_directional_efficiency": np.asarray(price_features.directional_efficiency[p], dtype=float),
        "meso_distributed_push_score": 1.0 - np.asarray(price_features.largest_bar_contribution[p], dtype=float),
        "meso_terminal_directional_location": close_loc,
        "meso_impulse_acceleration_bps": accel_bps,
        "meso_dir_delta_pressure": dir_pressure,
        "meso_dir_large_delta_pressure": dir_large_pressure,
        "meso_delta_pressure_acceleration": delta_accel,
        "meso_notional_speed_ratio": _safe_div(notional, pre_notional),
        "meso_trade_speed_ratio": _safe_div(trades, pre_trades),
    }


def _range_interval_sums(prefix: np.ndarray, left: np.ndarray, right_exclusive: np.ndarray) -> np.ndarray:
    out = np.full(len(left), np.nan, dtype=float)
    valid = (left >= 0) & (right_exclusive >= left) & (right_exclusive < len(prefix))
    out[valid] = prefix[right_exclusive[valid]] - prefix[left[valid]]
    return out


def _range_features(
    bars: pd.DataFrame,
    cache: RangeCache,
    positions: np.ndarray,
    *,
    side: int,
    impulse_window: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    p = np.asarray(positions, dtype=np.int64)
    index = bars.index
    signal_ns = _date_ns(index[p] + pd.Timedelta(minutes=1))
    start_ns = _date_ns(index[p - int(impulse_window)] + pd.Timedelta(minutes=1))
    mid_ns = start_ns + (signal_ns - start_ns) // 2

    left = np.searchsorted(cache.end_ns, start_ns, side="right")
    right = np.searchsorted(cache.end_ns, signal_ns, side="right")
    mid = np.searchsorted(cache.end_ns, mid_ns, side="right")
    count = (right - left).astype(float)
    valid = count > 0
    dir_sum = _range_interval_sums(cache.prefix_direction, left, right)
    duration_sum = _range_interval_sums(cache.prefix_duration, left, right)
    notional_sum = _range_interval_sums(cache.prefix_notional, left, right)
    delta_sum = _range_interval_sums(cache.prefix_delta, left, right)
    large_delta_sum = _range_interval_sums(cache.prefix_large_delta, left, right)
    trades_sum = _range_interval_sums(cache.prefix_trades, left, right)
    early_count = (mid - left).astype(float)
    late_count = (right - mid).astype(float)

    last_idx = right - 1
    last_valid = valid & (last_idx >= 0) & (last_idx < len(cache.end_ns))
    last_direction = np.full(len(p), np.nan, dtype=float)
    last_duration = np.full(len(p), np.nan, dtype=float)
    last_end_ns = np.full(len(p), np.iinfo(np.int64).min, dtype=np.int64)
    last_direction[last_valid] = cache.direction[last_idx[last_valid]]
    last_duration[last_valid] = cache.duration[last_idx[last_valid]]
    last_end_ns[last_valid] = cache.end_ns[last_idx[last_valid]]

    mean_duration = _safe_div(duration_sum, count)
    tag = cache.tag
    out = {
        f"range_{tag}_bar_count_per_min": count / float(impulse_window),
        f"range_{tag}_directional_ratio": float(side) * _safe_div(dir_sum, count),
        f"range_{tag}_late_speed_ratio": _safe_div(late_count + 0.5, early_count + 0.5),
        f"range_{tag}_terminal_speed_score": _safe_div(mean_duration, last_duration),
        f"range_{tag}_dir_delta_pressure": float(side) * _safe_div(delta_sum, notional_sum),
        f"range_{tag}_dir_large_delta_pressure": float(side) * _safe_div(large_delta_sum, notional_sum),
        f"range_{tag}_trades_per_second": _safe_div(trades_sum, duration_sum),
        f"range_{tag}_last_direction_support": float(side) * last_direction,
        f"range_{tag}_last_available_time_ns": last_end_ns,
    }
    for key, values in out.items():
        if not key.endswith("available_time_ns"):
            out[key] = np.where(valid, values, np.nan)
    return out, valid


def _feature_specs(feature_names: Iterable[str]) -> dict[str, BucketSpec]:
    specs: dict[str, BucketSpec] = {}
    for name in feature_names:
        if name.endswith("_dir_return_z"):
            edges, labels = (-np.inf, -1.0, 0.0, 1.0, np.inf), ("<=-1", "-1-0", "0-1", ">1")
            mechanism = "macro_direction"
        elif name.endswith("_directional_efficiency"):
            edges, labels = (-np.inf, 0.25, 0.50, 0.75, np.inf), ("<=0.25", "0.25-0.50", "0.50-0.75", ">0.75")
            mechanism = "directional_efficiency"
        elif name.endswith("_vol_compression_score"):
            edges, labels = (-np.inf, 0.75, 1.0, 1.5, 2.0, np.inf), ("<=0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0", ">2.0")
            mechanism = "pre_impulse_compression"
        elif name.endswith("_directional_range_location") or name == "meso_terminal_directional_location":
            edges, labels = (-np.inf, 0.25, 0.50, 0.75, np.inf), ("<=0.25", "0.25-0.50", "0.50-0.75", ">0.75")
            mechanism = "directional_location"
        elif "delta_pressure" in name:
            edges, labels = (-np.inf, -0.20, 0.0, 0.20, np.inf), ("<=-0.20", "-0.20-0", "0-0.20", ">0.20")
            mechanism = "orderflow_support"
        elif name == "meso_distributed_push_score":
            edges, labels = (-np.inf, 0.25, 0.50, 0.75, np.inf), ("<=0.25", "0.25-0.50", "0.50-0.75", ">0.75")
            mechanism = "distributed_vs_single_bar"
        elif name == "meso_impulse_acceleration_bps":
            edges, labels = (-np.inf, -10.0, 0.0, 10.0, np.inf), ("<=-10", "-10-0", "0-10", ">10")
            mechanism = "impulse_acceleration"
        elif name.endswith("_speed_ratio") or name in {"meso_notional_speed_ratio", "meso_trade_speed_ratio"}:
            edges, labels = (-np.inf, 0.75, 1.0, 1.5, 2.0, np.inf), ("<=0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0", ">2.0")
            mechanism = "formation_speed"
        elif name.endswith("_bar_count_per_min"):
            edges, labels = (-np.inf, 0.10, 0.25, 0.50, 1.0, np.inf), ("<=0.10", "0.10-0.25", "0.25-0.50", "0.50-1.0", ">1.0")
            mechanism = "range_bar_activity"
        elif name.endswith("_directional_ratio") or name.endswith("_last_direction_support"):
            edges, labels = (-np.inf, 0.0, 0.50, 0.80, np.inf), ("<=0", "0-0.50", "0.50-0.80", ">0.80")
            mechanism = "range_directional_consistency"
        elif name.endswith("_trades_per_second"):
            edges, labels = (-np.inf, 1.0, 3.0, 10.0, 30.0, np.inf), ("<=1", "1-3", "3-10", "10-30", ">30")
            mechanism = "range_trade_intensity"
        else:
            continue
        specs[name] = BucketSpec(mechanism=mechanism, feature_name=name, edges=tuple(edges), labels=tuple(labels))
    return specs


def _summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "std": np.nan}
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p25": float(np.quantile(x, 0.25)), "p75": float(np.quantile(x, 0.75)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
    }


def _bucket_codes(values: np.ndarray, spec: BucketSpec) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    code = np.full(len(x), -1, dtype=np.int16)
    valid = np.isfinite(x)
    if valid.any():
        code[valid] = np.digitize(x[valid], np.asarray(spec.edges[1:-1]), right=False).astype(np.int16)
    return code


def _path_distribution_rows(
    direction: str, window: int, threshold: float, event_set: str,
    selected: np.ndarray, flags: dict[str, np.ndarray], study_months: int,
) -> list[dict[str, Any]]:
    return r08._path_distribution_rows(
        direction=direction, window=window, threshold=threshold, event_set=event_set,
        selected=selected, flags=flags, months=study_months,
    )


def _feature_by_path_rows(
    *, direction: str, window: int, threshold: float, selected: np.ndarray,
    flags: dict[str, np.ndarray], features: dict[str, np.ndarray], layer: str,
) -> list[dict[str, Any]]:
    labels = flags["primary_path_label"]
    base = np.flatnonzero(selected)
    rows: list[dict[str, Any]] = []
    for path_label in PATH_LABELS:
        idx = base[labels[base] == path_label]
        for name, values in features.items():
            if name.endswith("_ns"):
                continue
            rows.append({
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": "deduplicated", "layer": layer, "path_label": path_label,
                "feature_name": name, "events": int(len(idx)), **_summary(values[idx]),
            })
    return rows


def _bucket_rows(
    *, direction: str, window: int, threshold: float, selected: np.ndarray,
    flags: dict[str, np.ndarray], features: dict[str, np.ndarray], specs: dict[str, BucketSpec],
    years: np.ndarray, min_bucket_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    selected_idx = np.flatnonzero(selected)
    primary = flags["primary_path_label"]
    for name, spec in specs.items():
        values = features[name]
        codes = _bucket_codes(values, spec)
        for bucket_order, bucket_label in enumerate(spec.labels):
            idx = selected_idx[codes[selected_idx] == bucket_order]
            total = int(len(idx))
            rec: dict[str, Any] = {
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": "deduplicated", "mechanism": spec.mechanism,
                "feature_name": name, "bucket_order": int(bucket_order), "bucket": bucket_label,
                "events": total, "sufficient_sample": bool(total >= int(min_bucket_events)),
                **_summary(values[idx]),
            }
            for path_label in PATH_LABELS:
                rec[f"{path_label}_rate"] = float(np.mean(primary[idx] == path_label)) if total else np.nan
            rec["early_failure_rate"] = float(np.mean(flags["early_failure"][idx])) if total else np.nan
            rec["runner_rate"] = float(np.mean(flags["immediate_runner"][idx] | flags["pullback_runner"][idx])) if total else np.nan
            rows.append(rec)
        for year in sorted(np.unique(years[selected_idx])):
            yi_base = selected_idx[years[selected_idx] == int(year)]
            for bucket_order, bucket_label in enumerate(spec.labels):
                idx = yi_base[codes[yi_base] == bucket_order]
                total = int(len(idx))
                yearly.append({
                    "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                    "year": int(year), "mechanism": spec.mechanism, "feature_name": name,
                    "bucket_order": int(bucket_order), "bucket": bucket_label, "events": total,
                    "runner_rate": float(np.mean(flags["immediate_runner"][idx] | flags["pullback_runner"][idx])) if total else np.nan,
                    "immediate_runner_rate": float(np.mean(flags["immediate_runner"][idx])) if total else np.nan,
                    "directional_failure_rate": float(np.mean(flags["directional_failure"][idx])) if total else np.nan,
                })
    return rows, yearly


def _decision_matrix(bucket_df: pd.DataFrame, yearly_df: pd.DataFrame, min_bucket_events: int) -> pd.DataFrame:
    if bucket_df.empty:
        return pd.DataFrame()
    combo_keys = ["direction", "impulse_window", "threshold", "mechanism", "feature_name"]
    combo_rows: list[dict[str, Any]] = []
    for keys, g in bucket_df.groupby(combo_keys, dropna=False):
        g = g.sort_values("bucket_order")
        low = g.iloc[0]
        high = g.iloc[-1]
        combo_rows.append({
            **dict(zip(combo_keys, keys)),
            "low_events": int(low["events"]), "high_events": int(high["events"]),
            "runner_lift_high_minus_low": float(high["runner_rate"] - low["runner_rate"]),
            "immediate_lift_high_minus_low": float(high["immediate_runner_rate"] - low["immediate_runner_rate"]),
            "failure_reduction_low_minus_high": float(low["directional_failure_rate"] - high["directional_failure_rate"]),
            "sample_ok": bool(min(low["events"], high["events"]) >= int(min_bucket_events)),
        })
    combo = pd.DataFrame(combo_rows)
    if combo.empty:
        return combo

    yearly_sign: dict[tuple[str, str, str], tuple[int, int]] = {}
    if not yearly_df.empty:
        tmp = yearly_df.copy()
        tmp["runner_hits"] = pd.to_numeric(tmp["runner_rate"], errors="coerce") * pd.to_numeric(tmp["events"], errors="coerce")
        tmp["failure_hits"] = pd.to_numeric(tmp["directional_failure_rate"], errors="coerce") * pd.to_numeric(tmp["events"], errors="coerce")
        agg = tmp.groupby(["direction", "feature_name", "year", "bucket_order"], dropna=False).agg(
            events=("events", "sum"), runner_hits=("runner_hits", "sum"), failure_hits=("failure_hits", "sum")
        ).reset_index()
        agg["runner_rate"] = agg["runner_hits"] / agg["events"].replace(0, np.nan)
        agg["directional_failure_rate"] = agg["failure_hits"] / agg["events"].replace(0, np.nan)
        yrows: list[dict[str, Any]] = []
        for keys, g in agg.groupby(["direction", "feature_name", "year"], dropna=False):
            g = g.sort_values("bucket_order")
            low, high = g.iloc[0], g.iloc[-1]
            if min(low["events"], high["events"]) < max(25, int(min_bucket_events // 4)):
                continue
            yrows.append({
                "direction": keys[0], "feature_name": keys[1], "year": int(keys[2]),
                "runner_positive": bool(high["runner_rate"] > low["runner_rate"]),
                "failure_positive": bool(high["directional_failure_rate"] < low["directional_failure_rate"]),
            })
        if yrows:
            yt = pd.DataFrame(yrows)
            for keys, g in yt.groupby(["direction", "feature_name"], dropna=False):
                yearly_sign[(keys[0], keys[1], "runner")] = (int(g["runner_positive"].sum()), int(len(g)))
                yearly_sign[(keys[0], keys[1], "failure")] = (int(g["failure_positive"].sum()), int(len(g)))

    rows: list[dict[str, Any]] = []
    for keys, g in combo.groupby(["direction", "mechanism", "feature_name"], dropna=False):
        valid = g[g["sample_ok"]].copy()
        total = int(len(valid))
        if total:
            runner_med = float(valid["runner_lift_high_minus_low"].median())
            failure_med = float(valid["failure_reduction_low_minus_high"].median())
            runner_positive_rate = float((valid["runner_lift_high_minus_low"] > 0).mean())
            failure_positive_rate = float((valid["failure_reduction_low_minus_high"] > 0).mean())
            window_coverage = int(valid["impulse_window"].nunique())
            threshold_coverage = int(valid["threshold"].nunique())
            min_extreme_events = int(valid[["low_events", "high_events"]].min(axis=1).min())
        else:
            runner_med = failure_med = runner_positive_rate = failure_positive_rate = np.nan
            window_coverage = threshold_coverage = min_extreme_events = 0
        yr_run = yearly_sign.get((keys[0], keys[2], "runner"), (0, 0))
        yr_fail = yearly_sign.get((keys[0], keys[2], "failure"), (0, 0))
        year_runner_rate = yr_run[0] / yr_run[1] if yr_run[1] else np.nan
        year_failure_rate = yr_fail[0] / yr_fail[1] if yr_fail[1] else np.nan

        status = "insufficient_evidence"
        if total >= 4 and window_coverage >= 2 and threshold_coverage >= 2:
            if (
                runner_med >= 0.02 and failure_med >= 0.02
                and runner_positive_rate >= 2 / 3 and failure_positive_rate >= 2 / 3
                and (not np.isfinite(year_runner_rate) or year_runner_rate >= 0.60)
            ):
                status = "retain_for_causal_validation"
            elif (
                runner_med >= 0.01 and failure_med >= 0.01
                and runner_positive_rate >= 0.55 and failure_positive_rate >= 0.55
            ):
                status = "weak_keep_for_more_anatomy"
            elif runner_med < 0 and failure_med < 0 and runner_positive_rate <= 1 / 3:
                status = "reject_expected_direction"
        rows.append({
            "direction": keys[0], "mechanism": keys[1], "feature_name": keys[2],
            "valid_window_threshold_combos": total, "window_coverage": window_coverage,
            "threshold_coverage": threshold_coverage, "min_extreme_bucket_events": min_extreme_events,
            "median_runner_lift_high_minus_low": runner_med,
            "median_failure_reduction_low_minus_high": failure_med,
            "runner_positive_combo_rate": runner_positive_rate,
            "failure_positive_combo_rate": failure_positive_rate,
            "year_runner_positive_count": yr_run[0], "year_comparisons": yr_run[1],
            "year_runner_positive_rate": year_runner_rate,
            "year_failure_positive_count": yr_fail[0], "year_failure_positive_rate": year_failure_rate,
            "status": status,
            "not_a_strategy_filter": True,
        })
    return pd.DataFrame(rows).sort_values(
        ["status", "direction", "median_runner_lift_high_minus_low"], ascending=[True, True, False]
    )


def _compact_events(
    *, bars: pd.DataFrame, positions: np.ndarray, selected: np.ndarray, direction: str, side: int,
    window: int, thresholds: tuple[float, ...], threshold_masks: dict[float, np.ndarray],
    price_features: Any, flags: dict[str, np.ndarray], macro: dict[str, np.ndarray],
    meso: dict[str, np.ndarray], range_features: dict[str, np.ndarray], event_id_start: int,
) -> pd.DataFrame:
    idx = np.flatnonzero(selected)
    p = positions[idx]
    signal_start = pd.to_datetime(bars.index[p])
    signal_time = signal_start + pd.Timedelta(minutes=1)
    frame = pd.DataFrame({
        "event_id": np.arange(event_id_start, event_id_start + len(idx), dtype=np.int64),
        "direction": direction, "impulse_window": int(window),
        "signal_bar_start": signal_start, "signal_bar_end": signal_time, "signal_time": signal_time,
        "entry_time": pd.to_datetime(bars.index[p + 1]),
        "entry_price": pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)[p + 1],
        "normalized_impulse": float(side) * np.asarray(price_features.normalized_impulse[p], dtype=float),
        "impulse_return": float(side) * np.asarray(price_features.impulse_return[p], dtype=float),
        "primary_path_label": flags["primary_path_label"][idx],
        "immediate_runner_flag": flags["immediate_runner"][idx],
        "pullback_runner_flag": flags["pullback_runner"][idx],
        "directional_failure_flag": flags["directional_failure"][idx],
        "two_sided_expansion_flag": flags["two_sided_expansion"][idx],
        "muted_flag": flags["muted"][idx],
        "macro_context_available_time": pd.to_datetime(macro["macro_context_end_time_ns"][idx]),
    })
    for t in thresholds:
        frame[f"threshold_ge_{str(t).replace('.', '_')}"] = threshold_masks[float(t)][idx]
    for source in (macro, meso, range_features):
        for name, values in source.items():
            if name.endswith("_ns"):
                if "available_time" in name:
                    ns = np.asarray(values[idx], dtype=np.int64)
                    valid = ns > 0
                    dt = np.full(len(ns), np.datetime64("NaT"), dtype="datetime64[ns]")
                    dt[valid] = ns[valid].astype("datetime64[ns]")
                    frame[name.replace("_ns", "")] = pd.to_datetime(dt)
                continue
            frame[name] = np.asarray(values[idx])
    return frame


def _signal_audit(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_id", "direction", "impulse_window", "signal_bar_start", "signal_bar_end",
        "signal_time", "entry_time", "entry_price", "macro_context_available_time",
    ]
    range_available = [c for c in events.columns if c.endswith("last_available_time")]
    out = events[cols + range_available].copy()
    out["expected_entry_time"] = pd.to_datetime(out["signal_time"])
    out["entry_not_next_open_flag"] = pd.to_datetime(out["entry_time"]) != pd.to_datetime(out["expected_entry_time"])
    out["macro_context_available_time_flag"] = (
        pd.to_datetime(out["macro_context_available_time"]) > pd.to_datetime(out["signal_time"])
    )
    range_flags = []
    for c in range_available:
        flag = f"{c}_after_signal_flag"
        out[flag] = pd.to_datetime(out[c], errors="coerce") > pd.to_datetime(out["signal_time"])
        range_flags.append(flag)
    out["future_path_label_used_in_feature_flag"] = False
    out["lookahead_flag"] = (
        out["entry_not_next_open_flag"] | out["macro_context_available_time_flag"]
        | out[range_flags].any(axis=1) if range_flags else out["entry_not_next_open_flag"] | out["macro_context_available_time_flag"]
    )
    return out


def _build_brief(decision: pd.DataFrame, range_audit: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    retained = decision[decision["status"] == "retain_for_causal_validation"] if not decision.empty else pd.DataFrame()
    weak = decision[decision["status"] == "weak_keep_for_more_anatomy"] if not decision.empty else pd.DataFrame()
    rejected = decision[decision["status"] == "reject_expected_direction"] if not decision.empty else pd.DataFrame()
    lines = [
        "# Round 10 — Macro/Meso Path Atlas", "",
        "## Scope", "",
        "This is a path-probability atlas, not a strategy backtest. Future path labels are outcomes only; every macro/meso feature is available by signal close.", "",
        "## Decision summary", "",
        f"- retain_for_causal_validation: {len(retained)} feature/direction rows",
        f"- weak_keep_for_more_anatomy: {len(weak)}",
        f"- reject_expected_direction: {len(rejected)}",
        f"- insufficient_evidence: {int((decision['status'] == 'insufficient_evidence').sum()) if not decision.empty else 0}", "",
        "A retained mechanism still is **not** an entry filter. It only means the expected high-vs-low ordering changed runner/failure path probabilities across adjacent windows, thresholds and years.", "",
        "## Local range-bar coverage", "",
    ]
    for row in range_audit:
        lines.append(f"- {row.get('range_code')}: {row.get('status')} rows={row.get('rows', 0)} {row.get('data_start', '')} -> {row.get('data_end', '')}")
    if not retained.empty:
        lines.extend(["", "## Mechanisms retained for a later causal experiment", ""])
        for row in retained.head(20).itertuples(index=False):
            lines.append(
                f"- {row.direction} / {row.feature_name}: median runner lift "
                f"{row.median_runner_lift_high_minus_low:+.2%}, failure reduction "
                f"{row.median_failure_reduction_low_minus_high:+.2%}, "
                f"combo consistency {row.runner_positive_combo_rate:.0%}."
            )
    lines.extend(["", "## Hard boundaries", "", "- No combined filter.", "- No best bucket selected.", "- No TP/SL or position simulation.", "- No ordinary K-line download.", "- Range bars are local-only; missing cache is HOLD, not silently rebuilt.", "", "## Next-step gate", "", "Only a mechanism marked retain_for_causal_validation may become the single variable in Round 11. If none are retained, do not stack weak variables; deepen footprint/order-book anatomy or reject the current macro/meso hypothesis."])
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation Research Log\n"
    marker = "## Round 10 — Macro/Meso Path Atlas"
    if marker in text:
        return
    block = f"""

{marker}

- 研究问题：哪些冲击前宏观环境与冲击形成结构，会稳定改变立即延续、回踩延续、失败、双向扩张等路径概率？
- 与上一轮相比改变：从单根 post1 价格/CVD 扩展为宏观环境（30/60/240m）+ 中观冲击形成（1m trade flow + range bars）。
- 研究边界：只做单机制路径解剖；不组合过滤、不优化参数、不模拟策略。
- 数据：本地 1m OKX trade bar + 本地 range bars {meta['range_pcts_loaded']}；所有 loader 为 local-only。
- 因果：宏观窗口在 impulse 开始前关闭；range bar end_ts <= signal_time；未来路径仅作为结果标签。
- 当前状态：等待生产运行。
"""
    log_path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def run_research(
    bars: pd.DataFrame,
    args: argparse.Namespace,
    *,
    range_caches_override: dict[float, RangeCache] | None = None,
) -> dict[str, Path]:
    windows = _parse_positive_ints(args.impulse_windows, name="impulse-windows")
    thresholds = _parse_positive_floats(args.thresholds, name="thresholds")
    macro_windows = _parse_positive_ints(args.macro_windows, name="macro-windows")
    range_pcts = _parse_positive_floats(args.range_pcts, name="range-pcts")
    max_path = int(args.max_path_minutes)
    if max_path < 60:
        raise ValueError("max-path-minutes must be at least 60 for fixed path labels")
    if max(macro_windows) + max(windows) >= int(args.vol_lookback_bars) * 2:
        print("[warning] macro history is large relative to volatility lookback", flush=True)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "11_events.csv"
    audit_path = out_dir / "12_signal_audit.csv"
    events_path.unlink(missing_ok=True)
    audit_path.unlink(missing_ok=True)

    validation = r01.validate_bars(bars, args)
    if range_caches_override is None:
        range_caches, range_audit = _load_range_caches(args, range_pcts)
    else:
        range_caches = range_caches_override
        range_audit = [
            {"range_pct": rp, "range_code": cache.tag, "status": "synthetic_override", "rows": len(cache.end_ns), "data_start": str(cache.data_start), "data_end": str(cache.data_end)}
            for rp, cache in range_caches.items()
        ]

    masks = r02._eligible_masks(bars, args, (max_path,))
    study_months = int(masks["study_months"])
    log_return, abs_change, historical_vol_series = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )
    arr = _trade_feature_arrays(bars, log_return, abs_change)
    historical_vol = pd.to_numeric(historical_vol_series, errors="coerce").to_numpy(dtype=float)
    rolling = _build_macro_rolling_cache(bars, log_return, macro_windows)
    n = len(bars)
    min_threshold = min(thresholds)

    count_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    macro_path_rows: list[dict[str, Any]] = []
    meso_path_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    compact_rows = 0

    with ProgressReporter(
        label="[macro-meso atlas] direction/windows", total=len(windows) * 2,
        every=max(1, int(args.progress_every)), enabled=not args.no_progress,
    ) as progress:
        done = 0
        for window in windows:
            price_features = r01.build_window_features(bars, window, log_return, abs_change, historical_vol_series)
            norm = price_features.normalized_impulse
            for direction, side in (("LONG", 1), ("SHORT", -1)):
                directed_norm = float(side) * norm
                all_min_positions = np.flatnonzero(np.isfinite(directed_norm) & (directed_norm >= float(min_threshold)))
                base_eligible = all_min_positions[masks["eligible"][all_min_positions]]

                # Require every macro context window to be fully observed before impulse start.
                macro_end = base_eligible - int(window)
                macro_start = macro_end - max(macro_windows) + 1
                macro_ok = _window_observed(arr, macro_start, macro_end)
                eligible_positions = base_eligible[macro_ok]

                rows, threshold_masks, dedup_masks = r07._event_count_rows(
                    direction=direction, window=window, thresholds=thresholds,
                    all_min_positions=all_min_positions, eligible_positions=eligible_positions,
                    directed_norm=directed_norm, masks=masks, study_months=study_months, n=n,
                )
                for rec in rows:
                    t = float(rec["threshold"])
                    raw_count = int(threshold_masks[t].sum())
                    dedup_count = int(dedup_masks[t].sum())
                    rec["events"] = raw_count if rec["event_set"] == "raw" else dedup_count
                    rec["events_per_month"] = float(rec["events"] / max(1, study_months))
                    rec["overlap_ratio"] = 1.0 - dedup_count / raw_count if raw_count else np.nan
                    rec["macro_context_max_minutes"] = int(max(macro_windows))
                    rec["eligible_after_macro_gap_check"] = int(len(eligible_positions))
                count_rows.extend(rows)

                if len(eligible_positions):
                    macro, _ = _macro_features(
                        bars, arr, rolling, historical_vol, eligible_positions,
                        side=side, impulse_window=window, macro_windows=macro_windows,
                    )
                    meso = _meso_features(
                        bars, arr, eligible_positions, side=side,
                        impulse_window=window, price_features=price_features,
                    )
                    range_features: dict[str, np.ndarray] = {}
                    range_valid: dict[str, int] = {}
                    for rp, cache in range_caches.items():
                        rf, valid = _range_features(
                            bars, cache, eligible_positions, side=side, impulse_window=window,
                        )
                        range_features.update(rf)
                        range_valid[cache.tag] = int(valid.sum())

                    feature_set = {
                        **{k: v for k, v in macro.items() if not k.endswith("_ns")},
                        **meso,
                        **{k: v for k, v in range_features.items() if not k.endswith("_ns")},
                    }
                    specs = _feature_specs(feature_set.keys())

                    with tempfile.TemporaryDirectory(prefix=f"dic_r10_{direction.lower()}_{window}m_") as tmp:
                        path = r07._build_path_memmaps(
                            bars, eligible_positions, side=side, max_path=max_path,
                            chunk_size=int(args.path_chunk_size), tmp_dir=Path(tmp),
                            label=f"[path build] {direction} {window}m chunks",
                            progress_enabled=not args.no_progress,
                        )
                        descriptors = r07._path_descriptor_arrays(
                            path["close_path"], path["running_mfe"], path["running_mae"],
                            activation_bps=(15, 25, 50), giveback_bps=(10,),
                        )
                        flags = r08._path_flags(descriptors)
                        years = pd.to_datetime(bars.index[eligible_positions]).year.to_numpy(dtype=int)

                        for threshold in thresholds:
                            t = float(threshold)
                            for event_set, selected in (("raw", threshold_masks[t]), ("deduplicated", dedup_masks[t])):
                                path_rows.extend(_path_distribution_rows(
                                    direction, window, t, event_set, selected, flags, study_months
                                ))
                            selected = dedup_masks[t]
                            macro_path_rows.extend(_feature_by_path_rows(
                                direction=direction, window=window, threshold=t, selected=selected,
                                flags=flags, features={k: v for k, v in macro.items() if not k.endswith("_ns")},
                                layer="macro_pre_impulse",
                            ))
                            meso_path_rows.extend(_feature_by_path_rows(
                                direction=direction, window=window, threshold=t, selected=selected,
                                flags=flags, features={**meso, **{k: v for k, v in range_features.items() if not k.endswith("_ns")}},
                                layer="meso_impulse_formation",
                            ))
                            br, yr = _bucket_rows(
                                direction=direction, window=window, threshold=t, selected=selected,
                                flags=flags, features=feature_set, specs=specs, years=years,
                                min_bucket_events=int(args.min_bucket_events),
                            )
                            bucket_rows.extend(br)
                            yearly_rows.extend(yr)

                        if not args.skip_events_csv:
                            selected = dedup_masks[float(min_threshold)]
                            compact = _compact_events(
                                bars=bars, positions=eligible_positions, selected=selected,
                                direction=direction, side=side, window=window, thresholds=thresholds,
                                threshold_masks=threshold_masks, price_features=price_features, flags=flags,
                                macro=macro, meso=meso, range_features=range_features,
                                event_id_start=event_id_cursor,
                            )
                            event_id_cursor += len(compact)
                            compact_rows += len(compact)
                            r01._write_stream_csv(compact, events_path, first_write=first_event_write)
                            first_event_write = False
                            audit = _signal_audit(compact)
                            if bool(audit["lookahead_flag"].any()):
                                sample = audit[audit["lookahead_flag"]].head(5).to_dict("records")
                                raise AssertionError(f"Round 10 lookahead audit failed: {sample}")
                            r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                            first_audit_write = False
                        for key in ("close_path", "running_mfe", "running_mae"):
                            path[key]._mmap.close()  # type: ignore[attr-defined]
                        del path, descriptors, flags
                done += 1
                progress.update(done)
            del price_features

    event_counts = pd.DataFrame(count_rows)
    path_distribution = pd.DataFrame(path_rows)
    macro_by_path = pd.DataFrame(macro_path_rows)
    meso_by_path = pd.DataFrame(meso_path_rows)
    buckets = pd.DataFrame(bucket_rows)
    yearly = pd.DataFrame(yearly_rows)
    decision = _decision_matrix(buckets, yearly, int(args.min_bucket_events))

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "impulse_window"]).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    # Range-scale comparison is kept separate, preventing cross-scale filter combinations.
    range_scale = decision[decision["feature_name"].astype(str).str.startswith("range_")].copy() if not decision.empty else pd.DataFrame()
    long_short = pd.DataFrame()
    if not decision.empty:
        keys = ["mechanism", "feature_name"]
        long = decision[decision["direction"] == "LONG"][keys + ["status", "median_runner_lift_high_minus_low", "median_failure_reduction_low_minus_high"]].rename(columns={
            "status": "long_status", "median_runner_lift_high_minus_low": "long_runner_lift",
            "median_failure_reduction_low_minus_high": "long_failure_reduction",
        })
        short = decision[decision["direction"] == "SHORT"][keys + ["status", "median_runner_lift_high_minus_low", "median_failure_reduction_low_minus_high"]].rename(columns={
            "status": "short_status", "median_runner_lift_high_minus_low": "short_runner_lift",
            "median_failure_reduction_low_minus_high": "short_failure_reduction",
        })
        long_short = long.merge(short, on=keys, how="outer")

    meta = {
        "script_name": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID,
        "portfolio_plan": "ETH_NOVA_PORTFOLIO", "title": TITLE,
        "status": "research_only_macro_meso_path_atlas",
        "symbol": args.symbol, "timeframe": args.timeframe,
        "data_source": "local OKX 1m trade bars + local OKX range bars",
        "ordinary_kline_download_enabled": False, "trade_bar_build_missing": False,
        "range_bar_auto_build_enabled": False, "range_bar_db_path": str(_range_db_path(args)),
        "range_pcts_requested": list(range_pcts), "range_pcts_loaded": sorted(range_caches.keys()),
        "range_cache_audit": range_audit,
        "warmup_start_date": args.warmup_start_date, "research_start": args.start_date,
        "research_end": args.end_date, "impulse_windows": list(windows),
        "thresholds": list(thresholds), "macro_windows": list(macro_windows),
        "max_path_minutes": max_path, "path_labels_are_ex_post_only": True,
        "future_path_label_used_in_feature": False,
        "macro_context_available": "close of bar immediately before impulse window begins",
        "range_context_available": "range_bar.end_ts <= signal_time",
        "reference_entry": "signal closed -> p+1 open",
        "combined_filters_tested": False, "parameter_optimization_performed": False,
        "strategy_backtest_performed": False,
        "decision_matrix_rule": "predeclared high-vs-low orientation; no best-bucket selection",
        "min_bucket_events": int(args.min_bucket_events), "path_chunk_size": int(args.path_chunk_size),
        "compact_event_rows_written": int(compact_rows), "input_rows": int(len(bars)),
        "validation": validation, "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    brief = _build_brief(decision, range_audit, meta)

    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (path_distribution, out_dir / "02_path_class_distribution.csv"),
        (macro_by_path, out_dir / "03_macro_feature_by_path.csv"),
        (meso_by_path, out_dir / "04_meso_feature_by_path.csv"),
        (buckets, out_dir / "05_causal_bucket_summary.csv"),
        (yearly, out_dir / "06_yearly_mechanism_stability.csv"),
        (range_scale, out_dir / "07_range_scale_comparison.csv"),
        (decision, out_dir / "08_mechanism_decision_matrix.csv"),
        (long_short, out_dir / "09_long_short_comparison.csv"),
    ]
    print("[artifacts] writing macro/meso path atlas", flush=True)
    with ProgressReporter(label="[artifacts] tables", total=len(artifacts) + 3, every=1, enabled=not args.no_progress) as progress:
        done = 0
        for frame, path in artifacts:
            frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
            done += 1
            progress.update(done)
        r01._write_json(meta, out_dir / "13_run_meta.json")
        done += 1
        progress.update(done)
        (out_dir / "14_research_brief.md").write_text(brief, encoding="utf-8")
        done += 1
        progress.update(done)
        _update_log(Path(__file__).resolve().with_name("00_research_log.md"), meta)
        done += 1
        progress.update(done)

    if not args.skip_review_pack:
        print("[review pack] packaging macro/meso atlas", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "events": events_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def _synthetic_range_cache(bars: pd.DataFrame, range_pct: float, every: int) -> RangeCache:
    positions = np.arange(every, len(bars), every, dtype=int)
    ends = _date_ns(bars.index[positions] + pd.Timedelta(minutes=1))
    direction = np.where(np.arange(len(positions)) % 3 == 0, -1.0, 1.0)
    duration = np.where(np.arange(len(positions)) % 4 == 0, 120.0, 35.0)
    notional = np.full(len(positions), 1_000_000.0)
    delta = direction * 150_000.0
    large_delta = direction * 30_000.0
    trades = np.full(len(positions), 500.0)
    return RangeCache(
        range_pct=range_pct, tag=range_code(range_pct), end_ns=ends,
        direction=direction, duration=duration, notional=notional, delta=delta,
        large_delta=large_delta, trades=trades,
        prefix_direction=_prefix(direction), prefix_duration=_prefix(duration),
        prefix_notional=_prefix(notional), prefix_delta=_prefix(delta),
        prefix_large_delta=_prefix(large_delta), prefix_trades=_prefix(trades),
        data_start=pd.Timestamp(ends[0]), data_end=pd.Timestamp(ends[-1]),
    )


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic macro/meso path atlas", flush=True)

    # Shared range loader keeps end_ts as both index and column.  Reproduce that
    # exact contract so the production load path cannot regress to the pandas
    # ambiguous-label failure seen in the first R10 build.
    duplicate_end_ts = pd.DataFrame(
        {
            "bar_id": [2, 1, 3],
            "end_ts": pd.to_datetime(["2024-01-01 00:02:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00"]),
            "direction": [1.0, -1.0, 1.0],
            "duration_seconds": [10.0, 20.0, 30.0],
            "notional": [1.0, 1.0, 1.0],
            "delta_notional": [0.1, -0.1, 0.2],
            "large_delta_notional": [0.0, 0.0, 0.0],
            "trades_count": [1.0, 1.0, 1.0],
        }
    ).set_index("end_ts", drop=False)
    normalized_rb = _normalize_loaded_range_bars(duplicate_end_ts, range_tag="selftest")
    if normalized_rb.index.name == "end_ts":
        raise AssertionError("range normalization left ambiguous end_ts index")
    if normalized_rb["bar_id"].tolist() != [1, 2, 3]:
        raise AssertionError("range normalization did not preserve end_ts/bar_id ordering")

    raw = r01._synthetic_bars()
    reg = r01._regularize_trade_bar_axis(raw)
    n = len(reg)
    close = pd.to_numeric(reg["close"], errors="coerce").to_numpy(dtype=float)
    ret = pd.Series(close, index=reg.index).pct_change().fillna(0.0).to_numpy()
    notional = np.full(n, 1_000_000.0)
    delta = np.tanh(ret * 200.0) * notional
    reg["notional"] = notional
    reg["buy_notional"] = (notional + delta) / 2.0
    reg["sell_notional"] = (notional - delta) / 2.0
    reg["delta_notional"] = delta
    reg["trades_count"] = 500.0
    reg["large_buy_notional"] = np.maximum(delta, 0.0) * 0.25
    reg["large_sell_notional"] = np.maximum(-delta, 0.0) * 0.25
    reg["large_delta_notional"] = reg["large_buy_notional"] - reg["large_sell_notional"]
    reg.attrs.update(raw.attrs)

    # Exact causal helper checks.
    lr, ac, hv = r01.build_base_volatility(reg, 300, 150)
    arr = _trade_feature_arrays(reg, lr, ac)
    rolling = _build_macro_rolling_cache(reg, lr, (30, 60, 240))
    pos = np.asarray([1000], dtype=int)
    pf = r01.build_window_features(reg, 10, lr, ac, hv)
    macro, _ = _macro_features(reg, arr, rolling, hv.to_numpy(dtype=float), pos, side=1, impulse_window=10, macro_windows=(30, 60, 240))
    expected_context_time = reg.index[990] + pd.Timedelta(minutes=1)
    got_context_time = pd.Timestamp(macro["macro_context_end_time_ns"][0])
    if got_context_time != expected_context_time:
        raise AssertionError(f"macro available time mismatch: {got_context_time} != {expected_context_time}")
    meso = _meso_features(reg, arr, pos, side=1, impulse_window=10, price_features=pf)
    if not np.isfinite(meso["meso_dir_delta_pressure"][0]):
        raise AssertionError("meso flow feature missing")

    caches = {
        0.0015: _synthetic_range_cache(reg, 0.0015, 2),
        0.0020: _synthetic_range_cache(reg, 0.0020, 3),
        0.0025: _synthetic_range_cache(reg, 0.0025, 4),
    }
    rf, valid = _range_features(reg, caches[0.0020], pos, side=1, impulse_window=10)
    if not valid[0] or rf["range_r0020_last_available_time_ns"][0] > float((reg.index[1000] + pd.Timedelta(minutes=1)).value):
        raise AssertionError("range feature used future end_ts")

    original = vars(args).copy()
    original_log = (Path(__file__).resolve().with_name("00_research_log.md").read_text(encoding="utf-8"))
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r10_selftest_") as tmp:
            args.out_dir = str(Path(tmp) / "report")
            args.start_date = str(reg.index[700].date())
            args.end_date = str(reg.index[-80])
            args.warmup_start_date = str(reg.index[0].date())
            args.impulse_windows = "5,10"
            args.thresholds = "0.5,1.0"
            args.macro_windows = "30,60,240"
            args.range_pcts = "0.0015,0.0020,0.0025"
            args.vol_lookback_bars = 300
            args.vol_min_periods = 150
            args.max_path_minutes = 60
            args.path_chunk_size = 200
            args.min_bucket_events = 5
            args.skip_review_pack = True
            args.no_progress = True
            args.skip_events_csv = False
            result = run_research(reg, args, range_caches_override=caches)
            report = result["report_dir"]
            required = [
                "01_event_counts.csv", "02_path_class_distribution.csv",
                "03_macro_feature_by_path.csv", "04_meso_feature_by_path.csv",
                "05_causal_bucket_summary.csv", "06_yearly_mechanism_stability.csv",
                "08_mechanism_decision_matrix.csv", "11_events.csv", "12_signal_audit.csv",
                "13_run_meta.json", "14_research_brief.md",
            ]
            missing = [name for name in required if not (report / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            counts = pd.read_csv(report / "01_event_counts.csv")
            if counts.empty or pd.to_numeric(counts["events"], errors="coerce").fillna(0).sum() <= 0:
                raise AssertionError("self-test produced no events")
            decision = pd.read_csv(report / "08_mechanism_decision_matrix.csv")
            if decision.empty:
                raise AssertionError("self-test produced no mechanism decision rows")
            meso_report = pd.read_csv(report / "04_meso_feature_by_path.csv")
            if not meso_report["feature_name"].astype(str).str.startswith("range_").any():
                raise AssertionError("self-test range-bar features missing")
            audit = pd.read_csv(report / "12_signal_audit.csv")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit failed")
            meta = json.loads((report / "13_run_meta.json").read_text(encoding="utf-8"))
            if meta["combined_filters_tested"] or meta["future_path_label_used_in_feature"]:
                raise AssertionError("research boundary metadata failed")
    finally:
        for key, value in original.items():
            setattr(args, key, value)
        Path(__file__).resolve().with_name("00_research_log.md").write_text(original_log, encoding="utf-8")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    bars = r01.load_bars(args)
    run_research(bars, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
