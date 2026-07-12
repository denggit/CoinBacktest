#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: Range-Bar activity directional validation.

Round 11 validates the strongest Round-10 mechanism without combining filters:
Does high Range-Bar formation activity *during the already-closed impulse* create
an executable directional first-passage advantage from the ordinary next-bar
open, or does it merely predict larger two-sided volatility?

Hard boundaries
---------------
- Local OKX 1m trade bars and local OKX Range Bars only.
- No ordinary K-line download and no implicit Range-Bar build.
- Signal is confirmed only after the 1m signal bar closes.
- Reference entry is the next 1m open.
- Range features use only bars with end_ts <= signal_time.
- Primary Range-Bar membership requires start_ts >= impulse_start_time and
  end_ts <= signal_time (fully-contained mode).
- End-time-only membership is retained only as a boundary sensitivity check.
- No macro filter, CVD filter, terminal-pullback filter, TP/SL optimization,
  parameter search, position sizing, or portfolio logic is introduced.

Performance
-----------
- One 1m load and one local read per Range-Bar scale.
- Prefix/order arrays and searchsorted provide O(log N) interval lookup.
- Fully-contained membership is obtained from end-time interval lookup plus
  exact correction for left-boundary crossings, including rare adjacent overlaps
  present in locally cached Range Bars.
- One first-passage scan per direction/impulse-window; thresholds, activity
  buckets, range scales, years and containment modes reuse the same arrays.
- No iterrows over market data and no full-history rescan per variant.
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


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r10 = _load_sibling("10_macro_meso_path_atlas.py", "directional_impulse_round10_for_r11")
r04 = _load_sibling("04_impulse_first_passage_path_study.py", "directional_impulse_round04_for_r11")
r08 = r10.r08
r07 = r10.r07
r02 = r10.r02
r01 = r10.r01

SCRIPT_NAME = "11_range_activity_directional_validation"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R11"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Range Activity Directional Validation"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "11_range_activity_directional_validation"
)

DEFAULT_WINDOWS = (5, 10, 15)
DEFAULT_THRESHOLDS = (1.5, 2.0, 2.5)
DEFAULT_RANGE_PCTS = (0.0015, 0.0020, 0.0025)
DEFAULT_HORIZONS = (5, 15, 30, 60)
DEFAULT_BARRIERS_BPS = (15, 25, 50)
DEFAULT_ACTIVITY_EDGES = (0.0, 0.25, 0.50, 1.00, float("inf"))
DEFAULT_ACTIVITY_LABELS = ("0-0.25", "0.25-0.50", "0.50-1.00", ">1.00")
PRIMARY_CONTAINMENT = "fully_contained"
SENSITIVITY_CONTAINMENT = "end_time_only"


@dataclass
class RangeActivityCache:
    range_pct: float
    tag: str
    start_ns: np.ndarray
    end_ns: np.ndarray
    bar_id: np.ndarray
    overlap_indices: np.ndarray
    data_start: pd.Timestamp
    data_end: pd.Timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate whether Range-Bar activity creates directional first-passage advantage.",
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
    p.add_argument("--range-pcts", default=",".join(map(str, DEFAULT_RANGE_PCTS)))
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--barriers-bps", default=",".join(map(str, DEFAULT_BARRIERS_BPS)))
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage", type=float, default=0.00020)
    p.add_argument("--exit-slippage", type=float, default=0.00020)
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


def _date_ns(values: Iterable[Any]) -> np.ndarray:
    return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _range_db_path(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    return data_dir / str(args.range_bar_db_name)


def _normalize_range_frame(rb: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    required = {"bar_id", "start_ts", "end_ts"}
    missing = sorted(required.difference(rb.columns))
    if missing:
        raise RuntimeError(f"Range-Bar frame {tag} missing required columns: {missing}")
    out = rb.reset_index(drop=True)
    out["start_ts"] = pd.to_datetime(out["start_ts"], errors="coerce")
    out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce")
    out = out.dropna(subset=["start_ts", "end_ts", "bar_id"]).copy()
    if out.empty:
        return out
    out = out[out["end_ts"] >= out["start_ts"]].copy()
    out["bar_id"] = out["bar_id"].astype("int64")
    end_ns = _date_ns(out["end_ts"])
    bar_ids = out["bar_id"].to_numpy(dtype=np.int64)
    if len(out) > 1:
        ordered = (end_ns[1:] > end_ns[:-1]) | (
            (end_ns[1:] == end_ns[:-1]) & (bar_ids[1:] >= bar_ids[:-1])
        )
        if not bool(np.all(ordered)):
            order = np.lexsort((bar_ids, end_ns))
            out = out.iloc[order].reset_index(drop=True)
    return out


def _load_range_caches(
    args: argparse.Namespace,
    range_pcts: tuple[float, ...],
) -> tuple[dict[float, RangeActivityCache], list[dict[str, Any]]]:
    db_path = _range_db_path(args)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Local Range-Bar DB not found: {db_path}. Round 11 never downloads or builds Range Bars."
        )
    caches: dict[float, RangeActivityCache] = {}
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
        rb = loader.load_local_data(start_date=args.warmup_start_date, end_date=loader_end)
        if rb.empty:
            audit.append({"range_pct": rp, "range_code": tag, "status": "missing_or_empty", "rows": 0})
            print(f"       {tag}: empty/missing; skipped", flush=True)
            continue
        rb = _normalize_range_frame(rb, tag=tag)
        if rb.empty:
            audit.append({"range_pct": rp, "range_code": tag, "status": "invalid_or_empty", "rows": 0})
            continue
        cache = RangeActivityCache(
            range_pct=float(rp),
            tag=tag,
            start_ns=_date_ns(rb["start_ts"]),
            end_ns=_date_ns(rb["end_ts"]),
            bar_id=rb["bar_id"].to_numpy(dtype=np.int64),
            overlap_indices=np.flatnonzero(
                _date_ns(rb["start_ts"])[1:] < _date_ns(rb["end_ts"])[:-1]
            ).astype(np.int64) + 1,
            data_start=pd.Timestamp(rb["end_ts"].min()),
            data_end=pd.Timestamp(rb["end_ts"].max()),
        )
        required_start = pd.Timestamp(args.start_date)
        required_end = r01._inclusive_loader_end(args.end_date)
        coverage_ok = cache.data_start <= required_start and cache.data_end >= required_end - pd.Timedelta(days=1)
        if not coverage_ok:
            audit.append({
                "range_pct": rp, "range_code": tag, "status": "partial_coverage_skipped",
                "rows": len(rb), "data_start": str(cache.data_start), "data_end": str(cache.data_end),
                "required_start": str(required_start), "required_end": str(required_end),
                "local_only": True, "auto_build_disabled": True,
            })
            print(f"       {tag}: partial coverage; skipped", flush=True)
            continue
        # The public cache is ordered by end_ts, but rare adjacent overlaps can
        # exist in locally prebuilt data (for example around raw-file boundaries).
        # They are not treated as fatal: strict containment applies an exact
        # correction for every overlapping bar beyond the ordinary first
        # left-boundary crossing.
        overlap_count = int(cache.overlap_indices.size)
        caches[float(rp)] = cache
        audit.append({
            "range_pct": rp, "range_code": tag, "status": "loaded", "rows": len(rb),
            "data_start": str(cache.data_start), "data_end": str(cache.data_end),
            "adjacent_overlap_count": overlap_count,
            "overlap_handling": "exact_left_boundary_correction",
            "local_only": True, "auto_build_disabled": True,
        })
        overlap_note = f" overlaps={overlap_count} (exactly corrected)" if overlap_count else ""
        print(f"       {tag}: rows={len(rb):,} range={cache.data_start}->{cache.data_end}{overlap_note}", flush=True)
    if not caches:
        raise RuntimeError("No requested local Range-Bar scale has complete coverage; Round 11 is HOLD")
    return caches, audit


def _activity_arrays(
    bars: pd.DataFrame,
    cache: RangeActivityCache,
    positions: np.ndarray,
    *,
    impulse_window: int,
) -> dict[str, np.ndarray]:
    """Return end-only and exact fully-contained Range-Bar activity.

    The cache is sorted by ``end_ts``. End-time-only membership is therefore a
    pair of ``searchsorted`` lookups. Strict membership additionally requires
    ``start_ts >= impulse_start``. In the normal sequential case only the first
    included Range Bar can cross the left boundary. Rare locally cached adjacent
    overlaps can make later included bars cross it as well; those bar indices are
    precomputed once and corrected exactly here without scanning event paths.
    """
    p = np.asarray(positions, dtype=np.int64)
    signal_ns = _date_ns(bars.index[p] + pd.Timedelta(minutes=1))
    impulse_start_ns = _date_ns(bars.index[p - int(impulse_window)] + pd.Timedelta(minutes=1))
    left_end = np.searchsorted(cache.end_ns, impulse_start_ns, side="right")
    right = np.searchsorted(cache.end_ns, signal_ns, side="right")
    end_count = (right - left_end).astype(np.int32)

    first_valid = left_end < right
    first_crossing = np.zeros(len(p), dtype=bool)
    valid_idx = np.flatnonzero(first_valid)
    if valid_idx.size:
        first_bar_idx = left_end[valid_idx]
        first_crossing[valid_idx] = cache.start_ns[first_bar_idx] < impulse_start_ns[valid_idx]

    extra_overlap_exclusions = np.zeros(len(p), dtype=np.int16)
    # Usually empty; in the user's production cache r0015 has only two such
    # indices across the full multi-year table. Complexity is O(events * rare K),
    # not O(events * bars-in-window).
    for bar_idx in np.asarray(cache.overlap_indices, dtype=np.int64):
        extra = (
            (left_end < bar_idx)
            & (bar_idx < right)
            & (cache.start_ns[bar_idx] < impulse_start_ns)
        )
        extra_overlap_exclusions += extra.astype(np.int16)

    strict_count = (
        end_count.astype(np.int64)
        - first_crossing.astype(np.int64)
        - extra_overlap_exclusions.astype(np.int64)
    )
    if np.any(strict_count < 0):
        raise AssertionError("fully-contained Range-Bar count became negative")
    strict_count = strict_count.astype(np.int32)

    last_idx = right - 1
    last_end = np.full(len(p), np.iinfo(np.int64).min, dtype=np.int64)
    has_any = end_count > 0
    last_end[has_any] = cache.end_ns[last_idx[has_any]]
    first_end = np.full(len(p), np.iinfo(np.int64).min, dtype=np.int64)
    first_start = np.full(len(p), np.iinfo(np.int64).min, dtype=np.int64)
    first_end[has_any] = cache.end_ns[left_end[has_any]]
    first_start[has_any] = cache.start_ns[left_end[has_any]]

    return {
        "end_time_only_count": end_count,
        "fully_contained_count": strict_count,
        "end_time_only_activity": end_count.astype(float) / float(impulse_window),
        "fully_contained_activity": strict_count.astype(float) / float(impulse_window),
        "left_boundary_crossing_flag": first_crossing | (extra_overlap_exclusions > 0),
        "first_boundary_crossing_flag": first_crossing,
        "extra_overlap_exclusion_count": extra_overlap_exclusions,
        "last_available_time_ns": last_end,
        "first_included_start_ns": first_start,
        "first_included_end_ns": first_end,
        "signal_time_ns": signal_ns,
        "impulse_start_time_ns": impulse_start_ns,
    }


def _bucket_codes(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    codes = np.full(len(v), -1, dtype=np.int8)
    valid = np.isfinite(v) & (v >= 0)
    if valid.any():
        codes[valid] = np.digitize(v[valid], DEFAULT_ACTIVITY_EDGES[1:-1], right=False).astype(np.int8)
    return codes


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 0 or np.std(yr) <= 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _forward_arrays(
    bars: pd.DataFrame,
    positions: np.ndarray,
    *,
    side: int,
    horizons: tuple[int, ...],
    path_cache: dict[int, Any],
) -> dict[int, dict[str, np.ndarray]]:
    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    entry = open_arr[positions + 1]
    out: dict[int, dict[str, np.ndarray]] = {}
    for h in horizons:
        gross = float(side) * (close_arr[positions + int(h)] / entry - 1.0)
        if int(side) == 1:
            mfe = path_cache[int(h)].future_high[positions] / entry - 1.0
            mae = path_cache[int(h)].future_low[positions] / entry - 1.0
        else:
            mfe = 1.0 - path_cache[int(h)].future_low[positions] / entry
            mae = 1.0 - path_cache[int(h)].future_high[positions] / entry
        out[int(h)] = {"gross": gross, "mfe": mfe, "mae": mae}
    return out


def _adequate_bucket_codes(codes: np.ndarray, selected: np.ndarray, min_events: int) -> list[int]:
    result: list[int] = []
    for code in range(len(DEFAULT_ACTIVITY_LABELS)):
        if int(np.sum(selected & (codes == code))) >= int(min_events):
            result.append(code)
    return result


def _path_outcomes(
    first_passage: dict[int, dict[str, np.ndarray]],
    forward: dict[int, dict[str, np.ndarray]],
    *,
    barrier_bps: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    fp = first_passage[int(barrier_bps)]
    favorable_first = fp["favorable_first_min"]
    adverse_first = fp["adverse_first_min"]
    h = int(horizon)
    barrier = float(barrier_bps) / 10_000.0
    target_touch = (favorable_first > 0) & (favorable_first <= h)
    stop_touch = (adverse_first > 0) & (adverse_first <= h)
    target_first = target_touch & (~stop_touch | (favorable_first < adverse_first))
    stop_first = stop_touch & (~target_touch | (adverse_first < favorable_first))
    ambiguous = target_touch & stop_touch & (favorable_first == adverse_first)
    neither = ~(target_first | stop_first | ambiguous)
    terminal = forward[h]["gross"]
    conservative = np.where(target_first, barrier, np.where(stop_first | ambiguous, -barrier, terminal))
    optimistic = np.where(target_first | ambiguous, barrier, np.where(stop_first, -barrier, terminal))
    return {
        "favorable_first": favorable_first,
        "adverse_first": adverse_first,
        "target_touch": target_touch,
        "stop_touch": stop_touch,
        "target_first": target_first,
        "stop_first": stop_first,
        "ambiguous": ambiguous,
        "neither": neither,
        "terminal": terminal,
        "conservative": conservative,
        "optimistic": optimistic,
    }


def _summary_stats(
    outcomes: dict[str, np.ndarray],
    forward: dict[int, dict[str, np.ndarray]],
    idx: np.ndarray,
    *,
    horizon: int,
    fee_cost: float,
    normal_cost: float,
) -> dict[str, Any]:
    if idx.size == 0:
        return {"events": 0}
    h = int(horizon)
    target_first = outcomes["target_first"][idx]
    stop_first = outcomes["stop_first"][idx]
    target_touch = outcomes["target_touch"][idx]
    stop_touch = outcomes["stop_touch"][idx]
    ambiguous = outcomes["ambiguous"][idx]
    neither = outcomes["neither"][idx]
    conservative = outcomes["conservative"][idx]
    optimistic = outcomes["optimistic"][idx]
    terminal = outcomes["terminal"][idx]
    mfe = forward[h]["mfe"][idx]
    mae = forward[h]["mae"][idx]
    c = r01._stats(conservative, conservative - fee_cost, conservative - normal_cost, mfe, mae)
    o = r01._stats(optimistic, optimistic - fee_cost, optimistic - normal_cost, mfe, mae)
    fixed = r01._stats(terminal, terminal - fee_cost, terminal - normal_cost, mfe, mae)
    fav = outcomes["favorable_first"][idx]
    adv = outcomes["adverse_first"][idx]
    target_median = float(np.median(fav[target_touch])) if target_touch.any() else np.nan
    stop_median = float(np.median(adv[stop_touch])) if stop_touch.any() else np.nan
    return {
        "events": int(idx.size),
        "target_touch_rate": float(target_touch.mean()),
        "stop_touch_rate": float(stop_touch.mean()),
        "target_first_rate": float(target_first.mean()),
        "stop_first_rate": float(stop_first.mean()),
        "directional_first_passage_gap": float(target_first.mean() - stop_first.mean()),
        "ambiguous_same_bar_rate": float(ambiguous.mean()),
        "neither_hit_rate": float(neither.mean()),
        "median_target_touch_min": target_median,
        "median_stop_touch_min": stop_median,
        "conservative_mean_gross": c["mean_gross"],
        "conservative_mean_net": c["mean_net"],
        "conservative_median_net": c["median_net"],
        "conservative_win_rate": c["win_rate"],
        "conservative_profit_factor": c["profit_factor"],
        "conservative_p05": c["p05"],
        "conservative_p95": c["p95"],
        "conservative_top_1_event_contribution": c["top_1_event_contribution"],
        "conservative_top_5_event_contribution": c["top_5_event_contribution"],
        "optimistic_mean_net": o["mean_net"],
        "optimistic_profit_factor": o["profit_factor"],
        "fixed_time_mean_gross": fixed["mean_gross"],
        "fixed_time_mean_net": fixed["mean_net"],
        "fixed_time_median_net": fixed["median_net"],
        "fixed_time_profit_factor": fixed["profit_factor"],
        "mean_mfe": c["mean_mfe"],
        "mean_mae": c["mean_mae"],
        "excursion_advantage": float(c["mean_mfe"] + c["mean_mae"]),
    }


def _build_decision(
    monotonicity: pd.DataFrame,
    sensitivity: pd.DataFrame,
    yearly: pd.DataFrame,
) -> pd.DataFrame:
    if monotonicity.empty:
        return pd.DataFrame()
    primary = monotonicity[
        (monotonicity["containment_mode"] == PRIMARY_CONTAINMENT)
        & (monotonicity["barrier_bps"] == 25)
        & (monotonicity["horizon"] == 15)
    ].copy()
    rows: list[dict[str, Any]] = []
    for keys, g in primary.groupby(["direction", "range_code"], dropna=False):
        valid = g[g["extreme_comparison_valid"]].copy()
        combos = int(len(valid))
        if combos:
            gap_lift_med = float(valid["high_minus_low_directional_gap"].median())
            target_lift_med = float(valid["high_minus_low_target_first"].median())
            stop_reduction_med = float(valid["low_minus_high_stop_first"].median())
            gross_lift_med = float(valid["high_minus_low_fixed_mean_gross"].median())
            positive_gap_rate = float((valid["high_minus_low_directional_gap"] > 0).mean())
            positive_target_rate = float((valid["high_minus_low_target_first"] > 0).mean())
            no_stop_worsening_rate = float((valid["low_minus_high_stop_first"] >= -0.01).mean())
            windows = int(valid["impulse_window"].nunique())
            thresholds = int(valid["threshold"].nunique())
            min_extreme = int(valid[["low_events", "high_events"]].min(axis=1).min())
            positive_net_combos = int((valid["high_conservative_mean_net"] > 0).sum())
        else:
            gap_lift_med = target_lift_med = stop_reduction_med = gross_lift_med = np.nan
            positive_gap_rate = positive_target_rate = no_stop_worsening_rate = np.nan
            windows = thresholds = min_extreme = positive_net_combos = 0
        sens = sensitivity[
            (sensitivity["direction"] == keys[0]) & (sensitivity["range_code"] == keys[1])
            & (sensitivity["barrier_bps"] == 25) & (sensitivity["horizon"] == 15)
        ]
        sensitivity_sign_agreement = float(sens["directional_gap_lift_sign_agreement"].mean()) if len(sens) else np.nan

        year_positive: list[bool] = []
        year_labels: set[int] = set()
        if not yearly.empty and combos:
            for rec in valid.itertuples(index=False):
                yg = yearly[
                    (yearly["direction"] == rec.direction)
                    & (yearly["impulse_window"] == rec.impulse_window)
                    & (yearly["threshold"] == rec.threshold)
                    & (yearly["range_code"] == rec.range_code)
                ]
                for year, gy in yg.groupby("year"):
                    low_y = gy[gy["activity_bucket"] == rec.low_bucket]
                    high_y = gy[gy["activity_bucket"] == rec.high_bucket]
                    if low_y.empty or high_y.empty:
                        continue
                    low_row = low_y.iloc[0]
                    high_row = high_y.iloc[0]
                    if min(int(low_row.get("events", 0)), int(high_row.get("events", 0))) < 25:
                        continue
                    lift = float(high_row["directional_first_passage_gap"] - low_row["directional_first_passage_gap"])
                    if np.isfinite(lift):
                        year_positive.append(lift > 0)
                        year_labels.add(int(year))
        year_positive_rate = float(np.mean(year_positive)) if year_positive else np.nan
        year_comparisons = int(len(year_positive))
        year_coverage = int(len(year_labels))

        status = "insufficient_evidence"
        if combos >= 6 and windows >= 2 and thresholds >= 2:
            if (
                gap_lift_med >= 0.05
                and positive_gap_rate >= 0.75
                and positive_target_rate >= 0.75
                and no_stop_worsening_rate >= 0.60
                and (not np.isfinite(sensitivity_sign_agreement) or sensitivity_sign_agreement >= 0.75)
                and (not np.isfinite(year_positive_rate) or year_positive_rate >= 0.65)
            ):
                status = "retain_directional_mechanism"
            elif (
                target_lift_med >= 0.05
                and abs(gap_lift_med) < 0.02
                and stop_reduction_med < 0.01
            ):
                status = "volatility_only_not_directional"
            elif gap_lift_med <= 0 and positive_gap_rate <= 0.40:
                status = "reject_directional_mechanism"
            else:
                status = "weak_or_mixed"
        rows.append({
            "direction": keys[0], "range_code": keys[1],
            "valid_window_threshold_combos": combos,
            "window_coverage": windows, "threshold_coverage": thresholds,
            "min_extreme_bucket_events": min_extreme,
            "median_directional_gap_lift_high_minus_low": gap_lift_med,
            "median_target_first_lift_high_minus_low": target_lift_med,
            "median_stop_first_reduction_low_minus_high": stop_reduction_med,
            "median_fixed_mean_gross_lift_high_minus_low": gross_lift_med,
            "positive_directional_gap_combo_rate": positive_gap_rate,
            "positive_target_first_combo_rate": positive_target_rate,
            "no_material_stop_worsening_combo_rate": no_stop_worsening_rate,
            "containment_sensitivity_sign_agreement": sensitivity_sign_agreement,
            "year_directional_gap_positive_count": int(sum(year_positive)),
            "year_comparisons": year_comparisons,
            "year_coverage": year_coverage,
            "year_directional_gap_positive_rate": year_positive_rate,
            "positive_normal_cost_combos_in_high_bucket": positive_net_combos,
            "status": status,
            "strategy_accepted": False,
        })
    return pd.DataFrame(rows).sort_values(["status", "direction", "range_code"])


def _build_brief(decision: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# Round 11 — Range Activity Directional Validation",
        "",
        "## Scope",
        "",
        "Single-variable causal validation of Range-Bar formation activity. No macro, CVD, terminal-pullback or other filter is combined.",
        "",
        "## Primary test",
        "",
        "- Signal closed -> next 1m open reference entry.",
        "- Primary Range-Bar membership: fully contained inside the impulse interval.",
        "- Primary diagnostic: 25bps symmetric first passage within 15 minutes.",
        "- Directional evidence requires target-first improvement without equivalent stop-first deterioration.",
        "",
        "## Decision matrix",
        "",
    ]
    if decision.empty:
        lines.append("No decision rows were produced.")
    else:
        for row in decision.itertuples(index=False):
            lines.append(
                f"- {row.direction} {row.range_code}: `{row.status}`, "
                f"median directional-gap lift={row.median_directional_gap_lift_high_minus_low:.4f}, "
                f"positive combo rate={row.positive_directional_gap_combo_rate:.1%}, "
                f"positive normal-cost high-bucket combos={row.positive_normal_cost_combos_in_high_bucket}."
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A retained mechanism is not yet a strategy. It must still survive entry/exit design, costs, delays, overlap, yearly splits and walk-forward validation.",
        "",
        "## Data and performance",
        "",
        f"- Loaded Range-Bar scales: {meta.get('range_pcts_loaded')}",
        f"- Event rows written: {meta.get('compact_event_rows_written')}",
        "- Range caches are local-only; no implicit build or download.",
        "- First-passage arrays are built once per direction/window and reused.",
    ])
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation Research Log\n"
    marker = "## Round 11 — Range activity directional validation"
    if marker in text:
        return
    addition = f"""

{marker}

- 研究问题：Round 10 中 Range Bar 单位时间生成密度的路径提升，是真正方向性 first-passage 优势，还是仅代表双向波动扩大？
- 研究假设：如果 activity 是方向机制，高 activity 应提高 target-first 与 stop-first 的差值，并在严格完整落入冲击窗口的 Range Bar 口径下保持。
- 与上一轮相比改变了什么：只保留 Range Bar activity 一个变量；修正结构性空桶比较；加入 end-time-only 与 fully-contained 边界敏感性；从 next-open 重新计算 first-passage、固定时间收益、成本、年度和事件依赖。
- 使用的数据：本地 ETH-USDT-SWAP 1m trade bar；本地 r0015/r0020/r0025 Range Bar；UTC+8 项目时间约定。
- 事件数：等待生产运行。
- 交易数或模拟事件数：等待生产运行，本轮仍为事件级方向验证，不是完整策略回测。
- 月均频率：等待生产运行。
- mean net / median net / win rate / PF：等待生产运行。
- 年度表现：等待生产运行。
- 结果解释：等待生产运行。
- 失败分支：若 high activity 同时提高 target-first 与 stop-first，且 directional gap 不改善，则判定为 volatility-only。
- 下一轮理由：只有 retain_directional_mechanism 才进入单变量入场/持仓结构研究；否则回到 footprint/订单簿或其他机制解剖。
- 因果约束：signal close 后 next-open；Range Bar end_ts <= signal_time；主结果还要求 start_ts >= impulse_start_time；无未来路径进入信号。
- 性能：Range Bar 单次本地读取、searchsorted 区间索引、一次 first-passage 扫描复用全部分层，禁止逐事件和逐组合全量扫描。
- 生成时间：{meta.get('created_at')}
"""
    log_path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def run_research(
    bars: pd.DataFrame,
    args: argparse.Namespace,
    *,
    range_caches_override: dict[float, RangeActivityCache] | None = None,
) -> dict[str, Path]:
    windows = _parse_positive_ints(args.impulse_windows, name="impulse-windows")
    thresholds = _parse_positive_floats(args.thresholds, name="thresholds")
    range_pcts = _parse_positive_floats(args.range_pcts, name="range-pcts")
    horizons = _parse_positive_ints(args.horizons, name="horizons")
    barriers = _parse_positive_ints(args.barriers_bps, name="barriers-bps")
    max_horizon = max(horizons)
    if max_horizon > 120:
        raise ValueError("Round 11 is intentionally bounded to <=120 minutes")
    fee_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_cost + args.entry_slippage + args.exit_slippage)

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
        range_audit = [{
            "range_pct": rp, "range_code": c.tag, "status": "synthetic_override",
            "rows": len(c.end_ns), "data_start": str(c.data_start), "data_end": str(c.data_end),
        } for rp, c in range_caches.items()]

    masks = r02._eligible_masks(bars, args, horizons)
    study_months = int(masks["study_months"])
    log_return, abs_change, historical_vol_series = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )
    path_cache = r01.build_path_cache(bars, horizons, progress_enabled=not args.no_progress)
    n = len(bars)
    min_threshold = min(thresholds)

    count_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    first_passage_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    monotonic_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    compact_rows = 0

    with ProgressReporter(
        label="[range activity validation] direction/windows",
        total=len(windows) * 2,
        every=max(1, int(args.progress_every)), enabled=not args.no_progress,
    ) as progress:
        done = 0
        for window in windows:
            price_features = r01.build_window_features(
                bars, window, log_return, abs_change, historical_vol_series
            )
            norm = price_features.normalized_impulse
            for direction, side in (("LONG", 1), ("SHORT", -1)):
                directed_norm = float(side) * norm
                all_min_positions = np.flatnonzero(
                    np.isfinite(directed_norm) & (directed_norm >= float(min_threshold))
                )
                eligible_positions = all_min_positions[masks["eligible"][all_min_positions]]
                rows, threshold_masks, dedup_masks = r07._event_count_rows(
                    direction=direction, window=window, thresholds=thresholds,
                    all_min_positions=all_min_positions, eligible_positions=eligible_positions,
                    directed_norm=directed_norm, masks=masks, study_months=study_months, n=n,
                )
                count_rows.extend(rows)
                if not len(eligible_positions):
                    done += 1
                    progress.update(done)
                    continue

                activity: dict[str, dict[str, np.ndarray]] = {}
                for rp, cache in range_caches.items():
                    activity[cache.tag] = _activity_arrays(
                        bars, cache, eligible_positions, impulse_window=window
                    )

                forward = _forward_arrays(
                    bars, eligible_positions, side=side, horizons=horizons, path_cache=path_cache
                )
                first_passage = r04._build_first_passage_arrays(
                    bars, eligible_positions, side=side, barriers_bps=barriers,
                    max_horizon=max_horizon, chunk_size=int(args.path_chunk_size),
                    label=f"[first passage] {direction} {window}m chunks",
                    progress_enabled=not args.no_progress,
                )
                years = pd.to_datetime(bars.index[eligible_positions]).year.to_numpy(dtype=int)
                outcomes_cache = {
                    (int(b), int(h)): _path_outcomes(first_passage, forward, barrier_bps=int(b), horizon=int(h))
                    for b in barriers for h in horizons
                }

                for threshold in thresholds:
                    selected = dedup_masks[float(threshold)]
                    for tag, act in activity.items():
                        for mode, values in (
                            (PRIMARY_CONTAINMENT, act["fully_contained_activity"]),
                            (SENSITIVITY_CONTAINMENT, act["end_time_only_activity"]),
                        ):
                            codes = _bucket_codes(values)
                            adequate = _adequate_bucket_codes(codes, selected, int(args.min_bucket_events))
                            for code, label in enumerate(DEFAULT_ACTIVITY_LABELS):
                                idx = np.flatnonzero(selected & (codes == code))
                                activity_rows.append({
                                    "direction": direction, "impulse_window": window,
                                    "threshold": float(threshold), "range_code": tag,
                                    "containment_mode": mode, "activity_bucket": label,
                                    "bucket_order": code, "events": int(idx.size),
                                    "events_per_month": float(idx.size / max(1, study_months)),
                                    "mean_activity": float(np.nanmean(values[idx])) if idx.size else np.nan,
                                    "median_activity": float(np.nanmedian(values[idx])) if idx.size else np.nan,
                                    "left_boundary_crossing_rate": float(np.mean(act["left_boundary_crossing_flag"][idx])) if idx.size else np.nan,
                                    "mean_extra_overlap_exclusions": float(np.mean(act["extra_overlap_exclusion_count"][idx])) if idx.size else np.nan,
                                    "adequate_for_extreme_comparison": code in adequate,
                                })
                                for h in horizons:
                                    gross = forward[int(h)]["gross"]
                                    stats = r01._stats(
                                        gross[idx], gross[idx] - fee_cost, gross[idx] - normal_cost,
                                        forward[int(h)]["mfe"][idx], forward[int(h)]["mae"][idx],
                                    )
                                    fixed_rows.append({
                                        "direction": direction, "impulse_window": window,
                                        "threshold": float(threshold), "range_code": tag,
                                        "containment_mode": mode, "activity_bucket": label,
                                        "bucket_order": code, "horizon": int(h),
                                        "events_per_month": float(idx.size / max(1, study_months)),
                                        **stats,
                                    })
                                for b in barriers:
                                    for h in horizons:
                                        stats = _summary_stats(
                                            outcomes_cache[(int(b), int(h))], forward, idx,
                                            horizon=int(h), fee_cost=fee_cost, normal_cost=normal_cost,
                                        )
                                        first_passage_rows.append({
                                            "direction": direction, "impulse_window": window,
                                            "threshold": float(threshold), "range_code": tag,
                                            "containment_mode": mode, "activity_bucket": label,
                                            "bucket_order": code, "barrier_bps": int(b),
                                            "horizon": int(h),
                                            "events_per_month": float(idx.size / max(1, study_months)),
                                            **stats,
                                        })
                                        if mode == PRIMARY_CONTAINMENT and b == 25 and h == 15:
                                            for year in sorted(np.unique(years[selected])):
                                                yidx = np.flatnonzero(selected & (codes == code) & (years == year))
                                                ys = _summary_stats(
                                                    outcomes_cache[(25, 15)], forward, yidx,
                                                    horizon=15, fee_cost=fee_cost, normal_cost=normal_cost,
                                                )
                                                yearly_rows.append({
                                                    "direction": direction, "impulse_window": window,
                                                    "threshold": float(threshold), "range_code": tag,
                                                    "containment_mode": mode, "activity_bucket": label,
                                                    "bucket_order": code, "year": int(year), **ys,
                                                })

                            for b in barriers:
                                for h in horizons:
                                    outcomes = outcomes_cache[(int(b), int(h))]
                                    base_idx = np.flatnonzero(selected & np.isfinite(values))
                                    low_code = adequate[0] if len(adequate) >= 2 else -1
                                    high_code = adequate[-1] if len(adequate) >= 2 else -1
                                    low_idx = np.flatnonzero(selected & (codes == low_code)) if low_code >= 0 else np.array([], dtype=int)
                                    high_idx = np.flatnonzero(selected & (codes == high_code)) if high_code >= 0 else np.array([], dtype=int)
                                    low = _summary_stats(outcomes, forward, low_idx, horizon=int(h), fee_cost=fee_cost, normal_cost=normal_cost)
                                    high = _summary_stats(outcomes, forward, high_idx, horizon=int(h), fee_cost=fee_cost, normal_cost=normal_cost)
                                    tf = outcomes["target_first"].astype(float)
                                    sf = outcomes["stop_first"].astype(float)
                                    edge_indicator = tf - sf
                                    gross = forward[int(h)]["gross"]
                                    valid_extreme = len(adequate) >= 2
                                    monotonic_rows.append({
                                        "direction": direction, "impulse_window": window,
                                        "threshold": float(threshold), "range_code": tag,
                                        "containment_mode": mode, "barrier_bps": int(b),
                                        "horizon": int(h), "events": int(base_idx.size),
                                        "adequate_bucket_count": len(adequate),
                                        "low_bucket": DEFAULT_ACTIVITY_LABELS[low_code] if low_code >= 0 else None,
                                        "high_bucket": DEFAULT_ACTIVITY_LABELS[high_code] if high_code >= 0 else None,
                                        "low_events": int(low.get("events", 0)),
                                        "high_events": int(high.get("events", 0)),
                                        "extreme_comparison_valid": valid_extreme,
                                        "spearman_activity_vs_target_first": _rank_correlation(values[base_idx], tf[base_idx]),
                                        "spearman_activity_vs_stop_first": _rank_correlation(values[base_idx], sf[base_idx]),
                                        "spearman_activity_vs_directional_indicator": _rank_correlation(values[base_idx], edge_indicator[base_idx]),
                                        "spearman_activity_vs_fixed_gross": _rank_correlation(values[base_idx], gross[base_idx]),
                                        "high_minus_low_target_first": float(high.get("target_first_rate", np.nan) - low.get("target_first_rate", np.nan)),
                                        "low_minus_high_stop_first": float(low.get("stop_first_rate", np.nan) - high.get("stop_first_rate", np.nan)),
                                        "high_minus_low_directional_gap": float(high.get("directional_first_passage_gap", np.nan) - low.get("directional_first_passage_gap", np.nan)),
                                        "high_minus_low_fixed_mean_gross": float(high.get("fixed_time_mean_gross", np.nan) - low.get("fixed_time_mean_gross", np.nan)),
                                        "low_directional_gap": low.get("directional_first_passage_gap", np.nan),
                                        "high_directional_gap": high.get("directional_first_passage_gap", np.nan),
                                        "high_conservative_mean_net": high.get("conservative_mean_net", np.nan),
                                        "high_conservative_profit_factor": high.get("conservative_profit_factor", np.nan),
                                    })

                if not args.skip_events_csv:
                    selected = dedup_masks[float(min_threshold)]
                    idx = np.flatnonzero(selected)
                    pos = eligible_positions[idx]
                    event = pd.DataFrame({
                        "event_id": np.arange(event_id_cursor, event_id_cursor + len(idx), dtype=np.int64),
                        "direction": direction,
                        "impulse_window": int(window),
                        "signal_bar_start": bars.index[pos],
                        "signal_bar_end": bars.index[pos] + pd.Timedelta(minutes=1),
                        "signal_time": bars.index[pos] + pd.Timedelta(minutes=1),
                        "entry_time": bars.index[pos + 1],
                        "expected_entry_time": bars.index[pos] + pd.Timedelta(minutes=1),
                        "entry_price": pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)[pos + 1],
                        "normalized_impulse": directed_norm[pos],
                    })
                    for t in thresholds:
                        event[f"threshold_ge_{str(t).replace('.', '_')}"] = threshold_masks[float(t)][idx]
                    for tag, act in activity.items():
                        for key in (
                            "end_time_only_count", "fully_contained_count", "end_time_only_activity",
                            "fully_contained_activity", "left_boundary_crossing_flag",
                            "extra_overlap_exclusion_count",
                        ):
                            event[f"range_{tag}_{key}"] = act[key][idx]
                        event[f"range_{tag}_last_available_time"] = pd.to_datetime(act["last_available_time_ns"][idx])
                    for h in horizons:
                        event[f"forward_return_{int(h)}m"] = forward[int(h)]["gross"][idx]
                        event[f"mfe_{int(h)}m"] = forward[int(h)]["mfe"][idx]
                        event[f"mae_{int(h)}m"] = forward[int(h)]["mae"][idx]
                    for b in barriers:
                        tag_b = f"{int(b)}bps"
                        event[f"favorable_first_{tag_b}_min"] = first_passage[int(b)]["favorable_first_min"][idx]
                        event[f"adverse_first_{tag_b}_min"] = first_passage[int(b)]["adverse_first_min"][idx]
                    event_id_cursor += len(event)
                    compact_rows += len(event)
                    r01._write_stream_csv(event, events_path, first_write=first_event_write)
                    first_event_write = False
                    audit = event[["event_id", "signal_time", "entry_time", "expected_entry_time"]].copy()
                    audit["entry_not_next_open_flag"] = audit["entry_time"] != audit["expected_entry_time"]
                    range_flags = []
                    for tag in activity:
                        col = f"range_{tag}_last_available_time"
                        audit[col] = event[col]
                        flag = pd.to_datetime(audit[col], errors="coerce") > pd.to_datetime(audit["signal_time"])
                        audit[f"range_{tag}_future_available_time_flag"] = flag.fillna(False)
                        range_flags.append(audit[f"range_{tag}_future_available_time_flag"].to_numpy(dtype=bool))
                    audit["lookahead_flag"] = audit["entry_not_next_open_flag"].to_numpy(dtype=bool)
                    if range_flags:
                        audit["lookahead_flag"] |= np.logical_or.reduce(range_flags)
                    if bool(audit["lookahead_flag"].any()):
                        raise AssertionError(f"Round 11 lookahead audit failed: {audit[audit['lookahead_flag']].head().to_dict('records')}")
                    r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                    first_audit_write = False

                done += 1
                progress.update(done)
            del price_features

    event_counts = pd.DataFrame(count_rows)
    activity_distribution = pd.DataFrame(activity_rows)
    first_passage_summary = pd.DataFrame(first_passage_rows)
    fixed_horizon_summary = pd.DataFrame(fixed_rows)
    monotonicity = pd.DataFrame(monotonic_rows)
    yearly = pd.DataFrame(yearly_rows)

    # Compare the sign and magnitude of strict vs end-time-only extreme lifts.
    sensitivity = pd.DataFrame()
    if not monotonicity.empty:
        keys = ["direction", "impulse_window", "threshold", "range_code", "barrier_bps", "horizon"]
        strict = monotonicity[monotonicity["containment_mode"] == PRIMARY_CONTAINMENT][
            keys + ["high_minus_low_directional_gap", "high_minus_low_target_first", "low_minus_high_stop_first"]
        ].rename(columns={
            "high_minus_low_directional_gap": "strict_directional_gap_lift",
            "high_minus_low_target_first": "strict_target_first_lift",
            "low_minus_high_stop_first": "strict_stop_first_reduction",
        })
        loose = monotonicity[monotonicity["containment_mode"] == SENSITIVITY_CONTAINMENT][
            keys + ["high_minus_low_directional_gap", "high_minus_low_target_first", "low_minus_high_stop_first"]
        ].rename(columns={
            "high_minus_low_directional_gap": "end_only_directional_gap_lift",
            "high_minus_low_target_first": "end_only_target_first_lift",
            "low_minus_high_stop_first": "end_only_stop_first_reduction",
        })
        sensitivity = strict.merge(loose, on=keys, how="outer")
        sensitivity["directional_gap_lift_difference_strict_minus_end_only"] = (
            sensitivity["strict_directional_gap_lift"] - sensitivity["end_only_directional_gap_lift"]
        )
        sensitivity["directional_gap_lift_sign_agreement"] = (
            np.sign(sensitivity["strict_directional_gap_lift"]) == np.sign(sensitivity["end_only_directional_gap_lift"])
        )

    decision = _build_decision(monotonicity, sensitivity, yearly)
    long_short = pd.DataFrame()
    if not decision.empty:
        long = decision[decision["direction"] == "LONG"].drop(columns=["direction"]).add_prefix("long_")
        short = decision[decision["direction"] == "SHORT"].drop(columns=["direction"]).add_prefix("short_")
        long_short = long.merge(short, left_on="long_range_code", right_on="short_range_code", how="outer")

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "impulse_window"]).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    meta = {
        "script_name": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID,
        "portfolio_plan": "ETH_NOVA_PORTFOLIO", "title": TITLE,
        "status": "research_only_single_mechanism_validation",
        "symbol": args.symbol, "timeframe": args.timeframe,
        "data_source": "local OKX 1m trade bars + local OKX Range Bars",
        "ordinary_kline_download_enabled": False, "trade_bar_build_missing": False,
        "range_bar_auto_build_enabled": False, "range_bar_db_path": str(_range_db_path(args)),
        "range_pcts_requested": list(range_pcts), "range_pcts_loaded": sorted(range_caches.keys()),
        "range_cache_audit": range_audit,
        "warmup_start_date": args.warmup_start_date, "research_start": args.start_date,
        "research_end": args.end_date, "impulse_windows": list(windows),
        "thresholds": list(thresholds), "horizons": list(horizons),
        "barriers_bps": list(barriers), "fee_only_cost": fee_cost,
        "normal_execution_cost": normal_cost,
        "primary_containment_mode": PRIMARY_CONTAINMENT,
        "sensitivity_containment_mode": SENSITIVITY_CONTAINMENT,
        "activity_bucket_edges_per_minute": [0.0, 0.25, 0.50, 1.00, "inf"],
        "activity_bucket_comparison": "lowest and highest predeclared bucket with >= min_bucket_events; empty structural buckets are skipped",
        "primary_diagnostic": "25bps symmetric first passage within 15m",
        "combined_filters_tested": False, "parameter_optimization_performed": False,
        "strategy_backtest_performed": False,
        "range_context_available": "range_bar.end_ts <= signal_time; primary additionally start_ts >= impulse_start_time",
        "reference_entry": "signal closed -> next 1m open",
        "compact_event_rows_written": int(compact_rows), "input_rows": int(len(bars)),
        "validation": validation, "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    brief = _build_brief(decision, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (activity_distribution, out_dir / "02_activity_distribution.csv"),
        (first_passage_summary, out_dir / "03_first_passage_summary.csv"),
        (fixed_horizon_summary, out_dir / "04_fixed_horizon_summary.csv"),
        (monotonicity, out_dir / "05_continuous_monotonicity.csv"),
        (yearly, out_dir / "06_yearly_stability.csv"),
        (sensitivity, out_dir / "07_containment_sensitivity.csv"),
        (decision, out_dir / "08_mechanism_decision_matrix.csv"),
        (long_short, out_dir / "09_long_short_comparison.csv"),
    ]
    print("[artifacts] writing Range activity validation", flush=True)
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
        print("[review pack] packaging Range activity validation", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "events": events_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def _synthetic_range_cache(bars: pd.DataFrame, range_pct: float, every: int) -> RangeActivityCache:
    pos = np.arange(every, len(bars) - 1, every, dtype=int)
    end = _date_ns(bars.index[pos] + pd.Timedelta(minutes=1))
    duration_seconds = np.where(np.arange(len(pos)) % 5 == 0, 90, 20)
    start = end - duration_seconds.astype("timedelta64[s]").astype("timedelta64[ns]").astype(np.int64)
    return RangeActivityCache(
        range_pct=float(range_pct), tag=range_code(range_pct), start_ns=start,
        end_ns=end, bar_id=np.arange(1, len(pos) + 1, dtype=np.int64),
        overlap_indices=np.flatnonzero(start[1:] < end[:-1]).astype(np.int64) + 1,
        data_start=pd.Timestamp(end[0]), data_end=pd.Timestamp(end[-1]),
    )


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic Range activity directional validation", flush=True)
    # Exact boundary membership check: first event bar ends inside the impulse but
    # starts before it, so end-only count must exceed fully-contained count by one.
    idx = pd.date_range("2024-01-01", periods=30, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}, index=idx)
    cache = RangeActivityCache(
        range_pct=0.0020, tag="r0020",
        start_ns=_date_ns(pd.to_datetime(["2024-01-01 00:10:30", "2024-01-01 00:11:20"])),
        end_ns=_date_ns(pd.to_datetime(["2024-01-01 00:11:20", "2024-01-01 00:12:00"])),
        bar_id=np.array([1, 2], dtype=np.int64),
        overlap_indices=np.array([], dtype=np.int64),
        data_start=idx[0], data_end=idx[-1],
    )
    act = _activity_arrays(bars, cache, np.array([11]), impulse_window=1)
    if int(act["end_time_only_count"][0]) != 2 or int(act["fully_contained_count"][0]) != 1:
        raise AssertionError(f"containment boundary mismatch: {act}")
    if not bool(act["left_boundary_crossing_flag"][0]):
        raise AssertionError("boundary crossing flag missing")

    # Rare adjacent-overlap contract: the second included bar also starts before
    # the impulse boundary. Strict membership must exclude both bars, while
    # end-time-only membership still counts both.
    overlap_cache = RangeActivityCache(
        range_pct=0.0015, tag="r0015",
        start_ns=_date_ns(pd.to_datetime([
            "2024-01-01 00:10:10", "2024-01-01 00:10:50", "2024-01-01 00:11:40"
        ])),
        end_ns=_date_ns(pd.to_datetime([
            "2024-01-01 00:11:10", "2024-01-01 00:11:20", "2024-01-01 00:12:00"
        ])),
        bar_id=np.array([1, 2, 3], dtype=np.int64),
        overlap_indices=np.array([1], dtype=np.int64),
        data_start=idx[0], data_end=idx[-1],
    )
    overlap_act = _activity_arrays(bars, overlap_cache, np.array([11]), impulse_window=1)
    if int(overlap_act["end_time_only_count"][0]) != 3:
        raise AssertionError(f"overlap end-only mismatch: {overlap_act}")
    if int(overlap_act["fully_contained_count"][0]) != 1:
        raise AssertionError(f"overlap strict containment mismatch: {overlap_act}")
    if int(overlap_act["extra_overlap_exclusion_count"][0]) != 1:
        raise AssertionError(f"overlap correction missing: {overlap_act}")

    raw = r01._synthetic_bars()
    reg = r01._regularize_trade_bar_axis(raw)
    original = vars(args).copy()
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r11_selftest_") as tmp:
            args.out_dir = str(Path(tmp) / "report")
            args.start_date = str(reg.index[500].date())
            args.end_date = str((reg.index[-1] - pd.Timedelta(days=1)).date())
            args.warmup_start_date = str(reg.index[0].date())
            args.impulse_windows = "5,10"
            args.thresholds = "0.3,0.6"
            args.range_pcts = "0.0015,0.0020"
            args.horizons = "5,15"
            args.barriers_bps = "15,25"
            args.vol_lookback_bars = 300
            args.vol_min_periods = 150
            args.min_bucket_events = 2
            args.path_chunk_size = 200
            args.no_progress = True
            args.skip_review_pack = True
            args.skip_events_csv = False
            caches = {
                0.0015: _synthetic_range_cache(reg, 0.0015, 2),
                0.0020: _synthetic_range_cache(reg, 0.0020, 3),
            }
            result = run_research(reg, args, range_caches_override=caches)
            report = Path(result["report_dir"])
            required = [
                "01_event_counts.csv", "02_activity_distribution.csv",
                "03_first_passage_summary.csv", "04_fixed_horizon_summary.csv",
                "05_continuous_monotonicity.csv", "06_yearly_stability.csv",
                "07_containment_sensitivity.csv", "08_mechanism_decision_matrix.csv",
                "09_long_short_comparison.csv", "11_events.csv", "12_signal_audit.csv",
                "13_run_meta.json", "14_research_brief.md",
            ]
            missing = [name for name in required if not (report / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            audit = pd.read_csv(report / "12_signal_audit.csv")
            if len(audit) and bool(audit["lookahead_flag"].astype(bool).any()):
                raise AssertionError("self-test lookahead flag")
            mono = pd.read_csv(report / "05_continuous_monotonicity.csv")
            if mono.empty:
                raise AssertionError("self-test produced no monotonicity rows")
            if not {PRIMARY_CONTAINMENT, SENSITIVITY_CONTAINMENT}.issubset(set(mono["containment_mode"])):
                raise AssertionError("containment modes missing")
            events = pd.read_csv(report / "11_events.csv")
            if events.empty:
                raise AssertionError("self-test produced no compact events")
            fp = pd.read_csv(report / "03_first_passage_summary.csv")
            if fp.empty or not {15, 25}.issubset(set(pd.to_numeric(fp["barrier_bps"], errors="coerce").dropna().astype(int))):
                raise AssertionError("self-test first-passage coverage missing")
    finally:
        for key, value in original.items():
            setattr(args, key, value)
        if original_log:
            log_path.write_text(original_log, encoding="utf-8")
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
