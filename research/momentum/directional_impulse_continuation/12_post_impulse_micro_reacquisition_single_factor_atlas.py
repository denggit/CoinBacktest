#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: post-impulse micro reacquisition single-factor atlas.

Round 12 asks one question:
Which *single* causal microstructure variable, observed after a 5m/10m/15m
price impulse, earliest separates residual continuation from failure strongly
enough to leave executable room after the variable is known?

Design boundaries
-----------------
- Event anchor remains the existing 1m directional impulse; no macro, Range-Bar,
  footprint, order-book or cross-factor filter is combined with a micro factor.
- Micro timeframes are evaluated independently: 1s, 3s, 5s, 15s.
- Common checkpoints are fixed at 15s, 30s, 60s and 120s so all timeframes are
  compared at identical information times.
- Every micro factor uses bars whose close/available time is <= checkpoint time.
- Reference entry is the next micro bar open at checkpoint time.
- All continuous buckets are predeclared natural bins; no threshold grid search,
  no in-sample winner selection and no cross-factor AND combinations.
- Missing micro cache tables are reported and skipped. No raw trades are
  downloaded and no missing trade bars are built.
- Micro bars are loaded in bounded calendar chunks. Each timeframe is read once
  per chunk and all factors/checkpoints reuse prefix arrays.
- 1m trade-count reconciliation is used as a cache-integrity audit. Events that
  touch a minute where micro trade counts disagree with the local 1m cache are
  excluded for that timeframe.

This is an event/path study, not a strategy backtest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sqlite3
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


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r11 = _load_sibling("11_range_activity_directional_validation.py", "directional_impulse_round11_for_r12")
r10 = r11.r10
r07 = r10.r07
r04 = r11.r04
r02 = r10.r02
r01 = r10.r01

SCRIPT_NAME = "12_post_impulse_micro_reacquisition_single_factor_atlas"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R12"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Micro Reacquisition Single-Factor Atlas"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "12_post_impulse_micro_reacquisition_single_factor_atlas"
)

DEFAULT_WINDOWS = (5, 10, 15)
DEFAULT_THRESHOLDS = (1.5, 2.0, 2.5)
DEFAULT_MICRO_TIMEFRAMES = ("1s", "3s", "5s", "15s")
DEFAULT_CHECKPOINTS = (15, 30, 60, 120)
DEFAULT_HORIZONS = (30, 60, 180, 300, 900)
DEFAULT_BARRIERS_BPS = (15, 25, 50)
DEFAULT_FIRST_PASSAGE_LIMITS = (30, 60, 180, 300)
PRIMARY_HORIZON_SECONDS = 300
PRIMARY_BARRIER_BPS = 25
PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS = 300


@dataclass(frozen=True)
class FactorSpec:
    name: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]
    expected_direction: str = "higher_better"


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        "directional_price_progress_bps",
        (-np.inf, -25.0, -10.0, 0.0, 10.0, 25.0, np.inf),
        ("<=-25", "-25--10", "-10-0", "0-10", "10-25", ">25"),
    ),
    FactorSpec(
        "directional_delta_pressure",
        (-np.inf, -0.20, -0.05, 0.0, 0.05, 0.20, np.inf),
        ("<=-0.20", "-0.20--0.05", "-0.05-0", "0-0.05", "0.05-0.20", ">0.20"),
    ),
    FactorSpec(
        "notional_speed_ratio",
        (-np.inf, 0.50, 1.00, 2.00, 4.00, np.inf),
        ("<=0.50", "0.50-1.00", "1.00-2.00", "2.00-4.00", ">4.00"),
    ),
    FactorSpec(
        "trade_speed_ratio",
        (-np.inf, 0.50, 1.00, 2.00, 4.00, np.inf),
        ("<=0.50", "0.50-1.00", "1.00-2.00", "2.00-4.00", ">4.00"),
    ),
    FactorSpec(
        "signed_path_efficiency",
        (-np.inf, -0.75, -0.25, 0.0, 0.25, 0.75, np.inf),
        ("<=-0.75", "-0.75--0.25", "-0.25-0", "0-0.25", "0.25-0.75", ">0.75"),
    ),
    FactorSpec(
        "price_per_delta_impact",
        (-np.inf, -200.0, -50.0, 0.0, 50.0, 200.0, np.inf),
        ("<=-200", "-200--50", "-50-0", "0-50", "50-200", ">200"),
    ),
    FactorSpec(
        "directional_large_delta_pressure",
        (-np.inf, -0.20, -0.05, 0.0, 0.05, 0.20, np.inf),
        ("<=-0.20", "-0.20--0.05", "-0.05-0", "0-0.05", "0.05-0.20", ">0.20"),
    ),
    FactorSpec(
        "aligned_delta_bar_ratio",
        (-np.inf, 0.25, 0.50, 0.75, np.inf),
        ("<=0.25", "0.25-0.50", "0.50-0.75", ">0.75"),
    ),
    FactorSpec(
        "delta_pressure_acceleration",
        (-np.inf, -0.20, -0.05, 0.0, 0.05, 0.20, np.inf),
        ("<=-0.20", "-0.20--0.05", "-0.05-0", "0-0.05", "0.05-0.20", ">0.20"),
    ),
)


@dataclass
class EventUniverse:
    frame: pd.DataFrame
    threshold_flags: dict[float, np.ndarray]
    count_rows: list[dict[str, Any]]
    study_months: int


@dataclass
class MicroArrays:
    valid: np.ndarray
    entry_time_ns: np.ndarray
    factors: dict[str, np.ndarray]
    gross: dict[int, np.ndarray]
    mfe: dict[int, np.ndarray]
    mae: dict[int, np.ndarray]
    favorable_first_seconds: dict[int, np.ndarray]
    adverse_first_seconds: dict[int, np.ndarray]
    cache_mismatch_flag: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-factor post-impulse microstructure atlas.",
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
    p.add_argument("--impulse-windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    p.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    p.add_argument("--micro-timeframes", default=",".join(DEFAULT_MICRO_TIMEFRAMES))
    p.add_argument("--checkpoints-seconds", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    p.add_argument("--horizons-seconds", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--barriers-bps", default=",".join(map(str, DEFAULT_BARRIERS_BPS)))
    p.add_argument("--first-passage-limits-seconds", default=",".join(map(str, DEFAULT_FIRST_PASSAGE_LIMITS)))
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage", type=float, default=0.00020)
    p.add_argument("--exit-slippage", type=float, default=0.00020)
    p.add_argument("--micro-chunk-days", type=int, default=7)
    p.add_argument("--path-batch-events", type=int, default=2000)
    p.add_argument("--min-bucket-events", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--skip-events-csv", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _parse_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(sorted(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_floats(raw: str, *, name: str) -> tuple[float, ...]:
    values = tuple(sorted(dict.fromkeys(float(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any((not math.isfinite(v)) or v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive finite values")
    return values


def _tf_seconds(tf: str) -> int:
    # Mirror OKXTradeBarLoader's public timeframe grammar without constructing
    # a loader merely to parse it (the loader constructor initializes SQLite).
    text = str(tf).strip()
    if len(text) < 2 or not text[:-1].isdigit():
        raise ValueError(f"invalid micro timeframe: {tf!r}")
    amount = int(text[:-1])
    unit = text[-1].lower()
    if amount <= 0 or unit not in {"s", "m", "h", "d"}:
        raise ValueError(f"invalid micro timeframe: {tf!r}")
    return amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _prefix(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values) + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(np.nan_to_num(values, nan=0.0), out=out[1:])
    return out


def _range_sum(prefix: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return prefix[end] - prefix[start]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan, dtype=np.float64)
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    np.divide(n, d, out=out, where=np.isfinite(n) & np.isfinite(d) & (np.abs(d) > 1e-12))
    return out


def _bucket_codes(values: np.ndarray, spec: FactorSpec) -> np.ndarray:
    out = np.full(len(values), -1, dtype=np.int16)
    valid = np.isfinite(values)
    if valid.any():
        out[valid] = np.digitize(values[valid], spec.edges[1:-1], right=False).astype(np.int16)
    return out


def _profit_factor(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else np.nan
    return gp / gl


def _summary(gross: np.ndarray, mfe: np.ndarray, mae: np.ndarray, fee_cost: float, normal_cost: float) -> dict[str, Any]:
    valid = np.isfinite(gross)
    g = np.asarray(gross, dtype=float)[valid]
    mf = np.asarray(mfe, dtype=float)[valid]
    ma = np.asarray(mae, dtype=float)[valid]
    if not len(g):
        return {
            "events": 0, "mean_gross": np.nan, "median_gross": np.nan,
            "mean_fee_only_net": np.nan, "mean_net": np.nan, "median_net": np.nan,
            "win_rate": np.nan, "profit_factor": np.nan, "mean_mfe": np.nan, "mean_mae": np.nan,
        }
    net = g - float(normal_cost)
    return {
        "events": int(len(g)),
        "mean_gross": float(np.mean(g)),
        "median_gross": float(np.median(g)),
        "mean_fee_only_net": float(np.mean(g - float(fee_cost))),
        "mean_net": float(np.mean(net)),
        "median_net": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": _profit_factor(net),
        "mean_mfe": float(np.nanmean(mf)) if np.isfinite(mf).any() else np.nan,
        "mean_mae": float(np.nanmean(ma)) if np.isfinite(ma).any() else np.nan,
    }


def _build_event_universe(
    bars: pd.DataFrame,
    args: argparse.Namespace,
    windows: tuple[int, ...],
    thresholds: tuple[float, ...],
    max_forward_minutes: int,
) -> EventUniverse:
    masks = r02._eligible_masks(bars, args, (int(max_forward_minutes),))
    log_return, abs_change, hist_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )
    notional_arr = pd.to_numeric(bars["notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    trades_arr = pd.to_numeric(bars["trades_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    notional_prefix = _prefix(notional_arr)
    trades_prefix = _prefix(trades_arr)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    n = len(bars)
    min_threshold = min(thresholds)
    count_rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    threshold_flag_parts: dict[float, list[np.ndarray]] = {float(t): [] for t in thresholds}
    event_id_cursor = 1

    for window in windows:
        feat = r01.build_window_features(bars, int(window), log_return, abs_change, hist_vol)
        norm = feat.normalized_impulse
        for direction, side in (("LONG", 1), ("SHORT", -1)):
            directed = float(side) * norm
            all_min = np.flatnonzero(np.isfinite(directed) & (directed >= float(min_threshold)))
            eligible = all_min[masks["eligible"][all_min]]
            rows, _, dedup_masks = r07._event_count_rows(
                direction=direction,
                window=int(window),
                thresholds=thresholds,
                all_min_positions=all_min,
                eligible_positions=eligible,
                directed_norm=directed,
                masks=masks,
                study_months=int(masks["study_months"]),
                n=n,
            )
            count_rows.extend(rows)
            if not len(eligible):
                continue
            union = np.zeros(len(eligible), dtype=bool)
            for t in thresholds:
                union |= dedup_masks[float(t)]
            local = np.flatnonzero(union)
            pos = eligible[local]
            if not len(pos):
                continue
            window_start = pos - int(window) + 1
            window_end = pos + 1
            impulse_notional = _range_sum(notional_prefix, window_start, window_end)
            impulse_trades = _range_sum(trades_prefix, window_start, window_end)
            signal_time = bars.index[pos] + pd.Timedelta(minutes=1)
            frame = pd.DataFrame({
                "event_id": np.arange(event_id_cursor, event_id_cursor + len(pos), dtype=np.int64),
                "direction": direction,
                "side": int(side),
                "impulse_window": int(window),
                "signal_bar_pos": pos,
                "signal_bar_start": bars.index[pos],
                "signal_time": signal_time,
                "signal_price": close_arr[pos],
                "normalized_impulse": directed[pos],
                "impulse_notional_per_second": impulse_notional / float(int(window) * 60),
                "impulse_trades_per_second": impulse_trades / float(int(window) * 60),
                "year": pd.DatetimeIndex(signal_time).year.astype(int),
                "month": pd.DatetimeIndex(signal_time).to_period("M").astype(str),
            })
            frames.append(frame)
            for t in thresholds:
                threshold_flag_parts[float(t)].append(dedup_masks[float(t)][local].astype(bool))
            event_id_cursor += len(frame)

    if not frames:
        raise RuntimeError("No eligible impulse events were built")
    all_events = pd.concat(frames, ignore_index=True)
    flags = {t: np.concatenate(parts) if parts else np.zeros(len(all_events), dtype=bool)
             for t, parts in threshold_flag_parts.items()}
    return EventUniverse(
        frame=all_events,
        threshold_flags=flags,
        count_rows=count_rows,
        study_months=int(masks["study_months"]),
    )


def _micro_db_audit(loader: OKXTradeBarLoader, args: argparse.Namespace) -> dict[str, Any]:
    db_path = loader.db_path
    result: dict[str, Any] = {
        "timeframe": loader.timeframe,
        "table_name": loader.table_name,
        "db_path": str(db_path),
        "status": "missing",
        "rows": 0,
    }
    if not db_path.exists():
        result["status"] = "db_missing"
        return result
    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (loader.table_name,)
        ).fetchone()
        if not table_exists:
            result["status"] = "table_missing"
            return result
        row = conn.execute(
            f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {loader.table_name}"
        ).fetchone()
        result.update({"rows": int(row[0] or 0), "data_start": row[1], "data_end": row[2]})
        try:
            coverage = conn.execute(
                "SELECT COUNT(DISTINCT utc_day), SUM(rows) FROM trade_bar_coverage WHERE table_name=?",
                (loader.table_name,),
            ).fetchone()
            result["coverage_days"] = int(coverage[0] or 0)
            result["coverage_rows_meta"] = int(coverage[1] or 0)
        except sqlite3.Error:
            result["coverage_days"] = 0
            result["coverage_rows_meta"] = 0
    result["status"] = "loaded" if result["rows"] > 0 else "empty"
    return result


def _regularize_micro(raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tf_seconds: int) -> pd.DataFrame:
    if raw.empty:
        return raw
    raw = raw.sort_index().copy()
    freq = pd.Timedelta(seconds=int(tf_seconds))
    grid = pd.date_range(start=start.floor(freq), end=end.floor(freq), freq=freq)
    out = raw.reindex(grid)
    observed = out["close"].notna().to_numpy(dtype=bool)
    close = pd.to_numeric(out["close"], errors="coerce").ffill().bfill()
    for col in ("open", "high", "low", "close"):
        series = pd.to_numeric(out[col], errors="coerce")
        out[col] = series.where(series.notna(), close)
    zero_cols = [
        "volume", "trades_count", "buy_volume", "sell_volume", "notional",
        "buy_notional", "sell_notional", "buy_trades_count", "sell_trades_count",
        "delta_volume", "delta_notional", "large_buy_notional", "large_sell_notional",
        "large_buy_trades_count", "large_sell_trades_count", "large_delta_notional",
        "large_trades_count", "max_trade_notional", "max_trade_size",
    ]
    for col in zero_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["source_micro_bar_observed_flag"] = observed
    out.index.name = "timestamp"
    return out


def _micro_minute_mismatch(raw: pd.DataFrame, bars_1m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if raw.empty:
        idx = pd.date_range(start.floor("1min"), end.ceil("1min"), freq="1min")
        return pd.Series(True, index=idx)
    micro_counts = pd.to_numeric(raw["trades_count"], errors="coerce").fillna(0.0).resample("1min").sum()
    one = pd.to_numeric(bars_1m["trades_count"], errors="coerce").reindex(micro_counts.index)
    mismatch = one.notna() & (np.abs(micro_counts - one) > 0.5)
    return mismatch.astype(bool)


def _allocate_micro_arrays(n_events: int, n_checkpoints: int, horizons: tuple[int, ...], barriers: tuple[int, ...]) -> MicroArrays:
    shape = (n_events, n_checkpoints)
    return MicroArrays(
        valid=np.zeros(shape, dtype=bool),
        entry_time_ns=np.full(shape, np.iinfo(np.int64).min, dtype=np.int64),
        factors={spec.name: np.full(shape, np.nan, dtype=np.float32) for spec in FACTOR_SPECS},
        gross={h: np.full(shape, np.nan, dtype=np.float32) for h in horizons},
        mfe={h: np.full(shape, np.nan, dtype=np.float32) for h in horizons},
        mae={h: np.full(shape, np.nan, dtype=np.float32) for h in horizons},
        favorable_first_seconds={b: np.zeros(shape, dtype=np.int16) for b in barriers},
        adverse_first_seconds={b: np.zeros(shape, dtype=np.int16) for b in barriers},
        cache_mismatch_flag=np.zeros(shape, dtype=bool),
    )


def _fill_first_passage(
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: np.ndarray,
    entry_price: np.ndarray,
    side: np.ndarray,
    tf_seconds: int,
    max_limit_seconds: int,
    barriers: tuple[int, ...],
    out: MicroArrays,
    event_rows: np.ndarray,
    checkpoint_col: int,
    batch_size: int,
) -> None:
    max_bars = int(max_limit_seconds // tf_seconds)
    offsets = np.arange(max_bars, dtype=np.int64)
    for batch_start in range(0, len(event_rows), int(batch_size)):
        batch_rows = event_rows[batch_start: batch_start + int(batch_size)]
        eidx = entry_idx[batch_start: batch_start + int(batch_size)]
        ep = entry_price[batch_start: batch_start + int(batch_size)]
        sd = side[batch_start: batch_start + int(batch_size)]
        gather = eidx[:, None] + offsets[None, :]
        ph = high[gather]
        pl = low[gather]
        if len(batch_rows) == 0:
            continue
        for bps in barriers:
            rate = float(bps) / 10_000.0
            fav = np.where(sd[:, None] == 1, ph >= ep[:, None] * (1.0 + rate), pl <= ep[:, None] * (1.0 - rate))
            adv = np.where(sd[:, None] == 1, pl <= ep[:, None] * (1.0 - rate), ph >= ep[:, None] * (1.0 + rate))
            fav_any = fav.any(axis=1)
            adv_any = adv.any(axis=1)
            fav_first = np.where(fav_any, fav.argmax(axis=1) + 1, 0).astype(np.int16)
            adv_first = np.where(adv_any, adv.argmax(axis=1) + 1, 0).astype(np.int16)
            out.favorable_first_seconds[int(bps)][batch_rows, checkpoint_col] = fav_first * int(tf_seconds)
            out.adverse_first_seconds[int(bps)][batch_rows, checkpoint_col] = adv_first * int(tf_seconds)


def _process_micro_frame_for_events(
    *,
    micro: pd.DataFrame,
    events: pd.DataFrame,
    event_rows: np.ndarray,
    checkpoints: tuple[int, ...],
    horizons: tuple[int, ...],
    barriers: tuple[int, ...],
    tf_seconds: int,
    mismatch_minutes: pd.Series,
    arrays: MicroArrays,
    path_batch_events: int,
    max_first_passage_limit_seconds: int,
) -> None:
    if not len(event_rows) or micro.empty:
        return
    idx_ns = micro.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    signal_ns = pd.to_datetime(events.loc[event_rows, "signal_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    start_idx = np.searchsorted(idx_ns, signal_ns, side="left")
    exact_start = (start_idx < len(idx_ns)) & (idx_ns[np.minimum(start_idx, len(idx_ns) - 1)] == signal_ns)
    if not exact_start.any():
        return

    open_arr = pd.to_numeric(micro["open"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(micro["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(micro["low"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(micro["close"], errors="coerce").to_numpy(dtype=float)
    notional = pd.to_numeric(micro["notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    trades = pd.to_numeric(micro["trades_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta = pd.to_numeric(micro["delta_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    large_buy = pd.to_numeric(micro["large_buy_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    large_sell = pd.to_numeric(micro["large_sell_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    large_delta = pd.to_numeric(micro["large_delta_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    abs_path = np.abs(np.diff(close_arr, prepend=close_arr[0]))

    p_notional = _prefix(notional)
    p_trades = _prefix(trades)
    p_delta = _prefix(delta)
    p_large_buy = _prefix(large_buy)
    p_large_sell = _prefix(large_sell)
    p_large_delta = _prefix(large_delta)
    p_abs_path = _prefix(abs_path)
    p_delta_pos = _prefix((delta > 0).astype(float))
    p_delta_neg = _prefix((delta < 0).astype(float))

    side_all = events.loc[event_rows, "side"].to_numpy(dtype=int)
    signal_price_all = events.loc[event_rows, "signal_price"].to_numpy(dtype=float)
    impulse_notional_speed = events.loc[event_rows, "impulse_notional_per_second"].to_numpy(dtype=float)
    impulse_trade_speed = events.loc[event_rows, "impulse_trades_per_second"].to_numpy(dtype=float)

    mismatch_index_ns = mismatch_minutes.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    mismatch_values = mismatch_minutes.to_numpy(dtype=bool)

    for ccol, checkpoint in enumerate(checkpoints):
        n_feat_bars = int(checkpoint // tf_seconds)
        n_max_future = int(max(max(horizons), int(max_first_passage_limit_seconds)) // tf_seconds)
        end_feat = start_idx + n_feat_bars
        entry_idx = end_feat
        enough = exact_start & (start_idx > 0) & (entry_idx + n_max_future <= len(micro))
        rows_local = np.flatnonzero(enough)
        if not len(rows_local):
            continue
        global_rows = event_rows[rows_local]
        s = start_idx[rows_local]
        e = end_feat[rows_local]
        ei = entry_idx[rows_local]
        side = side_all[rows_local]
        signal_price = signal_price_all[rows_local]

        # Cache-integrity flag: any mismatched 1m aggregate from signal through
        # the largest requested future horizon invalidates this event/timeframe.
        start_minute = signal_ns[rows_local].astype("datetime64[m]").astype(np.int64)
        end_minute = (signal_ns[rows_local] + int(checkpoint + max(horizons)) * 1_000_000_000).astype("datetime64[m]").astype(np.int64)
        mm_minutes = mismatch_index_ns.astype("datetime64[m]").astype(np.int64)
        mm_prefix = _prefix(mismatch_values.astype(float))
        ml = np.searchsorted(mm_minutes, start_minute, side="left")
        mr = np.searchsorted(mm_minutes, end_minute, side="right")
        cache_bad = _range_sum(mm_prefix, ml, mr) > 0
        arrays.cache_mismatch_flag[global_rows, ccol] = cache_bad

        valid = ~cache_bad
        if not valid.any():
            continue
        global_rows = global_rows[valid]
        s = s[valid]
        e = e[valid]
        ei = ei[valid]
        side = side[valid]
        signal_price = signal_price[valid]
        local_valid_rows = rows_local[valid]

        total_notional = _range_sum(p_notional, s, e)
        total_trades = _range_sum(p_trades, s, e)
        total_delta = _range_sum(p_delta, s, e)
        total_large = _range_sum(p_large_buy, s, e) + _range_sum(p_large_sell, s, e)
        total_large_delta = _range_sum(p_large_delta, s, e)
        total_abs_path = _range_sum(p_abs_path, s, e)
        aligned_count = np.where(
            side == 1,
            _range_sum(p_delta_pos, s, e),
            _range_sum(p_delta_neg, s, e),
        )
        progress_bps = side * (close_arr[e - 1] / signal_price - 1.0) * 10_000.0
        directional_delta_pressure = side * _safe_div(total_delta, total_notional)
        large_pressure = side * _safe_div(total_large_delta, total_large)
        notional_speed_ratio = _safe_div(total_notional / float(checkpoint), impulse_notional_speed[local_valid_rows])
        trade_speed_ratio = _safe_div(total_trades / float(checkpoint), impulse_trade_speed[local_valid_rows])
        signed_eff = _safe_div(side * (close_arr[e - 1] - signal_price), total_abs_path)
        impact = _safe_div(progress_bps, np.maximum(np.abs(directional_delta_pressure), 0.05))
        aligned_ratio = _safe_div(aligned_count, np.full(len(s), n_feat_bars, dtype=float))
        half = n_feat_bars // 2
        mid = s + half
        first_notional = _range_sum(p_notional, s, mid)
        second_notional = _range_sum(p_notional, mid, e)
        first_pressure = side * _safe_div(_range_sum(p_delta, s, mid), first_notional)
        second_pressure = side * _safe_div(_range_sum(p_delta, mid, e), second_notional)
        acceleration = second_pressure - first_pressure

        values = {
            "directional_price_progress_bps": progress_bps,
            "directional_delta_pressure": directional_delta_pressure,
            "notional_speed_ratio": notional_speed_ratio,
            "trade_speed_ratio": trade_speed_ratio,
            "signed_path_efficiency": signed_eff,
            "price_per_delta_impact": impact,
            "directional_large_delta_pressure": large_pressure,
            "aligned_delta_bar_ratio": aligned_ratio,
            "delta_pressure_acceleration": acceleration,
        }
        for name, value in values.items():
            arrays.factors[name][global_rows, ccol] = np.asarray(value, dtype=np.float32)
        arrays.valid[global_rows, ccol] = True
        arrays.entry_time_ns[global_rows, ccol] = idx_ns[ei]

        entry_price = open_arr[ei]
        for horizon in horizons:
            hb = int(horizon // tf_seconds)
            end_idx = ei + hb
            gross = side * (close_arr[end_idx - 1] / entry_price - 1.0)
            # Bounded vectorized gather; horizon <=900s and chunk events are small.
            offsets = np.arange(hb, dtype=np.int64)
            gather = ei[:, None] + offsets[None, :]
            ph = high_arr[gather]
            pl = low_arr[gather]
            favorable = np.where(side[:, None] == 1, ph / entry_price[:, None] - 1.0, 1.0 - pl / entry_price[:, None])
            adverse = np.where(side[:, None] == 1, pl / entry_price[:, None] - 1.0, 1.0 - ph / entry_price[:, None])
            arrays.gross[int(horizon)][global_rows, ccol] = gross.astype(np.float32)
            arrays.mfe[int(horizon)][global_rows, ccol] = np.nanmax(favorable, axis=1).astype(np.float32)
            arrays.mae[int(horizon)][global_rows, ccol] = np.nanmin(adverse, axis=1).astype(np.float32)

        _fill_first_passage(
            high=high_arr,
            low=low_arr,
            entry_idx=ei,
            entry_price=entry_price,
            side=side,
            tf_seconds=tf_seconds,
            max_limit_seconds=int(max_first_passage_limit_seconds),
            barriers=barriers,
            out=arrays,
            event_rows=global_rows,
            checkpoint_col=ccol,
            batch_size=int(path_batch_events),
        )


def _load_and_process_timeframe(
    *,
    bars: pd.DataFrame,
    events: pd.DataFrame,
    args: argparse.Namespace,
    timeframe: str,
    checkpoints: tuple[int, ...],
    horizons: tuple[int, ...],
    barriers: tuple[int, ...],
    micro_override: pd.DataFrame | None = None,
    max_first_passage_limit_seconds: int = PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS,
) -> tuple[MicroArrays | None, dict[str, Any]]:
    tf_seconds = _tf_seconds(timeframe)
    if any(c % tf_seconds != 0 for c in checkpoints):
        return None, {"timeframe": timeframe, "status": "checkpoint_not_divisible"}
    if any(h % tf_seconds != 0 for h in horizons):
        return None, {"timeframe": timeframe, "status": "horizon_not_divisible"}
    if any(l % tf_seconds != 0 for l in DEFAULT_FIRST_PASSAGE_LIMITS):
        return None, {"timeframe": timeframe, "status": "first_passage_limit_not_divisible"}

    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=timeframe,
        data_dir=args.data_dir,
        db_name=args.trade_bar_db_name,
        align_with_okx_loader_timezone=True,
    )
    audit = _micro_db_audit(loader, args) if micro_override is None else {
        "timeframe": timeframe, "table_name": "synthetic_override", "status": "synthetic_override",
        "rows": int(len(micro_override)), "data_start": str(micro_override.index.min()), "data_end": str(micro_override.index.max()),
    }
    if micro_override is None and audit.get("status") != "loaded":
        print(f"       {timeframe}: {audit.get('status')}; skipped", flush=True)
        return None, audit

    arrays = _allocate_micro_arrays(len(events), len(checkpoints), horizons, barriers)
    start = pd.Timestamp(args.start_date)
    end_exclusive = r01._date_bounds(args.start_date, args.end_date)[1]
    chunk_days = max(1, int(args.micro_chunk_days))
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start.normalize()
    while cur < end_exclusive:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_exclusive)
        chunks.append((cur, nxt))
        cur = nxt
    max_extra = max(checkpoints) + max(horizons)
    margin_before = pd.Timedelta(seconds=tf_seconds)
    margin_after = pd.Timedelta(seconds=max_extra + tf_seconds)

    print(f"[micro] {timeframe} chunks={len(chunks)} table={loader.table_name}", flush=True)
    with ProgressReporter(
        label=f"[micro {timeframe}] calendar chunks",
        total=len(chunks), every=max(1, int(args.progress_every)), enabled=not args.no_progress,
    ) as progress:
        for i, (core_start, core_end) in enumerate(chunks, start=1):
            event_mask = (pd.to_datetime(events["signal_time"]) >= core_start) & (pd.to_datetime(events["signal_time"]) < core_end)
            event_rows = np.flatnonzero(event_mask.to_numpy(dtype=bool))
            if not len(event_rows):
                progress.update(i)
                continue
            query_start = core_start - margin_before
            query_end = core_end + margin_after
            if micro_override is None:
                raw = loader.load_local_data(start_date=query_start, end_date=query_end)
            else:
                raw = micro_override.loc[(micro_override.index >= query_start) & (micro_override.index <= query_end)].copy()
            if raw.empty:
                arrays.cache_mismatch_flag[event_rows, :] = True
                progress.update(i)
                continue
            mismatch = _micro_minute_mismatch(raw, bars, query_start, query_end)
            micro = _regularize_micro(raw, query_start, query_end, tf_seconds)
            _process_micro_frame_for_events(
                micro=micro,
                events=events,
                event_rows=event_rows,
                checkpoints=checkpoints,
                horizons=horizons,
                barriers=barriers,
                tf_seconds=tf_seconds,
                mismatch_minutes=mismatch,
                arrays=arrays,
                path_batch_events=int(args.path_batch_events),
                max_first_passage_limit_seconds=int(max_first_passage_limit_seconds),
            )
            progress.update(i)
    audit["valid_event_checkpoint_cells"] = int(arrays.valid.sum())
    audit["cache_mismatch_cells"] = int(arrays.cache_mismatch_flag.sum())
    audit["timeframe_seconds"] = tf_seconds
    return arrays, audit


def _adequate_extremes(codes: np.ndarray, selected: np.ndarray, n_buckets: int, min_events: int) -> tuple[int, int] | None:
    adequate: list[int] = []
    for code in range(n_buckets):
        if int(np.sum(selected & (codes == code))) >= int(min_events):
            adequate.append(code)
    if len(adequate) < 2:
        return None
    return adequate[0], adequate[-1]


def _first_passage_stats(
    fav_seconds: np.ndarray,
    adv_seconds: np.ndarray,
    selected: np.ndarray,
    time_limit: int,
) -> dict[str, Any]:
    idx = np.flatnonzero(selected)
    if not len(idx):
        return {
            "events": 0, "target_first_rate": np.nan, "stop_first_rate": np.nan,
            "timeout_rate": np.nan, "same_time_rate": np.nan, "directional_gap": np.nan,
        }
    fav = fav_seconds[idx]
    adv = adv_seconds[idx]
    fav_in = (fav > 0) & (fav <= int(time_limit))
    adv_in = (adv > 0) & (adv <= int(time_limit))
    target_first = fav_in & (~adv_in | (fav < adv))
    stop_first = adv_in & (~fav_in | (adv <= fav))  # conservative same-time stop-first
    timeout = ~(target_first | stop_first)
    same = fav_in & adv_in & (fav == adv)
    return {
        "events": int(len(idx)),
        "target_first_rate": float(np.mean(target_first)),
        "stop_first_rate": float(np.mean(stop_first)),
        "timeout_rate": float(np.mean(timeout)),
        "same_time_rate": float(np.mean(same)),
        "directional_gap": float(np.mean(target_first) - np.mean(stop_first)),
    }


def _first_passage_stats_idx(
    fav_seconds: np.ndarray,
    adv_seconds: np.ndarray,
    idx: np.ndarray,
    time_limit: int,
) -> dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    if not len(idx):
        return {
            "events": 0, "target_first_rate": np.nan, "stop_first_rate": np.nan,
            "timeout_rate": np.nan, "same_time_rate": np.nan, "directional_gap": np.nan,
        }
    fav = fav_seconds[idx]
    adv = adv_seconds[idx]
    fav_in = (fav > 0) & (fav <= int(time_limit))
    adv_in = (adv > 0) & (adv <= int(time_limit))
    target_first = fav_in & (~adv_in | (fav < adv))
    stop_first = adv_in & (~fav_in | (adv <= fav))
    timeout = ~(target_first | stop_first)
    same = fav_in & adv_in & (fav == adv)
    return {
        "events": int(len(idx)),
        "target_first_rate": float(np.mean(target_first)),
        "stop_first_rate": float(np.mean(stop_first)),
        "timeout_rate": float(np.mean(timeout)),
        "same_time_rate": float(np.mean(same)),
        "directional_gap": float(np.mean(target_first) - np.mean(stop_first)),
    }


def _summarize_timeframe(
    *,
    universe: EventUniverse,
    arrays: MicroArrays,
    timeframe: str,
    checkpoints: tuple[int, ...],
    horizons: tuple[int, ...],
    barriers: tuple[int, ...],
    fp_limits: tuple[int, ...],
    thresholds: tuple[float, ...],
    windows: tuple[int, ...],
    fee_cost: float,
    normal_cost: float,
    min_bucket_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame]]:
    events = universe.frame
    fixed_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    event_state_rows: list[pd.DataFrame] = []
    direction_values = events["direction"].astype(str).to_numpy()
    window_values = events["impulse_window"].to_numpy(dtype=int)
    years = events["year"].to_numpy(dtype=int)

    for ccol, checkpoint in enumerate(checkpoints):
        valid_cp = arrays.valid[:, ccol]
        factor_values = {spec.name: arrays.factors[spec.name][:, ccol].astype(float) for spec in FACTOR_SPECS}
        factor_codes = {spec.name: _bucket_codes(factor_values[spec.name], spec) for spec in FACTOR_SPECS}

        for direction in ("LONG", "SHORT"):
            dmask = direction_values == direction
            for window in windows:
                wmask = window_values == int(window)
                for threshold in thresholds:
                    base_idx = np.flatnonzero(valid_cp & dmask & wmask & universe.threshold_flags[float(threshold)])
                    if not len(base_idx):
                        continue
                    for spec in FACTOR_SPECS:
                        values = factor_values[spec.name]
                        codes = factor_codes[spec.name]
                        local_codes = codes[base_idx]
                        adequate_codes = [
                            code for code in range(len(spec.labels))
                            if int(np.sum(local_codes == code)) >= int(min_bucket_events)
                        ]
                        for code, label in enumerate(spec.labels):
                            idx = base_idx[local_codes == code]
                            for horizon in horizons:
                                stats = _summary(
                                    arrays.gross[int(horizon)][idx, ccol],
                                    arrays.mfe[int(horizon)][idx, ccol],
                                    arrays.mae[int(horizon)][idx, ccol],
                                    fee_cost, normal_cost,
                                )
                                fixed_rows.append({
                                    "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                                    "micro_timeframe": timeframe, "checkpoint_seconds": int(checkpoint),
                                    "factor_name": spec.name, "factor_bucket": label, "bucket_order": int(code),
                                    "horizon_seconds": int(horizon), "events_per_month": float(stats["events"] / max(1, universe.study_months)),
                                    "mean_factor_value": float(np.nanmean(values[idx])) if len(idx) else np.nan,
                                    **stats,
                                })
                            for barrier in barriers:
                                for limit in fp_limits:
                                    fp = _first_passage_stats_idx(
                                        arrays.favorable_first_seconds[int(barrier)][:, ccol],
                                        arrays.adverse_first_seconds[int(barrier)][:, ccol],
                                        idx, int(limit),
                                    )
                                    fp_rows.append({
                                        "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                                        "micro_timeframe": timeframe, "checkpoint_seconds": int(checkpoint),
                                        "factor_name": spec.name, "factor_bucket": label, "bucket_order": int(code),
                                        "barrier_bps": int(barrier), "time_limit_seconds": int(limit), **fp,
                                    })

                        if len(adequate_codes) >= 2:
                            low_code, high_code = adequate_codes[0], adequate_codes[-1]
                            low_idx = base_idx[local_codes == low_code]
                            high_idx = base_idx[local_codes == high_code]
                            low_fp = _first_passage_stats_idx(
                                arrays.favorable_first_seconds[PRIMARY_BARRIER_BPS][:, ccol],
                                arrays.adverse_first_seconds[PRIMARY_BARRIER_BPS][:, ccol],
                                low_idx, PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS,
                            )
                            high_fp = _first_passage_stats_idx(
                                arrays.favorable_first_seconds[PRIMARY_BARRIER_BPS][:, ccol],
                                arrays.adverse_first_seconds[PRIMARY_BARRIER_BPS][:, ccol],
                                high_idx, PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS,
                            )
                            low_fixed = _summary(
                                arrays.gross[PRIMARY_HORIZON_SECONDS][low_idx, ccol],
                                arrays.mfe[PRIMARY_HORIZON_SECONDS][low_idx, ccol],
                                arrays.mae[PRIMARY_HORIZON_SECONDS][low_idx, ccol], fee_cost, normal_cost,
                            )
                            high_fixed = _summary(
                                arrays.gross[PRIMARY_HORIZON_SECONDS][high_idx, ccol],
                                arrays.mfe[PRIMARY_HORIZON_SECONDS][high_idx, ccol],
                                arrays.mae[PRIMARY_HORIZON_SECONDS][high_idx, ccol], fee_cost, normal_cost,
                            )
                            comparison_rows.append({
                                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                                "micro_timeframe": timeframe, "checkpoint_seconds": int(checkpoint),
                                "factor_name": spec.name,
                                "low_bucket": spec.labels[low_code], "high_bucket": spec.labels[high_code],
                                "low_events": int(low_fp["events"]), "high_events": int(high_fp["events"]),
                                "low_directional_gap": low_fp["directional_gap"], "high_directional_gap": high_fp["directional_gap"],
                                "high_minus_low_directional_gap": float(high_fp["directional_gap"] - low_fp["directional_gap"]),
                                "low_mean_gross_300s": low_fixed["mean_gross"], "high_mean_gross_300s": high_fixed["mean_gross"],
                                "high_minus_low_mean_gross_300s": float(high_fixed["mean_gross"] - low_fixed["mean_gross"]),
                                "low_mean_net_300s": low_fixed["mean_net"], "high_mean_net_300s": high_fixed["mean_net"],
                                "high_minus_low_mean_mfe_300s": float(high_fixed["mean_mfe"] - low_fixed["mean_mfe"]),
                                "high_minus_low_mean_mae_300s": float(high_fixed["mean_mae"] - low_fixed["mean_mae"]),
                            })
                            for year in sorted(np.unique(years[base_idx])):
                                for extreme_name, idx0, bucket_name in (
                                    ("low", low_idx, spec.labels[low_code]),
                                    ("high", high_idx, spec.labels[high_code]),
                                ):
                                    yidx = idx0[years[idx0] == int(year)]
                                    fp = _first_passage_stats_idx(
                                        arrays.favorable_first_seconds[PRIMARY_BARRIER_BPS][:, ccol],
                                        arrays.adverse_first_seconds[PRIMARY_BARRIER_BPS][:, ccol],
                                        yidx, PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS,
                                    )
                                    fixed = _summary(
                                        arrays.gross[PRIMARY_HORIZON_SECONDS][yidx, ccol],
                                        arrays.mfe[PRIMARY_HORIZON_SECONDS][yidx, ccol],
                                        arrays.mae[PRIMARY_HORIZON_SECONDS][yidx, ccol], fee_cost, normal_cost,
                                    )
                                    yearly_rows.append({
                                        "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                                        "micro_timeframe": timeframe, "checkpoint_seconds": int(checkpoint),
                                        "factor_name": spec.name, "extreme": extreme_name, "factor_bucket": bucket_name,
                                        "year": int(year), **fp,
                                        "mean_gross_300s": fixed["mean_gross"], "mean_net_300s": fixed["mean_net"],
                                        "median_net_300s": fixed["median_net"],
                                    })

        if timeframe == "5s":
            valid_idx = np.flatnonzero(valid_cp)
            if len(valid_idx):
                part = events.iloc[valid_idx][
                    ["event_id", "direction", "impulse_window", "signal_time"]
                ].reset_index(drop=True).copy()
                part["micro_timeframe"] = timeframe
                part["checkpoint_seconds"] = int(checkpoint)
                part["entry_time"] = pd.to_datetime(arrays.entry_time_ns[valid_idx, ccol])
                part["cache_mismatch_flag"] = arrays.cache_mismatch_flag[valid_idx, ccol]
                for t in thresholds:
                    part[f"event_threshold_{str(t).replace('.', 'p')}_flag"] = universe.threshold_flags[float(t)][valid_idx]
                for spec in FACTOR_SPECS:
                    part[spec.name] = arrays.factors[spec.name][valid_idx, ccol]
                for h in horizons:
                    part[f"gross_{h}s"] = arrays.gross[int(h)][valid_idx, ccol]
                for b in barriers:
                    part[f"favorable_first_{b}bps_seconds"] = arrays.favorable_first_seconds[int(b)][valid_idx, ccol]
                    part[f"adverse_first_{b}bps_seconds"] = arrays.adverse_first_seconds[int(b)][valid_idx, ccol]
                event_state_rows.append(part)

    return fixed_rows, fp_rows, yearly_rows, comparison_rows, event_state_rows

def _build_decision_matrix(comparison: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["direction", "factor_name"]
    for (direction, factor), part in comparison.groupby(group_cols, sort=False):
        lifts = pd.to_numeric(part["high_minus_low_directional_gap"], errors="coerce")
        gross_lifts = pd.to_numeric(part["high_minus_low_mean_gross_300s"], errors="coerce")
        valid = lifts.notna() & gross_lifts.notna()
        partv = part.loc[valid].copy()
        if partv.empty:
            rows.append({"direction": direction, "factor_name": factor, "status": "insufficient_evidence"})
            continue
        positive_gap_share = float(np.mean(partv["high_minus_low_directional_gap"] > 0))
        positive_gross_share = float(np.mean(partv["high_minus_low_mean_gross_300s"] > 0))
        median_gap_lift = float(np.median(partv["high_minus_low_directional_gap"]))
        median_gross_lift = float(np.median(partv["high_minus_low_mean_gross_300s"]))
        tf_coverage = int(partv["micro_timeframe"].nunique())
        checkpoint_coverage = int(partv["checkpoint_seconds"].nunique())
        window_coverage = int(partv["impulse_window"].nunique())
        threshold_coverage = int(partv["threshold"].nunique())

        year_sub = yearly[(yearly["direction"] == direction) & (yearly["factor_name"] == factor)].copy()
        year_lifts: list[float] = []
        if not year_sub.empty:
            keys = ["impulse_window", "threshold", "micro_timeframe", "checkpoint_seconds", "year"]
            pivot = year_sub.pivot_table(index=keys, columns="extreme", values="directional_gap", aggfunc="first")
            if {"low", "high"}.issubset(pivot.columns):
                year_lifts = (pivot["high"] - pivot["low"]).dropna().tolist()
        positive_year_share = float(np.mean(np.asarray(year_lifts) > 0)) if year_lifts else np.nan

        high_net_positive_share = float(np.mean(pd.to_numeric(partv["high_mean_net_300s"], errors="coerce") > 0))
        volatility_only = (
            median_gap_lift <= 0.0
            and float(np.nanmedian(pd.to_numeric(partv["high_minus_low_mean_mfe_300s"], errors="coerce"))) > 0
        )
        if (
            median_gap_lift >= 0.05
            and positive_gap_share >= 0.70
            and median_gross_lift > 0
            and positive_gross_share >= 0.60
            and tf_coverage >= 3
            and checkpoint_coverage >= 2
            and window_coverage >= 2
            and threshold_coverage >= 2
            and (not year_lifts or positive_year_share >= 0.60)
        ):
            status = "retain_for_causal_validation"
        elif (
            median_gap_lift >= 0.02
            and positive_gap_share >= 0.60
            and median_gross_lift > 0
            and tf_coverage >= 2
            and checkpoint_coverage >= 2
        ):
            status = "weak_keep_for_more_anatomy"
        elif volatility_only:
            status = "volatility_only_not_directional"
        elif median_gap_lift < 0 and positive_gap_share < 0.40:
            status = "reject_expected_direction"
        else:
            status = "insufficient_evidence"
        rows.append({
            "direction": direction,
            "factor_name": factor,
            "status": status,
            "comparisons": int(len(partv)),
            "median_directional_gap_lift": median_gap_lift,
            "positive_gap_lift_share": positive_gap_share,
            "median_mean_gross_300s_lift": median_gross_lift,
            "positive_gross_lift_share": positive_gross_share,
            "high_bucket_net_positive_share": high_net_positive_share,
            "positive_year_lift_share": positive_year_share,
            "timeframe_coverage": tf_coverage,
            "checkpoint_coverage": checkpoint_coverage,
            "window_coverage": window_coverage,
            "threshold_coverage": threshold_coverage,
        })
    return pd.DataFrame(rows).sort_values(["status", "direction", "factor_name"]).reset_index(drop=True)


def _timeframe_consistency(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    return (
        comparison.groupby(["direction", "factor_name", "micro_timeframe", "checkpoint_seconds"], as_index=False)
        .agg(
            comparisons=("high_minus_low_directional_gap", "count"),
            median_directional_gap_lift=("high_minus_low_directional_gap", "median"),
            positive_gap_lift_share=("high_minus_low_directional_gap", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce") > 0))),
            median_mean_gross_300s_lift=("high_minus_low_mean_gross_300s", "median"),
            positive_gross_lift_share=("high_minus_low_mean_gross_300s", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce") > 0))),
        )
    )


def _signal_audit(event_states: pd.DataFrame) -> pd.DataFrame:
    if event_states.empty:
        return pd.DataFrame(columns=["event_id", "lookahead_flag"])
    out = event_states[["event_id", "micro_timeframe", "checkpoint_seconds", "signal_time", "entry_time", "cache_mismatch_flag"]].copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"])
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["expected_entry_time"] = out["signal_time"] + pd.to_timedelta(out["checkpoint_seconds"], unit="s")
    out["entry_not_checkpoint_next_open_flag"] = out["entry_time"] != out["expected_entry_time"]
    out["factor_available_after_entry_flag"] = False
    out["future_path_used_in_factor_flag"] = False
    out["lookahead_flag"] = (
        out["entry_not_checkpoint_next_open_flag"].astype(bool)
        | out["factor_available_after_entry_flag"].astype(bool)
        | out["future_path_used_in_factor_flag"].astype(bool)
        | out["cache_mismatch_flag"].astype(bool)
    )
    return out


def _build_brief(decision: pd.DataFrame, consistency: pd.DataFrame, audit_df: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# Round 12 Research Brief",
        "",
        "## Question",
        "",
        "Which single post-impulse micro variable earliest leaves a directional residual edge after it becomes observable?",
        "",
        "## Guardrails",
        "",
        "- No cross-factor combinations.",
        "- Fixed natural buckets only; no threshold optimization.",
        "- Common 15s/30s/60s/120s checkpoints across 1s/3s/5s/15s bars.",
        "- Entry at checkpoint next micro open; factors use only fully closed micro bars.",
        "- Micro/1m trade-count mismatches invalidate the affected event/timeframe.",
        "",
        "## Decision matrix",
        "",
    ]
    if decision.empty:
        lines.append("No factor had enough valid evidence.")
    else:
        lines.extend([
            "| direction | factor | status | median gap lift | positive gap share | median 300s gross lift | timeframe coverage |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for _, row in decision.iterrows():
            lines.append(
                f"| {row['direction']} | {row['factor_name']} | {row['status']} | "
                f"{float(row.get('median_directional_gap_lift', np.nan)):.2%} | "
                f"{float(row.get('positive_gap_lift_share', np.nan)):.2%} | "
                f"{float(row.get('median_mean_gross_300s_lift', np.nan)):.4%} | "
                f"{int(row.get('timeframe_coverage', 0))} |"
            )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- `retain_for_causal_validation`: broad single-factor evidence only; still not a strategy.",
        "- `weak_keep_for_more_anatomy`: some cross-scale evidence but insufficient executable margin.",
        "- `volatility_only_not_directional`: MFE rises without directional first-passage improvement.",
        "- `reject_expected_direction`: the factor's higher bucket consistently worsens directional order.",
        "",
        "## Audit",
        "",
        f"- Event-state audit rows: {len(audit_df):,}.",
        f"- Lookahead/cache-integrity flags: {int(audit_df['lookahead_flag'].sum()) if len(audit_df) else 0:,}.",
        f"- Normal round-trip execution cost: {float(meta['normal_execution_cost']):.4%}.",
        "",
        "No factor should be combined in Round 13 unless it independently survives multiple timeframes, checkpoints, adjacent impulse windows/thresholds and years.",
    ])
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation Research Log\n"
    marker = "## Round 12 — Post-impulse micro reacquisition single-factor atlas"
    if marker in text:
        return
    addition = f"""

{marker}

- 研究问题：1s/3s/5s/15s trade bar 上，哪个单变量最早识别冲击后的原方向重新接管，并在状态可见后留下可执行空间？
- 改变：不再使用 Range activity 作为入场过滤；并行研究多个微观变量，但每个变量独立分层，禁止条件组合。
- 固定检查点：15s、30s、60s、120s；固定自然分箱，不做阈值搜索。
- 单变量：价格进展、delta pressure、成交额速度、成交笔数速度、路径效率、price-per-delta impact、大单 delta、顺向 delta bar 比例、delta 加速度。
- 因果：左标签 micro bar 仅在 timestamp+timeframe 后可见；checkpoint 后下一根 micro bar open 执行；未来路径不进入特征。
- 数据完整性：秒级 trade count 与本地 1m trade count 对账；不一致事件按 timeframe 排除；不下载、不补建。
- 性能：按日历块读取秒级缓存，一次构建 prefix arrays，所有单变量/检查点/阈值复用；无逐组合全历史扫描。
- 当前状态：等待生产运行和报告复核。
- 生成时间：{meta.get('created_at')}
"""
    log_path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def run_research(
    bars: pd.DataFrame,
    args: argparse.Namespace,
    *,
    micro_overrides: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    windows = _parse_ints(args.impulse_windows, name="impulse-windows")
    thresholds = _parse_floats(args.thresholds, name="thresholds")
    micro_timeframes = tuple(dict.fromkeys(x.strip() for x in str(args.micro_timeframes).split(",") if x.strip()))
    checkpoints = _parse_ints(args.checkpoints_seconds, name="checkpoints-seconds")
    horizons = _parse_ints(args.horizons_seconds, name="horizons-seconds")
    barriers = _parse_ints(args.barriers_bps, name="barriers-bps")
    fp_limits = _parse_ints(args.first_passage_limits_seconds, name="first-passage-limits-seconds")
    if tuple(windows) != DEFAULT_WINDOWS:
        print(f"[warning] non-default windows: {windows}", flush=True)
    if PRIMARY_HORIZON_SECONDS not in horizons:
        raise ValueError(f"horizons-seconds must include primary {PRIMARY_HORIZON_SECONDS}")
    if PRIMARY_BARRIER_BPS not in barriers:
        raise ValueError(f"barriers-bps must include primary {PRIMARY_BARRIER_BPS}")
    if PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS not in fp_limits:
        raise ValueError(f"first-passage-limits-seconds must include primary {PRIMARY_FIRST_PASSAGE_LIMIT_SECONDS}")

    fee_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_cost + args.entry_slippage + args.exit_slippage)
    max_forward_minutes = int(math.ceil((max(checkpoints) + max(horizons)) / 60.0)) + 1

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = r01.validate_bars(bars, args)
    universe = _build_event_universe(bars, args, windows, thresholds, max_forward_minutes)
    print(f"[events] union deduplicated event rows={len(universe.frame):,}", flush=True)

    fixed_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    event_states: list[pd.DataFrame] = []
    cache_audits: list[dict[str, Any]] = []

    for timeframe in micro_timeframes:
        override = micro_overrides.get(timeframe) if micro_overrides else None
        arrays, cache_audit = _load_and_process_timeframe(
            bars=bars,
            events=universe.frame,
            args=args,
            timeframe=timeframe,
            checkpoints=checkpoints,
            horizons=horizons,
            barriers=barriers,
            micro_override=override,
            max_first_passage_limit_seconds=max(fp_limits),
        )
        cache_audits.append(cache_audit)
        if arrays is None:
            continue
        f, p, y, c, e = _summarize_timeframe(
            universe=universe,
            arrays=arrays,
            timeframe=timeframe,
            checkpoints=checkpoints,
            horizons=horizons,
            barriers=barriers,
            fp_limits=fp_limits,
            thresholds=thresholds,
            windows=windows,
            fee_cost=fee_cost,
            normal_cost=normal_cost,
            min_bucket_events=int(args.min_bucket_events),
        )
        fixed_rows.extend(f)
        fp_rows.extend(p)
        yearly_rows.extend(y)
        comparison_rows.extend(c)
        event_states.extend(e)

    event_counts = pd.DataFrame(universe.count_rows)
    fixed = pd.DataFrame(fixed_rows)
    first_passage = pd.DataFrame(fp_rows)
    yearly = pd.DataFrame(yearly_rows)
    comparison = pd.DataFrame(comparison_rows)
    decision = _build_decision_matrix(comparison, yearly)
    consistency = _timeframe_consistency(comparison)
    cache_audit_df = pd.DataFrame(cache_audits)
    events_df = pd.concat(event_states, ignore_index=True) if event_states else pd.DataFrame()
    audit_df = _signal_audit(events_df)

    created_at = pd.Timestamp.now().isoformat()
    meta = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "created_at": created_at,
        "symbol": args.symbol,
        "base_timeframe": args.timeframe,
        "micro_timeframes_requested": list(micro_timeframes),
        "micro_timeframes_loaded": cache_audit_df.loc[cache_audit_df["status"].isin(["loaded", "synthetic_override"]), "timeframe"].tolist() if not cache_audit_df.empty else [],
        "warmup_start_date": args.warmup_start_date,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "impulse_windows": list(windows),
        "thresholds": list(thresholds),
        "checkpoints_seconds": list(checkpoints),
        "horizons_seconds": list(horizons),
        "barriers_bps": list(barriers),
        "first_passage_limits_seconds": list(fp_limits),
        "fee_only_cost": fee_cost,
        "normal_execution_cost": normal_cost,
        "event_union_rows": int(len(universe.frame)),
        "event_state_rows_written": int(len(events_df)),
        "factor_names": [x.name for x in FACTOR_SPECS],
        "binning_policy": "fixed_natural_bins_predeclared_in_source_no_quantile_fit",
        "combination_policy": "single_factor_only_no_cross_factor_AND",
        "micro_loader_policy": "local_cache_only_load_local_data_no_build_no_download",
        "micro_integrity_policy": "minute_trade_count_must_match_local_1m_trade_bar",
        "validation": validation,
    }

    artifacts: list[tuple[pd.DataFrame, Path]] = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (cache_audit_df, out_dir / "02_micro_cache_audit.csv"),
        (fixed, out_dir / "03_single_factor_horizon_summary.csv"),
        (first_passage, out_dir / "04_single_factor_first_passage.csv"),
        (comparison, out_dir / "05_extreme_bucket_comparison.csv"),
        (consistency, out_dir / "06_timeframe_checkpoint_consistency.csv"),
        (yearly, out_dir / "07_yearly_stability.csv"),
        (decision, out_dir / "08_mechanism_decision_matrix.csv"),
    ]
    with ProgressReporter(label="[artifacts] tables", total=len(artifacts) + 4, every=1, enabled=not args.no_progress) as progress:
        done = 0
        for frame, path in artifacts:
            frame.to_csv(path, index=False)
            done += 1
            progress.update(done)
        if args.skip_events_csv:
            pd.DataFrame(columns=["event_id", "micro_timeframe", "checkpoint_seconds"]).to_csv(out_dir / "10_events_5s.csv", index=False)
        else:
            events_df.to_csv(out_dir / "10_events_5s.csv", index=False)
        done += 1
        progress.update(done)
        audit_df.to_csv(out_dir / "11_signal_audit.csv", index=False)
        done += 1
        progress.update(done)
        (out_dir / "12_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        done += 1
        progress.update(done)
        brief = _build_brief(decision, consistency, audit_df, meta)
        (out_dir / "13_research_brief.md").write_text(brief, encoding="utf-8")
        done += 1
        progress.update(done)

    _update_log(Path(__file__).resolve().with_name("00_research_log.md"), meta)
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    return {
        "report_dir": out_dir,
        "events": out_dir / "10_events_5s.csv",
        "audit": out_dir / "11_signal_audit.csv",
        "review_pack": out_dir / "gpt_review_pack.zip",
    }


def _synthetic_micro_from_1m(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf_seconds = _tf_seconds(timeframe)
    start = bars.index.min()
    end = bars.index.max() + pd.Timedelta(minutes=1) - pd.Timedelta(seconds=tf_seconds)
    idx = pd.date_range(start=start, end=end, freq=pd.Timedelta(seconds=tf_seconds))
    minute_pos = np.minimum(((idx - start).total_seconds() // 60).astype(int), len(bars) - 1)
    base_open = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)[minute_pos]
    base_close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)[minute_pos]
    within = ((idx - idx.floor("1min")).total_seconds() + tf_seconds) / 60.0
    close = base_open + (base_close - base_open) * within
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * 1.00005
    low = np.minimum(open_, close) * 0.99995
    trades_1m = pd.to_numeric(bars["trades_count"], errors="coerce").fillna(60).to_numpy(dtype=float)[minute_pos]
    bars_per_min = 60 // tf_seconds
    trades = np.maximum(1.0, np.floor(trades_1m / bars_per_min))
    # exact reconciliation by distributing remainder to first bars of minute
    minute_group = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.floor("1min"))
    trades_out = np.zeros(len(idx), dtype=float)
    one_counts = pd.to_numeric(bars["trades_count"], errors="coerce").fillna(60).to_numpy(dtype=int)
    for minute_i, (_, locs) in enumerate(minute_group):
        loc = locs.to_numpy(dtype=int)
        total = int(one_counts[min(minute_i, len(one_counts)-1)])
        q, r = divmod(total, len(loc))
        trades_out[loc] = q
        if r:
            trades_out[loc[:r]] += 1
    notional = np.maximum(trades_out, 1.0) * close * 0.01
    sign = np.where(np.sin(np.arange(len(idx)) / 7.0) >= 0, 1.0, -1.0)
    delta = notional * 0.20 * sign
    buy = (notional + delta) / 2.0
    sell = (notional - delta) / 2.0
    frame = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": notional / close, "trades_count": trades_out,
        "buy_volume": buy / close, "sell_volume": sell / close,
        "notional": notional, "buy_notional": buy, "sell_notional": sell,
        "buy_trades_count": np.floor(trades_out / 2), "sell_trades_count": trades_out - np.floor(trades_out / 2),
        "delta_volume": delta / close, "delta_notional": delta,
        "cvd_volume": np.cumsum(delta / close), "cvd_notional": np.cumsum(delta),
        "taker_buy_ratio": _safe_div(buy, notional), "avg_trade_size": _safe_div(notional / close, trades_out),
        "vwap": close,
        "large_buy_notional": np.zeros(len(idx)), "large_sell_notional": np.zeros(len(idx)),
        "large_buy_trades_count": np.zeros(len(idx)), "large_sell_trades_count": np.zeros(len(idx)),
        "large_delta_notional": np.zeros(len(idx)), "large_trades_count": np.zeros(len(idx)),
        "max_trade_notional": notional, "max_trade_size": _safe_div(notional, close),
    }, index=idx)
    frame.index.name = "timestamp"
    return frame


def run_self_test(args: argparse.Namespace) -> int:
    raw = r01._synthetic_bars()
    reg = r01._regularize_trade_bar_axis(raw)
    # Round-01 synthetic OHLCV predates the richer trade-bar schema. Add a
    # deterministic flow/count contract so the Round-12 cache reconciliation
    # and speed baselines are exercised end to end.
    base_volume = pd.to_numeric(reg.get("volume", 1.0), errors="coerce").fillna(1.0)
    reg["trades_count"] = np.maximum(20, np.round(base_volume * 10)).astype(int)
    reg["notional"] = base_volume.to_numpy(dtype=float) * pd.to_numeric(reg["close"], errors="coerce").to_numpy(dtype=float)
    reg["buy_notional"] = reg["notional"] * 0.52
    reg["sell_notional"] = reg["notional"] * 0.48
    reg["delta_notional"] = reg["buy_notional"] - reg["sell_notional"]
    reg["large_buy_notional"] = 0.0
    reg["large_sell_notional"] = 0.0
    reg["large_delta_notional"] = 0.0
    original = vars(args).copy()
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r12_selftest_") as tmp:
            args.out_dir = str(Path(tmp) / "report")
            args.start_date = str(reg.index[500].date())
            args.end_date = str((reg.index[-1] - pd.Timedelta(days=1)).date())
            args.warmup_start_date = str(reg.index[0].date())
            args.impulse_windows = "5,10"
            args.thresholds = "0.3,0.6"
            args.micro_timeframes = "1s,3s,5s,15s"
            args.checkpoints_seconds = "15,30,60"
            args.horizons_seconds = "30,60,180,300"
            args.barriers_bps = "15,25"
            args.first_passage_limits_seconds = "30,60,180,300"
            args.vol_lookback_bars = 300
            args.vol_min_periods = 150
            args.micro_chunk_days = 2
            args.path_batch_events = 100
            args.min_bucket_events = 2
            args.no_progress = True
            args.skip_review_pack = True
            args.skip_events_csv = False
            overrides = {tf: _synthetic_micro_from_1m(reg, tf) for tf in ("1s", "3s", "5s", "15s")}
            result = run_research(reg, args, micro_overrides=overrides)
            report = Path(result["report_dir"])
            required = [
                "01_event_counts.csv", "02_micro_cache_audit.csv", "03_single_factor_horizon_summary.csv",
                "04_single_factor_first_passage.csv", "05_extreme_bucket_comparison.csv",
                "06_timeframe_checkpoint_consistency.csv", "07_yearly_stability.csv",
                "08_mechanism_decision_matrix.csv", "10_events_5s.csv", "11_signal_audit.csv",
                "12_run_meta.json", "13_research_brief.md",
            ]
            missing = [x for x in required if not (report / x).exists()]
            if missing:
                raise AssertionError(f"missing self-test outputs: {missing}")
            cache = pd.read_csv(report / "02_micro_cache_audit.csv")
            if set(cache["timeframe"]) != {"1s", "3s", "5s", "15s"}:
                raise AssertionError("micro timeframe coverage missing")
            fp = pd.read_csv(report / "04_single_factor_first_passage.csv")
            if fp.empty:
                raise AssertionError("first-passage output empty")
            events = pd.read_csv(report / "10_events_5s.csv")
            if events.empty:
                raise AssertionError("5s event state output empty")
            audit = pd.read_csv(report / "11_signal_audit.csv")
            if len(audit) and bool(audit["entry_not_checkpoint_next_open_flag"].astype(bool).any()):
                raise AssertionError("checkpoint entry timing failed")
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
