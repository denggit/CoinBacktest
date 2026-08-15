#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICT MSS Edge Discovery R02.

Research-only event study.  The study asks a narrow question first:
can a mechanically causal market-structure shift on ETH 5m predict future
returns after the project's next-open execution convention and costs?

Definitions are deliberately fixed before the run:
- order-2 symmetric swing highs/lows;
- a pivot becomes usable only after its right confirmation bars close;
- LONG MSS: two confirmed swing highs form a lower-high sequence, then a
  later closed bar closes above the latest lower high;
- SHORT MSS: two confirmed swing lows form a higher-low sequence, then a
  later closed bar closes below the latest higher low;
- signal is the close of the break bar; entry is next bar open.

No FVG, order block, liquidity sweep, session, displacement, HTF bias or
parameter mining is included.  Those are intentionally reserved for later
research versions if the base event has evidence of edge.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# research/<family>/<script>.py -> repository root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.event_study import CostConfig, EventStudyConfig, run_event_study  # noqa: E402
from src.research_common.event_study.reports import write_event_study_report  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "ict_mss_edge_discovery_r02"
SCRIPT_VERSION = "2.0.0"
EXPERIMENT_ID = "ETH_5M_ICT_MSS_EDGE_DISCOVERY_R02"
EDGE_ID = "RESEARCH_ONLY_ETH_ICT_MSS"
TITLE = "ETH 5m ICT MSS Edge Discovery R02"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/ict_mss_edge_discovery_r02"
FEE_ONE_WAY = 0.00055  # project default: 0.11% round trip


@dataclass(frozen=True)
class Pivot:
    pos: int
    time: pd.Timestamp
    price: float
    available_time: pd.Timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-build-missing", action="store_true")
    parser.add_argument("--pivot-order", type=int, default=2)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12, 24, 48])
    parser.add_argument("--mfe-mae-horizon", type=int, default=48)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _end_exclusive(value: str) -> pd.Timestamp:
    text = str(value).strip()
    ts = pd.Timestamp(text)
    return ts + pd.Timedelta(days=1) if len(text) <= 10 else ts + pd.Timedelta(microseconds=1)


def _pivot_mask(values: np.ndarray, order: int, *, high: bool) -> np.ndarray:
    n = len(values)
    o = int(order)
    if o < 1 or n < 2 * o + 1:
        return np.zeros(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    mask[:o] = False
    mask[-o:] = False
    for lag in range(1, o + 1):
        left = np.full(n, np.nan)
        right = np.full(n, np.nan)
        left[lag:] = values[:-lag]
        right[:-lag] = values[lag:]
        if high:
            mask &= values > left
            mask &= values >= right
        else:
            mask &= values < left
            mask &= values <= right
    return mask & np.isfinite(values)


def _build_causal_pivots(bars: pd.DataFrame, order: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build symmetric causal highs/lows; availability is pivot + (order+1) bars."""
    idx = pd.DatetimeIndex(bars.index)
    delta = idx[1] - idx[0]
    if delta <= pd.Timedelta(0):
        raise ValueError("bars must have increasing timestamps")
    if not np.all(np.diff(idx.asi8) == delta.value):
        raise ValueError("MSS R02 requires a gap-free primary execution axis")
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(float)
    hi_mask = _pivot_mask(high, order, high=True)
    lo_mask = _pivot_mask(low, order, high=False)
    positions_hi = np.flatnonzero(hi_mask)
    positions_lo = np.flatnonzero(lo_mask)
    def frame(positions: np.ndarray, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "pivot_pos": positions.astype(np.int64),
            "pivot_time": idx[positions],
            "pivot_price": values[positions],
            "available_time": idx[positions] + (order + 1) * delta,
        })
    return frame(positions_hi, high), frame(positions_lo, low)


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if args.timeframe != "5m":
        raise ValueError("R02 is intentionally frozen to 5m for the first MSS edge test")
    print(f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe, data_dir=args.data_dir, db_name=args.db_name)
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=not bool(args.no_build_missing),
    )
    required = ["open", "high", "low", "close"]
    bars = bars.loc[:, [c for c in required if c in bars.columns]].copy()
    if len(bars) < 100:
        raise RuntimeError("insufficient bars")
    bars.index = pd.to_datetime(bars.index, errors="coerce")
    bars = bars.loc[~bars.index.isna()].sort_index()
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    for c in required:
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    bars = bars.dropna(subset=required)
    print(f"[load] rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _build_mss_events(
    bars: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    *,
    show_progress: bool,
) -> pd.DataFrame:
    """Online MSS state machine; no future labels are used for event creation."""
    idx = pd.DatetimeIndex(bars.index)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)
    high_events = {int(idx.get_loc(pd.Timestamp(r.available_time))): r for r in highs.itertuples(index=False)}
    low_events = {int(idx.get_loc(pd.Timestamp(r.available_time))): r for r in lows.itertuples(index=False)}
    latest_highs: list[Pivot] = []
    latest_lows: list[Pivot] = []
    broken_long_pivot_pos: set[int] = set()
    broken_short_pivot_pos: set[int] = set()
    rows: list[dict[str, object]] = []
    progress = ProgressReporter(
        label="[MSS] scan",
        total=len(bars),
        every=max(1, len(bars) // 100),
        enabled=show_progress,
    )
    for pos, ts in enumerate(idx):
        # Pivots are inserted only at their causal availability bar.
        if pos in high_events:
            r = high_events[pos]
            latest_highs.append(Pivot(int(r.pivot_pos), pd.Timestamp(r.pivot_time), float(r.pivot_price), pd.Timestamp(r.available_time)))
            latest_highs = latest_highs[-2:]
        if pos in low_events:
            r = low_events[pos]
            latest_lows.append(Pivot(int(r.pivot_pos), pd.Timestamp(r.pivot_time), float(r.pivot_price), pd.Timestamp(r.available_time)))
            latest_lows = latest_lows[-2:]

        # Break is evaluated only on a closed bar and uses already-available pivots.
        c = close[pos]
        if not np.isfinite(c):
            progress.update(pos + 1)
            continue

        # Bullish MSS: lower-high sequence -> close breaks latest lower high.
        if len(latest_highs) == 2:
            h1, h2 = latest_highs
            if h2.price < h1.price and h2.pos not in broken_long_pivot_pos and c > h2.price:
                if research_start <= ts < research_end_exclusive:
                    rows.append({
                        "event_id": f"MSSL_{pos}",
                        "event_name": "MSS_LONG",
                        "side": 1,
                        "signal_time": ts,
                        "break_bar_pos": pos,
                        "broken_pivot_pos": h2.pos,
                        "broken_pivot_time": h2.time,
                        "broken_pivot_price": h2.price,
                        "prior_pivot_time": h1.time,
                        "prior_pivot_price": h1.price,
                        "broken_pivot_available_time": h2.available_time,
                        "structure_type": "lower_high_break",
                    })
                broken_long_pivot_pos.add(h2.pos)

        # Bearish MSS: higher-low sequence -> close breaks latest higher low.
        if len(latest_lows) == 2:
            l1, l2 = latest_lows
            if l2.price > l1.price and l2.pos not in broken_short_pivot_pos and c < l2.price:
                if research_start <= ts < research_end_exclusive:
                    rows.append({
                        "event_id": f"MSSS_{pos}",
                        "event_name": "MSS_SHORT",
                        "side": -1,
                        "signal_time": ts,
                        "break_bar_pos": pos,
                        "broken_pivot_pos": l2.pos,
                        "broken_pivot_time": l2.time,
                        "broken_pivot_price": l2.price,
                        "prior_pivot_time": l1.time,
                        "prior_pivot_price": l1.price,
                        "broken_pivot_available_time": l2.available_time,
                        "structure_type": "higher_low_break",
                    })
                broken_short_pivot_pos.add(l2.pos)
        progress.update(pos + 1)
    progress.close()
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.sort_values(["signal_time", "event_name", "event_id"], kind="mergesort").drop_duplicates("signal_time", keep="first")
    events = events.reset_index(drop=True)
    # Explicit causal assertion before event-study labels.
    bad = pd.to_datetime(events["broken_pivot_available_time"]) > pd.to_datetime(events["signal_time"])
    if bool(bad.any()):
        raise RuntimeError("MSS causal failure: a broken pivot was used before availability")
    return events


def _run_variant(bars: pd.DataFrame, events: pd.DataFrame, *, delay: int, cost_multiplier: float, args: argparse.Namespace):
    cost = CostConfig(
        entry_fee_rate=FEE_ONE_WAY * cost_multiplier,
        exit_fee_rate=FEE_ONE_WAY * cost_multiplier,
        entry_slippage_pct=0.0,
        exit_slippage_pct=0.0,
    )
    cfg = EventStudyConfig(
        horizons=tuple(int(x) for x in args.horizons if int(x) >= int(delay)),
        mfe_mae_horizon=max(int(args.mfe_mae_horizon), int(delay)),
        entry_delay_bars=int(delay),
        cost=cost,
        signal_time_col="signal_time",
        side_col="side",
        event_name_col="event_name",
        event_id_col="event_id",
        context_available_time_cols=("broken_pivot_available_time",),
        min_count=int(args.min_count),
        progress_every=0,
    )
    return run_event_study(bars, events, cfg)


def _write_extra_report(out_dir: Path, events: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame, base_result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "06_mss_events.csv", index=False, encoding="utf-8-sig")
    highs.to_csv(out_dir / "07_confirmed_swing_highs.csv", index=False, encoding="utf-8-sig")
    lows.to_csv(out_dir / "08_confirmed_swing_lows.csv", index=False, encoding="utf-8-sig")
    audit = events[["event_id", "signal_time", "broken_pivot_time", "broken_pivot_available_time", "prior_pivot_time"]].copy()
    audit["pivot_available_time_flag"] = pd.to_datetime(audit["broken_pivot_available_time"]) > pd.to_datetime(audit["signal_time"])
    audit["pivot_available_time_status"] = np.where(audit["pivot_available_time_flag"], "FAIL", "PASS")
    audit.to_csv(out_dir / "09_mss_causal_audit.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return Path(args.out_dir)
    bars = _load_bars(args)
    highs, lows = _build_causal_pivots(bars, int(args.pivot_order))
    print(f"[structure] confirmed_highs={len(highs):,} confirmed_lows={len(lows):,} order={args.pivot_order}", flush=True)
    events = _build_mss_events(
        bars,
        highs,
        lows,
        pd.Timestamp(args.start_date),
        _end_exclusive(args.end_date),
        show_progress=not args.no_progress,
    )
    if events.empty:
        raise RuntimeError("No MSS events found; inspect structure definition/data coverage.")
    print(f"[events] MSS events={len(events):,} long={(events.side == 1).sum():,} short={(events.side == -1).sum():,}", flush=True)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _run_variant(bars, events, delay=1, cost_multiplier=1.0, args=args)
    write_event_study_report(base, out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, write_review_pack=False)
    _write_extra_report(out_dir, events, highs, lows, base)

    stress_rows: list[dict[str, object]] = []
    variants = [(1, 1.0), (1, 1.5), (1, 2.0), (2, 1.0), (3, 1.0)]
    for delay, multiplier in variants:
        result = _run_variant(bars, events, delay=delay, cost_multiplier=multiplier, args=args)
        row = {
            "entry_delay_bars": delay,
            "cost_multiplier": multiplier,
            "round_trip_fee": 2 * FEE_ONE_WAY * multiplier,
            "events": len(result.events),
        }
        overview = result.overview
        if not overview.empty:
            for _, r in overview.iterrows():
                metric = str(r.get("metric", ""))
                if metric.endswith("_net"):
                    row[metric + "_mean"] = r.get("mean")
                    row[metric + "_win_rate"] = r.get("win_rate")
                    row[metric + "_profit_factor"] = r.get("profit_factor")
        stress_rows.append(row)
    pd.DataFrame(stress_rows).to_csv(out_dir / "10_cost_delay_stress.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_start_date": args.warmup_start_date,
        "research_start_date": args.start_date,
        "research_end_date": args.end_date,
        "pivot_order": int(args.pivot_order),
        "mss_definition": "lower-high sequence then closed-bar break for LONG; higher-low sequence then closed-bar break for SHORT",
        "entry_assumption": "next_open",
        "round_trip_fee": 2 * FEE_ONE_WAY,
        "future_features_excluded": ["liquidity_sweep", "FVG", "order_block", "session", "HTF_bias", "displacement_filter"],
        "event_count": int(len(events)),
        "notes": "R02 intentionally tests the base mechanical MSS event before adding ICT refinements.",
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    readme = f"""# {TITLE}

## Research question
Can a mechanically causal 5m market-structure shift (MSS) predict ETH returns after next-open execution and the project's 0.11% round-trip fee?

## Fixed event definition
- Swing order: {int(args.pivot_order)}.
- Pivot availability: pivot bar + (order + 1) execution bars; the pivot is not usable before that time.
- LONG MSS: the two latest available swing highs form a lower-high sequence and a later closed bar closes above the latest lower high.
- SHORT MSS: the two latest available swing lows form a higher-low sequence and a later closed bar closes below the latest higher low.
- Entry: next bar open.
- Excluded in R02: liquidity sweep, FVG, order block, session, HTF bias and displacement filters.

## Interpretation rule
This is an event study, not a strategy. A positive result is only a reason to continue research; it is not a promotion to live trading.

## Files
The standard Event Study files are written by `src.research_common.event_study`. Additional MSS structure, causal and cost/delay stress files are written by this script.
"""
    (out_dir / "RESEARCH_README.md").write_text(readme, encoding="utf-8")
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    return out_dir


def run_self_test() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="5min")
    close = np.array([100, 101, 99, 100, 98, 100, 97, 99, 96, 98, 95, 97, 94, 96, 93, 95, 92, 94, 91, 93], dtype=float)
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    # First verify the pivot builder's causal availability contract.
    highs, lows = _build_causal_pivots(bars, 2)
    if not highs.empty:
        if (pd.to_datetime(highs["available_time"]) <= pd.to_datetime(highs["pivot_time"])).any():
            raise AssertionError("pivot availability is not after pivot bar")
    if not lows.empty:
        if (pd.to_datetime(lows["available_time"]) <= pd.to_datetime(lows["pivot_time"])).any():
            raise AssertionError("pivot availability is not after pivot bar")

    # Then exercise the MSS state machine with explicitly causal confirmed
    # pivots.  This isolates the MSS transition test from the synthetic OHLC
    # geometry used by the pivot detector.
    confirmed_highs = pd.DataFrame([
        {"pivot_pos": 2, "pivot_time": idx[2], "pivot_price": 105.0, "available_time": idx[5]},
        {"pivot_pos": 6, "pivot_time": idx[6], "pivot_price": 103.0, "available_time": idx[9]},
    ])
    confirmed_lows = pd.DataFrame([
        {"pivot_pos": 3, "pivot_time": idx[3], "pivot_price": 95.0, "available_time": idx[6]},
        {"pivot_pos": 7, "pivot_time": idx[7], "pivot_price": 97.0, "available_time": idx[10]},
    ])
    test_close = close.copy()
    test_close[10] = 104.0
    test_close[12] = 94.0
    test_bars = bars.copy()
    test_bars["close"] = test_close
    test_bars["open"] = np.r_[100.0, test_close[:-1]]
    test_bars["high"] = np.maximum(test_bars["open"], test_bars["close"]) + 0.2
    test_bars["low"] = np.minimum(test_bars["open"], test_bars["close"]) - 0.2
    events = _build_mss_events(test_bars, confirmed_highs, confirmed_lows, idx[0], idx[-1] + pd.Timedelta(minutes=5), show_progress=False)
    if events.empty or set(events["event_name"]) != {"MSS_LONG", "MSS_SHORT"}:
        raise AssertionError(f"self-test expected both MSS directions, got {events.to_dict('records')}")
    if (pd.to_datetime(events["broken_pivot_available_time"]) > pd.to_datetime(events["signal_time"])).any():
        raise AssertionError("self-test found lookahead")
    result = _run_variant(test_bars, events, delay=1, cost_multiplier=1.0, args=argparse.Namespace(horizons=[1, 3], mfe_mae_horizon=3, min_count=1))
    if result.events.empty or result.causal_audit.empty:
        raise AssertionError("self-test event study failed")
    print(f"[self-test] passed events={len(events)}")


if __name__ == "__main__":
    run(parse_args())
