#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 causal unconsumed multi-timeframe Swing Low liquidity atlas.

This is deliberately an atlas, not a finished strategy backtest.  It starts
from every causally confirmed order-1 Swing Low on 15m/30m/1H/4H/1D, keeps each
level active until its first real sweep, and records the complete broad funnel:
activation -> approach -> touch -> first sweep -> reclaim/acceptance.

No OBI, footprint, wall, volume, trend, prominence, age, confluence or return
threshold is used to admit a level.  Those are attributes for later filtering.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.swing_liquidity_atlas import (  # noqa: E402
    AtlasConfig,
    attach_active_confluence,
    attach_forward_paths,
    build_event_table,
    build_level_lifecycle,
    build_swing_low_universe,
    normalize_primary_bars,
    normalize_timeframes,
)
from src.research_common.swing_liquidity_atlas.reports import (  # noqa: E402
    attribute_bin_summary,
    causal_audit,
    event_stage_summary,
    fixed_period,
    forward_path_summary,
    level_summary,
    lifecycle_summary,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "02_unconsumed_swing_liquidity_atlas"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_UNCONSUMED_SWING_LIQUIDITY_ATLAS_R02"
EDGE_ID = "RESEARCH_ONLY_SWING_LOW_LIQUIDITY_SWEEP_REVERSAL"
TITLE = "ETH Causal Unconsumed Swing Low Liquidity Atlas R02"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/unconsumed_swing_liquidity_atlas_r02"


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _comma_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in str(value).split(",") if item.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _comma_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a broad causal atlas of unconsumed 15m/30m/1H/4H/1D Swing Low liquidity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m", help="Primary execution axis; R02 requires 1m.")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--swing-timeframes", default="15m,30m,1H,4H,1D")
    parser.add_argument("--confirmation-orders", type=_comma_ints, default=(1, 2, 3, 5))
    parser.add_argument("--approach-distance-bp", type=float, default=200.0)
    parser.add_argument("--touch-distance-bp", type=float, default=5.0)
    parser.add_argument("--sweep-epsilon-bp", type=float, default=0.01)
    parser.add_argument("--acceptance-depth-bp", type=float, default=50.0)
    parser.add_argument("--acceptance-consecutive-closes", type=int, default=3)
    parser.add_argument("--resolution-horizon-bars", type=int, default=180)
    parser.add_argument("--forward-horizons", type=_comma_ints, default=(5, 15, 30, 60, 180))
    parser.add_argument("--confluence-tolerances-bp", type=_comma_floats, default=(5.0, 10.0, 25.0, 50.0))
    parser.add_argument("--data-source", choices=["trade_bar", "ohlcv_local"], default="trade_bar")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-build-missing", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--write-full-events", action="store_true")
    parser.add_argument("--event-sample-size", type=int, default=20_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _end_exclusive(value: str) -> pd.Timestamp:
    text = str(value).strip()
    timestamp = pd.Timestamp(text)
    if len(text) <= 10:
        timestamp += pd.Timedelta(days=1)
    else:
        timestamp += pd.Timedelta(microseconds=1)
    return timestamp


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.timeframe) != "1m":
        raise ValueError("R02 requires --timeframe 1m so HTF availability and event execution share one causal axis")
    print(
        f"[load] source={args.data_source} symbol={args.symbol} "
        f"window={args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    if args.data_source == "trade_bar":
        loader = OKXTradeBarLoader(
            symbol=args.symbol,
            timeframe="1m",
            data_dir=args.data_dir,
            db_name=args.db_name,
        )
        bars = loader.fetch_data_by_date_range(
            args.warmup_start_date,
            args.end_date,
            chunksize=int(args.chunksize),
            force_rebuild=bool(args.force_rebuild),
            build_missing=not bool(args.no_build_missing),
        )
    else:
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        bars = loader.load_local_data()
        if not bars.empty:
            start = pd.Timestamp(args.warmup_start_date)
            end = pd.Timestamp(args.end_date)
            bars = bars.loc[(bars.index >= start) & (bars.index <= end)]
    keep_columns = [
        "open", "high", "low", "close", "volume", "notional", "trades_count",
        "buy_notional", "sell_notional", "delta_notional",
        "large_buy_notional", "large_sell_notional", "large_delta_notional",
    ]
    available_columns = [name for name in keep_columns if name in bars.columns]
    bars = normalize_primary_bars(bars.loc[:, available_columns].copy())
    print(f"[load] rows={len(bars):,} cols={len(bars.columns)} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _config(args: argparse.Namespace) -> AtlasConfig:
    timeframes = normalize_timeframes(_comma_strings(args.swing_timeframes))
    return AtlasConfig(
        timeframes=timeframes,
        confirmation_orders=tuple(args.confirmation_orders),
        approach_distance_bp=float(args.approach_distance_bp),
        touch_distance_bp=float(args.touch_distance_bp),
        sweep_epsilon_bp=float(args.sweep_epsilon_bp),
        acceptance_depth_bp=float(args.acceptance_depth_bp),
        acceptance_consecutive_closes=int(args.acceptance_consecutive_closes),
        resolution_horizon_bars=int(args.resolution_horizon_bars),
        forward_horizons=tuple(int(v) for v in args.forward_horizons),
        confluence_tolerances_bp=tuple(float(v) for v in args.confluence_tolerances_bp),
    ).validate()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _data_quality(bars: pd.DataFrame, levels: pd.DataFrame, lifecycle: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    gaps = bars.index.to_series().diff().dropna()
    missing_minutes = int(np.maximum(gaps.dt.total_seconds().to_numpy(dtype=float) / 60.0 - 1.0, 0.0).sum()) if len(gaps) else 0
    max_gap_seconds = float(gaps.dt.total_seconds().max()) if len(gaps) else 0.0
    return pd.DataFrame(
        [
            {"metric": "primary_rows", "value": len(bars)},
            {"metric": "primary_start", "value": bars.index.min()},
            {"metric": "primary_end", "value": bars.index.max()},
            {"metric": "primary_duplicate_timestamps", "value": int(bars.index.duplicated().sum())},
            {"metric": "primary_estimated_missing_minutes", "value": missing_minutes},
            {"metric": "primary_max_gap_seconds", "value": max_gap_seconds},
            {"metric": "swing_levels", "value": len(levels)},
            {"metric": "lifecycle_rows", "value": len(lifecycle)},
            {"metric": "active_unconsumed_at_end", "value": int(lifecycle["stop_liquidity_state"].eq("active_unconsumed").sum()) if not lifecycle.empty else 0},
            {"metric": "first_sweeps", "value": int(lifecycle["sweep_pos"].ge(0).sum()) if not lifecycle.empty else 0},
            {"metric": "event_rows", "value": len(events)},
            {"metric": "unique_event_ids", "value": int(events["event_id"].nunique()) if not events.empty else 0},
        ]
    )


def _brief(config: AtlasConfig, levels: pd.DataFrame, lifecycle: pd.DataFrame, events: pd.DataFrame, audit: pd.DataFrame) -> str:
    stage_counts = events["event_stage"].value_counts().to_dict() if not events.empty else {}
    timeframe_counts = levels["source_timeframe"].value_counts().to_dict() if not levels.empty else {}
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) if not audit.empty else 0
    return f"""# Unconsumed Swing Low Liquidity Atlas R02

## Purpose

Build the broad causal structural universe before adding order-flow or Books filters.
This report is **not a strategy backtest** and does not select thresholds from returns.

## Universe

- Timeframes: {', '.join(tf for tf, _ in config.timeframes)}
- Confirmation orders tracked causally: {config.confirmation_orders}
- Initial admission rule: every order-1 causal Swing Low; no prominence, age, trend, volume, OBI, footprint, wall or confluence filter.
- Levels by timeframe: {timeframe_counts}
- Total levels: {len(levels):,}

## Lifecycle semantics

- A level enters the unconsumed pool only at `initial_available_time`, after the right-side HTF confirmation bar closes.
- Touching a level does not consume it.
- The first true downward sweep consumes the stop-liquidity pool.
- Reclaim versus acceptance below is recorded separately as the support outcome; a reclaimed level is not silently restored to the unconsumed stop pool.
- There is no arbitrary age expiry. A level can remain active for hours, days or months until first sweep or dataset end.

## Broad funnel

- Stage counts: {stage_counts}
- Swept levels: {int(lifecycle['sweep_pos'].ge(0).sum()) if not lifecycle.empty else 0:,}
- Still unconsumed at dataset end: {int(lifecycle['sweep_pos'].lt(0).sum()) if not lifecycle.empty else 0:,}

## What is intentionally not done

- No final entry/exit strategy.
- No fee-adjusted PnL claim.
- No OBI, Range Footprint or Liquidity Map filter.
- No parameter optimization.
- No requirement that a level be the latest low or be respected twice.

## Causal audit

Total violations: {violations}. Any non-zero value invalidates the atlas until fixed.

## Next research use

Use the atlas to test, one attribute at a time, which Swing Lows are valuable: timeframe, causal confirmation order, age, prior touches, cross-timeframe clustering, approach path, sweep depth, aggressive selling, price-impact efficiency, and finally Books replenishment/withdrawal during the 2025-10 to 2026-06 coverage window.
"""


def _synthetic_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=1_200, freq="1min")
    base = 2000.0 + np.sin(np.arange(len(index)) / 45.0) * 12.0 + np.sin(np.arange(len(index)) / 9.0) * 2.0
    close = base + np.sin(np.arange(len(index)) / 3.0) * 0.4
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "notional": 1_000_000.0, "trades_count": 100.0,
            "buy_notional": 500_000.0, "sell_notional": 500_000.0,
            "delta_notional": 0.0,
        },
        index=index,
    )


def run_self_test() -> None:
    config = AtlasConfig(timeframes=(("15m", 15), ("30m", 30), ("1H", 60)), confirmation_orders=(1, 2, 3)).validate()
    bars = _synthetic_bars()
    levels = build_swing_low_universe(bars, config)
    if levels.empty:
        raise RuntimeError("self-test produced no levels")
    lifecycle = build_level_lifecycle(bars, levels, config, show_progress=False)
    events = build_event_table(lifecycle, bars, config)
    events = attach_active_confluence(events, lifecycle, config)
    events = attach_forward_paths(events, bars, config)
    audit = causal_audit(levels, lifecycle, events, config)
    if int(audit["violations"].sum()) != 0:
        raise RuntimeError(f"self-test causal audit failed:\n{audit}")
    print(f"[self-test] passed levels={len(levels):,} events={len(events):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return Path(args.out_dir)
    config = _config(args)
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start} -> {research_end_exclusive}", flush=True)
    print(f"[swing] timeframes={config.timeframes} orders={config.confirmation_orders}", flush=True)

    bars = _load_bars(args)
    print("[stage] build broad causal Swing Low universe", flush=True)
    levels = build_swing_low_universe(bars, config)
    if levels.empty:
        raise RuntimeError("No causal Swing Low levels generated")
    print(f"[levels] total={len(levels):,} by_tf={levels['source_timeframe'].value_counts().to_dict()}", flush=True)

    print("[stage] track unconsumed lifecycle without arbitrary expiry", flush=True)
    lifecycle = build_level_lifecycle(bars, levels, config, show_progress=not args.no_progress)
    print(
        f"[lifecycle] levels={len(lifecycle):,} swept={int(lifecycle['sweep_pos'].ge(0).sum()):,} "
        f"active_end={int(lifecycle['sweep_pos'].lt(0).sum()):,}",
        flush=True,
    )

    print("[stage] build approach/touch/sweep/reclaim event funnel", flush=True)
    events = build_event_table(lifecycle, bars, config)
    events = attach_active_confluence(events, lifecycle, config)
    events = attach_forward_paths(events, bars, config)
    events = events.loc[
        (pd.to_datetime(events["event_available_time"]) >= research_start)
        & (pd.to_datetime(events["event_available_time"]) < research_end_exclusive)
    ].reset_index(drop=True)
    if events.empty:
        raise RuntimeError("No atlas events in requested research window")
    events["period"] = fixed_period(pd.to_datetime(events["event_available_time"]))
    print(f"[events] total={len(events):,} stages={events['event_stage'].value_counts().to_dict()}", flush=True)

    print("[stage] causal audit and descriptive reports", flush=True)
    audit = causal_audit(levels, lifecycle, events, config)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"Causal audit failed with {violations} violations:\n{audit.to_string(index=False)}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "warmup_start_date": str(args.warmup_start_date),
        "research_start": str(research_start),
        "research_end_exclusive": str(research_end_exclusive),
        "data_source": args.data_source,
        "config": asdict(config),
        "notes": [
            "Research atlas only; not a strategy backtest.",
            "All HTF levels use available_time after right-side bars close.",
            "No admission filter beyond causal order-1 Swing Low.",
            "First sweep consumes stop liquidity; support reclaim is a separate outcome.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(_data_quality(bars, levels, lifecycle, events), out_dir / "01_data_quality.csv")
    _write_csv(level_summary(levels), out_dir / "02_level_universe_summary.csv")
    _write_csv(levels, out_dir / "03_swing_level_table.csv")
    _write_csv(lifecycle, out_dir / "04_level_lifecycle_table.csv")
    _write_csv(event_stage_summary(events), out_dir / "05_event_stage_summary.csv")
    _write_csv(lifecycle_summary(lifecycle), out_dir / "06_lifecycle_summary.csv")
    _write_csv(forward_path_summary(events, config), out_dir / "07_forward_path_summary.csv")
    _write_csv(attribute_bin_summary(events, config), out_dir / "08_sweep_attribute_bins.csv")
    _write_csv(audit, out_dir / "09_causal_audit.csv")

    sample_size = max(0, int(args.event_sample_size))
    if sample_size:
        if len(events) <= sample_size:
            sample = events.copy()
        else:
            per_stage = max(1, sample_size // max(1, events["event_stage"].nunique()))
            pieces = [
                part.sample(min(len(part), per_stage), random_state=42)
                for _, part in events.groupby("event_stage", sort=False)
            ]
            sample = pd.concat(pieces, ignore_index=True, sort=False)
        _write_csv(sample, out_dir / "10_event_sample.csv")
    if args.write_full_events:
        _write_csv(events, out_dir / "11_full_event_table.csv")

    brief = _brief(config, levels, lifecycle, events, audit)
    (out_dir / "12_research_brief.md").write_text(brief, encoding="utf-8")
    review = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={review.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
