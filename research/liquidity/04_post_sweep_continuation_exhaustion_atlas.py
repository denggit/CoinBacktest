#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R04 Post-Sweep Continuation, Exhaustion and Reversal Atlas.

This is mechanism research, not a final strategy backtest. It starts from the
causal first-sweep Swing Liquidity Zone universe built in R03, then follows each
closed 1m bar after the sweep. It studies whether active selling remains price-
efficient, whether CVD/Delta continue lower without equivalent price progress,
whether repeated new-low attempts shrink, and which causal confirmation states
retain future MFE while reducing future MAE.

Large future-MFE opportunities and oracle turning points are extracted only as
future labels for feature discovery. They are never used to admit checkpoints or
choose thresholds in this script.
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

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.post_sweep_process import (  # noqa: E402
    PostSweepConfig,
    build_post_sweep_checkpoint_table,
    causal_audit,
    checkpoint_path_summary,
    confirmation_state_summary,
    large_mfe_feature_profile,
    large_mfe_summary,
    new_low_attempt_summary,
    oracle_turning_point_table,
    orderflow_fixed_bin_summary,
    period_stability_summary,
    split_checkpoint_features_labels,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.swing_liquidity_atlas import (  # noqa: E402
    AtlasConfig,
    build_level_lifecycle,
    build_swing_low_universe,
    normalize_primary_bars,
    normalize_timeframes,
)
from src.research_common.swing_liquidity_zone_study import (  # noqa: E402
    ZoneStudyConfig,
    attach_causal_market_features,
    build_causal_market_feature_frame,
    build_sweep_zone_events,
)

SCRIPT_NAME = "04_post_sweep_continuation_exhaustion_atlas"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_POST_SWEEP_CONTINUATION_EXHAUSTION_REVERSAL_R04"
EDGE_ID = "RESEARCH_ONLY_POST_SWEEP_STATE_TRANSITION"
TITLE = "ETH Post-Sweep Continuation, Exhaustion and Reversal Atlas R04"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/post_sweep_continuation_exhaustion_r04"
DEFAULT_R03_EVENTS = "data/reports/research/liquidity/swing_liquidity_zone_sweep_mechanism_r03/04_online_first_zone_feature_table.csv"


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _comma_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in str(value).split(",") if item.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _comma_floats(value: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(item.strip()) for item in str(value).split(",") if item.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study post-Swing-zone-sweep continuation, order-flow exhaustion and causal reversal confirmations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--swing-timeframes", default="15m,30m,1H,4H,1D")
    parser.add_argument("--confirmation-orders", type=_comma_ints, default=(1, 2, 3, 5))
    parser.add_argument("--zone-merge-tolerance-bp", type=float, default=10.0)
    parser.add_argument("--impulse-gap-bars", type=int, default=5)
    parser.add_argument("--impulse-price-tolerance-bp", type=float, default=50.0)
    parser.add_argument("--observation-horizon-bars", type=int, default=180)
    parser.add_argument("--dense-checkpoint-bars", type=int, default=30)
    parser.add_argument("--fixed-checkpoint-bars", type=_comma_ints, default=(45, 60, 90, 120, 180))
    parser.add_argument("--flow-windows", type=_comma_ints, default=(1, 3, 5, 15, 30))
    parser.add_argument("--future-horizons", type=_comma_ints, default=(5, 15, 30, 60, 180))
    parser.add_argument("--large-mfe-returns", type=_comma_floats, default=(0.005, 0.01, 0.02))
    parser.add_argument("--new-low-epsilon-bp", type=float, default=0.0)
    parser.add_argument("--r03-event-csv", default=DEFAULT_R03_EVENTS)
    parser.add_argument("--rebuild-zone-events", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-build-missing", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _end_exclusive(value: str) -> pd.Timestamp:
    text = str(value).strip()
    timestamp = pd.Timestamp(text)
    if len(text) <= 10:
        return timestamp + pd.Timedelta(days=1)
    return timestamp + pd.Timedelta(microseconds=1)


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[load] trade_bar 1m symbol={args.symbol} window={args.warmup_start_date}->{args.end_date}", flush=True)
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe="1m", data_dir=args.data_dir, db_name=args.db_name)
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=not bool(args.no_build_missing),
    )
    keep = [
        "open", "high", "low", "close", "volume", "notional", "trades_count",
        "buy_notional", "sell_notional", "buy_trades_count", "sell_trades_count",
        "delta_notional", "cvd_notional", "large_buy_notional", "large_sell_notional",
        "large_delta_notional", "large_buy_trades_count", "large_sell_trades_count",
        "max_trade_notional",
    ]
    bars = normalize_primary_bars(bars.loc[:, [name for name in keep if name in bars.columns]].copy())
    required = {"buy_notional", "sell_notional", "delta_notional", "notional"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise RuntimeError(f"R04 requires 1m trade-bar order flow; missing columns={missing}")
    print(f"[load] rows={len(bars):,} range={bars.index.min()}->{bars.index.max()} cols={len(bars.columns)}", flush=True)
    return bars


def _atlas_config(args: argparse.Namespace) -> AtlasConfig:
    return AtlasConfig(
        timeframes=normalize_timeframes(_comma_strings(args.swing_timeframes)),
        confirmation_orders=tuple(args.confirmation_orders),
        approach_distance_bp=200.0,
        touch_distance_bp=5.0,
        sweep_epsilon_bp=0.01,
        acceptance_depth_bp=50.0,
        acceptance_consecutive_closes=3,
        resolution_horizon_bars=180,
        forward_horizons=(5, 15, 30, 60, 180),
        confluence_tolerances_bp=(5.0, 10.0, 25.0, 50.0),
    ).validate()


def _zone_config(args: argparse.Namespace) -> ZoneStudyConfig:
    return ZoneStudyConfig(
        zone_merge_tolerance_bp=float(args.zone_merge_tolerance_bp),
        zone_merge_sensitivity_bp=(5.0, 10.0, 25.0, 50.0),
        impulse_gap_bars=int(args.impulse_gap_bars),
        impulse_price_tolerance_bp=float(args.impulse_price_tolerance_bp),
        path_horizons=(5, 15, 30, 60, 180),
        tp_returns=(0.0025, 0.005, 0.01, 0.02),
        control_max_per_zone=0,
    ).validate()


def _post_config(args: argparse.Namespace) -> PostSweepConfig:
    return PostSweepConfig(
        observation_horizon_bars=int(args.observation_horizon_bars),
        dense_checkpoint_bars=int(args.dense_checkpoint_bars),
        fixed_checkpoint_bars=tuple(int(v) for v in args.fixed_checkpoint_bars),
        flow_windows=tuple(int(v) for v in args.flow_windows),
        future_horizons=tuple(int(v) for v in args.future_horizons),
        large_mfe_returns=tuple(float(v) for v in args.large_mfe_returns),
        new_low_epsilon_bp=float(args.new_low_epsilon_bp),
        sample_rows=int(args.sample_rows),
    ).validate()


def _read_r03_events(path: Path, bars: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    print(f"[events] reuse R03 causal event table: {path}", flush=True)
    events = pd.read_csv(path, low_memory=False)
    for name in [c for c in events.columns if c.endswith("time")]:
        events[name] = pd.to_datetime(events[name], errors="coerce")
    events["event_pos"] = pd.to_numeric(events["event_pos"], errors="coerce").astype("Int64")
    events = events.dropna(subset=["event_pos", "event_available_time"]).copy()
    events["event_pos"] = events["event_pos"].astype(int)
    available = pd.to_datetime(events["event_available_time"], errors="coerce")
    events = events.loc[(available >= start) & (available < end_exclusive)].reset_index(drop=True)
    if events.empty:
        raise RuntimeError("R03 event table has no rows in the requested research window")
    if events["event_pos"].max() >= len(bars):
        raise RuntimeError("R03 event positions exceed loaded 1m bar frame; warmup/data window mismatch")
    expected = pd.DatetimeIndex(bars.index)[events["event_pos"].to_numpy()] + pd.Timedelta(minutes=1)
    mismatch = pd.to_datetime(events["event_available_time"]).to_numpy(dtype="datetime64[ns]") != expected.to_numpy(dtype="datetime64[ns]")
    if int(mismatch.sum()):
        raise RuntimeError(f"R03 event table does not align with current bars: mismatches={int(mismatch.sum())}")
    print(f"[events] reused={len(events):,}", flush=True)
    return events


def _build_zone_events(
    args: argparse.Namespace,
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    atlas_cfg: AtlasConfig,
    zone_cfg: ZoneStudyConfig,
) -> pd.DataFrame:
    print("[stage] rebuild broad causal Swing Low universe", flush=True)
    levels = build_swing_low_universe(bars, atlas_cfg)
    lifecycle = build_level_lifecycle(bars, levels, atlas_cfg, show_progress=not args.no_progress)
    zones = build_sweep_zone_events(lifecycle, bars, zone_cfg)
    event_time = pd.to_datetime(zones["event_available_time"], errors="coerce")
    zones = zones.loc[(event_time >= start) & (event_time < end_exclusive)].reset_index(drop=True)
    feature_frame = build_causal_market_feature_frame(bars, zone_cfg)
    zones = attach_causal_market_features(zones, bars, zone_cfg, feature_frame=feature_frame)
    zones = zones.loc[zones["is_impulse_first_event"].astype(bool)].reset_index(drop=True)
    print(f"[events] levels={len(levels):,} first-zone-sweeps={len(zones):,}", flush=True)
    return zones


def _load_or_build_events(
    args: argparse.Namespace,
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    atlas_cfg: AtlasConfig,
    zone_cfg: ZoneStudyConfig,
) -> pd.DataFrame:
    path = Path(args.r03_event_csv)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists() and not args.rebuild_zone_events:
        return _read_r03_events(path, bars, start, end_exclusive)
    return _build_zone_events(args, bars, start, end_exclusive, atlas_cfg, zone_cfg)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["empty_result"])
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", compression={"method": "gzip", "compresslevel": 1})


def _data_quality(bars: pd.DataFrame, events: pd.DataFrame, checkpoints: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    gaps = bars.index.to_series().diff().dropna()
    return pd.DataFrame([
        {"metric": "primary_rows", "value": len(bars)},
        {"metric": "primary_start", "value": bars.index.min()},
        {"metric": "primary_end", "value": bars.index.max()},
        {"metric": "primary_duplicate_timestamps", "value": int(bars.index.duplicated().sum())},
        {"metric": "primary_max_gap_seconds", "value": float(gaps.dt.total_seconds().max()) if len(gaps) else 0.0},
        {"metric": "zone_sweep_events", "value": len(events)},
        {"metric": "post_sweep_checkpoints", "value": len(checkpoints)},
        {"metric": "events_with_checkpoints", "value": int(checkpoints["zone_event_id"].nunique()) if len(checkpoints) else 0},
        {"metric": "median_checkpoints_per_event", "value": float(checkpoints.groupby("zone_event_id").size().median()) if len(checkpoints) else 0.0},
        {"metric": "new_low_attempt_checkpoints", "value": int(checkpoints["new_low_attempt_flag"].sum()) if len(checkpoints) else 0},
        {"metric": "oracle_turning_points_future_labelled", "value": len(oracle)},
        {"metric": "buy_sell_delta_columns_present", "value": all(name in bars.columns for name in ["buy_notional", "sell_notional", "delta_notional"])},
    ])


def _large_opportunity_sample(checkpoints: pd.DataFrame, cfg: PostSweepConfig, sample_rows: int) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    horizon = max(cfg.future_horizons)
    threshold = min(cfg.large_mfe_returns)
    fixed = checkpoints["elapsed_bars"].isin([1, 3, 5, 10, 15, 30, 60])
    selected = checkpoints.loc[
        fixed & (pd.to_numeric(checkpoints[f"future_mfe_{horizon}m"], errors="coerce") >= threshold)
    ].copy()
    selected = selected.sort_values(f"future_mfe_{horizon}m", ascending=False, kind="mergesort")
    return selected.head(max(0, int(sample_rows))).reset_index(drop=True)


def _brief(
    atlas_cfg: AtlasConfig,
    zone_cfg: ZoneStudyConfig,
    post_cfg: PostSweepConfig,
    events: pd.DataFrame,
    checkpoints: pd.DataFrame,
    oracle: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) if not audit.empty else 0
    return f"""# Post-Sweep Continuation, Exhaustion and Reversal Atlas R04

## Decision this study supports

After the first causal sweep of an active Swing Liquidity Zone, determine when selling remains effective, when price response to negative Delta begins to weaken, and which causal confirmation state retains enough future MFE while reducing future MAE.

## Universe

- First online zone sweeps: {len(events):,}
- Sparse process checkpoints: {len(checkpoints):,}
- Future-labelled oracle turning points: {len(oracle):,}
- Swing timeframes: {atlas_cfg.timeframes}
- Zone merge: {zone_cfg.zone_merge_tolerance_bp:g}bp

## Causal checkpoint design

- Every closed 1m bar is observed for the first {post_cfg.dense_checkpoint_bars} minutes.
- Later fixed checkpoints: {post_cfg.fixed_checkpoint_bars}.
- Every additional new-low attempt is retained even when it is not on the fixed schedule.
- Features use only information available by checkpoint close.
- Future paths begin at the next 1m open.

## Mechanisms measured

1. Buy/sell notional, Delta ratio, cumulative Delta and large-trade Delta.
2. Price movement per unit sell notional and per unit negative Delta.
3. CVD making a new low while price does not make an equivalent new low.
4. Repeated new-low extension, extension shrinkage and time since the last attempt.
5. Micro-high breaks, local no-new-low states and Zone reclaim states.

## Large opportunity extraction

Fixed future-MFE labels: {post_cfg.large_mfe_returns} over {max(post_cfg.future_horizons)} minutes.
These labels only select descriptive samples and feature profiles. They are not entry filters, and no threshold is fitted from their returns.

The oracle turning-point table explicitly uses future knowledge. It exists only to describe what the market looked like near eventual durable reversals.

## Overfitting controls

- No model is fitted.
- No irregular sample quantile becomes a strategy rule.
- Reports use fixed periods and rounded, predeclared outcome thresholds.
- Feature and future-label tables are physically separated.
- Holdout-period results are descriptive and do not modify admission logic.

## Deferred

- No final entry, stop, take-profit or PnL backtest.
- No Books, iceberg or Range Footprint enrichment yet.
- No 5s/10s model until the full-history 1m order-flow mechanism is validated.

## Causal audit

Total violations: {violations}. Any non-zero value invalidates the report.
"""


def _synthetic_bars(periods: int = 8_000) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1min")
    x = np.arange(periods, dtype=float)
    close = 2_000.0 + 18.0 * np.sin(x / 220.0) + 3.0 * np.sin(x / 19.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    delta = np.sin(x / 9.0) * 100_000.0
    notional = np.full(periods, 1_000_000.0)
    buy = (notional + delta) / 2.0
    sell = (notional - delta) / 2.0
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "notional": notional, "buy_notional": buy, "sell_notional": sell,
        "delta_notional": delta, "trades_count": 100.0,
        "large_buy_notional": np.maximum(delta, 0.0),
        "large_sell_notional": np.maximum(-delta, 0.0),
        "large_delta_notional": delta,
    }, index=index)


def run_self_test() -> None:
    bars = _synthetic_bars()
    atlas_cfg = AtlasConfig(timeframes=(("15m", 15), ("30m", 30), ("1H", 60)), confirmation_orders=(1, 2, 3)).validate()
    zone_cfg = ZoneStudyConfig(path_horizons=(5, 15, 30, 60, 180), tp_returns=(0.005, 0.01), control_max_per_zone=0).validate()
    post_cfg = PostSweepConfig(observation_horizon_bars=60, dense_checkpoint_bars=15, fixed_checkpoint_bars=(30, 60), future_horizons=(5, 15, 30, 60)).validate()
    levels = build_swing_low_universe(bars, atlas_cfg)
    lifecycle = build_level_lifecycle(bars, levels, atlas_cfg, show_progress=False)
    zones = build_sweep_zone_events(lifecycle, bars, zone_cfg)
    features_frame = build_causal_market_feature_frame(bars, zone_cfg)
    zones = attach_causal_market_features(zones, bars, zone_cfg, feature_frame=features_frame)
    zones = zones.loc[zones["is_impulse_first_event"].astype(bool)].head(50).reset_index(drop=True)
    checkpoints = build_post_sweep_checkpoint_table(zones, bars, post_cfg, show_progress=False)
    features, labels = split_checkpoint_features_labels(checkpoints)
    audit = causal_audit(features, labels)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if checkpoints.empty or violations:
        raise RuntimeError(f"R04 self-test failed checkpoints={len(checkpoints)} audit=\n{audit}")
    print(f"[self-test] passed events={len(zones):,} checkpoints={len(checkpoints):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir

    atlas_cfg = _atlas_config(args)
    zone_cfg = _zone_config(args)
    post_cfg = _post_config(args)
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start}->{research_end_exclusive}", flush=True)
    print(f"[scope] observation={post_cfg.observation_horizon_bars}m flow_windows={post_cfg.flow_windows} future={post_cfg.future_horizons}", flush=True)

    bars = _load_bars(args)
    events = _load_or_build_events(args, bars, research_start, research_end_exclusive, atlas_cfg, zone_cfg)
    if events.empty:
        raise RuntimeError("No causal first-sweep zone events")

    print("[stage] post-sweep sparse process checkpoints", flush=True)
    checkpoints = build_post_sweep_checkpoint_table(events, bars, post_cfg, show_progress=not args.no_progress)
    if checkpoints.empty:
        raise RuntimeError("No post-sweep checkpoints generated")
    print(f"[checkpoints] rows={len(checkpoints):,} events={checkpoints['zone_event_id'].nunique():,}", flush=True)

    features, labels = split_checkpoint_features_labels(checkpoints)
    audit = causal_audit(features, labels)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"Causal audit failed with {violations} violations:\n{audit.to_string(index=False)}")

    print("[stage] continuation, exhaustion and large-MFE descriptive reports", flush=True)
    attempts = new_low_attempt_summary(checkpoints, post_cfg)
    paths = checkpoint_path_summary(checkpoints, post_cfg)
    confirmations = confirmation_state_summary(checkpoints, post_cfg)
    large_summary = large_mfe_summary(checkpoints, post_cfg)
    large_profile = large_mfe_feature_profile(checkpoints, post_cfg, event_features=events)
    oracle = oracle_turning_point_table(checkpoints, post_cfg)
    fixed_bins = orderflow_fixed_bin_summary(checkpoints, post_cfg)
    stability = period_stability_summary(confirmations)
    large_sample = _large_opportunity_sample(checkpoints, post_cfg, args.sample_rows)

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
        "atlas_config": asdict(atlas_cfg),
        "zone_config": asdict(zone_cfg),
        "post_sweep_config": asdict(post_cfg),
        "r03_event_source": str(args.r03_event_csv),
        "notes": [
            "Mechanism research only; no final trade backtest.",
            "Large-MFE and oracle-turn labels use future outcomes only for descriptive feature extraction.",
            "All checkpoint features are available by the closed checkpoint bar.",
            "Future paths begin at the next 1m open.",
            "Irregular training-sample quantile boundaries are not promoted to strategy parameters.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(_data_quality(bars, events, checkpoints, oracle), out_dir / "01_data_quality.csv")
    _write_csv(attempts, out_dir / "02_new_low_attempt_summary.csv")
    _write_csv(paths, out_dir / "03_checkpoint_path_summary.csv")
    _write_csv(confirmations, out_dir / "04_confirmation_state_summary.csv")
    _write_csv(fixed_bins, out_dir / "05_orderflow_fixed_bin_summary.csv")
    _write_csv(stability, out_dir / "06_confirmation_period_stability.csv")
    _write_csv(large_summary, out_dir / "07_large_mfe_summary.csv")
    _write_csv(large_profile, out_dir / "08_large_mfe_feature_profile.csv")
    _write_csv(oracle.head(max(0, int(args.sample_rows))), out_dir / "09_oracle_turning_point_sample.csv")
    _write_csv(large_sample, out_dir / "10_large_reversal_opportunity_sample.csv")
    _write_csv(events, out_dir / "11_static_zone_event_features.csv")
    _write_csv(audit, out_dir / "12_causal_audit.csv")
    _write_gzip_csv(features, out_dir / "13_checkpoint_feature_table.csv.gz")
    _write_gzip_csv(labels, out_dir / "14_checkpoint_label_table.csv.gz")
    sample_size = max(0, int(args.sample_rows))
    sample = checkpoints if len(checkpoints) <= sample_size else checkpoints.sample(sample_size, random_state=42)
    _write_csv(sample, out_dir / "15_checkpoint_sample.csv")
    brief = _brief(atlas_cfg, zone_cfg, post_cfg, events, checkpoints, oracle, audit)
    (out_dir / "16_research_brief.md").write_text(brief, encoding="utf-8")
    review = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={review.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
