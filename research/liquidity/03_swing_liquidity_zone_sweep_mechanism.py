#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03 causal Swing Liquidity Zone sweep mechanism study.

R03 is deliberately not a final strategy backtest.  It converts the broad R02
level-centric first-sweep universe into causally observable same-bar price zones,
removes repeated signals with an online impulse rule, studies Swing Low quality,
normalizes sweep depth by pre-event volatility, and measures next-open MFE/MAE,
structural-low survival and TP-before-lower-low paths out to three days.

Order-book, CVD, footprint and iceberg filters are intentionally deferred.  The
model-ready feature table contains only information known by the closed sweep
bar; future outcomes live in a separate label table.
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

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
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
    attach_structural_path_outcomes,
    build_causal_market_feature_frame,
    build_matched_controls,
    build_sweep_zone_events,
    zone_merge_sensitivity_summary,
)
from src.research_common.swing_liquidity_zone_study.reports import (  # noqa: E402
    causal_audit,
    control_comparison,
    control_match_balance,
    feature_bin_summary,
    fixed_period,
    path_horizon_summary,
    structural_exit_summary,
)

SCRIPT_NAME = "03_swing_liquidity_zone_sweep_mechanism"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_SWING_LIQUIDITY_ZONE_SWEEP_MECHANISM_R03"
EDGE_ID = "RESEARCH_ONLY_SWING_LIQUIDITY_ZONE_FIRST_SWEEP_REVERSAL"
TITLE = "ETH Swing Liquidity Zone Sweep Mechanism Study R03"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/swing_liquidity_zone_sweep_mechanism_r03"


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
        description="Study causal unconsumed Swing Low zone sweeps, volatility-normalized depth and multi-day structural paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--swing-timeframes", default="15m,30m,1H,4H,1D")
    parser.add_argument("--confirmation-orders", type=_comma_ints, default=(1, 2, 3, 5))
    parser.add_argument("--approach-distance-bp", type=float, default=200.0)
    parser.add_argument("--touch-distance-bp", type=float, default=5.0)
    parser.add_argument("--sweep-epsilon-bp", type=float, default=0.01)
    parser.add_argument("--zone-merge-tolerance-bp", type=float, default=10.0)
    parser.add_argument("--zone-merge-sensitivity-bp", type=_comma_floats, default=(5.0, 10.0, 25.0, 50.0))
    parser.add_argument("--impulse-gap-bars", type=int, default=5)
    parser.add_argument("--impulse-price-tolerance-bp", type=float, default=50.0)
    parser.add_argument("--path-horizons", type=_comma_ints, default=(5, 15, 30, 60, 180, 360, 720, 1440, 2880, 4320))
    parser.add_argument("--tp-returns", type=_comma_floats, default=(0.0025, 0.005, 0.01, 0.02, 0.03))
    parser.add_argument("--structural-break-epsilon-bp", type=float, default=0.01)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--control-exclusion-bars", type=int, default=5)
    parser.add_argument("--control-min-downside-atr", type=float, default=0.25)
    parser.add_argument("--data-source", choices=["trade_bar", "ohlcv_local"], default="trade_bar")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-build-missing", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--event-sample-size", type=int, default=20_000)
    parser.add_argument("--write-full-lifecycle", action="store_true")
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
    if str(args.timeframe) != "1m":
        raise ValueError("R03 requires --timeframe 1m")
    print(f"[load] source={args.data_source} symbol={args.symbol} window={args.warmup_start_date}->{args.end_date}", flush=True)
    if args.data_source == "trade_bar":
        loader = OKXTradeBarLoader(symbol=args.symbol, timeframe="1m", data_dir=args.data_dir, db_name=args.db_name)
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
            if not isinstance(bars.index, pd.DatetimeIndex):
                bars.index = pd.to_datetime(bars.index, errors="coerce")
            bars = bars.loc[(bars.index >= start) & (bars.index <= end)]
    keep = [
        "open", "high", "low", "close", "volume", "notional", "trades_count",
        "buy_notional", "sell_notional", "delta_notional",
        "large_buy_notional", "large_sell_notional", "large_delta_notional",
    ]
    bars = normalize_primary_bars(bars.loc[:, [name for name in keep if name in bars.columns]].copy())
    print(f"[load] rows={len(bars):,} range={bars.index.min()}->{bars.index.max()} cols={len(bars.columns)}", flush=True)
    return bars


def _atlas_config(args: argparse.Namespace) -> AtlasConfig:
    return AtlasConfig(
        timeframes=normalize_timeframes(_comma_strings(args.swing_timeframes)),
        confirmation_orders=tuple(args.confirmation_orders),
        approach_distance_bp=float(args.approach_distance_bp),
        touch_distance_bp=float(args.touch_distance_bp),
        sweep_epsilon_bp=float(args.sweep_epsilon_bp),
        acceptance_depth_bp=50.0,
        acceptance_consecutive_closes=3,
        resolution_horizon_bars=180,
        forward_horizons=(5, 15, 30, 60, 180),
        confluence_tolerances_bp=(5.0, 10.0, 25.0, 50.0),
    ).validate()


def _zone_config(args: argparse.Namespace) -> ZoneStudyConfig:
    return ZoneStudyConfig(
        zone_merge_tolerance_bp=float(args.zone_merge_tolerance_bp),
        zone_merge_sensitivity_bp=tuple(float(v) for v in args.zone_merge_sensitivity_bp),
        impulse_gap_bars=int(args.impulse_gap_bars),
        impulse_price_tolerance_bp=float(args.impulse_price_tolerance_bp),
        path_horizons=tuple(int(v) for v in args.path_horizons),
        tp_returns=tuple(float(v) for v in args.tp_returns),
        structural_break_epsilon_bp=float(args.structural_break_epsilon_bp),
        control_exclusion_bars=int(args.control_exclusion_bars),
        control_min_downside_atr=float(args.control_min_downside_atr),
        control_max_per_zone=0 if args.no_controls else 1,
    ).validate()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _event_window(frame: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ts = pd.to_datetime(frame["event_available_time"], errors="coerce")
    return frame.loc[(ts >= start) & (ts < end_exclusive)].reset_index(drop=True)


def _feature_label_split(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_prefixes = (
        "entry_reference_", "first_lower_low_", "bars_to_lower_low", "first_zone_", "bars_to_zone_",
        "close_return_", "mfe_", "mae_", "structural_low_survival_", "zone_floor_reclaim_by_",
        "zone_ceiling_reclaim_by_", "first_tp_", "bars_to_tp_", "tp_",
    )
    id_columns = [name for name in ("zone_event_id", "event_kind", "matched_zone_event_id") if name in outcomes.columns]
    label_columns = id_columns + [name for name in outcomes.columns if name.startswith(label_prefixes)]
    label_columns = list(dict.fromkeys(label_columns))
    labels = outcomes.loc[:, label_columns].copy()
    feature_columns = [name for name in outcomes.columns if name not in set(label_columns) or name in id_columns]
    features = outcomes.loc[:, feature_columns].copy()
    # Future-completed online impulse size is intentionally absent. Only count-so-far is retained.
    forbidden = [name for name in features.columns if "future" in name.lower()]
    if forbidden:
        features = features.drop(columns=forbidden)
    return features, labels


def _data_quality(
    bars: pd.DataFrame,
    levels: pd.DataFrame,
    lifecycle: pd.DataFrame,
    zones: pd.DataFrame,
    decision_zones: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    raw_sweeps_all: int,
    raw_sweeps_research: int,
) -> pd.DataFrame:
    gaps = bars.index.to_series().diff().dropna()
    return pd.DataFrame(
        [
            {"metric": "primary_rows", "value": len(bars)},
            {"metric": "primary_start", "value": bars.index.min()},
            {"metric": "primary_end", "value": bars.index.max()},
            {"metric": "primary_duplicate_timestamps", "value": int(bars.index.duplicated().sum())},
            {"metric": "primary_max_gap_seconds", "value": float(gaps.dt.total_seconds().max()) if len(gaps) else 0.0},
            {"metric": "swing_levels", "value": len(levels)},
            {"metric": "raw_level_sweeps_all_loaded", "value": int(raw_sweeps_all)},
            {"metric": "raw_level_sweeps_research_window", "value": int(raw_sweeps_research)},
            {"metric": "same_bar_zone_events", "value": len(zones)},
            {"metric": "online_impulse_first_events", "value": len(decision_zones)},
            {"metric": "matched_controls", "value": len(controls)},
            {"metric": "zone_reduction_vs_raw_research", "value": 1.0 - len(zones) / max(int(raw_sweeps_research), 1)},
            {"metric": "impulse_reduction_vs_raw_research", "value": 1.0 - len(decision_zones) / max(int(raw_sweeps_research), 1)},
        ]
    )


def _brief(
    atlas_cfg: AtlasConfig,
    zone_cfg: ZoneStudyConfig,
    raw_sweeps: int,
    zones: pd.DataFrame,
    decisions: pd.DataFrame,
    controls: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) if not audit.empty else 0
    return f"""# Swing Liquidity Zone Sweep Mechanism Study R03

## Decision this study supports

Determine which causally visible, still-unconsumed 15m/30m/1H/4H/1D Swing Low zones are most likely to produce a durable reversal after their first sweep, before adding Books, CVD, footprint or iceberg confirmation.

## What changed from R02

- Raw level sweeps: {raw_sweeps:,}.
- Same-bar price-zone events at {zone_cfg.zone_merge_tolerance_bp:g}bp merge tolerance: {len(zones):,}.
- Online first events after causal five-minute impulse de-duplication: {len(decisions):,}.
- Matched non-zone downside controls: {len(controls):,}.
- A zone uses only levels already available by the closed sweep bar.
- Outcome paths start at the next 1m open.

## Structural questions

1. Does Swing timeframe, age, freshness, causal confirmation order, pivot shape or multi-timeframe composition improve later reversal paths?
2. Is absolute sweep depth useful after dividing it by pre-event 60m/240m/1440m ATR?
3. Does a swept structural low remain unbroken for hours or days, and how much MFE appears before a genuinely lower low?
4. Are zone sweeps better than matched downside impulses that did not sweep any active unconsumed Swing Low?

## Outcome design

- Horizons: {zone_cfg.path_horizons} minutes, up to {max(zone_cfg.path_horizons) / 1440:.1f} days.
- MFE uses future highs; MAE uses future lows; both are labels only.
- Structural invalidation is the first future low below the closed sweep bar low by {zone_cfg.structural_break_epsilon_bp:g}bp.
- TP-before-lower-low labels: {zone_cfg.tp_returns}.
- Feature and label tables are physically separated for later modelling.

## Intentionally deferred

- No final entry filter or strategy PnL.
- No OBI, Books replenishment, iceberg, CVD divergence or Range Footprint filter.
- No model fitting and no threshold chosen from holdout returns.
- No assumption that a five-minute reclaim is known at sweep time.

## Causal audit

Total violations: {violations}. Any non-zero value invalidates the report.

## Configuration

- Swing timeframes: {atlas_cfg.timeframes}
- Causal confirmation orders: {atlas_cfg.confirmation_orders}
- Zone sensitivity: {zone_cfg.zone_merge_sensitivity_bp}bp
"""


def _synthetic_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4_000, freq="1min")
    x = np.arange(len(index), dtype=float)
    close = 2_000.0 + 20.0 * np.sin(x / 180.0) + 4.0 * np.sin(x / 17.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "notional": 1_000_000.0, "trades_count": 100.0,
            "buy_notional": 500_000.0, "sell_notional": 500_000.0, "delta_notional": 0.0,
        },
        index=index,
    )


def run_self_test() -> None:
    bars = _synthetic_bars()
    atlas_cfg = AtlasConfig(timeframes=(("15m", 15), ("30m", 30), ("1H", 60)), confirmation_orders=(1, 2, 3)).validate()
    zone_cfg = ZoneStudyConfig(path_horizons=(5, 15, 60, 180), tp_returns=(0.0025, 0.005)).validate()
    levels = build_swing_low_universe(bars, atlas_cfg)
    lifecycle = build_level_lifecycle(bars, levels, atlas_cfg, show_progress=False)
    zones = build_sweep_zone_events(lifecycle, bars, zone_cfg)
    features_frame = build_causal_market_feature_frame(bars, zone_cfg)
    zones = attach_causal_market_features(zones, bars, zone_cfg, feature_frame=features_frame)
    decisions = zones.loc[zones["is_impulse_first_event"].astype(bool)].reset_index(drop=True)
    outcomes = attach_structural_path_outcomes(decisions, bars, zone_cfg)
    features, labels = _feature_label_split(outcomes)
    audit = causal_audit(features, labels)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"self-test causal audit failed:\n{audit}")
    print(f"[self-test] passed levels={len(levels):,} raw_sweeps={lifecycle['sweep_pos'].ge(0).sum():,} zones={len(zones):,} decisions={len(decisions):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    atlas_cfg = _atlas_config(args)
    zone_cfg = _zone_config(args)
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start}->{research_end_exclusive}", flush=True)
    print(f"[scope] swing={atlas_cfg.timeframes} merge={zone_cfg.zone_merge_tolerance_bp:g}bp horizons={zone_cfg.path_horizons}", flush=True)

    bars = _load_bars(args)
    print("[stage] broad causal Swing Low universe", flush=True)
    levels = build_swing_low_universe(bars, atlas_cfg)
    if levels.empty:
        raise RuntimeError("No Swing Low levels generated")
    print(f"[levels] total={len(levels):,} by_tf={levels['source_timeframe'].value_counts().to_dict()}", flush=True)

    print("[stage] unconsumed lifecycle and first sweeps", flush=True)
    lifecycle = build_level_lifecycle(bars, levels, atlas_cfg, show_progress=not args.no_progress)
    raw_sweeps_all = int(pd.to_numeric(lifecycle["sweep_pos"], errors="coerce").ge(0).sum())
    sweep_available = pd.to_datetime(lifecycle["sweep_available_time"], errors="coerce")
    raw_sweeps = int(((sweep_available >= research_start) & (sweep_available < research_end_exclusive)).sum())
    print(f"[lifecycle] levels={len(lifecycle):,} raw_level_sweeps_all={raw_sweeps_all:,} research_window={raw_sweeps:,}", flush=True)

    print("[stage] zone merge sensitivity and causal online impulse de-dup", flush=True)
    sensitivity_table = zone_merge_sensitivity_summary(
        lifecycle, bars, zone_cfg,
        tolerances_bp=zone_cfg.zone_merge_sensitivity_bp,
        research_start=research_start, research_end_exclusive=research_end_exclusive,
    )
    for row in sensitivity_table.itertuples(index=False):
        print(f"[zones] tolerance={float(row.zone_merge_tolerance_bp):g}bp zones={int(row.same_bar_zone_events):,} first_impulses={int(row.online_impulse_first_events):,}", flush=True)
    primary_zones = _event_window(build_sweep_zone_events(lifecycle, bars, zone_cfg), research_start, research_end_exclusive)
    if primary_zones.empty:
        raise RuntimeError("No zone events in research window")

    print("[stage] causal pre-sweep volatility and approach features", flush=True)
    market_features = build_causal_market_feature_frame(bars, zone_cfg)
    primary_zones = attach_causal_market_features(primary_zones, bars, zone_cfg, feature_frame=market_features)
    decision_zones = primary_zones.loc[primary_zones["is_impulse_first_event"].astype(bool)].reset_index(drop=True)
    print(f"[decision-universe] zones={len(primary_zones):,} online_first_events={len(decision_zones):,}", flush=True)

    controls = pd.DataFrame()
    if not args.no_controls:
        print("[stage] matched non-zone downside controls", flush=True)
        controls = build_matched_controls(
            decision_zones,
            lifecycle,
            bars,
            zone_cfg,
            research_start=research_start,
            research_end_exclusive=research_end_exclusive,
            feature_frame=market_features,
        )
        print(f"[controls] matched={len(controls):,}/{len(decision_zones):,}", flush=True)

    all_features = pd.concat([decision_zones, controls], ignore_index=True, sort=False) if not controls.empty else decision_zones.copy()
    print("[stage] next-open high/low MFE-MAE, multi-day survival and structural exits", flush=True)
    outcomes = attach_structural_path_outcomes(all_features, bars, zone_cfg)
    features, labels = _feature_label_split(outcomes)
    audit = causal_audit(features, labels)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"Causal audit failed with {violations} violations:\n{audit.to_string(index=False)}")

    print("[stage] descriptive reports and model-ready feature/label split", flush=True)
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
        "atlas_config": asdict(atlas_cfg),
        "zone_study_config": asdict(zone_cfg),
        "notes": [
            "Mechanism research only; not a final strategy backtest.",
            "Five-minute reclaim and all MFE/MAE fields are labels, never event admission filters.",
            "Zone members must be causally available by the closed sweep bar.",
            "Outcome entry reference is the next 1m open.",
            "Controls exclude all bars around any raw active Swing Low first sweep.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(
        _data_quality(
            bars, levels, lifecycle, primary_zones, decision_zones, controls,
            raw_sweeps_all=raw_sweeps_all, raw_sweeps_research=raw_sweeps,
        ),
        out_dir / "01_data_quality.csv",
    )
    _write_csv(sensitivity_table, out_dir / "02_zone_construction_sensitivity.csv")
    _write_csv(primary_zones, out_dir / "03_same_bar_zone_event_table.csv")
    _write_csv(decision_zones, out_dir / "04_online_first_zone_feature_table.csv")
    if not controls.empty:
        _write_csv(controls, out_dir / "05_matched_control_feature_table.csv")
    _write_csv(path_horizon_summary(outcomes, zone_cfg), out_dir / "06_path_horizon_summary.csv")
    _write_csv(structural_exit_summary(outcomes, zone_cfg), out_dir / "07_structural_exit_summary.csv")
    _write_csv(control_comparison(outcomes, zone_cfg), out_dir / "08_control_comparison.csv")
    _write_csv(control_match_balance(all_features), out_dir / "09_control_match_balance.csv")
    bin_summary, bin_edges = feature_bin_summary(outcomes, zone_cfg)
    _write_csv(bin_summary, out_dir / "10_zone_attribute_path_bins.csv")
    _write_csv(bin_edges, out_dir / "11_reference_bin_edges.csv")
    _write_csv(features, out_dir / "12_model_feature_table.csv")
    _write_csv(labels, out_dir / "13_model_label_table.csv")
    _write_csv(audit, out_dir / "14_causal_audit.csv")
    sample_size = max(0, int(args.event_sample_size))
    if sample_size:
        sample = outcomes if len(outcomes) <= sample_size else outcomes.sample(sample_size, random_state=42)
        _write_csv(sample, out_dir / "15_event_sample.csv")
    if args.write_full_lifecycle:
        _write_csv(lifecycle, out_dir / "16_full_level_lifecycle.csv")
    brief = _brief(atlas_cfg, zone_cfg, raw_sweeps, primary_zones, decision_zones, controls, audit)
    (out_dir / "17_research_brief.md").write_text(brief, encoding="utf-8")
    review = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={review.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
