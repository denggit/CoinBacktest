#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R09 structured Swing-Low stop-pool hypothesis research.

R09 returns to the foundational question: which causally identifiable Swing Low
structures appear to contain real stop liquidity when first swept, and which of
those structures subsequently reverse efficiently?

The study predeclares H1-H8 and never combines them into a mined super-filter.
Structure membership is known before the sweep.  Sweep-time order-flow release
and post-sweep paths are labels in separate tables.  Next-open execution is used
for all TP/SL outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
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
from src.research_common.swing_liquidity_atlas import build_level_lifecycle, build_swing_low_universe, normalize_primary_bars  # noqa: E402
from src.research_common.swing_liquidity_atlas.config import AtlasConfig  # noqa: E402
from src.research_common.swing_liquidity_zone_study import attach_structural_path_outcomes  # noqa: E402
from src.research_common.swing_liquidity_zone_study.config import ZoneStudyConfig  # noqa: E402
from src.research_common.structured_stop_pool import (  # noqa: E402
    StructuredStopPoolConfig,
    attach_first_touch_outcomes,
    attach_stop_release_labels,
    audit_r02_bar_alignment,
    build_r09_universe,
    calibrate_release_score,
    causal_audit,
    data_quality,
    family_overlap,
    family_path_summary,
    family_release_summary,
    family_scorecard,
    family_strategy_summary,
    family_timeframe_summary,
    hypothesis_definitions,
    hypothesis_universe_summary,
    load_or_build_r02,
    matched_release_comparison,
    period_stability,
    research_brief,
)

SCRIPT_NAME = "09_structured_swing_stop_pool_hypothesis_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_STRUCTURED_SWING_STOP_POOL_HYPOTHESES_R09"
EDGE_ID = "RESEARCH_ONLY_STRUCTURED_SWING_STOP_POOL_REVERSAL"
TITLE = "ETH Structured Swing-Low Stop-Pool Hypothesis Study R09"
DEFAULT_R02_DIR = "data/reports/research/liquidity/unconsumed_swing_liquidity_atlas_r02"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/structured_swing_stop_pool_hypotheses_r09"


def _end_exclusive(value: str) -> pd.Timestamp:
    text = str(value).strip()
    ts = pd.Timestamp(text)
    return ts + pd.Timedelta(days=1) if len(text) <= 10 else ts + pd.Timedelta(microseconds=1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--r02-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--rebuild-r02-if-missing", action="store_true")
    p.add_argument("--data-source", choices=["trade_bar", "ohlcv_local"], default="trade_bar")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--max-events", type=int, default=0, help="Deterministic smoke cap after online zone de-dup; 0 uses all.")
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.timeframe) != "1m":
        raise ValueError("R09 requires --timeframe 1m")
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
            if not isinstance(bars.index, pd.DatetimeIndex):
                bars.index = pd.to_datetime(bars.index, errors="coerce")
            bars = bars.loc[(bars.index >= pd.Timestamp(args.warmup_start_date)) & (bars.index <= pd.Timestamp(args.end_date))]
    keep = [
        "open", "high", "low", "close", "volume", "notional", "buy_notional", "sell_notional",
        "delta_notional", "trades_count", "large_buy_notional", "large_sell_notional",
        "large_buy_trades_count", "large_sell_trades_count", "large_delta_notional",
        "max_trade_notional", "max_trade_size",
    ]
    bars = normalize_primary_bars(bars.loc[:, [name for name in keep if name in bars.columns]].copy())
    required_flow = {"notional", "sell_notional", "delta_notional", "trades_count"}
    missing = sorted(required_flow.difference(bars.columns))
    if missing:
        raise RuntimeError(
            f"R09 real-liquidity release study requires trade-bar order-flow fields; missing={missing}. "
            "Use --data-source trade_bar with the complete OKX trade-bar cache."
        )
    for optional in ("large_sell_notional", "large_sell_trades_count", "max_trade_notional", "buy_notional"):
        if optional not in bars.columns:
            bars[optional] = np.nan
    print(f"[load] rows={len(bars):,} range={bars.index.min()}->{bars.index.max()} cols={len(bars.columns)}", flush=True)
    return bars


def _zone_config(cfg: StructuredStopPoolConfig) -> ZoneStudyConfig:
    return ZoneStudyConfig(
        zone_merge_tolerance_bp=float(cfg.zone_merge_tolerance_bp),
        zone_merge_sensitivity_bp=(float(cfg.zone_merge_tolerance_bp),),
        impulse_gap_bars=int(cfg.impulse_gap_bars),
        impulse_price_tolerance_bp=float(cfg.impulse_price_tolerance_bp),
        path_horizons=tuple(int(v) for v in cfg.path_horizons),
        tp_returns=tuple(float(v) for v in cfg.tp_returns),
        structural_break_epsilon_bp=float(cfg.structural_break_epsilon_bp),
        control_exclusion_bars=int(cfg.control_exclusion_bars),
        control_min_downside_atr=float(cfg.control_min_downside_atr),
        control_max_per_zone=1,
    ).validate()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _split_feature_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_prefixes = (
        "release_", "stop_release_score", "high_stop_release_label",
        "entry_reference_", "first_lower_low_", "bars_to_lower_low", "first_zone_", "bars_to_zone_",
        "close_return_", "mfe_", "mae_", "structural_low_survival_", "zone_floor_reclaim_by_",
        "zone_ceiling_reclaim_by_", "first_tp_", "bars_to_tp_", "tp_", "r09_entry_",
        "tp15_sl15_", "tp25_sl15_", "tp50_sl25_",
    )
    ids = [name for name in ("zone_event_id", "event_kind", "matched_zone_event_id", "period") if name in frame.columns]
    labels = ids + [name for name in frame.columns if name.startswith(label_prefixes)]
    labels = list(dict.fromkeys(labels))
    feature_names = [name for name in frame.columns if name not in set(labels) or name in ids]
    features = frame.loc[:, feature_names].copy()
    label_frame = frame.loc[:, labels].copy()
    forbidden = [name for name in features.columns if name.startswith(("release_", "mfe_", "mae_", "tp_"))]
    if forbidden:
        raise RuntimeError(f"future/release labels leaked into feature table: {forbidden[:10]}")
    return features, label_frame


def _synthetic_bars() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=14_000, freq="1min")
    x = np.arange(len(index), dtype=float)
    trend = 2_000.0 - 0.003 * x
    wave = 28.0 * np.sin(x / 390.0) + 7.0 * np.sin(x / 43.0)
    close = trend + wave
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    notional = 1_000_000.0 + 300_000.0 * (1.0 + np.sin(x / 29.0))
    sell = notional * (0.50 + 0.10 * np.sin(x / 17.0))
    buy = notional - sell
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": notional / close, "notional": notional,
            "buy_notional": buy, "sell_notional": sell,
            "delta_notional": buy - sell, "trades_count": 100.0 + (x % 30),
            "large_sell_notional": sell * 0.05,
            "large_sell_trades_count": 2.0 + (x % 3),
            "max_trade_notional": 50_000.0 + (x % 1_000),
        },
        index=index,
    )


def run_self_test() -> None:
    bars = _synthetic_bars()
    cfg = StructuredStopPoolConfig(timeframes=(("15m", 15), ("30m", 30)), path_horizons=(15, 60, 180)).validate()
    atlas = AtlasConfig(
        timeframes=cfg.timeframes, confirmation_orders=(1, 2, 3, 5), approach_distance_bp=200.0,
        touch_distance_bp=5.0, sweep_epsilon_bp=0.01, acceptance_depth_bp=50.0,
        acceptance_consecutive_closes=3, resolution_horizon_bars=180,
        forward_horizons=(5, 15, 30, 60, 180), confluence_tolerances_bp=(5.0, 10.0, 25.0, 50.0),
    ).validate()
    levels = build_swing_low_universe(bars, atlas)
    lifecycle = build_level_lifecycle(bars, levels, atlas, show_progress=False)
    if levels.empty:
        raise RuntimeError("R09 self-test generated no levels")
    level_features, thresholds, zones, controls, _ = build_r09_universe(
        levels, lifecycle, bars, cfg,
        research_start=pd.Timestamp("2023-01-02"), research_end_exclusive=pd.Timestamp("2023-01-11"),
        max_events=100, include_controls=False,
    )
    if zones.empty:
        raise RuntimeError("R09 self-test generated no sweep zones")
    combined = attach_stop_release_labels(zones, bars, cfg)
    combined, calibration = calibrate_release_score(combined, cfg)
    combined = attach_structural_path_outcomes(combined, bars, _zone_config(cfg))
    combined = attach_first_touch_outcomes(combined, bars, cfg)
    audit = causal_audit(level_features, zones, combined)
    failures = audit.loc[audit["status"].eq("FAIL")]
    if not failures.empty or thresholds.empty or calibration.empty:
        raise RuntimeError(f"R09 self-test failed:\n{failures.to_string(index=False)}")
    print(f"[self-test] passed levels={len(levels):,} zones={len(zones):,} outcome_rows={len(combined):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    cfg = replace(StructuredStopPoolConfig(), report_sample_rows=int(args.sample_rows)).validate()
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    if research_end_exclusive <= research_start:
        raise ValueError("end-date must be after start-date")
    started = time.perf_counter()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start}->{research_end_exclusive}", flush=True)
    print("[design] H1-H8 separate; pre-sweep structure -> stop release -> post-sweep reversal; no combination mining", flush=True)

    bars = _load_bars(args)
    r02_dir = PROJECT_ROOT / args.r02_dir
    print(f"[stage] load/rebuild R02 causal Swing Low atlas from {r02_dir}", flush=True)
    levels, lifecycle, r02_source = load_or_build_r02(
        r02_dir, bars, cfg,
        rebuild_if_missing=bool(args.rebuild_r02_if_missing),
        show_progress=not bool(args.no_progress),
    )
    alignment = audit_r02_bar_alignment(lifecycle, bars)
    if alignment["status"].eq("FAIL").any():
        raise RuntimeError(f"R02 report/bar alignment failed:\n{alignment.to_string(index=False)}")
    print(f"[r02] source={r02_source} levels={len(levels):,} swept={pd.to_numeric(lifecycle['sweep_pos'], errors='coerce').ge(0).sum():,}", flush=True)

    print("[stage] classify H1-H8 causally on each source timeframe", flush=True)
    level_features, structure_thresholds, zones, controls, all_zones = build_r09_universe(
        levels, lifecycle, bars, cfg,
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        max_events=int(args.max_events),
        include_controls=not bool(args.skip_controls),
    )
    if zones.empty:
        raise RuntimeError("R09 produced no online first sweep zones")
    print(f"[universe] levels={len(level_features):,} all_same_bar_zones={len(all_zones):,} online_first={len(zones):,} controls={len(controls):,}", flush=True)

    print("[stage] measure stop-like order-flow release at first sweep", flush=True)
    combined = pd.concat([zones, controls], ignore_index=True, sort=False) if not controls.empty else zones.copy()
    combined = attach_stop_release_labels(combined, bars, cfg)
    combined, release_calibration = calibrate_release_score(combined, cfg)
    print(f"[release] rows={len(combined):,} high_release={int(combined['high_stop_release_label'].sum()):,}", flush=True)

    print("[stage] next-open path, structural survival and conservative TP/SL outcomes", flush=True)
    combined = attach_structural_path_outcomes(combined, bars, _zone_config(cfg))
    combined = attach_first_touch_outcomes(combined, bars, cfg)
    feature_table, label_table = _split_feature_labels(combined)

    print("[stage] family reports, matched controls and frozen-period scorecard", flush=True)
    universe_summary = hypothesis_universe_summary(level_features, lifecycle, zones)
    release_summary = family_release_summary(combined)
    matched_summary = matched_release_comparison(combined)
    path_summary = family_path_summary(combined)
    strategy_summary = family_strategy_summary(combined)
    timeframe_summary = family_timeframe_summary(combined)
    stability = period_stability(release_summary, path_summary, strategy_summary)
    overlap = family_overlap(combined)
    scorecard = family_scorecard(combined, release_summary, strategy_summary, cfg)
    audit = pd.concat([alignment, causal_audit(level_features, zones, combined)], ignore_index=True)
    quality = data_quality(bars, levels, lifecycle, zones, controls, combined, structure_thresholds, release_calibration)

    reports = {
        "01_data_quality.csv": quality,
        "02_hypothesis_definitions.csv": hypothesis_definitions(),
        "03_structure_thresholds_frozen_early.csv": structure_thresholds,
        "04_hypothesis_universe_summary.csv": universe_summary,
        "05_family_stop_release_summary.csv": release_summary,
        "06_matched_control_release_comparison.csv": matched_summary,
        "07_family_reversal_path_summary.csv": path_summary,
        "08_family_first_touch_strategy_summary.csv": strategy_summary,
        "09_family_timeframe_summary.csv": timeframe_summary,
        "10_period_stability.csv": stability,
        "11_hypothesis_overlap.csv": overlap,
        "12_hypothesis_scorecard.csv": scorecard,
        "13_release_score_calibration.csv": release_calibration,
        "14_causal_audit.csv": audit,
    }
    for name, frame in reports.items():
        _write(frame, out_dir / name)

    sample_n = min(int(cfg.report_sample_rows), len(combined))
    combined.sort_values(["period", "event_pos", "event_kind"], kind="mergesort").head(sample_n).to_csv(out_dir / "15_event_sample.csv", index=False)
    feature_table.to_csv(out_dir / "16_zone_feature_table.csv.gz", index=False, compression="gzip")
    label_table.to_csv(out_dir / "17_zone_label_table.csv.gz", index=False, compression="gzip")
    level_features.to_csv(out_dir / "18_level_structure_feature_table.csv.gz", index=False, compression="gzip")
    (out_dir / "19_research_brief.md").write_text(research_brief(scorecard, universe_summary, matched_summary), encoding="utf-8")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "research_start": str(research_start),
        "research_end_exclusive": str(research_end_exclusive),
        "r02_source": r02_source,
        "r02_dir": str(r02_dir),
        "data_source": args.data_source,
        "config": asdict(cfg),
        "level_count": len(level_features),
        "zone_event_count": len(zones),
        "control_count": len(controls),
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "H1-H8 are predeclared and analyzed separately; combinations are not mined.",
            "Structure membership is available before the sweep; stop release and future reversal are labels.",
            "Controls are matched ordinary downside bars outside all raw Swing Low first-sweep neighborhoods.",
            "All trade outcomes enter at the next 1m open; same-bar TP/SL ambiguity is resolved conservatively as SL.",
            "Default round-trip fee assumption is 0.11%, plus 1bp slippage per side in first-touch outcome tables.",
            "Broad R09 uses complete 1m trade-bar order flow. Sparse raw-1s and Books validation are intentionally deferred until a family passes this screen.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    failures = quality.loc[quality["status"].eq("FAIL"), "check"].astype(str).tolist() + audit.loc[audit["status"].eq("FAIL"), "check"].astype(str).tolist()
    if failures:
        raise RuntimeError(f"R09 quality/causal gate failed: {failures}")
    if not args.skip_review_pack:
        review = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
        print(f"[done] review_pack={review.zip_path}", flush=True)
    print(f"[done] report={out_dir} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    print(f"[decision] {scorecard[['family_id','decision']].to_dict(orient='records')}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
