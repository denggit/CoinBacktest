#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R10 causal structured Higher-Low pullback-entry study.

The study does not wait for a liquidity Sweep. Once a Higher Low is causally
confirmed, it rests a buy limit at that level and places the structural stop
below the earlier low that invalidates the pullback thesis.
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
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars  # noqa: E402
from src.research_common.structured_pullback_entry import (  # noqa: E402
    StructuredPullbackConfig,
    attach_limit_fills,
    attach_trade_outcomes,
    build_pullback_candidate_universe,
    candidate_funnel_summary,
    causal_audit,
    data_quality,
    family_geometry_summary,
    family_outcome_summary,
    family_overlap,
    family_scorecard,
    family_timeframe_summary,
    fill_age_summary,
    hypothesis_definitions,
    load_or_build_r09_level_features,
    period_stability,
    research_brief,
)
from src.research_common.structured_stop_pool import (  # noqa: E402
    StructuredStopPoolConfig,
    audit_r02_bar_alignment,
    load_or_build_r02,
)

SCRIPT_NAME = "10_structured_pullback_entry_study"
SCRIPT_VERSION = "1.0.2"
EXPERIMENT_ID = "ETH_STRUCTURED_PULLBACK_ENTRY_R10"
EDGE_ID = "RESEARCH_ONLY_STRUCTURED_HIGHER_LOW_PULLBACK"
TITLE = "ETH Structured Higher-Low Pullback Entry Study R10"
DEFAULT_R02_DIR = "data/reports/research/liquidity/unconsumed_swing_liquidity_atlas_r02"
DEFAULT_R09_DIR = "data/reports/research/liquidity/structured_swing_stop_pool_hypotheses_r09"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/10_structured_pullback_entry_r10"


def _end_exclusive(value: str) -> pd.Timestamp:
    text = str(value).strip()
    timestamp = pd.Timestamp(text)
    return timestamp + pd.Timedelta(days=1) if len(text) <= 10 else timestamp + pd.Timedelta(microseconds=1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--r02-dir", default=DEFAULT_R02_DIR)
    parser.add_argument("--r09-dir", default=DEFAULT_R09_DIR)
    parser.add_argument("--rebuild-r02-if-missing", action="store_true")
    parser.add_argument("--rebuild-r09-if-missing", action="store_true")
    parser.add_argument("--data-source", choices=["trade_bar", "ohlcv_local"], default="trade_bar")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-build-missing", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0, help="Deterministic smoke cap on unique Higher-Low candidates; 0 uses all.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.timeframe) != "1m":
        raise ValueError("R10 requires --timeframe 1m")
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
            bars = bars.loc[
                (bars.index >= pd.Timestamp(args.warmup_start_date))
                & (bars.index <= pd.Timestamp(args.end_date))
            ]
    keep = [
        "open", "high", "low", "close", "volume", "notional", "buy_notional",
        "sell_notional", "delta_notional", "trades_count",
    ]
    bars = normalize_primary_bars(bars.loc[:, [name for name in keep if name in bars.columns]].copy())
    print(f"[load] rows={len(bars):,} range={bars.index.min()}->{bars.index.max()} cols={len(bars.columns)}", flush=True)
    return bars


def _resolve_r09_dir(value: str) -> Path:
    requested = PROJECT_ROOT / value
    candidates = [
        requested,
        PROJECT_ROOT / "data/reports/research/liquidity/09_structured_swing_stop_pool_hypotheses_r09",
        PROJECT_ROOT / DEFAULT_R09_DIR,
    ]
    for candidate in candidates:
        if (candidate / "18_level_structure_feature_table.csv.gz").exists():
            return candidate
    return requested


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _synthetic_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=600, freq="1min")
    close = np.full(len(index), 106.0)
    open_ = np.r_[106.0, close[:-1]]
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    # Causal candidate available at bar 100; later retest and then H0 target.
    low[112] = 99.5
    high[112] = 101.0
    close[112] = 100.5
    open_[112] = 103.0
    high[125] = 110.5
    close[125] = 110.0
    bars = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": 1_000.0, "notional": 1_000_000.0,
            "buy_notional": 500_000.0, "sell_notional": 500_000.0,
            "delta_notional": 0.0, "trades_count": 100.0,
        },
        index=index,
    )
    common = {
        "pivot_time": index[80],
        "previous2_swing_low_price": 105.0,
        "reference_swing_high_price": 112.0,
        "prior_leg_high_price": 108.0,
        "predecessor_decline_atr": 3.0,
        "rebound_before_current_atr": 2.0,
        "higher_low_gap_atr": 1.0,
        "pullback_fraction_of_rebound": 0.5,
        "prior_two_low_gap_atr": 0.1,
        "confirmation_reaction_high_bp": 100.0,
        "left_high_range_20_bp": 300.0,
        "consecutive_higher_low_count": 1,
        "bos_before_current_low": True,
        "higher_high_before_current_low": True,
        "failed_breakdown_previous_low": True,
        "is_higher_low": True,
        "hyp_h1_first_higher_low_after_decline": True,
        "hyp_h2_bos_pullback_higher_low": True,
        "hyp_h3_layered_base_higher_low": False,
        "hyp_h4_strong_displacement_origin": True,
        "hyp_h5_base_breakout_pullback": False,
        "hyp_h6_multitimeframe_confluence": False,
        "hyp_h7_trend_continuation_higher_low": False,
        "hyp_h8_failed_breakdown_then_higher_low": True,
    }
    levels = pd.DataFrame(
        [
            {
                **common,
                "level_id": 1,
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "level_price": 100.0,
                "structure_available_time": index[100],
                "previous_swing_low_price": 95.0,
                "current_leg_high_price": 110.0,
            },
            {
                **{**common, "is_higher_low": False},
                "level_id": 2,
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "pivot_time": index[250],
                "level_price": 102.0,
                "structure_available_time": index[300],
                "previous_swing_low_price": 100.0,
                "current_leg_high_price": 112.0,
            },
            {
                **{**common, "is_higher_low": False},
                "level_id": 3,
                "source_timeframe": "30m",
                "source_timeframe_min": 30,
                "pivot_time": index[40],
                "level_price": 100.05,
                "structure_available_time": index[60],
                "previous_swing_low_price": 94.0,
                "current_leg_high_price": 111.0,
            },
        ]
    )
    lifecycle = pd.DataFrame(
        [
            {"level_id": 1, "level_price": 100.0, "source_timeframe": "15m", "active_pos": 100, "sweep_pos": 112},
            {"level_id": 2, "level_price": 102.0, "source_timeframe": "15m", "active_pos": 300, "sweep_pos": -1},
            {"level_id": 3, "level_price": 100.05, "source_timeframe": "30m", "active_pos": 60, "sweep_pos": -1},
        ]
    )
    return bars, levels, lifecycle


def run_self_test() -> None:
    bars, level_features, lifecycle = _synthetic_fixture()
    cfg = StructuredPullbackConfig(timeframes=(("15m", 15), ("30m", 30))).validate()
    unique, family = build_pullback_candidate_universe(
        level_features,
        lifecycle,
        bars,
        cfg,
        research_start=bars.index[0],
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    replay = attach_limit_fills(
        family,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    replay = attach_trade_outcomes(
        replay,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    audit = causal_audit(unique, family, replay)
    if audit["status"].eq("FAIL").any():
        raise RuntimeError(f"R10 self-test causal failure:\n{audit.to_string(index=False)}")
    p1 = replay.loc[replay["family_id"].eq("P1")]
    if p1.empty or not p1["fill_status"].eq("FILLED").all() or not p1["h0_outcome"].eq("TP").all():
        raise RuntimeError("R10 self-test did not fill/target the expected P1 pullback")
    p6 = replay.loc[replay["family_id"].eq("P6")]
    if p6.empty:
        raise RuntimeError("R10 self-test did not causally identify formation-timeframe confluence")
    print(f"[self-test] passed unique={len(unique):,} family_rows={len(family):,} replay_rows={len(replay):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir

    cfg = replace(StructuredPullbackConfig(), report_sample_rows=int(args.sample_rows)).validate()
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    if research_end_exclusive <= research_start:
        raise ValueError("end-date must be after start-date")

    started = time.perf_counter()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start}->{research_end_exclusive}", flush=True)
    print("[design] confirmed Higher-Low limit retest -> prior structural-low stop -> H0/1R/2R/3R; families separate; no combination mining", flush=True)

    bars = _load_bars(args)
    r09_cfg = StructuredStopPoolConfig().validate()
    r02_dir = PROJECT_ROOT / args.r02_dir
    print(f"[stage] load/rebuild R02 causal levels/lifecycle from {r02_dir}", flush=True)
    levels, lifecycle, r02_source = load_or_build_r02(
        r02_dir,
        bars,
        r09_cfg,
        rebuild_if_missing=bool(args.rebuild_r02_if_missing),
        show_progress=not bool(args.no_progress),
    )
    alignment = audit_r02_bar_alignment(lifecycle, bars)
    if alignment["status"].eq("FAIL").any():
        raise RuntimeError(f"R02 report/bar alignment failed:\n{alignment.to_string(index=False)}")

    r09_dir = _resolve_r09_dir(args.r09_dir)
    print(f"[stage] load/rebuild causal R09 H1-H8 level features from {r09_dir}", flush=True)
    level_features, rebuilt_thresholds, r09_source = load_or_build_r09_level_features(
        r09_dir,
        levels,
        bars,
        rebuild_if_missing=bool(args.rebuild_r09_if_missing),
    )
    print(f"[structure] r02={r02_source} r09={r09_source} levels={len(level_features):,}", flush=True)

    print("[stage] build P1-P8 pullback families and formation-time multi-timeframe confluence", flush=True)
    unique_candidates, family_candidates = build_pullback_candidate_universe(
        level_features,
        lifecycle,
        bars,
        cfg,
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        max_candidates=int(args.max_candidates),
    )
    if family_candidates.empty:
        raise RuntimeError("R10 produced no valid Higher-Low family candidates")
    print(f"[universe] unique={len(unique_candidates):,} family_rows={len(family_candidates):,}", flush=True)

    print("[stage] causal resting-limit fills; cancel old order at next same-timeframe Swing Low", flush=True)
    replay = attach_limit_fills(
        family_candidates,
        bars,
        cfg,
        research_end_exclusive=research_end_exclusive,
        show_progress=not bool(args.no_progress),
    )
    print(
        f"[fills] filled={int(replay['fill_status'].eq('FILLED').sum()):,}/{len(replay):,} "
        f"h0_before_fill={int(replay['h0_traded_before_fill_flag'].sum()):,}",
        flush=True,
    )

    print("[stage] structural-stop replay for H0/1R/2R/3R with conservative same-bar ordering", flush=True)
    replay = attach_trade_outcomes(
        replay,
        bars,
        cfg,
        research_end_exclusive=research_end_exclusive,
        show_progress=not bool(args.no_progress),
    )

    print("[stage] cost, robustness, timeframe and period reports", flush=True)
    funnel = candidate_funnel_summary(replay)
    geometry = family_geometry_summary(replay)
    outcome_summary = family_outcome_summary(replay)
    timeframe_summary = family_timeframe_summary(replay)
    stability = period_stability(replay)
    ages = fill_age_summary(replay)
    overlap = family_overlap(replay)
    scorecard = family_scorecard(replay, outcome_summary, stability, cfg)
    audit = pd.concat([alignment, causal_audit(unique_candidates, family_candidates, replay)], ignore_index=True)
    quality = data_quality(
        bars,
        level_features,
        unique_candidates,
        family_candidates,
        replay,
        r09_source=r09_source,
    )

    reports = {
        "01_data_quality.csv": quality,
        "02_hypothesis_definitions.csv": hypothesis_definitions(),
        "03_candidate_funnel_summary.csv": funnel,
        "04_family_geometry_summary.csv": geometry,
        "05_family_target_outcome_summary.csv": outcome_summary,
        "06_family_timeframe_summary.csv": timeframe_summary,
        "07_period_stability.csv": stability,
        "08_fill_age_summary.csv": ages,
        "09_family_overlap.csv": overlap,
        "10_family_target_scorecard.csv": scorecard,
        "11_causal_audit.csv": audit,
    }
    if not rebuilt_thresholds.empty:
        reports["12_rebuilt_r09_thresholds.csv"] = rebuilt_thresholds
    for name, frame in reports.items():
        _write(frame, out_dir / name)

    sample_n = min(int(cfg.report_sample_rows), len(replay))
    replay.sort_values(
        ["structure_available_time", "source_timeframe_min", "level_id", "family_id"],
        kind="mergesort",
    ).head(sample_n).to_csv(out_dir / "13_event_sample.csv", index=False, encoding="utf-8-sig")

    # Candidate feature table intentionally excludes the future cancellation timestamp.
    candidate_feature_table = unique_candidates.drop(
        columns=["next_same_timeframe_structure_available_time"], errors="ignore"
    )
    candidate_feature_table.to_csv(out_dir / "14_candidate_feature_table.csv.gz", index=False, compression="gzip")
    family_candidates.to_csv(out_dir / "15_family_candidate_execution_plan.csv.gz", index=False, compression="gzip")
    replay.to_csv(out_dir / "16_trade_outcome_table.csv.gz", index=False, compression="gzip")
    (out_dir / "17_research_brief.md").write_text(research_brief(scorecard, funnel), encoding="utf-8")

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
        "r09_source": r09_source,
        "r02_dir": str(r02_dir),
        "r09_dir": str(r09_dir),
        "data_source": args.data_source,
        "config": asdict(cfg),
        "unique_candidate_count": len(unique_candidates),
        "family_candidate_count": len(family_candidates),
        "filled_family_count": int(replay["fill_status"].eq("FILLED").sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Report directory and artifacts use numeric prefixes for chronological sorting.",
            "R10 tests confirmed Higher-Low pullback entries; it does not require or predict a later liquidity Sweep.",
            "Orders activate only at structure_available_time and cancel at the next same-timeframe Swing Low availability.",
            "P3/P5 use the lower two-low base as the structural anchor; other families use the previous Swing Low.",
            "Base fee is 0.11% round trip. Realistic columns add 2bp total slippage; stressed columns double that total realistic cost.",
            "Families are evaluated separately. No family combinations or threshold grids are mined.",
            "This is event-level research with overlapping opportunities, not a final capital-constrained portfolio backtest.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    failures = (
        quality.loc[quality["status"].eq("FAIL"), "check"].astype(str).tolist()
        + audit.loc[audit["status"].eq("FAIL"), "check"].astype(str).tolist()
    )
    if failures:
        raise RuntimeError(f"R10 quality/causal gate failed: {failures}")

    if not args.skip_review_pack:
        review = finalize_research_report(
            out_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )
        print(f"[done] review_pack={review.zip_path}", flush=True)
    print(f"[done] report={out_dir} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    promoted = scorecard.loc[scorecard["decision"].ne("rejected"), ["family_id", "target", "decision"]]
    print(f"[decision] {promoted.to_dict(orient='records')}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
