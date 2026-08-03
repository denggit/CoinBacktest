#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R11 causal liquidity-magnet and stop-risk frontier study.

R09 confirmed that structured Swing-Low zones release unusually large downside
order flow when swept, but unconditional post-sweep reversal failed.  R11 moves
one step earlier: can price be traded *toward* an active lower liquidity pool,
and can any natural causal stop make that route viable after realistic costs?

The study is intentionally sparse:
- four predeclared distance checkpoints;
- three natural stop models;
- no parameter grid and no combination mining;
- next-open entry after a closed 1m signal bar;
- conservative same-bar target/stop ordering.
"""
from __future__ import annotations

import argparse
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
from src.research_common.liquidity_magnet import (  # noqa: E402
    LiquidityMagnetConfig,
    attach_risk_frontier_outcomes,
    build_liquidity_magnet_universe,
    candidate_funnel,
    causal_audit,
    data_quality,
    design_table,
    directional_magnet_summary,
    load_r02_and_r09_levels,
    manifest_json,
    period_stability,
    research_brief,
    risk_frontier_summary,
    scorecard,
    structure_family_summary,
    timeframe_confluence_summary,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars  # noqa: E402

SCRIPT_NAME = "11_liquidity_magnet_risk_frontier_study"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_LIQUIDITY_MAGNET_RISK_FRONTIER_R11"
EDGE_ID = "RESEARCH_ONLY_LIQUIDITY_MAGNET_ROUTE"
TITLE = "ETH Liquidity Magnet and Risk Frontier Study R11"
DEFAULT_R02_DIR = "data/reports/research/liquidity/unconsumed_swing_liquidity_atlas_r02"
DEFAULT_R09_DIR = "data/reports/research/liquidity/structured_swing_stop_pool_hypotheses_r09"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/11_liquidity_magnet_risk_frontier_r11"


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
    p.add_argument("--r09-dir", default=DEFAULT_R09_DIR)
    p.add_argument("--rebuild-r02-if-missing", action="store_true")
    p.add_argument("--rebuild-r09-if-missing", action="store_true")
    p.add_argument("--data-source", choices=["trade_bar", "ohlcv_local"], default="trade_bar")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--max-candidates", type=int, default=0, help="Deterministic cap after pool de-dup; 0 uses all.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.timeframe) != "1m":
        raise ValueError("R11 requires --timeframe 1m")
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
    keep = ["open", "high", "low", "close", "volume", "notional", "buy_notional", "sell_notional", "delta_notional", "trades_count"]
    bars = normalize_primary_bars(bars.loc[:, [name for name in keep if name in bars.columns]].copy())
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise RuntimeError(f"R11 requires OHLC fields; missing={missing}")
    print(f"[load] rows={len(bars):,} range={bars.index.min()}->{bars.index.max()} cols={len(bars.columns)}", flush=True)
    return bars


def _write(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression=compression)


def _split_candidate_feature_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [name for name in ("pool_event_id", "distance_band_bp", "event_available_time", "entry_time") if name in frame.columns]
    future = [name for name in frame.columns if name.startswith("future_")]
    labels = list(dict.fromkeys([*ids, *future]))
    features = frame.drop(columns=future, errors="ignore").copy()
    label_frame = frame.loc[:, labels].copy()
    leaked = [name for name in features.columns if name.startswith("future_")]
    if leaked:
        raise RuntimeError(f"future labels leaked into candidate feature table: {leaked}")
    return features, label_frame


def _synthetic_bars() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=2_000, freq="1min")
    x = np.arange(len(index), dtype=float)
    close = 2_000.0 - 0.03 * x + 8.0 * np.sin(x / 37.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def run_self_test() -> None:
    bars = _synthetic_bars()
    lifecycle = pd.DataFrame(
        {
            "level_id": [1, 2, 3],
            "source_timeframe": ["15m", "1H", "30m"],
            "source_timeframe_min": [15, 60, 30],
            "level_price": [1980.0, 1979.5, 1960.0],
            "pivot_time": pd.to_datetime(["2023-01-01 01:00", "2023-01-01 01:30", "2023-01-01 02:00"]),
            "initial_available_time": pd.to_datetime(["2023-01-01 02:00", "2023-01-01 02:30", "2023-01-01 03:00"]),
            "active_pos": [120, 150, 180],
            "sweep_pos": [800, 800, 1_500],
            "sweep_available_time": pd.to_datetime(["2023-01-01 13:21", "2023-01-01 13:21", "2023-01-02 01:01"]),
            "confirmed_order_at_sweep": [2, 3, 2],
            "confirmation_reaction_close_bp": [80.0, 100.0, 60.0],
            "confirmation_reaction_high_bp": [120.0, 180.0, 90.0],
            "left_high_range_20_bp": [220.0, 300.0, 150.0],
            "left_low_gap_20_bp": [40.0, 50.0, 30.0],
            "pivot_notional_vs_past20": [1.2, 1.5, 0.9],
            "pivot_trades_count_vs_past20": [1.1, 1.4, 1.0],
        }
    )
    from src.research_common.structured_stop_pool import FAMILY_COLUMNS
    features = lifecycle[["level_id"]].copy()
    for i, family in enumerate(FAMILY_COLUMNS):
        features[family] = [i == 0, i in (0, 5), False]
    cfg = replace(
        LiquidityMagnetConfig(),
        distance_bands_bp=(100.0, 50.0),
        horizon_minutes=120,
        minimum_spec_events=1,
        minimum_period_events=1,
    ).validate()
    candidates = build_liquidity_magnet_universe(
        lifecycle,
        features,
        bars,
        cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-03"),
        show_progress=False,
    )
    if candidates.empty:
        raise RuntimeError("R11 self-test produced no candidates")
    outcomes = attach_risk_frontier_outcomes(candidates, bars, cfg, show_progress=False)
    audit = causal_audit(candidates, outcomes)
    failures = audit.loc[audit["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(f"R11 self-test causal failure:\n{failures.to_string(index=False)}")
    if len(outcomes) != len(candidates) * 3:
        raise RuntimeError("R11 self-test outcome multiplier mismatch")
    print(f"[self-test] passed candidates={len(candidates):,} outcomes={len(outcomes):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    started = time.perf_counter()
    cfg = replace(LiquidityMagnetConfig(), report_sample_rows=int(args.sample_rows)).validate()
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    if research_end_exclusive <= research_start:
        raise ValueError("end date must be after start date")

    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start}->{research_end_exclusive}", flush=True)
    print(
        "[design] active lower Swing-Low pool -> first distance-band approach -> next-open short -> "
        "front-run pool target vs equal-distance/15m-high/60m-high stops; no combination mining",
        flush=True,
    )
    bars = _load_bars(args)
    print(f"[stage] load/rebuild R02 lifecycle from {PROJECT_ROOT / args.r02_dir}", flush=True)
    print(f"[stage] load/rebuild R09 causal formation features from {PROJECT_ROOT / args.r09_dir}", flush=True)
    levels, lifecycle, level_features, r02_source, r09_source = load_r02_and_r09_levels(
        r02_dir=PROJECT_ROOT / args.r02_dir,
        r09_dir=PROJECT_ROOT / args.r09_dir,
        primary=bars,
        rebuild_r02_if_missing=bool(args.rebuild_r02_if_missing),
        rebuild_r09_if_missing=bool(args.rebuild_r09_if_missing),
        show_progress=not bool(args.no_progress),
    )
    print(
        f"[structure] r02={r02_source} r09={r09_source} "
        f"levels={len(levels):,} lifecycle={len(lifecycle):,} features={len(level_features):,}",
        flush=True,
    )

    print("[stage] build causal first-approach liquidity-pool checkpoints", flush=True)
    candidates = build_liquidity_magnet_universe(
        lifecycle,
        level_features,
        bars,
        cfg,
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        max_candidates=int(args.max_candidates),
        show_progress=not bool(args.no_progress),
    )
    print(f"[universe] unique_pool_checkpoints={len(candidates):,}", flush=True)
    if candidates.empty:
        raise RuntimeError("R11 generated no candidates")

    print("[stage] target-before-stop replay and realistic-cost frontier", flush=True)
    outcomes = attach_risk_frontier_outcomes(candidates, bars, cfg, show_progress=not bool(args.no_progress))

    dq = data_quality(bars, lifecycle, candidates, outcomes)
    audit = causal_audit(candidates, outcomes)
    failures = pd.concat(
        [
            dq.loc[dq["status"].eq("FAIL"), ["check", "value", "status"]].rename(columns={"value": "violations"}),
            audit.loc[audit["status"].eq("FAIL")],
        ],
        ignore_index=True,
    )
    if not failures.empty:
        raise RuntimeError(f"R11 fail-fast gate failed:\n{failures.to_string(index=False)}")

    print("[stage] distance, stop, quality, timeframe and period reports", flush=True)
    design = design_table(cfg)
    funnel = candidate_funnel(candidates, outcomes)
    frontier = risk_frontier_summary(outcomes)
    directional = directional_magnet_summary(outcomes)
    timeframe = timeframe_confluence_summary(outcomes)
    structures = structure_family_summary(outcomes)
    stability = period_stability(outcomes)
    score = scorecard(frontier, stability, cfg)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "created_at_utc": str(pd.Timestamp.now("UTC")),
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "research_start": research_start,
        "research_end_exclusive": research_end_exclusive,
        "r02_source": r02_source,
        "r09_source": r09_source,
        "distance_bands_bp": cfg.distance_bands_bp,
        "horizon_minutes": cfg.horizon_minutes,
        "config": asdict(cfg),
        "candidate_rows": len(candidates),
        "outcome_rows": len(outcomes),
        "notes": [
            "Target is 5bp before the upper edge of the active lower liquidity pool.",
            "Equal-distance stop is the clean directional magnet test.",
            "Local-high stops use only highs from completed bars through the signal bar.",
            "Entry is the strict next contiguous 1m open after the closed checkpoint bar; candidates crossing data gaps are discarded.",
            "No distance/stop combination mining is performed.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(manifest_json(manifest), encoding="utf-8")
    _write(dq, out_dir / "01_data_quality.csv")
    _write(design, out_dir / "02_frozen_design.csv")
    _write(funnel, out_dir / "03_candidate_funnel.csv")
    _write(frontier, out_dir / "04_risk_frontier_summary.csv")
    _write(directional, out_dir / "05_equal_distance_directional_magnet.csv")
    _write(timeframe, out_dir / "06_timeframe_confluence_summary.csv")
    _write(structures, out_dir / "07_structure_family_summary.csv")
    _write(stability, out_dir / "08_period_stability.csv")
    _write(score, out_dir / "09_candidate_scorecard.csv")
    _write(audit, out_dir / "10_causal_audit.csv")
    sample = outcomes.head(int(cfg.report_sample_rows)).copy()
    _write(sample, out_dir / "11_outcome_sample.csv")
    candidate_features, candidate_labels = _split_candidate_feature_labels(candidates)
    _write(candidate_features, out_dir / "12_candidate_feature_table.csv.gz", compression="gzip")
    _write(candidate_labels, out_dir / "13_candidate_future_label_table.csv.gz", compression="gzip")
    _write(outcomes, out_dir / "14_risk_frontier_outcome_table.csv.gz", compression="gzip")
    brief = research_brief(manifest=manifest, frontier=frontier, directional=directional, score=score)
    (out_dir / "15_research_brief.md").write_text(brief, encoding="utf-8")

    if not args.skip_review_pack:
        result = finalize_research_report(
            out_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )
        print(f"[done] review_pack={result.zip_path}", flush=True)
    elapsed = time.perf_counter() - started
    counts = score["decision"].value_counts().to_dict() if not score.empty else {}
    print(f"[done] report={out_dir} elapsed={elapsed:.1f}s", flush=True)
    print(
        "[decision-summary] "
        f"promote_to_backtest={int(counts.get('promote_to_backtest', 0))} "
        f"research_continue={int(counts.get('research_continue', 0))} "
        f"rejected={int(counts.get('rejected', 0))}",
        flush=True,
    )
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
