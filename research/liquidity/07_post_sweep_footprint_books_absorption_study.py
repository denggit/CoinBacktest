#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R07 full-history Range-Footprint + limited-coverage Books absorption study.

R07 asks whether completed causal Range-Footprints can distinguish durable
post-sweep turning attempts from failed lows, and whether compact 5000-level
Books + Trades liquidity-map features add independent evidence over their actual
local coverage.

This is mechanism research, not a strategy backtest:
- the broad universe contains every R04 new-low attempt;
- R06 oracle/prior/control cohorts are future-labelled only for retrospective
  matched diagnostics;
- strategy-facing Footprints use only range bars completed by the checkpoint's
  available time;
- Books use persisted `available_time` and are backward-asof only;
- no parameter grid or PnL optimization is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
from src.research_common.post_sweep_footprint_books import (  # noqa: E402
    PostSweepFootprintBooksConfig,
    aggregate_footprint_bars,
    attach_books_context,
    books_coverage_table,
    build_footprint_context,
    build_research_brief,
    causal_audit,
    cohort_feature_summary,
    data_quality_report,
    feature_outcome_auc,
    frozen_quantile_lift,
    load_r04_attempt_universe,
    load_r06_matched_universe,
    mechanism_scorecard,
    paired_feature_profile,
    pair_overlap_summary,
)
from src.research_common.post_sweep_footprint_books.reports import (  # noqa: E402
    PRIMARY_BOOK_METRICS,
    PRIMARY_FOOTPRINT_METRICS,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "07_post_sweep_footprint_books_absorption_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_POST_SWEEP_FOOTPRINT_BOOKS_ABSORPTION_R07"
EDGE_ID = "RESEARCH_ONLY_POST_SWEEP_FOOTPRINT_BOOKS_ABSORPTION"
TITLE = "ETH Post-Sweep Range-Footprint + Books Absorption Study R07"
DEFAULT_R04_DIR = "data/reports/research/liquidity/post_sweep_continuation_exhaustion_r04"
DEFAULT_R06_DIR = "data/reports/research/liquidity/post_sweep_micro_turning_point_r06"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/post_sweep_footprint_books_absorption_r07"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Study full-history causal Range Footprints and limited-coverage Books replenishment after swing-zone sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--r04-dir", default=DEFAULT_R04_DIR)
    p.add_argument("--r06-dir", default=DEFAULT_R06_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--footprint-db-name", default="okx_range_footprints.db")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--footprint-price-step", type=float, default=1.0)
    p.add_argument("--footprint-chunk-days", type=int, default=120)
    p.add_argument("--books-depth", type=int, default=5000)
    p.add_argument("--books-lookback-seconds", type=int, default=60)
    p.add_argument("--books-max-staleness-seconds", type=int, default=15)
    p.add_argument("--skip-books", action="store_true")
    p.add_argument("--books-required", action="store_true")
    p.add_argument("--max-events", type=int, default=0, help="Deterministic broad-universe smoke cap; 0 uses all attempts.")
    p.add_argument("--max-pairs", type=int, default=0, help="Deterministic matched-pair smoke cap; 0 uses all pairs.")
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _config(args: argparse.Namespace) -> PostSweepFootprintBooksConfig:
    return PostSweepFootprintBooksConfig(
        range_pct=float(args.range_pct),
        footprint_price_step=float(args.footprint_price_step),
        footprint_chunk_days=int(args.footprint_chunk_days),
        books_depth=int(args.books_depth),
        books_lookback_seconds=int(args.books_lookback_seconds),
        books_max_staleness_seconds=int(args.books_max_staleness_seconds),
        sample_rows=int(args.sample_rows),
    ).validate()


def _synthetic_footprint() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.DataFrame(
        {
            "bar_id": [1, 2, 3],
            "start_ts": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "end_ts": pd.to_datetime(["2025-01-01 00:01:00", "2025-01-01 00:02:00", "2025-01-01 00:03:00"]),
            "duration_seconds": [60.0, 60.0, 60.0],
            "open": [100.0, 99.8, 99.6],
            "high": [100.1, 99.9, 99.8],
            "low": [99.8, 99.6, 99.5],
            "close": [99.8, 99.6, 99.7],
            "direction": [-1.0, -1.0, 1.0],
            "notional": [10e6, 12e6, 9e6],
            "buy_notional": [3e6, 3e6, 5e6],
            "sell_notional": [7e6, 9e6, 4e6],
            "delta_notional": [-4e6, -6e6, 1e6],
            "large_buy_notional": [0.2e6, 0.2e6, 0.4e6],
            "large_sell_notional": [0.5e6, 0.8e6, 0.3e6],
            "large_delta_notional": [-0.3e6, -0.6e6, 0.1e6],
            "max_trade_notional": [100_000.0, 150_000.0, 120_000.0],
        }
    )
    rows = []
    for bar_id, low in zip(bars["bar_id"], bars["low"], strict=True):
        for offset in range(4):
            sell = float((4 - offset) * 1_000_000)
            buy = float((offset + 1) * 250_000)
            rows.append(
                {
                    "bar_id": bar_id,
                    "price_bucket": low + offset,
                    "notional": sell + buy,
                    "trades_count": 10 + offset,
                    "buy_notional": buy,
                    "sell_notional": sell,
                    "delta_notional": buy - sell,
                    "large_buy_notional": 0.0,
                    "large_sell_notional": sell * 0.1,
                    "large_delta_notional": -sell * 0.1,
                    "max_trade_notional": 50_000.0 + offset,
                }
            )
    return bars, pd.DataFrame(rows)


def run_self_test() -> None:
    cfg = PostSweepFootprintBooksConfig(footprint_price_step=1.0).validate()
    bars, footprints = _synthetic_footprint()
    aggregated = aggregate_footprint_bars(bars, footprints, cfg)
    if len(aggregated) != 3:
        raise RuntimeError(f"R07 self-test expected 3 footprint bars, got {len(aggregated)}")
    events = pd.DataFrame(
        {
            "checkpoint_id": ["A", "B"],
            "checkpoint_available_time": pd.to_datetime(["2025-01-01 00:01:30", "2025-01-01 00:02:30"]),
        }
    )
    from src.research_common.post_sweep_footprint_books.footprint import attach_footprint_context

    attached = attach_footprint_context(events, aggregated)
    if attached["fp_bar_id"].tolist() != [1, 2]:
        raise RuntimeError(f"R07 self-test causal asof failed: {attached[['checkpoint_id', 'fp_bar_id']].to_dict('records')}")
    if not attached["fp_causal_valid"].all():
        raise RuntimeError("R07 self-test causal audit failed")
    print("[self-test] passed footprint aggregation and causal as-of", flush=True)


def _sample(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= rows:
        return frame.copy()
    return frame.sort_values([name for name in ("period", "checkpoint_available_time", "checkpoint_id") if name in frame.columns], kind="mergesort").head(rows).copy()


def _merge_context(features: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return features.copy()
    payload = context.drop(columns=["checkpoint_available_time"], errors="ignore")
    return features.merge(payload, on="checkpoint_id", how="left", validate="many_to_one")


def _distinct_pair_features(matched: pd.DataFrame) -> pd.DataFrame:
    oracle = matched.loc[matched["cohort"] == "ORACLE_TURN", ["pair_id", "fp_bar_id"]].rename(columns={"fp_bar_id": "oracle_bar"})
    prior = matched.loc[matched["cohort"] == "PRIOR_FAILED_ATTEMPT", ["pair_id", "fp_bar_id"]].rename(columns={"fp_bar_id": "prior_bar"})
    pairs = oracle.merge(prior, on="pair_id", how="inner", validate="one_to_one")
    valid = pairs["oracle_bar"].notna() & pairs["prior_bar"].notna() & (pairs["oracle_bar"] != pairs["prior_bar"])
    ids = set(pairs.loc[valid, "pair_id"].astype(str))
    return matched.loc[matched["pair_id"].astype(str).isin(ids)].copy()


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    cfg = _config(args)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    r04_dir = PROJECT_ROOT / args.r04_dir
    r06_dir = PROJECT_ROOT / args.r06_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    range_db = data_dir / args.range_db_name
    footprint_db = data_dir / args.footprint_db_name
    if not range_db.exists():
        raise FileNotFoundError(f"range-bar DB missing: {range_db}")
    if not footprint_db.exists():
        raise FileNotFoundError(f"range-footprint DB missing: {footprint_db}")

    timings: dict[str, float] = {}
    started = time.perf_counter()
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] {start} -> {end}", flush=True)
    print(f"[footprint] range={cfg.range_pct:.4%} step={cfg.footprint_price_step:g} full-history", flush=True)
    print(f"[books] depth={cfg.books_depth} lookback={cfg.books_lookback_seconds}s optional={not args.books_required}", flush=True)

    stage = time.perf_counter()
    print("[stage] load broad R04 new-low attempts and matched R06 cohorts", flush=True)
    broad_features, broad_labels = load_r04_attempt_universe(
        r04_dir,
        start=start,
        end=end,
        max_events=int(args.max_events),
    )
    matched_features, matched_labels = load_r06_matched_universe(
        r06_dir,
        start=start,
        end=end,
        max_pairs=int(args.max_pairs),
    )
    timings["load_universes_seconds"] = time.perf_counter() - stage
    print(f"[universe] broad={len(broad_features):,} matched={len(matched_features):,}", flush=True)

    combined_events = pd.concat(
        [
            broad_features[["checkpoint_id", "checkpoint_available_time"]],
            matched_features[["checkpoint_id", "checkpoint_available_time"]],
        ],
        ignore_index=True,
    ).drop_duplicates("checkpoint_id", keep="first")

    stage = time.perf_counter()
    print("[stage] causal completed Range-Footprint context", flush=True)
    range_loader = OKXRangeBarLoader(
        symbol=args.symbol,
        range_pct=cfg.range_pct,
        data_dir=data_dir,
        db_name=args.range_db_name,
    )
    footprint_loader = OKXRangeFootprintLoader(
        symbol=args.symbol,
        range_pct=cfg.range_pct,
        price_step=cfg.footprint_price_step,
        data_dir=data_dir,
        db_name=args.footprint_db_name,
    )
    footprint_result = build_footprint_context(
        combined_events,
        range_loader=range_loader,
        footprint_loader=footprint_loader,
        config=cfg,
        progress=not args.no_progress,
    )
    broad_features = _merge_context(broad_features, footprint_result.context)
    matched_features = _merge_context(matched_features, footprint_result.context)
    timings["footprint_seconds"] = time.perf_counter() - stage
    fp_coverage = float(broad_features.get("fp_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).mean())
    print(f"[footprint] attached={fp_coverage:.2%} chunks={len(footprint_result.audit)}", flush=True)
    if fp_coverage < 0.90:
        raise RuntimeError(
            f"R07 footprint coverage gate failed: {fp_coverage:.2%}. "
            "Check r0020 step1 database coverage and table parameters before interpreting the report."
        )

    stage = time.perf_counter()
    print("[stage] optional compact 5000-level Books mechanism context", flush=True)
    books_audit = pd.DataFrame()
    books_coverage = pd.DataFrame()
    if not args.skip_books:
        books_loader = OKXLiquidityMapLoader(
            symbol=args.symbol,
            books_depth=cfg.books_depth,
            data_dir=str(data_dir),
        )
        books_coverage = books_coverage_table(books_loader)
        books_result = attach_books_context(
            combined_events,
            loader=books_loader,
            config=cfg,
            progress=not args.no_progress,
        )
        broad_features = _merge_context(broad_features, books_result.context)
        matched_features = _merge_context(matched_features, books_result.context)
        books_audit = books_result.audit
        books_rows = int(broad_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        matched_books_rows = int(matched_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        print(
            f"[books] compact_days={len(books_coverage):,} broad_causal={books_rows:,} "
            f"matched_causal={matched_books_rows:,}",
            flush=True,
        )
        if args.books_required and books_rows == 0:
            raise RuntimeError(
                "R07 --books-required was set but no causal compact liquidity-map rows were found. "
                "Prebuild the local 5000-level offline liquidity map or run without --books-required."
            )
    else:
        print("[books] skipped by user", flush=True)
    timings["books_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    print("[stage] matched profiles, broad outcome discrimination, and fixed gates", flush=True)
    footprint_cohort = cohort_feature_summary(matched_features, metrics=PRIMARY_FOOTPRINT_METRICS)
    footprint_paired = paired_feature_profile(matched_features, metrics=PRIMARY_FOOTPRINT_METRICS)
    distinct_matched = _distinct_pair_features(matched_features)
    footprint_distinct_paired = paired_feature_profile(distinct_matched, metrics=PRIMARY_FOOTPRINT_METRICS)
    overlap = pair_overlap_summary(matched_features)
    auc_report = feature_outcome_auc(
        broad_features,
        broad_labels,
        metrics=PRIMARY_FOOTPRINT_METRICS,
        minimum_events=cfg.minimum_period_events,
    )
    lift_report = frozen_quantile_lift(
        broad_features,
        broad_labels,
        reference_period=cfg.frozen_reference_period,
        metrics=PRIMARY_FOOTPRINT_METRICS,
        minimum_events=cfg.minimum_period_events,
    )
    books_broad_valid = broad_features.loc[
        broad_features.get("books_causal_valid", pd.Series(False, index=broad_features.index)).fillna(False).astype(bool)
    ].copy()
    books_matched_valid = matched_features.loc[
        matched_features.get("books_causal_valid", pd.Series(False, index=matched_features.index)).fillna(False).astype(bool)
    ].copy()
    books_cohort = cohort_feature_summary(books_matched_valid, metrics=PRIMARY_BOOK_METRICS)
    books_paired = paired_feature_profile(books_matched_valid, metrics=PRIMARY_BOOK_METRICS)
    books_auc_report = feature_outcome_auc(
        books_broad_valid,
        broad_labels,
        metrics=PRIMARY_BOOK_METRICS,
        minimum_events=cfg.minimum_period_events,
    )
    scorecard = mechanism_scorecard(
        auc_report,
        lift_report,
        footprint_paired,
        reference_period=cfg.frozen_reference_period,
    )
    audit = causal_audit(broad_features, matched_features)
    quality = data_quality_report(
        broad_features,
        broad_labels,
        matched_features,
        matched_labels,
        footprint_result.audit,
        books_audit,
    )
    if (audit["status"] == "FAIL").any() or (quality["status"] == "FAIL").any():
        raise RuntimeError(
            "R07 causal/data-quality gate failed:\n"
            + pd.concat([quality.loc[quality["status"] == "FAIL"], audit.loc[audit["status"] == "FAIL"]], ignore_index=True).to_string(index=False)
        )
    timings["reports_seconds"] = time.perf_counter() - stage

    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "start_date": str(start),
        "end_date": str(end),
        "symbol": args.symbol,
        "config": asdict(cfg),
        "sources": {
            "r04_dir": str(r04_dir),
            "r06_dir": str(r06_dir),
            "range_db": str(range_db),
            "footprint_db": str(footprint_db),
            "books_depth": cfg.books_depth,
        },
        "rows": {
            "broad_features": len(broad_features),
            "broad_labels": len(broad_labels),
            "matched_features": len(matched_features),
            "matched_labels": len(matched_labels),
            "books_broad_causal": int(broad_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "books_matched_causal": int(matched_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        },
        "causal_policy": {
            "footprint": "latest completed range footprint with end_ts <= checkpoint_available_time",
            "books": "compact liquidity-map available_time <= checkpoint_available_time",
            "future_labels_physically_separate": True,
            "oracle_selection_uses_future": True,
        },
        "timings_seconds": {**timings, "total": time.perf_counter() - started},
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    quality.to_csv(out_dir / "01_data_quality.csv", index=False)
    (
        broad_features.groupby("period", dropna=False).size().rename("broad_attempts").reset_index()
        .merge(matched_features.groupby(["period", "cohort"], dropna=False).size().rename("matched_events").reset_index(), on="period", how="outer")
        .to_csv(out_dir / "02_attempt_universe_summary.csv", index=False)
    )
    footprint_result.audit.to_csv(out_dir / "03_footprint_cache_audit.csv", index=False)
    footprint_cohort.to_csv(out_dir / "04_footprint_cohort_summary.csv", index=False)
    footprint_paired.to_csv(out_dir / "05_footprint_oracle_prior_paired_profile.csv", index=False)
    footprint_distinct_paired.to_csv(out_dir / "06_footprint_distinct_bar_paired_profile.csv", index=False)
    overlap.to_csv(out_dir / "07_footprint_pair_overlap_summary.csv", index=False)
    auc_report.to_csv(out_dir / "08_footprint_outcome_auc.csv", index=False)
    lift_report.to_csv(out_dir / "09_footprint_frozen_quantile_lift.csv", index=False)
    books_coverage.to_csv(out_dir / "10_books_coverage.csv", index=False)
    books_audit.to_csv(out_dir / "11_books_cache_audit.csv", index=False)
    books_cohort.to_csv(out_dir / "12_books_cohort_summary.csv", index=False)
    books_paired.to_csv(out_dir / "13_books_oracle_prior_paired_profile.csv", index=False)
    books_auc_report.to_csv(out_dir / "14_books_outcome_auc.csv", index=False)
    scorecard.to_csv(out_dir / "15_mechanism_scorecard.csv", index=False)
    audit.to_csv(out_dir / "16_causal_audit.csv", index=False)
    _sample(broad_features, cfg.sample_rows).to_csv(out_dir / "17_footprint_event_sample.csv", index=False)
    _sample(books_broad_valid, cfg.sample_rows).to_csv(out_dir / "18_books_event_sample.csv", index=False)
    broad_features.to_csv(out_dir / "19_all_attempt_feature_table.csv.gz", index=False, compression="gzip")
    broad_labels.to_csv(out_dir / "20_all_attempt_label_table.csv.gz", index=False, compression="gzip")
    matched_features.to_csv(out_dir / "21_matched_feature_table.csv.gz", index=False, compression="gzip")
    matched_labels.to_csv(out_dir / "22_matched_label_table.csv.gz", index=False, compression="gzip")
    build_research_brief(
        output_path=out_dir / "23_research_brief.md",
        broad_features=broad_features,
        matched_features=matched_features,
        scorecard=scorecard,
        overlap=overlap,
        books_coverage=books_coverage,
    )
    if not args.skip_review_pack:
        finalize_research_report(
            out_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )
    print(f"[done] report={out_dir}", flush=True)
    print(f"[timing] total={time.perf_counter() - started:.1f}s footprint={timings['footprint_seconds']:.1f}s books={timings['books_seconds']:.1f}s", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
