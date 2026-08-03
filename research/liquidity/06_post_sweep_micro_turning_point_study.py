#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R06 Post-Sweep 1s / Range-Bar Turning-Point Mechanism Study.

R06 asks whether the final durable turning attempt can be distinguished from the
previous failed new-low attempt early enough to lower post-entry MAE without
blindly catching a continuing selloff.

This is mechanism research, not a final strategy backtest:
- Oracle turning attempts are selected with future labels only to define the
  retrospective comparison cohort.
- Candidate trigger features are causal and use closed 1s bars only.
- Candidate entry is the next 1s bar open.
- `ORACLE_LOW_PLUS_1S` is written as a labelled upper bound and is never included
  in the causal scorecard.
- Threshold variants are a small predeclared natural neighborhood; no parameter
  grid or in-sample winner optimization is performed.
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

from src.data_feed.okx_event_trade_window_loader import OKXEventTradeWindowLoader  # noqa: E402
from src.research_common.post_sweep_micro import (  # noqa: E402
    PostSweepMicroConfig,
    analyze_micro_window,
    attach_optional_oi_context,
    build_attempt_universe,
    build_research_brief,
    candidate_scorecard,
    causal_audit,
    cohort_low_feature_summary,
    data_quality_report,
    extract_range_context,
    load_binance_oi_context,
    load_optional_r05_oi,
    load_r04_micro_source,
    paired_micro_profile,
    range_pair_overlap_summary,
    range_pair_profile,
    raw_hourly_coverage_report,
    trigger_occurrence_summary,
    trigger_path_summary,
    trigger_relative_to_baselines,
    validate_micro_data_gate,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "06_post_sweep_micro_turning_point_study"
SCRIPT_VERSION = "1.2.0"
EXPERIMENT_ID = "ETH_POST_SWEEP_MICRO_TURNING_POINT_R06"
EDGE_ID = "RESEARCH_ONLY_POST_SWEEP_MICRO_ABSORPTION_ENTRY"
TITLE = "ETH Post-Sweep 1s / Range-Bar Turning-Point Mechanism Study R06"
DEFAULT_R04_DIR = "data/reports/research/liquidity/post_sweep_continuation_exhaustion_r04"
DEFAULT_R05_DIR = "data/reports/research/liquidity/post_sweep_binance_oi_mechanism_r05"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/post_sweep_micro_turning_point_r06"


def _comma_floats(value: str) -> tuple[float, ...]:
    result = tuple(sorted(set(float(item.strip()) for item in str(value).split(",") if item.strip())))
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare final turning attempts with prior failed lows using sparse 1s raw-trade windows and Range Bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--r04-dir", default=DEFAULT_R04_DIR)
    p.add_argument("--r05-dir", default=DEFAULT_R05_DIR, help="Fallback only when direct Binance OI DB is unavailable.")
    p.add_argument("--oi-source", choices=("binance_db", "r05", "none"), default="binance_db")
    p.add_argument("--binance-symbol", default="ETHUSDT")
    p.add_argument("--binance-metrics-db", default="binance_futures_metrics.db")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--raw-chunksize", type=int, default=300_000)
    p.add_argument("--allow-download-missing-raw", action="store_true")
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--range-pcts", type=_comma_floats, default=(0.0015, 0.0020, 0.0025))
    p.add_argument("--range-chunk-days", type=int, default=365)
    p.add_argument("--pre-window-seconds", type=int, default=60)
    p.add_argument("--post-window-seconds", type=int, default=660)
    p.add_argument("--control-multiplier", type=float, default=1.0)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--max-pairs", type=int, default=0, help="Deterministic smoke/debug cap; 0 uses the full universe.")
    p.add_argument("--skip-micro", action="store_true")
    p.add_argument("--skip-range", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--micro-max-no-trade-rate", type=float, default=0.02)
    p.add_argument("--micro-fail-fast-min-windows", type=int, default=10)
    p.add_argument("--micro-max-hour-no-trade-rate", type=float, default=0.10)
    p.add_argument("--micro-hour-min-windows", type=int, default=20)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _config(args: argparse.Namespace) -> PostSweepMicroConfig:
    return PostSweepMicroConfig(
        pre_window_seconds=int(args.pre_window_seconds),
        post_window_seconds=int(args.post_window_seconds),
        range_pcts=tuple(float(v) for v in args.range_pcts),
        control_multiplier=float(args.control_multiplier),
        round_trip_cost=float(args.round_trip_cost),
        sample_rows=int(args.sample_rows),
    ).validate()


def _synthetic_micro(event: pd.Series, cfg: PostSweepMicroConfig) -> pd.DataFrame:
    index = pd.date_range(event["start_time"], event["end_time"], freq="1s", inclusive="left")
    n = len(index)
    anchor = int((pd.Timestamp(event["checkpoint_time"]) - pd.Timestamp(event["start_time"])).total_seconds())
    price = np.full(n, 100.0)
    # Selloff into a final low, then impact collapse/reclaim while sell flow stays high.
    for i in range(max(0, anchor - 20), min(n, anchor + 25)):
        price[i] = 100.0 - max(0, i - (anchor - 20)) * 0.015
    for i in range(anchor + 25, min(n, anchor + 80)):
        price[i] = price[anchor + 24] + (i - (anchor + 24)) * 0.02
    price = pd.Series(price).replace(100.0, np.nan).ffill().bfill().to_numpy()
    sell = np.full(n, 1_000_000.0)
    buy = np.full(n, 500_000.0)
    buy[anchor + 25: anchor + 60] = 900_000.0
    out = pd.DataFrame(
        {
            "window_id": event["window_id"], "timestamp": index,
            "available_time": index + pd.Timedelta(seconds=1),
            "open": price, "high": price + 0.005, "low": price - 0.005, "close": price,
            "volume": 1.0, "trades_count": 10, "buy_volume": 0.4, "sell_volume": 0.6,
            "notional": buy + sell, "buy_notional": buy, "sell_notional": sell,
            "buy_trades_count": 4, "sell_trades_count": 6,
            "delta_volume": -0.2, "delta_notional": buy - sell,
            "taker_buy_ratio": buy / (buy + sell), "large_buy_notional": 0.0,
            "large_sell_notional": 0.0, "large_delta_notional": 0.0,
            "large_buy_trades_count": 0, "large_sell_trades_count": 0,
            "large_trades_count": 0, "max_trade_notional": 50_000.0,
            "max_trade_size": 1.0, "vwap": price,
        }
    )
    return out


def run_self_test() -> None:
    cfg = PostSweepMicroConfig(pre_window_seconds=60, post_window_seconds=420, future_horizons_seconds=(30, 60, 180, 300)).validate()
    anchor = pd.Timestamp("2025-01-01 12:00:00")
    event = pd.Series(
        {
            "window_id": "TEST", "checkpoint_id": "C1", "zone_event_id": "E1",
            "pair_id": "E1", "cohort": "ORACLE_TURN", "period": "TEST",
            "checkpoint_time": anchor, "start_time": anchor - pd.Timedelta(seconds=60),
            "end_time": anchor + pd.Timedelta(seconds=420),
            "prior_running_low_before_attempt": 99.9,
        }
    )
    raw = _synthetic_micro(event, cfg)
    feature, triggers, audit = analyze_micro_window(raw, event, cfg)
    if feature is None or audit.get("status") != "complete":
        raise RuntimeError(f"R06 self-test failed feature/audit={audit}")
    causal = [row for row in triggers if not row["signal_uses_future"]]
    if not causal or any(not row["entry_is_next_bar_open"] for row in causal):
        raise RuntimeError("R06 self-test failed causal next-open audit")
    print(f"[self-test] passed triggers={len(triggers)}", flush=True)


def _cap_universe(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    pair_audit: pd.DataFrame,
    max_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if max_pairs <= 0 or len(pair_audit) <= max_pairs:
        return features, labels, pair_audit
    selected_pairs = set(pair_audit.sort_values(["period", "pair_id"], kind="mergesort").head(max_pairs)["pair_id"].astype(str))
    keep = features["pair_id"].astype(str).isin(selected_pairs)
    ids = set(features.loc[keep, "window_id"].astype(str))
    return (
        features.loc[keep].reset_index(drop=True),
        labels.loc[labels["window_id"].astype(str).isin(ids)].reset_index(drop=True),
        pair_audit.loc[pair_audit["pair_id"].astype(str).isin(selected_pairs)].reset_index(drop=True),
    )


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    cfg = _config(args)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    r04_dir = PROJECT_ROOT / args.r04_dir
    r05_dir = PROJECT_ROOT / args.r05_dir if args.r05_dir else None
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] {start} -> {end}", flush=True)
    print(f"[source] R04={r04_dir}", flush=True)
    print(f"[micro] sparse raw-trade windows timeframe=1s pre={cfg.pre_window_seconds}s post={cfg.post_window_seconds}s", flush=True)
    print(f"[range] pcts={cfg.range_pcts} db={args.range_db_name}", flush=True)

    print("[stage] load R04 features/labels and optional R05 OI context", flush=True)
    r04_features, r04_labels, static = load_r04_micro_source(r04_dir)
    time_mask = (
        (r04_features["checkpoint_available_time"] >= start)
        & (r04_features["checkpoint_available_time"] <= end)
    )
    checkpoint_ids = set(r04_features.loc[time_mask, "checkpoint_id"].astype(str))
    r04_features = r04_features.loc[time_mask].reset_index(drop=True)
    r04_labels = r04_labels.loc[r04_labels["checkpoint_id"].astype(str).isin(checkpoint_ids)].reset_index(drop=True)
    active_events = set(r04_features["zone_event_id"].astype(str))
    static = static.loc[static["zone_event_id"].astype(str).isin(active_events)].reset_index(drop=True)
    print(f"[source] checkpoints={len(r04_features):,} events={len(active_events):,}", flush=True)

    print("[stage] build future-labelled oracle/prior/control attempt universe", flush=True)
    universe, labels, pair_audit = build_attempt_universe(r04_features, r04_labels, static, cfg)
    universe, labels, pair_audit = _cap_universe(universe, labels, pair_audit, int(args.max_pairs))
    selected_checkpoint_ids = set(universe["checkpoint_id"].astype(str))
    if args.oi_source == "binance_db":
        print("[stage] load indexed Binance OI features and causally align selected checkpoints", flush=True)
        oi = load_binance_oi_context(
            universe,
            symbol=args.binance_symbol,
            data_dir=args.data_dir,
            db_name=args.binance_metrics_db,
        )
        if oi.empty:
            raise RuntimeError(
                "Binance OI database returned no aligned rows. Run "
                "tools\\prebuild_binance_futures_metrics.py first, or use --oi-source none explicitly."
            )
    elif args.oi_source == "r05":
        print("[stage] stream selected R05 OI rows from compressed report", flush=True)
        oi = load_optional_r05_oi(r05_dir, selected_checkpoint_ids)
    else:
        print("[stage] OI context disabled", flush=True)
        oi = pd.DataFrame()
    universe = attach_optional_oi_context(universe, oi)
    print(f"[oi] source={args.oi_source} selected_rows={len(oi):,}", flush=True)
    del r04_features, r04_labels, static, oi, checkpoint_ids, active_events, selected_checkpoint_ids, time_mask
    print(
        f"[universe] windows={len(universe):,} pairs={len(pair_audit):,} "
        f"cohorts={universe['cohort'].value_counts().to_dict()}", flush=True,
    )

    window_feature_rows: list[dict[str, object]] = []
    trigger_rows: list[dict[str, object]] = []
    micro_audit_rows: list[dict[str, object]] = []
    raw_coverage_parts: list[pd.DataFrame] = []

    if not args.skip_micro:
        print("[stage] stream project-local raw archive days; reconstruct cross-midnight 1s event windows", flush=True)
        loader = OKXEventTradeWindowLoader(symbol=args.symbol, data_dir=args.data_dir)
        prepared, _ = loader.prepare_windows(universe[["window_id", "start_time", "end_time"]])
        total_days = int(prepared["archive_day"].nunique()) if not prepared.empty else 0
        reporter = ProgressReporter(
            label="[micro raw-archive-days]",
            total=total_days,
            every=max(1, total_days // 100),
            enabled=not args.no_progress,
        )
        done_days = 0
        for batch in loader.iter_daily_window_bars(
            universe[["window_id", "start_time", "end_time"]],
            timeframe="1s",
            chunksize=int(args.raw_chunksize),
            allow_download_missing=bool(args.allow_download_missing_raw),
        ):
            raw_coverage_parts.append(batch.coverage)
            if batch.archive_day.year == 1970:
                continue
            done_days += 1
            reporter.update(done_days)
            day_ids = set(batch.coverage["window_id"].astype(str))
            events = universe.loc[universe["window_id"].astype(str).isin(day_ids)]
            grouped = {str(key): grp for key, grp in batch.bars.groupby("window_id", sort=False)} if not batch.bars.empty else {}
            for _, event in events.iterrows():
                raw = grouped.get(str(event["window_id"]), pd.DataFrame())
                feature, triggers, audit = analyze_micro_window(raw, event, cfg)
                micro_audit_rows.append(audit)
                if feature is not None:
                    window_feature_rows.append(feature)
                trigger_rows.extend(triggers)
        reporter.close()
    else:
        print("[stage] micro skipped by CLI", flush=True)

    window_features = pd.DataFrame(window_feature_rows)
    triggers = pd.DataFrame(trigger_rows)
    micro_audit = pd.DataFrame(micro_audit_rows)
    raw_coverage = pd.concat(raw_coverage_parts, ignore_index=True) if raw_coverage_parts else pd.DataFrame()
    print(f"[micro] complete_windows={len(window_features):,} trigger_rows={len(triggers):,}", flush=True)
    if not args.skip_micro:
        micro_gate = validate_micro_data_gate(
            universe,
            window_features,
            triggers,
            raw_coverage,
            max_no_trade_rate=float(args.micro_max_no_trade_rate),
            min_checked_windows=int(args.micro_fail_fast_min_windows),
            max_hour_no_trade_rate=float(args.micro_max_hour_no_trade_rate),
            min_hour_windows=int(args.micro_hour_min_windows),
        )
        print(f"[micro-gate] PASS {micro_gate}", flush=True)

    if not args.skip_range:
        print("[stage] causal Range-Bar context in bounded calendar chunks", flush=True)
        range_total = len(cfg.range_pcts) * max(1, int(np.ceil((end.normalize() - start.normalize()).days / max(1, args.range_chunk_days))))
        range_reporter = ProgressReporter(
            label="[range chunks]", total=range_total, every=max(1, range_total // 100), enabled=not args.no_progress
        )
        def range_progress(done: int, total: int, pct: float, core_start: pd.Timestamp, core_end: pd.Timestamp) -> None:
            del total, pct, core_start, core_end
            range_reporter.total = max(range_reporter.total, done)
            range_reporter.update(done)
        range_features, range_audit = extract_range_context(
            universe, cfg, symbol=args.symbol, data_dir=args.data_dir,
            db_name=args.range_db_name, chunk_days=int(args.range_chunk_days),
            progress_callback=range_progress,
        )
        range_reporter.close()
    else:
        print("[stage] range skipped by CLI", flush=True)
        range_features, range_audit = pd.DataFrame(), pd.DataFrame()
    print(f"[range] feature_rows={len(range_features):,}", flush=True)

    print("[stage] summaries, matched differences, candidate scorecard and causal audit", flush=True)
    reports = {
        "01_data_quality.csv": data_quality_report(universe, labels, pair_audit, micro_audit, range_audit, raw_coverage),
        "01b_raw_hourly_coverage.csv": raw_hourly_coverage_report(raw_coverage),
        "02_attempt_universe_summary.csv": (
            universe.groupby(["period", "cohort"], dropna=False).size().rename("events").reset_index()
        ),
        "03_micro_low_feature_summary.csv": cohort_low_feature_summary(window_features),
        "04_micro_oracle_vs_prior_paired_profile.csv": paired_micro_profile(window_features),
        "05_trigger_occurrence_summary.csv": trigger_occurrence_summary(universe, triggers),
        "06_trigger_path_summary.csv": trigger_path_summary(triggers),
        "07_trigger_relative_to_baselines.csv": trigger_relative_to_baselines(triggers),
        "08_range_pair_overlap_summary.csv": range_pair_overlap_summary(range_features),
        "09_range_oracle_vs_prior_paired_profile.csv": range_pair_profile(range_features),
    }
    scorecard = candidate_scorecard(universe, triggers)
    audit = causal_audit(universe, triggers)
    reports["10_candidate_scorecard.csv"] = scorecard
    reports["11_causal_audit.csv"] = audit
    violations = int(pd.to_numeric(audit.get("violations"), errors="coerce").fillna(0).sum()) if not audit.empty else 0
    fail_rows = audit.loc[audit["status"].eq("FAIL")] if not audit.empty else pd.DataFrame()
    if violations or not fail_rows.empty:
        raise RuntimeError(f"R06 causal audit failed:\n{audit.to_string(index=False)}")

    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "research_only": True,
        "start_date": str(start),
        "end_date": str(end),
        "config": asdict(cfg),
        "attempt_windows": int(len(universe)),
        "oracle_prior_pairs": int(len(pair_audit)),
        "micro_complete_windows": int(len(window_features)),
        "trigger_rows": int(len(triggers)),
        "range_feature_rows": int(len(range_features)),
        "notes": [
            "Oracle cohort selection uses future labels only for retrospective mechanism comparison.",
            "Causal triggers use closed 1s bars and next-1s-open execution.",
            "ORACLE_LOW_PLUS_1S is an upper-bound label and excluded from the candidate scorecard.",
            "Raw ZIP filename dates use the project-local calendar (UTC+8 by default); trade rows remain Unix UTC milliseconds.",
            "Cross-project-midnight windows are reconstructed from adjacent archive files instead of being skipped.",
            "Raw trades are streamed by bounded archive-day batches; full multi-year 1s bars are not materialized.",
            "Round-trip cost defaults to 0.11%.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name, report in reports.items():
        report.to_csv(out_dir / name, index=False)

    pair_audit.to_csv(out_dir / "12_oracle_prior_pair_audit.csv", index=False)
    raw_coverage.to_csv(out_dir / "13_raw_window_coverage.csv", index=False)
    micro_audit.to_csv(out_dir / "14_micro_window_audit.csv", index=False)
    range_audit.to_csv(out_dir / "15_range_cache_audit.csv", index=False)
    sample_n = int(cfg.sample_rows)
    window_features.head(sample_n).to_csv(out_dir / "16_micro_window_feature_sample.csv", index=False)
    triggers.head(sample_n).to_csv(out_dir / "17_trigger_path_sample.csv", index=False)
    range_features.head(sample_n).to_csv(out_dir / "18_range_feature_sample.csv", index=False)
    universe.to_csv(out_dir / "19_attempt_feature_table.csv.gz", index=False, compression="gzip")
    labels.to_csv(out_dir / "20_attempt_label_table.csv.gz", index=False, compression="gzip")
    window_features.to_csv(out_dir / "21_micro_window_feature_table.csv.gz", index=False, compression="gzip")
    triggers.to_csv(out_dir / "22_trigger_path_table.csv.gz", index=False, compression="gzip")
    range_features.to_csv(out_dir / "23_range_feature_table.csv.gz", index=False, compression="gzip")
    (out_dir / "24_research_brief.md").write_text(
        build_research_brief(universe, micro_audit, scorecard), encoding="utf-8"
    )

    if not args.skip_review_pack:
        finalize_research_report(
            out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE
        )
    print(f"[done] report={out_dir}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
