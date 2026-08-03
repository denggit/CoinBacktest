#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R12 causal post-sweep rejection versus acceptance study.

R09 established that structured Swing-Low sweeps release abnormal downside flow,
but unconditional reversal and pre-sweep routes failed.  R12 returns to the
original manual hypothesis and tests the missing branch explicitly: after the
sweep, do lower prices fail to gain acceptance (long) or does the market remain
below the zone / fail a reclaim (short)?

No exact-bottom prediction, condition-combination mining, or parameter grid is
used.  State is measured at 1/3/5/10 closed minutes and entry is the strict next
1m open.
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
from src.research_common.post_sweep_acceptance import (  # noqa: E402
    PostSweepAcceptanceConfig,
    attach_checkpoint_outcomes,
    build_post_sweep_checkpoints,
    causal_audit,
    data_quality,
    design_table,
    direction_outcome_summary,
    family_timeframe_summary,
    load_r09_zone_events,
    manifest_json,
    period_stability,
    release_interaction,
    research_brief,
    scorecard,
    state_distribution,
    state_feature_profile,
    transition_matrix,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.structured_stop_pool import FAMILY_COLUMNS  # noqa: E402
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars  # noqa: E402

SCRIPT_NAME = "12_post_sweep_rejection_acceptance_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_POST_SWEEP_REJECTION_ACCEPTANCE_R12"
EDGE_ID = "RESEARCH_ONLY_SWEEP_REJECT_ACCEPT_BRANCH"
TITLE = "ETH Post-Sweep Rejection vs Acceptance Study R12"
DEFAULT_R09_DIR = "data/reports/research/liquidity/structured_swing_stop_pool_hypotheses_r09"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/12_post_sweep_rejection_acceptance_r12"


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
    p.add_argument("--r09-dir", default=DEFAULT_R09_DIR)
    p.add_argument("--data-source", choices=["trade_bar", "ohlcv_local"], default="trade_bar")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--max-events", type=int, default=0, help="Deterministic R09 sweep cap; 0 uses all.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.timeframe) != "1m":
        raise ValueError("R12 requires --timeframe 1m")
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
        "open", "high", "low", "close", "volume", "notional", "buy_notional",
        "sell_notional", "delta_notional", "trades_count",
    ]
    bars = normalize_primary_bars(bars.loc[:, [name for name in keep if name in bars.columns]].copy())
    required = {"open", "high", "low", "close", "notional", "sell_notional", "delta_notional", "trades_count"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise RuntimeError(
            f"R12 requires complete trade-bar OHLC/order-flow fields; missing={missing}. "
            "Use --data-source trade_bar."
        )
    print(f"[load] rows={len(bars):,} range={bars.index.min()}->{bars.index.max()} cols={len(bars.columns)}", flush=True)
    return bars


def _write(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression=compression)


def _feature_table(checkpoints: pd.DataFrame) -> pd.DataFrame:
    forbidden_prefixes = ("r1p", "r2p", "r3p", "future_")
    leaked = [name for name in checkpoints.columns if name.startswith(forbidden_prefixes)]
    if leaked:
        raise RuntimeError(f"future/outcome labels leaked into checkpoint features: {leaked[:10]}")
    return checkpoints.copy()


def _outcome_label_table(outcomes: pd.DataFrame, config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    ids = [
        "zone_event_id", "checkpoint_minutes", "checkpoint_available_time", "entry_time",
        "state", "state_direction", "trade_direction", "period", "high_release",
    ]
    labels = ["natural_stop_price", "natural_stop_distance_bp", "mfe_bp", "mae_bp", "horizon_end_pos"]
    for r in config.target_r_multiples:
        token = str(float(r)).replace(".", "p")
        labels.extend([name for name in outcomes.columns if name.startswith(f"r{token}_")])
    columns = [name for name in [*ids, *labels] if name in outcomes.columns]
    return outcomes.loc[:, columns].copy()


def _synthetic_bars() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=240, freq="1min")
    close = np.full(len(index), 100.0, dtype=float)
    # Reject event at 30: sweep, then recover above the zone.
    close[30:36] = [98.8, 99.2, 100.2, 100.8, 101.2, 101.5]
    # Accept event at 100: sweep and remain below.
    close[100:112] = [94.7, 94.4, 94.2, 94.0, 93.8, 93.6, 93.5, 93.4, 93.2, 93.1, 93.0, 92.9]
    # Add favorable paths after the checkpoints for both preferred directions.
    close[36:70] = np.linspace(101.5, 106.0, 34)
    close[112:150] = np.linspace(92.9, 88.0, 38)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    notional = np.full(len(index), 1_000_000.0)
    sell = np.full(len(index), 550_000.0)
    buy = notional - sell
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": notional / close, "notional": notional,
            "buy_notional": buy, "sell_notional": sell,
            "delta_notional": buy - sell, "trades_count": 100.0,
        },
        index=index,
    )


def _synthetic_zones(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_id, pos, floor, ceiling in (("Z_REJECT", 30, 99.0, 100.0), ("Z_ACCEPT", 100, 95.0, 95.0)):
        row = {
            "zone_event_id": event_id,
            "event_kind": "swing_zone_sweep",
            "event_pos": pos,
            "event_bar_time": bars.index[pos],
            "event_available_time": bars.index[pos] + pd.Timedelta(minutes=1),
            "zone_latest_level_available_time": bars.index[pos] - pd.Timedelta(minutes=30),
            "zone_floor_price": floor,
            "zone_ceiling_price": ceiling,
            "zone_center_price": np.sqrt(floor * ceiling),
            "zone_timeframe_count": 2,
            "zone_max_timeframe_min": 60,
            "sweep_low": float(bars["low"].iloc[pos]),
            "high_stop_release_label": True,
            "stop_release_score": 1.0,
            "period": "EARLY_2023_2024",
        }
        for family in FAMILY_COLUMNS:
            row[family] = family.endswith("multitimeframe_confluence")
        rows.append(row)
    return pd.DataFrame(rows)


def run_self_test() -> None:
    bars = _synthetic_bars()
    zones = _synthetic_zones(bars)
    cfg = replace(
        PostSweepAcceptanceConfig(),
        checkpoints_minutes=(1, 3, 5, 10),
        horizon_minutes=60,
        minimum_spec_events=1,
        minimum_period_events=1,
        minimum_promote_events=1,
    ).validate()
    checkpoints = build_post_sweep_checkpoints(
        zones, bars, cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    if checkpoints.empty:
        raise RuntimeError("R12 self-test produced no checkpoints")
    outcomes = attach_checkpoint_outcomes(checkpoints, bars, cfg, show_progress=False)
    audit = causal_audit(checkpoints, outcomes)
    failures = audit.loc[audit["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(f"R12 self-test causal failure:\n{failures.to_string(index=False)}")
    reject_states = set(checkpoints.loc[checkpoints["zone_event_id"].eq("Z_REJECT"), "state"])
    accept_states = set(checkpoints.loc[checkpoints["zone_event_id"].eq("Z_ACCEPT"), "state"])
    if not reject_states.intersection({"REJECT", "STRONG_REJECT"}):
        raise RuntimeError(f"R12 self-test reject classification failed: {reject_states}")
    if not accept_states.intersection({"PERSISTENT_ACCEPT", "RECLAIM_FAILED"}):
        raise RuntimeError(f"R12 self-test accept classification failed: {accept_states}")
    print(f"[self-test] passed checkpoints={len(checkpoints):,} outcomes={len(outcomes):,}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    cfg = replace(PostSweepAcceptanceConfig(), report_sample_rows=int(args.sample_rows)).validate()
    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    if research_end_exclusive <= research_start:
        raise ValueError("end-date must be after start-date")
    started = time.perf_counter()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={research_start}->{research_end_exclusive}", flush=True)
    print("[design] R09 real-liquidity sweep -> 1/3/5/10m rejection or acceptance -> next-open long/short; no exact-bottom or combination mining", flush=True)

    bars = _load_bars(args)
    r09_dir = PROJECT_ROOT / args.r09_dir
    print(f"[stage] load R09 causal zone/release cache from {r09_dir}", flush=True)
    zones, r09_source = load_r09_zone_events(r09_dir)
    zones = zones.loc[
        pd.to_datetime(zones["event_available_time"], errors="coerce").ge(research_start)
        & pd.to_datetime(zones["event_available_time"], errors="coerce").lt(research_end_exclusive)
    ].copy()
    print(f"[r09] source={r09_source} zone_sweeps={len(zones):,}", flush=True)

    print("[stage] build causal 1/3/5/10m post-sweep states", flush=True)
    checkpoints = build_post_sweep_checkpoints(
        zones,
        bars,
        cfg,
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        max_events=int(args.max_events),
        show_progress=not bool(args.no_progress),
    )
    print(f"[states] rows={len(checkpoints):,} events={checkpoints['zone_event_id'].nunique() if not checkpoints.empty else 0:,}", flush=True)

    print("[stage] next-open natural-stop long/short replay", flush=True)
    outcomes = attach_checkpoint_outcomes(checkpoints, bars, cfg, show_progress=not bool(args.no_progress))
    print("[stage] state, release, family, period and cost reports", flush=True)
    quality = data_quality(bars, zones, checkpoints, outcomes, cfg)
    audit = causal_audit(checkpoints, outcomes)
    failures = pd.concat([quality.loc[quality["status"].eq("FAIL")], audit.loc[audit["status"].eq("FAIL")]], ignore_index=True)
    if not failures.empty:
        raise RuntimeError(f"R12 fail-fast gate failed:\n{failures.to_string(index=False)}")

    design = design_table(cfg)
    distribution = state_distribution(checkpoints)
    feature_profile = state_feature_profile(checkpoints)
    long_summary = direction_outcome_summary(outcomes, cfg, "LONG")
    short_summary = direction_outcome_summary(outcomes, cfg, "SHORT")
    periods = period_stability(outcomes, cfg)
    transitions = transition_matrix(checkpoints)
    release = release_interaction(outcomes, cfg)
    families = family_timeframe_summary(outcomes, cfg)
    decisions = scorecard(long_summary, short_summary, periods, cfg)

    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "warmup_start": args.warmup_start_date,
        "research_start": research_start,
        "research_end_exclusive": research_end_exclusive,
        "r09_source": r09_source,
        "r09_dir": str(r09_dir),
        "out_dir": str(out_dir),
        "zone_events": int(zones["zone_event_id"].nunique()),
        "checkpoint_rows": int(len(checkpoints)),
        "outcome_rows": int(len(outcomes)),
        **asdict(cfg),
    }
    (out_dir / "00_manifest.json").write_text(manifest_json(manifest), encoding="utf-8")
    _write(quality, out_dir / "01_data_quality.csv")
    _write(design, out_dir / "02_frozen_design.csv")
    _write(distribution, out_dir / "03_state_distribution.csv")
    _write(feature_profile, out_dir / "04_state_feature_profile.csv")
    _write(long_summary, out_dir / "05_rejection_long_summary.csv")
    _write(short_summary, out_dir / "06_acceptance_short_summary.csv")
    _write(periods, out_dir / "07_period_stability.csv")
    _write(transitions, out_dir / "08_state_transition_matrix.csv")
    _write(release, out_dir / "09_release_interaction.csv")
    _write(families, out_dir / "10_family_timeframe_summary.csv")
    _write(decisions, out_dir / "11_candidate_scorecard.csv")
    _write(audit, out_dir / "12_causal_audit.csv")
    sample = outcomes.head(int(cfg.report_sample_rows)).copy()
    _write(sample, out_dir / "13_event_sample.csv")
    _write(_feature_table(checkpoints), out_dir / "14_checkpoint_feature_table.csv.gz", compression="gzip")
    _write(_outcome_label_table(outcomes, cfg), out_dir / "15_outcome_label_table.csv.gz", compression="gzip")
    (out_dir / "16_research_brief.md").write_text(research_brief(manifest, decisions, long_summary, short_summary), encoding="utf-8")

    if not args.skip_review_pack:
        result = finalize_research_report(
            out_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )
        print(f"[done] review_pack={result.zip_path}", flush=True)
    elapsed = time.perf_counter() - started
    counts = decisions["decision"].value_counts().to_dict() if not decisions.empty else {}
    print(f"[done] report={out_dir} elapsed={elapsed:.1f}s", flush=True)
    print(
        "[decision-summary] "
        + " ".join(f"{name}={int(counts.get(name, 0))}" for name in ("promote_to_backtest", "research_continue", "rejected")),
        flush=True,
    )
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
