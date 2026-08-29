#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R25 — fixed r0020 directional-run exhaustion reversal."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.research_common.ict_mss2.r13 import data_coverage_audit  # noqa: E402
from src.research_common.ict_mss2.r25 import (  # noqa: E402
    R25Config,
    build_r25_gate,
    build_range_run_events,
    r25_causal_audit,
    range_source_audit,
    range_temporal_quality,
    simulate_range_run_reversal,
    summarize_r25,
    summarize_r25_periods,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

TITLE = "ETH ICT MSS2 R25 r0020 Directional-Run Exhaustion"
EXPERIMENT_ID = "ETH_ICT_MSS2_R0020_DIRECTIONAL_RUN_EXHAUSTION_R25"
EDGE_ID = "RESEARCH_ONLY_R0020_DIRECTIONAL_RUN_EXHAUSTION"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r25_r0020_directional_run_exhaustion"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--end-date", default="2025-06-30 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-review-pack", action="store_true")
    return p.parse_args(argv)


def _manual_review(out: Path, trades: pd.DataFrame) -> None:
    review = out / "manual_review"
    review.mkdir(parents=True, exist_ok=True)
    closed = trades.loc[
        trades["path_status"].eq("included") & trades["entry_delay_minutes"].eq(0)
    ].copy()
    closed.sort_values("entry_time").tail(100).to_csv(review / "01_recent_100.csv", index=False)
    closed.sort_values("net_return_cost2x", ascending=False).head(50).to_csv(review / "02_best_50.csv", index=False)
    closed.sort_values("net_return_cost2x").head(50).to_csv(review / "03_worst_50.csv", index=False)
    for direction, tag in (("Long", "long"), ("Short", "short")):
        sample = closed.loc[closed["direction"].eq(direction)].sort_values("entry_time")
        sample.groupby(pd.to_datetime(sample["entry_time"]).dt.year, group_keys=False).tail(12).to_csv(
            review / f"04_{tag}_year_samples.csv", index=False
        )
    boundary = trades.loc[trades["path_status"].eq("boundary_censored")]
    boundary.to_csv(review / "05_boundary_censored.csv", index=False)
    (review / "README.md").write_text(
        "# R25 manual review\n\n"
        "Verify the four-plus same-direction r0020 run, first opposite completed confirmation, "
        "strictly-later 1m entry, run-origin target, sequence-extreme stop, and stop-first replay.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R25Config().validate()
    if pd.Timestamp(args.end_date) >= cfg.embargo_start:
        raise ValueError("R25 end date must remain before the July embargo")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r25] load bare ETH 1m through src.data_feed", flush=True)
    bars_1m = OKXDataLoader(args.symbol, "1m").fetch_data_by_date_range(
        args.warmup_start_date, args.end_date
    )
    coverage = data_coverage_audit(
        bars_1m,
        requested_start=pd.Timestamp(args.warmup_start_date),
        requested_end=pd.Timestamp(args.end_date),
    )

    print("[r25] load local-only fixed r0020 through src.data_feed", flush=True)
    range_loader = OKXRangeBarLoader(
        symbol=args.symbol,
        range_pct=cfg.range_pct,
        data_dir=args.data_dir,
        db_name=args.range_db_name,
        initialize_db=False,
    )
    if not range_loader.db_path.exists():
        raise RuntimeError(f"local range-bar cache is missing: {range_loader.db_path}")
    range_bars_loaded = range_loader.load_local_data(
        start_date=args.warmup_start_date,
        end_date=args.end_date,
        columns=(
            "bar_id", "start_ts", "end_ts", "duration_seconds", "open", "high",
            "low", "close", "direction", "notional", "delta_notional",
        ),
    )
    if range_bars_loaded.empty:
        raise RuntimeError("local r0020 cache returned no visible rows")
    source_audit = range_source_audit(range_bars_loaded, cutoff=cfg.embargo_start)
    temporal_quality = range_temporal_quality(range_bars_loaded)
    range_bars = range_bars_loaded.reset_index(drop=True)
    range_bars["end_ts"] = pd.to_datetime(range_bars["end_ts"], errors="coerce")
    range_bars = range_bars.loc[range_bars["end_ts"].lt(cfg.embargo_start)].copy()
    events = build_range_run_events(range_bars, config=cfg)
    if events.empty:
        raise RuntimeError("R25 produced no range-run events")

    pieces: list[pd.DataFrame] = []
    for delay in cfg.entry_delays_minutes:
        for split, start, end in (
            ("discovery", cfg.discovery_start, cfg.validation_start),
            ("validation", cfg.validation_start, cfg.embargo_start),
        ):
            for direction in (1, -1):
                part = simulate_range_run_reversal(
                    bars_1m,
                    events,
                    direction=direction,
                    split=split,
                    split_start=start,
                    split_end=end,
                    entry_delay_minutes=delay,
                    config=cfg,
                )
                if not part.empty:
                    pieces.append(part)
    trades = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    if trades.empty:
        raise RuntimeError("R25 produced no simulated paths")

    score = summarize_r25(trades, config=cfg)
    years, quarters = summarize_r25_periods(trades)
    gate = build_r25_gate(score, years)
    causal = r25_causal_audit(trades, events, config=cfg)
    funnel = (
        trades.groupby(["entry_delay_minutes", "research_split", "direction", "path_status"], dropna=False)
        .size().rename("rows").reset_index()
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "market": args.symbol,
        "window": [args.warmup_start_date, args.end_date],
        "range_source": {"scale": "r0020", "db": str(range_loader.db_path), "loaded_rows": len(range_bars_loaded)},
        "signal": {"min_run_bars": cfg.min_run_bars, "confirmation": "first opposite completed r0020"},
        "execution": {"entry_delays_minutes": cfg.entry_delays_minutes, "target": "run origin", "stop": "run+confirmation extreme", "same_bar": "stop_first", "time_exit": None},
        "costs": {"roundtrip": cfg.market_roundtrip_cost, "scales": cfg.cost_scales},
        "splits": {"discovery": [cfg.discovery_start, cfg.validation_start], "validation": [cfg.validation_start, cfg.embargo_start]},
        "embargo_rows_loaded": 0,
        "holdout_rows_loaded": 0,
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    coverage.to_csv(out / "01_1m_data_coverage.csv", index=False)
    source_audit.to_csv(out / "02_range_source_audit.csv", index=False)
    temporal_quality.to_csv(out / "02b_range_temporal_quality.csv", index=False)
    events.to_csv(out / "03_range_run_events.csv.gz", index=False, compression="gzip", float_format="%.17g")
    funnel.to_csv(out / "04_funnel.csv", index=False)
    trades.to_csv(out / "05_trade_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    score.to_csv(out / "06_scorecard.csv", index=False)
    years.to_csv(out / "07_years.csv", index=False)
    quarters.to_csv(out / "08_quarters.csv", index=False)
    gate.to_csv(out / "09_candidate_gate.csv", index=False)
    causal.to_csv(out / "10_causal_audit.csv", index=False)
    _manual_review(out, trades)
    (out / "R25_GENERATED_NOTE.md").write_text(
        "# R25 generated note\n\nFixed r0020 run-exhaustion reversal. July and holdout are absent.\n",
        encoding="utf-8",
    )
    if not args.skip_review_pack:
        finalize_research_report(
            out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE
        )
    print(score.to_string(index=False), flush=True)
    print(gate.to_string(index=False), flush=True)
    print(causal.to_string(index=False), flush=True)
    print(f"[r25] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
