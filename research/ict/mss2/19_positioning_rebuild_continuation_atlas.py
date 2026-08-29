#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R19 — positioning rebuild / continuation-resumption path atlas."""
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

from src.data_feed.binance_futures_metrics_loader import BinanceFuturesMetricsLoader  # noqa: E402
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.r13 import data_coverage_audit  # noqa: E402
from src.research_common.ict_mss2.r18 import r18_data_quality_audit  # noqa: E402
from src.research_common.ict_mss2.r19 import (  # noqa: E402
    R19Config,
    build_positioning_rebuild_events,
    build_positioning_rebuild_paths,
    r19_causal_audit,
    summarize_r19_funnel,
    summarize_r19_paths,
    summarize_r19_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "19.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_POSITIONING_REBUILD_CONTINUATION_ATLAS_R19"
EDGE_ID = "RESEARCH_ONLY_BINANCE_PROXY_POSITIONING_REBUILD_LONG_SHORT"
TITLE = "ETH ICT MSS2 R19 Positioning Rebuild Continuation-Resumption Atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r19_positioning_rebuild_continuation_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--binance-symbol", default="ETHUSDT")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-08-15 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _contract(cfg: R19Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "build", "value": "prior causal 1h directional OKX price move with rising Binance base OI"},
            {"field": "release", "value": "first causal 5m base-OI sign transition from >=0 to <0"},
            {"field": "rebuild", "value": "first subsequent causal 5m base-OI observation >=0"},
            {"field": "continuation", "value": "rebuild-time completed OKX 5m close beyond frozen release-bar high/low"},
            {"field": "rebuild_window_minutes", "value": cfg.rebuild_window_minutes},
            {"field": "entry", "value": "first OKX 1m open at or after all rebuild information is available"},
            {"field": "stop", "value": f"release-through-rebuild extreme plus {cfg.stop_buffer_atr:g}x causal 5m ATR(12)"},
            {"field": "maximum_stop_distance_pct", "value": cfg.max_stop_distance_pct},
            {"field": "primary_target", "value": "1.0x causal rebuild-time 1h high-low range from entry"},
            {"field": "fixed_r_targets", "value": "|".join(f"{x:g}" for x in cfg.fixed_r_targets)},
            {"field": "path_horizon_minutes", "value": cfg.path_horizon_minutes},
            {"field": "same_bar_policy", "value": "stop_first"},
            {"field": "holdout_unsealed", "value": 0},
        ]
    )


def _manual_review(out: Path, events: pd.DataFrame, paths: pd.DataFrame) -> None:
    directory = out / "manual_review"
    directory.mkdir(parents=True, exist_ok=True)
    executable = events.loc[events["setup_status"].eq("executable")].copy()
    if not executable.empty:
        executable.sort_values("entry_time", kind="stable").tail(80).to_csv(
            directory / "01_recent_80_executable_setups.csv", index=False, encoding="utf-8-sig"
        )
        for (year, direction), part in executable.groupby(
            [pd.to_datetime(executable["entry_time"]).dt.year, "direction"], sort=True
        ):
            part.sort_values("entry_time", kind="stable").tail(20).to_csv(
                directory / f"02_{int(year)}_{str(direction).lower()}_recent_20.csv",
                index=False,
                encoding="utf-8-sig",
            )
    if not paths.empty:
        comparator = paths.loc[paths["target_model"].eq("R2")].copy()
        comparator.sort_values("net_return_cost2x", kind="stable").head(50).to_csv(
            directory / "03_r2_worst_50.csv", index=False, encoding="utf-8-sig"
        )
        comparator.sort_values("net_return_cost2x", ascending=False, kind="stable").head(50).to_csv(
            directory / "04_r2_best_50.csv", index=False, encoding="utf-8-sig"
        )
    (directory / "README.md").write_text(
        "# R19 manual review\n\n"
        "Verify the directional 1h price/base-OI build, first release sign transition, uninterrupted "
        "negative-OI episode, first rebuild observation, original-direction release-range break, and "
        "next-observable OKX 1m entry. Binance OI is a cross-exchange proxy. Rankings are diagnostic only.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R19Config().validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r19] load OKX bare 1m K through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
        args.warmup_start_date, args.end_date
    )
    coverage = data_coverage_audit(
        bars,
        requested_start=pd.Timestamp(args.warmup_start_date),
        requested_end=pd.Timestamp(args.end_date),
    )
    covered = coverage.loc[coverage["check"].eq("requested_end_covered"), "value"]
    if covered.empty or int(covered.iloc[0]) != 1:
        raise RuntimeError("R19 requested end is not covered by bare OKX 1m data")

    print("[r19] load causal Binance 5m positioning proxy through src.data_feed", flush=True)
    loader = BinanceFuturesMetricsLoader(symbol=args.binance_symbol, data_dir=args.data_dir)
    oi = loader.load_relative_features(
        args.warmup_start_date,
        args.end_date,
        windows=("5m", "1h"),
        publication_lag=cfg.publication_lag,
        baseline_tolerance=f"{int(cfg.baseline_tolerance_seconds)}s",
        index_mode="none",
    )
    if oi.empty:
        raise RuntimeError("R19 Binance positioning cache is empty")
    oi_coverage = loader.coverage()
    oi_days = loader.coverage_by_day()

    print("[r19] build frozen build -> release -> first rebuild continuation events", flush=True)
    events, seal, engineering = build_positioning_rebuild_events(bars, oi, config=cfg)
    print("[r19] calculate discovery/validation-only exact 1m first passage", flush=True)
    paths = build_positioning_rebuild_paths(bars, events, config=cfg)
    funnel = summarize_r19_funnel(events)
    scorecard = summarize_r19_paths(paths, config=cfg)
    years = summarize_r19_years(paths)
    pre_embargo_bars = bars.loc[pd.to_datetime(bars.index) < cfg.embargo_start]
    quality = r18_data_quality_audit(pre_embargo_bars, oi, oi_days, config=cfg)
    audit = r19_causal_audit(events, paths, config=cfg)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "execution_market": args.symbol,
        "positioning_proxy": f"Binance USD-M {args.binance_symbol}",
        "requested_start_date": args.warmup_start_date,
        "requested_end_date": args.end_date,
        "binance_cache": {
            "rows": oi_coverage.rows,
            "start_project_time": str(oi_coverage.start),
            "end_project_time": str(oi_coverage.end),
            "complete_days": oi_coverage.complete_days,
            "partial_days": oi_coverage.partial_days,
            "requested_end_covered": bool(oi_coverage.end is not None and oi_coverage.end >= pd.Timestamp(args.end_date)),
        },
        "splits": {
            "discovery": "2023-01-01 through 2024-12-31",
            "validation": "2025-01-01 through 2025-06-30",
            "embargo": "2025-07-01 through 2025-07-31",
            "holdout_start": str(cfg.holdout_start),
            "holdout_unsealed": False,
        },
        "event_sequence": [
            "prior completed 1h directional price move plus rising Binance base OI",
            "first 5m base-OI release transition",
            "first nonnegative base-OI rebuild within 60 minutes",
            "completed OKX 5m close breaks frozen release bar in original direction",
            "next observable OKX 1m open",
        ],
        "excluded_admission_data": ["OI USD", "ratios", "funding", "magnitude thresholds", "future OI", "oracle turning points"],
        "risk": {
            "rebuild_window_minutes": cfg.rebuild_window_minutes,
            "atr_window_5m": cfg.atr_window_5m,
            "stop_buffer_atr": cfg.stop_buffer_atr,
            "max_stop_distance_pct": cfg.max_stop_distance_pct,
        },
        "targets": ["H0_1H_VOLATILITY_RANGE", *[f"R{x:g}" for x in cfg.fixed_r_targets]],
        "path_horizon_minutes": cfg.path_horizon_minutes,
        "same_bar_policy": "stop_first",
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "strategy_status": "mechanism/path atlas only; no automatic promotion",
        "known_external_validation_limit": "Other repository projects inspected overlapping 2025-2026 data; eventual live approval needs new forward data.",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_okx_data_coverage_audit.csv", index=False)
    quality.to_csv(out / "02_binance_and_join_data_quality.csv", index=False)
    seal.to_csv(out / "03_holdout_seal.csv", index=False)
    _contract(cfg).to_csv(out / "04_precommitted_event_contract.csv", index=False)
    funnel.to_csv(out / "05_setup_funnel.csv", index=False)
    events.to_csv(out / "06_causal_event_table.csv.gz", index=False, compression="gzip", float_format="%.17g")
    paths.to_csv(out / "07_first_passage_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    scorecard.to_csv(out / "08_direction_target_scorecard.csv", index=False)
    years.to_csv(out / "09_direction_target_years.csv", index=False)
    audit.to_csv(out / "10_causal_audit.csv", index=False)
    engineering.to_csv(out / "11_engineering_audit.csv", index=False)
    (out / "R19_GENERATED_NOTE.md").write_text(
        "# R19 generated note\n\nThis is the frozen positioning-rebuild continuation-resumption path atlas. "
        "Binance OI is a cross-exchange proxy, Long and Short remain separate, July/holdout economics are absent, "
        "and the 24-hour horizon is diagnostic.\n",
        encoding="utf-8",
    )
    _manual_review(out, events, paths)
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r19] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
