#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R18 — independent causal Binance-positioning unwind path atlas."""
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
from src.research_common.ict_mss2.r18 import (  # noqa: E402
    R18Config,
    build_positioning_unwind_events,
    build_positioning_unwind_paths,
    r18_causal_audit,
    r18_data_quality_audit,
    summarize_r18_funnel,
    summarize_r18_paths,
    summarize_r18_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "18.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_INDEPENDENT_POSITIONING_UNWIND_ATLAS_R18"
EDGE_ID = "RESEARCH_ONLY_BINANCE_PROXY_POSITIONING_UNWIND_LONG_SHORT"
TITLE = "ETH ICT MSS2 R18 Independent Positioning-Unwind Path Atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r18_positioning_unwind_path_atlas"


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


def _contract(cfg: R18Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "build", "value": "prior causal 1h price direction with rising Binance base OI"},
            {"field": "release", "value": "first causal 5m Binance base-OI change from >=0 to <0"},
            {"field": "stabilization", "value": "completed OKX 5m close beyond immediately prior 5m high/low"},
            {"field": "entry", "value": "first OKX 1m open at or after all information is available"},
            {"field": "publication_lag", "value": str(cfg.publication_lag)},
            {"field": "gap_rule", "value": f"{cfg.metric_min_gap} through {cfg.metric_max_gap}; no interpolation"},
            {"field": "stop", "value": f"two-bar stabilization extreme plus {cfg.stop_buffer_atr:g}x causal 5m ATR(12)"},
            {"field": "maximum_stop_distance_pct", "value": cfg.max_stop_distance_pct},
            {"field": "structural_target", "value": "opposite extreme of frozen 1h build range"},
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
        "# R18 manual review\n\n"
        "Verify that the Binance OI observation is treated only as a cross-exchange proxy, its publication "
        "lag precedes entry, the immediately prior 1h price/OI state is directional build, the OI release "
        "is a first sign transition, and the OKX 5m close genuinely reacquires the prior extreme. "
        "R2 rankings are diagnostic only; R18 does not select a target or promote a strategy.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R18Config().validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r18] load OKX bare 1m K through src.data_feed", flush=True)
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
        raise RuntimeError("R18 requested end is not covered by bare OKX 1m data")

    print("[r18] load causal Binance 5m positioning proxy through src.data_feed", flush=True)
    oi_loader = BinanceFuturesMetricsLoader(symbol=args.binance_symbol, data_dir=args.data_dir)
    oi = oi_loader.load_relative_features(
        args.warmup_start_date,
        args.end_date,
        windows=("5m", "1h"),
        publication_lag=cfg.publication_lag,
        baseline_tolerance=f"{int(cfg.baseline_tolerance_seconds)}s",
        index_mode="none",
    )
    if oi.empty:
        raise RuntimeError("R18 Binance positioning cache is empty")
    oi_coverage = oi_loader.coverage()
    oi_days = oi_loader.coverage_by_day()

    print("[r18] build frozen all-market build -> release -> reacquisition events", flush=True)
    events, seal, engineering = build_positioning_unwind_events(bars, oi, config=cfg)
    print("[r18] calculate discovery/validation-only exact 1m first passage", flush=True)
    paths = build_positioning_unwind_paths(bars, events, config=cfg)
    funnel = summarize_r18_funnel(events)
    scorecard = summarize_r18_paths(paths, config=cfg)
    years = summarize_r18_years(paths)
    pre_embargo_bars = bars.loc[pd.to_datetime(bars.index) < cfg.embargo_start]
    quality = r18_data_quality_audit(pre_embargo_bars, oi, oi_days, config=cfg)
    audit = r18_causal_audit(events, paths, config=cfg)

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
            "first completed 5m Binance base-OI release sign transition",
            "completed OKX 5m close reacquires prior bar extreme",
            "next observable OKX 1m open",
        ],
        "excluded_admission_data": [
            "OI USD",
            "taker ratio",
            "top-trader and global account ratios",
            "funding",
            "future OI",
            "oracle turning points",
        ],
        "risk": {
            "atr_window_5m": cfg.atr_window_5m,
            "stop_buffer_atr": cfg.stop_buffer_atr,
            "max_stop_distance_pct": cfg.max_stop_distance_pct,
        },
        "targets": ["H0_1H_BUILD_RANGE", *[f"R{x:g}" for x in cfg.fixed_r_targets]],
        "path_horizon_minutes": cfg.path_horizon_minutes,
        "same_bar_policy": "stop_first",
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "strategy_status": "mechanism/path atlas only; no automatic promotion",
        "known_external_validation_limit": (
            "Other repository projects inspected overlapping 2025-2026 data; eventual live approval needs new forward data."
        ),
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
    (out / "R18_GENERATED_NOTE.md").write_text(
        "# R18 generated note\n\n"
        "This is the frozen independent positioning-unwind mechanism/path atlas. Binance OI is a "
        "cross-exchange proxy, Long and Short remain separate, the 24-hour horizon is diagnostic, "
        "and the existing holdout remains sealed.\n",
        encoding="utf-8",
    )
    _manual_review(out, events, paths)
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r18] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
