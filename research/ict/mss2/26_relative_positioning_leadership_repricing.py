#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R26 — causal relative-positioning leadership repricing study."""
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
from src.research_common.ict_mss2.r26 import (  # noqa: E402
    R26Config,
    build_positioning_leadership_events,
    build_positioning_leadership_paths,
    r26_causal_audit,
    r26_data_quality_audit,
    summarize_r26_funnel,
    summarize_r26_paths,
    summarize_r26_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "26.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_RELATIVE_POSITIONING_LEADERSHIP_REPRICING_R26"
EDGE_ID = "RESEARCH_ONLY_TOP_TRADER_GLOBAL_ACCOUNT_LEADERSHIP_CROSS"
TITLE = "ETH ICT MSS2 R26 Relative Positioning Leadership Repricing"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r26_relative_positioning_leadership_repricing"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--binance-symbol", default="ETHUSDT")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _contract(cfg: R26Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "relative_spread", "value": "top_trader_position_long_share - global_account_long_share"},
            {"field": "long_cross", "value": "prior spread <= 0 and current spread > 0"},
            {"field": "short_cross", "value": "prior spread >= 0 and current spread < 0"},
            {"field": "publication_lag", "value": str(cfg.publication_lag)},
            {"field": "metric_gap", "value": f"{cfg.metric_min_gap} through {cfg.metric_max_gap}; no interpolation"},
            {"field": "confirmation_window", "value": str(cfg.confirmation_window)},
            {"field": "price_confirmation", "value": "first completed 5m close through prior 5m high/low while spread retains sign"},
            {"field": "entry", "value": "first OKX 1m open at or after all information is available"},
            {"field": "stop", "value": f"two-bar confirmation extreme plus {cfg.stop_buffer_atr:g}x causal ATR(12)"},
            {"field": "maximum_stop_distance_pct", "value": cfg.max_stop_distance_pct},
            {"field": "structural_target", "value": "direction-side extreme of cross-time completed 1h range"},
            {"field": "diagnostic_targets", "value": "|".join(f"R{x:g}" for x in cfg.fixed_r_targets)},
            {"field": "same_bar_policy", "value": "stop_first"},
            {"field": "input_cutoff", "value": str(cfg.embargo_start)},
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
    if not paths.empty and "gross_return" in paths:
        primary = paths.loc[
            paths["target_model"].eq("H0_CROSS_TIME_1H_RANGE")
            & paths["position_selected"].eq(True)
            & paths["gross_return"].notna()
        ].copy()
        primary.sort_values("net_return_cost2x", kind="stable").head(50).to_csv(
            directory / "03_structural_worst_50.csv", index=False, encoding="utf-8-sig"
        )
        primary.sort_values("net_return_cost2x", ascending=False, kind="stable").head(50).to_csv(
            directory / "04_structural_best_50.csv", index=False, encoding="utf-8-sig"
        )
    (directory / "README.md").write_text(
        "# R26 manual review\n\n"
        "Verify that the Binance ratio cross is publication-lagged, gap-safe, and independent of base OI; "
        "the cross-time one-hour range is frozen before confirmation; the relative spread retains its new "
        "sign; the completed OKX five-minute close confirms direction; and entry occurs at the next eligible "
        "one-minute open. Structural rankings are review aids only and cannot select a rule.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R26Config().validate()
    requested_end = pd.Timestamp(args.end_date)
    if requested_end >= cfg.embargo_start:
        raise ValueError("R26 physically forbids loading July 2025 or the sealed holdout")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r26] load visible OKX bare 1m K through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
        args.warmup_start_date, args.end_date
    )
    coverage = data_coverage_audit(
        bars,
        requested_start=pd.Timestamp(args.warmup_start_date),
        requested_end=requested_end,
    )
    covered = coverage.loc[coverage["check"].eq("requested_end_covered"), "value"]
    if covered.empty or int(covered.iloc[0]) != 1:
        raise RuntimeError("R26 requested end is not covered by bare OKX 1m data")

    print("[r26] load visible Binance ratio fields through src.data_feed", flush=True)
    metric_loader = BinanceFuturesMetricsLoader(symbol=args.binance_symbol, data_dir=args.data_dir)
    metrics = metric_loader.load_metrics(
        args.warmup_start_date,
        args.end_date,
        publication_lag=cfg.publication_lag,
        index_mode="none",
    )
    if metrics.empty:
        raise RuntimeError("R26 Binance metrics cache is empty")
    metric_coverage = metric_loader.coverage()
    metric_days = metric_loader.coverage_by_day(args.warmup_start_date, args.end_date)

    print("[r26] build frozen leadership-cross and price-confirmation events", flush=True)
    events, seal, engineering = build_positioning_leadership_events(bars, metrics, config=cfg)
    print("[r26] replay exact visible split paths with non-overlap", flush=True)
    paths = build_positioning_leadership_paths(bars, events, config=cfg)
    funnel = summarize_r26_funnel(events, paths)
    scorecard = summarize_r26_paths(paths, config=cfg)
    years = summarize_r26_years(paths)
    quality = r26_data_quality_audit(bars, metrics, metric_days, config=cfg)
    audit = r26_causal_audit(events, paths, config=cfg)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "execution_market": args.symbol,
        "positioning_proxy": f"Binance USD-M {args.binance_symbol}",
        "requested_start_date": args.warmup_start_date,
        "requested_end_date": args.end_date,
        "physical_cutoff": str(cfg.embargo_start),
        "binance_cache_metadata": {
            "rows_all_local_periods": metric_coverage.rows,
            "start_project_time": str(metric_coverage.start),
            "end_project_time": str(metric_coverage.end),
            "loaded_rows_visible_only": int(len(metrics)),
        },
        "splits": {
            "discovery": "2023-01-01 through 2024-12-31",
            "validation": "2025-01-01 through 2025-06-30",
            "embargo_start": str(cfg.embargo_start),
            "holdout_start": str(cfg.holdout_start),
            "holdout_unsealed": False,
        },
        "event_sequence": [
            "top-trader position share crosses global-account share",
            "relative spread retains new sign for at most one hour",
            "first same-direction completed OKX 5m close through prior bar extreme",
            "next observable OKX 1m open",
        ],
        "excluded_admission_data": [
            "base OI level or change",
            "OI USD",
            "taker ratio",
            "funding or basis",
            "liquidity sweeps",
            "session or volatility filters",
            "future or oracle fields",
        ],
        "targets": ["H0_CROSS_TIME_1H_RANGE", *[f"R{x:g}" for x in cfg.fixed_r_targets]],
        "path_horizon_minutes": cfg.path_horizon_minutes,
        "position_overlap": "separate non-overlapping simulation per split/direction/target",
        "same_bar_policy": "stop_first",
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "strategy_status": "mechanism/path study only; promotion requires frozen gate",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_okx_data_coverage_audit.csv", index=False)
    quality.to_csv(out / "02_binance_ratio_data_quality.csv", index=False)
    seal.to_csv(out / "03_holdout_seal.csv", index=False)
    _contract(cfg).to_csv(out / "04_precommitted_event_contract.csv", index=False)
    funnel.to_csv(out / "05_setup_funnel.csv", index=False)
    events.to_csv(out / "06_causal_event_table.csv.gz", index=False, compression="gzip", float_format="%.17g")
    paths.to_csv(out / "07_first_passage_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    scorecard.to_csv(out / "08_direction_target_scorecard.csv", index=False)
    years.to_csv(out / "09_direction_target_years.csv", index=False)
    audit.to_csv(out / "10_causal_audit.csv", index=False)
    engineering.to_csv(out / "11_engineering_audit.csv", index=False)
    (out / "R26_GENERATED_NOTE.md").write_text(
        "# R26 generated note\n\n"
        "This study physically loads only data before 2025-07-01. Binance ratios are a cross-exchange "
        "positioning proxy; all price execution remains OKX ETH-USDT-SWAP. Long and Short stay separate, "
        "fixed-R paths are diagnostic, and the holdout remains sealed.\n",
        encoding="utf-8",
    )
    _manual_review(out, events, paths)
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r26] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
