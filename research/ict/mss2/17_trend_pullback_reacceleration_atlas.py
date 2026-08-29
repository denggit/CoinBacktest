#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R17 — causal trend-pullback reclaim/re-acceleration path atlas."""
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
from src.research_common.ict_mss2.r13 import data_coverage_audit  # noqa: E402
from src.research_common.ict_mss2.r17 import (  # noqa: E402
    R17Config,
    build_first_passage_paths,
    build_pullback_setup_atlas,
    r17_causal_audit,
    summarize_path_models,
    summarize_path_years,
    summarize_setup_funnel,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "17.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_TREND_PULLBACK_REACCELERATION_ATLAS_R17"
EDGE_ID = "RESEARCH_ONLY_TREND_PULLBACK_REACCELERATION_LONG_SHORT"
TITLE = "ETH ICT MSS2 R17 Trend Pullback Re-acceleration Path Atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r17_trend_pullback_reacceleration_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-08-15 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _contract(cfg: R17Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "trend_state", "value": "aligned causal 1D and 4H order-1 HH/HL or LH/LL"},
            {"field": "pullback", "value": "causally confirmed 30m order-1 pivot against trend"},
            {"field": "reclaim", "value": "15m close beyond 30m pivot-bar range"},
            {"field": "reacceleration", "value": "later 5m close beyond reclaim-bar high/low"},
            {"field": "entry", "value": "1m open at closed-5m availability boundary"},
            {"field": "stop", "value": f"30m pivot extreme plus {cfg.stop_buffer_atr:g}x causal 30m ATR"},
            {"field": "maximum_stop_distance_pct", "value": cfg.max_stop_distance_pct},
            {"field": "structural_target", "value": "latest causally confirmed 4H pivot extreme in trend direction"},
            {"field": "fixed_r_targets", "value": "|".join(f"{x:g}" for x in cfg.fixed_r_targets)},
            {"field": "setup_expiry_minutes", "value": cfg.setup_expiry_minutes},
            {"field": "path_horizon_minutes", "value": cfg.path_horizon_minutes},
            {"field": "same_bar_policy", "value": "stop_first"},
            {"field": "holdout_unsealed", "value": 0},
        ]
    )


def _manual_review(out: Path, setups: pd.DataFrame, paths: pd.DataFrame) -> None:
    directory = out / "manual_review"
    directory.mkdir(parents=True, exist_ok=True)
    executable = setups.loc[setups["setup_status"].eq("executable")].copy()
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
        "# R17 manual review\n\n"
        "Check that 1D/4H trend structure was already visible, the 30m pivot is a genuine local pullback, "
        "the 15m reclaim and later 5m break occur in sequence, and entry is the next 1m open. "
        "R2 best/worst files are diagnostic only; R17 does not select a target or promote a strategy.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R17Config().validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("[r17] load bare 1m K through src.data_feed", flush=True)
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
        raise RuntimeError("R17 requested end is not covered by bare 1m data")

    print("[r17] causal 1D/4H trend -> 30m pullback -> 15m reclaim -> 5m re-acceleration", flush=True)
    setups, seal, engineering = build_pullback_setup_atlas(bars, config=cfg)
    print("[r17] exact 1m first-passage paths", flush=True)
    paths = build_first_passage_paths(bars, setups, config=cfg)
    funnel = summarize_setup_funnel(setups)
    scorecard = summarize_path_models(paths, config=cfg)
    years = summarize_path_years(paths)
    audit = r17_causal_audit(setups, paths, config=cfg)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "requested_start_date": args.warmup_start_date,
        "requested_end_date": args.end_date,
        "splits": {
            "discovery": "2023-01-01 through 2024-12-31",
            "validation": "2025-01-01 through 2025-06-30",
            "embargo": "2025-07-01 through 2025-07-31",
            "holdout_start": str(cfg.holdout_start),
            "holdout_unsealed": False,
        },
        "event_sequence": [
            "aligned causal 1D+4H structural trend",
            "causally confirmed 30m counter-trend pivot",
            "15m pivot-range reclaim close",
            "later 5m reclaim-bar break close",
            "next observable 1m open",
        ],
        "risk": {
            "stop_buffer_atr": cfg.stop_buffer_atr,
            "max_stop_distance_pct": cfg.max_stop_distance_pct,
        },
        "targets": ["H0_4H_STRUCTURAL", *[f"R{x:g}" for x in cfg.fixed_r_targets]],
        "path_horizon_minutes": cfg.path_horizon_minutes,
        "same_bar_policy": "stop_first",
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "strategy_status": "mechanism/path atlas only; no automatic promotion",
        "known_external_validation_limit": (
            "Other repository projects inspected overlapping 2025-2026 data; any eventual live approval needs new forward data."
        ),
    }
    (out / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    coverage.to_csv(out / "01_actual_data_coverage_audit.csv", index=False)
    seal.to_csv(out / "02_holdout_seal.csv", index=False)
    _contract(cfg).to_csv(out / "03_precommitted_event_contract.csv", index=False)
    funnel.to_csv(out / "04_setup_funnel.csv", index=False)
    setups.to_csv(out / "05_setup_atlas.csv.gz", index=False, compression="gzip")
    paths.to_csv(out / "06_first_passage_paths.csv.gz", index=False, compression="gzip")
    scorecard.to_csv(out / "07_direction_target_scorecard.csv", index=False)
    years.to_csv(out / "08_direction_target_years.csv", index=False)
    audit.to_csv(out / "09_causal_audit.csv", index=False)
    engineering.to_csv(out / "10_engineering_audit.csv", index=False)
    (out / "R17_GENERATED_NOTE.md").write_text(
        "# R17 generated note\n\n"
        "This output is the precommitted trend-pullback mechanism/path atlas. Long and Short remain separate. "
        "The 72-hour horizon exit is diagnostic and is not a proposed final strategy exit. Holdout is sealed.\n",
        encoding="utf-8",
    )
    _manual_review(out, setups, paths)
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r17] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

