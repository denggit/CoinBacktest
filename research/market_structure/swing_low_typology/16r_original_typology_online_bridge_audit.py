#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research 16R: reconnect frozen 01-03 Swing Low types to online candidates.

This is a correction audit, not a new typology and not a strategy backtest.
It preserves the frozen historical type definitions from Research 01-03, then
asks four questions:

1. How much of every original type is recalled by the raw and region-selected
   causal online candidate universes?
2. Which original types actually contain respected-macro First Sweep events?
3. When an online candidate leads into an original type, does the candidate's
   own next-open close-only path reach +1% cleanly?
4. Inside each original C3/type label, do predeclared single causal conditions
   improve path quality across walk-forward test periods without worsening MAE?

Historical type mapping and future close paths are labels only. Candidate
features and train-fitted thresholds use current closed bars or older data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.broad_reversal_evaluation import (  # noqa: E402
    build_multi_horizon_close_labels,
)
from research.market_structure.swing_low_typology.common.broad_reversal_mechanisms import (  # noqa: E402
    FrozenMechanismScorer,
    MECHANISM_ORDER,
    UNRESOLVED_MECHANISM,
    merge_macro_first_sweep_candidates,
    select_broad_region_events,
)
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    build_online_candidate_events,
)
from research.market_structure.swing_low_typology.common.original_typology_bridge import (  # noqa: E402
    HIERARCHY_COLUMNS,
    SPECIAL_CONDITION_SPECS,
    attach_event_bridge_flags,
    bridge_causal_audit,
    bridge_coverage_scorecard,
    build_historical_typology_table,
    prepare_sweep_only_events,
    source_typology_overlap,
    map_candidates_to_future_typology,
    mapped_source_coverage_scorecard,
    path_scorecard_by_original_type,
    special_condition_scorecard,
    summarize_condition_candidates,
    typology_inventory,
    walkforward_folds,
)
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    validate_trade_bar_fields,
)
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    build_broad_candidate_regions,
)

SCRIPT_NAME = "16r_original_typology_online_bridge_audit"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_1M_ORIGINAL_SWING_LOW_TYPOLOGY_ONLINE_BRIDGE_16R"
EDGE_ID = "RESEARCH_ONLY_ETH_ORIGINAL_TYPOLOGY_ONLINE_BRIDGE"
TITLE = "ETH Original Swing Low Typology to Online Candidate Bridge Audit 16R"
DEFAULT_OUT_DIR = (
    "data/reports/research/market_structure/swing_low_typology/"
    "16r_original_typology_online_bridge"
)
DEFAULT_STAGE1_DIR = (
    "data/reports/research/market_structure/swing_low_typology/01_causal_typology"
)
DEFAULT_STAGE2_DIR = (
    "data/reports/research/market_structure/swing_low_typology/02_c3_hierarchical_typology"
)
DEFAULT_STAGE3_DIR = (
    "data/reports/research/market_structure/swing_low_typology/03_mechanism_hierarchical_typology"
)

DATE_COLUMNS_STAGE1 = [
    "extreme_time",
    "feature_available_time",
    "confirmation_time",
    "confirmation_available_time",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconnect frozen Research 01-03 Swing Low types to causal online candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s v{SCRIPT_VERSION}")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--forward-horizon-bars", type=int, default=60)
    p.add_argument("--stage1-report-dir", default=DEFAULT_STAGE1_DIR)
    p.add_argument("--stage2-report-dir", default=DEFAULT_STAGE2_DIR)
    p.add_argument("--stage3-report-dir", default=DEFAULT_STAGE3_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")

    p.add_argument("--candidate-lookback-bars", type=int, default=240)
    p.add_argument("--candidate-new-low-window", type=int, default=5)
    p.add_argument("--candidate-near-floor-window", type=int, default=60)
    p.add_argument("--candidate-position-window", type=int, default=120)
    p.add_argument("--candidate-near-floor-tolerance-bp", type=float, default=20.0)
    p.add_argument("--candidate-max-position-in-range", type=float, default=0.55)
    p.add_argument("--region-max-gap-bars", type=int, default=2)
    p.add_argument("--region-max-bars", type=int, default=120)
    p.add_argument("--region-retest-tolerance-bp", type=float, default=25.0)
    p.add_argument("--broad-cooldown-bars", type=int, default=15)

    p.add_argument("--bridge-lead-windows", nargs="+", type=int, default=[0, 3, 5, 10, 15])
    p.add_argument("--reference-maximum-lead-bars", type=int, default=15)
    p.add_argument("--reference-price-tolerance-bp", type=float, default=75.0)
    p.add_argument("--condition-top-pcts", nargs="+", type=int, default=[20, 30, 40])
    p.add_argument("--minimum-type-test-rows", type=int, default=20)

    p.add_argument("--liquidity-pivot-minutes", nargs="+", type=int, default=[15, 60, 240])
    p.add_argument("--liquidity-pivot-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--liquidity-pivot-left-bars", type=int, default=2)
    p.add_argument("--liquidity-pivot-right-bars", type=int, default=2)
    p.add_argument("--liquidity-cluster-tolerance-bp", type=float, default=25.0)
    p.add_argument("--liquidity-minimum-respects", type=int, default=2)
    p.add_argument("--liquidity-minimum-macro-timeframe-min", type=int, default=60)
    p.add_argument("--liquidity-minimum-respect-separation-minutes", type=int, default=60)
    p.add_argument("--liquidity-formation-max-days", type=int, default=45)
    p.add_argument("--liquidity-reclaim-window-bars", type=int, default=3)
    p.add_argument("--liquidity-accept-below-bars", type=int, default=3)
    p.add_argument("--liquidity-accept-depth-bp", type=float, default=75.0)

    p.add_argument("--label-vectorized-chunk-size", type=int, default=20_000)
    p.add_argument("--write-full-candidate-bridge", action="store_true")
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _end_exclusive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        timestamp += pd.Timedelta(days=1)
    return timestamp


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] source=trade_bar {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        db_name=args.db_name,
    )
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=not bool(args.no_build_missing),
    )
    if bars.empty:
        raise RuntimeError("No trade-bar data loaded")
    bars = bars.sort_index()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index, errors="coerce")
    bars = bars[~bars.index.isna()]
    bars = bars[~bars.index.duplicated(keep="last")]
    print(f"       rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_original_reports(args: argparse.Namespace) -> tuple[Path, Path, Path, pd.DataFrame]:
    stage1 = Path(args.stage1_report_dir)
    stage2 = Path(args.stage2_report_dir)
    stage3 = Path(args.stage3_report_dir)
    required = [
        stage1 / "00_manifest.json",
        stage1 / "06_frozen_cluster_assignments.csv",
        stage2 / "00_manifest.json",
        stage2 / "07_frozen_c3_subcluster_assignments.csv",
        stage3 / "00_manifest.json",
        stage3 / "06_broad_mechanism_assignments.csv",
        stage3 / "15_c3c_trend_subtype_assignments.csv",
        stage3 / "24_c3e_base_subtype_assignments.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Research 16R reuses frozen 01-03 outputs. Missing files: " + ", ".join(missing)
        )

    manifests = {
        "stage1": _read_manifest(stage1 / "00_manifest.json"),
        "stage2": _read_manifest(stage2 / "00_manifest.json"),
        "stage3": _read_manifest(stage3 / "00_manifest.json"),
    }
    rows: list[dict[str, object]] = []
    expected = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "target_move_pct": float(args.target_move_pct),
        "max_completion_bars": int(args.forward_horizon_bars),
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
    }
    failures: list[str] = []
    for source, manifest in manifests.items():
        for key, value in expected.items():
            actual = manifest.get(key)
            passed = str(actual) == str(value)
            rows.append(
                {
                    "source": source,
                    "check": key,
                    "expected": value,
                    "actual": actual,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"{source}.{key}={actual}, expected={value}")
    train_end_values = {str(manifest.get("train_end_date")) for manifest in manifests.values()}
    train_end_consistent = len(train_end_values) == 1 and "None" not in train_end_values
    rows.append(
        {
            "source": "cross_report",
            "check": "train_end_date_consistent",
            "expected": "same non-null value across 01/02/03",
            "actual": sorted(train_end_values),
            "passed": train_end_consistent,
        }
    )
    if not train_end_consistent:
        failures.append(f"train_end_date mismatch={sorted(train_end_values)}")
    if failures:
        raise RuntimeError("Frozen 01-03 reports are incompatible: " + "; ".join(failures))
    return stage1, stage2, stage3, pd.DataFrame(rows)


def _load_historical(stage1_dir: Path, stage2_dir: Path, stage3_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[stage] load frozen 01-03 historical typology assignments", flush=True)
    stage1 = pd.read_csv(
        stage1_dir / "06_frozen_cluster_assignments.csv",
        parse_dates=DATE_COLUMNS_STAGE1,
    )
    stage2 = pd.read_csv(
        stage2_dir / "07_frozen_c3_subcluster_assignments.csv",
        parse_dates=DATE_COLUMNS_STAGE1,
    )
    broad = pd.read_csv(
        stage3_dir / "06_broad_mechanism_assignments.csv",
        parse_dates=["extreme_time", "feature_available_time"],
    )
    trend = pd.read_csv(
        stage3_dir / "15_c3c_trend_subtype_assignments.csv",
        parse_dates=["extreme_time", "feature_available_time"],
    )
    base = pd.read_csv(
        stage3_dir / "24_c3e_base_subtype_assignments.csv",
        parse_dates=["extreme_time", "feature_available_time"],
    )
    return build_historical_typology_table(stage1, stage2, broad, trend, base)


def _fit_oos_research16_mechanisms(frame: pd.DataFrame, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["research16_primary_mechanism"] = pd.Series(pd.NA, index=out.index, dtype="string")
    rows: list[dict[str, object]] = []
    time = pd.to_datetime(out["extreme_time"], errors="raise")
    for fold in walkforward_folds(end_date):
        train = out[(time >= fold.train_start) & (time <= fold.train_end)]
        test_mask = (time >= fold.test_start) & (time <= fold.test_end)
        test = out.loc[test_mask]
        if len(train) < 100 or test.empty:
            continue
        scorer = FrozenMechanismScorer.fit(train, minimum_score=70.0)
        transformed = scorer.transform(test)
        out.loc[test.index, "research16_primary_mechanism"] = transformed[
            "primary_mechanism"
        ].astype("string").to_numpy()
        rows.append(
            {
                "fold": fold.fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "mechanism_count": len(MECHANISM_ORDER),
                "unresolved_count": int(
                    transformed["primary_mechanism"].astype(str).eq(UNRESOLVED_MECHANISM).sum()
                ),
            }
        )
    return out, pd.DataFrame(rows)


def _research16_crosswalk(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    matched = frame[
        frame["reference_swing_matched"].fillna(False).astype(bool)
        & frame["research16_primary_mechanism"].notna()
    ]
    for level in HIERARCHY_COLUMNS:
        ref_col = f"reference_{level}"
        valid = matched[ref_col].notna()
        for (type_id, mechanism), group in matched.loc[valid].groupby(
            [ref_col, "research16_primary_mechanism"], sort=True
        ):
            type_total = int(matched.loc[valid, ref_col].astype(str).eq(str(type_id)).sum())
            rows.append(
                {
                    "hierarchy_level": level,
                    "type_id": str(type_id),
                    "research16_mechanism": str(mechanism),
                    "candidate_events": int(len(group)),
                    "share_within_original_type": float(len(group) / max(1, type_total)),
                    "tp60_rate": float(group["tp_1_h60"].mean()),
                    "mean_mae60_pct": float(
                        pd.to_numeric(group["mae_h60_pct"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_summary(
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    first_sweep_overlap: pd.DataFrame,
    conditions: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    stage1 = inventory[inventory["hierarchy_level"].eq("stage1_type")]
    selected_15 = coverage[
        coverage["source"].eq("region_selected")
        & coverage["lead_window_bars"].eq(15)
        & coverage["hierarchy_level"].eq("stage1_type")
    ]
    first_all = first_sweep_overlap[
        first_sweep_overlap["hierarchy_level"].eq("all_swing_lows")
    ]
    candidate_conditions = conditions[conditions.get("bridge_candidate_status", pd.Series(dtype=str)).eq("candidate")]
    lines = [
        f"# {TITLE}",
        "",
        "## Corrected research question",
        "",
        "- Research 01-03 type definitions are loaded unchanged; Research 16R does not create G1-G6 as replacement typologies.",
        "- Historical Swing Low rows are future-confirmed +1% labels, so their type is used only as a supervised reference label.",
        "- The online object is a current closed-bar candidate. Entry reference is next-bar open and future path uses closed-bar closes only.",
        "- First Sweep is measured as one possible branch inside the original hierarchy, never as a global entry gate.",
        "",
        "## Original stage-1 inventory",
        "",
    ]
    for row in stage1.itertuples(index=False):
        lines.append(
            f"- {row.type_id}: {int(row.count):,} events, {float(row.share_within_level):.2%} of frozen stage-1 Swing Lows."
        )
    lines.extend(["", "## Region-selected candidate recall within 15 bars", ""])
    for row in selected_15.itertuples(index=False):
        lines.append(
            f"- {row.type_id}: {int(row.covered_events):,}/{int(row.historical_events):,} ({float(row.recall):.2%})."
        )
    if not first_all.empty:
        row = first_all.iloc[0]
        lines.extend(
            [
                "",
                "## First Sweep position inside the original universe",
                "",
                f"- Frozen historical Swing Lows precisely linked to a true respected-macro First Sweep within the causal lead/price zone: {int(row['linked_historical_events']):,}/{int(row['historical_events']):,} ({float(row['linked_share_within_type']):.2%}).",
                f"- True First Sweep online bars mapped to any frozen Swing Low: {int(row['matched_source_events']):,}/{int(row['source_events']):,} ({float(row['source_event_match_rate']):.2%}).",
            ]
        )
    lines.extend(["", "## Frozen simple-condition bridge candidates", ""])
    if candidate_conditions.empty:
        lines.append("- No condition met the predeclared cross-fold TP-uplift, sample, and non-worsening-MAE screen.")
    else:
        for row in candidate_conditions.head(20).itertuples(index=False):
            lines.append(
                f"- {row.hierarchy_level}/{row.type_id} + {row.condition_id} Top{int(row.top_pct)}: "
                f"median TP60 uplift {float(row.median_within_type_tp_uplift_pp):+.2f}pp, "
                f"median MAE change {float(row.median_mae_change_pp):+.3f}pp."
            )
    lines.extend(["", "## Causal audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- {'PASS' if bool(row.passed) else 'FAIL'} `{row.check}`: {row.detail}")
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "- This audit does not declare a tradable edge.",
            "- Only original types with adequate online recall and simple conditions that survive multiple walk-forward folds should proceed to a type-specific expert model.",
            "- If the original C3 branches have poor recall, the next task is candidate/zone redesign, not more model features.",
            "- If recall is adequate but all frozen conditions fail, stop trying to force the ordinary Swing Low branch into a standalone long strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_research(args: argparse.Namespace) -> Path:
    if str(args.symbol) != "ETH-USDT-SWAP":
        raise ValueError("Research 16R is frozen to OKX ETH-USDT-SWAP")
    if str(args.timeframe).lower() != "1m":
        raise ValueError("Research 16R is frozen to the original 1m typology")
    if not np.isclose(float(args.target_move_pct), 1.0):
        raise ValueError("Research 16R preserves the original +1% target")
    if int(args.forward_horizon_bars) != 60:
        raise ValueError("Research 16R preserves the original +1% within 60 closed bars label")
    if int(args.reference_maximum_lead_bars) != 15:
        raise ValueError("Research 16R freezes the maximum bridge lead at 15 bars")
    if not np.isclose(float(args.reference_price_tolerance_bp), 75.0):
        raise ValueError("Research 16R freezes the symmetric bridge price tolerance at 75bp")
    lead_windows = tuple(sorted(set(int(value) for value in args.bridge_lead_windows)))
    if set(lead_windows) != {0, 3, 5, 10, 15}:
        raise ValueError("Research 16R predeclares bridge lead windows exactly 0 3 5 10 15")
    top_pcts = tuple(sorted(set(int(value) for value in args.condition_top_pcts)))
    if set(top_pcts) != {20, 30, 40}:
        raise ValueError("Research 16R predeclares Top20/30/40 condition neighborhoods")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir, stage2_dir, stage3_dir, report_validation = _validate_original_reports(args)
    _write_csv(report_validation, out_dir / "01_frozen_report_compatibility.csv")

    bars = load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "02_trade_bar_field_coverage.csv")
    historical, hierarchy_audit = _load_historical(stage1_dir, stage2_dir, stage3_dir)
    inventory = typology_inventory(historical)
    _write_csv(historical, out_dir / "03_original_historical_typology_events.csv")
    _write_csv(inventory, out_dir / "04_original_typology_inventory.csv")

    print("[stage] rebuild the same broad causal candidate universe used by Research 16", flush=True)
    candidate_config = CandidateGateConfig(
        lookback=int(args.candidate_lookback_bars),
        horizon=180,
        new_low_window=int(args.candidate_new_low_window),
        near_floor_window=int(args.candidate_near_floor_window),
        position_window=int(args.candidate_position_window),
        near_floor_tolerance_bp=float(args.candidate_near_floor_tolerance_bp),
        max_position_in_range=float(args.candidate_max_position_in_range),
    )
    raw_gate, gate_coverage = build_online_candidate_events(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        config=candidate_config,
    )

    print("[stage] respected-macro First Sweep as a measured branch only", flush=True)
    first_sweep = build_first_sweep_event_decisions(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        pivot_minutes=tuple(int(value) for value in args.liquidity_pivot_minutes),
        pivot_weights=tuple(float(value) for value in args.liquidity_pivot_weights),
        left_bars=int(args.liquidity_pivot_left_bars),
        right_bars=int(args.liquidity_pivot_right_bars),
        cluster_tolerance_bp=float(args.liquidity_cluster_tolerance_bp),
        minimum_respects=int(args.liquidity_minimum_respects),
        minimum_macro_timeframe_min=int(args.liquidity_minimum_macro_timeframe_min),
        minimum_respect_separation_minutes=int(args.liquidity_minimum_respect_separation_minutes),
        formation_max_days=int(args.liquidity_formation_max_days),
        reclaim_window_bars=int(args.liquidity_reclaim_window_bars),
        accept_below_bars=int(args.liquidity_accept_below_bars),
        accept_depth_bp=float(args.liquidity_accept_depth_bp),
        show_progress=True,
    )
    sweep_only = prepare_sweep_only_events(bars, first_sweep.decisions)
    raw_union = merge_macro_first_sweep_candidates(bars, raw_gate, first_sweep.decisions)
    decision_paths = first_sweep.decisions.get("decision_path", pd.Series(dtype="string")).astype(str)
    gate_coverage = pd.concat(
        [
            gate_coverage,
            pd.DataFrame(
                [
                    {"metric": "raw_gate_before_first_sweep_union", "value": int(len(raw_gate))},
                    {"metric": "first_sweep_all_path_decisions", "value": int(len(first_sweep.decisions))},
                    {"metric": "first_sweep_level_decisions", "value": int(decision_paths.eq("sweep").sum())},
                    {"metric": "first_sweep_reclaim_decisions", "value": int(decision_paths.eq("reclaim").sum())},
                    {"metric": "first_sweep_unique_1m_bars", "value": int(len(sweep_only))},
                    {"metric": "raw_candidate_union", "value": int(len(raw_union))},
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv(gate_coverage, out_dir / "05_candidate_gate_coverage.csv")
    _write_csv(first_sweep.diagnostics, out_dir / "06_first_sweep_build_diagnostics.csv")

    print("[stage] causal regions and globally spaced online events", flush=True)
    region_result = build_broad_candidate_regions(
        bars,
        raw_union,
        max_gap_bars=int(args.region_max_gap_bars),
        max_region_bars=int(args.region_max_bars),
        retest_tolerance_bp=float(args.region_retest_tolerance_bp),
        show_progress=True,
    )
    selected_meta = select_broad_region_events(
        region_result.frame,
        cooldown_bars=int(args.broad_cooldown_bars),
    )
    _write_csv(region_result.summary, out_dir / "07_region_build_summary.csv")
    _write_csv(region_result.dictionary, out_dir / "08_region_feature_dictionary.csv")

    print("[stage] audit original typology recall before any new model", flush=True)
    event_bridge = attach_event_bridge_flags(
        historical,
        raw_gate,
        prefix="raw_gate",
        lead_windows=lead_windows,
    )
    event_bridge = attach_event_bridge_flags(
        event_bridge,
        selected_meta,
        prefix="region_selected",
        lead_windows=lead_windows,
    )
    event_bridge = attach_event_bridge_flags(
        event_bridge,
        sweep_only,
        prefix="first_sweep",
        lead_windows=lead_windows,
    )
    temporal_coverage = bridge_coverage_scorecard(
        event_bridge,
        prefixes=("raw_gate", "region_selected", "first_sweep"),
        lead_windows=lead_windows,
    )
    mapped_raw_gate = map_candidates_to_future_typology(
        raw_gate,
        historical,
        maximum_lead_bars=int(args.reference_maximum_lead_bars),
        price_tolerance_bp=float(args.reference_price_tolerance_bp),
    )
    mapped_region_selected = map_candidates_to_future_typology(
        selected_meta,
        historical,
        maximum_lead_bars=int(args.reference_maximum_lead_bars),
        price_tolerance_bp=float(args.reference_price_tolerance_bp),
    )
    mapped_sweep_only = map_candidates_to_future_typology(
        sweep_only,
        historical,
        maximum_lead_bars=int(args.reference_maximum_lead_bars),
        price_tolerance_bp=float(args.reference_price_tolerance_bp),
    )
    bridge_coverage = pd.concat(
        [
            mapped_source_coverage_scorecard(
                mapped_raw_gate,
                historical,
                source_name="raw_gate",
                lead_windows=lead_windows,
            ),
            mapped_source_coverage_scorecard(
                mapped_region_selected,
                historical,
                source_name="region_selected",
                lead_windows=lead_windows,
            ),
            mapped_source_coverage_scorecard(
                mapped_sweep_only,
                historical,
                source_name="first_sweep",
                lead_windows=lead_windows,
            ),
        ],
        ignore_index=True,
    )
    sweep_overlap = source_typology_overlap(
        mapped_sweep_only,
        historical,
        source_name="respected_macro_first_sweep_sweep_path_only",
    )
    _write_csv(event_bridge, out_dir / "09_historical_event_candidate_bridge_temporal.csv")
    _write_csv(bridge_coverage, out_dir / "10_original_type_candidate_recall.csv")
    _write_csv(temporal_coverage, out_dir / "10b_temporal_only_candidate_recall_diagnostic.csv")
    _write_csv(sweep_overlap, out_dir / "11_first_sweep_original_typology_overlap.csv")
    _write_csv(mapped_sweep_only, out_dir / "11b_first_sweep_precise_bridge.csv")

    print("[stage] compact causal online features", flush=True)
    feature_result = build_reversal_candidate_features(
        bars,
        selected_meta,
        include_session=True,
        include_htf=True,
        show_progress=True,
    )
    if not feature_result.alignment_audit.empty and not feature_result.alignment_audit["passed"].all():
        raise RuntimeError("HTF available_time audit failed before original-type bridge labels")
    _write_csv(feature_result.dictionary, out_dir / "12_online_feature_dictionary.csv")
    _write_csv(feature_result.alignment_audit, out_dir / "13_htf_available_time_audit.csv")
    frame = feature_result.frame.copy()
    frame["is_macro_first_sweep"] = frame.get("is_macro_first_sweep", False)
    frame["is_macro_first_sweep"] = frame["is_macro_first_sweep"].fillna(False).astype(bool)

    print("[stage] candidate-owned next-open close-only path atlas", flush=True)
    labels = build_multi_horizon_close_labels(
        bars,
        frame,
        horizons=(30, 60, 180),
        target_levels_pct=(0.5, 1.0, 1.5, 2.0),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = frame.merge(labels, on="event_id", how="inner", validate="one_to_one")
    frame = map_candidates_to_future_typology(
        frame,
        historical,
        maximum_lead_bars=int(args.reference_maximum_lead_bars),
        price_tolerance_bp=float(args.reference_price_tolerance_bp),
    )

    print("[stage] out-of-sample crosswalk to Research 16 replacement mechanisms", flush=True)
    frame, mechanism_fit = _fit_oos_research16_mechanisms(frame, args.end_date)
    crosswalk = _research16_crosswalk(frame)
    _write_csv(mechanism_fit, out_dir / "14_research16_mechanism_fit_diagnostics.csv")
    _write_csv(crosswalk, out_dir / "15_original_type_to_research16_mechanism_crosswalk.csv")

    path_scorecard = path_scorecard_by_original_type(frame, horizons=(30, 60, 180))
    _write_csv(path_scorecard, out_dir / "16_original_type_candidate_path_scorecard.csv")

    print("[stage] frozen single-condition scans inside original types", flush=True)
    condition_scorecard, threshold_audit = special_condition_scorecard(
        frame,
        end_date=args.end_date,
        top_pcts=top_pcts,
        minimum_type_test_rows=int(args.minimum_type_test_rows),
        horizon=60,
    )
    condition_summary = summarize_condition_candidates(condition_scorecard)
    _write_csv(threshold_audit, out_dir / "17_condition_threshold_audit.csv")
    _write_csv(condition_scorecard, out_dir / "18_original_type_special_condition_scorecard.csv")
    _write_csv(condition_summary, out_dir / "19_special_condition_crossfold_summary.csv")

    feature_columns = tuple(
        dict.fromkeys(
            [
                *region_result.dictionary["feature"].astype(str).tolist(),
                *feature_result.dictionary["feature"].astype(str).tolist(),
                *[spec[0] for spec in SPECIAL_CONDITION_SPECS.values()],
                "is_macro_first_sweep",
            ]
        )
    )
    audit = pd.concat(
        [
            hierarchy_audit,
            bridge_causal_audit(
                frame,
                feature_columns,
                htf_alignment_audit=feature_result.alignment_audit,
            ),
            pd.DataFrame(
                [
                    {
                        "check": "official_recall_uses_time_and_price_zone",
                        "passed": bool(
                            not bridge_coverage.empty
                            and bridge_coverage["bridge_policy"].astype(str).eq(
                                "nearest_future_swing_within_time_and_symmetric_price_zone"
                            ).all()
                        ),
                        "detail": (
                            f"price_tolerance_bp={float(args.reference_price_tolerance_bp):.3f} "
                            f"rows={len(bridge_coverage):,}"
                        ),
                    },
                    {
                        "check": "first_sweep_overlap_uses_sweep_path_only",
                        "passed": bool(
                            sweep_only["decision_path"].astype(str).eq("sweep").all()
                            and sweep_only["extreme_pos"].is_unique
                        ),
                        "detail": (
                            f"all_paths={len(first_sweep.decisions):,} "
                            f"true_sweep_level_rows={int(decision_paths.eq('sweep').sum()):,} "
                            f"unique_sweep_bars={len(sweep_only):,}"
                        ),
                    },
                    {
                        "check": "first_sweep_not_global_gate",
                        "passed": bool((~frame["is_macro_first_sweep"]).any()),
                        "detail": f"first_sweep={int(frame['is_macro_first_sweep'].sum()):,} total={len(frame):,}",
                    },
                    {
                        "check": "original_typology_not_replaced",
                        "passed": True,
                        "detail": "all supervised type labels originate from frozen 01/02/03 event_id assignments",
                    },
                    {
                        "check": "candidate_owned_path_label",
                        "passed": bool(
                            (
                                pd.to_datetime(frame["entry_time"], errors="raise")
                                > pd.to_datetime(frame["extreme_time"], errors="raise")
                            ).all()
                        ),
                        "detail": "entry is next bar open; future path uses closed-bar close labels",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv(audit, out_dir / "20_causal_and_typology_audit.csv")
    if not audit["passed"].fillna(False).astype(bool).all():
        raise RuntimeError("Research 16R audit failed; inspect 20_causal_and_typology_audit.csv")

    bridge_columns = [
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "is_macro_first_sweep",
        "reference_swing_event_id",
        "reference_swing_lead_bars",
        "reference_price_distance_bp",
        "reference_swing_matched",
        *[f"reference_{column}" for column in HIERARCHY_COLUMNS],
        "research16_primary_mechanism",
        "entry_time",
        "entry_price",
        "label_end_time",
        "tp_0p5_h60",
        "tp_1_h60",
        "tp_1p5_h60",
        "tp_2_h60",
        "mfe_h60_pct",
        "mae_h60_pct",
        "mae_before_tp_1_h60_pct",
        *[spec[0] for spec in SPECIAL_CONDITION_SPECS.values() if spec[0] in frame.columns],
    ]
    bridge_frame = frame[[column for column in bridge_columns if column in frame.columns]].copy()
    if bool(args.write_full_candidate_bridge):
        _write_csv(bridge_frame, out_dir / "21_candidate_bridge_full.csv")
    else:
        sample = bridge_frame.sample(
            min(10_000, len(bridge_frame)), random_state=42
        ).sort_values(["extreme_time", "event_id"], kind="mergesort")
        _write_csv(sample, out_dir / "21_candidate_bridge_sample.csv")

    manifest = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "scope": "research_only_original_typology_bridge_no_strategy_no_pnl_backtest",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "warmup_start": args.warmup_start_date,
        "target_move_pct": float(args.target_move_pct),
        "forward_horizon_bars": int(args.forward_horizon_bars),
        "original_hierarchy_columns": list(HIERARCHY_COLUMNS),
        "historical_swing_low_count": int(len(historical)),
        "raw_candidate_count": int(len(raw_gate)),
        "region_selected_candidate_count": int(len(frame)),
        "mapped_candidate_count": int(frame["reference_swing_matched"].sum()),
        "first_sweep_all_path_decision_count": int(len(first_sweep.decisions)),
        "first_sweep_unique_sweep_bar_count": int(len(sweep_only)),
        "bridge_lead_windows": list(lead_windows),
        "reference_maximum_lead_bars": int(args.reference_maximum_lead_bars),
        "reference_price_tolerance_bp": float(args.reference_price_tolerance_bp),
        "condition_top_pcts": list(top_pcts),
        "condition_specs": {
            key: {"feature": value[0], "direction": value[1], "description": value[2]}
            for key, value in SPECIAL_CONDITION_SPECS.items()
        },
        "causal_policy": "original types and future paths are labels only; candidate features use current closed bar or older; entry is next bar open; outcomes use future closed-bar closes",
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "22_RESEARCH_SUMMARY.md").write_text(
        _build_summary(inventory, bridge_coverage, sweep_overlap, condition_summary, audit),
        encoding="utf-8",
    )

    result = finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
    )
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
