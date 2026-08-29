#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICT MSS edge discovery R01 using only causal 1m bare candles.

Mechanism
---------
1. Aggregate the project's 1m OHLC into complete 15m/30m/1H/4H candles.
2. Build swing-high/swing-low liquidity levels.  A swing cannot become active
   until all right-confirmation HTF candles have closed.
3. On the 1m axis find the first true sweep through each active HTF level.
4. After a sweep require a causal 1m MSS: a displacement candle closes through
   the latest confirmed opposite micro swing and leaves a three-candle FVG.
5. The FVG completion candle must close before the order exists.  Starting from
   the next 1m candle, place a limit at the FVG near edge (bull: third-candle
   low; bear: third-candle high).  Stop is the sweep-to-FVG extreme.
6. Evaluate fixed-R exits as research diagnostics under conservative bare-OHLC
   path rules.  No same-bar optimistic target/stop ordering is allowed.

R01 intentionally predeclares a small structural/displacement atlas before any
outcomes are inspected.  2023-2024 are development evidence, 2025 is validation
and 2026H1 is sealed confirmation.  Promotion rules that select a candidate use
only 2023-2025; the sealed period is opened only for the final edge gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ict.mss.common.evaluation import (  # noqa: E402
    build_edge_gate,
    cost_stress_table,
    evaluate_specs,
    filter_spec,
    spec_definition_table,
)
from research.ict.mss.common.execution import attach_limit_entry_and_outcomes  # noqa: E402
from research.ict.mss.common.models import DisplacementSpec, MSSResearchSpec  # noqa: E402
from research.ict.mss.common.structure import (  # noqa: E402
    build_displacement_fvgs,
    build_htf_liquidity_levels,
    build_micro_structure_context,
    build_sweep_episodes,
    normalize_bars,
    pair_sweeps_with_mss_fvgs,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "01_ict_mss_edge_discovery"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS_EDGE_DISCOVERY_R01"
EDGE_ID = "ICT_MSS_HTF_LIQUIDITY_SWEEP_DISPLACEMENT_FVG"
TITLE = "ETH ICT MSS Edge Discovery R01"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss/01_ict_mss_edge_discovery"
HTF_TIMEFRAMES = (("15m", 15), ("30m", 30), ("1H", 60), ("4H", 240))
HTF_CONFIRMATION_ORDERS = (1, 2, 3, 5)
MICRO_ORDERS = (2, 3, 5)
TARGET_RS = (1.0, 1.5, 2.0, 3.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ICT MSS edge discovery from 1m bare OKX candles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m", choices=["1m"])
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sweep-epsilon-bp", type=float, default=0.01)
    p.add_argument("--displacement-rolling-window", type=int, default=60)
    p.add_argument("--max-sweep-to-fvg-search-bars", type=int, default=180)
    p.add_argument("--max-precomputed-fill-wait-bars", type=int, default=120)
    p.add_argument("--outcome-horizon-bars", type=int, default=240)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--write-full-setups", action="store_true")
    return p.parse_args(argv)


def _date_end_inclusive(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    raw = str(value).strip()
    if len(raw) <= 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return ts


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_bare_1m(args: argparse.Namespace) -> pd.DataFrame:
    end = _date_end_inclusive(args.end_date)
    print(
        f"[load] OKXDataLoader bare OHLC {args.symbol} 1m "
        f"{args.warmup_start_date} -> {end}",
        flush=True,
    )
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    raw = loader.fetch_data_by_date_range(pd.Timestamp(args.warmup_start_date), end)
    if raw.empty:
        raise RuntimeError(
            "No 1m candle data loaded through src.data_feed.OKXDataLoader. "
            "Check data/crypto_history.db coverage or network access."
        )
    bars = normalize_bars(raw.loc[:, ["open", "high", "low", "close"]])
    print(f"[load] rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _data_quality(bars: pd.DataFrame) -> pd.DataFrame:
    expected = pd.date_range(bars.index.min(), bars.index.max(), freq="1min")
    missing = int(len(expected.difference(bars.index)))
    diffs = bars.index.to_series().diff().dropna()
    return pd.DataFrame(
        [
            {
                "rows": int(len(bars)),
                "start": bars.index.min(),
                "end": bars.index.max(),
                "duplicate_timestamps": int(bars.index.duplicated().sum()),
                "missing_1m_bars": missing,
                "gap_count_gt_1m": int((diffs > pd.Timedelta(minutes=1)).sum()),
                "max_gap_minutes": float(diffs.max().total_seconds() / 60.0) if len(diffs) else 0.0,
                "nonpositive_prices": int((bars[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
                "ohlc_inconsistent": int(((bars["high"] < bars[["open", "close", "low"]].max(axis=1)) | (bars["low"] > bars[["open", "close", "high"]].min(axis=1))).sum()),
            }
        ]
    )


def _displacement_specs() -> dict[str, DisplacementSpec]:
    # Fixed absolute mechanism thresholds; no outcome-fitted quantiles.
    return {
        "weak": DisplacementSpec("weak", 1.40, 1.35, 0.52, 0.30, 1.0),
        "core_lo": DisplacementSpec("core_lo", 1.70, 1.55, 0.58, 0.24, 2.0),
        "core": DisplacementSpec("core", 2.00, 1.80, 0.64, 0.18, 3.0),
        "core_hi": DisplacementSpec("core_hi", 2.30, 2.05, 0.69, 0.14, 4.0),
        "strong": DisplacementSpec("strong", 2.80, 2.40, 0.72, 0.12, 5.0),
    }


def _research_specs() -> tuple[MSSResearchSpec, ...]:
    # Every variation is declared before reading outcomes.  This is intentionally
    # small: mechanism neighborhoods, not a parameter optimizer.
    return (
        MSSResearchSpec("B00_loose", "Loose mechanism sanity baseline", min_htf_confirmed_order=1, max_sweep_to_displacement_bars=90, displacement_name="weak", max_fill_wait_bars=90),
        MSSResearchSpec("C01_core_lo", "Core displacement lower neighbor", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core_lo", max_fill_wait_bars=60, neighborhood_group="core_displacement"),
        MSSResearchSpec("C02_core", "Core MSS definition", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60, neighborhood_group="core_displacement"),
        MSSResearchSpec("C03_core_hi", "Core displacement upper neighbor", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core_hi", max_fill_wait_bars=60, neighborhood_group="core_displacement"),
        MSSResearchSpec("D04_strong", "Very strong displacement", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="strong", max_fill_wait_bars=60),
        MSSResearchSpec("L05_long_core", "Long-only sell-side liquidity sweep", side=1, min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("S06_short_core", "Short-only buy-side liquidity sweep", side=-1, min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("M07_micro3", "Slower 1m swing structure order=3", micro_order=3, min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("M08_micro5", "Major 1m swing structure order=5", micro_order=5, min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("R09_rolling", "Allow post-sweep confirmed 1m structure before MSS", structure_mode="rolling", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("H10_1h_plus", "Only sweeps including >=1H liquidity", min_htf_confirmed_order=2, min_max_timeframe_min=60, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("H11_4h", "Only 4H liquidity sweeps", min_htf_confirmed_order=2, min_max_timeframe_min=240, max_sweep_to_displacement_bars=90, displacement_name="core", max_fill_wait_bars=90),
        MSSResearchSpec("Q12_htf_order3", "More obvious HTF swing order>=3", min_htf_confirmed_order=3, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("Q13_confluence", "Same-bar liquidity sweep across >=2 HTFs", min_htf_confirmed_order=2, min_swept_timeframe_count=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("T14_fast_mss", "MSS within 15m of liquidity sweep", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=15, displacement_name="core", max_fill_wait_bars=60),
        MSSResearchSpec("F15_fast_fill", "FVG retest must fill within 30m", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=60, displacement_name="core", max_fill_wait_bars=30),
        MSSResearchSpec("W16_wide_window", "Slower MSS/retest structural sensitivity", min_htf_confirmed_order=2, max_sweep_to_displacement_bars=120, displacement_name="core", max_fill_wait_bars=120),
    )


def _causal_audit(
    levels: pd.DataFrame,
    level_sweeps: pd.DataFrame,
    paired: pd.DataFrame,
    attached: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    if not levels.empty:
        checks.append(
            {
                "check": "htf_initial_available_after_pivot_bar_close",
                "violations": int((pd.to_datetime(levels["initial_available_time"]) <= pd.to_datetime(levels["pivot_bar_end_time"])).sum()),
            }
        )
    if not level_sweeps.empty:
        checks.append(
            {
                "check": "liquidity_level_available_before_sweep_bar",
                "violations": int((pd.to_datetime(level_sweeps["initial_available_time"]) > pd.to_datetime(level_sweeps["sweep_bar_time"])).sum()),
            }
        )
    if not paired.empty:
        checks.extend(
            [
                {
                    "check": "micro_structure_available_before_displacement_bar",
                    "violations": int((paired["micro_structure_available_pos"] > paired["displacement_pos"]).sum()),
                },
                {
                    "check": "fvg_completion_strictly_after_sweep",
                    "violations": int((paired["fvg_completion_pos"] <= paired["sweep_pos"]).sum()),
                },
                {
                    "check": "fvg_available_after_completion_bar",
                    "violations": int((pd.to_datetime(paired["fvg_available_time"]) <= pd.to_datetime(paired["fvg_completion_time"])).sum()),
                },
            ]
        )
    if not attached.empty:
        checks.extend(
            [
                {
                    "check": "limit_order_active_only_after_fvg_completion",
                    "violations": int((attached["order_active_pos"] <= attached["fvg_completion_pos"]).sum()),
                },
                {
                    "check": "fill_never_before_order_activation",
                    "violations": int(((attached["first_fill_pos"] >= 0) & (attached["first_fill_pos"] < attached["order_active_pos"])).sum()),
                },
            ]
        )
    return pd.DataFrame(checks)


def _side_summary(setups: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for side, part in setups.groupby("side", sort=True):
        rows.append(
            {
                "side": "long" if int(side) == 1 else "short",
                "paired_mss_fvgs": int(len(part)),
                "valid_structural_stop": int(part["entry_structure_valid"].sum()),
                "filled_within_120m": int((part["first_fill_pos"].ge(0) & part["fill_wait_bars"].le(120)).sum()),
                "median_sweep_to_mss_bars": float(part["sweep_to_displacement_bars"].median()),
                "median_fvg_size_bp": float(part["fvg_size_bp"].median()),
                "median_risk_pct": float(part.loc[part["entry_structure_valid"], "risk_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def _write_summary(
    path: Path,
    *,
    bars: pd.DataFrame,
    levels: pd.DataFrame,
    episodes: pd.DataFrame,
    fvgs: pd.DataFrame,
    paired: pd.DataFrame,
    edge_gate: pd.DataFrame,
    causal_audit: pd.DataFrame,
) -> None:
    passing = edge_gate.loc[edge_gate["edge_found"]].copy() if not edge_gate.empty else pd.DataFrame()
    audit_ok = bool(causal_audit.empty or int(causal_audit["violations"].sum()) == 0)
    decision = "promote_to_backtest" if audit_ok and not passing.empty else "research_continue"
    lines = [
        f"# {TITLE}",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Data: 1m bare OHLC only, `{bars.index.min()}` -> `{bars.index.max()}`",
        f"- HTF liquidity levels: `{len(levels):,}`",
        f"- First-sweep episodes: `{len(episodes):,}`",
        f"- Raw directional FVGs: `{len(fvgs):,}`",
        f"- Sweep -> causal MSS/FVG pairs: `{len(paired):,}`",
        f"- Causal audit violations: `{int(causal_audit['violations'].sum()) if not causal_audit.empty else 0}`",
        f"- Edge-gate passes: `{len(passing)}`",
        f"- Decision: **{decision}**",
        "",
        "## Timing rules",
        "",
        "HTF swings become usable only after right-confirmation bars close. The 1m MSS may only break a micro swing already confirmed before the displacement candle opens. The FVG is known only after its third candle closes, so the limit order begins on the following 1m candle. Bare-OHLC same-bar target/stop ambiguity is resolved as stop-first.",
        "",
        "## Holdout discipline",
        "",
        "2023-2024 are development evidence; 2025 is validation. Candidate freezing, 2x-cost stress and top-10-winner-removal gates use only 2023-2025. 2026H1 is opened only for final confirmation and is never used to rank or tune variants.",
        "",
        "## Next action",
        "",
    ]
    if passing.empty:
        lines.append("No predeclared R01 candidate cleared the full edge gate. Do not build a trading strategy from this version; inspect the mechanism/slice diagnostics and continue with a new hypothesis-driven R02 instead of tuning R01 to losses.")
    else:
        lines.append("At least one predeclared candidate cleared the frozen gate and sealed 2026H1 confirmation. Build a separate executable backtest only for the frozen passing definition(s), then run latency/slippage/order-fill stress before considering live migration.")
        lines.append("")
        lines.append("Passing candidates:")
        for row in passing.sort_values(["spec_id", "target_r"]).itertuples(index=False):
            lines.append(f"- `{row.spec_id}` @ `{row.target_r:g}R`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    show_progress = not bool(args.no_progress)
    research_start = pd.Timestamp(args.start_date)
    research_end = _date_end_inclusive(args.end_date)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_bare_1m(args)
    if bars.index.min() > pd.Timestamp(args.warmup_start_date) + pd.Timedelta(minutes=1):
        raise RuntimeError(f"Warmup coverage starts too late: {bars.index.min()}")
    if bars.index.max() < research_end - pd.Timedelta(minutes=1):
        raise RuntimeError(f"Research coverage ends too early: {bars.index.max()} < {research_end}")

    quality = _data_quality(bars)
    print("[stage] causal HTF swing liquidity 15m/30m/1H/4H", flush=True)
    levels = build_htf_liquidity_levels(
        bars,
        timeframes=HTF_TIMEFRAMES,
        confirmation_orders=HTF_CONFIRMATION_ORDERS,
    )
    print(f"[levels] rows={len(levels):,}", flush=True)

    print("[stage] first causal liquidity sweeps on 1m", flush=True)
    level_sweeps, episodes = build_sweep_episodes(
        bars,
        levels,
        confirmation_orders=HTF_CONFIRMATION_ORDERS,
        sweep_epsilon_bp=float(args.sweep_epsilon_bp),
        show_progress=show_progress,
    )
    print(f"[sweeps] level_sweeps={len(level_sweeps):,} episodes={len(episodes):,}", flush=True)

    print("[stage] causal 1m micro swings + displacement FVG atlas", flush=True)
    micro = build_micro_structure_context(bars, orders=MICRO_ORDERS)
    fvgs = build_displacement_fvgs(bars, rolling_window=int(args.displacement_rolling_window))
    print(f"[fvg] directional_fvgs={len(fvgs):,}", flush=True)

    print("[stage] HTF sweep -> 1m MSS + displacement/FVG", flush=True)
    paired = pair_sweeps_with_mss_fvgs(
        bars,
        episodes,
        fvgs,
        micro,
        max_search_bars=int(args.max_sweep_to_fvg_search_bars),
        show_progress=show_progress,
    )
    print(f"[mss] causal_pairs={len(paired):,}", flush=True)

    print("[stage] next-bar-active FVG limit fills + conservative R outcomes", flush=True)
    attached = attach_limit_entry_and_outcomes(
        bars,
        paired,
        max_fill_wait_bars=int(args.max_precomputed_fill_wait_bars),
        outcome_horizon_bars=int(args.outcome_horizon_bars),
        target_rs=TARGET_RS,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        show_progress=show_progress,
    )

    displacement_specs = _displacement_specs()
    specs = _research_specs()
    print(f"[stage] fixed research atlas specs={len(specs)} targets={len(TARGET_RS)}", flush=True)
    overall, periods, funnel, slices = evaluate_specs(
        attached,
        specs,
        displacement_specs,
        target_rs=TARGET_RS,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        research_start=research_start,
        research_end=research_end,
    )
    cost_stress = cost_stress_table(
        slices,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        multipliers=(1.0, 2.0, 3.0),
    )
    edge_gate = build_edge_gate(overall, periods, cost_stress, specs)
    causal_audit = _causal_audit(levels, level_sweeps, paired, attached)
    if not causal_audit.empty and int(causal_audit["violations"].sum()) > 0:
        raise RuntimeError(f"Causal audit failed:\n{causal_audit.to_string(index=False)}")

    # Report artifacts: compact summaries by default; optional full setup file
    # stays outside source code and may be too large for the GPT review pack.
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(spec_definition_table(specs, displacement_specs), out_dir / "02_fixed_spec_definitions.csv")
    _write_csv(causal_audit, out_dir / "03_causal_audit.csv")
    _write_csv(funnel, out_dir / "04_candidate_funnel.csv")
    _write_csv(_side_summary(attached), out_dir / "05_side_mechanism_summary.csv")
    _write_csv(overall, out_dir / "06_overall_fixed_r_metrics.csv")
    _write_csv(periods, out_dir / "07_period_metrics.csv")
    _write_csv(cost_stress, out_dir / "08_cost_stress.csv")
    _write_csv(edge_gate, out_dir / "09_edge_gate.csv")
    if not attached.empty:
        audit_cols = [
            "setup_key", "sweep_id", "side", "sweep_bar_time", "sweep_available_time",
            "swept_timeframes", "max_timeframe_min", "max_confirmed_order",
            "micro_order", "structure_mode", "micro_structure_pivot_time",
            "micro_structure_available_time", "micro_structure_level", "displacement_pos",
            "fvg_completion_time", "fvg_available_time", "fvg_near_price", "fvg_size_bp",
            "stop_extreme", "order_active_time", "first_fill_time", "entry_price", "stop_price",
            "risk_pct", "fill_wait_bars", "sweep_to_displacement_bars",
            "displacement_body_vs_past_median", "displacement_range_vs_past_median",
            "displacement_body_fraction", "displacement_close_from_extreme_fraction",
            "mfe_r_horizon", "mae_r_horizon", "gap_through_stop_on_fill_flag",
        ]
        audit_cols += [c for c in attached.columns if c.startswith(("result_r", "net_r_r", "gross_r_r"))]
        audit_cols = [c for c in audit_cols if c in attached.columns]
        sample = attached.loc[:, audit_cols].sort_values("fvg_completion_time").head(5000)
        _write_csv(sample, out_dir / "10_trade_replay_audit_sample.csv")
        if bool(args.write_full_setups):
            _write_csv(attached.loc[:, audit_cols], out_dir / "10b_trade_replay_audit_full.csv")

    _write_summary(
        out_dir / "11_summary.md",
        bars=bars,
        levels=levels,
        episodes=episodes,
        fvgs=fvgs,
        paired=paired,
        edge_gate=edge_gate,
        causal_audit=causal_audit,
    )
    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "data_source": "src.data_feed.okx_loader.OKXDataLoader",
        "input_semantics": "1m bare OHLC only",
        "warmup_start": str(args.warmup_start_date),
        "research_start": str(args.start_date),
        "research_end": str(args.end_date),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "htf_timeframes": HTF_TIMEFRAMES,
        "htf_confirmation_orders": HTF_CONFIRMATION_ORDERS,
        "micro_orders": MICRO_ORDERS,
        "target_rs": TARGET_RS,
        "rows": {
            "bars": len(bars),
            "levels": len(levels),
            "level_sweeps": len(level_sweeps),
            "sweep_episodes": len(episodes),
            "directional_fvgs": len(fvgs),
            "paired_mss_fvgs": len(paired),
        },
        "causal_audit_violations": int(causal_audit["violations"].sum()) if not causal_audit.empty else 0,
        "edge_found_count": int(edge_gate["edge_found"].sum()) if not edge_gate.empty else 0,
        "decision": "promote_to_backtest" if (not edge_gate.empty and bool(edge_gate["edge_found"].any())) else "research_continue",
        "notes": [
            "No future-fitted quantiles are used in the signal mechanism.",
            "HTF context is used only after its causal available time.",
            "FVG limit orders become active on the bar after FVG completion.",
            "Same-bar target/stop ambiguity is resolved stop-first.",
            "2026H1 never participates in candidate freezing or pre-holdout stress gates.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    pack = finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
        print_log=True,
    )
    print(f"[done] report_dir={out_dir}", flush=True)
    print(f"[done] edge_found={manifest['edge_found_count']}", flush=True)
    print(f"[done] review_pack={pack.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
