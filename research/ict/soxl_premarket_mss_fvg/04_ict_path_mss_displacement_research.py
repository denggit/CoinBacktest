#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R04: liquidity sweep -> structural MSS -> displacement leg -> FVG.

This revision corrects R02's concept error: the MSS break candle is NOT required
itself to be a large displacement candle or the third candle of an FVG.  MSS,
displacement and FVG are modeled as related but distinct parts of the reversal
process.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader  # noqa: E402
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE  # noqa: E402
from src.research_common.ict.premarket_mss_fvg import (  # noqa: E402
    BASE_SCENARIO,
    NY_TZ,
    ReplayScenario,
    ResearchConfig,
    add_analysis_dimensions,
    build_data_quality_table,
    compound_account,
    eligible_ny_dates,
    enforce_single_lifecycle,
    make_synthetic_ict_day,
    ny_date_bounds_to_source_naive,
    replay_attempts,
    report_trade_history,
    source_naive_to_new_york,
    summarize_variant,
)
from src.research_common.ict.premarket_mss_fvg_v2 import (  # noqa: E402
    SweepEpisodeConfig,
    build_all_premarket_levels_v2,
    build_sweep_events_v2,
    filter_liquidity_mode_v2,
    summarize_by_groups,
)
from src.research_common.ict.premarket_mss_fvg_v3 import (  # noqa: E402
    ICTPathConfig,
    build_causal_audit_v3,
    build_signal_attempts_v3,
)
from src.research_common.ict.spot_perp_overlap import (  # noqa: E402
    build_equity_proxy_data_quality_table,
    densify_equity_minutes_causally,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

DEFAULT_SYMBOL = "SOXL-USDT-SWAP"
DEFAULT_START_DATE = "2026-05-20"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r04_ict_path_mss_displacement"
LIQUIDITY_MODES = ("extremes_only", "extremes_plus_strong_15m_swing")


def _csv_numbers(text: str, *, cast=float) -> list[Any]:
    out = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError(f"empty numeric list: {text!r}")
    return out


def _csv_names(text: str) -> list[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("empty name list")
    return out


def _source_offset_hours(text: str) -> int:
    value = str(text).strip().upper().replace("UTC", "")
    try:
        return int(value)
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOXL ICT R04 path-based MSS/displacement/FVG causal research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-source", choices=("okx", "alpaca"), default="okx")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", choices=("sip", "iex", "boats"), default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--include-us-equity-holidays", action="store_true")
    p.add_argument("--required-day-coverage", type=float, default=0.995)
    p.add_argument("--execution-timeframes", default="1,2,5")
    p.add_argument("--liquidity-modes", default=",".join(LIQUIDITY_MODES))
    p.add_argument("--premarket-pivot-left", type=int, default=2)
    p.add_argument("--premarket-pivot-right", type=int, default=2)
    p.add_argument("--mss-pivot-left", type=int, default=1)
    p.add_argument("--mss-pivot-right", type=int, default=1)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1,2,3")
    p.add_argument("--order-delay-minutes", default="0,1,2")
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=2.0)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-platform-reports", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    tfs = _csv_numbers(args.execution_timeframes, cast=int)
    if any(tf not in {1, 2, 5} for tf in tfs):
        raise ValueError("execution_timeframes must be a subset of 1,2,5")
    invalid = set(_csv_names(args.liquidity_modes)) - set(LIQUIDITY_MODES)
    if invalid:
        raise ValueError(f"invalid liquidity modes: {sorted(invalid)}")
    if not 0.90 <= float(args.required_day_coverage) <= 1.0:
        raise ValueError("required_day_coverage must be in [0.90,1.0]")


def _base_cfg(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        mss_pivot_left=int(args.mss_pivot_left),
        mss_pivot_right=int(args.mss_pivot_right),
        premarket_pivot_left=int(args.premarket_pivot_left),
        premarket_pivot_right=int(args.premarket_pivot_right),
        required_day_coverage=float(args.required_day_coverage),
        round_trip_cost=float(args.round_trip_cost),
        risk_fraction=float(args.risk_fraction),
        max_notional_multiple=float(args.max_notional_multiple),
    )


def _path_cfg(args: argparse.Namespace) -> ICTPathConfig:
    return ICTPathConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        mss_pivot_left=int(args.mss_pivot_left),
        mss_pivot_right=int(args.mss_pivot_right),
        require_relative_impulse=True,
    )


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        loader = AlpacaStockLoader(
            symbol=args.alpaca_symbol,
            timeframe="1Min",
            feed=args.alpaca_feed,
            adjustment=args.alpaca_adjustment,
            data_dir=args.data_dir,
        )
        print(f"[load] Alpaca {args.alpaca_symbol} 1Min {args.alpaca_feed}/{args.alpaca_adjustment} NY={args.start_date}->{args.end_date}", flush=True)
        raw = loader.fetch_data_by_date_range(
            start_ny.tz_convert("UTC"),
            end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1),
            local_only=bool(args.local_only),
        )
        if raw.empty:
            raise RuntimeError("Alpaca loader returned no data")
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        bars = raw.copy()
        bars.index = idx.tz_convert(NY_TZ)
        bars.index.name = "bar_start_ny"
    else:
        offset = _source_offset_hours(OKX_LOADER_TIMEZONE)
        start_src, end_src = ny_date_bounds_to_source_naive(args.start_date, args.end_date, source_offset_hours=offset)
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        print(f"[load] OKX {args.symbol} 1m NY={args.start_date}->{args.end_date}", flush=True)
        if args.local_only:
            raw = loader.load_local_data()
            if not raw.empty:
                raw = raw.loc[(raw.index >= start_src) & (raw.index <= end_src)].copy()
        else:
            raw = loader.fetch_data_by_date_range(start_src, end_src)
        if raw.empty:
            raise RuntimeError("OKX loader returned no data")
        bars = source_naive_to_new_york(raw, source_offset_hours=offset)

    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()  # NY 04:00-16:30
    if args.data_source == "alpaca":
        bars = densify_equity_minutes_causally(bars)
    print(f"[load] rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _design(args: argparse.Namespace, cfg: ResearchConfig, valid_days: int) -> str:
    return f"""# SOXL ICT R04 — path-based MSS / displacement / FVG

## Why R04 exists
R02 incorrectly required the MSS break candle itself to be a large-body displacement candle, close near its extreme, and simultaneously be the third candle of the FVG. Those are removed.

## ICT process modeled
1. Freeze NY 04:00-08:30 external premarket high/low plus genuinely strong causal 15m internal swings.
2. Trade only 08:30-16:30. A sweep of fresh liquidity opens a sweep episode and tracks the current terminal extreme.
3. On 1m/2m/5m, MSS is the first completed close through the latest causally valid opposing short-term pivot. That pivot may already exist before the terminal extreme (direct V reversal) or form after the terminal extreme during the developing reversal (new small STH/STL).
4. Displacement is the *whole reversal leg* from terminal extreme through MSS. It is not one candle. The base non-tuned relative-impulse rule rejects a reversal that delivers more slowly than the inbound reference->terminal leg: outbound directional speed must be >= inbound speed.
5. At least one directional three-candle FVG must exist anywhere inside that reversal leg and be known by MSS confirmation. The MSS candle does not need to be the FVG third candle.
6. If several FVGs exist in the displacement leg, use the latest one known by MSS confirmation; enter at the third candle's near edge (bullish third-candle low / bearish third-candle high).
7. Stop = sweep terminal extreme. Target = opposite fresh absolute premarket extreme. Pending order cancels if target arrives first, extreme invalidates first, or at 16:30.

## Causality
All higher-timeframe bars use explicit available_time. No pivot/FVG/MSS is used before the necessary bars have closed. Orders become active only from signal time onward.

## Frozen costs
Round trip={cfg.round_trip_cost:.4%}; cost stress={args.cost_multipliers}x; delay stress={args.order_delay_minutes} minutes. Valid sessions={valid_days}.
"""


def _replay_summaries(bars: pd.DataFrame, attempts: pd.DataFrame, args: argparse.Namespace, cfg: ResearchConfig):
    modes = _csv_names(args.liquidity_modes)
    costs = _csv_numbers(args.cost_multipliers, cast=float)
    delays = _csv_numbers(args.order_delay_minutes, cast=int)
    lifecycle_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    total = len(cfg.execution_timeframes) * len(modes) * (len(costs) + max(0, len(delays) - 1))
    prog = ProgressReporter(label="[replay] variants/stress", total=max(1, total), every=1, enabled=not args.no_progress)
    done = 0
    for tf in cfg.execution_timeframes:
        for mode in modes:
            subset = attempts.loc[attempts["execution_tf_minutes"] == int(tf)].copy()
            subset = filter_liquidity_mode_v2(subset, mode)
            scenarios = []
            for c in costs:
                scenarios.append((ReplayScenario("base" if c == 1 else f"cost_{c:g}x", c, 0), "cost"))
            scenarios += [(ReplayScenario(f"delay_{d}m", 1.0, d), "delay") for d in delays if d != 0]
            for scenario, family in scenarios:
                replayed = replay_attempts(
                    bars, subset, scenario=scenario, round_trip_cost=cfg.round_trip_cost,
                    risk_fraction=cfg.risk_fraction, max_notional_multiple=cfg.max_notional_multiple,
                )
                kept, skipped = enforce_single_lifecycle(replayed)
                if not kept.empty:
                    kept["liquidity_mode"] = mode
                    kept["stress_family"] = family
                    kept["variant_key"] = f"tf={tf}m|liq={mode}|disp=relative_leg|scenario={scenario.name}"
                    lifecycle_parts.append(kept)
                summaries.append({
                    "execution_tf": f"{tf}m", "execution_tf_minutes": tf, "liquidity_mode": mode,
                    "displacement_model": "relative_leg_speed_ge_inbound", "scenario": scenario.name,
                    "stress_family": family, "cost_multiple": scenario.cost_multiple,
                    "order_delay_minutes": scenario.order_delay_minutes,
                    **summarize_variant(kept, skipped_overlap=skipped, initial_capital=float(args.initial_capital)),
                })
                done += 1
                prog.update(done)
    prog.close()
    lifecycle = pd.concat(lifecycle_parts, ignore_index=True) if lifecycle_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    base = summary.loc[(summary["scenario"] == "base") & (summary["stress_family"] == "cost")].copy() if not summary.empty else pd.DataFrame()
    return lifecycle, base, summary.loc[summary["stress_family"] == "cost"].copy() if not summary.empty else pd.DataFrame(), summary.loc[summary["stress_family"] == "delay"].copy() if not summary.empty else pd.DataFrame()


def _emit_platform_reports(base_lifecycle: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    if base_lifecycle.empty or args.skip_platform_reports:
        return
    root = out_dir / "platform_full_reports"
    for (tf, mode), group in base_lifecycle.groupby(["execution_tf", "liquidity_mode"], sort=True):
        filled = group.loc[group["filled"].fillna(False).astype(bool)].copy()
        if filled.empty:
            continue
        account = compound_account(filled, initial_capital=float(args.initial_capital))
        history = report_trade_history(account)
        if not history:
            continue
        print_full_report(
            trade_history=history, df=bars, initial_capital=float(args.initial_capital),
            capital=float(account["capital"].iloc[-1]), strategy_name=f"SOXL_ICT_R04_{tf}_{mode}",
            total_days=max((bars.index.max() - bars.index.min()).total_seconds() / 86400.0, 1.0),
            ai_enabled=False, symbol=args.symbol, report_dir=root / f"{tf}_{mode}",
        )


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    cfg = _base_cfg(args)
    path_cfg = _path_cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = eligible_ny_dates(bars, start_date=args.start_date, end_date=args.end_date,
                             exclude_equity_holidays=not args.include_us_equity_holidays)
    quality = build_equity_proxy_data_quality_table(bars, days) if args.data_source == "alpaca" else build_data_quality_table(bars, days, required_coverage=cfg.required_day_coverage)
    valid_text = set(quality.loc[quality["coverage_pass"], "ny_date"].astype(str)) if not quality.empty else set()
    valid_days = [pd.Timestamp(x).date() for x in sorted(valid_text)]
    if not valid_days:
        raise RuntimeError("No valid sessions after coverage gate")
    print(f"[coverage] eligible={len(days)} valid={len(valid_days)}", flush=True)

    stage = ProgressReporter(label="[research] build stages", total=5, every=1, enabled=not args.no_progress)
    levels = build_all_premarket_levels_v2(bars, valid_days, pivot_left=cfg.premarket_pivot_left,
                                            pivot_right=cfg.premarket_pivot_right, episode_config=SweepEpisodeConfig())
    stage.update(1)
    sweeps = build_sweep_events_v2(bars, levels)
    stage.update(2)
    attempts, funnel = build_signal_attempts_v3(bars, sweeps, config=path_cfg, progress_enabled=not args.no_progress)
    stage.update(3)
    lifecycle, base_summary, cost_stress, delay_stress = _replay_summaries(bars, attempts, args, cfg) if not attempts.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    stage.update(4)
    base_lifecycle = add_analysis_dimensions(lifecycle.loc[(lifecycle["scenario"] == "base") & (lifecycle["stress_family"] == "cost")].copy()) if not lifecycle.empty else pd.DataFrame()
    audit = build_causal_audit_v3(attempts)
    stage.update(5); stage.close()

    (out_dir / "00_research_design.md").write_text(_design(args, cfg, len(valid_days)), encoding="utf-8")
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(levels, out_dir / "02_premarket_liquidity_levels.csv")
    _write_csv(sweeps, out_dir / "03_sweep_events.csv")
    _write_csv(funnel, out_dir / "04_mss_displacement_funnel.csv")
    _write_csv(attempts, out_dir / "05_signal_attempts.csv")
    _write_csv(base_lifecycle, out_dir / "06_base_trade_lifecycle.csv")
    _write_csv(base_summary, out_dir / "07_base_variant_summary.csv")
    _write_csv(cost_stress, out_dir / "08_cost_stress.csv")
    _write_csv(delay_stress, out_dir / "09_order_delay_stress.csv")
    filled = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy() if not base_lifecycle.empty else pd.DataFrame()
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "execution_tf"]), out_dir / "10_execution_timeframe_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "level_type"]), out_dir / "11_liquidity_type_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "trade_side"]), out_dir / "12_long_short_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "weekday"]), out_dir / "13_weekday_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "sweep_time_bucket"]), out_dir / "14_sweep_time_compare.csv")
    _write_csv(audit, out_dir / "15_causal_audit.csv")

    findings = [
        "# R04 Findings", "",
        f"- Valid sessions: **{len(valid_days)}**.",
        f"- Fresh/eligible sweeps: **{int(sweeps.get('setup_eligible_at_sweep', pd.Series(dtype=bool)).fillna(False).sum()) if not sweeps.empty else 0}**.",
        f"- ICT path-qualified attempts: **{len(attempts)}**.",
        "- R04 does not require the MSS break candle to be a special displacement candle or an FVG third candle.",
        "- Displacement is measured over terminal-extreme -> MSS as a whole reversal leg; FVG can occur anywhere inside that leg.",
    ]
    if not funnel.empty:
        by = funnel.groupby("execution_tf").agg(sweeps=("fresh_sweep","sum"), mss=("mss_found","sum"), fvg=("fvg_in_displacement_leg_found","sum"), impulse=("relative_impulse_pass","sum"), attempts=("attempt_emitted","sum")).reset_index()
        findings += ["", "## Funnel"] + [f"- `{r.execution_tf}`: sweeps={int(r.sweeps)}, MSS={int(r.mss)}, FVG-in-leg={int(r.fvg)}, relative-impulse={int(r.impulse)}, attempts={int(r.attempts)}." for r in by.itertuples(index=False)]
    if not base_summary.empty:
        ranked = base_summary.sort_values(["profit_factor","mean_net_return"], ascending=False, na_position="last")
        findings += ["", "## Base variants"]
        for r in ranked.head(8).to_dict("records"):
            findings.append(f"- `{r['execution_tf']} / {r['liquidity_mode']}`: trades={int(r.get('filled_trades',0) or 0)}, win={float(r.get('win_rate',np.nan)):.1%}, PF={float(r.get('profit_factor',np.nan)):.3f}, mean_net={float(r.get('mean_net_return',np.nan)):.3%}.")
    (out_dir / "16_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": "SOXL_ICT_MSS_R04_PATH_DISPLACEMENT",
        "edge_id": "SOXL_ICT_PM_SWEEP_MSS_FVG",
        "data_source": args.data_source,
        "start_date": args.start_date, "end_date": args.end_date,
        "valid_sessions": len(valid_days),
        "session_ny": "04:00-16:30", "premarket_ny": "04:00-08:30", "trade_ny": "08:30-16:30",
        "mss_definition": "liquidity sweep then close through latest causally confirmed opposing short-term pivot; post-terminal STH/STL is valid",
        "displacement_definition": "terminal-extreme to MSS reversal leg; outbound speed >= inbound reference-to-terminal speed",
        "fvg_definition": "directional 3-candle imbalance anywhere inside reversal leg; MSS candle need not be FVG candle3",
        "entry": "latest FVG in displacement leg known by MSS; third-candle near edge",
        "round_trip_cost": cfg.round_trip_cost,
        "causal_audit_passed": bool(not audit.empty and audit["passed"].fillna(False).all()),
    }
    _write_json(manifest, out_dir / "17_manifest.json")
    _emit_platform_reports(base_lifecycle, bars, args, out_dir)
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=manifest["experiment_id"], edge_id=manifest["edge_id"], title="SOXL ICT R04 Path MSS / Displacement / FVG", print_log=True)
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    bars = make_synthetic_ict_day()
    with tempfile.TemporaryDirectory(prefix="soxl_ict_r04_") as tmp:
        args.start_date = args.end_date = "2026-06-02"
        args.out_dir = tmp
        args.include_us_equity_holidays = True
        args.required_day_coverage = 1.0
        args.skip_platform_reports = True
        args.skip_review_pack = True
        args.no_progress = True
        result = run_research(bars, args)
        if not (result["report_dir"] / "15_causal_audit.csv").exists():
            raise AssertionError("missing causal audit")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    if args.self_test:
        return run_self_test(args)
    run_research(_load_1m(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
