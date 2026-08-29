#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 SOXL ICT sweep-episode state machine -> dynamic MSS -> displacement FVG backtest.

Requested process
-----------------
1. Use only weekday / US-equity trading sessions.
2. Freeze New York 04:00-08:30 premarket liquidity from 1m data:
   * absolute premarket high/low;
   * strongest causally confirmed 15m internal swing high/low.
3. Trade only New York 08:30-16:30.
4. A liquidity sweep opens a causal sweep episode. Track its terminal extreme
   and dynamically update the relevant opposing short-term pivot on 1m/2m/5m.
5. Require a completed execution bar to close through the dynamic pivot (MSS), show
   displacement, and itself form a three-candle FVG.
6. Place a limit at the third FVG candle's low (long) / high (short) no earlier
   than the completed signal bar's available time.
7. Stop at the post-sweep extreme observed through signal time.
8. Target the opposite absolute premarket extreme only while that liquidity is still fresh.
9. Cancel an unfilled order when the opposite target is reached first, when the
   sweep extreme is invalidated first, or at 16:30.
10. Close any open position at 16:30; no overnight exposure.

All market data enters through ``src.data_feed`` loaders. OKX perpetual and
Alpaca SOXL spot-proxy bars share the same causal research engine; the research
package itself performs no direct HTTP/SQLite access.
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
    build_causal_audit_v2,
    build_signal_attempts_v2,
    build_sweep_events_v2,
    filter_liquidity_mode_v2,
    summarize_by_groups,
)
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE  # noqa: E402
from src.data_feed.alpaca_stock_loader import AlpacaStockLoader  # noqa: E402
from src.research_common.ict.spot_perp_overlap import (  # noqa: E402
    build_equity_proxy_data_quality_table,
    densify_equity_minutes_causally,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

DEFAULT_SYMBOL = "SOXL-USDT-SWAP"
# OKX listed SOXL-USDT-SWAP on 2026-05-19 09:00 UTC. The first complete
# requested 04:00-08:30 New York premarket window is therefore 2026-05-20.
DEFAULT_START_DATE = "2026-05-20"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r02_state_machine"
LIQUIDITY_MODES = ("extremes_only", "extremes_plus_strong_15m_swing")


def _csv_numbers(text: str, *, cast=float) -> list[Any]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    if not values:
        raise ValueError(f"empty numeric list: {text!r}")
    return values


def _csv_names(text: str) -> list[str]:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("empty name list")
    return values


def _source_offset_hours(text: str) -> int:
    value = str(text).strip().upper().replace("UTC", "")
    if value.startswith("+") or value.startswith("-"):
        try:
            return int(value)
        except ValueError:
            pass
    return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOXL ICT R02 sweep-episode -> dynamic MSS -> displacement FVG causal research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-source", choices=("okx", "alpaca"), default="okx")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="OKX perpetual symbol when --data-source=okx")
    p.add_argument("--alpaca-symbol", default="SOXL", help="US-stock symbol when --data-source=alpaca")
    p.add_argument("--alpaca-feed", choices=("sip", "iex", "boats"), default="sip")
    p.add_argument("--alpaca-adjustment", default="split", help="Use split for multi-year SOXL research")
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="New York calendar date, inclusive")
    p.add_argument("--end-date", default=DEFAULT_END_DATE, help="New York calendar date, inclusive")
    p.add_argument("--data-dir", default="data", help="Passed to src.data_feed loader; research never opens DB directly")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true", help="Use only the selected loader's local cache; do not fetch missing history")
    p.add_argument("--include-us-equity-holidays", action="store_true", help="By default full NYSE-style holidays are excluded in addition to weekends")
    p.add_argument("--required-day-coverage", type=float, default=0.995)

    p.add_argument("--execution-timeframes", default="1,2,5", help="Execution FVG/MSS timeframes in minutes; all are aggregated from 1m")
    p.add_argument("--liquidity-modes", default=",".join(LIQUIDITY_MODES), choices=None)
    p.add_argument("--premarket-pivot-left", type=int, default=2)
    p.add_argument("--premarket-pivot-right", type=int, default=2)
    p.add_argument("--mss-pivot-left", type=int, default=1)
    p.add_argument("--mss-pivot-right", type=int, default=1)
    p.add_argument("--displacement-body-mult", type=float, default=1.50)
    p.add_argument("--displacement-body-window", type=int, default=20)
    p.add_argument("--displacement-min-periods", type=int, default=10)
    p.add_argument("--displacement-close-location", type=float, default=0.75)
    p.add_argument("--displacement-sensitivity", default="1.25,1.50,1.75")

    p.add_argument("--round-trip-cost", type=float, default=0.0011, help="Project conservative default: 0.11% open+close")
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
        raise ValueError("execution_timeframes must be a subset of 1,2,5 for this research")
    modes = _csv_names(args.liquidity_modes)
    invalid_modes = sorted(set(modes) - set(LIQUIDITY_MODES))
    if invalid_modes:
        raise ValueError(f"invalid liquidity modes: {invalid_modes}")
    if not (0.90 <= float(args.required_day_coverage) <= 1.0):
        raise ValueError("required_day_coverage must be in [0.90, 1.0]")
    if not (0.5 <= float(args.displacement_close_location) < 1.0):
        raise ValueError("displacement_close_location must be in [0.5, 1.0)")
    if float(args.round_trip_cost) < 0:
        raise ValueError("round_trip_cost must be >= 0")
    if float(args.risk_fraction) <= 0 or float(args.max_notional_multiple) <= 0:
        raise ValueError("risk_fraction and max_notional_multiple must be positive")


def _config(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        displacement_body_mult=float(args.displacement_body_mult),
        displacement_body_window=int(args.displacement_body_window),
        displacement_min_periods=int(args.displacement_min_periods),
        displacement_close_location=float(args.displacement_close_location),
        mss_pivot_left=int(args.mss_pivot_left),
        mss_pivot_right=int(args.mss_pivot_right),
        premarket_pivot_left=int(args.premarket_pivot_left),
        premarket_pivot_right=int(args.premarket_pivot_right),
        required_day_coverage=float(args.required_day_coverage),
        round_trip_cost=float(args.round_trip_cost),
        risk_fraction=float(args.risk_fraction),
        max_notional_multiple=float(args.max_notional_multiple),
    )


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.data_source) == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny_exclusive = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        start_utc = start_ny.tz_convert("UTC")
        end_utc = end_ny_exclusive.tz_convert("UTC") - pd.Timedelta(minutes=1)
        loader = AlpacaStockLoader(
            symbol=args.alpaca_symbol,
            timeframe="1Min",
            feed=args.alpaca_feed,
            adjustment=args.alpaca_adjustment,
            data_dir=args.data_dir,
        )
        print(
            f"[load] src.data_feed.AlpacaStockLoader symbol={args.alpaca_symbol} timeframe=1Min "
            f"feed={args.alpaca_feed} adjustment={args.alpaca_adjustment} "
            f"NY={args.start_date}->{args.end_date}",
            flush=True,
        )
        raw = loader.fetch_data_by_date_range(start_utc, end_utc, local_only=bool(args.local_only))
        if raw.empty:
            raise RuntimeError(
                "No SOXL 1m spot data returned by src.data_feed.AlpacaStockLoader. "
                "Check the local cache/table and requested feed/adjustment."
            )
        bars_ny = raw.copy()
        idx = pd.DatetimeIndex(bars_ny.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        bars_ny.index = idx.tz_convert(NY_TZ)
        bars_ny.index.name = "bar_start_ny"
    else:
        source_offset = _source_offset_hours(OKX_LOADER_TIMEZONE)
        start_source, end_source = ny_date_bounds_to_source_naive(
            args.start_date,
            args.end_date,
            source_offset_hours=source_offset,
        )
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        print(
            f"[load] src.data_feed.OKXDataLoader symbol={args.symbol} timeframe=1m "
            f"NY={args.start_date}->{args.end_date} source={start_source}->{end_source}",
            flush=True,
        )
        if args.local_only:
            raw = loader.load_local_data()
            if not raw.empty:
                raw = raw.loc[(raw.index >= start_source) & (raw.index <= end_source)].copy()
        else:
            raw = loader.fetch_data_by_date_range(start_source, end_source)
        if raw.empty:
            raise RuntimeError(
                "No SOXL 1m perpetual data returned by src.data_feed.OKXDataLoader. "
                "The research does not implement a fallback data interface."
            )
        bars_ny = source_naive_to_new_york(raw, source_offset_hours=source_offset)

    # Strategy context deliberately ignores the derivative's extra 24h session.
    # All downstream builders use only NY 04:00-16:30; retaining only that
    # interval here also makes spot/perpetual research directly comparable.
    mins = bars_ny.index.hour * 60 + bars_ny.index.minute
    session_mask = (mins >= 4 * 60) & (mins < 16 * 60 + 30)
    bars_ny = bars_ny.loc[session_mask].copy()
    if str(args.data_source) == "alpaca":
        bars_ny = densify_equity_minutes_causally(bars_ny)
    print(
        f"[load] source={args.data_source} rows={len(bars_ny):,} NY={bars_ny.index.min()} -> {bars_ny.index.max()}",
        flush=True,
    )
    return bars_ny


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[write] {path.name}", flush=True)


def _research_design(args: argparse.Namespace, cfg: ResearchConfig, valid_days: int) -> str:
    return f"""# SOXL ICT Sweep Episode -> Dynamic MSS -> Displacement FVG R02

## Research question
Does the corrected ICT-style sequence have a measurable edge on SOXL?

## Frozen session definition
- Time zone: `America/New_York`, DST-aware.
- Premarket liquidity window: `04:00 <= t < 08:30`.
- Trading window: `08:30 <= t < 16:30`.
- Weekends excluded.
- Full US-equity holidays excluded by default: `{not bool(args.include_us_equity_holidays)}`.
- Valid session coverage threshold: `{cfg.required_day_coverage:.3f}`.
- Valid sessions in this run: `{valid_days}`.

## Liquidity definition
1. `extremes_only`: absolute premarket high and low.
2. `extremes_plus_strong_15m_swing`: extremes plus the strongest internal 15m swing on each side only when it is genuinely obvious.

The internal 15m swing still uses causal left/right = `{cfg.premarket_pivot_left}/{cfg.premarket_pivot_right}` confirmation, but R02 no longer calls the best-looking pivot "major" automatically. Its minimum two-sided excursion must be at least one median completed premarket 15m range. Rejected weak pivots remain visible in the liquidity-level report with `tradable_level=False`.

## Fresh-liquidity rule
The opposite absolute premarket extreme is the target. If that target was already traded through before or during the current sweep bar, the setup is rejected as consumed liquidity. A later MSS cannot revive that old target.

## Sweep-episode state machine
A first cross of a tradable premarket level opens a sweep episode. For each execution timeframe (`{','.join(str(x) + 'm' for x in cfg.execution_timeframes)}`):
1. Track the most adverse completed 1m price reached since the sweep; this is the current terminal extreme.
2. Recompute the relevant opposing short-term pivot from causally confirmed structure that lies before the current terminal extreme.
3. A direct V reversal may use the latest valid pre-sweep pivot.
4. If price later makes a more extreme low/high after forming a newer pivot, that newer pivot replaces the old reference, producing the intended W/M-style dynamic MSS structure.
5. Bullish MSS requires a completed execution bar close above that dynamic short-term high; bearish MSS mirrors it.

## Displacement + FVG
R02 intentionally keeps R01's strict displacement/FVG semantics so this revision isolates the structure fix:
- break-bar body >= `{cfg.displacement_body_mult:.2f}x` prior `{cfg.displacement_body_window}`-bar median body;
- close in the reversal-direction outer `{(1.0 - cfg.displacement_close_location) * 100:.0f}%`;
- break bar is the third candle of a strict three-candle FVG;
- the FVG sequence begins only after the current terminal extreme is already observable.

## Execution
- Entry: FVG third-candle near edge.
- Stop: sweep-episode terminal extreme through signal completion.
- Target: fresh opposite absolute premarket extreme.
- Pending cancellation: target reached first, terminal extreme invalidated first, or 16:30.
- No overnight exposure.
- Same-minute path ambiguity is resolved against the strategy.

## Costs and robustness
- Base round-trip cost: `{cfg.round_trip_cost:.4%}`.
- Cost stress: `{args.cost_multipliers}x`.
- Order activation delay stress: `{args.order_delay_minutes}` minutes.
- Displacement sensitivity remains diagnostic only: `{args.displacement_sensitivity}x`; it is not used to optimize this short sample.

## Causality
Every 2m/5m/15m bar has explicit available time. Dynamic pivots need only be confirmed by signal time, never by hindsight after the signal. The terminal extreme is built only from completed 1m bars available by that signal.
"""

def _summaries_for_run(
    bars_ny: pd.DataFrame,
    attempts: pd.DataFrame,
    *,
    args: argparse.Namespace,
    cfg: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modes = _csv_names(args.liquidity_modes)
    cost_mults = _csv_numbers(args.cost_multipliers, cast=float)
    delays = _csv_numbers(args.order_delay_minutes, cast=int)
    base_mult = float(cfg.displacement_body_mult)
    sensitivity_mults = _csv_numbers(args.displacement_sensitivity, cast=float)

    all_lifecycle_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    total_jobs = len(cfg.execution_timeframes) * len(modes) * (
        len(cost_mults) + max(0, len(delays) - 1) + max(0, len(sensitivity_mults) - 1)
    )
    progress = ProgressReporter(
        label="[replay] variants/stress",
        total=max(total_jobs, 1),
        every=1,
        enabled=not bool(args.no_progress),
    )
    done = 0

    def run_one(tf: int, mode: str, disp: float, scenario: ReplayScenario, family: str) -> None:
        nonlocal done
        subset = attempts.loc[
            (attempts["execution_tf_minutes"] == int(tf))
            & np.isclose(pd.to_numeric(attempts["displacement_body_mult"], errors="coerce"), float(disp))
        ].copy()
        subset = filter_liquidity_mode_v2(subset, mode)
        replayed = replay_attempts(
            bars_ny,
            subset,
            scenario=scenario,
            round_trip_cost=cfg.round_trip_cost,
            risk_fraction=cfg.risk_fraction,
            max_notional_multiple=cfg.max_notional_multiple,
        )
        kept, skipped = enforce_single_lifecycle(replayed)
        if not kept.empty:
            kept["liquidity_mode"] = mode
            kept["stress_family"] = family
            kept["variant_key"] = f"tf={tf}m|liq={mode}|disp={disp:.2f}|scenario={scenario.name}"
            all_lifecycle_parts.append(kept)
        rec = {
            "execution_tf": f"{tf}m",
            "execution_tf_minutes": int(tf),
            "liquidity_mode": mode,
            "displacement_body_mult": float(disp),
            "scenario": scenario.name,
            "stress_family": family,
            "cost_multiple": float(scenario.cost_multiple),
            "order_delay_minutes": int(scenario.order_delay_minutes),
            **summarize_variant(kept, skipped_overlap=skipped, initial_capital=float(args.initial_capital)),
        }
        summary_rows.append(rec)
        done += 1
        progress.update(done)

    for tf in cfg.execution_timeframes:
        for mode in modes:
            for cost_mult in cost_mults:
                name = "base" if abs(cost_mult - 1.0) < 1e-12 else f"cost_{cost_mult:g}x"
                run_one(tf, mode, base_mult, ReplayScenario(name, cost_mult, 0), "cost")
            for delay in delays:
                if int(delay) == 0:
                    continue
                run_one(tf, mode, base_mult, ReplayScenario(f"delay_{delay}m", 1.0, int(delay)), "delay")
            for disp in sensitivity_mults:
                if abs(float(disp) - base_mult) < 1e-12:
                    continue
                run_one(tf, mode, float(disp), BASE_SCENARIO, "displacement_sensitivity")

    progress.close()
    lifecycle = pd.concat(all_lifecycle_parts, ignore_index=True) if all_lifecycle_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return lifecycle, summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base = summary.loc[
        np.isclose(pd.to_numeric(summary["displacement_body_mult"], errors="coerce"), base_mult)
        & (summary["scenario"] == "base")
        & (summary["stress_family"] == "cost")
    ].copy()
    cost = summary.loc[summary["stress_family"] == "cost"].copy()
    delay = summary.loc[summary["stress_family"] == "delay"].copy()
    disp = summary.loc[
        (summary["stress_family"] == "displacement_sensitivity")
        | (
            np.isclose(pd.to_numeric(summary["displacement_body_mult"], errors="coerce"), base_mult)
            & (summary["scenario"] == "base")
            & (summary["stress_family"] == "cost")
        )
    ].copy()
    return lifecycle, base, cost, delay, disp


def _base_lifecycle(lifecycle: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    if lifecycle.empty:
        return lifecycle.copy()
    return lifecycle.loc[
        np.isclose(pd.to_numeric(lifecycle["displacement_body_mult"], errors="coerce"), cfg.displacement_body_mult)
        & (lifecycle["scenario"] == "base")
        & (lifecycle["stress_family"] == "cost")
    ].copy()


def _emit_platform_reports(
    base_lifecycle: pd.DataFrame,
    bars_ny: pd.DataFrame,
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    if base_lifecycle.empty or bool(args.skip_platform_reports):
        return
    report_root = out_dir / "platform_full_reports"
    reporter = ProgressReporter(
        label="[report] print_full_report variants",
        total=base_lifecycle.groupby(["execution_tf", "liquidity_mode"]).ngroups,
        every=1,
        enabled=not bool(args.no_progress),
    )
    done = 0
    for (tf, mode), group in base_lifecycle.groupby(["execution_tf", "liquidity_mode"], sort=True):
        filled = group.loc[group["filled"].fillna(False).astype(bool)].copy()
        if filled.empty:
            done += 1
            reporter.update(done)
            continue
        account = compound_account(filled, initial_capital=float(args.initial_capital))
        history = report_trade_history(account)
        if not history:
            done += 1
            reporter.update(done)
            continue
        strategy_name = f"SOXL_ICT_R02_{tf}_{mode}"
        total_days = max((bars_ny.index.max() - bars_ny.index.min()).total_seconds() / 86400.0, 1.0)
        print_full_report(
            trade_history=history,
            df=bars_ny,
            initial_capital=float(args.initial_capital),
            capital=float(account["capital"].iloc[-1]),
            strategy_name=strategy_name,
            total_days=total_days,
            ai_enabled=False,
            symbol=args.symbol,
            report_dir=report_root / f"{tf}_{mode}",
        )
        done += 1
        reporter.update(done)
    reporter.close()


def _findings_markdown(
    *,
    args: argparse.Namespace,
    valid_days: int,
    levels: pd.DataFrame,
    sweeps: pd.DataFrame,
    attempts: pd.DataFrame,
    base_summary: pd.DataFrame,
    causal_audit: pd.DataFrame,
) -> str:
    strong = int(levels.get("tradable_level", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not levels.empty else 0
    stale = int((~sweeps.get("opposite_target_fresh_at_sweep", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not sweeps.empty else 0
    lines = [
        "# R02 Findings",
        "",
        "## Structural corrections",
        "- R01 frozen-at-sweep MSS reference has been replaced by a dynamic sweep-episode terminal-extreme state machine.",
        "- Opposite premarket target must be fresh when the sweep occurs.",
        "- Internal 15m pivots are no longer forced into a major-swing label; weak candidates are reported but not traded.",
        "- Aggregate diagnostics are split by liquidity mode so duplicate extreme trades are not pooled across variants.",
        "",
        "## Sample",
        f"- Valid fully-covered sessions: **{valid_days}**.",
        f"- Reported liquidity rows: **{len(levels)}**; tradable rows: **{strong}**.",
        f"- First-sweep events: **{len(sweeps)}**; sweeps rejected because opposite target was already consumed: **{stale}**.",
        f"- Dynamic MSS+displacement+FVG attempts: **{len(attempts)}**.",
        "",
        "## Causality",
    ]
    if causal_audit.empty:
        lines.append("- No causal audit rows were generated.")
    else:
        failed = causal_audit.loc[~causal_audit["passed"].fillna(False).astype(bool)]
        lines.append("- All implemented state-machine / available-time checks passed." if failed.empty else f"- **FAILED** causal checks: {', '.join(failed['check'].astype(str))}.")
    lines += ["", "## Base variants"]
    if base_summary.empty:
        lines.append("- No base variant produced a replay summary.")
    else:
        ranked = base_summary.copy()
        ranked["_pf"] = pd.to_numeric(ranked.get("profit_factor"), errors="coerce")
        ranked["_mean"] = pd.to_numeric(ranked.get("mean_net_return"), errors="coerce")
        ranked = ranked.sort_values(["_pf", "_mean", "filled_trades"], ascending=[False, False, False], na_position="last")
        for row in ranked.head(8).to_dict("records"):
            lines.append(
                f"- `{row['execution_tf']} / {row['liquidity_mode']}`: "
                f"trades={int(row.get('filled_trades', 0) or 0)}, "
                f"win={float(row.get('win_rate', np.nan)):.1%}, PF={float(row.get('profit_factor', np.nan)):.3f}, "
                f"mean net={float(row.get('mean_net_return', np.nan)):.3%}, "
                f"account={float(row.get('account_total_return', np.nan)):.2%}, MDD={float(row.get('account_max_drawdown', np.nan)):.2%}."
            )
    lines += [
        "",
        "## Decision rule",
        "Do not promote or tune on the short OKX sample. First compare R02 against R01, then extend history with an independently audited SOXL spot proxy and validate spot/perpetual structural agreement on the overlapping 2026 sample.",
    ]
    return "\n".join(lines) + "\n"

def run_research(bars_ny: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    cfg = _config(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_days = eligible_ny_dates(
        bars_ny,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_equity_holidays=not bool(args.include_us_equity_holidays),
    )
    if str(args.data_source) == "alpaca":
        quality = build_equity_proxy_data_quality_table(bars_ny, all_days)
    else:
        quality = build_data_quality_table(
            bars_ny,
            all_days,
            required_coverage=cfg.required_day_coverage,
        )
    valid_day_text = set(quality.loc[quality["coverage_pass"], "ny_date"].astype(str)) if not quality.empty else set()
    valid_days = [pd.Timestamp(x).date() for x in sorted(valid_day_text)]
    if not valid_days:
        raise RuntimeError("No fully covered eligible New York sessions after data-quality gate")
    print(f"[coverage] eligible={len(all_days)} valid={len(valid_days)} threshold={cfg.required_day_coverage:.3f}", flush=True)

    stage = ProgressReporter(label="[research] build stages", total=5, every=1, enabled=not bool(args.no_progress))
    levels = build_all_premarket_levels_v2(
        bars_ny,
        valid_days,
        pivot_left=cfg.premarket_pivot_left,
        pivot_right=cfg.premarket_pivot_right,
        episode_config=SweepEpisodeConfig(),
    )
    stage.update(1)
    sweeps = build_sweep_events_v2(bars_ny, levels)
    stage.update(2)
    sensitivity = sorted(set(_csv_numbers(args.displacement_sensitivity, cast=float) + [cfg.displacement_body_mult]))
    attempts = build_signal_attempts_v2(
        bars_ny,
        sweeps,
        config=cfg,
        displacement_body_multipliers=sensitivity,
        progress_enabled=not bool(args.no_progress),
    )
    stage.update(3)
    if attempts.empty:
        print("[research] no strict MSS+FVG attempts; writing diagnostic artifacts", flush=True)
    lifecycle, base_summary, cost_stress, delay_stress, disp_stress = _summaries_for_run(
        bars_ny,
        attempts,
        args=args,
        cfg=cfg,
    ) if not attempts.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    stage.update(4)
    base_lifecycle = add_analysis_dimensions(_base_lifecycle(lifecycle, cfg))
    causal_audit = build_causal_audit_v2(
        attempts.loc[np.isclose(pd.to_numeric(attempts.get("displacement_body_mult", pd.Series(dtype=float)), errors="coerce"), cfg.displacement_body_mult)].copy()
        if not attempts.empty else attempts,
        base_lifecycle,
    )
    stage.update(5)
    stage.close()

    design = _research_design(args, cfg, len(valid_days))
    (out_dir / "00_research_design.md").write_text(design, encoding="utf-8")
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(levels, out_dir / "02_premarket_liquidity_levels.csv")
    _write_csv(sweeps, out_dir / "03_sweep_events.csv")
    _write_csv(attempts, out_dir / "04_signal_attempts.csv")
    _write_csv(base_lifecycle, out_dir / "05_base_trade_lifecycle.csv")
    _write_csv(base_summary, out_dir / "06_base_variant_summary.csv")
    _write_csv(cost_stress, out_dir / "07_cost_stress.csv")
    _write_csv(disp_stress, out_dir / "08_displacement_sensitivity.csv")
    _write_csv(delay_stress, out_dir / "09_order_delay_stress.csv")

    filled_base = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy() if not base_lifecycle.empty else pd.DataFrame()
    _write_csv(summarize_by_groups(filled_base, ["liquidity_mode", "execution_tf"]), out_dir / "10_execution_timeframe_compare.csv")
    _write_csv(summarize_by_groups(filled_base, ["liquidity_mode", "level_type"]), out_dir / "11_liquidity_level_type_compare.csv")
    _write_csv(summarize_by_groups(filled_base, ["liquidity_mode", "trade_side"]), out_dir / "12_long_short_compare.csv")
    _write_csv(summarize_by_groups(filled_base, ["liquidity_mode", "weekday"]), out_dir / "13_weekday_compare.csv")
    _write_csv(summarize_by_groups(filled_base, ["liquidity_mode", "sweep_time_bucket"]), out_dir / "14_sweep_time_bucket_compare.csv")
    _write_csv(summarize_by_groups(filled_base, ["liquidity_mode", "mss_reference_source"]), out_dir / "15_mss_reference_source_compare.csv")
    _write_csv(causal_audit, out_dir / "16_causal_audit.csv")

    findings = _findings_markdown(
        args=args,
        valid_days=len(valid_days),
        levels=levels,
        sweeps=sweeps,
        attempts=attempts,
        base_summary=base_summary,
        causal_audit=causal_audit,
    )
    (out_dir / "17_findings.md").write_text(findings, encoding="utf-8")

    manifest = {
        "experiment_id": "SOXL_ICT_PREMARKET_MSS_FVG_R02",
        "edge_id": "SOXL_ICT_PM_SWEEP_MSS_FVG",
        "title": "SOXL ICT Sweep Episode -> Dynamic MSS -> Displacement FVG R02",
        "symbol": args.symbol,
        "new_york_start_date": args.start_date,
        "new_york_end_date": args.end_date,
        "data_source": str(args.data_source),
        "source_loader": (
            "src.data_feed.alpaca_stock_loader.AlpacaStockLoader"
            if str(args.data_source) == "alpaca"
            else "src.data_feed.okx_loader.OKXDataLoader"
        ),
        "source_symbol": args.alpaca_symbol if str(args.data_source) == "alpaca" else args.symbol,
        "source_feed": args.alpaca_feed if str(args.data_source) == "alpaca" else "okx",
        "source_adjustment": args.alpaca_adjustment if str(args.data_source) == "alpaca" else "n/a",
        "source_timeframe": "1m",
        "source_timezone_policy": (
            "UTC tz-aware -> America/New_York DST-aware"
            if str(args.data_source) == "alpaca"
            else f"project-local naive {OKX_LOADER_TIMEZONE} -> America/New_York DST-aware"
        ),
        "strategy_session_clip_ny": "04:00-16:30",
        "valid_sessions": len(valid_days),
        "premarket_window_ny": "04:00-08:30",
        "trade_window_ny": "08:30-16:30",
        "execution_timeframes": list(cfg.execution_timeframes),
        "liquidity_modes": _csv_names(args.liquidity_modes),
        "base_displacement_body_mult": cfg.displacement_body_mult,
        "displacement_sensitivity": sensitivity,
        "round_trip_cost": cfg.round_trip_cost,
        "cost_multipliers": _csv_numbers(args.cost_multipliers, cast=float),
        "order_delay_minutes": _csv_numbers(args.order_delay_minutes, cast=int),
        "risk_fraction": cfg.risk_fraction,
        "max_notional_multiple": cfg.max_notional_multiple,
        "causal_audit_passed": bool(not causal_audit.empty and causal_audit["passed"].fillna(False).all()),
        "known_limitation": (
            "Alpaca SOXL spot is a long-history proxy, not the OKX perpetual; require the separate overlap audit before treating long-history findings as transferable."
            if str(args.data_source) == "alpaca"
            else "OKX SOXL perpetual listed 2026-05-19; use a separately audited SOXL spot proxy for longer history only after overlap validation."
        ),
        "r02_semantic_fix": "dynamic sweep episode terminal extreme + fresh opposite liquidity + strong 15m swing gate",
        "decision_policy": "research_continue unless future/independent history confirms robustness; do not optimize this short sample",
    }
    _write_json(manifest, out_dir / "18_manifest.json")

    _emit_platform_reports(base_lifecycle, bars_ny, args=args, out_dir=out_dir)
    if not bool(args.skip_review_pack):
        finalize_research_report(
            out_dir,
            experiment_id=manifest["experiment_id"],
            edge_id=manifest["edge_id"],
            title=manifest["title"],
            print_log=True,
        )
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] SOXL ICT R02 sweep episode/dynamic MSS/FVG", flush=True)
    bars = make_synthetic_ict_day()
    with tempfile.TemporaryDirectory(prefix="soxl_ict_r02_") as tmp:
        args.start_date = "2026-06-02"
        args.end_date = "2026-06-02"
        args.out_dir = tmp
        args.include_us_equity_holidays = True
        args.required_day_coverage = 1.0
        args.execution_timeframes = "1,2,5"
        args.displacement_sensitivity = "1.25,1.50,1.75"
        args.cost_multipliers = "1,2"
        args.order_delay_minutes = "0,1"
        args.skip_platform_reports = True
        args.skip_review_pack = True
        args.no_progress = True
        result = run_research(bars, args)
        required = [
            "00_research_design.md",
            "01_data_quality.csv",
            "02_premarket_liquidity_levels.csv",
            "03_sweep_events.csv",
            "16_causal_audit.csv",
            "18_manifest.json",
        ]
        missing = [name for name in required if not (result["report_dir"] / name).exists()]
        if missing:
            raise AssertionError(f"self-test missing artifacts: {missing}")
        audit = pd.read_csv(result["report_dir"] / "16_causal_audit.csv")
        if not audit.empty and not bool(audit["passed"].astype(bool).all()):
            raise AssertionError(f"self-test causal audit failed:\n{audit}")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    if args.self_test:
        return run_self_test(args)
    bars_ny = _load_1m(args)
    run_research(bars_ny, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
