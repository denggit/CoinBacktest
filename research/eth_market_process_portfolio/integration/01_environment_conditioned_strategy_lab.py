#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02: environment-conditioned ETH strategy lab.

The study pre-declares three complete market processes and replays causal
entries/exits with realistic costs:

1. ``compression_breakout``: low-volatility balance -> directional expansion.
2. ``expansion_exhaustion``: extended expansion -> absorption/reclaim reversal.
3. ``balance_failed_auction``: balanced auction -> failed edge excursion.

Data access is exclusively through ``src.data_feed`` loaders. Trade bars and
range bars use the project's timezone-aligned (tzplus8) local caches. Range-bar
features are available only at ``end_ts``. The study loads one calendar year at
a time with bounded warmup/tail overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.research_common.market_process.environment_features import (  # noqa: E402,F401
    DEFINITIONS,
    Definition,
    apply_cooldown,
    build_market_features,
    build_range_context,
    build_strategy_candidates,
)
from src.research_common.market_process.strategy_replay import (  # noqa: E402,F401
    FAMILY_EXITS,
    SCENARIOS,
    BarArrays,
    LabConfig,
    Scenario,
    _promotion_table,
    build_bar_arrays,
    build_period_tables,
    enforce_nonoverlap,
    simulate_candidate,
    simulate_candidates,
    summarize_trades,
)
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.trade_bar_orderflow import validate_trade_bar_orderflow  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--range-pct", type=float, default=0.0020)
    parser.add_argument("--baseline-window", type=int, default=240)
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--max-notional-multiple", type=float, default=2.0)
    parser.add_argument("--cooldown-minutes", type=int, default=15)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "data"
        / "reports"
        / "research"
        / "eth_market_process_portfolio"
        / "integration"
        / "01_environment_conditioned_strategy_lab",
    )
    parser.add_argument("--no-candidate-export", action="store_true")
    return parser.parse_args(argv)


def _year_windows(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = pd.Timestamp(year=start.year, month=1, day=1)
    while cursor <= end:
        left = max(start, cursor)
        right = min(end, pd.Timestamp(year=cursor.year, month=12, day=31, hour=23, minute=59, second=59))
        if left <= right:
            yield left, right
        cursor = pd.Timestamp(year=cursor.year + 1, month=1, day=1)


def _write_report(
    path: Path,
    cfg: LabConfig,
    overview: pd.DataFrame,
    promotion: pd.DataFrame,
    yearly: pd.DataFrame,
    field_coverage: pd.DataFrame,
    candidate_count: int,
    trade_count: int,
) -> None:
    base = (
        overview[(overview["scenario"] == "base") & (overview["definition"] == "base")]
        if not overview.empty
        else pd.DataFrame()
    )
    lines = [
        "# R02 Environment-Conditioned Strategy Lab",
        "",
        "## Scope",
        "",
        f"- Symbol: `{cfg.symbol}`",
        f"- Window: `{cfg.start}` to `{cfg.end}`",
        "- Trade bars: `OKXTradeBarLoader`, timezone-aligned `tzplus8` cache",
        f"- Range bars: `OKXRangeBarLoader`, range `{cfg.range_pct:.4%}`, available at `end_ts`",
        "- Signal: closed 1m bar; execution: next 1m open",
        f"- Base round-trip cost: `{cfg.round_trip_cost:.4%}`",
        f"- Risk per trade: `{cfg.risk_fraction:.2%}` equity, notional cap `{cfg.max_notional_multiple:.2f}x`",
        f"- Candidate events: `{candidate_count:,}`; simulated trades across scenarios/definitions: `{trade_count:,}`",
        "",
        "This is a complete strategy screen, not a fixed-horizon event study. It uses structural stops, mechanism targets, causal trailing updates, pessimistic same-bar ambiguity and fail-safe maximum holding periods.",
        "",
        "## Frozen strategy families",
        "",
        "- `compression_breakout`: prior compression plus directional breakout and effective aligned order flow.",
        "- `expansion_exhaustion`: prior expansion plus sweep, absorption and reclaim-style flow reversal.",
        "- `balance_failed_auction`: prior balanced auction plus failed edge excursion and order-flow reversal.",
        "",
        "## Primary base results",
        "",
        base.to_markdown(index=False) if not base.empty else "No primary trades.",
        "",
        "## Promotion screen",
        "",
        promotion.to_markdown(index=False) if not promotion.empty else "No promotion rows.",
        "",
        "A family is only screened forward when PF >= 1.15, base/2x-fee/1m-delay/2026 holdout returns are positive, frequency is adequate, and removing its ten best trades does not destroy the result. Passing this screen is not yet live approval.",
        "",
        "## Base yearly results",
        "",
    ]
    base_year = (
        yearly[(yearly["scenario"] == "base") & (yearly["definition"] == "base")]
        if not yearly.empty
        else pd.DataFrame()
    )
    lines.append(base_year.to_markdown(index=False) if not base_year.empty else "No yearly rows.")
    lines.extend(["", "## Trade-bar field coverage", ""])
    lines.append(field_coverage.to_markdown(index=False) if not field_coverage.empty else "No coverage rows.")
    lines.extend(
        [
            "",
            "## Causal and robustness rules",
            "",
            "- Range context uses only completed bars whose `end_ts <= signal_time`.",
            "- Trailing changes observed on a bar activate on the following bar.",
            "- Stop wins when stop and target are both touched inside one 1m bar.",
            "- `loose/base/strict` are coherent neighbouring definitions; they are reported together and are not selected by best return.",
            "- No Books, OI, funding or liquidation filters are used in R02.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    cfg = LabConfig(
        symbol=args.symbol,
        start=args.start_date,
        end=args.end_date,
        range_pct=float(args.range_pct),
        baseline_window=int(args.baseline_window),
        round_trip_cost=float(args.round_trip_cost),
        risk_fraction=float(args.risk_fraction),
        max_notional_multiple=float(args.max_notional_multiple),
        cooldown_minutes=int(args.cooldown_minutes),
    )
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    trade_loader = OKXTradeBarLoader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        align_with_okx_loader_timezone=True,
    )
    range_loader = OKXRangeBarLoader(
        symbol=cfg.symbol,
        range_pct=cfg.range_pct,
        align_with_okx_loader_timezone=True,
    )

    print("[run] ETH Market Process Portfolio - Environment Strategy Lab R02")
    print(f"[window] {start} -> {end} timezone=tzplus8")
    print(f"[data] trade_bars=1m range_bars={cfg.range_pct:.4%} local-cache-only")
    print(f"[cost] round_trip={cfg.round_trip_cost:.4%} risk={cfg.risk_fraction:.2%} notional_cap={cfg.max_notional_multiple:.2f}x")

    all_candidates: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    field_coverage: pd.DataFrame | None = None
    windows = list(_year_windows(start, end))
    for chunk_number, (left, right) in enumerate(windows, start=1):
        load_left = left - pd.Timedelta(days=cfg.warmup_days)
        load_right = right + pd.Timedelta(minutes=cfg.tail_minutes)
        print(f"[chunk {chunk_number}/{len(windows)}] load {load_left} -> {load_right}")
        trade_bars = trade_loader.fetch_data_by_date_range(
            load_left,
            load_right,
            cvd_mode="range",
            build_missing=False,
        )
        if trade_bars.empty:
            raise RuntimeError(f"no local trade bars for {load_left} -> {load_right}")
        # Public local-cache API: never trigger downloads/builds from research.
        range_bars = range_loader.load_local_data(load_left - pd.Timedelta(hours=2), load_right)
        if range_bars.empty:
            raise RuntimeError(f"no local range bars for {load_left} -> {load_right}")
        if field_coverage is None:
            field_coverage = validate_trade_bar_orderflow(trade_bars)

        print(f"[chunk {chunk_number}/{len(windows)}] trade_rows={len(trade_bars):,} range_rows={len(range_bars):,} build_features")
        features = build_market_features(trade_bars, range_bars, baseline_window=cfg.baseline_window)
        chunk_candidates: list[pd.DataFrame] = []
        for definition in DEFINITIONS:
            candidate = build_strategy_candidates(features, definition)
            if not candidate.empty:
                candidate = candidate[(candidate["signal_time"] >= left) & (candidate["signal_time"] <= right)].copy()
                chunk_candidates.append(candidate)
        candidates = pd.concat(chunk_candidates, ignore_index=True) if chunk_candidates else pd.DataFrame()
        candidates = apply_cooldown(candidates, cfg.cooldown_minutes)
        all_candidates.append(candidates)
        counts = candidates.groupby(["definition", "family"]).size().to_dict() if not candidates.empty else {}
        print(f"[chunk {chunk_number}/{len(windows)}] candidates={len(candidates):,} groups={counts}")

        trades = simulate_candidates(
            candidates,
            features,
            SCENARIOS,
            cfg,
            progress_label=f"[chunk {chunk_number}/{len(windows)}] replay",
        )
        all_trades.append(trades)
        print(f"[chunk {chunk_number}/{len(windows)}] trades={len(trades):,}")
        del trade_bars, range_bars, features, candidates, trades

    candidates_all = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trades_all.empty:
        trades_all = enforce_nonoverlap(trades_all)
        trades_all = trades_all.sort_values(["scenario", "definition", "family", "entry_time"], kind="stable").reset_index(drop=True)

    overview = summarize_trades(trades_all, ["scenario", "definition", "family"])
    yearly, quarterly, monthly = build_period_tables(trades_all)
    promotion = _promotion_table(overview, yearly)

    report_dir: Path = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(report_dir / "overview.csv", index=False)
    yearly.to_csv(report_dir / "yearly.csv", index=False)
    quarterly.to_csv(report_dir / "quarterly.csv", index=False)
    monthly.to_csv(report_dir / "monthly.csv", index=False)
    promotion.to_csv(report_dir / "promotion_screen.csv", index=False)
    trades_all.to_csv(report_dir / "trades.csv", index=False)
    (field_coverage if field_coverage is not None else pd.DataFrame()).to_csv(report_dir / "field_coverage.csv", index=False)
    if not args.no_candidate_export:
        candidates_all.to_csv(report_dir / "candidates.csv", index=False)
    (report_dir / "run_config.json").write_text(
        json.dumps(
            {
                "lab": asdict(cfg),
                "definitions": [asdict(x) for x in DEFINITIONS],
                "scenarios": [asdict(x) for x in SCENARIOS],
                "family_exits": {key: asdict(value) for key, value in FAMILY_EXITS.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        report_dir / "report.md",
        cfg,
        overview,
        promotion,
        yearly,
        field_coverage if field_coverage is not None else pd.DataFrame(),
        len(candidates_all),
        len(trades_all),
    )

    print(f"[done] candidates={len(candidates_all):,} trades={len(trades_all):,}")
    primary = (
        overview[(overview["scenario"] == "base") & (overview["definition"] == "base")]
        if not overview.empty
        else pd.DataFrame()
    )
    if not primary.empty:
        print(primary.to_string(index=False))
    if not promotion.empty:
        print("[promotion]")
        print(promotion[["family", "screen_pass"]].to_string(index=False))
    print(f"[report] {report_dir / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
