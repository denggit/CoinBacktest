#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01 causal research for liquidity-hunt reversal and liquidity-void momentum.

This is deliberately an event study plus a fixed-rule strategy probe, not a
parameter-mining strategy factory.  It reuses CoinBacktest's public data-feed
loaders and processes Books features one local day at a time.

Causal contract
---------------
1. A range bar is usable only after ``end_ts``.
2. Liquidity context is the latest row with ``available_time <= end_ts``.
3. The paired event uses bar i-1 and completed bar i only.
4. Entry is at the first range-bar open with start_ts strictly after the signal.
5. Dynamic exits are decided on completed range bars and filled next open.
6. If stop and target are both touched in one range bar, stop wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.liquidity.liquidity_hunt_momentum_r01.core import (  # noqa: E402
    BOOK_CONTEXT_COLUMNS,
    LiquidityHuntConfig,
    StrategyVariant,
    aggregate_footprint_features,
    attach_book_context,
    attach_forward_time_outcomes,
    build_causal_audit,
    build_events,
    build_range_features,
    chronological_split_labels,
    datetime_index_to_ns_int64,
    profit_factor,
    prepare_book_features,
    simulate_events,
    summarize_returns,
)
from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader, range_code  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

SCRIPT_NAME = "01_liquidity_hunt_momentum_event_study"
SCRIPT_VERSION = "1.0.2"
EXPERIMENT_ID = "ETH_LIQUIDITY_HUNT_MOMENTUM_R01"
EDGE_ID = "ETH_LIQUIDITY_HUNT_MOMENTUM"
TITLE = "ETH Liquidity Hunt Momentum - Causal Event Study R01"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/liquidity_hunt_momentum_r01"
DEFAULT_RANGE_PCTS = (0.0015, 0.0020, 0.0025)
STRICT_STAGES = ("M1_FLOW_RECLAIM_OBI_REBUILD", "M2_TWO_BAR_ATTACK_OBI_VOID")
FORWARD_HORIZONS = (5, 15, 30, 60)


def _csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one float is required")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal liquidity-hunt reversal and liquidity-void momentum research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2025-09-01")
    p.add_argument("--start-date", default="2025-10-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--range-pcts", default=",".join(map(str, DEFAULT_RANGE_PCTS)))
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--books-depth", type=int, default=5000)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--range-bar-db-name", default="okx_range_bars.db")
    p.add_argument("--range-footprint-db-name", default="okx_range_footprints.db")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--attack-notional-multiple", type=float, default=1.50)
    p.add_argument("--cooldown-minutes", type=int, default=15)
    p.add_argument("--book-tolerance-seconds", type=int, default=10)
    p.add_argument("--skip-footprint", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--skip-full-report", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _iter_months(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start.normalize().replace(day=1)
    while cursor <= end:
        nxt = cursor + pd.offsets.MonthBegin(1)
        yield max(start, cursor), min(end, nxt - pd.Timedelta(microseconds=1))
        cursor = nxt


def _load_range_frames(
    args: argparse.Namespace,
    range_pcts: tuple[float, ...],
    cfg: LiquidityHuntConfig,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    progress = ProgressReporter(
        label="[range-bars] load",
        total=len(range_pcts),
        every=1,
        enabled=not args.no_progress,
    )
    for i, value in enumerate(range_pcts, start=1):
        tag = range_code(value)
        loader = OKXRangeBarLoader(
            symbol=args.symbol,
            range_pct=value,
            data_dir=args.data_dir,
            db_name=args.range_bar_db_name,
        )
        raw = loader.load_local_data(args.warmup_start_date, args.end_date)
        if raw.empty:
            raise RuntimeError(
                f"No local range bars for {tag}. Prebuild them before running this research."
            )
        # OKXRangeBarLoader exposes end_ts both as index and column.  Existing
        # CoinBacktest research normalizes that index before column operations.
        raw = raw.reset_index(drop=True)
        end_limit = pd.Timestamp(args.end_date)
        raw = raw.loc[pd.to_datetime(raw["end_ts"], errors="coerce") <= end_limit].copy()
        if raw.empty:
            raise RuntimeError(f"No completed local range bars within the requested window for {tag}.")
        frames[tag] = build_range_features(raw, cfg)
        progress.update(i)
    progress.close()
    return frames


def _attach_footprints(
    args: argparse.Namespace,
    range_pcts: tuple[float, ...],
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    if args.skip_footprint:
        for tag, frame in frames.items():
            frame["footprint_missing_flag"] = True
            diagnostics[tag] = {"footprint_rows": 0, "footprint_bars": 0, "skipped": True}
        return diagnostics

    start = pd.Timestamp(args.warmup_start_date)
    end = pd.Timestamp(args.end_date)
    months = list(_iter_months(start, end))
    total = len(range_pcts) * len(months)
    progress = ProgressReporter(
        label="[footprint] monthly aggregation",
        total=total,
        every=1,
        enabled=not args.no_progress,
    )
    done = 0
    for value in range_pcts:
        tag = range_code(value)
        loader = OKXRangeFootprintLoader(
            symbol=args.symbol,
            range_pct=value,
            price_step=args.price_step,
            data_dir=args.data_dir,
            db_name=args.range_footprint_db_name,
        )
        parts: list[pd.DataFrame] = []
        raw_rows = 0
        for month_start, month_end in months:
            raw = loader.load_local_data(month_start, month_end)
            raw_rows += len(raw)
            compact = aggregate_footprint_features(raw)
            if not compact.empty:
                parts.append(compact)
            done += 1
            progress.update(done)
        compact_all = (
            pd.concat(parts, ignore_index=True)
            .sort_values("bar_id", kind="stable")
            .drop_duplicates("bar_id", keep="last")
            if parts
            else pd.DataFrame(columns=["bar_id"])
        )
        if compact_all.empty:
            frames[tag]["footprint_missing_flag"] = True
        else:
            frames[tag] = frames[tag].merge(compact_all, on="bar_id", how="left", validate="one_to_one")
            frames[tag]["footprint_missing_flag"] = frames[tag]["fp_total_notional"].isna()
        diagnostics[tag] = {
            "footprint_rows": int(raw_rows),
            "footprint_bars": int(len(compact_all)),
            "skipped": False,
        }
    progress.close()
    return diagnostics


def _attach_books_daily(
    args: argparse.Namespace,
    frames: dict[str, pd.DataFrame],
    cfg: LiquidityHuntConfig,
) -> dict[str, Any]:
    loader = OKXLiquidityMapLoader(
        symbol=args.symbol,
        books_depth=args.books_depth,
        data_dir=args.data_dir,
    )
    # Books are strategy context, not Range-Bar warmup.  Start from the
    # research window and request only the trailing reference lead-in per day.
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date)
    days = list(pd.date_range(start, end.normalize(), freq="D"))
    progress = ProgressReporter(
        label="[books] causal daily alignment",
        total=len(days),
        every=1,
        enabled=not args.no_progress,
    )

    # Pre-create all aligned columns once.  Daily slices are found with
    # searchsorted rather than scanning every range frame for every day.
    frame_end_ns: dict[str, np.ndarray] = {}
    for tag, frame in frames.items():
        end_ts = pd.DatetimeIndex(pd.to_datetime(frame["end_ts"], errors="coerce"))
        if end_ts.hasnans or not end_ts.is_monotonic_increasing:
            raise ValueError(f"{tag} end_ts must be valid and sorted before Books alignment")
        frame_end_ns[tag] = datetime_index_to_ns_int64(end_ts)
        for source_col in BOOK_CONTEXT_COLUMNS:
            col = f"book_{source_col}"
            if col not in frame.columns:
                if source_col == "available_time":
                    frame[col] = pd.NaT
                else:
                    frame[col] = np.nan
        frame["book_context_missing_flag"] = True
        frame["book_available_after_signal_flag"] = False

    touched_days = 0
    feature_rows = 0
    for done, day in enumerate(days, start=1):
        day_end = day + pd.Timedelta(days=1)
        day_ns = int(day.value)
        day_end_ns = int(day_end.value)
        subsets: dict[str, np.ndarray] = {}
        for tag, end_ns in frame_end_ns.items():
            left = int(np.searchsorted(end_ns, day_ns, side="left"))
            right = int(np.searchsorted(end_ns, day_end_ns, side="left"))
            if right > left:
                subsets[tag] = np.arange(left, right, dtype=np.int64)
        if not subsets:
            progress.update(done)
            continue
        lead = pd.Timedelta(minutes=cfg.book_reference_minutes + 2)
        raw = loader.load_features(
            day - lead,
            day_end + pd.Timedelta(seconds=args.book_tolerance_seconds),
            project_time=True,
            index_mode="none",
        )
        prepared = prepare_book_features(raw, cfg) if not raw.empty else pd.DataFrame()
        feature_rows += len(prepared)
        if not prepared.empty:
            touched_days += 1
        for tag, positions in subsets.items():
            subset = frames[tag].iloc[positions].copy()
            aligned = attach_book_context(
                subset,
                prepared,
                tolerance=pd.Timedelta(seconds=args.book_tolerance_seconds),
            )
            new_cols = [c for c in aligned.columns if c.startswith("book_")]
            for col in new_cols:
                frames[tag].loc[positions, col] = aligned[col].to_numpy()
        progress.update(done)
    progress.close()
    return {
        "coverage_days": len(loader.coverage()),
        "requested_days": len(days),
        "days_with_features": touched_days,
        "feature_rows_processed": int(feature_rows),
    }


def _bool_series(frame: pd.DataFrame, name: str, default: bool) -> pd.Series:
    """Return a nullable-safe boolean Series aligned to ``frame``."""

    if name not in frame.columns:
        return pd.Series(bool(default), index=frame.index, dtype=bool)
    values = pd.Series(pd.array(frame[name], dtype="boolean"), index=frame.index)
    return values.fillna(bool(default)).astype(bool)


def _event_stage_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for key, part in events.groupby(["range_tag", "mode", "stage", "side_name"], observed=True):
        times = pd.to_datetime(part["signal_time"])
        months = max(1.0, (times.max() - times.min()).total_seconds() / (86400.0 * 30.4375))
        rows.append(
            {
                "range_tag": key[0],
                "mode": key[1],
                "stage": key[2],
                "side_name": key[3],
                "events": int(len(part)),
                "events_per_month": float(len(part) / months),
                "book_missing_rate": float(_bool_series(part, "book_context_missing_flag", True).mean()),
                "footprint_missing_rate": float(_bool_series(part, "footprint_missing_flag", True).mean()),
                "mean_obi_5s": float(pd.to_numeric(part["book_obi_5s"], errors="coerce").mean()),
                "mean_notional_multiple": float(pd.to_numeric(part["notional_multiple"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _forward_summary(events: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for horizon in FORWARD_HORIZONS:
        col = f"h{horizon}_net_return"
        part = summarize_returns(
            events,
            value_col=col,
            group_cols=["range_tag", "mode", "stage", "side_name", "split"],
        )
        if not part.empty:
            part["horizon_minutes"] = horizon
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _trade_summary(trades: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for key, part in trades.groupby(list(group_cols), observed=True, dropna=False):
        values = pd.to_numeric(part["net_return"], errors="coerce").dropna().to_numpy(dtype=float)
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(group_cols, keys, strict=False)}
        if len(values) == 0:
            continue
        row.update(
            {
                "trades": int(len(values)),
                "mean_net": float(np.mean(values)),
                "median_net": float(np.median(values)),
                "win_rate": float(np.mean(values > 0)),
                "profit_factor": profit_factor(values),
                "mean_r": float(pd.to_numeric(part["r_multiple"], errors="coerce").mean()),
                "same_bar_both_rate": float(_bool_series(part, "same_bar_both_hit_flag", False).mean()),
                "mean_holding_minutes": float(
                    (pd.to_datetime(part["exit_time"]) - pd.to_datetime(part["entry_time"])).dt.total_seconds().mean() / 60.0
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _fixed_feature_uplift(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    specs = {
        "book_obi_5s": [-np.inf, -0.30, -0.10, 0.10, 0.30, np.inf],
        "book_ask_depth_25bps_ref_ratio": [-np.inf, 0.40, 0.60, 0.80, 1.00, 1.50, np.inf],
        "book_bid_depth_25bps_ref_ratio": [-np.inf, 0.40, 0.60, 0.80, 1.00, 1.50, np.inf],
        "notional_multiple": [-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf],
        "fp_low_zone_delta_ratio": [-np.inf, -0.30, -0.10, 0.10, 0.30, np.inf],
        "fp_high_zone_delta_ratio": [-np.inf, -0.30, -0.10, 0.10, 0.30, np.inf],
    }
    rows: list[pd.DataFrame] = []
    for feature, bins in specs.items():
        if feature not in events.columns:
            continue
        labels = [f"[{bins[i]},{bins[i + 1]})" for i in range(len(bins) - 1)]
        temp = events.copy()
        temp["feature_bin"] = pd.cut(
            pd.to_numeric(temp[feature], errors="coerce"),
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        ).astype("object")
        summary = summarize_returns(
            temp,
            value_col="h15_net_return",
            group_cols=["range_tag", "mode", "stage", "side_name", "split", "feature_bin"],
        )
        if not summary.empty:
            summary["feature"] = feature
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _data_quality(
    frames: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    footprint_diag: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for tag, frame in frames.items():
        sample = frame[
            (pd.to_datetime(frame["signal_time"]) >= pd.Timestamp(frame.attrs.get("research_start", "1900-01-01")))
        ] if frame.attrs.get("research_start") else frame
        rows.append(
            {
                "range_tag": tag,
                "range_bars": int(len(sample)),
                "start_time": str(pd.to_datetime(sample["signal_time"]).min()) if len(sample) else None,
                "end_time": str(pd.to_datetime(sample["signal_time"]).max()) if len(sample) else None,
                "book_context_coverage": float(1.0 - _bool_series(sample, "book_context_missing_flag", True).mean()),
                "footprint_coverage": float(1.0 - _bool_series(sample, "footprint_missing_flag", True).mean()),
                "causal_book_violation_count": int(_bool_series(sample, "book_available_after_signal_flag", False).sum()),
                "events": int((events["range_tag"] == tag).sum()) if not events.empty else 0,
                **footprint_diag.get(tag, {}),
            }
        )
    return pd.DataFrame(rows)


def _report_trade_history(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> tuple[list[dict[str, Any]], float]:
    capital = float(initial_capital)
    history: list[dict[str, Any]] = []
    for row in trades.sort_values("entry_time").itertuples(index=False):
        stop_pct = max(float(row.risk_return), 1e-6)
        # Respect the user's <=1% equity risk target while capping gross
        # notional at 5x for a readable diagnostic report.
        notional_fraction = min(5.0, 0.01 / stop_pct)
        fee = capital * notional_fraction * float(row.round_trip_cost)
        pnl = capital * notional_fraction * float(row.net_return)
        capital += pnl
        history.append(
            {
                "entry_time": pd.Timestamp(row.entry_time),
                "exit_time": pd.Timestamp(row.exit_time),
                "type": f"{row.mode}_{row.side_name}",
                "entry": float(row.entry_price),
                "exit": float(row.exit_price),
                "pnl": float(pnl),
                "fee": float(fee),
                "capital": float(capital),
                "mfe_r": float(row.mfe / stop_pct) if stop_pct > 0 else np.nan,
                "mae_r": float(row.mae / stop_pct) if stop_pct > 0 else np.nan,
            }
        )
    return history, capital


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_brief(
    event_summary: pd.DataFrame,
    forward_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    lines = [
        "# ETH Liquidity Hunt Momentum R01",
        "",
        "> This is a causal event study and fixed-rule probe, not an accepted edge.",
        "",
        "## What is being tested",
        "",
        "- M1: prior-level sweep + aggressive attack + reclaim + OBI reversal + same-side liquidity rebuild.",
        "- M2: two consecutive aggressive range bars + sustained OBI + thin opposing depth.",
        "- Every layer is reported separately so an apparent edge cannot be hidden inside a final filter.",
        "",
        "## Causal rules",
        "",
        "- Liquidity rows must have `available_time <= signal_time`.",
        "- Entry is the first Range-Bar open with `start_ts > signal_time`.",
        "- Dynamic exits act only after a completed range bar and fill next open.",
        "- Same-bar stop and target is scored as stop.",
        "",
        "## Data limits",
        "",
        "Books coverage determines the usable research window. Range bars may begin earlier only to provide warmup baselines.",
        "The phrase 'spoof order was eaten' is not treated as ground truth. Cancellation, estimated consumption and replenishment remain separate observables.",
        "",
        "## Automatic diagnostics",
        "",
    ]
    if quality.empty:
        lines.append("No data-quality rows were produced.")
    else:
        lines.append(quality.to_markdown(index=False))
    lines.extend(["", "## Event stages", ""])
    lines.append(event_summary.head(60).to_markdown(index=False) if not event_summary.empty else "No events.")
    lines.extend(["", "## Fixed strategy probe", ""])
    lines.append(strategy_summary.head(60).to_markdown(index=False) if not strategy_summary.empty else "No trades.")
    lines.extend(
        [
            "",
            "## Promotion gate",
            "",
            "Do not promote unless the strict stage remains positive in validation and holdout, survives 1.5x/2x cost and 3-range-bar delay, appears on nearby range sizes, and has enough trades for the intended MHF/MF frequency.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        from research.liquidity.liquidity_hunt_momentum_r01.selftest import run_self_test

        run_self_test()
        return Path(args.out_dir)

    range_pcts = _csv_floats(args.range_pcts)
    cfg = LiquidityHuntConfig(
        round_trip_cost=float(args.round_trip_cost),
        attack_notional_multiple=float(args.attack_notional_multiple),
        cooldown_minutes=int(args.cooldown_minutes),
    )
    cfg.validate()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {TITLE}", flush=True)
    print(f"[window] warmup={args.warmup_start_date} research={args.start_date} -> {args.end_date}", flush=True)
    print(f"[range] {range_pcts}", flush=True)

    frames = _load_range_frames(args, range_pcts, cfg)
    footprint_diag = _attach_footprints(args, range_pcts, frames)
    book_diag = _attach_books_daily(args, frames, cfg)
    if int(book_diag.get("feature_rows_processed", 0)) <= 0:
        raise RuntimeError(
            "No liquidity feature rows were loaded for the research window. "
            "Raw Books metadata alone is insufficient; verify .features.npz coverage, "
            "data_dir, timezone alignment, and datetime resolution before rerunning."
        )

    all_events: list[pd.DataFrame] = []
    baseline_trades: list[pd.DataFrame] = []
    strict_combined_parts: list[pd.DataFrame] = []
    cost_stress_parts: list[pd.DataFrame] = []
    delay_stress_parts: list[pd.DataFrame] = []
    causal_parts: list[pd.DataFrame] = []

    progress = ProgressReporter(
        label="[research] range variants",
        total=len(frames),
        every=1,
        enabled=not args.no_progress,
    )
    for done, (tag, frame) in enumerate(frames.items(), start=1):
        frame.attrs["research_start"] = args.start_date
        in_window = frame[
            (pd.to_datetime(frame["signal_time"]) >= pd.Timestamp(args.start_date))
            & (pd.to_datetime(frame["signal_time"]) <= pd.Timestamp(args.end_date))
        ].copy()
        events = build_events(in_window, cfg, range_tag=tag)
        if not events.empty:
            events["split"] = chronological_split_labels(events["signal_time"])
            events = attach_forward_time_outcomes(
                events,
                frame,
                horizons_minutes=FORWARD_HORIZONS,
                entry_delay_bars=1,
                round_trip_cost=cfg.round_trip_cost,
            )
            all_events.append(events)

            for stage in sorted(events["stage"].unique()):
                stage_events = events[events["stage"] == stage]
                trades = simulate_events(
                    stage_events,
                    frame,
                    cfg,
                    StrategyVariant(name="baseline"),
                )
                if not trades.empty:
                    trades["split"] = chronological_split_labels(trades["signal_time"])
                    baseline_trades.append(trades)

            strict_events = events[events["stage"].isin(STRICT_STAGES)]
            strict_trades = simulate_events(
                strict_events,
                frame,
                cfg,
                StrategyVariant(name="strict_combined"),
            )
            if not strict_trades.empty:
                strict_trades["split"] = chronological_split_labels(strict_trades["signal_time"])
                strict_combined_parts.append(strict_trades)

            for multiplier in (1.0, 1.5, 2.0):
                trades = simulate_events(
                    strict_events,
                    frame,
                    cfg,
                    StrategyVariant(name=f"cost_{multiplier:g}x", cost_multiplier=multiplier),
                )
                if not trades.empty:
                    trades["cost_multiplier"] = multiplier
                    cost_stress_parts.append(trades)
            for delay in (1, 2, 3):
                trades = simulate_events(
                    strict_events,
                    frame,
                    cfg,
                    StrategyVariant(name=f"delay_{delay}", entry_delay_bars=delay),
                )
                if not trades.empty:
                    trades["entry_delay_bars"] = delay
                    delay_stress_parts.append(trades)

            audit_trades = pd.concat(
                [trades for trades in baseline_trades if not trades.empty and (trades["range_tag"] == tag).any()],
                ignore_index=True,
            ) if baseline_trades else pd.DataFrame()
            causal_parts.append(build_causal_audit(events, audit_trades))
        progress.update(done)
    progress.close()

    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    baseline_all = pd.concat(baseline_trades, ignore_index=True) if baseline_trades else pd.DataFrame()
    strict_all = pd.concat(strict_combined_parts, ignore_index=True) if strict_combined_parts else pd.DataFrame()
    cost_all = pd.concat(cost_stress_parts, ignore_index=True) if cost_stress_parts else pd.DataFrame()
    delay_all = pd.concat(delay_stress_parts, ignore_index=True) if delay_stress_parts else pd.DataFrame()
    causal_all = pd.concat(causal_parts, ignore_index=True) if causal_parts else pd.DataFrame()

    event_summary = _event_stage_summary(events_all)
    forward_summary = _forward_summary(events_all)
    split_summary = summarize_returns(
        events_all,
        value_col="h15_net_return",
        group_cols=["range_tag", "mode", "stage", "side_name", "split"],
    )
    strategy_summary = _trade_summary(
        baseline_all,
        ["range_tag", "mode", "stage", "side_name", "variant", "split"],
    )
    cost_summary = _trade_summary(cost_all, ["range_tag", "mode", "side_name", "cost_multiplier"])
    delay_summary = _trade_summary(delay_all, ["range_tag", "mode", "side_name", "entry_delay_bars"])
    range_summary = _trade_summary(strict_all, ["range_tag", "mode", "side_name"])
    feature_uplift = _fixed_feature_uplift(events_all)
    quality = _data_quality(frames, events_all, footprint_diag)

    yearly = pd.DataFrame()
    monthly = pd.DataFrame()
    if not strict_all.empty:
        strict_all["year"] = pd.to_datetime(strict_all["exit_time"]).dt.year
        strict_all["month"] = pd.to_datetime(strict_all["exit_time"]).dt.to_period("M").astype(str)
        yearly = _trade_summary(strict_all, ["range_tag", "mode", "side_name", "year"])
        monthly = _trade_summary(strict_all, ["range_tag", "mode", "side_name", "month"])

    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "range_pcts": list(range_pcts),
        "range_tags": list(frames),
        "config": cfg.to_dict(),
        "book_diagnostics": book_diag,
        "footprint_diagnostics": footprint_diag,
        "causal_policy": {
            "signal": "completed range bar",
            "book_alignment": "latest available_time <= signal_time",
            "entry": "first range-bar open with start_ts > signal_time",
            "same_bar_path": "stop wins",
            "dynamic_exit": "completed bar decision, next-open fill",
        },
        "selection_policy": "fixed thresholds; chronological 60/20/20 reporting; no holdout tuning",
        "status": "research_only_not_promoted",
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(event_summary, out_dir / "02_event_stage_summary.csv")
    _write_csv(forward_summary, out_dir / "03_forward_path_summary.csv")
    _write_csv(split_summary, out_dir / "04_split_summary.csv")
    _write_csv(strategy_summary, out_dir / "05_strategy_summary.csv")
    _write_csv(cost_summary, out_dir / "06_cost_stress.csv")
    _write_csv(delay_summary, out_dir / "07_delay_stress.csv")
    _write_csv(range_summary, out_dir / "08_range_neighborhood.csv")
    _write_csv(yearly, out_dir / "09_yearly.csv")
    _write_csv(monthly, out_dir / "10_monthly.csv")
    _write_csv(feature_uplift, out_dir / "11_fixed_feature_uplift.csv")
    _write_csv(causal_all, out_dir / "12_causal_audit.csv")
    _write_csv(events_all.head(50_000), out_dir / "13_event_sample.csv")
    _write_csv(baseline_all.head(50_000), out_dir / "14_trade_sample.csv")
    threshold_rows = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in cfg.to_dict().items()]
    )
    _write_csv(threshold_rows, out_dir / "15_predeclared_thresholds.csv")
    brief = _build_brief(event_summary, forward_summary, strategy_summary, quality)
    (out_dir / "16_research_brief.md").write_text(brief, encoding="utf-8")

    if not args.skip_full_report and not strict_all.empty:
        primary_tag = "r0020" if "r0020" in set(strict_all["range_tag"]) else str(strict_all.iloc[0]["range_tag"])
        primary = strict_all[strict_all["range_tag"] == primary_tag].copy()
        history, capital = _report_trade_history(primary)
        if history:
            report_axis = frames[primary_tag].set_index("end_ts").sort_index()
            report_axis = report_axis[
                (report_axis.index >= pd.Timestamp(args.start_date))
                & (report_axis.index <= pd.Timestamp(args.end_date))
            ]
            total_days = max(1.0, (report_axis.index.max() - report_axis.index.min()).total_seconds() / 86400.0)
            print_full_report(
                history,
                report_axis,
                10_000.0,
                capital,
                f"Liquidity Hunt Momentum R01 FixedRisk {primary_tag}",
                total_days,
                False,
                symbol=args.symbol,
                report_dir=str(out_dir),
            )

    if not args.skip_review_pack:
        finalize_research_report(
            out_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )

    print(f"[done] report={out_dir.resolve()}", flush=True)
    print(
        f"[result] events={len(events_all):,} baseline_trades={len(baseline_all):,} "
        f"strict_trades={len(strict_all):,} causal_failures="
        f"{int(_bool_series(causal_all, 'causal_fail_flag', False).sum()) if not causal_all.empty else 0}",
        flush=True,
    )
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
