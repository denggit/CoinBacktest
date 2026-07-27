#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03: broad multi-window order-flow and single-PA path atlas.

This study deliberately starts from the widest practical candidate universe:
all valid closed 1m trade bars whose rolling aggressive-flow pressure is outside
one fixed neutral band.  It then studies adjacent-window pressure transitions
and adds exactly one price-action context at a time.

It is a hypothesis screen, not a final strategy backtest.  Its purpose is to
find which pressure window/path and which single PA context create enough gross
return thickness to justify a later causal TP/SL strategy implementation.

Data access is exclusively through ``src.data_feed.OKXTradeBarLoader`` using the
project's timezone-aligned (tzplus8) cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.market_process.broad_order_flow_paths import (  # noqa: E402
    BAND_NAMES,
    FLOW_WINDOWS,
    HORIZONS,
    PA_CONTEXTS,
    PRESSURE_EDGES,
    TRANSITION_TYPES,
    add_cross_year_diagnostics,
    build_incremental_pa_table,
    build_outcome_arrays,
    build_pa_context_arrays,
    build_pressure_paths,
    build_transition_arrays,
    combine_sufficient_stats,
    directional_outcomes,
    finalize_sufficient_stats,
    pa_context_mask,
    pressure_band_codes,
    sufficient_stats_by_band,
)
from src.research_common.trade_bar_orderflow import (  # noqa: E402
    trade_bar_field_coverage,
    validate_trade_bar_orderflow,
)


@dataclass(frozen=True)
class StudyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1m"
    start: str = "2023-01-01"
    end: str = "2026-06-30 23:59:59"
    flow_windows: tuple[int, ...] = FLOW_WINDOWS
    horizons: tuple[int, ...] = HORIZONS
    round_trip_cost: float = 0.0011
    warmup_minutes: int = 240
    sample_events_per_group: int = 100


GROUP_COLS = [
    "atlas_type",
    "event_type",
    "flow_window",
    "pressure_band",
    "band_code",
    "flow_side",
    "trade_side",
    "pa_context",
    "horizon",
]


def _parse_int_tuple(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--flow-windows", default=",".join(map(str, FLOW_WINDOWS)))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument("--sample-events-per-group", type=int, default=100)
    parser.add_argument("--export-events", action="store_true")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "data"
        / "reports"
        / "research"
        / "eth_market_process_portfolio"
        / "order_flow"
        / "02_broad_order_flow_pa_path_atlas",
    )
    return parser.parse_args(argv)


def _year_windows(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    year = start.year
    while year <= end.year:
        left = max(start, pd.Timestamp(year=year, month=1, day=1))
        right = min(end, pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59))
        if left <= right:
            yield left, right
        year += 1


def _month_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"))


def _trade_side_name(event_type: str, band_code: int) -> str:
    flow_buy = int(band_code) > 0
    if event_type == "weakening_fade":
        flow_buy = not flow_buy
    return "BUY" if flow_buy else "SELL"


def _append_stats(
    destination: list[dict[str, object]],
    *,
    year: int,
    atlas_type: str,
    event_type: str,
    flow_window: int,
    pa_context: str,
    band_codes: np.ndarray,
    selection: np.ndarray,
    directional_by_horizon: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    for horizon, (gross, net, mfe, mae) in directional_by_horizon.items():
        rows = sufficient_stats_by_band(band_codes, selection, gross, net, mfe, mae)
        for row in rows:
            band_code = int(row["band_code"])
            row.update(
                {
                    "year": int(year),
                    "atlas_type": atlas_type,
                    "event_type": event_type,
                    "flow_window": int(flow_window),
                    "trade_side": _trade_side_name(event_type, band_code),
                    "pa_context": pa_context,
                    "horizon": int(horizon),
                }
            )
            destination.append(row)


def _sample_transition_events(
    *,
    index: pd.DatetimeIndex,
    core_mask: np.ndarray,
    window: int,
    pressure: np.ndarray,
    transition_name: str,
    transition: object,
    pa: dict[str, np.ndarray],
    max_rows: int,
) -> pd.DataFrame:
    mask = core_mask & transition.event_mask
    pos = np.flatnonzero(mask)
    if len(pos) == 0 or max_rows <= 0:
        return pd.DataFrame()
    if len(pos) > max_rows:
        take = np.linspace(0, len(pos) - 1, num=max_rows, dtype=int)
        pos = pos[take]
    side = transition.trade_side[pos]
    rows = pd.DataFrame(
        {
            "signal_time": index[pos],
            "flow_window": int(window),
            "event_type": transition_name,
            "pressure": pressure[pos],
            "prior_pressure_equal_window": transition.prior_pressure[pos],
            "pressure_change": transition.pressure_change[pos],
            "pressure_band": [BAND_NAMES[int(code)] for code in transition.band_code[pos]],
            "flow_side": np.where(transition.flow_side[pos] > 0, "BUY", "SELL"),
            "trade_side": np.where(side > 0, "BUY", "SELL"),
        }
    )
    for context in PA_CONTEXTS[1:]:
        rows[context] = pa_context_mask(context, transition.trade_side, pa)[pos]
    return rows


def _finalize_yearly(raw: pd.DataFrame, months_by_year: dict[int, int]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    pieces: list[pd.DataFrame] = []
    for year, frame in raw.groupby("year", sort=True):
        combined = frame.groupby(GROUP_COLS, dropna=False, sort=True)[
            [
                "events",
                "sum_gross",
                "sum_net",
                "sum_sq_net",
                "wins",
                "gross_gains",
                "gross_losses",
                "net_gains",
                "net_losses",
                "sum_mfe",
                "sum_mae",
            ]
        ].sum().reset_index()
        combined["events"] = combined["events"].astype(int)
        combined["wins"] = combined["wins"].astype(int)
        finished = finalize_sufficient_stats(combined, months_by_year[int(year)])
        finished.insert(0, "year", int(year))
        pieces.append(finished)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _build_screen(overall: pd.DataFrame, incremental: pd.DataFrame, cost: float) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    out = overall.copy()
    retention = incremental[GROUP_COLS + ["retention_vs_parent"]].copy() if not incremental.empty else pd.DataFrame()
    if not retention.empty:
        out = out.merge(retention, on=GROUP_COLS, how="left", validate="one_to_one")
    out["retention_vs_parent"] = out["retention_vs_parent"].fillna(1.0)
    out["frequency_ok"] = out["events_per_month"] >= 30.0
    out["sample_ok"] = (out["events"] >= 500) & (out["min_year_events"] >= 50)
    out["gross_clears_cost"] = out["mean_gross"] > float(cost)
    out["net_positive"] = out["mean_net"] > 0.0
    out["pf_ok"] = out["profit_factor_net"] >= 1.05
    out["year_consistency_ok"] = out["positive_net_years"] >= 3
    out["retention_ok"] = out["retention_vs_parent"] >= 0.05
    out["followup_candidate"] = out[
        [
            "frequency_ok",
            "sample_ok",
            "gross_clears_cost",
            "net_positive",
            "pf_ok",
            "year_consistency_ok",
            "retention_ok",
        ]
    ].all(axis=1)
    return out.sort_values(
        ["followup_candidate", "mean_net", "events"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _write_report(
    path: Path,
    cfg: StudyConfig,
    overall: pd.DataFrame,
    incremental: pd.DataFrame,
    screen: pd.DataFrame,
    funnel: pd.DataFrame,
) -> None:
    base = overall[overall["pa_context"] == "all"].copy()
    base_top = base[base["events"] >= 500].sort_values("mean_net", ascending=False).head(30)
    inc_top = incremental[
        (incremental["events_per_month"] >= 30.0)
        & (incremental["retention_vs_parent"] >= 0.05)
    ].sort_values("delta_mean_net", ascending=False).head(30) if not incremental.empty else pd.DataFrame()
    promoted = screen[screen["followup_candidate"]].copy() if not screen.empty else pd.DataFrame()
    concise = [
        "atlas_type", "event_type", "flow_window", "pressure_band", "trade_side",
        "pa_context", "horizon", "events", "events_per_month", "mean_gross",
        "mean_net", "profit_factor_net", "win_rate_net", "mean_mfe", "mean_mae",
        "positive_net_years", "min_year_events",
    ]
    promoted_view = promoted[[c for c in concise if c in promoted.columns]].head(30) if not promoted.empty else pd.DataFrame()
    base_top_view = base_top[[c for c in concise if c in base_top.columns]] if not base_top.empty else pd.DataFrame()
    inc_cols = [
        "atlas_type", "event_type", "flow_window", "pressure_band", "trade_side",
        "pa_context", "horizon", "events", "events_per_month", "retention_vs_parent",
        "parent_mean_net", "mean_net", "delta_mean_net", "profit_factor_net",
        "positive_net_years", "min_year_events",
    ]
    inc_top_view = inc_top[[c for c in inc_cols if c in inc_top.columns]] if not inc_top.empty else pd.DataFrame()

    lines = [
        "# R03 Broad Order-Flow + Single-PA Path Atlas",
        "",
        "## Purpose",
        "",
        "This is the widest order-flow path screen, not a final strategy backtest. It starts from all valid closed 1m pressure observations, including weak pressure below 3%, compares multiple pressure windows, then adds exactly one PA context at a time.",
        "",
        "## Frozen design",
        "",
        f"- Window: `{cfg.start}` to `{cfg.end}`",
        "- Source: `OKXTradeBarLoader`, timezone-aligned `tzplus8` cache",
        f"- Flow windows: `{cfg.flow_windows}` minutes",
        f"- Outcome horizons: `{cfg.horizons}` minutes",
        f"- Fixed pressure bands: weak `<{PRESSURE_EDGES[0]:.2f}`, mild `>={PRESSURE_EDGES[0]:.2f}`, moderate `>={PRESSURE_EDGES[1]:.2f}`, strong `>={PRESSURE_EDGES[2]:.2f}`",
        "- Signal information: closed 1m bar only",
        "- Entry assumption: next 1m open",
        f"- Round-trip cost: `{cfg.round_trip_cost:.4%}`",
        "- No Range Bar, Books, OI, volatility, large-trade or other filter",
        "- No arbitrary cooldown in event construction",
        "",
        "## Atlases",
        "",
        "- `state / pressure_level_follow`: every non-neutral pressure observation, traded in the pressure direction.",
        "- `transition / band_entry_follow`: the first bar entering a different non-neutral pressure band.",
        "- `transition / strengthening_follow`: current pressure is stronger than the preceding equal-length window, same direction.",
        "- `transition / weakening_follow`: pressure weakens but the test still follows the current pressure direction.",
        "- `transition / weakening_fade`: the same weakening event tested contrarian.",
        "- `transition / reversal_follow`: current pressure direction is opposite to the preceding equal-length window.",
        "",
        "Each PA row is an independent child of the pressure-only parent: prior 60m trend aligned, prior trend opposed, 30m sweep/reclaim, or 30m breakout acceptance. PA conditions are never stacked together in this run.",
        "",
        "## Candidate screen",
        "",
        f"Follow-up rows passing the frozen exploratory screen: `{len(promoted):,}`",
        "",
    ]
    lines.append(promoted_view.to_markdown(index=False) if not promoted_view.empty else "No row passed the follow-up screen.")
    lines.extend(["", "## Best pressure-only rows with at least 500 observations", ""])
    lines.append(base_top_view.to_markdown(index=False) if not base_top_view.empty else "No qualifying rows.")
    lines.extend(["", "## Largest single-PA net-return increments", ""])
    lines.append(inc_top_view.to_markdown(index=False) if not inc_top_view.empty else "No qualifying incremental rows.")
    lines.extend(["", "## Frequency funnel", ""])
    lines.append(funnel.to_markdown(index=False) if not funnel.empty else "No funnel rows.")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- State rows overlap heavily and describe conditional paths; they are not independent trades.",
            "- `t_stat_naive` is diagnostic only because overlapping horizons create serial dependence.",
            "- A positive row is not a strategy. It must later survive de-overlap, causal TP/SL replay, delay/slippage, walk-forward and holdout checks.",
            "- A PA condition is useful only when return thickness improves without collapsing frequency or cross-year consistency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    flow_windows = _parse_int_tuple(args.flow_windows, name="flow-windows")
    horizons = _parse_int_tuple(args.horizons, name="horizons")
    cfg = StudyConfig(
        symbol=args.symbol,
        start=args.start_date,
        end=args.end_date,
        flow_windows=flow_windows,
        horizons=horizons,
        round_trip_cost=float(args.round_trip_cost),
        warmup_minutes=max(240, 2 * max(flow_windows) + 61),
        sample_events_per_group=max(0, int(args.sample_events_per_group)),
    )
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if end < start:
        raise ValueError("end-date must be >= start-date")

    loader = OKXTradeBarLoader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        align_with_okx_loader_timezone=True,
    )
    print("[run] ETH Market Process Portfolio - Broad Order Flow + PA R03")
    print(f"[window] {start} -> {end} timezone=tzplus8")
    print(f"[flow] windows={cfg.flow_windows} bands={PRESSURE_EDGES}")
    print(f"[outcome] horizons={cfg.horizons} entry=next_open cost={cfg.round_trip_cost:.4%}")

    raw_stats: list[dict[str, object]] = []
    funnel_rows: list[dict[str, object]] = []
    samples: list[pd.DataFrame] = []
    all_transition_events: list[pd.DataFrame] = []
    coverage: pd.DataFrame | None = None
    months_by_year: dict[int, int] = {}
    runtime_rows: list[dict[str, object]] = []
    run_started = time.perf_counter()

    windows = list(_year_windows(start, end))
    for chunk_no, (left, right) in enumerate(windows, start=1):
        load_left = left - pd.Timedelta(minutes=cfg.warmup_minutes)
        load_right = right + pd.Timedelta(minutes=max(cfg.horizons) + 2)
        print(f"[chunk {chunk_no}/{len(windows)}] load {load_left} -> {load_right}")
        chunk_started = time.perf_counter()
        stage_started = time.perf_counter()
        bars = loader.fetch_data_by_date_range(
            load_left,
            load_right,
            cvd_mode="range",
            build_missing=False,
        )
        if bars.empty:
            raise RuntimeError(f"no local tzplus8 trade bars for {left} -> {right}")
        load_seconds = time.perf_counter() - stage_started
        bars = bars.sort_index(kind="stable")
        bars = bars[~bars.index.duplicated(keep="last")]
        validate_trade_bar_orderflow(bars, require_large_fields=False)
        if coverage is None:
            coverage = trade_bar_field_coverage(bars)

        core_mask = np.asarray((bars.index >= left) & (bars.index <= right), dtype=bool)
        months_by_year[int(left.year)] = _month_count(left, right)
        print(f"[chunk {chunk_no}/{len(windows)}] rows={len(bars):,} core={int(core_mask.sum()):,}")

        stage_started = time.perf_counter()
        pressure_by_window = build_pressure_paths(bars, cfg.flow_windows)
        pa = build_pa_context_arrays(bars)
        labels_by_horizon = {h: build_outcome_arrays(bars, h) for h in cfg.horizons}
        feature_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()

        for window_no, window in enumerate(cfg.flow_windows, start=1):
            pressure = pressure_by_window[window]
            bands = pressure_band_codes(pressure)
            flow_side = np.sign(bands).astype(np.int8)
            valid_state = core_mask & (bands != 0)
            state_counts = {BAND_NAMES[code]: int((valid_state & (bands == code)).sum()) for code in BAND_NAMES}
            print(
                f"[chunk {chunk_no}/{len(windows)}][flow {window_no}/{len(cfg.flow_windows)}] "
                f"window={window}m non_neutral={int(valid_state.sum()):,}"
            )

            state_directional = {
                h: directional_outcomes(labels, flow_side, cfg.round_trip_cost)
                for h, labels in labels_by_horizon.items()
            }
            for context in PA_CONTEXTS:
                selection = valid_state & pa_context_mask(context, flow_side, pa)
                _append_stats(
                    raw_stats,
                    year=left.year,
                    atlas_type="state",
                    event_type="pressure_level_follow",
                    flow_window=window,
                    pa_context=context,
                    band_codes=bands,
                    selection=selection,
                    directional_by_horizon=state_directional,
                )

            transitions = build_transition_arrays(pressure, window)
            funnel = {
                "year": int(left.year),
                "flow_window": int(window),
                "core_bars": int(core_mask.sum()),
                "valid_pressure_bars": int((core_mask & np.isfinite(pressure)).sum()),
                "non_neutral_pressure_bars": int(valid_state.sum()),
                **{f"band_{name}": count for name, count in state_counts.items()},
            }
            for transition_name in TRANSITION_TYPES:
                transition = transitions[transition_name]
                event_base = core_mask & transition.event_mask
                funnel[f"events_{transition_name}"] = int(event_base.sum())
                transition_directional = {
                    h: directional_outcomes(labels, transition.trade_side, cfg.round_trip_cost)
                    for h, labels in labels_by_horizon.items()
                }
                for context in PA_CONTEXTS:
                    selection = event_base & pa_context_mask(context, transition.trade_side, pa)
                    _append_stats(
                        raw_stats,
                        year=left.year,
                        atlas_type="transition",
                        event_type=transition_name,
                        flow_window=window,
                        pa_context=context,
                        band_codes=transition.band_code,
                        selection=selection,
                        directional_by_horizon=transition_directional,
                    )
                sample = _sample_transition_events(
                    index=pd.DatetimeIndex(bars.index),
                    core_mask=core_mask,
                    window=window,
                    pressure=pressure,
                    transition_name=transition_name,
                    transition=transition,
                    pa=pa,
                    max_rows=cfg.sample_events_per_group,
                )
                if not sample.empty:
                    samples.append(sample)
                if args.export_events and event_base.any():
                    pos = np.flatnonzero(event_base)
                    event_frame = pd.DataFrame(
                        {
                            "signal_time": bars.index[pos],
                            "year": int(left.year),
                            "flow_window": int(window),
                            "event_type": transition_name,
                            "pressure": pressure[pos],
                            "prior_pressure_equal_window": transition.prior_pressure[pos],
                            "pressure_change": transition.pressure_change[pos],
                            "pressure_band": [BAND_NAMES[int(code)] for code in transition.band_code[pos]],
                            "flow_side": np.where(transition.flow_side[pos] > 0, "BUY", "SELL"),
                            "trade_side": np.where(transition.trade_side[pos] > 0, "BUY", "SELL"),
                        }
                    )
                    all_transition_events.append(event_frame)
            funnel_rows.append(funnel)

        aggregate_seconds = time.perf_counter() - stage_started
        chunk_seconds = time.perf_counter() - chunk_started
        runtime_rows.append(
            {
                "year": int(left.year),
                "loaded_rows": int(len(bars)),
                "core_rows": int(core_mask.sum()),
                "load_seconds": load_seconds,
                "feature_seconds": feature_seconds,
                "aggregate_seconds": aggregate_seconds,
                "chunk_seconds": chunk_seconds,
            }
        )
        print(
            f"[chunk {chunk_no}/{len(windows)}] elapsed={chunk_seconds:.1f}s "
            f"load={load_seconds:.1f}s features={feature_seconds:.1f}s aggregate={aggregate_seconds:.1f}s"
        )
        del bars, pressure_by_window, pa, labels_by_horizon

    raw = pd.DataFrame(raw_stats)
    if raw.empty:
        raise RuntimeError("no valid order-flow observations were generated")
    total_months = _month_count(start, end)
    yearly = _finalize_yearly(raw, months_by_year)
    overall = combine_sufficient_stats(raw, GROUP_COLS, total_months)
    overall = add_cross_year_diagnostics(overall, yearly)
    incremental = build_incremental_pa_table(overall)
    screen = _build_screen(overall, incremental, cfg.round_trip_cost)
    funnel = pd.DataFrame(funnel_rows).sort_values(["flow_window", "year"]).reset_index(drop=True)
    event_sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(report_dir / "overview.csv", index=False)
    yearly.to_csv(report_dir / "yearly.csv", index=False)
    incremental.to_csv(report_dir / "incremental_pa.csv", index=False)
    screen.to_csv(report_dir / "followup_screen.csv", index=False)
    funnel.to_csv(report_dir / "frequency_funnel.csv", index=False)
    event_sample.to_csv(report_dir / "transition_event_sample.csv", index=False)
    runtime_profile = pd.DataFrame(runtime_rows)
    total_row: dict[str, object] = {
        "year": "TOTAL",
        "loaded_rows": int(runtime_profile["loaded_rows"].sum()),
        "core_rows": int(runtime_profile["core_rows"].sum()),
        "total_wall_seconds": time.perf_counter() - run_started,
    }
    for col in ("load_seconds", "feature_seconds", "aggregate_seconds", "chunk_seconds"):
        total_row[col] = float(runtime_profile[col].sum())
    runtime_profile["year"] = runtime_profile["year"].astype(str)
    runtime_profile["total_wall_seconds"] = np.nan
    runtime_profile = pd.concat([runtime_profile, pd.DataFrame([total_row])], ignore_index=True)
    runtime_profile.to_csv(report_dir / "runtime_profile.csv", index=False)
    (coverage if coverage is not None else pd.DataFrame()).to_csv(report_dir / "field_coverage.csv", index=False)
    if args.export_events and all_transition_events:
        pd.concat(all_transition_events, ignore_index=True).to_csv(
            report_dir / "transition_events.csv.gz",
            index=False,
            compression="gzip",
        )
    (report_dir / "run_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(report_dir / "report.md", cfg, overall, incremental, screen, funnel)

    candidate_count = int(screen["followup_candidate"].sum()) if not screen.empty else 0
    print(f"[done] summary_rows={len(overall):,} followup_candidates={candidate_count:,}")
    print(f"[report] {report_dir / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
