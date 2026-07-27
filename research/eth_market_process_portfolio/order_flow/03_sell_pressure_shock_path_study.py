#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R04: broad sell-pressure shock, spike, reclaim and continuation study.

Hypothesis
----------
A sudden aggressive-sell shock does not imply one direction by itself. The
price response may separate two causal paths:

1. sell shock -> downside spike/sweep -> absorption/reclaim -> long reversal;
2. sell shock -> sweep/breakdown -> acceptance below -> short continuation.

The study starts with broad multi-window shock candidates and adds one PA or
activity condition at a time. Both follow and fade directions are reported to
avoid survivorship bias. This is an event/path screen, not a final TP/SL model.

Data access is exclusively through ``src.data_feed.OKXTradeBarLoader`` and the
project's local tzplus8 trade-bar cache.
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
    combine_sufficient_stats,
    finalize_sufficient_stats,
    sufficient_stats_by_band,
)
from src.research_common.market_process.sell_pressure_shock_paths import (  # noqa: E402
    ACCEPTANCE_BARS,
    ACTIVITY_THRESHOLDS,
    FLOW_WINDOWS,
    HORIZONS,
    RECLAIM_WAITS,
    SHOCK_TYPES,
    build_outcome_arrays,
    build_post_shock_events,
    build_sell_shock_arrays,
    build_sell_shock_pa,
    directional_outcomes,
    fixed_side_array,
    rolling_activity_ratio,
    rolling_pressure_ratio,
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
    reclaim_waits: tuple[int, ...] = RECLAIM_WAITS
    acceptance_bars: tuple[int, ...] = ACCEPTANCE_BARS
    activity_thresholds: tuple[float, ...] = ACTIVITY_THRESHOLDS
    round_trip_cost: float = 0.0011
    baseline_minutes: int = 240
    sample_events_per_group: int = 100


GROUP_COLS = [
    "shock_type",
    "flow_window",
    "pressure_band",
    "band_code",
    "path_type",
    "trade_side",
    "horizon",
]


def _parse_int_tuple(raw: str, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def _parse_float_tuple(raw: str, name: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values or min(values) <= 0.0:
        raise argparse.ArgumentTypeError(f"{name} must contain positive numbers")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--flow-windows", default=",".join(map(str, FLOW_WINDOWS)))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--reclaim-waits", default=",".join(map(str, RECLAIM_WAITS)))
    parser.add_argument("--acceptance-bars", default=",".join(map(str, ACCEPTANCE_BARS)))
    parser.add_argument("--activity-thresholds", default=",".join(map(str, ACTIVITY_THRESHOLDS)))
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
        / "03_sell_pressure_shock_path_study",
    )
    return parser.parse_args(argv)


def _year_windows(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    for year in range(start.year, end.year + 1):
        left = max(start, pd.Timestamp(year=year, month=1, day=1))
        right = min(end, pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59))
        if left <= right:
            yield left, right


def _month_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"))


def _append_stats(
    rows: list[dict[str, object]],
    *,
    year: int,
    shock_type: str,
    flow_window: int,
    path_type: str,
    trade_side_name: str,
    band_codes: np.ndarray,
    selection: np.ndarray,
    outcomes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    for horizon, (gross, net, mfe, mae) in outcomes.items():
        for row in sufficient_stats_by_band(band_codes, selection, gross, net, mfe, mae):
            row.update(
                {
                    "year": int(year),
                    "shock_type": shock_type,
                    "flow_window": int(flow_window),
                    "path_type": path_type,
                    "trade_side": trade_side_name,
                    "horizon": int(horizon),
                }
            )
            rows.append(row)


def _finalize_yearly(raw: pd.DataFrame, months_by_year: dict[int, int]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    sum_cols = [
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
    pieces: list[pd.DataFrame] = []
    for year, frame in raw.groupby("year", sort=True):
        combined = frame.groupby(GROUP_COLS, dropna=False, sort=True)[sum_cols].sum().reset_index()
        combined["events"] = combined["events"].astype(int)
        combined["wins"] = combined["wins"].astype(int)
        finished = finalize_sufficient_stats(combined, months_by_year[int(year)])
        finished.insert(0, "year", int(year))
        pieces.append(finished)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()



def _add_cross_year_diagnostics(overall: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return overall.copy()
    keys = GROUP_COLS
    diag = (
        yearly.groupby(keys, dropna=False, sort=False)
        .agg(
            years_present=("year", "nunique"),
            positive_net_years=("mean_net", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            positive_gross_years=("mean_gross", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            min_year_events=("events", "min"),
        )
        .reset_index()
    )
    return overall.merge(diag, on=keys, how="left", validate="one_to_one")

def _parent_path(path_type: str) -> str:
    return "shock_fade_long" if path_type.endswith("_long") else "shock_follow_short"


def _build_incremental(overall: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    base_keys = ["shock_type", "flow_window", "pressure_band", "band_code", "trade_side", "horizon"]
    parents = overall[overall["path_type"].isin(["shock_fade_long", "shock_follow_short"])].copy()
    parent_cols = base_keys + [
        "path_type",
        "events",
        "mean_gross",
        "mean_net",
        "profit_factor_net",
        "win_rate_net",
        "mean_mfe",
        "mean_mae",
        "positive_net_years",
        "min_year_events",
    ]
    parents = parents[parent_cols].rename(
        columns={
            "path_type": "parent_path_type",
            "events": "parent_events",
            "mean_gross": "parent_mean_gross",
            "mean_net": "parent_mean_net",
            "profit_factor_net": "parent_profit_factor_net",
            "win_rate_net": "parent_win_rate_net",
            "mean_mfe": "parent_mean_mfe",
            "mean_mae": "parent_mean_mae",
            "positive_net_years": "parent_positive_net_years",
            "min_year_events": "parent_min_year_events",
        }
    )
    child = overall[~overall["path_type"].isin(["shock_fade_long", "shock_follow_short"])].copy()
    child["parent_path_type"] = child["path_type"].map(_parent_path)
    merged = child.merge(parents, on=base_keys + ["parent_path_type"], how="left", validate="many_to_one")
    merged["retention_vs_parent"] = merged["events"] / merged["parent_events"].replace(0.0, np.nan)
    merged["delta_mean_gross"] = merged["mean_gross"] - merged["parent_mean_gross"]
    merged["delta_mean_net"] = merged["mean_net"] - merged["parent_mean_net"]
    merged["delta_pf_net"] = merged["profit_factor_net"] - merged["parent_profit_factor_net"]
    merged["delta_win_rate_net"] = merged["win_rate_net"] - merged["parent_win_rate_net"]
    return merged


def _build_screen(overall: pd.DataFrame, incremental: pd.DataFrame, cost: float) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    out = overall.copy()
    retention = incremental[GROUP_COLS + ["retention_vs_parent", "delta_mean_gross"]] if not incremental.empty else pd.DataFrame()
    if not retention.empty:
        out = out.merge(retention, on=GROUP_COLS, how="left", validate="one_to_one")
    out["retention_vs_parent"] = out["retention_vs_parent"].fillna(1.0)
    out["delta_mean_gross"] = out["delta_mean_gross"].fillna(0.0)
    out["sample_ok"] = (out["events"] >= 300) & (out["min_year_events"] >= 30)
    out["frequency_ok"] = out["events_per_month"] >= 5.0
    out["gross_clears_cost"] = out["mean_gross"] > float(cost)
    out["net_positive"] = out["mean_net"] > 0.0
    out["pf_ok"] = out["profit_factor_net"] >= 1.05
    out["year_consistency_ok"] = out["positive_net_years"] >= 3
    out["retention_ok"] = out["retention_vs_parent"] >= 0.03
    out["incremental_ok"] = out["delta_mean_gross"] >= 0.0002
    out["followup_candidate"] = out[
        [
            "sample_ok",
            "frequency_ok",
            "gross_clears_cost",
            "net_positive",
            "pf_ok",
            "year_consistency_ok",
            "retention_ok",
        ]
    ].all(axis=1)
    return out.sort_values(
        ["followup_candidate", "mean_net", "events"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _sample_events(
    *,
    index: pd.DatetimeIndex,
    core_mask: np.ndarray,
    shock_type: str,
    window: int,
    shock: object,
    activity_ratio: np.ndarray,
    pa: object,
    max_rows: int,
) -> pd.DataFrame:
    pos = np.flatnonzero(core_mask & shock.event_mask)
    if len(pos) == 0 or max_rows <= 0:
        return pd.DataFrame()
    if len(pos) > max_rows:
        pos = pos[np.linspace(0, len(pos) - 1, num=max_rows, dtype=int)]
    return pd.DataFrame(
        {
            "signal_time": index[pos],
            "shock_type": shock_type,
            "flow_window": int(window),
            "pressure": shock.current_pressure[pos],
            "prior_pressure": shock.prior_pressure[pos],
            "pressure_change": shock.pressure_change[pos],
            "pressure_band": [BAND_NAMES[int(x)] for x in shock.band_code[pos]],
            "activity_ratio": activity_ratio[pos],
            "window_return": pa.window_return[pos],
            "downside_excursion": pa.downside_excursion[pos],
            "lower_wick_fraction": pa.lower_wick_fraction[pos],
            "close_recovery_fraction": pa.close_recovery_fraction[pos],
            "prior_low_sweep": pa.prior_low_sweep[pos],
            "same_window_sweep_reclaim": pa.same_window_sweep_reclaim[pos],
            "sweep_without_reclaim": pa.sweep_without_reclaim[pos],
        }
    )


def _write_report(
    path: Path,
    cfg: StudyConfig,
    overall: pd.DataFrame,
    incremental: pd.DataFrame,
    screen: pd.DataFrame,
    funnel: pd.DataFrame,
) -> None:
    lines = [
        "# R04 Sell-Pressure Shock Path Study",
        "",
        "## Hypothesis",
        "",
        "A sudden aggressive-sell shock can produce either continuation or reversal. "
        "The price response—spike, sweep, reclaim or acceptance—is tested as the differentiator.",
        "",
        "## Causal contract",
        "",
        "- All pressure/activity/PA fields use closed data only.",
        "- Prior-low references exclude the complete shock window.",
        "- Same-window reclaim is traded from the next 1m open.",
        "- Delayed reclaim and acceptance are signalled only on their confirmation close, then entered next open.",
        "- Base/follow and fade directions are both reported.",
        "",
        "## Fixed configuration",
        "",
        f"- Window: `{cfg.start}` to `{cfg.end}`",
        f"- Flow windows: `{cfg.flow_windows}`",
        f"- Forward horizons: `{cfg.horizons}`",
        f"- Reclaim waits: `{cfg.reclaim_waits}`",
        f"- Acceptance confirmations: `{cfg.acceptance_bars}`",
        f"- Activity thresholds: `{cfg.activity_thresholds}`",
        f"- Round-trip cost: `{cfg.round_trip_cost:.4%}`",
        "",
        "## Follow-up screen",
        "",
    ]
    columns = [
        "shock_type",
        "flow_window",
        "pressure_band",
        "path_type",
        "trade_side",
        "horizon",
        "events",
        "events_per_month",
        "mean_gross",
        "mean_net",
        "profit_factor_net",
        "positive_net_years",
        "followup_candidate",
    ]
    top = screen.head(30)[columns] if not screen.empty else pd.DataFrame(columns=columns)
    lines.append(top.to_markdown(index=False))
    lines.extend(["", "## Largest single-condition gross-return improvements", ""])
    if incremental.empty:
        lines.append("No incremental rows.")
    else:
        inc_cols = [
            "shock_type",
            "flow_window",
            "pressure_band",
            "path_type",
            "trade_side",
            "horizon",
            "events",
            "retention_vs_parent",
            "delta_mean_gross",
            "mean_gross",
            "mean_net",
            "positive_net_years",
        ]
        lines.append(
            incremental.sort_values(["delta_mean_gross", "events"], ascending=[False, False])
            .head(40)[inc_cols]
            .to_markdown(index=False)
        )
    lines.extend(["", "## Frequency funnel", ""])
    lines.append(funnel.to_markdown(index=False) if not funnel.empty else "No funnel rows.")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This atlas contains multiple prespecified windows/paths; isolated top rows are hypotheses, not strategies.",
            "- A valid direction must retain enough samples, clear 0.11% cost, and persist across years/window neighbours.",
            "- No TP/SL search is performed here; weak gross edge cannot be rescued by exit optimisation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    cfg = StudyConfig(
        symbol=args.symbol,
        start=str(start),
        end=str(end),
        flow_windows=_parse_int_tuple(args.flow_windows, "flow_windows"),
        horizons=_parse_int_tuple(args.horizons, "horizons"),
        reclaim_waits=_parse_int_tuple(args.reclaim_waits, "reclaim_waits"),
        acceptance_bars=_parse_int_tuple(args.acceptance_bars, "acceptance_bars"),
        activity_thresholds=_parse_float_tuple(args.activity_thresholds, "activity_thresholds"),
        round_trip_cost=float(args.round_trip_cost),
        sample_events_per_group=max(0, int(args.sample_events_per_group)),
    )
    loader = OKXTradeBarLoader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        align_with_okx_loader_timezone=True,
    )
    raw_stats: list[dict[str, object]] = []
    funnel_rows: list[dict[str, object]] = []
    samples: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    months_by_year: dict[int, int] = {}
    coverage: pd.DataFrame | None = None
    run_started = time.perf_counter()
    windows = list(_year_windows(start, end))
    warmup = cfg.baseline_minutes + max(cfg.flow_windows) + 35
    right_pad = max(cfg.horizons) + max(cfg.reclaim_waits) + 5

    for chunk_no, (left, right) in enumerate(windows, start=1):
        load_left = left - pd.Timedelta(minutes=warmup)
        load_right = right + pd.Timedelta(minutes=right_pad)
        print(f"[chunk {chunk_no}/{len(windows)}] load {load_left} -> {load_right}")
        chunk_started = time.perf_counter()
        stage = time.perf_counter()
        bars = loader.fetch_data_by_date_range(load_left, load_right, cvd_mode="range", build_missing=False)
        if bars.empty:
            raise RuntimeError(f"no local tzplus8 trade bars for {left} -> {right}")
        load_seconds = time.perf_counter() - stage
        bars = bars.sort_index(kind="stable")
        bars = bars[~bars.index.duplicated(keep="last")]
        validate_trade_bar_orderflow(bars, require_large_fields=False)
        if coverage is None:
            coverage = trade_bar_field_coverage(bars)
        core_mask = np.asarray((bars.index >= left) & (bars.index <= right), dtype=bool)
        months_by_year[int(left.year)] = _month_count(left, right)
        print(f"[chunk {chunk_no}/{len(windows)}] rows={len(bars):,} core={int(core_mask.sum()):,}")

        stage = time.perf_counter()
        labels = {h: build_outcome_arrays(bars, h) for h in cfg.horizons}
        long_side = fixed_side_array(len(bars), 1)
        short_side = fixed_side_array(len(bars), -1)
        long_outcomes = {h: directional_outcomes(label, long_side, cfg.round_trip_cost) for h, label in labels.items()}
        short_outcomes = {h: directional_outcomes(label, short_side, cfg.round_trip_cost) for h, label in labels.items()}
        feature_seconds = time.perf_counter() - stage
        stage = time.perf_counter()

        for window_no, window in enumerate(cfg.flow_windows, start=1):
            pressure = rolling_pressure_ratio(bars["delta_notional"], bars["notional"], window)
            shocks = build_sell_shock_arrays(pressure, window)
            activity = rolling_activity_ratio(bars["notional"], window, cfg.baseline_minutes)
            pa = build_sell_shock_pa(bars, window)
            print(f"[chunk {chunk_no}/{len(windows)}][flow {window_no}/{len(cfg.flow_windows)}] window={window}m")

            for shock_type in SHOCK_TYPES:
                shock = shocks[shock_type]
                base = core_mask & shock.event_mask
                post = build_post_shock_events(
                    bars["close"],
                    shock.event_mask,
                    pa.sweep_without_reclaim,
                    pa.prior_low_30,
                    cfg.reclaim_waits,
                    cfg.acceptance_bars,
                )
                post_band_codes = np.zeros(len(bars), dtype=np.int8)
                valid_source = post.source_shock_index >= 0
                if valid_source.any():
                    post_band_codes[valid_source] = shock.band_code[post.source_shock_index[valid_source]]

                path_masks: dict[
                    str,
                    tuple[
                        np.ndarray,
                        str,
                        dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                        np.ndarray,
                    ],
                ] = {
                    "shock_follow_short": (base, "SELL", short_outcomes, shock.band_code),
                    "shock_fade_long": (base, "BUY", long_outcomes, shock.band_code),
                    "downside_impulse_follow_short": (base & pa.downside_impulse, "SELL", short_outcomes, shock.band_code),
                    "downside_impulse_fade_long": (base & pa.downside_impulse, "BUY", long_outcomes, shock.band_code),
                    "lower_wick_follow_short": (base & pa.lower_wick, "SELL", short_outcomes, shock.band_code),
                    "lower_wick_fade_long": (base & pa.lower_wick, "BUY", long_outcomes, shock.band_code),
                    "deep_lower_wick_fade_long": (base & pa.deep_lower_wick, "BUY", long_outcomes, shock.band_code),
                    "prior_low_sweep_follow_short": (base & pa.prior_low_sweep, "SELL", short_outcomes, shock.band_code),
                    "prior_low_sweep_fade_long": (base & pa.prior_low_sweep, "BUY", long_outcomes, shock.band_code),
                    "same_window_sweep_reclaim_long": (base & pa.same_window_sweep_reclaim, "BUY", long_outcomes, shock.band_code),
                    "sweep_without_reclaim_short": (base & pa.sweep_without_reclaim, "SELL", short_outcomes, shock.band_code),
                }
                for threshold in cfg.activity_thresholds:
                    key = str(threshold).replace(".", "p")
                    path_masks[f"activity_ge_{key}_follow_short"] = (
                        base & (activity >= threshold),
                        "SELL",
                        short_outcomes,
                        shock.band_code,
                    )
                    path_masks[f"activity_ge_{key}_fade_long"] = (
                        base & (activity >= threshold),
                        "BUY",
                        long_outcomes,
                        shock.band_code,
                    )
                for wait, mask in post.delayed_reclaim.items():
                    path_masks[f"delayed_reclaim_{wait}m_long"] = (
                        core_mask & mask,
                        "BUY",
                        long_outcomes,
                        post_band_codes,
                    )
                for count, mask in post.breakdown_acceptance.items():
                    path_masks[f"breakdown_acceptance_{count}bar_short"] = (
                        core_mask & mask,
                        "SELL",
                        short_outcomes,
                        post_band_codes,
                    )

                funnel = {
                    "year": int(left.year),
                    "flow_window": int(window),
                    "shock_type": shock_type,
                    "core_bars": int(core_mask.sum()),
                    "base_shocks": int(base.sum()),
                    "activity_ge_1p5": int((base & (activity >= 1.5)).sum()),
                    "activity_ge_2p5": int((base & (activity >= 2.5)).sum()),
                    "downside_impulse": int((base & pa.downside_impulse).sum()),
                    "lower_wick": int((base & pa.lower_wick).sum()),
                    "deep_lower_wick": int((base & pa.deep_lower_wick).sum()),
                    "prior_low_sweep": int((base & pa.prior_low_sweep).sum()),
                    "same_window_sweep_reclaim": int((base & pa.same_window_sweep_reclaim).sum()),
                    "sweep_without_reclaim": int((base & pa.sweep_without_reclaim).sum()),
                }
                for wait, mask in post.delayed_reclaim.items():
                    funnel[f"delayed_reclaim_{wait}m"] = int((core_mask & mask).sum())
                for count, mask in post.breakdown_acceptance.items():
                    funnel[f"breakdown_acceptance_{count}bar"] = int((core_mask & mask).sum())
                funnel_rows.append(funnel)

                for path_type, (selection, side_name, outcome_map, path_band_codes) in path_masks.items():
                    _append_stats(
                        raw_stats,
                        year=left.year,
                        shock_type=shock_type,
                        flow_window=window,
                        path_type=path_type,
                        trade_side_name=side_name,
                        band_codes=path_band_codes,
                        selection=selection,
                        outcomes=outcome_map,
                    )

                sample = _sample_events(
                    index=pd.DatetimeIndex(bars.index),
                    core_mask=core_mask,
                    shock_type=shock_type,
                    window=window,
                    shock=shock,
                    activity_ratio=activity,
                    pa=pa,
                    max_rows=cfg.sample_events_per_group,
                )
                if not sample.empty:
                    samples.append(sample)
                if args.export_events and base.any():
                    pos = np.flatnonzero(base)
                    all_events.append(
                        pd.DataFrame(
                            {
                                "signal_time": bars.index[pos],
                                "year": int(left.year),
                                "flow_window": int(window),
                                "shock_type": shock_type,
                                "pressure": shock.current_pressure[pos],
                                "prior_pressure": shock.prior_pressure[pos],
                                "pressure_change": shock.pressure_change[pos],
                                "pressure_band": [BAND_NAMES[int(x)] for x in shock.band_code[pos]],
                                "activity_ratio": activity[pos],
                                "window_return": pa.window_return[pos],
                                "downside_excursion": pa.downside_excursion[pos],
                                "lower_wick_fraction": pa.lower_wick_fraction[pos],
                                "close_recovery_fraction": pa.close_recovery_fraction[pos],
                                "prior_low_sweep": pa.prior_low_sweep[pos],
                                "same_window_sweep_reclaim": pa.same_window_sweep_reclaim[pos],
                                "sweep_without_reclaim": pa.sweep_without_reclaim[pos],
                            }
                        )
                    )

        aggregate_seconds = time.perf_counter() - stage
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
        del bars, labels, long_outcomes, short_outcomes

    raw = pd.DataFrame(raw_stats)
    if raw.empty:
        raise RuntimeError("no sell-pressure shock statistics were generated")
    total_months = _month_count(start, end)
    yearly = _finalize_yearly(raw, months_by_year)
    overall = combine_sufficient_stats(raw, GROUP_COLS, total_months)
    overall = _add_cross_year_diagnostics(overall, yearly)
    incremental = _build_incremental(overall)
    screen = _build_screen(overall, incremental, cfg.round_trip_cost)
    funnel = pd.DataFrame(funnel_rows).sort_values(["flow_window", "shock_type", "year"]).reset_index(drop=True)
    event_sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(report_dir / "overview.csv", index=False)
    yearly.to_csv(report_dir / "yearly.csv", index=False)
    incremental.to_csv(report_dir / "incremental_paths.csv", index=False)
    screen.to_csv(report_dir / "followup_screen.csv", index=False)
    funnel.to_csv(report_dir / "frequency_funnel.csv", index=False)
    event_sample.to_csv(report_dir / "shock_event_sample.csv", index=False)
    runtime = pd.DataFrame(runtime_rows)
    total_row = {
        "year": "TOTAL",
        "loaded_rows": int(runtime["loaded_rows"].sum()),
        "core_rows": int(runtime["core_rows"].sum()),
        "load_seconds": float(runtime["load_seconds"].sum()),
        "feature_seconds": float(runtime["feature_seconds"].sum()),
        "aggregate_seconds": float(runtime["aggregate_seconds"].sum()),
        "chunk_seconds": float(runtime["chunk_seconds"].sum()),
        "total_wall_seconds": time.perf_counter() - run_started,
    }
    runtime["year"] = runtime["year"].astype(str)
    runtime["total_wall_seconds"] = np.nan
    runtime = pd.concat([runtime, pd.DataFrame([total_row])], ignore_index=True)
    runtime.to_csv(report_dir / "runtime_profile.csv", index=False)
    (coverage if coverage is not None else pd.DataFrame()).to_csv(report_dir / "field_coverage.csv", index=False)
    if args.export_events and all_events:
        pd.concat(all_events, ignore_index=True).to_csv(
            report_dir / "shock_events.csv.gz", index=False, compression="gzip"
        )
    (report_dir / "run_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
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
