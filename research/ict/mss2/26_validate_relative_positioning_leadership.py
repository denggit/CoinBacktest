#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent raw-source validator for the frozen R26 outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.binance_futures_metrics_loader import BinanceFuturesMetricsLoader  # noqa: E402
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars  # noqa: E402
from src.research_common.ict_mss2.r26 import R26Config  # noqa: E402

DEFAULT_REPORT_DIR = "data/reports/research/ict/mss2/r26_relative_positioning_leadership_repricing"
EPS = 1e-10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--binance-symbol", default="ETHUSDT")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def _close(a: object, b: object, tolerance: float = EPS) -> bool:
    x, y = float(a), float(b)
    return bool(np.isfinite(x) and np.isfinite(y) and abs(x - y) <= tolerance * max(1.0, abs(x), abs(y)))


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(frame["high"])
    low = pd.to_numeric(frame["low"])
    previous = pd.to_numeric(frame["close"]).shift(1)
    return pd.concat([(high - low).abs(), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)


def _five_state(bars: pd.DataFrame, cfg: R26Config) -> pd.DataFrame:
    five = aggregate_bars(bars, 5).copy()
    five["price_bar_time"] = five.index
    five["price_available_time"] = pd.to_datetime(five["bar_end_time"])
    five["prior_high"] = pd.to_numeric(five["high"]).shift(1)
    five["prior_low"] = pd.to_numeric(five["low"]).shift(1)
    five["range_high_1h"] = pd.to_numeric(five["high"]).rolling(12, min_periods=12).max()
    five["range_low_1h"] = pd.to_numeric(five["low"]).rolling(12, min_periods=12).min()
    five["atr"] = _true_range(five).rolling(12, min_periods=12).mean()
    return five.reset_index(drop=True).sort_values("price_available_time", kind="stable")


def _metric_state(metrics: pd.DataFrame, five: pd.DataFrame, cfg: R26Config) -> pd.DataFrame:
    ratio = metrics.reset_index(drop=True).copy()
    ratio["timestamp"] = pd.to_datetime(ratio["timestamp"])
    ratio["available_time"] = pd.to_datetime(ratio["available_time"])
    ratio = ratio.sort_values("available_time", kind="stable").drop_duplicates("available_time", keep="last")
    aligned = pd.merge_asof(
        ratio,
        five,
        left_on="available_time",
        right_on="price_available_time",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
    )
    aligned["top"] = pd.to_numeric(aligned["top_trader_position_long_share"], errors="coerce")
    aligned["broad"] = pd.to_numeric(aligned["global_account_long_share"], errors="coerce")
    aligned["spread"] = aligned["top"] - aligned["broad"]
    aligned["valid"] = aligned["top"].between(0, 1) & aligned["broad"].between(0, 1)
    aligned["gap"] = aligned["timestamp"].diff()
    aligned["gap_valid"] = aligned["gap"].between(cfg.metric_min_gap, cfg.metric_max_gap)
    aligned["price_step_valid"] = pd.to_datetime(aligned["price_bar_time"]).diff().eq(pd.Timedelta("5min"))
    return aligned.reset_index(drop=True)


def _validate_events(events: pd.DataFrame, aligned: pd.DataFrame, bars: pd.DataFrame, cfg: R26Config) -> list[dict[str, object]]:
    failures: dict[str, int] = {
        "event_metric_lookup": 0,
        "cross_direction": 0,
        "cross_gap_and_validity": 0,
        "retained_sign_and_no_gap": 0,
        "first_price_confirmation": 0,
        "confirmation_window": 0,
        "cross_time_target": 0,
        "next_observed_entry": 0,
        "stop_formula": 0,
        "physical_cutoff": 0,
    }
    by_metric = {pd.Timestamp(value): idx for idx, value in enumerate(aligned["timestamp"])}
    bar_ns = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for event in events.itertuples(index=False):
        cross_idx = by_metric.get(pd.Timestamp(event.cross_metric_time))
        confirm_idx = by_metric.get(pd.Timestamp(event.confirmation_metric_time))
        if cross_idx is None or confirm_idx is None or confirm_idx <= cross_idx:
            failures["event_metric_lookup"] += 1
            continue
        cross = aligned.iloc[cross_idx]
        prior = aligned.iloc[cross_idx - 1]
        confirm = aligned.iloc[confirm_idx]
        direction = int(event.trade_direction)
        expected_cross = (prior["spread"] <= 0 < cross["spread"]) if direction > 0 else (prior["spread"] >= 0 > cross["spread"])
        failures["cross_direction"] += int(not expected_cross)
        failures["cross_gap_and_validity"] += int(
            not (bool(prior["valid"]) and bool(cross["valid"]) and bool(cross["gap_valid"]))
        )
        window = aligned.iloc[cross_idx + 1 : confirm_idx + 1]
        retained = (
            window["valid"].all()
            and window["gap_valid"].all()
            and window["price_step_valid"].all()
            and ((window["spread"] > 0).all() if direction > 0 else (window["spread"] < 0).all())
        )
        failures["retained_sign_and_no_gap"] += int(not retained)
        qualifies = (
            window["close"].gt(window["prior_high"])
            if direction > 0
            else window["close"].lt(window["prior_low"])
        )
        first_qualifying = int(np.flatnonzero(qualifies.to_numpy())[0]) if qualifies.any() else -1
        failures["first_price_confirmation"] += int(first_qualifying < 0 or first_qualifying != len(window) - 1)
        delay = pd.Timestamp(confirm["available_time"]) - pd.Timestamp(cross["available_time"])
        failures["confirmation_window"] += int(delay <= pd.Timedelta(0) or delay > cfg.confirmation_window)
        target = float(cross["range_high_1h"] if direction > 0 else cross["range_low_1h"])
        failures["cross_time_target"] += int(not _close(target, event.structural_target_price))
        signal = max(pd.Timestamp(confirm["available_time"]), pd.Timestamp(confirm["price_available_time"]))
        entry_pos = int(np.searchsorted(bar_ns, np.datetime64(signal, "ns").astype(np.int64), side="left"))
        if entry_pos >= len(bars):
            failures["next_observed_entry"] += 1
        else:
            failures["next_observed_entry"] += int(
                pd.Timestamp(bars.index[entry_pos]) != pd.Timestamp(event.entry_time)
                or not _close(bars.iloc[entry_pos]["open"], event.entry_price)
            )
        atr = float(confirm["atr"])
        expected_stop = (
            min(float(confirm["low"]), float(confirm["prior_low"])) - cfg.stop_buffer_atr * atr
            if direction > 0
            else max(float(confirm["high"]), float(confirm["prior_high"])) + cfg.stop_buffer_atr * atr
        )
        failures["stop_formula"] += int(not _close(expected_stop, event.stop_price))
        failures["physical_cutoff"] += int(pd.Timestamp(event.entry_time) >= cfg.embargo_start)
    return [
        {"check": name, "violations": count, "status": "PASS" if count == 0 else "FAIL"}
        for name, count in failures.items()
    ]


def _replay_path(row: object, bars: pd.DataFrame, cfg: R26Config) -> tuple[str, pd.Timestamp, float, float]:
    index_ns = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    entry_time = pd.Timestamp(row.entry_time)
    start = int(np.searchsorted(index_ns, np.datetime64(entry_time, "ns").astype(np.int64), side="left"))
    end = start + cfg.path_horizon_minutes - 1
    direction = int(row.trade_direction)
    target = float(row.target_price)
    stop = float(row.stop_price)
    for pos in range(start, end + 1):
        high = float(bars.iloc[pos]["high"])
        low = float(bars.iloc[pos]["low"])
        stop_hit = low <= stop if direction > 0 else high >= stop
        target_hit = high >= target if direction > 0 else low <= target
        if stop_hit:
            open_price = float(bars.iloc[pos]["open"])
            fill = min(stop, open_price) if direction > 0 else max(stop, open_price)
            return "sl_first", pd.Timestamp(bars.index[pos]), fill, direction * (fill / float(row.entry_price) - 1.0)
        if target_hit:
            return "tp_first", pd.Timestamp(bars.index[pos]), target, direction * (target / float(row.entry_price) - 1.0)
    fill = float(bars.iloc[end]["close"])
    return "horizon_exit", pd.Timestamp(bars.index[end]), fill, direction * (fill / float(row.entry_price) - 1.0)


def _validate_paths(paths: pd.DataFrame, bars: pd.DataFrame, cfg: R26Config) -> list[dict[str, object]]:
    failures = {
        "first_passage_outcome": 0,
        "first_passage_exit_time": 0,
        "first_passage_exit_price": 0,
        "gross_return": 0,
        "cost_arithmetic": 0,
        "split_boundary": 0,
        "position_non_overlap": 0,
    }
    valid = paths.loc[paths["gross_return"].notna()].copy()
    for row in valid.itertuples(index=False):
        outcome, exit_time, exit_price, gross = _replay_path(row, bars, cfg)
        failures["first_passage_outcome"] += int(outcome != row.outcome)
        failures["first_passage_exit_time"] += int(exit_time != pd.Timestamp(row.exit_time))
        failures["first_passage_exit_price"] += int(not _close(exit_price, row.exit_price))
        failures["gross_return"] += int(not _close(gross, row.gross_return))
        for scale in cfg.cost_scales:
            expected = gross - cfg.market_roundtrip_cost * float(scale)
            failures["cost_arithmetic"] += int(not _close(expected, getattr(row, f"net_return_cost{float(scale):g}x")))
        split_end = cfg.validation_start if row.research_split == "discovery" else cfg.embargo_start
        failures["split_boundary"] += int(pd.Timestamp(row.exit_time) >= split_end)
    selected = valid.loc[valid["position_selected"].eq(True)]
    for _, part in selected.groupby(["research_split", "direction", "target_model"], sort=True):
        ordered = part.sort_values("entry_time", kind="stable")
        if len(ordered) > 1:
            failures["position_non_overlap"] += int(
                (pd.to_datetime(ordered["entry_time"]).iloc[1:].to_numpy() <= pd.to_datetime(ordered["exit_time"]).iloc[:-1].to_numpy()).sum()
            )
    return [
        {"check": name, "violations": count, "status": "PASS" if count == 0 else "FAIL"}
        for name, count in failures.items()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R26Config().validate()
    if pd.Timestamp(args.end_date) >= cfg.embargo_start:
        raise ValueError("validator physically forbids embargo and holdout data")
    report = Path(args.report_dir)
    events = pd.read_csv(report / "06_causal_event_table.csv.gz", compression="gzip")
    paths = pd.read_csv(report / "07_first_passage_paths.csv.gz", compression="gzip")
    for column in [name for name in events if name.endswith("_time")]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    for column in ("entry_time", "exit_time", "signal_available_time", "cross_available_time"):
        if column in paths:
            paths[column] = pd.to_datetime(paths[column], errors="coerce")
    if "position_selected" in paths:
        paths["position_selected"] = paths["position_selected"].astype(str).str.lower().eq("true")

    print("[r26-validator] load visible OKX and Binance sources", flush=True)
    bars = normalize_1m_bars(
        OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
            args.warmup_start_date, args.end_date
        )
    )
    metrics = BinanceFuturesMetricsLoader(symbol=args.binance_symbol, data_dir=args.data_dir).load_metrics(
        args.warmup_start_date,
        args.end_date,
        publication_lag=cfg.publication_lag,
        index_mode="none",
    )
    aligned = _metric_state(metrics, _five_state(bars, cfg), cfg)
    checks = _validate_events(events, aligned, bars, cfg) + _validate_paths(paths, bars, cfg)
    audit = pd.DataFrame(checks)
    audit.to_csv(report / "12_independent_replay_audit.csv", index=False)
    failed = audit.loc[audit["violations"].ne(0)]
    if not failed.empty:
        print(failed.to_string(index=False))
        return 1
    print(f"[r26-validator] PASS {len(audit)} checks across {len(events):,} events and {len(paths):,} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
