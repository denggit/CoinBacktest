#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent structural-state replay for R23."""
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

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

DEFAULT_REPORT_DIR = Path("data/reports/research/ict/mss2/r23_frozen_panic_wick_structural_long")
EXPERIMENT_ID = "ETH_ICT_MSS2_FROZEN_PANIC_WICK_LONG_R23"
EDGE_ID = "RESEARCH_ONLY_FROZEN_PANIC_WICK_STRUCTURAL_LONG"
TITLE = "ETH ICT MSS2 R23 Frozen Panic-Wick Structural Long"
TOL = 1e-10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _calendar(raw: pd.DataFrame) -> pd.DataFrame:
    source = raw.sort_index()
    index = pd.date_range(source.index.min(), source.index.max(), freq="1min")
    observed = pd.Series(1, index=source.index).reindex(index, fill_value=0)
    out = source.reindex(index)
    prior_close = pd.to_numeric(out["close"], errors="coerce").ffill()
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(prior_close)
    out["observed"] = observed.to_numpy(dtype=np.int8)
    return out


def replay(trades: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    bars = _calendar(raw)
    index = bars.index
    open_ = bars["open"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    observed = bars["observed"].to_numpy(dtype=np.int8)
    rows = []
    for trade in trades.itertuples(index=False):
        entry_time = pd.Timestamp(trade.entry_time)
        entry_pos = int(index.searchsorted(entry_time))
        event_low = float(trade.event_low)
        event_high = float(trade.event_high)
        deeper = float(trade.deeper_failure_price)
        swept = 0
        prior_below = False
        high_reclaimed = False
        trail = np.nan
        expected_reason = None
        expected_decision = pd.NaT
        expected_exit = pd.NaT
        gap_seen = False
        end_time = pd.Timestamp(trade.exit_decision_bar_time)
        end_pos = int(index.searchsorted(end_time))
        for pos in range(entry_pos, end_pos + 1):
            if observed[pos] != 1 or observed[pos + 1] != 1:
                gap_seen = True
                break
            bar_low = float(low[pos])
            bar_close = float(close[pos])
            if bar_low < event_low:
                if not prior_below:
                    swept += 1
                prior_below = True
            else:
                prior_below = False
            if bar_close >= event_high:
                high_reclaimed = True
                if not np.isfinite(trail):
                    trail = event_low
            if pos >= 2:
                pivot = pos - 1
                if low[pivot] <= low[pivot - 1] and low[pivot] <= low[pos] and low[pivot] > event_low:
                    candidate = float(low[pivot])
                    if high_reclaimed and (not np.isfinite(trail) or candidate > trail):
                        trail = candidate
            if swept >= 2 and bar_low < deeper and bar_close < event_low:
                expected_reason = "MULTI_SWEEP_DEEPER_FAIL"
            elif high_reclaimed and np.isfinite(trail) and bar_close < trail:
                expected_reason = "HIGHER_LOW_TRAIL_BREAK"
            if expected_reason is not None:
                expected_decision = index[pos]
                expected_exit = index[pos + 1]
                break
        exit_pos = int(index.searchsorted(expected_exit)) if pd.notna(expected_exit) else -1
        expected_price = float(open_[exit_pos]) if exit_pos >= 0 else np.nan
        expected_gross = expected_price / float(open_[entry_pos]) - 1.0 if np.isfinite(expected_price) else np.nan
        checks = {
            "entry_source_observed": observed[entry_pos] == 1,
            "entry_raw_open": abs(float(trade.entry_price) - float(open_[entry_pos])) <= TOL,
            "no_path_gap": not gap_seen,
            "first_reason_replayed": expected_reason == str(trade.exit_reason),
            "decision_time_replayed": expected_decision == pd.Timestamp(trade.exit_decision_bar_time),
            "exit_time_replayed": expected_exit == pd.Timestamp(trade.exit_time),
            "exit_price_replayed": abs(expected_price - float(trade.exit_price)) <= TOL,
            "gross_replayed": abs(expected_gross - float(trade.gross_return)) <= TOL,
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append({"trade_id": trade.trade_id, **{name: int(value) for name, value in checks.items()}, "violations": len(failed), "failed_checks": "PASS" if not failed else ",".join(failed)})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    trades = pd.read_csv(report_dir / "03_trade_paths.csv.gz", parse_dates=["entry_time", "exit_decision_bar_time", "exit_time"])
    trades = trades.loc[trades["path_status"].eq("included")]
    raw = OKXTradeBarLoader(args.symbol, "1m").fetch_data_by_date_range(args.start_date, args.end_date, build_missing=False)
    detail = replay(trades, raw)
    checks = [column for column in detail.columns if column not in {"trade_id", "violations", "failed_checks"}]
    audit = pd.DataFrame([
        {"check": check, "rows_checked": len(detail), "violations": int(detail[check].eq(0).sum()), "status": "PASS" if bool(detail[check].eq(1).all()) else "FAIL"}
        for check in checks
    ])
    detail.to_csv(report_dir / "08_independent_trade_replay.csv", index=False)
    audit.to_csv(report_dir / "09_independent_replay_audit.csv", index=False)
    if not args.skip_review_pack:
        finalize_research_report(report_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(audit.to_string(index=False))
    return 0 if int(audit["violations"].sum()) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

