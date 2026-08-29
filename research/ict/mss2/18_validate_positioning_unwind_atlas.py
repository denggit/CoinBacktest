#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent raw-bar and arithmetic reconciliation for the saved R18 atlas."""
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

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.r18 import R18Config  # noqa: E402

DEFAULT_REPORT_DIR = Path("data/reports/research/ict/mss2/r18_positioning_unwind_path_atlas")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args(argv)


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R18Config().validate()
    root = Path(args.report_dir)
    events = pd.read_csv(
        root / "06_causal_event_table.csv.gz",
        parse_dates=["entry_time", "signal_available_time"],
        float_precision="round_trip",
    )
    paths = pd.read_csv(
        root / "07_first_passage_paths.csv.gz",
        parse_dates=["entry_time", "exit_time", "signal_available_time"],
        float_precision="round_trip",
    )
    score = pd.read_csv(root / "08_direction_target_scorecard.csv", float_precision="round_trip")
    if paths.empty:
        raise RuntimeError("saved R18 path table is empty")

    start = pd.Timestamp(paths["entry_time"].min()).normalize()
    end = pd.Timestamp(paths["entry_time"].max()) + pd.Timedelta(minutes=cfg.path_horizon_minutes)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(start, end)
    index = pd.to_datetime(bars.index)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)

    ordering_violations = 0
    exit_time_violations = 0
    exit_price_violations = 0
    replay_rows: list[tuple[str, str, float, float]] = []
    for row in paths.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        pos = int(index.searchsorted(entry_time, side="left"))
        if pos >= len(index) or pd.Timestamp(index[pos]) != entry_time:
            ordering_violations += 1
            continue
        final = min(len(index) - 1, pos + cfg.path_horizon_minutes - 1)
        direction = int(row.trade_direction)
        stop = float(row.stop_price)
        target = float(row.target_price)
        replay_outcome = "horizon_exit"
        replay_pos = final
        for current in range(pos, final + 1):
            stop_hit = low[current] <= stop if direction > 0 else high[current] >= stop
            target_hit = high[current] >= target if direction > 0 else low[current] <= target
            if stop_hit:
                replay_outcome, replay_pos = "sl_first", current
                break
            if target_hit:
                replay_outcome, replay_pos = "tp_first", current
                break
        replay_price = stop if replay_outcome == "sl_first" else target if replay_outcome == "tp_first" else float(close[replay_pos])
        ordering_violations += int(replay_outcome != str(row.outcome))
        exit_time_violations += int(pd.Timestamp(index[replay_pos]) != pd.Timestamp(row.exit_time))
        exit_price_violations += int(not np.isclose(replay_price, float(row.exit_price), rtol=0.0, atol=1e-10))
        replay_rows.append((str(row.setup_id), str(row.target_model), replay_price, direction * (replay_price / float(row.entry_price) - 1.0)))

    replay = pd.DataFrame(replay_rows, columns=["setup_id", "target_model", "replay_exit_price", "replay_gross_return"])
    joined = paths.merge(replay, on=["setup_id", "target_model"], how="left", validate="one_to_one")
    gross_diff = (pd.to_numeric(joined["gross_return"]) - pd.to_numeric(joined["replay_gross_return"])).abs()
    expected_2x = pd.to_numeric(joined["gross_return"]) - 2.0 * cfg.market_roundtrip_cost
    cost_diff = (pd.to_numeric(joined["net_return_cost2x"]) - expected_2x).abs()

    regrouped = []
    for keys, part in paths.groupby(["research_split", "direction", "target_model"], sort=True):
        regrouped.append((*keys, len(part), _pf(part["net_return_cost2x"])))
    regrouped_frame = pd.DataFrame(
        regrouped,
        columns=["research_split", "direction", "target_model", "replay_trades", "replay_pf2x"],
    )
    compare = score.merge(
        regrouped_frame,
        on=["research_split", "direction", "target_model"],
        how="outer",
        validate="one_to_one",
    )
    pf_diff = (pd.to_numeric(compare["net_pf_cost2x"]) - pd.to_numeric(compare["replay_pf2x"])).abs()
    rows = [
        {"check": "raw_bar_ordering", "violations": ordering_violations, "max_abs_difference": 0.0},
        {"check": "raw_bar_exit_time", "violations": exit_time_violations, "max_abs_difference": 0.0},
        {"check": "raw_bar_exit_price", "violations": exit_price_violations, "max_abs_difference": 0.0},
        {"check": "gross_return_formula", "violations": int((gross_diff > 1e-12).sum()), "max_abs_difference": float(gross_diff.max())},
        {"check": "cost2x_formula", "violations": int((cost_diff > 1e-12).sum()), "max_abs_difference": float(cost_diff.max())},
        {"check": "scorecard_trade_counts", "violations": int((pd.to_numeric(compare["trades"]) != pd.to_numeric(compare["replay_trades"])).sum()), "max_abs_difference": 0.0},
        {"check": "scorecard_pf2x", "violations": int((pf_diff > 1e-12).sum()), "max_abs_difference": float(pf_diff.max())},
        {"check": "unique_event_ids", "violations": int(events["setup_id"].duplicated().sum()), "max_abs_difference": 0.0},
        {"check": "unique_setup_target_paths", "violations": int(paths.duplicated(["setup_id", "target_model"]).sum()), "max_abs_difference": 0.0},
        {"check": "four_paths_per_included_setup", "violations": int(paths.groupby("setup_id").size().ne(4).sum()), "max_abs_difference": 0.0},
        {"check": "holdout_or_embargo_paths", "violations": int((pd.to_datetime(paths["entry_time"]) >= cfg.embargo_start).sum()), "max_abs_difference": 0.0},
    ]
    audit = pd.DataFrame(rows)
    audit["status"] = np.where(audit["violations"].eq(0), "PASS", "FAIL")
    audit.to_csv(root / "12_independent_reconciliation.csv", index=False)
    print(audit.to_string(index=False))
    if int(audit["violations"].sum()) != 0:
        raise RuntimeError("positioning path-atlas independent reconciliation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
