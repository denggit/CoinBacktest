#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R17: no-stop opposite-liquidity diagnostic.

Frozen question:
    08:30 prominent 15m liquidity pair
    -> one side swept
    -> first visible 1m MSS
    -> break-associated FVG near-edge limit entry
    -> NO intraday stop after fill
    -> opposite external liquidity TP, otherwise 16:30 ET session close.

This is a diagnostic experiment, not a live strategy proposal.  It isolates the
value/cost of the terminal-extreme stop by changing exactly one post-fill rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE
from src.research_common.ict.no_stop_opposite_target import (
    NoStopReplayConfig,
    replay_no_stop_to_opposite_or_close,
    summarize_eod_failures,
    summarize_no_stop,
    summarize_no_stop_by_period,
)
from src.research_common.ict.premarket_mss_fvg import NY_TZ, ny_date_bounds_to_source_naive, source_naive_to_new_york
from src.research_common.ict.spot_perp_overlap import densify_equity_minutes_causally
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_R16_CACHE = "data/reports/research/ict/soxl/mss/r16_entry_archetype_survival_atlas_alpaca_2023_2026_08"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r17_no_stop_opposite_liquidity_diagnostic"
DEFAULT_RANGE_MODEL = "prominent_15m_pair_0830"
DEFAULT_ARCHETYPE = "mss_first_visible_break_fvg_near"
DEFAULT_EXECUTION_TF = "1m"


def _source_offset_hours(text: str) -> int:
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R17 no-stop opposite-liquidity diagnostic")
    p.add_argument("--data-source", choices=("alpaca", "okx"), default="alpaca")
    p.add_argument("--symbol", default="SOXL-USDT-SWAP")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--r16-cache-dir", default=DEFAULT_R16_CACHE)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--range-model", default=DEFAULT_RANGE_MODEL)
    p.add_argument("--entry-archetype", default=DEFAULT_ARCHETYPE)
    p.add_argument("--execution-tf", default=DEFAULT_EXECUTION_TF)
    p.add_argument("--golden-date", default="2026-08-05")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


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
        bars = densify_equity_minutes_causally(bars)
    else:
        offset = _source_offset_hours(OKX_LOADER_TIMEZONE)
        start_src, end_src = ny_date_bounds_to_source_naive(args.start_date, args.end_date, source_offset_hours=offset)
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        raw = loader.load_local_data() if args.local_only else loader.fetch_data_by_date_range(start_src, end_src)
        if args.local_only and not raw.empty:
            raw = raw.loc[(raw.index >= start_src) & (raw.index <= end_src)].copy()
        if raw.empty:
            raise RuntimeError("OKX loader returned no data")
        bars = source_naive_to_new_york(raw, source_offset_hours=offset)
    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()
    print(f"[load] source={args.data_source} rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _validate_manifest(cache: Path, args: argparse.Namespace) -> dict[str, object]:
    path = cache / "13_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"R16 manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("research_id")) != "R16":
        raise RuntimeError(f"expected R16 cache, got research_id={data.get('research_id')}")
    for key in ("data_source", "start_date", "end_date"):
        expected = str(getattr(args, key))
        if str(data.get(key)) != expected:
            raise RuntimeError(f"R16 cache {key}={data.get(key)} does not match requested {expected}")
    return data


def _read_frozen_lifecycle(cache: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = cache / "05_entry_survival_lifecycle.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = [
        "ny_date","range_model","entry_archetype","entry_family","execution_tf","entry_order_type",
        "trade_side","path_event_id","event_id","first_raid_time","source_level_price","target_price",
        "entry_available_time","entry_price","entry_price_replay","stop_price","filled","fill_time","fill_wait_minutes",
        "stop_hit","stop_time","stop_minutes_after_fill","milestone_100_before_stop","net_return_exit_100","exit_reason_100",
        "causal_visibility_percentile","terminal_extreme_time","terminal_extreme_price","mss_reference_time","mss_reference_price",
        "break_bar_start","break_available_time","break_close_cross","fvg_middle_relation_to_break","fvg_near_edge_entry",
        "initial_risk_abs","initial_risk_frac_range","signal_minutes_from_raid","raid_count_so_far_at_entry",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in usecols if c in header]
    q = pd.read_csv(path, usecols=use, low_memory=False)
    mask = (
        q["range_model"].astype(str).eq(str(args.range_model))
        & q["entry_archetype"].astype(str).eq(str(args.entry_archetype))
        & q["execution_tf"].astype(str).eq(str(args.execution_tf))
    )
    selected = q.loc[mask].copy()
    if selected.empty:
        available = q.groupby(["range_model","entry_archetype","execution_tf"], dropna=False).size().sort_values(ascending=False).head(25)
        raise RuntimeError(
            "No R16 lifecycle rows matched the frozen selection. Top available groups:\n"
            + available.to_string()
        )
    if selected["filled"].dtype != bool:
        selected["filled"] = selected["filled"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    frozen = selected.loc[selected["filled"]].copy()
    for c in [x for x in frozen.columns if x.endswith("_time") or x in {"fill_time","break_bar_start","entry_available_time"}]:
        frozen[c] = pd.to_datetime(frozen[c], errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    return selected, frozen


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _design_text(args: argparse.Namespace, all_rows: int, fills: int) -> str:
    return f"""# SOXL ICT R17 — No-Stop Opposite-Liquidity Diagnostic

## Frozen hypothesis
This experiment changes exactly one rule after the R16 entry has filled.

- liquidity map: `{args.range_model}`;
- entry: `{args.entry_archetype}` on `{args.execution_tf}`;
- R16 entry signal, order price and fill time are frozen;
- **no intraday stop after fill**;
- only TP is the opposite frozen external-liquidity boundary;
- if TP is not reached by 16:30 ET, exit at the final 1m session close;
- round-trip cost = `{args.round_trip_cost:.6f}`;
- original terminal-extreme stop is retained only as a counterfactual diagnostic.

Selected R16 lifecycle candidates: {all_rows:,}; frozen fills: {fills:,}.

## Interpretation boundary
This is not a proposal to trade without a stop.  Its purpose is to distinguish:
1. entries that the old terminal stop prematurely washed out but later reached opposite liquidity;
2. genuinely wrong reversals whose no-stop EOD/MAE tail becomes unacceptable.

No 25/50/75 milestone is used in the strategy outcome.  Future bars are used only for TP/EOD counterfactual replay after the already-frozen fill.
"""


def _self_test() -> int:
    from src.research_common.ict.no_stop_opposite_target import replay_no_stop_to_opposite_or_close
    idx = pd.date_range("2026-08-05 08:30", periods=8, freq="1min", tz=NY_TZ)
    bars = pd.DataFrame({
        "open":  [100,100,99,98,99,101,104,105],
        "high":  [101,101,100,99,102,104,106,106],
        "low":   [99,99,97,96,98,100,103,104],
        "close": [100,99,98,98,101,103,105,105],
    }, index=idx)
    fills = pd.DataFrame([{
        "ny_date":"2026-08-05","filled":True,"fill_time":idx[1],"trade_side":"LONG","entry_order_type":"limit",
        "entry_price":100.0,"entry_price_replay":float("nan"),"target_price":105.0,"stop_price":98.0,
        "stop_hit":True,"stop_time":idx[2],"milestone_100_before_stop":False,"net_return_exit_100":-0.0211,
    }])
    out = replay_no_stop_to_opposite_or_close(bars, fills, config=NoStopReplayConfig(round_trip_cost=0.0011))
    assert len(out) == 1
    r = out.iloc[0]
    assert bool(r["no_stop_tp_hit"])
    assert bool(r["rescued_after_old_terminal_stop"])
    assert r["no_stop_exit_reason"] == "opposite_liquidity_tp"
    print("R17 self-test PASS", flush=True)
    return 0


def run_research(args: argparse.Namespace) -> bool:
    if args.self_test:
        _self_test(); return True
    cache = Path(args.r16_cache_dir)
    manifest = _validate_manifest(cache, args)
    all_selected, fills = _read_frozen_lifecycle(cache, args)
    print(
        f"[selection] range={args.range_model} archetype={args.entry_archetype} tf={args.execution_tf} "
        f"candidates={len(all_selected):,} fills={len(fills):,}", flush=True,
    )
    bars = _load_1m(args)
    cfg = NoStopReplayConfig(round_trip_cost=float(args.round_trip_cost))
    replayed = replay_no_stop_to_opposite_or_close(bars, fills, config=cfg)
    score = summarize_no_stop(replayed)
    period = summarize_no_stop_by_period(replayed)
    eod = summarize_eod_failures(replayed)

    counter_cols = [
        c for c in [
            "ny_date","path_event_id","trade_side","fill_time","entry_price","target_price","stop_price",
            "old_terminal_stop_hit","old_terminal_stop_before_tp","rescued_after_old_terminal_stop",
            "no_stop_tp_hit","no_stop_tp_time","no_stop_exit_reason","no_stop_exit_time","no_stop_net_return",
            "no_stop_mae_pct","no_stop_mae_old_r","reward_to_opposite_old_r",
        ] if c in replayed.columns
    ]
    counter = replayed[counter_cols].copy() if counter_cols else pd.DataFrame()
    if not counter.empty:
        counter = counter.sort_values(["rescued_after_old_terminal_stop","ny_date"], ascending=[False, True], kind="mergesort")

    golden = replayed.loc[replayed["ny_date"].astype(str).eq(str(args.golden_date))].copy() if "ny_date" in replayed else pd.DataFrame()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "00_research_design.md").write_text(_design_text(args, len(all_selected), len(fills)), encoding="utf-8")
    _write(all_selected, out / "01_frozen_r16_selection.csv")
    _write(replayed, out / "02_no_stop_trade_lifecycle.csv")
    _write(score, out / "03_no_stop_vs_old_stop_scorecard.csv")
    _write(period, out / "04_period_stability.csv")
    _write(counter, out / "05_old_stop_rescue_audit.csv")
    _write(eod, out / "06_session_close_tail_risk.csv")
    _write(golden, out / f"07_golden_replay_{args.golden_date}.csv")
    mf = {
        "research_id":"R17",
        "diagnostic":"no_stop_opposite_liquidity_or_session_close",
        "data_source":args.data_source,
        "start_date":args.start_date,
        "end_date":args.end_date,
        "range_model":args.range_model,
        "entry_archetype":args.entry_archetype,
        "execution_tf":args.execution_tf,
        "r16_cache":str(cache),
        "r16_manifest":manifest,
        "selected_candidates":len(all_selected),
        "frozen_fills":len(fills),
        "round_trip_cost":float(args.round_trip_cost),
    }
    (out / "08_manifest.json").write_text(json.dumps(mf, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id="R17_NO_STOP_OPPOSITE_LIQUIDITY", title="SOXL ICT R17 No-Stop Opposite-Liquidity Diagnostic")
    print(f"[done] report={out}", flush=True)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return 0 if run_research(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
