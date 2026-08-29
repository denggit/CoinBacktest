#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R13: semantic consolidation + target-state / FVG-entry atlas.

R13 follows the 2026-08-05 OKX golden replay rather than adding stricter
filters.  It keeps R12's wide causal atlas but consolidates it into a practical
research universe:

* partial/equal-like vs accepted/deep liquidity consumption is explicit;
* partially-raided opposite session liquidity may remain a conservative target;
* the old session level expires with the session/date; later persistence must
  come from a separately confirmed swing;
* one sweep can still have micro and later stronger MSS attempts;
* replay uses an outermost-barrier + visibility-tier narrative view so thousands
  of mathematical pivots do not become thousands of fake opportunities;
* FVG train execution is compared without hard-capping by Swing +/- $0.10;
* close-break next-open market entry is repaired and directly compared.

No target-state threshold or FVG choice is declared final in R13.  The purpose
is to discover which interpretation survives 2023H2-2024 discovery, 2025
forward, and 2026 late holdout while keeping opportunity frequency healthy.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE
from src.research_common.ict.entry_expansion import (
    EntryExpansionConfig,
    build_intraday_15m_swing_catalog,
    build_intraday_15m_sweep_events,
)
from src.research_common.ict.premarket_mss_fvg import (
    EPS,
    NY_TZ,
    ReplayScenario,
    aggregate_closed_bars,
    build_data_quality_table,
    eligible_ny_dates,
    make_synthetic_ict_day,
    ny_date_bounds_to_source_naive,
    replay_attempts,
    slice_ny_day,
    source_naive_to_new_york,
)
from src.research_common.ict.premarket_mss_fvg_v2 import SweepEpisodeConfig, build_all_premarket_levels_v2
from src.research_common.ict.semantic_consolidation import (
    LiquidityConsumptionConfig,
    add_market_next_open_choices,
    attach_consumption_state_to_fvg_rows,
    attach_market_next_open_from_bars,
    build_liquidity_consumption_query_index,
    consolidate_fvg_entry_choices,
    expand_market_target_state_variants,
    expand_target_state_variants,
    select_reference_narratives,
    select_tiered_primary_narratives,
)
from src.research_common.ict.spot_perp_overlap import build_equity_proxy_data_quality_table, densify_equity_minutes_causally
from src.research_common.ict.structure_entry_semantics import (
    StructureSemanticConfig,
    build_causal_sweep_events_for_levels,
    build_dual_session_liquidity_levels,
    build_r13_primary_break_fvg_compact,
    build_visible_swing_catalog,
)
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r13_semantic_consolidation"


def _csv_ints(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals:
        raise ValueError("empty integer list")
    return vals


def _source_offset_hours(text: str) -> int:
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R13 semantic consolidation atlas")
    p.add_argument("--data-source", choices=("okx", "alpaca"), default="alpaca")
    p.add_argument("--symbol", default="SOXL-USDT-SWAP")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--include-us-equity-holidays", action="store_true")
    p.add_argument("--required-day-coverage", type=float, default=0.995)
    p.add_argument("--execution-timeframes", default="1,2,5")
    p.add_argument("--structure-lookback-minutes", type=int, default=150)
    p.add_argument("--entry-buffer", type=float, default=0.10)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=2.0)
    p.add_argument("--golden-date", default="2026-08-05")
    p.add_argument("--no-progress", action="store_true")
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
    if args.data_source == "alpaca":
        bars = densify_equity_minutes_causally(bars)
    print(f"[load] source={args.data_source} rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _period_label(ts) -> str:
    t = pd.Timestamp(ts)
    if t < pd.Timestamp("2025-01-01", tz=t.tz):
        return "discovery_2023h2_2024"
    if t < pd.Timestamp("2026-01-01", tz=t.tz):
        return "forward_2025"
    return "late_holdout_2026"


def _make_replay_key(df: pd.DataFrame) -> pd.Series:
    """Exact factorised replay identity without per-row Python string joins."""
    cols = ["ny_date", "trade_side", "entry_order_type", "entry_available_time", "entry_price", "stop_price", "target_price_r13"]
    work = df[cols].copy()
    for c in ["entry_price", "stop_price", "target_price_r13"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").round(8)
    work["entry_available_time"] = pd.to_datetime(work["entry_available_time"])
    mi = pd.MultiIndex.from_frame(work, names=cols)
    codes, _ = pd.factorize(mi, sort=False)
    return pd.Series(codes.astype(np.int64), index=df.index, name="replay_key_r13")

def _replay_limit_unique(bars: pd.DataFrame, rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Fast exact replay for R13 limit entries using per-day NumPy arrays.

    This preserves the conservative same-bar semantics of ``replay_attempts``
    while avoiding a Python loop over every minute bar for every variant.
    """
    if rows.empty:
        return pd.DataFrame()
    q = rows.copy()
    q["replay_key_r13"] = _make_replay_key(q)
    unique = q.drop_duplicates("replay_key_r13", keep="first").copy()
    cache: dict[str, dict[str, object]] = {}
    out_rows: list[dict[str, object]] = []
    prog = ProgressReporter(label="[stage8-limit] replay", total=max(1, len(unique)), every=5000, enabled=not args.no_progress)
    for r in unique.to_dict("records"):
        day_text = str(r["ny_date"])
        data = cache.get(day_text)
        if data is None:
            day = slice_ny_day(bars, pd.Timestamp(day_text).date(), pd.Timestamp("08:30").time(), pd.Timestamp("16:30").time())
            idx = pd.DatetimeIndex(day.index).as_unit("ns")
            data = {
                "idx": idx,
                "idx_ns": idx.asi8,
                "open": pd.to_numeric(day.get("open"), errors="coerce").to_numpy(float),
                "high": pd.to_numeric(day.get("high"), errors="coerce").to_numpy(float),
                "low": pd.to_numeric(day.get("low"), errors="coerce").to_numpy(float),
                "close": pd.to_numeric(day.get("close"), errors="coerce").to_numpy(float),
            }
            cache[day_text] = data
        idx = data["idx"]; idx_ns = data["idx_ns"]; high = data["high"]; low = data["low"]; close = data["close"]
        is_long = str(r["trade_side"]) == "LONG"
        signal_time = pd.Timestamp(r["entry_available_time"])
        limit_price = float(r["entry_price"]); stop = float(r["stop_price"]); target = float(r["target_price_r13"])
        risk_abs = limit_price - stop if is_long else stop - limit_price
        risk_pct = risk_abs / limit_price if np.isfinite(limit_price) and abs(limit_price) > EPS else np.nan
        session_end = pd.Timestamp(day_text).tz_localize(NY_TZ) + pd.Timedelta(hours=16, minutes=30)
        rec = {
            "replay_key_r13": r["replay_key_r13"], "filled": False, "fill_time": pd.NaT, "entry_price": np.nan,
            "exit_time": pd.NaT, "exit_price": np.nan, "exit_reason": "", "gross_return": np.nan, "net_return": np.nan,
            "gross_r": np.nan, "net_r": np.nan, "mfe_r": np.nan, "mae_r": np.nan, "bars_held_1m": 0,
            "notional_multiple": np.nan, "account_return": np.nan, "round_trip_cost": float(args.round_trip_cost),
            "order_status": "pending", "cancel_reason": "", "same_bar_entry_stop_ambiguous": False,
            "same_bar_entry_target_ambiguous": False, "same_bar_stop_target_ambiguous": False,
        }
        if not np.isfinite(risk_abs) or risk_abs <= EPS or not np.isfinite(risk_pct) or risk_pct <= 0:
            rec.update(order_status="invalid", cancel_reason="invalid_risk"); out_rows.append(rec); prog.update(1); continue
        if signal_time >= session_end:
            rec.update(order_status="cancelled", cancel_reason="order_activated_at_or_after_session_end"); out_rows.append(rec); prog.update(1); continue
        start_pos = int(np.searchsorted(idx_ns, int(signal_time.value), side="left"))
        if start_pos >= len(idx_ns):
            rec.update(order_status="cancelled", cancel_reason="no_1m_path_after_order"); out_rows.append(rec); prog.update(1); continue
        hi = high[start_pos:]; lo = low[start_pos:]; cl = close[start_pos:]
        entry_touch = lo <= limit_price if is_long else hi >= limit_price
        stop_touch = lo <= stop if is_long else hi >= stop
        target_touch = hi >= target if is_long else lo <= target
        first_event = entry_touch | stop_touch | target_touch
        ev_rel = int(np.flatnonzero(first_event)[0]) if np.any(first_event) else -1
        if ev_rel < 0:
            rec.update(order_status="cancelled", cancel_reason="session_end_unfilled"); out_rows.append(rec); prog.update(1); continue
        abs_ev = start_pos + ev_rel
        bar_end = pd.Timestamp(idx[abs_ev]) + pd.Timedelta(minutes=1)
        et = bool(entry_touch[ev_rel]); st = bool(stop_touch[ev_rel]); tp = bool(target_touch[ev_rel])
        if tp:
            rec.update(order_status="cancelled", cancel_reason="target_and_entry_same_bar_ambiguous_cancel" if et else "opposite_premarket_extreme_reached_before_fill", same_bar_entry_target_ambiguous=bool(et))
            out_rows.append(rec); prog.update(1); continue
        if st and not et:
            rec.update(order_status="cancelled", cancel_reason="sweep_extreme_invalidated_before_fill")
            out_rows.append(rec); prog.update(1); continue
        # Entry touched.  If stop also touched in the same bar, use the same
        # conservative entry-then-stop assumption as the reference replay.
        rec.update(filled=True, order_status="filled", fill_time=pd.Timestamp(idx[abs_ev]), entry_price=limit_price)
        entry_rel = ev_rel
        if st:
            exit_price = stop; exit_time = bar_end; reason = "entry_then_stop_same_bar_conservative"
            mfe = 0.0; mae = -1.0; bars_held = 1; rec["same_bar_entry_stop_ambiguous"] = True
        else:
            post_hi = hi[entry_rel+1:]; post_lo = lo[entry_rel+1:]; post_cl = cl[entry_rel+1:]
            post_st = post_lo <= stop if is_long else post_hi >= stop
            post_tp = post_hi >= target if is_long else post_lo <= target
            hit_any = post_st | post_tp
            hit_rel = int(np.flatnonzero(hit_any)[0]) if np.any(hit_any) else -1
            pre_n = hit_rel if hit_rel >= 0 else len(post_hi)
            if pre_n > 0:
                pre_hi = post_hi[:pre_n]; pre_lo = post_lo[:pre_n]
                mfe = float(np.nanmax((pre_hi-limit_price)/risk_abs)) if is_long else float(np.nanmax((limit_price-pre_lo)/risk_abs))
                mae = float(np.nanmin((pre_lo-limit_price)/risk_abs)) if is_long else float(np.nanmin((limit_price-pre_hi)/risk_abs))
                if not np.isfinite(mfe): mfe = 0.0
                if not np.isfinite(mae): mae = 0.0
                mfe = max(0.0, mfe); mae = min(0.0, mae)
            else:
                mfe = 0.0; mae = 0.0
            if hit_rel >= 0:
                abs_hit = abs_ev + 1 + hit_rel
                st2 = bool(post_st[hit_rel]); tp2 = bool(post_tp[hit_rel])
                if st2:
                    exit_price = stop; reason = "stop_first_same_bar_both_conservative" if tp2 else "structural_sweep_extreme_stop"; mae = min(mae, -1.0)
                    if tp2: rec["same_bar_stop_target_ambiguous"] = True
                else:
                    exit_price = target; reason = "opposite_premarket_extreme_target"; mfe = max(mfe, (target-limit_price)/risk_abs if is_long else (limit_price-target)/risk_abs)
                exit_time = pd.Timestamp(idx[abs_hit]) + pd.Timedelta(minutes=1); bars_held = hit_rel + 2
            else:
                # Filled but no stop/target before session close.  Include all
                # post-fill bars in MFE/MAE and close at the final 1m close.
                if len(post_hi) > 0:
                    full_mfe = float(np.nanmax((post_hi-limit_price)/risk_abs)) if is_long else float(np.nanmax((limit_price-post_lo)/risk_abs))
                    full_mae = float(np.nanmin((post_lo-limit_price)/risk_abs)) if is_long else float(np.nanmin((limit_price-post_hi)/risk_abs))
                    if np.isfinite(full_mfe): mfe = max(mfe, full_mfe)
                    if np.isfinite(full_mae): mae = min(mae, full_mae)
                exit_price = float(cl[-1]); exit_time = session_end; reason = "session_1630_close"; bars_held = len(hi) - entry_rel
        gross=(exit_price/limit_price-1.0)*(1.0 if is_long else -1.0); net=gross-float(args.round_trip_cost); notional=min(float(args.max_notional_multiple),float(args.risk_fraction)/risk_pct)
        rec.update(exit_time=exit_time,exit_price=float(exit_price),exit_reason=reason,gross_return=float(gross),net_return=float(net),gross_r=float(gross/risk_pct),net_r=float(net/risk_pct),mfe_r=float(mfe),mae_r=float(mae),bars_held_1m=int(max(1,bars_held)),notional_multiple=float(notional),account_return=float(net*notional),cancel_reason="")
        out_rows.append(rec); prog.update(1)
    prog.close()
    life = pd.DataFrame(out_rows)
    outcomes = [
        "replay_key_r13", "filled", "fill_time", "entry_price", "exit_time", "exit_price", "exit_reason",
        "gross_return", "net_return", "gross_r", "net_r", "mfe_r", "mae_r", "bars_held_1m",
        "notional_multiple", "account_return", "round_trip_cost", "order_status", "cancel_reason",
        "same_bar_entry_stop_ambiguous", "same_bar_entry_target_ambiguous", "same_bar_stop_target_ambiguous",
    ]
    return q.drop(columns=[c for c in outcomes if c in q.columns and c != "replay_key_r13"], errors="ignore").merge(
        life[[c for c in outcomes if c in life.columns]], on="replay_key_r13", how="left", validate="many_to_one"
    )

def _replay_market_unique(bars: pd.DataFrame, rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    q = rows.copy()
    q["replay_key_r13"] = _make_replay_key(q)
    unique = q.drop_duplicates("replay_key_r13", keep="first").copy()
    out_rows = []
    cache: dict[str, dict[str, object]] = {}
    prog = ProgressReporter(label="[stage8-market] replay", total=max(1, len(unique)), every=5000, enabled=not args.no_progress)
    for j, r in enumerate(unique.to_dict("records"), start=1):
        day_text = str(r["ny_date"])
        data = cache.get(day_text)
        if data is None:
            day = slice_ny_day(bars, pd.Timestamp(day_text).date(), pd.Timestamp("08:30").time(), pd.Timestamp("16:30").time())
            idx = pd.DatetimeIndex(day.index).as_unit("ns")
            data = {
                "idx": idx,
                "idx_ns": idx.asi8,
                "high": pd.to_numeric(day.get("high"), errors="coerce").to_numpy(float),
                "low": pd.to_numeric(day.get("low"), errors="coerce").to_numpy(float),
                "close": pd.to_numeric(day.get("close"), errors="coerce").to_numpy(float),
            }
            cache[day_text] = data
        idx = data["idx"]; idx_ns = data["idx_ns"]; high = data["high"]; low = data["low"]; close = data["close"]
        is_long = str(r["trade_side"]) == "LONG"
        entry = float(r["entry_price"]); stop = float(r["stop_price"]); target = float(r["target_price_r13"])
        fill = pd.Timestamp(r["market_next_open_time"])
        risk = entry - stop if is_long else stop - entry
        rec = {"replay_key_r13":r["replay_key_r13"],"filled":False,"fill_time":fill,"entry_price":entry,"exit_time":pd.NaT,"exit_price":np.nan,"exit_reason":"","gross_return":np.nan,"net_return":np.nan,"gross_r":np.nan,"net_r":np.nan,"mfe_r":0.0,"mae_r":0.0,"bars_held_1m":0,"notional_multiple":np.nan,"account_return":np.nan,"round_trip_cost":float(args.round_trip_cost),"order_status":"invalid","cancel_reason":""}
        if not np.isfinite(risk) or risk <= EPS:
            rec["cancel_reason"]="invalid_risk"; out_rows.append(rec); prog.update(1); continue
        start_pos = int(np.searchsorted(idx_ns, int(fill.value), side="left"))
        if start_pos >= len(idx_ns):
            rec["cancel_reason"]="no_path"; out_rows.append(rec); prog.update(1); continue
        hi = high[start_pos:]; lo = low[start_pos:]; cl = close[start_pos:]
        st = lo <= stop if is_long else hi >= stop
        tp = hi >= target if is_long else lo <= target
        hit_any = st | tp
        hit_rel = int(np.flatnonzero(hit_any)[0]) if np.any(hit_any) else -1
        rec["filled"] = True; rec["order_status"] = "filled"
        pre_n = hit_rel if hit_rel >= 0 else len(hi)
        if pre_n > 0:
            pre_hi = hi[:pre_n]; pre_lo = lo[:pre_n]
            mfe = float(np.nanmax((pre_hi-entry)/risk)) if is_long else float(np.nanmax((entry-pre_lo)/risk))
            mae = float(np.nanmin((pre_lo-entry)/risk)) if is_long else float(np.nanmin((entry-pre_hi)/risk))
            if not np.isfinite(mfe): mfe = 0.0
            if not np.isfinite(mae): mae = 0.0
            mfe = max(0.0, mfe); mae = min(0.0, mae)
        else:
            mfe = 0.0; mae = 0.0
        if hit_rel >= 0:
            abs_pos = start_pos + hit_rel
            if bool(st[hit_rel]):
                exit_price = stop; reason = "stop_first_same_bar_both_conservative" if bool(tp[hit_rel]) else "structural_sweep_extreme_stop"; mae = -1.0
            else:
                exit_price = target; reason = "target"; mfe = max(mfe, (target-entry)/risk if is_long else (entry-target)/risk)
            exit_time = pd.Timestamp(idx[abs_pos]) + pd.Timedelta(minutes=1)
            bars_held = hit_rel + 1
        else:
            exit_price = float(cl[-1]); exit_time = pd.Timestamp(idx[-1]) + pd.Timedelta(minutes=1); reason = "session_1630_close"; bars_held = len(hi)
        gross=(exit_price/entry-1.0)*(1.0 if is_long else -1.0); net=gross-float(args.round_trip_cost); risk_pct=risk/entry; notional=min(float(args.max_notional_multiple),float(args.risk_fraction)/risk_pct)
        rec.update(exit_time=exit_time,exit_price=exit_price,exit_reason=reason,gross_return=gross,net_return=net,gross_r=gross/risk_pct,net_r=net/risk_pct,mfe_r=mfe,mae_r=mae,bars_held_1m=int(bars_held),notional_multiple=notional,account_return=net*notional,cancel_reason="")
        out_rows.append(rec); prog.update(1)
    prog.close()
    life = pd.DataFrame(out_rows)
    return q.merge(life, on="replay_key_r13", how="left", validate="many_to_one", suffixes=("", "_replay"))

def _summary(life: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if life.empty:
        return pd.DataFrame()
    rows=[]
    for key,g in life.groupby(group_cols,dropna=False,sort=True):
        if not isinstance(key, tuple): key=(key,)
        f=g.loc[g["filled"].fillna(False).astype(bool)]
        x=pd.to_numeric(f["net_return"],errors="coerce").dropna()
        gross=pd.to_numeric(f["gross_return"],errors="coerce").dropna()
        gains=float(x[x>0].sum()); losses=float(-x[x<0].sum()); gg=float(gross[gross>0].sum()); gl=float(-gross[gross<0].sum())
        rows.append({**dict(zip(group_cols,key)),"attempts":len(g),"filled_trades":len(f),"fill_rate":len(f)/len(g) if len(g) else np.nan,"win_rate":float((x>0).mean()) if len(x) else np.nan,"profit_factor":gains/losses if losses>0 else (np.inf if gains>0 else np.nan),"gross_profit_factor":gg/gl if gl>0 else (np.inf if gg>0 else np.nan),"mean_net_return":float(x.mean()) if len(x) else np.nan,"mean_net_r":float(pd.to_numeric(f.get("net_r"),errors="coerce").mean()) if len(f) else np.nan})
    return pd.DataFrame(rows)


def _cap_diagnostic(life: pd.DataFrame) -> pd.DataFrame:
    if life.empty or "legacy_swing_0p10_cap_pass_r13" not in life:
        return pd.DataFrame()
    q=life.loc[life["filled"].fillna(False).astype(bool) & life["legacy_swing_0p10_cap_pass_r13"].notna()].copy()
    if q.empty: return pd.DataFrame()
    q["legacy_cap_group"] = np.where(q["legacy_swing_0p10_cap_pass_r13"].astype(bool),"inside_swing_0p10","would_be_cut_by_swing_0p10")
    return _summary(q,["execution_tf","entry_model_r13","target_model_r13","legacy_cap_group","period_r13"])


def _distance_atlas(life: pd.DataFrame) -> pd.DataFrame:
    if life.empty: return pd.DataFrame()
    q=life.loc[life["filled"].fillna(False).astype(bool)].copy()
    x=pd.to_numeric(q.get("entry_distance_abs_r13"),errors="coerce")
    q=q.loc[x.notna()].copy()
    if q.empty:return pd.DataFrame()
    # Discovery-only quartile edges; forward/holdout reuse exactly the same bins.
    rows=[]
    for (tf,model),g in q.groupby(["execution_tf","entry_model_r13"],sort=True):
        disc=g.loc[g["period_r13"].eq("discovery_2023h2_2024")]
        vals=pd.to_numeric(disc["entry_distance_abs_r13"],errors="coerce").dropna()
        if len(vals)<20: continue
        edges=np.unique(np.nanquantile(vals,[0,0.25,0.5,0.75,1.0]))
        if len(edges)<3: continue
        edges[0]=-np.inf; edges[-1]=np.inf
        gg=g.copy(); gg["distance_bin_r13"]=pd.cut(pd.to_numeric(gg["entry_distance_abs_r13"],errors="coerce"),bins=edges,include_lowest=True,duplicates="drop").astype(str)
        s=_summary(gg,["execution_tf","entry_model_r13","target_model_r13","distance_bin_r13","period_r13"])
        s["discovery_edges"]="|".join(str(float(v)) for v in edges)
        rows.append(s)
    return pd.concat(rows,ignore_index=True,sort=False) if rows else pd.DataFrame()


def _write(df: pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); print(f"[write] {path.name} rows={len(df):,}",flush=True)


def run_research(bars: pd.DataFrame,args:argparse.Namespace)->dict[str,Path]:
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    days=eligible_ny_dates(bars,start_date=args.start_date,end_date=args.end_date,exclude_equity_holidays=not args.include_us_equity_holidays)
    quality=build_equity_proxy_data_quality_table(bars,days) if args.data_source=="alpaca" else build_data_quality_table(bars,days,required_coverage=float(args.required_day_coverage))
    valid=set(quality.loc[quality["coverage_pass"],"ny_date"].astype(str)); days=[pd.Timestamp(x).date() for x in sorted(valid)]
    if not days: raise RuntimeError("no valid sessions")
    cfg=StructureSemanticConfig(execution_timeframes=_csv_ints(args.execution_timeframes),structure_lookback_minutes=int(args.structure_lookback_minutes),absolute_entry_buffer=float(args.entry_buffer))
    c_cfg=LiquidityConsumptionConfig()
    stage=ProgressReporter(label="[research] R13 stages",total=12,every=1,enabled=not args.no_progress)

    pm=build_all_premarket_levels_v2(bars,days,pivot_left=2,pivot_right=2,episode_config=SweepEpisodeConfig())
    major=pm.loc[pm["level_type"].eq("major_15m_swing")].copy() if not pm.empty else pd.DataFrame()
    if not major.empty: major["liquidity_family"]="major_15m_swing"
    dual,running=build_dual_session_liquidity_levels(bars,days,config=cfg)
    levels=pd.concat([x for x in (dual,major) if not x.empty],ignore_index=True,sort=False) if (not dual.empty or not major.empty) else pd.DataFrame(); stage.update(1)

    sweeps=build_causal_sweep_events_for_levels(bars,levels)
    intraday_cfg=EntryExpansionConfig(intraday_pivot_left=1,intraday_pivot_right=1)
    intraday_catalog=build_intraday_15m_swing_catalog(bars,days,pm,config=intraday_cfg)
    intraday_sweeps=build_intraday_15m_sweep_events(bars,intraday_catalog,config=intraday_cfg)
    if not intraday_sweeps.empty:
        intraday_sweeps=intraday_sweeps.copy(); intraday_sweeps["setup_eligible_at_sweep"]=True
    all_sweeps=pd.concat([x for x in (sweeps,intraday_sweeps) if not x.empty],ignore_index=True,sort=False) if (not sweeps.empty or not intraday_sweeps.empty) else pd.DataFrame(); stage.update(2)

    swing_catalog=build_visible_swing_catalog(bars,days,config=cfg); stage.update(3)
    print(f"[stage 4/12] R13 compact semantic scan sweeps={len(all_sweeps):,} swings={len(swing_catalog):,}",flush=True)
    # Long-history R13 must not materialise R12's full pivot x FVG Cartesian
    # atlas.  The previous implementation produced 7.8M x ~110 object cells
    # on this exact SOXL window and exhausted >6 GiB during DataFrame creation.
    # Build only the predeclared outermost/tiered primary narratives and the
    # small union of FVG rows required by R13's fixed execution selectors.
    primary_parts=[]; fvg_parts=[]; audit_parts=[]
    sweep_groups={str(k):g for k,g in all_sweeps.groupby("ny_date",sort=True)} if not all_sweeps.empty else {}
    swing_groups={str(k):g for k,g in swing_catalog.groupby("ny_date",sort=False)} if not swing_catalog.empty else {}
    stage4_days=ProgressReporter(label="[stage4] compact semantic days",total=max(1,len(sweep_groups)),every=10,enabled=not args.no_progress)
    for day_text,day_sweeps in sweep_groups.items():
        day_swings=swing_groups.get(str(day_text),pd.DataFrame())
        if not day_swings.empty:
            p_day,f_day,a_day=build_r13_primary_break_fvg_compact(bars,day_sweeps,day_swings,config=cfg)
            if not p_day.empty: primary_parts.append(p_day)
            if not f_day.empty: fvg_parts.append(f_day)
            if not a_day.empty: audit_parts.append(a_day)
        stage4_days.update(1)
    stage4_days.close()
    primary=pd.concat(primary_parts,ignore_index=True,sort=False) if primary_parts else pd.DataFrame()
    fvgs_compact=pd.concat(fvg_parts,ignore_index=True,sort=False) if fvg_parts else pd.DataFrame()
    stage4_audit=pd.concat(audit_parts,ignore_index=True,sort=False) if audit_parts else pd.DataFrame()
    if not primary.empty:
        primary=primary.sort_values(["event_id","execution_tf","break_available_time","mss_reference_price"],kind="mergesort").reset_index(drop=True)
        primary["narrative_attempt_sequence_r13"]=primary.groupby(["event_id","execution_tf"],sort=False).cumcount()+1
    stage.update(4)
    if not stage4_audit.empty:
        avoided=int(pd.to_numeric(stage4_audit["r12_wide_fvg_rows_equivalent"],errors="coerce").fillna(0).sum())
        kept=int(pd.to_numeric(stage4_audit["r13_compact_fvg_rows"],errors="coerce").fillna(0).sum())
        print(f"[stage 4/12] compacted wide_fvg_equivalent={avoided:,} -> compact_fvg_rows={kept:,} primary={len(primary):,}",flush=True)

    all_levels_for_state=pd.concat([x for x in (levels,intraday_catalog) if not x.empty],ignore_index=True,sort=False) if (not levels.empty or not intraday_catalog.empty) else pd.DataFrame()
    state_index=build_liquidity_consumption_query_index(bars,all_levels_for_state,config=c_cfg)
    fvgs_state=attach_consumption_state_to_fvg_rows(fvgs_compact,state_index=state_index,config=c_cfg); stage.update(5)

    print(f"[stage 6/12] vectorized FVG selector primary={len(primary):,} compact_fvg={len(fvgs_state):,}", flush=True)
    fvg_entries=consolidate_fvg_entry_choices(primary,fvgs_state)
    print(f"[stage 6/12] selected FVG entries={len(fvg_entries):,}; expanding targets", flush=True)
    limit_variants=expand_target_state_variants(fvg_entries)
    print(f"[stage 6/12] limit variants={len(limit_variants):,}", flush=True); stage.update(6)

    print(f"[stage 7/12] market-next-open choices from primary={len(primary):,}", flush=True)
    market=add_market_next_open_choices(primary)
    market=attach_market_next_open_from_bars(bars,market)
    market_variants=expand_market_target_state_variants(market,state_index=state_index,config=c_cfg)
    print(f"[stage 7/12] market variants={len(market_variants):,}", flush=True); stage.update(7)

    print(f"[stage 8/12] replay limit_variants={len(limit_variants):,} market_variants={len(market_variants):,}", flush=True)
    limit_life=_replay_limit_unique(bars,limit_variants,args)
    market_life=_replay_market_unique(bars,market_variants,args)
    life=pd.concat([x for x in (limit_life,market_life) if not x.empty],ignore_index=True,sort=False) if (not limit_life.empty or not market_life.empty) else pd.DataFrame()
    if not life.empty:
        life["period_r13"]=[_period_label(x) for x in pd.to_datetime(life["entry_available_time"])]
    stage.update(8)

    perf=_summary(life,["execution_tf","structure_visibility_tier_r13","entry_model_r13","target_model_r13"])
    period=_summary(life,["execution_tf","structure_visibility_tier_r13","entry_model_r13","target_model_r13","period_r13"])
    target_state=_summary(life,["execution_tf","entry_model_r13","target_model_r13","target_liquidity_state","period_r13"]) if (not life.empty and "target_liquidity_state" in life) else pd.DataFrame()
    source_state=_summary(life,["execution_tf","entry_model_r13","source_liquidity_state","period_r13"]) if (not life.empty and "source_liquidity_state" in life) else pd.DataFrame(); stage.update(9)

    cap=_cap_diagnostic(life); distance=_distance_atlas(life); stage.update(10)
    freq=pd.DataFrame()
    if not all_sweeps.empty:
        q=all_sweeps.copy(); q["ny_date"]=q["ny_date"].astype(str)
        rows=[]
        sessions=max(1,len(days))
        for fam,g in q.groupby("liquidity_family",dropna=False,sort=True):
            rows.append({"liquidity_family":fam,"physical_sweeps":len(g),"sessions":sessions,"sweeps_per_session":len(g)/sessions,"active_days":g["ny_date"].nunique(),"active_day_rate":g["ny_date"].nunique()/sessions})
        freq=pd.DataFrame(rows)
    stage.update(11)

    golden=pd.DataFrame()
    for frame,name in ((all_sweeps,"sweep"),(swing_catalog,"swing"),(primary,"primary_mss"),(fvgs_state,"compact_fvg"),(limit_variants,"limit_variant"),(market_variants,"market_variant"),(life,"trade")):
        if not frame.empty and "ny_date" in frame:
            g=frame.loc[frame["ny_date"].astype(str)==str(args.golden_date)].copy(); g.insert(0,"golden_record_type",name); golden=pd.concat([golden,g],ignore_index=True,sort=False)

    design=f"""# R13 Semantic Consolidation Atlas\n\n- Source: {args.data_source}\n- Window: {args.start_date} -> {args.end_date}\n- 2026-08-05 remains the golden semantic replay.\n- Session liquidity consumption is not binary: continuous penetration/acceptance/reclaim features are exported.\n- Descriptive states: fresh / shallow_probe_equal_like / partial_consumed / accepted_or_deep_consumed.\n- A shallow/partial opposite level is still tested as a conservative target; only the `external_if_not_fully_consumed` variant drops accepted/deep targets.\n- Session levels expire with the day. Persistence into later days must come from independently confirmed swing liquidity.\n- Long-history replay uses outermost newly-broken barrier + causal visibility tiers to keep micro and later stronger MSS attempts without replaying every mathematical pivot.\n- Stage 4 is memory-bounded: the R12-wide pivot x FVG Cartesian atlas is counted for audit but not materialised; only the predeclared R13 primary narratives and FVG selector rows are retained.\n- Swing +/- ${args.entry_buffer:.2f} is diagnostic only, not a gate.\n- FVG models: first train, last pre/on-break, break-middle, closest-to-broken-swing, plus CE variants and close-break next-open market.\n- All FVG/market target and entry choices are predeclared; no PnL-derived selector is used.\n"""
    (out/"00_research_design.md").write_text(design,encoding="utf-8")
    _write(quality,out/"01_data_quality.csv"); _write(levels,out/"02_session_and_major_liquidity_levels.csv"); _write(running,out/"03_late_premarket_running_extremes.csv"); _write(all_sweeps,out/"04_physical_sweep_events.csv"); _write(swing_catalog,out/"05_causal_swing_catalog.csv"); _write(stage4_audit,out/"06_stage4_compaction_audit.csv"); _write(primary,out/"07_r13_primary_mss_narratives.csv"); _write(fvgs_state,out/"08_fvg_train_with_liquidity_state.csv"); _write(limit_variants,out/"09_limit_entry_target_variants.csv"); _write(market_variants,out/"10_market_entry_target_variants.csv"); _write(life,out/"11_trade_lifecycle.csv"); _write(perf,out/"12_entry_target_performance.csv"); _write(period,out/"13_period_validation.csv"); _write(target_state,out/"14_target_consumption_state_performance.csv"); _write(source_state,out/"15_source_consumption_state_performance.csv"); _write(cap,out/"16_swing_0p10_cap_keep_vs_cut_performance.csv"); _write(distance,out/"17_entry_chase_distance_atlas.csv"); _write(freq,out/"18_opportunity_frequency.csv"); _write(golden,out/f"19_golden_replay_{args.golden_date}.csv")
    manifest={"experiment_id":"SOXL_ICT_MSS_R13_SEMANTIC_CONSOLIDATION","data_source":args.data_source,"start_date":args.start_date,"end_date":args.end_date,"valid_sessions":len(days),"physical_sweeps":len(all_sweeps),"r12_wide_breaks_equivalent":int(pd.to_numeric(stage4_audit.get("r12_wide_break_rows_equivalent",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()) if not stage4_audit.empty else 0,"r13_primary_narratives":len(primary),"limit_variants":len(limit_variants),"market_variants":len(market_variants),"round_trip_cost":float(args.round_trip_cost),"protocol":"R13 semantic consolidation; no PnL-derived hard filter"}
    (out/"20_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    stage.update(12); stage.close()
    if not args.skip_review_pack:
        finalize_research_report(out,experiment_id=manifest["experiment_id"],edge_id="SOXL_ICT_SWEEP_MSS_SEMANTIC_CONSOLIDATION",title="SOXL ICT R13 Semantic Consolidation Atlas",print_log=True)
    return {"report_dir":out,"review_pack":out/"gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    bars=make_synthetic_ict_day("2026-06-02")
    with tempfile.TemporaryDirectory(prefix="soxl_r13_") as tmp:
        args.start_date=args.end_date="2026-06-02"; args.out_dir=tmp; args.include_us_equity_holidays=True; args.skip_review_pack=True; args.no_progress=True
        res=run_research(bars,args)
        if not (res["report_dir"]/"16_swing_0p10_cap_keep_vs_cut_performance.csv").exists(): raise AssertionError("missing cap diagnostic")
        if not (res["report_dir"]/"10_market_entry_target_variants.csv").exists(): raise AssertionError("missing market variants")
        life = pd.read_csv(res["report_dir"]/"11_trade_lifecycle.csv", low_memory=False)
        market = life.loc[life["entry_order_type"].astype(str).eq("market_next_open")]
        if not market.empty and not market["filled"].astype(str).str.lower().eq("true").any():
            raise AssertionError("market-next-open bug regression: no market entry filled")
    print("[self-test] PASS",flush=True); return 0


def main(argv: Sequence[str] | None=None)->int:
    args=parse_args(argv)
    if args.self_test: return run_self_test(args)
    return 0 if run_research(_load_1m(args),args) else 1


if __name__=="__main__":
    raise SystemExit(main())
