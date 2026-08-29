#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-timeframe trade-management research primitives for SOXL ICT.

R10 freezes the entry universe and studies post-fill management only.  It does
not use future pivots before their confirmation time and does not alter MSS/FVG
entry logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .premarket_mss_fvg import NY_TZ, aggregate_closed_bars, compound_account, slice_ny_day
from .premarket_mss_fvg_v2 import confirmed_pivots_with_excursion


@dataclass(frozen=True)
class TradeManagementConfig:
    structure_timeframes: tuple[int, ...] = (1, 2, 5, 15)
    pivot_left: int = 1
    pivot_right: int = 1
    internal_partial_fraction: float = 0.50
    main_target_fraction: float = 0.80
    runner_timeframes: tuple[int, ...] = (2, 5, 15)
    session_end_hour: int = 16
    session_end_minute: int = 30


def _as_ns_index(values: pd.Series | pd.Index) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    return idx.as_unit("ns").asi8


def _build_intermediate_pivots(st: pd.DataFrame) -> pd.DataFrame:
    """Build causal ITH/ITL from triples of same-side short-term pivots.

    A center STH is ITH once a later STH is itself confirmed and both adjacent
    STH prices are lower.  ITL is the mirrored construction.  Availability is
    therefore the later neighbour's confirmation time, never the center time.
    """
    if st.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for side in ("high", "low"):
        g = st.loc[st["pivot_side"].astype(str) == side].sort_values("pivot_time", kind="mergesort").reset_index(drop=True)
        if len(g) < 3:
            continue
        px = pd.to_numeric(g["pivot_price"], errors="coerce").to_numpy(float)
        for i in range(1, len(g) - 1):
            ok = px[i] > px[i - 1] and px[i] > px[i + 1] if side == "high" else px[i] < px[i - 1] and px[i] < px[i + 1]
            if not ok:
                continue
            center = g.iloc[i]
            right = g.iloc[i + 1]
            rows.append({
                "pivot_side": side,
                "pivot_time": pd.Timestamp(center["pivot_time"]),
                "pivot_price": float(center["pivot_price"]),
                "confirmation_available_time": max(pd.Timestamp(center["confirmation_available_time"]), pd.Timestamp(right["confirmation_available_time"])),
                "hierarchy": "IT",
                "source_left_pivot_time": pd.Timestamp(g.iloc[i - 1]["pivot_time"]),
                "source_right_pivot_time": pd.Timestamp(right["pivot_time"]),
            })
    return pd.DataFrame(rows)


def build_management_structure_catalog(day_1m: pd.DataFrame, *, config: TradeManagementConfig = TradeManagementConfig()) -> pd.DataFrame:
    if day_1m.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for tf in config.structure_timeframes:
        frame = aggregate_closed_bars(day_1m, int(tf))
        piv = confirmed_pivots_with_excursion(frame, left=config.pivot_left, right=config.pivot_right)
        if piv.empty:
            continue
        st = piv[["pivot_side", "pivot_time", "pivot_price", "confirmation_available_time"]].copy()
        st["hierarchy"] = "ST"
        st["structure_tf"] = int(tf)
        parts.append(st)
        it = _build_intermediate_pivots(st)
        if not it.empty:
            it["structure_tf"] = int(tf)
            parts.append(it)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["confirmation_available_time"] = pd.to_datetime(out["confirmation_available_time"])
    out["pivot_time"] = pd.to_datetime(out["pivot_time"])
    return out.sort_values(["confirmation_available_time", "structure_tf", "hierarchy"], kind="mergesort").reset_index(drop=True)


def _signed_r(price: float, entry: float, risk_abs: float, is_long: bool) -> float:
    return float((price - entry) / risk_abs if is_long else (entry - price) / risk_abs)


def _signed_return(price: float, entry: float, is_long: bool) -> float:
    return float((price / entry - 1.0) * (1.0 if is_long else -1.0))


def select_known_internal_target(structures: pd.DataFrame, *, fill_time: pd.Timestamp, entry: float, main_target: float, is_long: bool, tf: int | None = None, hierarchy: str | None = None) -> dict[str, object] | None:
    if structures.empty:
        return None
    g = structures.loc[pd.to_datetime(structures["confirmation_available_time"]) <= fill_time].copy()
    if tf is not None:
        g = g.loc[pd.to_numeric(g["structure_tf"], errors="coerce") == int(tf)]
    if hierarchy is not None:
        g = g.loc[g["hierarchy"].astype(str) == hierarchy]
    wanted_side = "high" if is_long else "low"
    g = g.loc[g["pivot_side"].astype(str) == wanted_side].copy()
    px = pd.to_numeric(g["pivot_price"], errors="coerce")
    if is_long:
        g = g.loc[(px > entry) & (px < main_target)]
        if g.empty:
            return None
        g = g.assign(_distance=pd.to_numeric(g["pivot_price"], errors="coerce") - entry)
    else:
        g = g.loc[(px < entry) & (px > main_target)]
        if g.empty:
            return None
        g = g.assign(_distance=entry - pd.to_numeric(g["pivot_price"], errors="coerce"))
    row = g.sort_values(["_distance", "confirmation_available_time"], kind="mergesort").iloc[0]
    return {k: row.get(k) for k in row.index if k != "_distance"}


def _known_latest_trail_level(structures: pd.DataFrame, *, at_time: pd.Timestamp, is_long: bool, tf: int, after_time: pd.Timestamp, current_stop: float) -> tuple[float, pd.Timestamp] | None:
    if structures.empty:
        return None
    wanted = "low" if is_long else "high"
    g = structures.loc[
        (structures["hierarchy"].astype(str) == "ST")
        & (pd.to_numeric(structures["structure_tf"], errors="coerce") == int(tf))
        & (structures["pivot_side"].astype(str) == wanted)
        & (pd.to_datetime(structures["confirmation_available_time"]) <= at_time)
        & (pd.to_datetime(structures["confirmation_available_time"]) > after_time)
    ].copy()
    if g.empty:
        return None
    g = g.sort_values("confirmation_available_time", kind="mergesort")
    for _, row in g.iloc[::-1].iterrows():
        px = float(row["pivot_price"])
        if (is_long and px > current_stop) or ((not is_long) and px < current_stop):
            return px, pd.Timestamp(row["confirmation_available_time"])
    return None


def _scenario_specs(config: TradeManagementConfig) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [{"name": "baseline", "internal": None, "main_fraction": 1.0, "runner_tf": None}]
    for tf in (2, 5, 15):
        specs.append({"name": f"internal_{tf}m_partial50", "internal": (tf, "ST", "fixed"), "main_fraction": 1.0, "runner_tf": None})
    specs.append({"name": "internal_ith_cost_cover", "internal": (None, "IT", "cost_cover"), "main_fraction": 1.0, "runner_tf": None})
    for tf in config.runner_timeframes:
        specs.append({"name": f"main80_runner_{tf}m", "internal": None, "main_fraction": config.main_target_fraction, "runner_tf": int(tf)})
    specs.append({"name": "ith_costcover_main80_runner_5m", "internal": (None, "IT", "cost_cover"), "main_fraction": config.main_target_fraction, "runner_tf": 5})
    return specs


def replay_trade_management(day_1m: pd.DataFrame, trade: dict[str, object], structures: pd.DataFrame, *, round_trip_cost: float, scenario_name: str, config: TradeManagementConfig = TradeManagementConfig()) -> dict[str, object]:
    spec = next(x for x in _scenario_specs(config) if x["name"] == scenario_name)
    result = dict(trade)
    result["management_scenario"] = scenario_name
    if not bool(trade.get("filled", False)):
        return result
    if scenario_name == "baseline":
        gross = float(trade.get("gross_return", np.nan))
        entry0 = float(trade.get("entry_price", np.nan)); stop0b = float(trade.get("stop_price", np.nan))
        risk_pct0 = abs(entry0 - stop0b) / entry0 if np.isfinite(entry0) and entry0 > 0 else np.nan
        net0 = gross - float(round_trip_cost)
        gross_r0 = gross / risk_pct0 if np.isfinite(risk_pct0) and risk_pct0 > 0 else np.nan
        net_r0 = net0 / risk_pct0 if np.isfinite(risk_pct0) and risk_pct0 > 0 else np.nan
        nm0 = float(trade.get("notional_multiple", np.nan))
        ar0 = net0 * nm0 if np.isfinite(nm0) else np.nan
        result.update({
            "management_exit_count": 1,
            "management_exit_ledger": f"1.0000@{float(trade.get('exit_price', np.nan)):.6f}:{trade.get('exit_reason', '')}",
            "management_gross_return": gross,
            "management_net_return": net0,
            "management_gross_r": gross_r0,
            "management_net_r": net_r0,
            "management_account_return": ar0,
            "management_exit_time": trade.get("exit_time"),
            "management_exit_price": float(trade.get("exit_price", np.nan)),
            "management_exit_reason": str(trade.get("exit_reason", "")),
            "management_mfe_r": float(trade.get("mfe_r", np.nan)),
            "management_mae_r": float(trade.get("mae_r", np.nan)),
            "internal_target_available": False,
            "internal_target_price": np.nan,
            "internal_target_r": np.nan,
            "internal_partial_fraction": np.nan,
            "internal_target_tf": np.nan,
            "internal_target_hierarchy": "",
            "main_target_touched": bool(str(trade.get("exit_reason", "")) == "opposite_premarket_extreme_target"),
            "runner_started": False,
        })
        return result
    fill_time = pd.Timestamp(trade["fill_time"])
    entry = float(trade["entry_price"])
    stop0 = float(trade["stop_price"])
    main_target = float(trade["target_price"])
    is_long = str(trade["trade_side"]).upper() == "LONG"
    risk_abs = abs(entry - stop0)
    risk_pct = risk_abs / entry
    if risk_abs <= 0 or risk_pct <= 0:
        raise ValueError("invalid trade risk")
    end = pd.Timestamp(f"{trade['ny_date']} {config.session_end_hour:02d}:{config.session_end_minute:02d}", tz=NY_TZ)
    path = day_1m.loc[(day_1m.index >= fill_time) & (day_1m.index < end)].copy()
    if path.empty:
        return result

    internal = None
    if spec["internal"] is not None:
        tf, hierarchy, mode = spec["internal"]
        internal = select_known_internal_target(structures, fill_time=fill_time, entry=entry, main_target=main_target, is_long=is_long, tf=tf, hierarchy=hierarchy)
        if internal is not None:
            target_r = _signed_r(float(internal["pivot_price"]), entry, risk_abs, is_long)
            cost_r = float(round_trip_cost) / risk_pct
            if mode == "cost_cover":
                fraction = float(np.clip((1.0 + cost_r) / max(target_r + 1.0, 1e-12), 0.0, 1.0))
            else:
                fraction = float(config.internal_partial_fraction)
            internal["partial_fraction"] = fraction
            internal["target_r"] = target_r
            internal["mode"] = mode

    remaining = 1.0
    exits: list[dict[str, object]] = []
    current_stop = stop0
    runner_started = False
    main_touched = False
    internal_done = False
    runner_anchor_time = fill_time
    mfe_r = 0.0
    mae_r = 0.0

    for bar_start, bar in path.iterrows():
        bar_start = pd.Timestamp(bar_start)
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        favourable = _signed_r(high if is_long else low, entry, risk_abs, is_long)
        adverse = _signed_r(low if is_long else high, entry, risk_abs, is_long)
        mfe_r = max(mfe_r, favourable)
        mae_r = min(mae_r, adverse)

        stop_touch = low <= current_stop if is_long else high >= current_stop
        if stop_touch:
            exits.append({"fraction": remaining, "price": current_stop, "time": bar_start + pd.Timedelta(minutes=1), "reason": "structural_trail_stop" if current_stop != stop0 else "initial_structural_stop"})
            remaining = 0.0
            break

        if internal is not None and not internal_done:
            ip = float(internal["pivot_price"])
            touched = high >= ip if is_long else low <= ip
            if touched:
                frac = min(remaining, float(internal["partial_fraction"]))
                if frac > 0:
                    exits.append({"fraction": frac, "price": ip, "time": bar_start + pd.Timedelta(minutes=1), "reason": f"internal_{internal['hierarchy']}_{int(internal['structure_tf'])}m_partial"})
                    remaining -= frac
                internal_done = True
                if remaining <= 1e-12:
                    break

        main_touch = high >= main_target if is_long else low <= main_target
        if main_touch and not main_touched:
            frac = min(remaining, float(spec["main_fraction"]) * remaining)
            exits.append({"fraction": frac, "price": main_target, "time": bar_start + pd.Timedelta(minutes=1), "reason": "opposite_external_liquidity_main_tp"})
            remaining -= frac
            main_touched = True
            if remaining <= 1e-12:
                break
            runner_started = spec["runner_tf"] is not None
            runner_anchor_time = bar_start + pd.Timedelta(minutes=1)

        if runner_started and remaining > 1e-12:
            trail = _known_latest_trail_level(structures, at_time=bar_start, is_long=is_long, tf=int(spec["runner_tf"]), after_time=runner_anchor_time, current_stop=current_stop)
            if trail is not None:
                px, avail = trail
                if (is_long and px < close and px > current_stop) or ((not is_long) and px > close and px < current_stop):
                    current_stop = px
                    result["latest_runner_trail_available_time"] = avail
                    result["latest_runner_trail_price"] = px

    if remaining > 1e-12:
        last_time = pd.Timestamp(path.index[-1]) + pd.Timedelta(minutes=1)
        last_close = float(path["close"].iloc[-1])
        exits.append({"fraction": remaining, "price": last_close, "time": last_time, "reason": "session_close_runner" if runner_started else "session_close"})
        remaining = 0.0

    weighted_gross = float(sum(float(x["fraction"]) * _signed_return(float(x["price"]), entry, is_long) for x in exits))
    net = weighted_gross - float(round_trip_cost)
    gross_r = weighted_gross / risk_pct
    net_r = net / risk_pct
    notional_multiple = float(trade.get("notional_multiple", np.nan))
    account_return = net * notional_multiple if np.isfinite(notional_multiple) else np.nan
    final_exit = exits[-1]
    result.update({
        "management_exit_count": len(exits),
        "management_exit_ledger": "|".join(f"{x['fraction']:.4f}@{x['price']:.6f}:{x['reason']}" for x in exits),
        "management_gross_return": weighted_gross,
        "management_net_return": net,
        "management_gross_r": gross_r,
        "management_net_r": net_r,
        "management_account_return": account_return,
        "management_exit_time": final_exit["time"],
        "management_exit_price": final_exit["price"],
        "management_exit_reason": final_exit["reason"],
        "management_mfe_r": mfe_r,
        "management_mae_r": mae_r,
        "internal_target_available": internal is not None,
        "internal_target_price": float(internal["pivot_price"]) if internal is not None else np.nan,
        "internal_target_r": float(internal["target_r"]) if internal is not None else np.nan,
        "internal_partial_fraction": float(internal["partial_fraction"]) if internal is not None else np.nan,
        "internal_target_tf": int(internal["structure_tf"]) if internal is not None else np.nan,
        "internal_target_hierarchy": str(internal["hierarchy"]) if internal is not None else "",
        "main_target_touched": bool(main_touched),
        "runner_started": bool(runner_started),
    })
    return result


def replay_management_scenarios(bars_ny: pd.DataFrame, base_lifecycle: pd.DataFrame, *, round_trip_cost: float, cost_multipliers: tuple[float, ...] = (1.0,), config: TradeManagementConfig = TradeManagementConfig(), progress=None) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    filled = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    day_cache: dict[str, pd.DataFrame] = {}
    struct_cache: dict[str, pd.DataFrame] = {}
    specs = _scenario_specs(config)
    rows: list[dict[str, object]] = []
    done = 0
    for trade in filled.to_dict("records"):
        day = str(trade["ny_date"])
        if day not in day_cache:
            day_cache[day] = slice_ny_day(bars_ny, pd.Timestamp(day).date(), pd.Timestamp("04:00").time(), pd.Timestamp("16:30").time())
            struct_cache[day] = build_management_structure_catalog(day_cache[day], config=config)
        entry = float(trade.get("entry_price", np.nan)); stop = float(trade.get("stop_price", np.nan))
        risk_pct = abs(entry - stop) / entry if np.isfinite(entry) and entry > 0 else np.nan
        notional_multiple = float(trade.get("notional_multiple", np.nan))
        for spec in specs:
            name = str(spec["name"])
            depends_on_cost = "costcover" in name or "cost_cover" in name
            replay_costs = cost_multipliers if depends_on_cost else (1.0,)
            cached: dict[str, object] | None = None
            for cm in replay_costs:
                row = replay_trade_management(day_cache[day], trade, struct_cache[day], round_trip_cost=round_trip_cost * float(cm), scenario_name=name, config=config)
                row["management_cost_multiple"] = float(cm)
                rows.append(row)
                cached = row
                done += 1
                if progress is not None:
                    progress.update(done)
            if not depends_on_cost and cached is not None:
                gross = float(cached.get("management_gross_return", np.nan))
                for cm in cost_multipliers:
                    if float(cm) == 1.0:
                        continue
                    clone = dict(cached)
                    net = gross - float(round_trip_cost) * float(cm)
                    clone["management_cost_multiple"] = float(cm)
                    clone["management_net_return"] = net
                    clone["management_net_r"] = net / risk_pct if np.isfinite(risk_pct) and risk_pct > 0 else np.nan
                    clone["management_account_return"] = net * notional_multiple if np.isfinite(notional_multiple) else np.nan
                    rows.append(clone)
    return pd.DataFrame(rows)


def _safe_median(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce")
    x = x.loc[np.isfinite(x.to_numpy(float))]
    return float(x.median()) if len(x) else np.nan


def management_summary_table(managed: pd.DataFrame, *, initial_capital: float) -> pd.DataFrame:
    if managed.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, g in managed.groupby(["execution_tf", "liquidity_family", "management_scenario", "management_cost_multiple"], dropna=False, sort=True):
        tf, fam, scenario, cost_multiple = keys
        x = pd.to_numeric(g["management_net_return"], errors="coerce").dropna()
        ar = pd.to_numeric(g["management_account_return"], errors="coerce")
        tmp = g.copy()
        tmp["account_return"] = ar
        account = compound_account(tmp, initial_capital=initial_capital)
        gains = float(x[x > 0].sum()); losses = float(-x[x < 0].sum())
        fill_dt = pd.to_datetime(g["fill_time"])
        months = fill_dt.dt.strftime("%Y-%m")
        monthly = g.assign(_month=months).groupby("_month")["management_account_return"].apply(lambda s: float(np.prod(1.0 + pd.to_numeric(s, errors="coerce").fillna(0.0)) - 1.0))
        rows.append({
            "execution_tf": tf, "liquidity_family": fam, "management_scenario": scenario, "management_cost_multiple": cost_multiple,
            "trades": int(len(g)), "win_rate": float((x > 0).mean()) if len(x) else np.nan,
            "mean_net_return": float(x.mean()) if len(x) else np.nan,
            "mean_net_r": float(pd.to_numeric(g["management_net_r"], errors="coerce").mean()),
            "profit_factor": gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan),
            "final_capital": float(account["capital"].iloc[-1]) if not account.empty else initial_capital,
            "account_total_return": float(account["capital"].iloc[-1] / initial_capital - 1.0) if not account.empty else 0.0,
            "account_max_drawdown": float(pd.to_numeric(account["drawdown"], errors="coerce").min()) if not account.empty else 0.0,
            "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "median_internal_target_r": _safe_median(g["internal_target_r"]),
            "median_internal_partial_fraction": _safe_median(g["internal_partial_fraction"]),
            "main_target_touch_rate": float(g["main_target_touched"].fillna(False).mean()),
            "runner_start_rate": float(g["runner_started"].fillna(False).mean()),
        })
    return pd.DataFrame(rows)


def management_period_table(managed: pd.DataFrame) -> pd.DataFrame:
    if managed.empty:
        return pd.DataFrame()
    out = managed.copy()
    d = pd.to_datetime(out["ny_date"], errors="coerce")
    out["analysis_period"] = np.select([d.dt.year <= 2024, d.dt.year == 2025, d.dt.year >= 2026], ["discovery_through_2024", "forward_2025", "late_holdout_2026"], default="unknown")
    rows=[]
    for keys,g in out.groupby(["execution_tf","liquidity_family","management_scenario","management_cost_multiple","analysis_period"],dropna=False,sort=True):
        x=pd.to_numeric(g["management_net_return"],errors="coerce").dropna(); gains=float(x[x>0].sum()); losses=float(-x[x<0].sum())
        rows.append({"execution_tf":keys[0],"liquidity_family":keys[1],"management_scenario":keys[2],"management_cost_multiple":keys[3],"analysis_period":keys[4],"trades":len(g),"win_rate":float((x>0).mean()) if len(x) else np.nan,"profit_factor":gains/losses if losses>0 else (np.inf if gains>0 else np.nan),"mean_net_return":float(x.mean()) if len(x) else np.nan})
    return pd.DataFrame(rows)


def structure_target_availability_table(managed: pd.DataFrame) -> pd.DataFrame:
    if managed.empty:
        return pd.DataFrame()
    g=managed.loc[managed["management_scenario"].astype(str).str.startswith("internal_")].copy()
    if g.empty: return pd.DataFrame()
    rows=[]
    for keys,x in g.groupby(["execution_tf","liquidity_family","management_scenario","management_cost_multiple"],dropna=False,sort=True):
        rows.append({"execution_tf":keys[0],"liquidity_family":keys[1],"management_scenario":keys[2],"management_cost_multiple":keys[3],"trades":len(x),"target_available_rate":float(x["internal_target_available"].fillna(False).mean()),"median_target_r":_safe_median(x["internal_target_r"]),"median_partial_fraction":_safe_median(x["internal_partial_fraction"])})
    return pd.DataFrame(rows)
