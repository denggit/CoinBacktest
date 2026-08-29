#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R16 structural and behavioral stop atlas for frozen SSL acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R16Config:
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    hard_stop_buffer_bps: float = 2.0
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R16Config":
        if self.hard_stop_buffer_bps < 0 or self.market_roundtrip_cost < 0:
            raise ValueError("buffer and cost must be non-negative")
        if any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("cost scales must be positive")
        return self


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def prepare_stop_atlas_universe(r14_entries: pd.DataFrame, r14_features: pd.DataFrame) -> pd.DataFrame:
    """Freeze all pre-holdout SSL root-close entries; stop model is the only variable."""
    if r14_entries.empty or r14_features.empty:
        return pd.DataFrame()
    e = r14_entries.copy()
    for col in ("root_sweep_time", "signal_available_time", "entry_time"):
        e[col] = pd.to_datetime(e[col], errors="coerce")
    e = e.loc[
        e["root_side"].eq("SSL")
        & e["entry_model"].eq("root_close_outside")
        & e["entry_status"].eq("filled")
        & e["research_split"].isin(["discovery", "validation"])
    ].copy()
    cols = ["path_horizon_minutes", "root_bar_high", "root_zone_high", "deeper_same_side_touch_price"]
    add_cols = [c for c in cols if c in r14_features and c not in e.columns]
    f = r14_features.loc[:, ["root_event_id", *add_cols]].drop_duplicates("root_event_id")
    q = e.merge(f, on="root_event_id", how="left", validate="one_to_one")
    q = q.dropna(subset=[
        "root_event_id", "root_sweep_time", "signal_available_time", "entry_time",
        "entry_price", "stop_price", "path_horizon_minutes", "root_bar_high",
        "root_zone_high", "deeper_same_side_touch_price",
    ])
    return q.sort_values("entry_time", kind="stable").reset_index(drop=True)


def _first_short_target_or_stop(
    high_tree: SegmentThresholdIndex,
    low_tree: SegmentThresholdIndex,
    *,
    target: float,
    stop: float,
    start: int,
    end: int,
) -> tuple[str, int]:
    tp = int(low_tree.first_leq(start, end, target))
    sl = int(high_tree.first_geq(start, end, stop))
    if sl >= 0 and (tp < 0 or sl <= tp):
        return "sl_first", sl
    if tp >= 0:
        return "tp_first", tp
    return "censored", -1


def _profit_record(
    *,
    entry: float,
    target: float,
    initial_stop: float,
    exit_price: float,
    outcome: str,
    exit_time: pd.Timestamp | pd.NaT,
    holding_minutes: float,
    cost: float,
    scales: tuple[float, ...],
) -> dict[str, object]:
    target_ret = entry / target - 1.0
    initial_risk = initial_stop / entry - 1.0
    gross = entry / exit_price - 1.0 if pd.notna(exit_price) else np.nan
    rec: dict[str, object] = {
        "outcome": outcome, "exit_time": exit_time, "exit_price": exit_price,
        "holding_minutes": holding_minutes, "target_distance_pct": target_ret,
        "initial_risk_distance_pct": initial_risk,
        "structural_rr": target_ret / initial_risk if initial_risk > EPS else np.nan,
        "gross_return": gross,
        "gross_r": gross / initial_risk if pd.notna(gross) and initial_risk > EPS else np.nan,
    }
    for scale in scales:
        net = gross - cost * float(scale) if pd.notna(gross) else np.nan
        rec[f"net_return_cost{float(scale):g}x"] = net
        rec[f"net_r_cost{float(scale):g}x"] = net / initial_risk if pd.notna(net) and initial_risk > EPS else np.nan
    return rec


def build_stop_model_outcomes(
    bars_1m: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    config: R16Config | None = None,
) -> pd.DataFrame:
    """Compare region touch, root-bar extreme and close-reclaim behavioral stops."""
    cfg = (config or R16Config()).validate()
    if universe.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    idx = b.index
    high = _num(b, "high").to_numpy(float)
    low = _num(b, "low").to_numpy(float)
    close = _num(b, "close").to_numpy(float)
    open_ = _num(b, "open").to_numpy(float)
    high_tree = SegmentThresholdIndex(high)
    low_tree = SegmentThresholdIndex(low)
    rows: list[dict[str, object]] = []
    for r in universe.itertuples(index=False):
        entry_pos = int(idx.searchsorted(pd.Timestamp(r.entry_time), side="left"))
        root_pos = int(idx.searchsorted(pd.Timestamp(r.root_sweep_time), side="left"))
        if entry_pos >= len(b) or idx[entry_pos] != pd.Timestamp(r.entry_time):
            continue
        end = min(len(b) - 1, root_pos + int(r.path_horizon_minutes))
        entry = float(r.entry_price)
        target = float(r.deeper_same_side_touch_price)
        region_stop = float(r.stop_price)
        hard_stop = float(r.root_bar_high) * (1.0 + cfg.hard_stop_buffer_bps / 10000.0)
        base = {k: getattr(r, k) for k in (
            "root_event_id", "root_sweep_time", "research_split", "year",
            "signal_available_time", "entry_time", "entry_price",
        )}
        for model, stop in (("region_edge_touch", region_stop), ("root_bar_extreme_touch", hard_stop)):
            rec = dict(base)
            rec.update({"stop_model": model, "target_price": target, "initial_stop_price": stop})
            if not (target < entry < stop):
                rec.update({"outcome": "invalid_geometry"})
                rows.append(rec)
                continue
            outcome, exit_pos = _first_short_target_or_stop(
                high_tree, low_tree, target=target, stop=stop, start=entry_pos, end=end
            )
            exit_price = target if outcome == "tp_first" else (stop if outcome == "sl_first" else np.nan)
            rec.update(_profit_record(
                entry=entry, target=target, initial_stop=stop, exit_price=exit_price,
                outcome=outcome, exit_time=idx[exit_pos] if exit_pos >= 0 else pd.NaT,
                holding_minutes=float(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
                cost=cfg.market_roundtrip_cost, scales=cfg.cost_scales,
            ))
            rows.append(rec)

        rec = dict(base)
        rec.update({"stop_model": "close_reclaim_plus_extreme", "target_price": target, "initial_stop_price": hard_stop})
        if not (target < entry < hard_stop):
            rec.update({"outcome": "invalid_geometry"})
            rows.append(rec)
            continue
        target_pos = int(low_tree.first_leq(entry_pos, end, target))
        hard_pos = int(high_tree.first_geq(entry_pos, end, hard_stop))
        reclaim_hits = np.flatnonzero(close[entry_pos : end + 1] > float(r.root_zone_high) + EPS)
        reclaim_pos = entry_pos + int(reclaim_hits[0]) if len(reclaim_hits) else -1
        failures = [x for x in (hard_pos, reclaim_pos) if x >= 0]
        first_failure = min(failures) if failures else -1
        # Target must occur on an earlier bar. A target and reclaim/hard stop on
        # the same OHLC bar is pessimistically a failure.
        if target_pos >= 0 and (first_failure < 0 or target_pos < first_failure):
            outcome = "tp_first"
            exit_pos = target_pos
            exit_price = target
        elif first_failure < 0:
            outcome = "censored"
            exit_pos = -1
            exit_price = np.nan
        elif hard_pos >= 0 and hard_pos == first_failure:
            outcome = "hard_stop_first"
            exit_pos = hard_pos
            exit_price = hard_stop
        else:
            next_pos = reclaim_pos + 1
            if next_pos >= len(b):
                outcome = "censored_after_reclaim_signal"
                exit_pos = -1
                exit_price = np.nan
            else:
                outcome = "close_reclaim_exit"
                exit_pos = next_pos
                exit_price = float(open_[next_pos])
        rec.update(_profit_record(
            entry=entry, target=target, initial_stop=hard_stop, exit_price=exit_price,
            outcome=outcome, exit_time=idx[exit_pos] if exit_pos >= 0 else pd.NaT,
            holding_minutes=float(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
            cost=cfg.market_roundtrip_cost, scales=cfg.cost_scales,
        ))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exit_time"] = pd.to_datetime(out.get("exit_time"), errors="coerce")
    return out


def _pf(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gain = float(x[x > 0].sum()); loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _top_removed(values: pd.Series, n: int) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return _pf(x.iloc[int(n) :]) if len(x) > int(n) else np.nan


def summarize_stop_models(paths: pd.DataFrame, *, config: R16Config | None = None) -> pd.DataFrame:
    cfg = (config or R16Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    rows = []
    for key, p in paths.groupby(["research_split", "stop_model"], sort=True):
        resolved = p.loc[p["gross_return"].notna()].copy()
        months_n = 24 if key[0] == "discovery" else 6
        net2 = pd.to_numeric(resolved.get("net_return_cost2x"), errors="coerce").dropna()
        calendar = pd.period_range("2023-01", "2024-12", freq="M") if key[0] == "discovery" else pd.period_range("2025-01", "2025-06", freq="M")
        monthly = pd.Series(0.0, index=calendar)
        observed = resolved.assign(month=pd.to_datetime(resolved["entry_time"]).dt.to_period("M")).groupby("month")["net_return_cost2x"].sum()
        monthly.loc[monthly.index.intersection(observed.index)] = observed.reindex(monthly.index.intersection(observed.index))
        times = pd.to_datetime(p["entry_time"], errors="coerce").dropna().sort_values()
        rec: dict[str, object] = {
            "research_split": key[0], "stop_model": key[1], "trades": len(p),
            "trades_per_month": len(p) / months_n, "resolved": len(resolved),
            "target_rate": p["outcome"].eq("tp_first").mean(),
            "gross_pf": _pf(resolved.get("gross_return")),
            "mean_gross_r": _num(resolved, "gross_r").mean(),
            "positive_month_rate_cost2x": float((monthly > 0).mean()),
            "longest_entry_gap_days": float(times.diff().max() / pd.Timedelta(days=1)) if len(times) >= 2 else np.nan,
            "median_initial_risk_pct": _num(resolved, "initial_risk_distance_pct").median(),
            "median_structural_rr": _num(resolved, "structural_rr").median(),
            "median_holding_minutes": _num(resolved, "holding_minutes").median(),
            "net_pf_cost2x_top5_removed": _top_removed(net2, 5),
            "net_pf_cost2x_top10_removed": _top_removed(net2, 10),
        }
        for scale in cfg.cost_scales:
            net = pd.to_numeric(resolved.get(f"net_return_cost{float(scale):g}x"), errors="coerce").dropna()
            nr = pd.to_numeric(resolved.get(f"net_r_cost{float(scale):g}x"), errors="coerce").dropna()
            rec[f"mean_net_return_cost{float(scale):g}x"] = net.mean()
            rec[f"net_pf_cost{float(scale):g}x"] = _pf(net)
            rec[f"mean_net_r_cost{float(scale):g}x"] = nr.mean()
            rec[f"r_pf_cost{float(scale):g}x"] = _pf(nr)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_stop_years(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    q = paths.loc[paths["gross_return"].notna()]
    rows = []
    for key, p in q.groupby(["stop_model", "year"], sort=True):
        net = pd.to_numeric(p["net_return_cost2x"], errors="coerce").dropna()
        rows.append({
            "stop_model": key[0], "year": key[1], "trades": len(p),
            "target_rate": p["outcome"].eq("tp_first").mean(),
            "mean_net_return_cost2x": net.mean(), "net_pf_cost2x": _pf(net),
            "net_pf_cost2x_top5_removed": _top_removed(net, 5),
        })
    return pd.DataFrame(rows)


def r16_causal_audit(paths: pd.DataFrame, *, holdout_start: pd.Timestamp) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    entry = pd.to_datetime(paths["entry_time"], errors="coerce")
    signal = pd.to_datetime(paths["signal_available_time"], errors="coerce")
    root = pd.to_datetime(paths["root_sweep_time"], errors="coerce")
    target = pd.to_numeric(paths["target_price"], errors="coerce")
    price = pd.to_numeric(paths["entry_price"], errors="coerce")
    stop = pd.to_numeric(paths["initial_stop_price"], errors="coerce")
    return pd.DataFrame([
        {"check": "entry_not_before_signal", "violations": int(entry.lt(signal).sum())},
        {"check": "sealed_holdout_absent", "violations": int(root.ge(holdout_start).sum())},
        {"check": "single_row_per_root_stop_model", "violations": int(paths.duplicated(["root_event_id", "stop_model"]).sum())},
        {"check": "short_target_below_entry", "violations": int((~target.lt(price)).sum())},
        {"check": "initial_stop_above_entry", "violations": int((~stop.gt(price)).sum())},
        {"check": "resolved_exit_not_before_entry", "violations": int((pd.to_datetime(paths["exit_time"], errors="coerce").notna() & pd.to_datetime(paths["exit_time"], errors="coerce").lt(entry)).sum())},
    ])
