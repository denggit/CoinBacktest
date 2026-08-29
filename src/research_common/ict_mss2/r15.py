#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R15 fixed-R first-passage diagnostics for frozen R14 SSL root acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R15Config:
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    r_multiples: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R15Config":
        if not self.r_multiples or any(float(x) <= 0 for x in self.r_multiples):
            raise ValueError("R multiples must be positive")
        if tuple(sorted(set(self.r_multiples))) != tuple(self.r_multiples):
            raise ValueError("R multiples must be unique and increasing")
        if self.market_roundtrip_cost < 0 or any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("costs must be non-negative and scales positive")
        return self


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def prepare_fixed_r_universe(r14_entries: pd.DataFrame, r14_features: pd.DataFrame) -> pd.DataFrame:
    """Freeze the higher-frequency SSL root-close entry; no R15 admission filter."""
    if r14_entries.empty or r14_features.empty:
        return pd.DataFrame()
    e = r14_entries.copy()
    for col in ("root_sweep_time", "signal_available_time", "entry_time", "exit_time"):
        if col in e:
            e[col] = pd.to_datetime(e[col], errors="coerce")
    e = e.loc[
        e["root_side"].eq("SSL")
        & e["entry_model"].eq("root_close_outside")
        & e["entry_status"].eq("filled")
    ].copy()
    feature_cols = ["root_event_id", "path_horizon_minutes"]
    f = r14_features.loc[:, [c for c in feature_cols if c in r14_features]].drop_duplicates("root_event_id")
    q = e.merge(f, on="root_event_id", how="left", validate="one_to_one")
    q = q.dropna(subset=[
        "root_event_id", "root_sweep_time", "signal_available_time", "entry_time",
        "entry_price", "stop_price", "path_horizon_minutes",
    ])
    q = q.loc[q["research_split"].isin(["discovery", "validation"])]
    q = q.loc[_num(q, "stop_price").gt(_num(q, "entry_price"))]
    return q.sort_values("entry_time", kind="stable").reset_index(drop=True)


def _first_barrier(
    high_tree: SegmentThresholdIndex,
    low_tree: SegmentThresholdIndex,
    *,
    direction: int,
    target: float,
    stop: float,
    start: int,
    end: int,
) -> tuple[str, int]:
    if direction == 1:
        tp = int(high_tree.first_geq(start, end, target))
        sl = int(low_tree.first_leq(start, end, stop))
    else:
        tp = int(low_tree.first_leq(start, end, target))
        sl = int(high_tree.first_geq(start, end, stop))
    if sl >= 0 and (tp < 0 or sl <= tp):
        return "sl_first", sl
    if tp >= 0:
        return "tp_first", tp
    return "censored", -1


def build_fixed_r_first_passage(
    bars_1m: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    config: R15Config | None = None,
) -> pd.DataFrame:
    """Replay exact fixed-R target vs the unchanged reclaim stop, stop-first."""
    cfg = (config or R15Config()).validate()
    if universe.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    idx = b.index
    high = _num(b, "high").to_numpy(float)
    low = _num(b, "low").to_numpy(float)
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
        stop = float(r.stop_price)
        direction = -1
        risk_price = stop - entry
        if risk_price <= EPS:
            continue
        for multiple in cfg.r_multiples:
            target = entry - float(multiple) * risk_price
            rec = {k: getattr(r, k) for k in (
                "root_event_id", "root_sweep_time", "research_split", "year",
                "signal_available_time", "entry_time", "entry_price", "stop_price",
            )}
            rec.update({"r_target": float(multiple), "target_price": target})
            if target <= EPS:
                rec.update({"outcome": "invalid_target", "exit_time": pd.NaT})
                rows.append(rec)
                continue
            outcome, exit_pos = _first_barrier(
                high_tree, low_tree, direction=direction, target=target, stop=stop,
                start=entry_pos, end=end,
            )
            risk = risk_price / entry
            gross = float(multiple) * risk if outcome == "tp_first" else (-risk if outcome == "sl_first" else np.nan)
            rec.update({
                "outcome": outcome,
                "exit_time": idx[exit_pos] if exit_pos >= 0 else pd.NaT,
                "holding_minutes": float(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
                "risk_distance_pct": risk,
                "gross_return": gross,
                "gross_r": gross / risk if pd.notna(gross) else np.nan,
            })
            for scale in cfg.cost_scales:
                net = gross - cfg.market_roundtrip_cost * float(scale) if pd.notna(gross) else np.nan
                rec[f"net_return_cost{float(scale):g}x"] = net
                rec[f"net_r_cost{float(scale):g}x"] = net / risk if pd.notna(net) else np.nan
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    return out


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _top_removed(values: pd.Series, n: int) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return _profit_factor(x.iloc[int(n) :]) if len(x) > int(n) else np.nan


def summarize_fixed_r(paths: pd.DataFrame, *, config: R15Config | None = None) -> pd.DataFrame:
    cfg = (config or R15Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    rows = []
    for key, p in paths.groupby(["research_split", "r_target"], sort=True):
        resolved = p.loc[p["outcome"].isin(["tp_first", "sl_first"])].copy()
        months_n = 24 if key[0] == "discovery" else 6
        times = pd.to_datetime(p["entry_time"], errors="coerce").dropna().sort_values()
        net2 = pd.to_numeric(resolved.get("net_return_cost2x"), errors="coerce").dropna()
        monthly = resolved.assign(month=pd.to_datetime(resolved["entry_time"]).dt.to_period("M")).groupby("month")["net_return_cost2x"].sum()
        rec: dict[str, object] = {
            "research_split": key[0], "r_target": key[1], "trades": len(p),
            "trades_per_month": len(p) / months_n,
            "resolved": len(resolved), "censored": int(p["outcome"].eq("censored").sum()),
            "tp_before_sl_rate": resolved["outcome"].eq("tp_first").mean() if len(resolved) else np.nan,
            "gross_pf": _profit_factor(resolved.get("gross_return")),
            "mean_gross_r": _num(resolved, "gross_r").mean(),
            "positive_active_month_rate_cost2x": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "longest_entry_gap_days": float(times.diff().max() / pd.Timedelta(days=1)) if len(times) >= 2 else np.nan,
            "median_risk_distance_pct": _num(resolved, "risk_distance_pct").median(),
            "median_holding_minutes": _num(resolved, "holding_minutes").median(),
            "net_pf_cost2x_top5_removed": _top_removed(net2, 5),
            "net_pf_cost2x_top10_removed": _top_removed(net2, 10),
        }
        for scale in cfg.cost_scales:
            vals = pd.to_numeric(resolved.get(f"net_return_cost{float(scale):g}x"), errors="coerce").dropna()
            nr = pd.to_numeric(resolved.get(f"net_r_cost{float(scale):g}x"), errors="coerce").dropna()
            rec[f"mean_net_return_cost{float(scale):g}x"] = vals.mean()
            rec[f"net_pf_cost{float(scale):g}x"] = _profit_factor(vals)
            rec[f"mean_net_r_cost{float(scale):g}x"] = nr.mean()
            rec[f"r_pf_cost{float(scale):g}x"] = _profit_factor(nr)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_fixed_r_years(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    q = paths.loc[paths["outcome"].isin(["tp_first", "sl_first"])]
    rows = []
    for key, p in q.groupby(["r_target", "year"], sort=True):
        net = pd.to_numeric(p["net_return_cost2x"], errors="coerce").dropna()
        rows.append({
            "r_target": key[0], "year": key[1], "trades": len(p),
            "tp_before_sl_rate": p["outcome"].eq("tp_first").mean(),
            "mean_net_return_cost2x": net.mean(), "net_pf_cost2x": _profit_factor(net),
            "net_pf_cost2x_top5_removed": _top_removed(net, 5),
        })
    return pd.DataFrame(rows)


def r15_causal_audit(paths: pd.DataFrame, *, holdout_start: pd.Timestamp) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    entry = pd.to_datetime(paths["entry_time"], errors="coerce")
    signal = pd.to_datetime(paths["signal_available_time"], errors="coerce")
    root = pd.to_datetime(paths["root_sweep_time"], errors="coerce")
    target = pd.to_numeric(paths["target_price"], errors="coerce")
    price = pd.to_numeric(paths["entry_price"], errors="coerce")
    stop = pd.to_numeric(paths["stop_price"], errors="coerce")
    return pd.DataFrame([
        {"check": "entry_not_before_signal_available", "violations": int(entry.lt(signal).sum())},
        {"check": "entry_after_root_bar", "violations": int(entry.le(root).sum())},
        {"check": "sealed_holdout_absent", "violations": int(root.ge(holdout_start).sum())},
        {"check": "single_row_per_root_r_target", "violations": int(paths.duplicated(["root_event_id", "r_target"]).sum())},
        {"check": "short_target_below_entry", "violations": int((~target.lt(price)).sum())},
        {"check": "short_stop_above_entry", "violations": int((~stop.gt(price)).sum())},
    ])
