#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R20 frozen LF V10B component visible-window falsification helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class R20Config:
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01")
    market_roundtrip_cost: float = 0.0015
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    expected_bar_spacing: pd.Timedelta = pd.Timedelta(hours=4)

    def validate(self) -> "R20Config":
        if not (self.discovery_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("invalid R20 split ordering")
        if self.market_roundtrip_cost <= 0 or not self.cost_scales or any(scale <= 0 for scale in self.cost_scales):
            raise ValueError("invalid R20 cost contract")
        return self


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > EPS else (np.inf if gains > EPS else np.nan)


def _research_split(entry: pd.Timestamp, cfg: R20Config) -> str:
    if entry < cfg.discovery_start:
        return "warmup"
    if entry < cfg.validation_start:
        return "discovery"
    if entry < cfg.embargo_start:
        return "validation"
    if entry < cfg.holdout_start:
        return "embargo"
    return "holdout"


def prepare_r20_trades(raw_trades: pd.DataFrame, *, config: R20Config | None = None) -> pd.DataFrame:
    """Convert zero-cost V10B trades into unlevered, boundary-safe R20 rows."""
    cfg = (config or R20Config()).validate()
    if raw_trades.empty:
        return pd.DataFrame()
    required = {"entry_time", "exit_time", "type", "engine", "avg_entry", "exit", "note"}
    missing = sorted(required.difference(raw_trades.columns))
    if missing:
        raise ValueError(f"R20 raw trade schema missing {missing}")

    work = raw_trades.copy().reset_index(drop=True)
    work["entry_time"] = pd.to_datetime(work["entry_time"], errors="coerce")
    work["exit_time"] = pd.to_datetime(work["exit_time"], errors="coerce")
    work["direction"] = work["type"].astype(str).str.upper().map({"LONG": "Long", "SHORT": "Short"})
    work["trade_direction"] = work["direction"].map({"Long": 1, "Short": -1})
    work["research_split"] = work["entry_time"].map(lambda value: _research_split(pd.Timestamp(value), cfg))
    work["signal_time"] = work["entry_time"] - cfg.expected_bar_spacing
    work["component"] = work["engine"].astype(str) + " / " + work["direction"].astype(str)
    work["trade_id"] = [
        f"R20_{str(engine)}_{str(direction).upper()}_{pd.Timestamp(entry).strftime('%Y%m%dT%H%M%S')}_{ordinal:04d}"
        for ordinal, (engine, direction, entry) in enumerate(
            zip(work["engine"], work["direction"], work["entry_time"], strict=True), start=1
        )
    ]

    avg_entry = _num(work, "avg_entry")
    exit_price = _num(work, "exit")
    direction = _num(work, "trade_direction")
    work["gross_return"] = direction * (exit_price / avg_entry - 1.0)
    for scale in cfg.cost_scales:
        work[f"net_return_cost{int(scale)}x"] = work["gross_return"] - scale * cfg.market_roundtrip_cost

    split_end = work["research_split"].map({"discovery": cfg.validation_start, "validation": cfg.embargo_start})
    work["path_status"] = "excluded_split"
    visible = work["research_split"].isin(["discovery", "validation"])
    complete = visible & work["exit_time"].lt(pd.to_datetime(split_end))
    work.loc[visible, "path_status"] = "boundary_censored"
    work.loc[complete, "path_status"] = "included"
    work.loc[work["note"].astype(str).eq("FORCE_CLOSE_END"), "path_status"] = "right_edge_censored"

    keep = [
        "trade_id", "research_split", "path_status", "component", "engine", "direction", "trade_direction",
        "signal_time", "entry_time", "exit_time", "avg_entry", "exit", "units", "note", "gross_return",
        *[f"net_return_cost{int(scale)}x" for scale in cfg.cost_scales],
    ]
    for column in keep:
        if column not in work:
            work[column] = np.nan
    return work[keep].sort_values(["entry_time", "trade_id"], kind="stable").reset_index(drop=True)


def _month_index(split: str, cfg: R20Config) -> pd.PeriodIndex:
    if split == "discovery":
        return pd.period_range(cfg.discovery_start, cfg.validation_start - pd.Timedelta(seconds=1), freq="M")
    return pd.period_range(cfg.validation_start, cfg.embargo_start - pd.Timedelta(seconds=1), freq="M")


def summarize_r20_components(trades: pd.DataFrame, *, config: R20Config | None = None) -> pd.DataFrame:
    cfg = (config or R20Config()).validate()
    included = trades.loc[trades.get("path_status", pd.Series(dtype=str)).eq("included")].copy()
    if included.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, component, engine, direction), part in included.groupby(
        ["research_split", "component", "engine", "direction"], sort=True
    ):
        gross = _num(part, "gross_return")
        net1 = _num(part, "net_return_cost1x")
        net2 = _num(part, "net_return_cost2x")
        net3 = _num(part, "net_return_cost3x")
        top = net2.sort_values(ascending=False, kind="stable")
        without5 = net2.drop(index=top.head(5).index)
        without10 = net2.drop(index=top.head(10).index)
        months = _month_index(str(split), cfg)
        realized = net2.groupby(pd.to_datetime(part["exit_time"]).dt.to_period("M")).sum().reindex(months, fill_value=0.0)
        entries = pd.to_datetime(part["entry_time"], errors="coerce").sort_values()
        rows.append(
            {
                "research_split": split,
                "component": component,
                "engine": engine,
                "direction": direction,
                "trades": int(len(part)),
                "trades_per_month": float(len(part) / len(months)),
                "win_rate_gross": float(gross.gt(0).mean()),
                "gross_pf": _pf(gross),
                "mean_gross_return": float(gross.mean()),
                "net_pf_cost1x": _pf(net1),
                "mean_net_return_cost1x": float(net1.mean()),
                "net_pf_cost2x": _pf(net2),
                "mean_net_return_cost2x": float(net2.mean()),
                "net_pf_cost3x": _pf(net3),
                "mean_net_return_cost3x": float(net3.mean()),
                "positive_month_rate_cost2x": float(realized.gt(0).mean()),
                "longest_entry_gap_days": float(entries.diff().max() / pd.Timedelta(days=1)) if len(entries) > 1 else np.nan,
                "net_pf_cost2x_top5_removed": _pf(without5),
                "net_sum_cost2x_top5_removed": float(without5.sum()),
                "net_pf_cost2x_top10_removed": _pf(without10),
                "net_sum_cost2x_top10_removed": float(without10.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_r20_years(trades: pd.DataFrame) -> pd.DataFrame:
    included = trades.loc[trades.get("path_status", pd.Series(dtype=str)).eq("included")].copy()
    if included.empty:
        return pd.DataFrame()
    included["year"] = pd.to_datetime(included["exit_time"], errors="coerce").dt.year
    rows: list[dict[str, object]] = []
    for (year, component, direction), part in included.groupby(["year", "component", "direction"], sort=True):
        net2 = _num(part, "net_return_cost2x")
        rows.append(
            {
                "year": int(year),
                "component": component,
                "direction": direction,
                "trades": int(len(part)),
                "net_pf_cost2x": _pf(net2),
                "mean_net_return_cost2x": float(net2.mean()),
                "net_sum_cost2x": float(net2.sum()),
                "net_pf_cost2x_top5_removed": _pf(net2.drop(index=net2.sort_values(ascending=False).head(5).index)),
            }
        )
    return pd.DataFrame(rows)


def build_r20_gate(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for component, part in scorecard.groupby("component", sort=True):
        indexed = part.set_index("research_split")
        reasons: list[str] = []
        for split in ("discovery", "validation"):
            if split not in indexed.index:
                reasons.append(f"missing_{split}")
                continue
            row = indexed.loc[split]
            if int(row["trades"]) < 12:
                reasons.append(f"{split}_sample")
            if float(row["net_pf_cost2x"]) < 1.4:
                reasons.append(f"{split}_pf2x")
            if float(row["mean_net_return_cost2x"]) <= 0:
                reasons.append(f"{split}_expectancy")
            top5 = float(row["net_pf_cost2x_top5_removed"])
            if not np.isfinite(top5) or top5 <= 1.0:
                reasons.append(f"{split}_top5")
        rows.append(
            {
                "component": component,
                "forward_incubation_eligible": int(not reasons),
                "reason": "PASS" if not reasons else "FAIL_" + ",".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def r20_causal_audit(trades: pd.DataFrame, features: pd.DataFrame, *, config: R20Config | None = None) -> pd.DataFrame:
    cfg = (config or R20Config()).validate()
    if trades.empty:
        return pd.DataFrame([{"check": "nonempty_trade_table", "violations": 1, "status": "FAIL"}])
    included = trades.loc[trades["path_status"].eq("included")].copy()
    lookup = features.reindex(pd.to_datetime(trades["signal_time"], errors="coerce"))
    signal = pd.to_numeric(lookup.get("signal"), errors="coerce").reset_index(drop=True)
    selected = lookup.get("selected_engine", pd.Series("", index=lookup.index)).astype(str).reset_index(drop=True)
    expected_direction = _num(trades, "trade_direction").reset_index(drop=True)
    rows = [
        {"check": "unique_trade_id", "violations": int(trades["trade_id"].duplicated().sum())},
        {"check": "valid_direction", "violations": int((~trades["trade_direction"].isin([-1, 1])).sum())},
        {"check": "exit_after_entry", "violations": int((pd.to_datetime(trades["exit_time"]) < pd.to_datetime(trades["entry_time"])).sum())},
        {"check": "next_4h_open_entry", "violations": int(((pd.to_datetime(trades["entry_time"]) - pd.to_datetime(trades["signal_time"])) != cfg.expected_bar_spacing).sum())},
        {"check": "signal_direction_matches", "violations": int(signal.ne(expected_direction).sum())},
        {"check": "selected_engine_matches", "violations": int(selected.ne(trades["engine"].astype(str).reset_index(drop=True)).sum())},
        {"check": "finite_entry_exit", "violations": int((~np.isfinite(_num(trades, "avg_entry")) | ~np.isfinite(_num(trades, "exit"))).sum())},
        {"check": "positive_entry_exit", "violations": int((_num(trades, "avg_entry").le(0) | _num(trades, "exit").le(0)).sum())},
        {"check": "included_exits_before_boundary", "violations": int(((included["research_split"].eq("discovery") & pd.to_datetime(included["exit_time"]).ge(cfg.validation_start)) | (included["research_split"].eq("validation") & pd.to_datetime(included["exit_time"]).ge(cfg.embargo_start))).sum())},
        {"check": "embargo_or_holdout_absent", "violations": int(trades["research_split"].isin(["embargo", "holdout"]).sum())},
        {"check": "gross_return_formula", "violations": int((_num(trades, "gross_return") - _num(trades, "trade_direction") * (_num(trades, "exit") / _num(trades, "avg_entry") - 1.0)).abs().gt(1e-12).sum())},
    ]
    for scale in cfg.cost_scales:
        expected = _num(trades, "gross_return") - scale * cfg.market_roundtrip_cost
        rows.append(
            {
                "check": f"cost{int(scale)}x_formula",
                "violations": int((_num(trades, f"net_return_cost{int(scale)}x") - expected).abs().gt(1e-12).sum()),
            }
        )
    audit = pd.DataFrame(rows)
    audit["status"] = np.where(audit["violations"].eq(0), "PASS", "FAIL")
    return audit

