#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R14 completed-trend liquidity acceptance/continuation helpers.

R14 is a strategic pivot from failed universal reversal entries.  It asks
whether a completed-trend liquidity sweep that remains accepted outside the
consumed region can continue in the sweep direction to the already-frozen
deeper same-side completed-trend liquidity.  Full reclaim of the swept region
is the structural invalidation.

Every persistence signal is computed from completed post-root bars and enters
at the next eligible 1m open.  Same-bar target/stop ambiguity is stop-first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R14Config:
    research_start: pd.Timestamp = pd.Timestamp("2023-01-01 00:00:00")
    discovery_end: pd.Timestamp = pd.Timestamp("2024-12-31 23:59:59")
    validation_end: pd.Timestamp = pd.Timestamp("2025-06-30 23:59:59")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    acceptance_windows_minutes: tuple[int, ...] = (5, 15)
    persistence_shares: tuple[float, ...] = (0.60, 0.80, 1.00)
    stop_buffer_bps: float = 2.0
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R14Config":
        if not (self.research_start <= self.discovery_end < self.validation_end < self.holdout_start):
            raise ValueError("research split timestamps must be strictly ordered")
        if not self.acceptance_windows_minutes or any(int(x) <= 0 for x in self.acceptance_windows_minutes):
            raise ValueError("acceptance windows must be positive")
        if not self.persistence_shares or any(not 0 < float(x) <= 1 for x in self.persistence_shares):
            raise ValueError("persistence shares must be in (0, 1]")
        if tuple(sorted(set(self.persistence_shares))) != tuple(self.persistence_shares):
            raise ValueError("persistence shares must be unique and increasing")
        if self.stop_buffer_bps < 0 or self.market_roundtrip_cost < 0:
            raise ValueError("cost and buffer must be non-negative")
        if any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("cost scales must be positive")
        return self


TIME_COLUMNS = (
    "root_sweep_time",
    "root_sweep_available_time",
    "path_start_time",
    "next_open_time",
    "deeper_same_side_touch_time",
    "opposite_1_touch_time",
)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def prepare_continuation_universe(
    r12_paths: pd.DataFrame,
    *,
    config: R14Config | None = None,
    include_holdout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the deeper-same-side continuation universe and holdout audit."""
    cfg = (config or R14Config()).validate()
    q = r12_paths.copy()
    for col in TIME_COLUMNS:
        if col in q:
            q[col] = pd.to_datetime(q[col], errors="coerce")
    q = q.dropna(subset=[
        "root_event_id", "root_sweep_time", "root_sweep_available_time",
        "root_side", "root_zone_low", "root_zone_high",
        "deeper_same_side_touch_price", "path_horizon_minutes",
    ])
    q = q.loc[q["root_side"].isin(["SSL", "BSL"])]
    q = q.loc[_num(q, "same_bar_two_sided_root_flag").fillna(0).eq(0)]
    q = q.loc[_num(q, "deeper_same_side_available_flag").fillna(0).eq(1)]
    target = _num(q, "deeper_same_side_touch_price")
    valid_target = np.where(
        q["root_side"].eq("BSL"),
        target.gt(_num(q, "root_zone_high")),
        target.lt(_num(q, "root_zone_low")),
    )
    q = q.loc[valid_target].copy()
    q["continuation_direction"] = np.where(q["root_side"].eq("BSL"), "long", "short")
    q["same_side_first_label"] = _num(q, "same_side_first_flag").fillna(0).astype(int)
    q["year"] = q["root_sweep_time"].dt.year.astype(int)
    q["research_split"] = np.select(
        [
            q["root_sweep_time"].le(cfg.discovery_end),
            q["root_sweep_time"].le(cfg.validation_end),
            q["root_sweep_time"].lt(cfg.holdout_start),
        ],
        ["discovery", "validation", "embargo"],
        default="sealed_holdout",
    )
    holdout_rows = int(q["research_split"].eq("sealed_holdout").sum())
    seal = pd.DataFrame([{
        "holdout_start": cfg.holdout_start,
        "available_holdout_rows_in_r12": holdout_rows,
        "included_in_r14_outputs": holdout_rows if include_holdout else 0,
        "status": "UNSEALED_EXPLICITLY" if include_holdout else "SEALED_FROM_R14_RULE_DISCOVERY",
        "qualification": "not pristine relative to R12 aggregate path atlas; untouched for R14 acceptance/rule selection",
    }])
    allowed = ["discovery", "validation"] + (["sealed_holdout"] if include_holdout else [])
    q = q.loc[q["research_split"].isin(allowed)]
    return q.sort_values(["root_sweep_time", "root_side"], kind="stable").reset_index(drop=True), seal


def _outside(close: np.ndarray, *, side: str, zone_low: float, zone_high: float) -> np.ndarray:
    if side == "BSL":
        return close > zone_high + EPS
    return close < zone_low - EPS


def _model_name(window: int, share: float) -> str:
    return f"accept_{int(window)}m_p{int(round(float(share) * 100)):03d}"


def attach_acceptance_features(
    bars_1m: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    config: R14Config | None = None,
) -> pd.DataFrame:
    """Attach root/5m/15m acceptance using only bars closed by availability."""
    cfg = (config or R14Config()).validate()
    if universe.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    idx = b.index
    close = _num(b, "close").to_numpy(float)
    rows: list[dict[str, object]] = []
    for r in universe.itertuples(index=False):
        row = r._asdict()
        root_time = pd.Timestamp(r.root_sweep_time)
        pos = int(idx.searchsorted(root_time, side="left"))
        if pos >= len(b) or idx[pos] != root_time:
            continue
        side = str(r.root_side)
        zone_low = float(r.root_zone_low)
        zone_high = float(r.root_zone_high)
        root_outside = bool(_outside(np.asarray([close[pos]]), side=side, zone_low=zone_low, zone_high=zone_high)[0])
        row["root_close_outside_flag"] = int(root_outside)
        row["root_acceptance_available_time"] = idx[pos] + pd.Timedelta(minutes=1)
        for window in cfg.acceptance_windows_minutes:
            end = pos + int(window)
            prefix = f"accept_{int(window)}m"
            row[f"{prefix}_available_time"] = idx[end] + pd.Timedelta(minutes=1) if end < len(b) else pd.NaT
            if end >= len(b):
                row[f"{prefix}_outside_close_share"] = np.nan
                row[f"{prefix}_final_outside_flag"] = 0
                for share in cfg.persistence_shares:
                    row[f"{_model_name(window, share)}_signal"] = 0
                continue
            post_close = close[pos + 1 : end + 1]
            outside = _outside(post_close, side=side, zone_low=zone_low, zone_high=zone_high)
            outside_share = float(np.mean(outside))
            final_outside = bool(outside[-1])
            row[f"{prefix}_outside_close_share"] = outside_share
            row[f"{prefix}_final_outside_flag"] = int(final_outside)
            for share in cfg.persistence_shares:
                row[f"{_model_name(window, share)}_signal"] = int(final_outside and outside_share + EPS >= float(share))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["root_sweep_time", "root_side"], kind="stable").reset_index(drop=True)


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
    if start > end:
        return "censored", -1
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


def _excursions(high: np.ndarray, low: np.ndarray, entry: float, direction: int) -> tuple[float, float]:
    if not len(high) or entry <= EPS:
        return np.nan, np.nan
    if direction == 1:
        return float(np.nanmax(high) / entry - 1.0), float(max(0.0, 1.0 - np.nanmin(low) / entry))
    return float(entry / np.nanmin(low) - 1.0), float(max(0.0, np.nanmax(high) / entry - 1.0))


def _entry_result(
    bars: pd.DataFrame,
    *,
    direction: int,
    entry_pos: int,
    entry: float,
    target: float,
    stop: float,
    outcome: str,
    exit_pos: int,
    cost: float,
    cost_scales: Sequence[float],
) -> dict[str, object]:
    target_ret = target / entry - 1.0 if direction == 1 else entry / target - 1.0
    risk = 1.0 - stop / entry if direction == 1 else stop / entry - 1.0
    if target_ret <= EPS or risk <= EPS:
        return {"entry_status": "invalid_entry_geometry", "outcome": "no_entry", "entry_price": entry}
    gross = target_ret if outcome == "tp_first" else (-risk if outcome == "sl_first" else np.nan)
    end = exit_pos if exit_pos >= 0 else entry_pos
    seg = bars.iloc[entry_pos : end + 1]
    mfe, mae = _excursions(_num(seg, "high").to_numpy(float), _num(seg, "low").to_numpy(float), entry, direction)
    rec: dict[str, object] = {
        "entry_status": "filled",
        "outcome": outcome,
        "entry_time": bars.index[entry_pos],
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "target_distance_pct": target_ret,
        "risk_distance_pct": risk,
        "structural_rr": target_ret / risk,
        "exit_time": bars.index[exit_pos] if exit_pos >= 0 else pd.NaT,
        "holding_minutes": float(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
        "gross_return": gross,
        "gross_r": gross / risk if pd.notna(gross) else np.nan,
        "mfe_pct": mfe,
        "mae_pct": mae,
    }
    for scale in cost_scales:
        net = gross - float(cost) * float(scale) if pd.notna(gross) else np.nan
        rec[f"net_return_cost{float(scale):g}x"] = net
        rec[f"net_r_cost{float(scale):g}x"] = net / risk if pd.notna(net) else np.nan
    return rec


def build_continuation_entries(
    bars_1m: pd.DataFrame,
    features: pd.DataFrame,
    *,
    config: R14Config | None = None,
) -> pd.DataFrame:
    """Replay a small monotone family of causal acceptance market entries."""
    cfg = (config or R14Config()).validate()
    if features.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    idx = b.index
    high = _num(b, "high").to_numpy(float)
    low = _num(b, "low").to_numpy(float)
    open_ = _num(b, "open").to_numpy(float)
    high_tree = SegmentThresholdIndex(high)
    low_tree = SegmentThresholdIndex(low)
    models: list[tuple[str, str, str]] = [
        ("root_close_outside", "root_acceptance_available_time", "root_close_outside_flag")
    ]
    for window in cfg.acceptance_windows_minutes:
        for share in cfg.persistence_shares:
            name = _model_name(window, share)
            models.append((name, f"accept_{int(window)}m_available_time", f"{name}_signal"))
    rows: list[dict[str, object]] = []
    for r in features.itertuples(index=False):
        base = r._asdict()
        root_pos = int(idx.searchsorted(pd.Timestamp(r.root_sweep_time), side="left"))
        first_path = root_pos + 1
        horizon_end = min(len(b) - 1, root_pos + int(r.path_horizon_minutes))
        direction = 1 if str(r.root_side) == "BSL" else -1
        target = float(r.deeper_same_side_touch_price)
        buffer = cfg.stop_buffer_bps / 10000.0
        stop = float(r.root_zone_low) * (1.0 - buffer) if direction == 1 else float(r.root_zone_high) * (1.0 + buffer)
        for model, time_col, flag_col in models:
            rec = {k: base.get(k) for k in (
                "root_event_id", "root_sweep_time", "root_side", "research_split", "year",
                "same_side_first_label", "path_outcome", "root_zone_low", "root_zone_high",
                "deeper_same_side_touch_price",
            )}
            signal_time = base.get(time_col)
            rec.update({"entry_model": model, "entry_kind": "market", "signal_available_time": signal_time})
            if int(float(base.get(flag_col, 0) or 0)) != 1 or signal_time is None or pd.isna(signal_time):
                rec.update({"entry_status": "no_causal_signal", "outcome": "no_entry"})
                rows.append(rec)
                continue
            entry_pos = int(idx.searchsorted(pd.Timestamp(signal_time), side="left"))
            if entry_pos < first_path or entry_pos > horizon_end or entry_pos >= len(b):
                rec.update({"entry_status": "no_causal_signal", "outcome": "no_entry"})
                rows.append(rec)
                continue
            stale, _ = _first_barrier(
                high_tree, low_tree, direction=direction, target=target, stop=stop,
                start=first_path, end=entry_pos - 1,
            )
            if stale != "censored":
                rec.update({"entry_status": "barrier_before_entry", "outcome": "stale"})
                rows.append(rec)
                continue
            entry = float(open_[entry_pos])
            result, exit_pos = _first_barrier(
                high_tree, low_tree, direction=direction, target=target, stop=stop,
                start=entry_pos, end=horizon_end,
            )
            rec.update(_entry_result(
                b, direction=direction, entry_pos=entry_pos, entry=entry,
                target=target, stop=stop, outcome=result, exit_pos=exit_pos,
                cost=cfg.market_roundtrip_cost, cost_scales=cfg.cost_scales,
            ))
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["entry_time"] = pd.to_datetime(out.get("entry_time"), errors="coerce")
        out["exit_time"] = pd.to_datetime(out.get("exit_time"), errors="coerce")
        out["signal_available_time"] = pd.to_datetime(out.get("signal_available_time"), errors="coerce")
    return out


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _top_removed_pf(values: pd.Series, n: int) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return _profit_factor(x.iloc[int(n) :]) if len(x) > int(n) else np.nan


def _split_months(split: str, cfg: R14Config) -> pd.PeriodIndex:
    if split == "discovery":
        return pd.period_range(cfg.research_start, cfg.discovery_end, freq="M")
    if split == "validation":
        return pd.period_range(cfg.discovery_end + pd.Timedelta(seconds=1), cfg.validation_end, freq="M")
    return pd.PeriodIndex([], freq="M")


def summarize_continuation_models(
    entries: pd.DataFrame,
    *,
    config: R14Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R14Config()).validate()
    if entries.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, p in entries.groupby(["research_split", "root_side", "entry_model"], sort=True):
        filled = p.loc[p["entry_status"].eq("filled")]
        resolved = filled.loc[filled["outcome"].isin(["tp_first", "sl_first"])].copy()
        months = _split_months(str(key[0]), cfg)
        net2 = pd.to_numeric(resolved.get("net_return_cost2x"), errors="coerce").dropna()
        monthly = pd.Series(0.0, index=months)
        if len(resolved) and len(months):
            observed = resolved.assign(_month=pd.to_datetime(resolved["entry_time"]).dt.to_period("M")).groupby("_month")["net_return_cost2x"].sum()
            monthly.loc[monthly.index.intersection(observed.index)] = observed.reindex(monthly.index.intersection(observed.index))
        times = pd.to_datetime(resolved["entry_time"], errors="coerce").dropna().sort_values()
        rec: dict[str, object] = {
            "research_split": key[0], "root_side": key[1], "entry_model": key[2],
            "opportunities": len(p), "signals": int(p["entry_status"].ne("no_causal_signal").sum()),
            "filled": len(filled), "resolved": len(resolved),
            "trades_per_month": len(resolved) / len(months) if len(months) else np.nan,
            "tp_before_sl_rate": float(resolved["outcome"].eq("tp_first").mean()) if len(resolved) else np.nan,
            "gross_pf": _profit_factor(resolved.get("gross_return")),
            "positive_months": int((monthly > 0).sum()) if len(monthly) else 0,
            "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "longest_entry_gap_days": float(times.diff().max() / pd.Timedelta(days=1)) if len(times) >= 2 else np.nan,
            "median_risk_distance_pct": _num(resolved, "risk_distance_pct").median(),
            "median_structural_rr": _num(resolved, "structural_rr").median(),
            "median_holding_minutes": _num(resolved, "holding_minutes").median(),
            "median_mfe_pct": _num(resolved, "mfe_pct").median(),
            "median_mae_pct": _num(resolved, "mae_pct").median(),
            "net_pf_cost2x_top5_removed": _top_removed_pf(net2, 5),
            "net_pf_cost2x_top10_removed": _top_removed_pf(net2, 10),
        }
        for scale in cfg.cost_scales:
            col = f"net_return_cost{float(scale):g}x"
            vals = pd.to_numeric(resolved.get(col), errors="coerce").dropna()
            rec[f"mean_net_return_cost{float(scale):g}x"] = vals.mean()
            rec[f"net_pf_cost{float(scale):g}x"] = _profit_factor(vals)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_continuation_years(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    q = entries.loc[entries["entry_status"].eq("filled") & entries["outcome"].isin(["tp_first", "sl_first"])]
    rows = []
    for key, p in q.groupby(["root_side", "entry_model", "year"], sort=True):
        net = pd.to_numeric(p["net_return_cost2x"], errors="coerce").dropna()
        rows.append({
            "root_side": key[0], "entry_model": key[1], "year": key[2], "trades": len(p),
            "tp_before_sl_rate": p["outcome"].eq("tp_first").mean(),
            "mean_net_return_cost2x": net.mean(), "net_pf_cost2x": _profit_factor(net),
            "net_pf_cost2x_top5_removed": _top_removed_pf(net, 5),
        })
    return pd.DataFrame(rows)


def summarize_continuation_months(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    q = entries.loc[entries["entry_status"].eq("filled") & entries["outcome"].isin(["tp_first", "sl_first"])].copy()
    q["month"] = pd.to_datetime(q["entry_time"]).dt.to_period("M").astype(str)
    rows = []
    for key, p in q.groupby(["research_split", "root_side", "entry_model", "month"], sort=True):
        net = pd.to_numeric(p["net_return_cost2x"], errors="coerce").dropna()
        rows.append({
            "research_split": key[0], "root_side": key[1], "entry_model": key[2], "month": key[3],
            "trades": len(p), "net_return_sum_cost2x": net.sum(), "net_pf_cost2x": _profit_factor(net),
        })
    return pd.DataFrame(rows)


def r14_causal_audit(
    features: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    holdout_start: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not features.empty:
        root_av = pd.to_datetime(features["root_sweep_available_time"], errors="coerce")
        for col in [c for c in features if c.endswith("available_time") and c != "root_sweep_available_time"]:
            available = pd.to_datetime(features[col], errors="coerce")
            rows.append({"check": f"{col}_not_before_root_available", "violations": int((available.notna() & available.lt(root_av)).sum())})
        rows.append({"check": "sealed_holdout_absent_from_features", "violations": int(pd.to_datetime(features["root_sweep_time"]).ge(holdout_start).sum())})
    if not entries.empty:
        filled = entries["entry_status"].eq("filled")
        signal = pd.to_datetime(entries["signal_available_time"], errors="coerce")
        entry = pd.to_datetime(entries["entry_time"], errors="coerce")
        rows.append({"check": "filled_entry_not_before_signal_available", "violations": int((filled & signal.notna() & entry.lt(signal)).sum())})
        rows.append({"check": "single_entry_row_per_model_root", "violations": int(entries.duplicated(["root_event_id", "entry_model"]).sum())})
        risk = pd.to_numeric(entries.get("risk_distance_pct"), errors="coerce")
        target = pd.to_numeric(entries.get("target_distance_pct"), errors="coerce")
        rows.append({"check": "filled_positive_structural_risk", "violations": int((filled & ~risk.gt(0)).sum())})
        rows.append({"check": "filled_positive_target_distance", "violations": int((filled & ~target.gt(0)).sum())})
    return pd.DataFrame(rows)
