#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal market-state features and descriptive attribution for R03.4.2.17."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.modeling import profit_factor

from .config import StateGateDiagnosticConfig


def _as_datetime_ns(values: pd.Series | pd.Index) -> pd.Series | pd.DatetimeIndex:
    """Normalize merge keys to timezone-naive datetime64[ns].

    Pandas requires exact datetime resolution equality for merge_asof. Real CSV/DB
    inputs may arrive as datetime64[us] while date_range/resample paths are ns.
    """
    converted = pd.to_datetime(values, errors="raise")
    if isinstance(converted, pd.DatetimeIndex):
        if converted.tz is not None:
            converted = converted.tz_convert("UTC").tz_localize(None)
        return converted.astype("datetime64[ns]")
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return converted.astype("datetime64[ns]")


def _ohlc_resample(minute: pd.DataFrame, rule: str, availability: pd.Timedelta, prefix: str) -> pd.DataFrame:
    bars = minute.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    bars.index = bars.index + availability
    bars.index.name = "available_time"
    bars = bars.rename(columns={column: f"{prefix}_{column}" for column in bars.columns})
    bars[f"{prefix}_source_start"] = bars.index - availability
    return bars


def _true_range(frame: pd.DataFrame, prefix: str) -> pd.Series:
    high = frame[f"{prefix}_high"].astype(float)
    low = frame[f"{prefix}_low"].astype(float)
    close = frame[f"{prefix}_close"].astype(float)
    previous = close.shift(1)
    return pd.concat([(high - low), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)


def _timeframe_features(frame: pd.DataFrame, prefix: str, *, is_daily: bool) -> pd.DataFrame:
    out = frame.copy()
    close = out[f"{prefix}_close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out[f"{prefix}_ema20"] = ema20
    out[f"{prefix}_ema50"] = ema50
    out[f"{prefix}_close_rel_ema20"] = close / ema20 - 1.0
    out[f"{prefix}_close_rel_ema50"] = close / ema50 - 1.0
    out[f"{prefix}_ema20_rel_ema50"] = ema20 / ema50 - 1.0
    out[f"{prefix}_ema20_slope3"] = ema20.pct_change(3, fill_method=None)
    if is_daily:
        out[f"{prefix}_ret5"] = close.pct_change(5, fill_method=None)
        out[f"{prefix}_ret20"] = close.pct_change(20, fill_method=None)
        out[f"{prefix}_ret60"] = close.pct_change(60, fill_method=None)
        rolling_high = out[f"{prefix}_high"].rolling(90, min_periods=60).max()
        tr = _true_range(out, prefix)
        atr14 = tr.rolling(14, min_periods=10).mean()
        atr60 = tr.rolling(60, min_periods=40).mean()
    else:
        out[f"{prefix}_ret6"] = close.pct_change(6, fill_method=None)
        out[f"{prefix}_ret18"] = close.pct_change(18, fill_method=None)
        out[f"{prefix}_ret42"] = close.pct_change(42, fill_method=None)
        rolling_high = out[f"{prefix}_high"].rolling(540, min_periods=360).max()  # 90 days of 4h bars
        tr = _true_range(out, prefix)
        atr14 = tr.rolling(14, min_periods=10).mean()
        atr60 = tr.rolling(60, min_periods=40).mean()
    out[f"{prefix}_drawdown_90d"] = close / rolling_high - 1.0
    out[f"{prefix}_atr14_pct"] = atr14 / close
    out[f"{prefix}_atr60_pct"] = atr60 / close
    out[f"{prefix}_vol_ratio"] = out[f"{prefix}_atr14_pct"] / out[f"{prefix}_atr60_pct"]
    return out


def classify_trend(close_rel_ema20: pd.Series, ema20_rel_ema50: pd.Series, ema20_slope3: pd.Series) -> pd.Series:
    up = (close_rel_ema20 > 0) & (ema20_rel_ema50 > 0) & (ema20_slope3 > 0)
    down = (close_rel_ema20 < 0) & (ema20_rel_ema50 < 0) & (ema20_slope3 < 0)
    values = np.select([up, down], ["UP", "DOWN"], default="MIXED")
    return pd.Series(values, index=close_rel_ema20.index, dtype="object")


def classify_combined_state(trend_1d: pd.Series, trend_4h: pd.Series) -> pd.Series:
    conditions = [
        trend_1d.eq("UP") & trend_4h.eq("UP"),
        ~trend_1d.eq("DOWN") & trend_4h.eq("UP"),
        trend_1d.eq("DOWN") & trend_4h.eq("DOWN"),
        ~trend_1d.eq("UP") & trend_4h.eq("DOWN"),
    ]
    choices = ["BULL_ALIGNED", "BULL_TACTICAL", "BEAR_ALIGNED", "BEAR_TACTICAL"]
    return pd.Series(np.select(conditions, choices, default="MIXED"), index=trend_1d.index, dtype="object")


def build_state_timeline(minute: pd.DataFrame, config: StateGateDiagnosticConfig) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(minute.columns))
    if missing:
        raise RuntimeError(f"state path missing OHLC: {missing}")
    work = minute.loc[:, ["open", "high", "low", "close"]].copy().sort_index()
    work.index = _as_datetime_ns(work.index)
    tf4h = _timeframe_features(_ohlc_resample(work, "4h", pd.Timedelta(hours=4), "tf4h"), "tf4h", is_daily=False)
    tf1d = _timeframe_features(_ohlc_resample(work, "1D", pd.Timedelta(days=1), "tf1d"), "tf1d", is_daily=True)
    grid = pd.DataFrame(
        {"decision_time": pd.date_range(
            pd.Timestamp(config.analysis_start),
            pd.Timestamp(config.analysis_end).floor("15min"),
            freq=f"{config.decision_interval_minutes}min",
        )}
    )
    grid["decision_time"] = _as_datetime_ns(grid["decision_time"])
    for source in (tf4h.reset_index(), tf1d.reset_index()):
        source["available_time"] = _as_datetime_ns(source["available_time"])
        grid = pd.merge_asof(
            grid.sort_values("decision_time"),
            source.sort_values("available_time"),
            left_on="decision_time",
            right_on="available_time",
            direction="backward",
        )
        if "available_time" in grid.columns:
            prefix = "tf4h" if "tf4h_close" in source.columns else "tf1d"
            grid = grid.rename(columns={"available_time": f"{prefix}_available_time"})
    grid["trend_4h"] = classify_trend(grid["tf4h_close_rel_ema20"], grid["tf4h_ema20_rel_ema50"], grid["tf4h_ema20_slope3"])
    grid["trend_1d"] = classify_trend(grid["tf1d_close_rel_ema20"], grid["tf1d_ema20_rel_ema50"], grid["tf1d_ema20_slope3"])
    grid["combined_state"] = classify_combined_state(grid["trend_1d"], grid["trend_4h"])
    drawdown = grid["tf1d_drawdown_90d"].astype(float)
    grid["drawdown_state"] = np.select(
        [drawdown >= config.near_90d_high_drawdown, drawdown <= config.deep_90d_drawdown],
        ["NEAR_90D_HIGH", "DEEP_DRAWDOWN"],
        default="CORRECTION",
    )
    vol_ratio = grid["tf4h_vol_ratio"].astype(float)
    grid["vol_state"] = np.select(
        [vol_ratio >= config.high_vol_ratio, vol_ratio <= config.low_vol_ratio],
        ["VOL_EXPANDING", "VOL_COMPRESSED"],
        default="VOL_NORMAL",
    )
    grid["above_1d_ema50"] = grid["tf1d_close_rel_ema50"].astype(float) > 0
    grid["state_ready"] = grid[["tf4h_available_time", "tf1d_available_time"]].notna().all(axis=1)
    grid["context_available_time_flag"] = (
        (pd.to_datetime(grid["tf4h_available_time"]) <= pd.to_datetime(grid["decision_time"]))
        & (pd.to_datetime(grid["tf1d_available_time"]) <= pd.to_datetime(grid["decision_time"]))
    )
    return grid


def align_state(frame: pd.DataFrame, state: pd.DataFrame, *, time_column: str = "decision_time") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work[time_column] = _as_datetime_ns(work[time_column])
    lookup = state.copy()
    lookup["decision_time"] = _as_datetime_ns(lookup["decision_time"])
    return pd.merge_asof(
        work.sort_values(time_column),
        lookup.sort_values("decision_time"),
        left_on=time_column,
        right_on="decision_time",
        direction="backward",
        suffixes=("", "_state"),
    )


def _closed_cycle_metrics(values: Iterable[float]) -> dict[str, float]:
    returns = np.asarray(list(values), dtype=float)
    returns = returns[np.isfinite(returns)]
    if not len(returns):
        return {"trades": 0, "total_return": np.nan, "closed_cycle_mdd": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "mean_return": np.nan}
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = equity / peak - 1.0
    return {
        "trades": int(len(returns)),
        "total_return": float(equity[-1] - 1.0),
        "closed_cycle_mdd": float(drawdown.min()),
        "win_rate": float(np.mean(returns > 0)),
        "profit_factor": float(profit_factor(returns)),
        "mean_return": float(np.mean(returns)),
    }


def summarize_c2_by_state(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if cycles.empty:
        return pd.DataFrame()
    dimensions = ("combined_state", "trend_1d", "trend_4h", "drawdown_state", "vol_state")
    for period, period_frame in cycles.groupby("analysis_period", sort=False):
        for dimension in dimensions:
            for value, group in period_frame.groupby(dimension, dropna=False, sort=False):
                metrics = _closed_cycle_metrics(group["cycle_return"].astype(float))
                rows.append({
                    "analysis_period": period,
                    "state_dimension": dimension,
                    "state_value": value,
                    **metrics,
                    "pnl_sum": float(group["cycle_return"].sum()),
                    "hard_stop_share": float(group["hard_stop_exit"].astype(bool).mean()),
                    "soft_failure_share": float(group["soft_failure_exit"].astype(bool).mean()),
                    "mean_cycle_mae": float(
                        pd.to_numeric(group.get("full_mae", pd.Series(index=group.index, dtype=float)), errors="coerce").mean()
                    ),
                    "mean_cycle_account_drawdown": float(
                        pd.to_numeric(group.get("cycle_max_drawdown", pd.Series(index=group.index, dtype=float)), errors="coerce").mean()
                    ),
                })
    return pd.DataFrame(rows)


def summarize_fixed6h_by_state(trades: pd.DataFrame, config: StateGateDiagnosticConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["net_return"] = work["gross_return"].astype(float) - config.base_round_trip_cost * config.anchor_cost_multiplier
    for period, period_frame in work.groupby("analysis_period", sort=False):
        for value, group in period_frame.groupby("combined_state", dropna=False, sort=False):
            metrics = _closed_cycle_metrics(group["net_return"].astype(float))
            rows.append({
                "analysis_period": period,
                "combined_state": value,
                **metrics,
                "mean_mfe": float(group["mfe"].astype(float).mean()),
                "mean_mae": float(group["mae"].astype(float).mean()),
            })
    return pd.DataFrame(rows)


def gate_masks(cycles: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "G0_ALL": pd.Series(True, index=cycles.index),
        "G1_EXCLUDE_BEAR_ALIGNED": ~cycles["combined_state"].eq("BEAR_ALIGNED"),
        "G2_4H_UP_ONLY": cycles["trend_4h"].eq("UP"),
        "G3_NOT_1D_DOWN": ~cycles["trend_1d"].eq("DOWN"),
        "G4_BULL_CONTEXT_ONLY": cycles["combined_state"].isin(["BULL_ALIGNED", "BULL_TACTICAL"]),
        "G5_ABOVE_1D_EMA50": cycles["above_1d_ema50"].astype(bool),
    }


def counterfactual_gate_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if cycles.empty:
        return pd.DataFrame()
    for period, period_frame in cycles.groupby("analysis_period", sort=False):
        for gate, mask in gate_masks(period_frame).items():
            accepted = period_frame.loc[mask]
            metrics = _closed_cycle_metrics(accepted["cycle_return"].astype(float))
            rows.append({
                "analysis_period": period,
                "gate": gate,
                "source_trades": int(len(period_frame)),
                "accepted_trades": int(len(accepted)),
                "coverage": float(len(accepted) / max(len(period_frame), 1)),
                **metrics,
                "interpretation": "descriptive only; designed after 2026 was opened and requires future untouched validation",
            })
    return pd.DataFrame(rows)


def summarize_score_state(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scores.empty:
        return pd.DataFrame()
    work = scores.copy()
    work["above_q70"] = work["score"].astype(float) >= float(threshold)
    period_frames = list(work.groupby("analysis_period", sort=False))
    h1 = work.loc[work["analysis_period"].isin(["2026_Q1", "2026_Q2"])].copy()
    if not h1.empty:
        period_frames.append(("2026_H1", h1))
    for period, period_frame in period_frames:
        for state_value, group in period_frame.groupby("combined_state", dropna=False, sort=False):
            score = group["score"].astype(float)
            rows.append({
                "analysis_period": period,
                "combined_state": state_value,
                "decision_rows": int(len(group)),
                "mean_score": float(score.mean()),
                "median_score": float(score.median()),
                "q70_exceedance_rate": float(group["above_q70"].mean()),
                "q90_score": float(score.quantile(0.90)),
            })
    return pd.DataFrame(rows)


def monthly_market_vs_c2(
    minute: pd.DataFrame,
    cycles: pd.DataFrame,
    scores: pd.DataFrame,
    state: pd.DataFrame,
    threshold: float,
    account_monthly_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = minute["close"].astype(float).resample("ME").last()
    market = close.pct_change(fill_method=None).rename("eth_month_return").reset_index()
    # The DatetimeIndex can carry a loader-specific name (for example ``timestamp``).
    # Never assume reset_index() creates a literal ``index`` column.
    market = market.rename(columns={market.columns[0]: "month_end"})
    market["month"] = pd.to_datetime(market["month_end"]).dt.to_period("M").astype(str)
    cycle_rows: list[dict[str, object]] = []
    for month, group in cycles.groupby(pd.to_datetime(cycles["entry_time"]).dt.to_period("M").astype(str), sort=True):
        metrics = _closed_cycle_metrics(group["cycle_return"].astype(float))
        cycle_rows.append({
            "month": month,
            "c2_entry_trades": metrics["trades"],
            "c2_entry_cohort_return": metrics["total_return"],
            "c2_entry_cohort_pf": metrics["profit_factor"],
        })
    score_rows: list[dict[str, object]] = []
    if not scores.empty:
        temp = scores.copy()
        temp["month"] = pd.to_datetime(temp["decision_time"]).dt.to_period("M").astype(str)
        temp["above_q70"] = temp["score"].astype(float) >= float(threshold)
        for month, group in temp.groupby("month", sort=True):
            score_rows.append({"month": month, "decision_rows": int(len(group)), "q70_exceedance_rate": float(group["above_q70"].mean())})
    state_rows: list[dict[str, object]] = []
    temp_state = state.copy()
    temp_state["month"] = pd.to_datetime(temp_state["decision_time"]).dt.to_period("M").astype(str)
    for month, group in temp_state.groupby("month", sort=True):
        dominant = group["combined_state"].value_counts().index[0] if len(group) else "UNKNOWN"
        state_rows.append({"month": month, "dominant_state": dominant, "bear_state_share": float(group["combined_state"].isin(["BEAR_ALIGNED", "BEAR_TACTICAL"]).mean()), "bull_state_share": float(group["combined_state"].isin(["BULL_ALIGNED", "BULL_TACTICAL"]).mean())})
    out = market.merge(pd.DataFrame(cycle_rows), on="month", how="left")
    if account_monthly_returns is not None and not account_monthly_returns.empty:
        exact = account_monthly_returns.copy()
        exact["month"] = exact["month"].astype(str)
        exact = exact[["month", "c2_account_return"]].drop_duplicates("month", keep="last")
        out = out.merge(exact, on="month", how="left")
    else:
        # Fallback is explicitly labeled as an entry-cohort return, not a calendar account return.
        out["c2_account_return"] = np.nan
    if score_rows:
        out = out.merge(pd.DataFrame(score_rows), on="month", how="left")
    out = out.merge(pd.DataFrame(state_rows), on="month", how="left")
    start_month = pd.Timestamp(cycles["entry_time"].min()).to_period("M") if not cycles.empty else pd.Period("2024-01")
    end_month = pd.Timestamp(cycles["entry_time"].max()).to_period("M") if not cycles.empty else pd.Period("2026-07")
    periods = pd.PeriodIndex(out["month"], freq="M")
    return out.loc[(periods >= start_month) & (periods <= end_month)].reset_index(drop=True)


def build_attribution_findings(
    c2_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    rows: list[dict[str, object]] = []

    def c2(period: str, state: str) -> pd.Series | None:
        hit = c2_summary.loc[(c2_summary["analysis_period"] == period) & (c2_summary["state_dimension"] == "combined_state") & (c2_summary["state_value"] == state)]
        return hit.iloc[0] if len(hit) == 1 else None

    h1_bear = c2("2026_H1", "BEAR_ALIGNED")
    h1_bull = pd.concat([
        c2_summary.loc[(c2_summary["analysis_period"] == "2026_H1") & (c2_summary["state_dimension"] == "combined_state") & c2_summary["state_value"].isin(["BULL_ALIGNED", "BULL_TACTICAL"])]
    ], ignore_index=True)
    h1_bear_mean = float(h1_bear["mean_return"]) if h1_bear is not None else np.nan
    h1_bull_mean = (
        float(np.average(h1_bull["mean_return"], weights=h1_bull["trades"]))
        if not h1_bull.empty and float(h1_bull["trades"].sum()) > 0
        else np.nan
    )
    regime_separation = bool(
        np.isfinite(h1_bear_mean)
        and h1_bear_mean < 0
        and np.isfinite(h1_bull_mean)
        and h1_bull_mean > 0
    )
    rows.append({
        "finding": "h1_regime_separation",
        "supported": regime_separation,
        "detail": (
            f"2026 H1 BEAR_ALIGNED mean={h1_bear_mean:.3%}; "
            f"BULL_ALIGNED/BULL_TACTICAL weighted mean={h1_bull_mean:.3%}"
        ),
    })

    cal = score_summary.loc[score_summary["analysis_period"] == "CAL_Q4_2025"]
    h1 = score_summary.loc[score_summary["analysis_period"] == "2026_H1"]
    common = sorted(set(cal["combined_state"]) & set(h1["combined_state"]))
    drift_deltas = []
    for state in common:
        a = cal.loc[cal["combined_state"] == state]
        b = h1.loc[h1["combined_state"] == state]
        if len(a) == 1 and len(b) == 1:
            drift_deltas.append(float(b.iloc[0]["q70_exceedance_rate"] - a.iloc[0]["q70_exceedance_rate"]))
    drift_broad = bool(drift_deltas and np.nanmedian(drift_deltas) >= 0.15)
    rows.append({"finding": "broad_conditional_score_drift", "supported": drift_broad, "detail": f"median H1-minus-calibration q70 exceedance delta={np.nanmedian(drift_deltas) if drift_deltas else np.nan:.3f}"})

    july_fixed = fixed_summary.loc[fixed_summary["analysis_period"] == "2026_JULY"]
    july_gate = gate_summary.loc[(gate_summary["analysis_period"] == "2026_JULY") & (gate_summary["gate"] == "G0_ALL")]
    exit_overlay_dependence = bool(
        not july_fixed.empty
        and float(np.average(july_fixed["mean_return"], weights=july_fixed["trades"])) <= 0
        and len(july_gate) == 1
        and float(july_gate.iloc[0]["total_return"]) > 0
    )
    rows.append({"finding": "july_exit_overlay_dependence", "supported": exit_overlay_dependence, "detail": "July C2 is positive while fixed-6h entry expectancy is non-positive"})

    required_periods = {"2024", "2025", "2026_H1", "2026_JULY"}
    positive_gate_name: str | None = None
    uplift_gate_name: str | None = None
    if not gate_summary.empty:
        baseline = gate_summary.loc[gate_summary["gate"] == "G0_ALL", ["analysis_period", "total_return"]].rename(
            columns={"total_return": "baseline_total_return"}
        )
        for gate, group in gate_summary.loc[gate_summary["gate"] != "G0_ALL"].groupby("gate"):
            if not required_periods.issubset(set(group["analysis_period"])):
                continue
            if int((group["total_return"] > 0).sum()) == 4 and float(group["coverage"].min()) >= 0.25:
                positive_gate_name = positive_gate_name or str(gate)
            compared = group.merge(baseline, on="analysis_period", how="inner")
            if len(compared) == 4 and bool((compared["total_return"] > compared["baseline_total_return"]).all()):
                uplift_gate_name = str(gate)
                break
    rows.append({
        "finding": "simple_gate_positive_all_periods",
        "supported": positive_gate_name is not None,
        "detail": (
            f"{positive_gate_name} stays positive in all four opened periods, but positivity alone is not uplift or validation"
            if positive_gate_name
            else "no predeclared gate stays positive in all four opened periods at >=25% minimum coverage"
        ),
    })
    rows.append({
        "finding": "simple_gate_uplift_all_periods",
        "supported": uplift_gate_name is not None,
        "detail": (
            f"{uplift_gate_name} beats G0_ALL in all four opened periods"
            if uplift_gate_name
            else "no predeclared gate beats the frozen G0_ALL return in every opened period"
        ),
    })

    if regime_separation and drift_broad:
        decision = "DIAGNOSIS_REGIME_DEPENDENCE_AND_SCORE_DRIFT"
        reason = "2026 weakness is consistent with both Long-regime dependence and broad score-calibration drift; no simple gate is qualified on already-opened data."
    elif regime_separation:
        decision = "DIAGNOSIS_REGIME_DEPENDENCE_SUPPORTED"
        reason = "C2 performance separates by causal 1D/4H Long regime, but any gate designed here is development-only and needs future untouched validation."
    elif drift_broad:
        decision = "DIAGNOSIS_SCORE_DRIFT_DOMINANT"
        reason = "Score calibration shifts broadly across states, while simple causal Long-state separation is insufficient."
    else:
        decision = "DIAGNOSIS_MIXED_NO_SIMPLE_GATE"
        reason = "The failure cannot be cleanly explained by a simple causal trend gate or broad score drift alone."
    return pd.DataFrame(rows), decision, reason
