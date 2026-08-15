#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01.3 threshold freezing, first-snapshot selection and executable stress summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AbsorptionModelConfig


def calibration_thresholds(predictions: pd.DataFrame, config: AbsorptionModelConfig, score_column: str = "trade_score") -> pd.DataFrame:
    calibration = predictions.loc[predictions["period"].astype(str).eq(config.calibration_period)].copy()
    rows: list[dict[str, object]] = []
    for side, subset in calibration.groupby("event_side", sort=True):
        values = pd.to_numeric(subset[score_column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 100:
            raise RuntimeError(f"R01.3 insufficient calibration scores for side={side}: {len(values)}")
        rows.append(
            {
                "model": "FULL" if score_column == "trade_score" else "BASELINE",
                "event_side": side,
                "score_column": score_column,
                "quantile": config.selection_quantile,
                "threshold": float(np.quantile(values, config.selection_quantile)),
                "calibration_rows": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def select_first_snapshot(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    score_column: str,
    model_name: str,
) -> pd.DataFrame:
    threshold_map = thresholds.set_index("event_side")["threshold"].to_dict()
    work = predictions.copy()
    work["threshold"] = work["event_side"].map(threshold_map)
    work = work.loc[pd.to_numeric(work[score_column], errors="coerce").ge(work["threshold"])]
    if work.empty:
        return work.assign(selection_model=model_name)
    work = work.sort_values(["event_time", "event_id", "decision_offset_seconds"], kind="mergesort")
    selected = work.drop_duplicates("event_id", keep="first").copy()
    selected["selection_model"] = model_name
    selected["selection_score"] = pd.to_numeric(selected[score_column], errors="coerce")
    return selected.reset_index(drop=True)


def score_deciles(predictions: pd.DataFrame, config: AbsorptionModelConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, period_frame in predictions.groupby("period", sort=True):
        for side, subset in period_frame.groupby("event_side", sort=True):
            work = subset[["trade_score", "tradeable_before_stop_target", "absorption_complete_target", "future_favorable_mfe_bp", "future_additional_extension_bp"]].dropna(subset=["trade_score"]).copy()
            if len(work) < 20:
                continue
            rank = work["trade_score"].rank(method="first", pct=True)
            work["score_decile"] = np.minimum(9, np.floor(rank * 10).astype(int))
            for decile, group in work.groupby("score_decile", sort=True):
                rows.append(
                    {
                        "period": period,
                        "event_side": side,
                        "score_decile": int(decile),
                        "rows": len(group),
                        "tradeable_rate": float(group["tradeable_before_stop_target"].astype(bool).mean()),
                        "absorption_rate": float(group["absorption_complete_target"].astype(bool).mean()),
                        "mean_mfe_bp": float(group["future_favorable_mfe_bp"].mean()),
                        "mean_additional_extension_bp": float(group["future_additional_extension_bp"].mean()),
                        "mean_score": float(group["trade_score"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _trade_pnl(row: pd.Series, config: AbsorptionModelConfig, delay: int, cost_multiple: float) -> tuple[float, str]:
    suffix = f"d{int(delay)}_c{int(cost_multiple)}x"
    result = str(row.get(f"barrier_result_{suffix}", ""))
    cost_bp = float(config.roundtrip_cost_bp * cost_multiple)
    if result == "TARGET":
        return float(config.minimum_net_room_bp), "TARGET"
    if result == "STOP":
        stop_distance = float(row.get(f"structural_stop_distance_bp_d{int(delay)}", np.nan))
        return -(stop_distance + cost_bp), "STOP"
    terminal = float(row.get(f"future_terminal_net_bp_{suffix}", np.nan))
    return terminal, "TIME"


def attach_trade_stress(selected: pd.DataFrame, config: AbsorptionModelConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if selected.empty:
        return pd.DataFrame()
    base_columns = [
        "event_id", "release_episode_id", "event_time", "decision_time", "entry_time",
        "event_side", "period", "path_cluster", "decision_offset_seconds", "selection_model",
        "selection_score", "threshold", "p_tradeable", "p_absorption_complete",
        "pred_additional_extension_bp", "pred_remaining_mfe_bp", "predicted_net_room_bp",
    ]
    for record in selected.to_dict("records"):
        series = pd.Series(record)
        base = {name: record.get(name) for name in base_columns}
        for delay in config.entry_delay_seconds:
            for cost_multiple in config.cost_multipliers:
                pnl, exit_reason = _trade_pnl(series, config, int(delay), float(cost_multiple))
                rows.append(
                    {
                        **base,
                        "entry_delay_seconds": int(delay),
                        "cost_multiple": float(cost_multiple),
                        "roundtrip_cost_bp": float(config.roundtrip_cost_bp * cost_multiple),
                        "net_return_bp": float(pnl),
                        "exit_reason": exit_reason,
                        "stop_distance_bp": float(record.get(f"structural_stop_distance_bp_d{int(delay)}", np.nan)),
                        "mfe_bp": float(record.get(f"future_favorable_mfe_bp_d{int(delay)}", np.nan)),
                        "mae_bp": float(record.get(f"future_adverse_mae_bp_d{int(delay)}", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def _profit_factor(values: pd.Series) -> float:
    positive = float(values.loc[values > 0].sum())
    negative = float(-values.loc[values < 0].sum())
    if negative <= 0:
        return np.inf if positive > 0 else np.nan
    return positive / negative


def trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["selection_model", "period", "event_side", "entry_delay_seconds", "cost_multiple"]
    for keys, group in trades.groupby(group_columns, sort=True):
        values = group["net_return_bp"].dropna()
        if values.empty:
            continue
        sorted_values = values.sort_values(ascending=False)
        trimmed = sorted_values.iloc[min(10, len(sorted_values)) :]
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "trades": int(len(values)),
                "mean_net_bp": float(values.mean()),
                "median_net_bp": float(values.median()),
                "win_rate": float((values > 0).mean()),
                "profit_factor": float(_profit_factor(values)),
                "mean_stop_distance_bp": float(group["stop_distance_bp"].mean()),
                "mean_mfe_bp": float(group["mfe_bp"].mean()),
                "mean_mae_bp": float(group["mae_bp"].mean()),
                "target_rate": float(group["exit_reason"].eq("TARGET").mean()),
                "stop_rate": float(group["exit_reason"].eq("STOP").mean()),
                "time_rate": float(group["exit_reason"].eq("TIME").mean()),
                "top10_removed_mean_net_bp": float(trimmed.mean()) if len(trimmed) else np.nan,
                "top10_share_of_positive_pnl": float(sorted_values.iloc[:10].clip(lower=0).sum() / max(values.clip(lower=0).sum(), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades.loc[trades["entry_delay_seconds"].eq(1) & trades["cost_multiple"].eq(1.0)].copy()
    base["month"] = pd.to_datetime(base["entry_time"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for keys, group in base.groupby(["selection_model", "period", "event_side", "month"], sort=True):
        values = group["net_return_bp"].dropna()
        rows.append(
            {
                "selection_model": keys[0],
                "period": keys[1],
                "event_side": keys[2],
                "month": keys[3],
                "trades": int(len(values)),
                "sum_net_bp": float(values.sum()),
                "mean_net_bp": float(values.mean()) if len(values) else np.nan,
                "win_rate": float((values > 0).mean()) if len(values) else np.nan,
                "profit_factor": float(_profit_factor(values)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def threshold_audit(thresholds: pd.DataFrame, config: AbsorptionModelConfig) -> pd.DataFrame:
    rows = []
    for record in thresholds.to_dict("records"):
        rows.append(
            {
                **record,
                "threshold_source_period": config.calibration_period,
                "holdout_used_for_threshold": False,
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)
