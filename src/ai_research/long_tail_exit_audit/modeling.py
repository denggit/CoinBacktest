#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen base-model scoring, trade metrics and robustness gates for R03.4.2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_state_calibration.modeling import LongBaseModel, fit_base_long_model
from src.ai_research.state_context_ablation.modeling import AblationPeriodData

from .config import LongTailExitAuditConfig
from .simulator import ScoreTimeline, SimulatedTrade


@dataclass(frozen=True)
class FoldScoreBundle:
    base_model: LongBaseModel
    calibration_score: np.ndarray
    test_score: np.ndarray
    timeline: ScoreTimeline
    feature_schema_hash: str


def feature_schema_hash(columns: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def fit_frozen_base_scores(
    fit: AblationPeriodData,
    calibration: AblationPeriodData,
    test: AblationPeriodData,
    config: LongTailExitAuditConfig,
) -> FoldScoreBundle:
    # The imported fitter uses the same frozen parameters as R03.4.1. Validate
    # those parameters explicitly so a future upstream change cannot silently
    # alter this research.
    from src.ai_research.long_state_calibration.config import LongStateCalibrationConfig

    frozen = LongStateCalibrationConfig(
        train_sample_cap=config.train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )
    model = fit_base_long_model(fit, frozen)
    calibration_score = model.predict(np.asarray(calibration.base_x, dtype=np.float32))
    test_score = model.predict(np.asarray(test.base_x, dtype=np.float32))
    quantiles = sorted(
        set(
            [
                *config.signal_quantiles,
                0.50,
                0.60,
                0.70,
            ]
        )
    )
    thresholds = {float(q): float(np.nanquantile(calibration_score, q)) for q in quantiles}
    timeline = ScoreTimeline(
        decision_times_ns=np.asarray(test.timestamps_ns, dtype=np.int64),
        scores=np.asarray(test_score, dtype=float),
        calibration_thresholds=thresholds,
    )
    return FoldScoreBundle(
        base_model=model,
        calibration_score=calibration_score,
        test_score=test_score,
        timeline=timeline,
        feature_schema_hash=feature_schema_hash(test.base_columns),
    )


def trades_to_frame(
    trades: list[SimulatedTrade],
    *,
    fold_id: str,
) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    frame = pd.DataFrame([trade.to_dict() for trade in trades])
    frame.insert(0, "fold_id", fold_id)
    frame["year"] = pd.to_datetime(frame["entry_time"]).dt.year
    frame["quarter"] = pd.to_datetime(frame["entry_time"]).dt.to_period("Q").astype(str)
    frame["month"] = pd.to_datetime(frame["entry_time"]).dt.to_period("M").astype(str)
    return frame


def profit_factor(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    gains = float(array[array > 0].sum())
    losses = float(-array[array < 0].sum())
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def _maximum_drawdown(returns: np.ndarray) -> tuple[float, float]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = equity / peak - 1.0
    return float(drawdown.min()), float(equity[-1] - 1.0)


def _risk_sized_returns(
    frame: pd.DataFrame,
    *,
    net_column: str,
    config: LongTailExitAuditConfig,
) -> np.ndarray:
    risk = frame["initial_risk_pct"].to_numpy(dtype=float)
    net = frame[net_column].to_numpy(dtype=float)
    notional = np.full(len(frame), 1.0, dtype=float)
    valid = np.isfinite(risk) & (risk > 0)
    notional[valid] = np.minimum(
        config.risk_budget_fraction / risk[valid],
        config.maximum_notional_multiple,
    )
    # The diagnostic time baseline has no structural stop and remains 1x notional.
    return net * notional


def summarize_trades(
    frame: pd.DataFrame,
    *,
    cost_multiplier: float,
    config: LongTailExitAuditConfig,
) -> dict[str, object]:
    if frame.empty:
        return {
            "trades": 0,
            "mean_net_return": np.nan,
            "profit_factor": np.nan,
        }
    cost = config.base_round_trip_cost * float(cost_multiplier)
    work = frame.copy()
    work["net_return"] = work["gross_return"].astype(float) - cost
    net = work["net_return"].to_numpy(dtype=float)
    winners = net[net > 0]
    losers = net[net < 0]
    risk_returns = _risk_sized_returns(work, net_column="net_return", config=config)
    max_drawdown, total_return = _maximum_drawdown(risk_returns)
    top_count = min(10, len(work))
    sorted_net = np.sort(net)[::-1]
    total_profit = float(net[net > 0].sum())
    top10_profit_share = float(sorted_net[:top_count].sum() / total_profit) if total_profit > 0 else np.nan
    without_top = sorted_net[top_count:] if len(sorted_net) > top_count else np.empty(0)
    return {
        "trades": int(len(work)),
        "mean_gross_return": float(work["gross_return"].mean()),
        "median_gross_return": float(work["gross_return"].median()),
        "mean_net_return": float(np.mean(net)),
        "median_net_return": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": profit_factor(net),
        "mean_winner": float(np.mean(winners)) if len(winners) else np.nan,
        "mean_loser": float(np.mean(losers)) if len(losers) else np.nan,
        "payoff_ratio": float(np.mean(winners) / abs(np.mean(losers))) if len(winners) and len(losers) else np.nan,
        "mean_mfe": float(work["mfe"].mean()),
        "mean_mae": float(work["mae"].mean()),
        "mean_realized_r": float(work["realized_r"].mean()) if work["realized_r"].notna().any() else np.nan,
        "median_holding_minutes": float(work["holding_minutes"].median()),
        "mean_holding_minutes": float(work["holding_minutes"].mean()),
        "risk_sized_total_return": total_return,
        "risk_sized_max_drawdown": max_drawdown,
        "top10_profit_share": top10_profit_share,
        "mean_net_without_top10": float(np.mean(without_top)) if len(without_top) else np.nan,
        "safety_cap_share": float(np.mean(work["exit_reason"] == "safety_time_cap")),
    }


def build_summary_tables(
    trades: pd.DataFrame,
    execution_audit: pd.DataFrame,
    config: LongTailExitAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []
    duration_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []

    keys = ["fold_id", "signal_quantile", "recipe", "delay_minutes"]
    for group_values, group in trades.groupby(keys, sort=False):
        common = dict(zip(keys, group_values, strict=True))
        for cost_multiplier in config.cost_stress_multipliers:
            metrics = summarize_trades(group, cost_multiplier=cost_multiplier, config=config)
            summary_rows.append({**common, "cost_multiplier": cost_multiplier, **metrics})
            stress_rows.append({**common, "cost_multiplier": cost_multiplier, **metrics})
        for period_kind, column in (("quarter", "quarter"), ("month", "month")):
            for period, period_group in group.groupby(column, sort=True):
                for period_cost in (1.0, 2.0):
                    metrics = summarize_trades(period_group, cost_multiplier=period_cost, config=config)
                    period_rows.append(
                        {
                            **common,
                            "period_kind": period_kind,
                            "period": period,
                            "cost_multiplier": period_cost,
                            **metrics,
                        }
                    )
        counts = group["exit_reason"].value_counts(dropna=False)
        for reason, count in counts.items():
            exit_rows.append(
                {
                    **common,
                    "exit_reason": reason,
                    "count": int(count),
                    "share": float(count / len(group)),
                    "mean_gross_return": float(group.loc[group["exit_reason"] == reason, "gross_return"].mean()),
                }
            )
        bins = pd.cut(
            group["holding_minutes"],
            bins=[-1, 60, 180, 360, 720, 1440, 2880, np.inf],
            labels=["<=1h", "1-3h", "3-6h", "6-12h", "12-24h", "24-48h", ">48h"],
        )
        duration_counts = bins.value_counts(sort=False)
        for bucket, count in duration_counts.items():
            duration_rows.append({**common, "duration_bucket": str(bucket), "count": int(count), "share": float(count / len(group))})
        one_x = summarize_trades(group, cost_multiplier=1.0, config=config)
        two_x = summarize_trades(group, cost_multiplier=2.0, config=config)
        concentration_rows.append(
            {
                **common,
                "top10_profit_share_1x": one_x["top10_profit_share"],
                "mean_net_without_top10_1x": one_x["mean_net_without_top10"],
                "mean_net_without_top10_2x": two_x["mean_net_without_top10"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not execution_audit.empty:
        summary = summary.merge(execution_audit, on=keys, how="left", validate="many_to_one")
    return (
        summary,
        pd.DataFrame(period_rows),
        pd.DataFrame(exit_rows),
        pd.DataFrame(duration_rows),
        pd.DataFrame(stress_rows),
        pd.DataFrame(concentration_rows),
    )


def select_stable_candidates(
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    concentration: pd.DataFrame,
    config: LongTailExitAuditConfig,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    primary = summary.loc[
        (summary["signal_quantile"] == config.primary_signal_quantile)
        & (summary["delay_minutes"] == 1)
    ].copy()
    rows: list[dict[str, object]] = []
    for recipe, group in primary.groupby("recipe", sort=False):
        if recipe == "fixed_6h_diagnostic":
            continue
        by_fold_cost = group.set_index(["fold_id", "cost_multiplier"])
        expected = {(fold, cost) for fold in ("WF_2024", "WF_2025") for cost in (1.0, 2.0, 3.0)}
        if not expected.issubset(set(by_fold_cost.index)):
            continue
        fold_metrics = {key: by_fold_cost.loc[key] for key in expected}
        positive_1x = all(float(fold_metrics[(fold, 1.0)]["mean_net_return"]) > 0 for fold in ("WF_2024", "WF_2025"))
        positive_2x = all(float(fold_metrics[(fold, 2.0)]["mean_net_return"]) > 0 for fold in ("WF_2024", "WF_2025"))
        positive_3x = all(float(fold_metrics[(fold, 3.0)]["mean_net_return"]) > 0 for fold in ("WF_2024", "WF_2025"))
        pf_2x = all(float(fold_metrics[(fold, 2.0)]["profit_factor"]) >= config.minimum_2x_profit_factor for fold in ("WF_2024", "WF_2025"))
        enough = all(int(fold_metrics[(fold, 1.0)]["trades"]) >= config.minimum_trades_per_year for fold in ("WF_2024", "WF_2025"))
        drawdown = all(abs(float(fold_metrics[(fold, 1.0)]["risk_sized_max_drawdown"])) <= config.maximum_risk_sized_drawdown for fold in ("WF_2024", "WF_2025"))
        safety = all(float(fold_metrics[(fold, 1.0)]["safety_cap_share"]) <= config.maximum_safety_cap_share for fold in ("WF_2024", "WF_2025"))

        conc = concentration.loc[
            (concentration["recipe"] == recipe)
            & (concentration["signal_quantile"] == config.primary_signal_quantile)
            & (concentration["delay_minutes"] == 1)
        ]
        top10_ok = (
            len(conc) == 2
            and bool((conc["top10_profit_share_1x"] <= config.maximum_top10_profit_share).all())
            and bool((conc["mean_net_without_top10_1x"] > 0).all())
            and bool((conc["mean_net_without_top10_2x"] > 0).all())
        )
        relevant_periods = periods.loc[
            (periods["recipe"] == recipe)
            & (periods["signal_quantile"] == config.primary_signal_quantile)
            & (periods["delay_minutes"] == 1)
            & (periods["period_kind"] == "quarter")
            & (periods["cost_multiplier"] == 2.0)
        ]
        positive_quarters = int((relevant_periods["mean_net_return"] > 0).sum())
        delay_group = summary.loc[
            (summary["recipe"] == recipe)
            & (summary["signal_quantile"] == config.primary_signal_quantile)
            & (summary["delay_minutes"] == 3)
            & (summary["cost_multiplier"] == 1.0)
        ]
        delay_ok = len(delay_group) == 2 and bool((delay_group["mean_net_return"] > 0).all())
        baseline = primary.loc[
            (primary["recipe"] == "fixed_6h_diagnostic")
            & (primary["cost_multiplier"] == 2.0)
        ].set_index("fold_id")
        retention = True
        retention_values: list[float] = []
        for fold in ("WF_2024", "WF_2025"):
            candidate_expectancy = float(fold_metrics[(fold, 2.0)]["mean_net_return"])
            baseline_expectancy = float(baseline.loc[fold, "mean_net_return"]) if fold in baseline.index else np.nan
            ratio = candidate_expectancy / baseline_expectancy if np.isfinite(baseline_expectancy) and baseline_expectancy > 0 else np.nan
            retention_values.append(ratio)
            if not np.isfinite(ratio) or ratio < config.minimum_2x_expectancy_retention_vs_fixed6h:
                retention = False
        passes = bool(
            positive_1x
            and positive_2x
            and pf_2x
            and enough
            and drawdown
            and safety
            and top10_ok
            and positive_quarters >= config.minimum_positive_quarters
            and delay_ok
            and retention
        )
        min_2x_expectancy = min(float(fold_metrics[(fold, 2.0)]["mean_net_return"]) for fold in ("WF_2024", "WF_2025"))
        min_2x_pf = min(float(fold_metrics[(fold, 2.0)]["profit_factor"]) for fold in ("WF_2024", "WF_2025"))
        rows.append(
            {
                "recipe": recipe,
                "passes_positive_expectancy_gate": passes,
                "positive_1x_both_years": positive_1x,
                "positive_2x_both_years": positive_2x,
                "positive_3x_both_years": positive_3x,
                "pf_2x_gate": pf_2x,
                "minimum_trade_count_gate": enough,
                "drawdown_gate": drawdown,
                "safety_cap_gate": safety,
                "top10_robustness_gate": top10_ok,
                "delay_3m_gate": delay_ok,
                "expectancy_retention_gate": retention,
                "minimum_2x_expectancy_retention_vs_fixed6h": float(np.nanmin(retention_values)),
                "positive_quarters": positive_quarters,
                "minimum_2x_expectancy": min_2x_expectancy,
                "minimum_2x_profit_factor": min_2x_pf,
                "stability_score": min_2x_expectancy * 10_000.0 + min_2x_pf + positive_quarters / 10.0,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            ["passes_positive_expectancy_gate", "stability_score"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)
    return output
