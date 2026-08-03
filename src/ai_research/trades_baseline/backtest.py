#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prediction-to-trade conversion and realistic stress evaluation for R01."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TradesBaselineConfig
from .dataset import feature_columns, load_month_shard
from .modeling import OnlineRegressionMetrics, PredictionMetrics, Regressor, WalkForwardFold, predict_model


@dataclass
class TradeRecord:
    fold_id: str
    model: str
    horizon_seconds: int
    quantile: float
    latency_seconds: float
    cost_multiplier: float
    decision_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    prediction: float
    gross_return: float
    cost_rate: float
    net_return: float
    capital: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ScenarioAccumulator:
    fold_id: str
    model_name: str
    horizon_seconds: int
    quantile: float
    latency_seconds: float
    cost_multiplier: float
    initial_capital: float
    next_free_ns: int = -1
    capital: float = field(init=False)
    peak: float = field(init=False)
    max_drawdown: float = 0.0
    records: list[TradeRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.capital = float(self.initial_capital)
        self.peak = float(self.initial_capital)

    def consume(
        self,
        *,
        decision_ns: int,
        direction: int,
        prediction: float,
        signed_gross_return: float,
        cost_rate: float,
    ) -> None:
        if direction == 0 or not np.isfinite(signed_gross_return):
            return
        if decision_ns < self.next_free_ns:
            return
        latency_seconds = max(1, int(math.ceil(self.latency_seconds)))
        entry_ns = decision_ns + latency_seconds * 1_000_000_000
        exit_ns = entry_ns + self.horizon_seconds * 1_000_000_000
        net = float(signed_gross_return - cost_rate)
        self.capital *= max(1e-9, 1.0 + net)
        self.peak = max(self.peak, self.capital)
        self.max_drawdown = max(self.max_drawdown, (self.peak - self.capital) / self.peak if self.peak > 0 else 0.0)
        self.records.append(
            TradeRecord(
                fold_id=self.fold_id,
                model=self.model_name,
                horizon_seconds=self.horizon_seconds,
                quantile=self.quantile,
                latency_seconds=self.latency_seconds,
                cost_multiplier=self.cost_multiplier,
                decision_time=pd.Timestamp(decision_ns),
                entry_time=pd.Timestamp(entry_ns),
                exit_time=pd.Timestamp(exit_ns),
                direction="long" if direction > 0 else "short",
                prediction=float(prediction),
                gross_return=float(signed_gross_return),
                cost_rate=float(cost_rate),
                net_return=net,
                capital=float(self.capital),
            )
        )
        self.next_free_ns = exit_ns

    def summary(self) -> dict[str, object]:
        returns = np.asarray([record.net_return for record in self.records], dtype=float)
        gross = np.asarray([record.gross_return for record in self.records], dtype=float)
        if len(returns) == 0:
            return {
                "fold_id": self.fold_id,
                "model": self.model_name,
                "horizon_seconds": self.horizon_seconds,
                "quantile": self.quantile,
                "latency_seconds": self.latency_seconds,
                "cost_multiplier": self.cost_multiplier,
                "trades": 0,
                "win_rate": 0.0,
                "mean_net_return": 0.0,
                "median_net_return": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "positive_month_ratio": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "top10_removed_total_return": 0.0,
                "mean_gross_return": 0.0,
            }
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("inf")
        times = pd.DatetimeIndex([record.exit_time for record in self.records])
        monthly = pd.Series(returns, index=times).groupby(times.to_period("M")).sum()
        removed = returns.copy()
        if len(removed) > 10:
            top_idx = np.argpartition(removed, -10)[-10:]
            removed[top_idx] = 0.0
        top10_removed_compound = float(np.prod(1.0 + removed) - 1.0)
        return {
            "fold_id": self.fold_id,
            "model": self.model_name,
            "horizon_seconds": self.horizon_seconds,
            "quantile": self.quantile,
            "latency_seconds": self.latency_seconds,
            "cost_multiplier": self.cost_multiplier,
            "trades": int(len(returns)),
            "win_rate": float((returns > 0).mean()),
            "mean_net_return": float(returns.mean()),
            "median_net_return": float(np.median(returns)),
            "profit_factor": profit_factor,
            "total_return": float(self.capital / self.initial_capital - 1.0),
            "max_drawdown": float(self.max_drawdown),
            "positive_month_ratio": float((monthly > 0).mean()) if len(monthly) else 0.0,
            "long_trades": int(sum(record.direction == "long" for record in self.records)),
            "short_trades": int(sum(record.direction == "short" for record in self.records)),
            "top10_removed_total_return": top10_removed_compound,
            "mean_gross_return": float(gross.mean()),
        }


@dataclass(frozen=True)
class FoldEvaluation:
    prediction_metrics: PredictionMetrics
    scenario_summaries: pd.DataFrame
    base_trades: pd.DataFrame


def evaluate_model_on_fold(
    *,
    model: Regressor,
    model_name: str,
    thresholds: dict[str, float],
    paths: Iterable[Path],
    fold: WalkForwardFold,
    horizon: int,
    config: TradesBaselineConfig,
) -> FoldEvaluation:
    features = feature_columns(config)
    base_target = f"gross_ret_h{horizon}_lat{int(config.base_latency_seconds * 1000)}"
    metrics = OnlineRegressionMetrics()
    scenarios: dict[tuple[float, float, float], ScenarioAccumulator] = {}
    for quantile in config.signal_quantiles:
        for latency in config.latency_scenarios_seconds:
            for cost_mult in config.cost_stress_multipliers:
                key = (quantile, latency, cost_mult)
                scenarios[key] = ScenarioAccumulator(
                    fold_id=fold.fold_id,
                    model_name=model_name,
                    horizon_seconds=horizon,
                    quantile=quantile,
                    latency_seconds=latency,
                    cost_multiplier=cost_mult,
                    initial_capital=config.initial_capital,
                )

    base_cost = config.round_trip_fee_rate + 2.0 * config.slippage_rate_per_side
    for path in paths:
        shard = load_month_shard(path)
        if tuple(features) != shard.feature_names:
            raise RuntimeError(f"feature schema mismatch in {path}")
        label_map = shard.label_index
        if base_target not in label_map:
            raise RuntimeError(f"missing target {base_target} in {path}")
        pos = shard.positions(fold.test_start, fold.test_end)
        x_view = shard.features[pos]
        base_y_view = shard.labels[pos, label_map[base_target]]
        ts_view = shard.timestamps_ns[pos]
        valid = np.isfinite(base_y_view) & np.isfinite(x_view).all(axis=1)
        if not valid.any():
            continue
        x = np.asarray(x_view[valid], dtype=np.float32)
        prediction = predict_model(model, x)
        metrics.update(np.asarray(base_y_view[valid], dtype=float), prediction)
        decision_ns = np.asarray(ts_view[valid], dtype=np.int64)
        valid_positions = np.flatnonzero(valid)
        latency_paths: dict[float, np.ndarray] = {}
        for latency in config.latency_scenarios_seconds:
            target_col = f"gross_ret_h{horizon}_lat{int(round(latency * 1000))}"
            if target_col not in label_map:
                raise RuntimeError(f"missing target {target_col} in {path}")
            latency_paths[latency] = np.asarray(
                shard.labels[pos, label_map[target_col]][valid_positions], dtype=float
            )

        for quantile in config.signal_quantiles:
            long_threshold = float(thresholds[f"q{quantile:.3f}_long"])
            short_threshold = float(thresholds[f"q{quantile:.3f}_short"])
            long_expected = float(thresholds.get(f"q{quantile:.3f}_long_expected_gross", float("nan")))
            short_expected = float(thresholds.get(f"q{quantile:.3f}_short_expected_gross", float("nan")))
            directions = np.zeros(len(prediction), dtype=np.int8)
            if np.isfinite(long_expected) and long_expected > base_cost:
                directions[prediction >= long_threshold] = 1
            if np.isfinite(short_expected) and short_expected > base_cost:
                directions[prediction <= short_threshold] = -1
            signal_positions = np.flatnonzero(directions)
            if len(signal_positions) == 0:
                continue
            for latency in config.latency_scenarios_seconds:
                gross_signed_path = latency_paths[latency]
                for cost_mult in config.cost_stress_multipliers:
                    accumulator = scenarios[(quantile, latency, cost_mult)]
                    cost_rate = base_cost * cost_mult
                    for signal_pos in signal_positions:
                        direction = int(directions[signal_pos])
                        raw_gross = gross_signed_path[signal_pos]
                        signed_gross = raw_gross if direction > 0 else -raw_gross
                        accumulator.consume(
                            decision_ns=int(decision_ns[signal_pos]),
                            direction=direction,
                            prediction=float(prediction[signal_pos]),
                            signed_gross_return=float(signed_gross),
                            cost_rate=float(cost_rate),
                        )

    summaries = pd.DataFrame([acc.summary() for acc in scenarios.values()])
    base_records: list[dict[str, object]] = []
    for quantile in config.signal_quantiles:
        base_acc = scenarios[(quantile, config.base_latency_seconds, 1.0)]
        base_records.extend(record.to_dict() for record in base_acc.records)
    base_trades = pd.DataFrame(base_records)
    return FoldEvaluation(metrics.finalize(), summaries, base_trades)

def select_validation_champion(summaries: pd.DataFrame, config: TradesBaselineConfig) -> dict[str, object] | None:
    """Select a champion using WF_2025 only; WF_2026 remains sealed.

    The selection is deliberately conservative: the same model/horizon/quantile
    must have positive expectation under base conditions, positive total return
    at 2x cost, and positive total return at 1s latency. No 2026 metric enters
    this decision.
    """
    if summaries.empty:
        return None
    validation = summaries.loc[summaries["fold_id"] == "WF_2025"].copy()
    base = validation.loc[
        (validation["latency_seconds"] == config.base_latency_seconds)
        & (validation["cost_multiplier"] == 1.0)
    ]
    robust_cost = validation.loc[
        (validation["latency_seconds"] == config.base_latency_seconds)
        & (validation["cost_multiplier"] == 2.0)
    ][["model", "horizon_seconds", "quantile", "total_return"]].rename(columns={"total_return": "return_2x"})
    robust_delay = validation.loc[
        (validation["latency_seconds"] == 1.0)
        & (validation["cost_multiplier"] == 1.0)
    ][["model", "horizon_seconds", "quantile", "total_return"]].rename(columns={"total_return": "return_1s"})
    candidates = base.merge(robust_cost, on=["model", "horizon_seconds", "quantile"], how="left")
    candidates = candidates.merge(robust_delay, on=["model", "horizon_seconds", "quantile"], how="left")
    candidates = candidates.loc[
        (candidates["trades"] >= 300)
        & (candidates["mean_net_return"] > 0)
        & (candidates["profit_factor"] > 1.05)
        & (candidates["return_2x"] > 0)
        & (candidates["return_1s"] > 0)
        & (candidates["top10_removed_total_return"] > 0)
    ].copy()
    if candidates.empty:
        return None

    def neighbour_is_stable(row: pd.Series) -> bool:
        peers = base.loc[
            (base["model"] == row["model"])
            & (base["horizon_seconds"] == row["horizon_seconds"])
            & (base["quantile"] != row["quantile"])
        ].copy()
        if peers.empty:
            return False
        peers["distance"] = (peers["quantile"] - float(row["quantile"])).abs()
        neighbour = peers.sort_values("distance", kind="stable").iloc[0]
        return bool(
            int(neighbour["trades"]) >= 100
            and float(neighbour["mean_net_return"]) > 0
            and float(neighbour["profit_factor"]) > 1.0
            and float(neighbour["top10_removed_total_return"]) > 0
        )

    candidates["neighbour_stable"] = candidates.apply(neighbour_is_stable, axis=1)
    candidates = candidates.loc[candidates["neighbour_stable"]].copy()
    if candidates.empty:
        return None
    candidates["robust_score"] = (
        candidates["return_2x"]
        + candidates["return_1s"]
        + candidates["total_return"]
    ) / (1.0 + candidates["max_drawdown"].clip(lower=0.0))
    row = candidates.sort_values(
        ["robust_score", "profit_factor", "trades"], ascending=[False, False, False], kind="stable"
    ).iloc[0]
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in row.to_dict().items()}
