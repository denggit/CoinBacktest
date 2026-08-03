#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reports and frozen decision gates for R11 liquidity-magnet research."""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.structured_stop_pool import FAMILY_COLUMNS

from .config import LiquidityMagnetConfig, stop_model_definitions


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame.loc[frame["outcome"].ne("INVALID")].copy()
    net1 = pd.to_numeric(valid["net_return_1x_cost"], errors="coerce")
    net2 = pd.to_numeric(valid["net_return_2x_cost"], errors="coerce")
    gross = pd.to_numeric(valid["gross_return"], errors="coerce")
    return {
        "events": int(len(valid)),
        "target_before_stop_rate": float(valid["target_before_stop"].mean()) if len(valid) else np.nan,
        "stop_rate": float(valid["stopped"].mean()) if len(valid) else np.nan,
        "time_exit_rate": float(valid["time_exit"].mean()) if len(valid) else np.nan,
        "gross_mean_bp": float(gross.mean() * 10_000.0) if gross.notna().any() else np.nan,
        "net_1x_mean_bp": float(net1.mean() * 10_000.0) if net1.notna().any() else np.nan,
        "net_1x_median_bp": float(net1.median() * 10_000.0) if net1.notna().any() else np.nan,
        "net_2x_mean_bp": float(net2.mean() * 10_000.0) if net2.notna().any() else np.nan,
        "net_positive_rate": float(net1.gt(0).mean()) if net1.notna().any() else np.nan,
        "profit_factor_1x": float(_profit_factor(net1)),
        "profit_factor_2x": float(_profit_factor(net2)),
        "median_target_distance_bp": float(pd.to_numeric(valid["target_distance_bp"], errors="coerce").median()) if len(valid) else np.nan,
        "median_stop_distance_bp": float(pd.to_numeric(valid["stop_distance_bp"], errors="coerce").median()) if len(valid) else np.nan,
        "median_nominal_reward_risk": float(pd.to_numeric(valid["nominal_reward_risk"], errors="coerce").median()) if len(valid) else np.nan,
        "median_mfe_short_bp": float(pd.to_numeric(valid["mfe_short_bp"], errors="coerce").median()) if len(valid) else np.nan,
        "median_mae_short_bp": float(pd.to_numeric(valid["mae_short_bp"], errors="coerce").median()) if len(valid) else np.nan,
    }


def design_table(config: LiquidityMagnetConfig) -> pd.DataFrame:
    cfg = config.validate()
    rows: list[dict[str, object]] = []
    for band in cfg.distance_bands_bp:
        for definition in stop_model_definitions():
            rows.append(
                {
                    "distance_band_bp": float(band),
                    **definition,
                    "front_run_buffer_bp": float(cfg.front_run_buffer_bp),
                    "horizon_minutes": int(cfg.horizon_minutes),
                    "fee_rate_per_side": float(cfg.fee_rate_per_side),
                    "slippage_rate_per_side": float(cfg.slippage_rate_per_side),
                    "stressed_cost_multiplier": float(cfg.stressed_cost_multiplier),
                }
            )
    return pd.DataFrame(rows)


def data_quality(
    bars: pd.DataFrame,
    lifecycle: pd.DataFrame,
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(bars.index)
    primary_non_1m_gaps = int((index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).sum())
    checks = [
        ("primary_rows", len(bars), "INFO"),
        ("primary_non_1m_gaps", primary_non_1m_gaps, "INFO"),
        ("lifecycle_rows", len(lifecycle), "INFO"),
        ("candidate_rows", len(candidates), "INFO"),
        ("outcome_rows", len(outcomes), "INFO"),
        ("candidate_unique_ids", int(candidates["pool_event_id"].nunique()) if not candidates.empty else 0, "PASS" if candidates.empty or not candidates["pool_event_id"].duplicated().any() else "FAIL"),
        ("outcome_expected_multiplier", len(candidates) * len(stop_model_definitions()), "PASS" if len(outcomes) == len(candidates) * len(stop_model_definitions()) else "FAIL"),
        ("invalid_outcome_rows", int(outcomes["outcome"].eq("INVALID").sum()) if not outcomes.empty else 0, "PASS" if outcomes.empty or int(outcomes["outcome"].eq("INVALID").sum()) == 0 else "FAIL"),
        ("positive_target_distance_rows", int(pd.to_numeric(candidates.get("tradable_target_distance_bp"), errors="coerce").gt(0).sum()) if not candidates.empty else 0, "PASS" if candidates.empty or pd.to_numeric(candidates["tradable_target_distance_bp"], errors="coerce").gt(0).all() else "FAIL"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "status"])


def candidate_funnel(candidates: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows = [
        {"stage": "unique_pool_checkpoints", "rows": int(len(candidates)), "share_of_candidates": 1.0},
        {
            "stage": "multi_timeframe_active_pool",
            "rows": int(pd.to_numeric(candidates["active_timeframe_count_10p0bp"], errors="coerce").ge(2).sum()),
            "share_of_candidates": float(pd.to_numeric(candidates["active_timeframe_count_10p0bp"], errors="coerce").ge(2).mean()),
        },
        {
            "stage": "high_timeframe_pool_1h_plus",
            "rows": int(pd.to_numeric(candidates["pool_max_timeframe_min"], errors="coerce").ge(60).sum()),
            "share_of_candidates": float(pd.to_numeric(candidates["pool_max_timeframe_min"], errors="coerce").ge(60).mean()),
        },
        {
            "stage": "any_r09_structured_family",
            "rows": int(candidates["has_any_structured_family"].astype(bool).sum()),
            "share_of_candidates": float(candidates["has_any_structured_family"].astype(bool).mean()),
        },
    ]
    if not outcomes.empty:
        eq = outcomes.loc[outcomes["stop_model"].eq("EQUAL_DISTANCE")]
        rows.append(
            {
                "stage": "equal_distance_target_before_stop",
                "rows": int(eq["target_before_stop"].sum()),
                "share_of_candidates": float(eq["target_before_stop"].mean()) if len(eq) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _quality_slices(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "ALL", frame
    yield "MULTITF_10BP", frame.loc[pd.to_numeric(frame["active_timeframe_count_10p0bp"], errors="coerce").ge(2)]
    yield "HIGH_TF_1H_PLUS", frame.loc[pd.to_numeric(frame["pool_max_timeframe_min"], errors="coerce").ge(60)]
    yield "MULTITF_AND_HIGH_TF", frame.loc[
        pd.to_numeric(frame["active_timeframe_count_10p0bp"], errors="coerce").ge(2)
        & pd.to_numeric(frame["pool_max_timeframe_min"], errors="coerce").ge(60)
    ]
    yield "ANY_R09_STRUCTURE", frame.loc[frame["has_any_structured_family"].astype(bool)]


def risk_frontier_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for quality, subset in _quality_slices(outcomes):
        for (band, stop_model), part in subset.groupby(["distance_band_bp", "stop_model"], sort=True):
            rows.append(
                {
                    "quality_slice": quality,
                    "distance_band_bp": float(band),
                    "stop_model": str(stop_model),
                    **_metrics(part),
                }
            )
    return pd.DataFrame(rows)


def directional_magnet_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Equal-distance lower-target vs upper-risk barrier is the clean direction test."""
    if outcomes.empty:
        return pd.DataFrame()
    eq = outcomes.loc[outcomes["stop_model"].eq("EQUAL_DISTANCE")].copy()
    rows: list[dict[str, object]] = []
    for quality, subset in _quality_slices(eq):
        for band, part in subset.groupby("distance_band_bp", sort=True):
            metrics = _metrics(part)
            rows.append(
                {
                    "quality_slice": quality,
                    "distance_band_bp": float(band),
                    **metrics,
                    "directional_edge_pp_vs_50": (float(metrics["target_before_stop_rate"]) - 0.5) * 100.0 if np.isfinite(metrics["target_before_stop_rate"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def timeframe_confluence_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    work = outcomes.copy()
    work["timeframe_bucket"] = pd.cut(
        pd.to_numeric(work["pool_max_timeframe_min"], errors="coerce"),
        bins=[0, 15, 30, 60, 240, np.inf],
        labels=["15m", "30m", "1H", "4H", "1D"],
        include_lowest=True,
    ).astype(str)
    work["confluence_bucket"] = np.where(
        pd.to_numeric(work["active_timeframe_count_10p0bp"], errors="coerce").ge(2),
        "MULTITF",
        "SINGLE_TF",
    )
    for keys, part in work.groupby(["distance_band_bp", "stop_model", "timeframe_bucket", "confluence_bucket"], observed=True, sort=True):
        band, model, timeframe, confluence = keys
        rows.append(
            {
                "distance_band_bp": float(band),
                "stop_model": str(model),
                "timeframe_bucket": str(timeframe),
                "confluence_bucket": str(confluence),
                **_metrics(part),
            }
        )
    return pd.DataFrame(rows)


def structure_family_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    eq = outcomes.loc[outcomes["stop_model"].eq("EQUAL_DISTANCE")]
    for family in FAMILY_COLUMNS:
        if family not in eq.columns:
            continue
        part = eq.loc[eq[family].astype(bool)]
        for band, group in part.groupby("distance_band_bp", sort=True):
            rows.append(
                {
                    "family_feature": family,
                    "distance_band_bp": float(band),
                    **_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def period_stability(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for quality, subset in _quality_slices(outcomes):
        for keys, part in subset.groupby(["distance_band_bp", "stop_model", "period"], sort=True):
            band, model, period = keys
            rows.append(
                {
                    "quality_slice": quality,
                    "distance_band_bp": float(band),
                    "stop_model": str(model),
                    "period": str(period),
                    **_metrics(part),
                }
            )
    return pd.DataFrame(rows)


def scorecard(
    frontier: pd.DataFrame,
    stability: pd.DataFrame,
    config: LiquidityMagnetConfig,
) -> pd.DataFrame:
    cfg = config.validate()
    if frontier.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for item in frontier.to_dict("records"):
        mask = (
            stability["quality_slice"].eq(item["quality_slice"])
            & pd.to_numeric(stability["distance_band_bp"], errors="coerce").eq(float(item["distance_band_bp"]))
            & stability["stop_model"].eq(item["stop_model"])
        )
        periods = stability.loc[mask].copy()
        valid_periods = periods.loc[pd.to_numeric(periods["events"], errors="coerce").ge(cfg.minimum_period_events)]
        positive_periods_1x = int(pd.to_numeric(valid_periods["net_1x_mean_bp"], errors="coerce").gt(0).sum())
        positive_periods_2x = int(pd.to_numeric(valid_periods["net_2x_mean_bp"], errors="coerce").gt(0).sum())
        enough = int(item["events"]) >= int(cfg.minimum_spec_events)
        directional = float(item["target_before_stop_rate"]) >= float(cfg.minimum_directional_target_rate)
        gross_positive = float(item["gross_mean_bp"]) > 0
        net1_positive = float(item["net_1x_mean_bp"]) > 0 and float(item["profit_factor_1x"]) > 1.0
        net2_positive = float(item["net_2x_mean_bp"]) > 0 and float(item["profit_factor_2x"]) > 1.0
        if enough and directional and net2_positive and positive_periods_2x >= cfg.minimum_positive_periods:
            decision = "promote_to_backtest"
        elif enough and directional and gross_positive and positive_periods_1x >= 2:
            decision = "research_continue"
        else:
            decision = "rejected"
        rows.append(
            {
                **item,
                "valid_period_count": int(len(valid_periods)),
                "positive_periods_1x": positive_periods_1x,
                "positive_periods_2x": positive_periods_2x,
                "gate_enough_events": bool(enough),
                "gate_directional_rate": bool(directional),
                "gate_gross_positive": bool(gross_positive),
                "gate_net_1x_positive": bool(net1_positive),
                "gate_net_2x_positive": bool(net2_positive),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision", "net_2x_mean_bp", "events"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def causal_audit(candidates: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    checks: list[tuple[str, int, str]] = []
    if candidates.empty:
        return pd.DataFrame([("nonempty_candidates", 0, "FAIL")], columns=["check", "violations", "status"])
    signal = pd.to_datetime(candidates["event_available_time"], errors="coerce")
    entry = pd.to_datetime(candidates["entry_time"], errors="coerce")
    member_available = pd.to_datetime(candidates["pool_member_initial_available_time_max"], errors="coerce")
    event_bar = pd.to_datetime(candidates["event_bar_time"], errors="coerce")
    event_pos = pd.to_numeric(candidates["event_pos"], errors="coerce")
    entry_pos = pd.to_numeric(candidates["entry_pos"], errors="coerce")
    expected_available = event_bar + pd.Timedelta(minutes=1)
    checks.append(("member_available_after_signal", int((member_available > signal).sum()), "PASS" if not (member_available > signal).any() else "FAIL"))
    checks.append(("event_available_not_bar_plus_1m", int(signal.ne(expected_available).sum()), "PASS" if signal.eq(expected_available).all() else "FAIL"))
    checks.append(("entry_pos_not_event_pos_plus_one", int(entry_pos.ne(event_pos + 1).sum()), "PASS" if entry_pos.eq(event_pos + 1).all() else "FAIL"))
    checks.append(("entry_not_at_signal_next_open", int(entry.ne(signal).sum()), "PASS" if entry.eq(signal).all() else "FAIL"))
    checks.append(("target_not_below_entry", int((pd.to_numeric(candidates["front_run_target_price"], errors="coerce") >= pd.to_numeric(candidates["entry_price"], errors="coerce")).sum()), "PASS" if (pd.to_numeric(candidates["front_run_target_price"], errors="coerce") < pd.to_numeric(candidates["entry_price"], errors="coerce")).all() else "FAIL"))
    checks.append(("future_sweep_feature_columns", int(sum(name.startswith("future_") for name in candidates.columns)), "INFO"))
    if not outcomes.empty:
        checks.append(("invalid_outcomes", int(outcomes["outcome"].eq("INVALID").sum()), "PASS" if not outcomes["outcome"].eq("INVALID").any() else "FAIL"))
        checks.append(("exit_before_entry", int((pd.to_datetime(outcomes["exit_time"], errors="coerce") < pd.to_datetime(outcomes["entry_time"], errors="coerce")).sum()), "PASS" if not (pd.to_datetime(outcomes["exit_time"], errors="coerce") < pd.to_datetime(outcomes["entry_time"], errors="coerce")).any() else "FAIL"))
    return pd.DataFrame(checks, columns=["check", "violations", "status"])


def research_brief(
    *,
    manifest: dict[str, object],
    frontier: pd.DataFrame,
    directional: pd.DataFrame,
    score: pd.DataFrame,
) -> str:
    promoted = score.loc[score["decision"].eq("promote_to_backtest")] if not score.empty else pd.DataFrame()
    continued = score.loc[score["decision"].eq("research_continue")] if not score.empty else pd.DataFrame()
    best = frontier.sort_values("net_1x_mean_bp", ascending=False).head(8) if not frontier.empty else pd.DataFrame()
    directional_best = directional.sort_values("target_before_stop_rate", ascending=False).head(8) if not directional.empty else pd.DataFrame()
    return f"""# R11 Liquidity Magnet and Risk Frontier Research Brief

## Research question

Do causally active, unconsumed Swing-Low liquidity pools attract price strongly
enough to support a short trade toward the pool, and can any predeclared stop
model make the route tradable after realistic costs?

R11 does **not** assume that a pool sweep reverses.  It studies the path before
the sweep and treats stop placement as a first-class research question.

## Frozen design

- Distance checkpoints: `{manifest.get("distance_bands_bp")}` bp above the pool.
- Entry: next 1m open after the closed bar first enters a checkpoint band.
- Tradable target: 5bp before the upper edge of the active lower-liquidity pool.
- Stops: equal-distance, prior completed 15m high + 5bp, prior completed 60m high + 5bp.
- Horizon: `{manifest.get("horizon_minutes")}` minutes.
- Same-bar target and stop: conservative stop.
- Costs: 0.11% fees plus 2bp round-trip slippage at 1x; doubled in stress.
- Families and quality slices are reported separately; no combination mining.

## Decision summary

- Promote to backtest rows: `{len(promoted)}`
- Research-continue rows: `{len(continued)}`
- Rejected rows: `{int(score["decision"].eq("rejected").sum()) if not score.empty else 0}`

## Best net 1x frontier rows

```text
{best.to_string(index=False) if not best.empty else "<none>"}
```

## Strongest equal-distance directional rows

```text
{directional_best.to_string(index=False) if not directional_best.empty else "<none>"}
```

## Interpretation rule

A target hit rate above 50% with the equal-distance stop supports a directional
magnet effect.  It is not enough for trading: target distance must exceed costs,
net expectancy must be positive, and results must remain stable across all
three periods.  Local-high stops are judged separately because they can reduce
false exits but may create poor capital efficiency.

If no specification survives costs, the correct conclusion is not to tune more
distance bands.  The liquidity map may still be useful as a target map, but the
pre-sweep short route is not independently tradable with the tested risk models.
"""


def manifest_json(data: dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str, sort_keys=True)
