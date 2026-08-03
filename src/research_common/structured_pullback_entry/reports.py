#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Answer-first reports for R10 structured pullback-entry research."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StructuredPullbackConfig, target_specs
from .universe import ALL_FAMILY_IDS, FAMILY_NAMES


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def _rate(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.astype(bool).mean()) if len(values) else np.nan


def _profit_factor(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    gains = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _periods_present(frame: pd.DataFrame) -> int:
    return int(frame.loc[frame["fill_status"].eq("FILLED"), "period"].dropna().nunique())


def data_quality(
    bars: pd.DataFrame,
    level_features: pd.DataFrame,
    unique_candidates: pd.DataFrame,
    family_candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    r09_source: str,
) -> pd.DataFrame:
    gaps = bars.index.to_series().diff().dropna()
    missing_minutes = int(
        np.maximum(gaps.dt.total_seconds().to_numpy(dtype=float) / 60.0 - 1.0, 0.0).sum()
    ) if len(gaps) else 0
    rows = [
        {"check": "primary_rows_positive", "value": len(bars), "status": "PASS" if len(bars) else "FAIL"},
        {"check": "primary_duplicate_timestamps", "value": int(bars.index.duplicated().sum()), "status": "PASS" if not bars.index.duplicated().any() else "FAIL"},
        {"check": "primary_estimated_missing_minutes", "value": missing_minutes, "status": "INFO"},
        {"check": "r09_level_features", "value": len(level_features), "status": "PASS" if len(level_features) else "FAIL"},
        {"check": "r09_feature_source", "value": r09_source, "status": "INFO"},
        {"check": "unique_higher_low_candidates", "value": len(unique_candidates), "status": "PASS" if len(unique_candidates) else "FAIL"},
        {"check": "expanded_family_candidates", "value": len(family_candidates), "status": "PASS" if len(family_candidates) else "FAIL"},
        {"check": "family_candidate_id_duplicates", "value": int(family_candidates.get("candidate_family_id", pd.Series(dtype=object)).duplicated().sum()), "status": "PASS" if not family_candidates.get("candidate_family_id", pd.Series(dtype=object)).duplicated().any() else "FAIL"},
        {"check": "filled_family_rows", "value": int(outcomes.get("fill_status", pd.Series(dtype=object)).eq("FILLED").sum()), "status": "INFO"},
        {"check": "valid_filled_trades", "value": int(outcomes.get("valid_filled_trade", pd.Series(dtype=bool)).fillna(False).sum()), "status": "INFO"},
    ]
    return pd.DataFrame(rows)


def candidate_funnel_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_id in ALL_FAMILY_IDS:
        part = frame.loc[frame["family_id"].eq(family_id)].copy()
        if part.empty:
            continue
        filled = part["fill_status"].eq("FILLED")
        valid = part["family_geometry_valid"].astype(bool)
        rows.append(
            {
                "family_id": family_id,
                "family_name": FAMILY_NAMES[family_id],
                "candidate_rows": len(part),
                "valid_geometry_rows": int(valid.sum()),
                "filled_rows": int(filled.sum()),
                "fill_rate": float(filled.mean()),
                "median_order_age_minutes_to_fill": _median(part.loc[filled, "order_age_minutes_to_fill"]),
                "h0_traded_before_fill_rate": _rate(part["h0_traded_before_fill_flag"]),
                "unfilled_rows": int((~filled).sum()),
                "periods_with_fills": _periods_present(part),
            }
        )
    return pd.DataFrame(rows)


def family_geometry_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family_id, timeframe), part in frame.groupby(["family_id", "source_timeframe"], sort=False):
        valid = part.loc[part["family_geometry_valid"].astype(bool)].copy()
        rows.append(
            {
                "family_id": family_id,
                "family_name": FAMILY_NAMES.get(str(family_id), str(family_id)),
                "source_timeframe": timeframe,
                "candidate_rows": len(part),
                "valid_geometry_rows": len(valid),
                "median_risk_distance_bp": _median(valid["risk_distance_return"]) * 10_000.0,
                "median_h0_reward_bp": _median(valid["h0_reward_return"]) * 10_000.0,
                "median_h0_reward_risk_ratio": _median(valid["h0_reward_risk_ratio"]),
                "h0_rr_ge_1_rate": _rate(valid["h0_reward_risk_ratio"].ge(1.0)),
                "h0_rr_ge_2_rate": _rate(valid["h0_reward_risk_ratio"].ge(2.0)),
            }
        )
    return pd.DataFrame(rows)


def _target_summary(part: pd.DataFrame, family_id: str, target: str) -> dict[str, object]:
    token = target.lower()
    trades = part.loc[part["valid_filled_trade"].fillna(False)].copy()
    net = pd.to_numeric(trades[f"{token}_net_r_realistic"], errors="coerce")
    net_2x = pd.to_numeric(trades[f"{token}_net_r_2x_cost"], errors="coerce")
    outcome = trades[f"{token}_outcome"].astype(str)
    sorted_net = net.sort_values(ascending=False)
    ex_top10 = sorted_net.iloc[min(10, len(sorted_net)) :]
    return {
        "family_id": family_id,
        "family_name": FAMILY_NAMES.get(family_id, family_id),
        "target": target,
        "candidate_rows": len(part),
        "filled_trades": len(trades),
        "fill_rate": float(part["fill_status"].eq("FILLED").mean()) if len(part) else np.nan,
        "tp_rate": float(outcome.eq("TP").mean()) if len(trades) else np.nan,
        "sl_rate": float(outcome.str.startswith("SL").mean()) if len(trades) else np.nan,
        "time_exit_rate": float(outcome.eq("TIME").mean()) if len(trades) else np.nan,
        "same_bar_both_rate": _rate(trades[f"{token}_same_bar_both_flag"]),
        "positive_net_r_rate": float(net.gt(0).mean()) if len(net.dropna()) else np.nan,
        "mean_net_r_realistic": float(net.mean()) if len(net.dropna()) else np.nan,
        "median_net_r_realistic": float(net.median()) if len(net.dropna()) else np.nan,
        "profit_factor_net_r_realistic": _profit_factor(net),
        "mean_net_r_2x_cost": float(net_2x.mean()) if len(net_2x.dropna()) else np.nan,
        "profit_factor_net_r_2x_cost": _profit_factor(net_2x),
        "mean_net_r_realistic_ex_top10": float(ex_top10.mean()) if len(ex_top10) else np.nan,
        "median_mae_bp": _median(trades[f"{token}_mae_return"]) * 10_000.0,
        "median_mfe_bp": _median(trades[f"{token}_mfe_return"]) * 10_000.0,
        "median_bars_to_exit": _median(trades[f"{token}_bars_to_exit"]),
        "periods_with_fills": _periods_present(trades),
    }


def family_outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_id in ALL_FAMILY_IDS:
        part = frame.loc[frame["family_id"].eq(family_id)].copy()
        if part.empty:
            continue
        for spec in target_specs():
            rows.append(_target_summary(part, family_id, spec.name))
    return pd.DataFrame(rows)


def family_timeframe_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family_id, timeframe), part in frame.groupby(["family_id", "source_timeframe"], sort=False):
        for spec in target_specs():
            row = _target_summary(part, str(family_id), spec.name)
            row["source_timeframe"] = timeframe
            rows.append(row)
    return pd.DataFrame(rows)


def period_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family_id, period), part in frame.groupby(["family_id", "period"], sort=False):
        for spec in target_specs():
            row = _target_summary(part, str(family_id), spec.name)
            row["period"] = period
            rows.append(row)
    return pd.DataFrame(rows)


def fill_age_summary(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.loc[frame["fill_status"].eq("FILLED")].copy()
    if filled.empty:
        return pd.DataFrame()
    age = pd.to_numeric(filled["order_age_minutes_to_fill"], errors="coerce")
    filled["fill_age_bucket"] = pd.cut(
        age,
        bins=[-np.inf, 15, 60, 240, 1_440, 10_080, np.inf],
        labels=["<=15m", "16-60m", "61-240m", "241-1440m", "1-7d", ">7d"],
        right=True,
    )
    rows: list[dict[str, object]] = []
    for (family_id, bucket), part in filled.groupby(["family_id", "fill_age_bucket"], observed=True, sort=False):
        rows.append(
            {
                "family_id": family_id,
                "family_name": FAMILY_NAMES.get(str(family_id), str(family_id)),
                "fill_age_bucket": str(bucket),
                "filled_trades": len(part),
                "share_within_family": len(part) / max(1, int(filled["family_id"].eq(family_id).sum())),
                "h0_mean_net_r_realistic": _mean(part["h0_net_r_realistic"]),
                "r2_mean_net_r_realistic": _mean(part["r2_net_r_realistic"]),
                "median_actual_risk_bp": _median(part["actual_risk_distance_bp"]),
            }
        )
    return pd.DataFrame(rows)


def family_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    unique = frame.loc[:, ["level_id", "family_id"]].drop_duplicates()
    counts = unique.groupby("level_id")["family_id"].nunique()
    return pd.DataFrame(
        {
            "family_memberships_per_level": counts.value_counts().sort_index().index,
            "level_count": counts.value_counts().sort_index().to_numpy(),
            "share": counts.value_counts(normalize=True).sort_index().to_numpy(),
        }
    )


def family_scorecard(
    outcomes: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    config: StructuredPullbackConfig,
) -> pd.DataFrame:
    cfg = config.validate()
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        family_id = str(row.family_id)
        target = str(row.target)
        periods = stability.loc[
            stability["family_id"].eq(family_id)
            & stability["target"].eq(target)
            & pd.to_numeric(stability["filled_trades"], errors="coerce").ge(cfg.minimum_period_fills)
        ].copy()
        positive_periods = int(pd.to_numeric(periods["mean_net_r_realistic"], errors="coerce").gt(0).sum())
        positive_2x_periods = int(pd.to_numeric(periods["mean_net_r_2x_cost"], errors="coerce").gt(0).sum())
        enough = int(row.candidate_rows) >= cfg.minimum_family_candidates and int(row.filled_trades) >= cfg.minimum_family_fills
        base_positive = np.isfinite(row.mean_net_r_realistic) and float(row.mean_net_r_realistic) > 0
        stress_positive = np.isfinite(row.mean_net_r_2x_cost) and float(row.mean_net_r_2x_cost) > 0
        pf = float(row.profit_factor_net_r_realistic) if pd.notna(row.profit_factor_net_r_realistic) else np.nan
        robust_top = np.isfinite(row.mean_net_r_realistic_ex_top10) and float(row.mean_net_r_realistic_ex_top10) > 0
        if (
            enough
            and base_positive
            and stress_positive
            and np.isfinite(pf)
            and pf >= 1.30
            and positive_periods >= 2
            and positive_2x_periods >= 2
            and robust_top
        ):
            decision = "promote_to_backtest"
        elif enough and base_positive and np.isfinite(pf) and pf >= 1.10 and positive_periods >= 2:
            decision = "research_continue"
        else:
            decision = "rejected"
        rows.append(
            {
                **row._asdict(),
                "eligible_period_rows": len(periods),
                "positive_periods_realistic": positive_periods,
                "positive_periods_2x_cost": positive_2x_periods,
                "enough_sample": enough,
                "top10_removal_positive": robust_top,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def causal_audit(
    unique_candidates: pd.DataFrame,
    family_candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(check: str, violations: int) -> None:
        rows.append({"check": check, "violations": int(violations), "status": "PASS" if int(violations) == 0 else "FAIL"})

    signal = pd.to_datetime(outcomes.get("structure_available_time"), errors="coerce")
    active = pd.to_datetime(outcomes.get("order_active_time"), errors="coerce")
    fill = pd.to_datetime(outcomes.get("fill_time"), errors="coerce")
    next_signal = pd.to_datetime(outcomes.get("next_same_timeframe_structure_available_time"), errors="coerce")
    add("unique_level_id", int(unique_candidates["level_id"].duplicated().sum()))
    add("unique_candidate_family_id", int(family_candidates["candidate_family_id"].duplicated().sum()))
    add("order_not_active_before_structure_available", int((active.notna() & signal.notna() & active.lt(signal)).sum()))
    add("fill_not_before_order_active", int((fill.notna() & active.notna() & fill.lt(active)).sum()))
    add("fill_before_next_structure_cancellation", int((fill.notna() & next_signal.notna() & fill.ge(next_signal)).sum()))
    add("stop_strictly_below_entry", int((pd.to_numeric(outcomes["stop_price"], errors="coerce") >= pd.to_numeric(outcomes["entry_limit_price"], errors="coerce")).sum()))
    add("h0_strictly_above_entry_for_valid_geometry", int((
        outcomes["family_geometry_valid"].astype(bool)
        & pd.to_numeric(outcomes["structural_target_h0_price"], errors="coerce").le(pd.to_numeric(outcomes["entry_limit_price"], errors="coerce"))
    ).sum()))
    forbidden = [name for name in unique_candidates.columns if name.startswith(("future_", "mfe_", "mae_", "tp_", "net_", "fill_"))]
    add("no_future_outcome_columns_in_candidate_features", len(forbidden))
    return pd.DataFrame(rows)


def research_brief(scorecard: pd.DataFrame, funnel: pd.DataFrame) -> str:
    promoted = scorecard.loc[scorecard["decision"].eq("promote_to_backtest"), ["family_id", "target"]].to_dict(orient="records")
    continued = scorecard.loc[scorecard["decision"].eq("research_continue"), ["family_id", "target"]].to_dict(orient="records")
    return f"""# R10 Structured Pullback Entry Study

## Question

After a Higher Low is causally confirmed, can a resting buy limit at that Higher Low,
with the stop below the earlier structural low, produce a tradable risk-adjusted edge?

## Design

- No liquidity Sweep is required before entry.
- The order starts only when the Higher Low is causally available.
- The old order is cancelled when the next same-timeframe Swing Low becomes available.
- P3/P5 stop below the lower two-low base; other families stop below the immediately prior Swing Low.
- Targets are the prior upswing high H0 and fixed 1R/2R/3R.
- Base fee is 0.11% round trip; the realistic column adds 2bp total slippage.
- Same-bar ambiguity is conservative.
- Families are evaluated separately; no combination mining is performed.

## Candidate funnel

```text\n{funnel.to_string(index=False) if not funnel.empty else "No candidates."}\n```

## Automated screen

- Promote-to-backtest rows: {promoted}
- Research-continue rows: {continued}

This is an event-level research replay. Overlapping signals, capital concurrency, exchange order
lifecycle and portfolio drawdown still require a dedicated candidate backtest before any live claim.
"""


__all__ = [
    "data_quality",
    "candidate_funnel_summary",
    "family_geometry_summary",
    "family_outcome_summary",
    "family_timeframe_summary",
    "period_stability",
    "fill_age_summary",
    "family_overlap",
    "family_scorecard",
    "causal_audit",
    "research_brief",
]
