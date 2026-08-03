#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R12 reports and frozen decision gates."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research_common.structured_stop_pool import FAMILY_COLUMNS

from .config import PostSweepAcceptanceConfig, STATE_ORDER, state_direction


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def design_table(config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    cfg = config.validate()
    rows = []
    for checkpoint in cfg.checkpoints_minutes:
        for state in STATE_ORDER:
            rows.append(
                {
                    "checkpoint_minutes": int(checkpoint),
                    "state": state,
                    "preferred_direction": state_direction(state),
                    "stop_model": "visible path extreme +/- 5bp",
                    "targets_r": "|".join(str(v) for v in cfg.target_r_multiples),
                    "horizon_minutes": int(cfg.horizon_minutes),
                    "persistent_accept_share": float(cfg.persistent_accept_share),
                    "fee_rate_per_side": float(cfg.fee_rate_per_side),
                    "slippage_rate_per_side": float(cfg.slippage_rate_per_side),
                }
            )
    return pd.DataFrame(rows)


def data_quality(bars: pd.DataFrame, zones: pd.DataFrame, checkpoints: pd.DataFrame, outcomes: pd.DataFrame, config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    cfg = config.validate()
    index = pd.DatetimeIndex(bars.index)
    gaps = int((index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).sum())
    expected = len(checkpoints) * 2
    invalid = 0
    if not outcomes.empty:
        for r in cfg.target_r_multiples:
            token = str(float(r)).replace(".", "p")
            invalid += int(outcomes[f"r{token}_outcome"].eq("INVALID").sum())
    checks = [
        ("primary_rows", len(bars), "INFO"),
        ("primary_non_1m_gaps", gaps, "INFO"),
        ("r09_zone_sweeps", len(zones), "INFO"),
        ("checkpoint_rows", len(checkpoints), "INFO"),
        ("outcome_rows", len(outcomes), "INFO"),
        ("checkpoint_unique_key", len(checkpoints), "PASS" if checkpoints.empty or not checkpoints.duplicated(["zone_event_id", "checkpoint_minutes"]).any() else "FAIL"),
        ("outcome_expected_direction_multiplier", expected, "PASS" if len(outcomes) == expected else "FAIL"),
        ("invalid_target_outcomes", invalid, "PASS" if invalid == 0 else "FAIL"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "status"])


def causal_audit(checkpoints: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame([{"check": "checkpoint_rows", "violations": 0, "status": "FAIL"}])
    event_time = pd.to_datetime(checkpoints["event_bar_time"], errors="coerce")
    event_avail = pd.to_datetime(checkpoints["event_available_time"], errors="coerce")
    cp_time = pd.to_datetime(checkpoints["checkpoint_bar_time"], errors="coerce")
    cp_avail = pd.to_datetime(checkpoints["checkpoint_available_time"], errors="coerce")
    entry_time = pd.to_datetime(checkpoints["entry_time"], errors="coerce")
    event_pos = pd.to_numeric(checkpoints["event_pos"], errors="coerce")
    cp_pos = pd.to_numeric(checkpoints["checkpoint_pos"], errors="coerce")
    entry_pos = pd.to_numeric(checkpoints["entry_pos"], errors="coerce")
    cp_minutes = pd.to_numeric(checkpoints["checkpoint_minutes"], errors="coerce")
    checks = [
        ("event_available_not_bar_plus_1m", int((event_avail != event_time + pd.Timedelta(minutes=1)).sum())),
        ("checkpoint_pos_mismatch", int((cp_pos != event_pos + cp_minutes).sum())),
        ("checkpoint_available_not_bar_plus_1m", int((cp_avail != cp_time + pd.Timedelta(minutes=1)).sum())),
        ("entry_not_checkpoint_next_open", int(((entry_pos != cp_pos + 1) | (entry_time != cp_avail)).sum())),
        ("immediate_entry_not_sweep_next_open", int((pd.to_datetime(checkpoints["immediate_entry_time"], errors="coerce") != event_avail).sum())),
        ("zone_not_available_by_sweep", int((pd.to_datetime(checkpoints["zone_latest_level_available_time"], errors="coerce") > event_avail).sum())),
        ("unknown_state", int((~checkpoints["state"].isin(STATE_ORDER)).sum())),
        ("future_named_feature_columns", int(sum(name.startswith("future_") for name in checkpoints.columns))),
    ]
    if not outcomes.empty:
        checks.append(("natural_stop_nonpositive", int(pd.to_numeric(outcomes["natural_stop_distance_bp"], errors="coerce").le(0).sum())))
    return pd.DataFrame(
        [{"check": name, "violations": value, "status": "PASS" if value == 0 else "FAIL"} for name, value in checks]
    )


def state_distribution(checkpoints: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    rows = []
    for checkpoint, part in checkpoints.groupby("checkpoint_minutes", sort=True):
        total = len(part)
        for state, subset in part.groupby("state", sort=False):
            rows.append(
                {
                    "checkpoint_minutes": int(checkpoint),
                    "state": str(state),
                    "events": int(len(subset)),
                    "share": float(len(subset) / total) if total else np.nan,
                    "high_release_rate": float(subset["high_release"].mean()),
                    "high_timeframe_rate": float(subset["high_timeframe_zone"].mean()),
                    "multitimeframe_rate": float(subset["multitimeframe_zone"].mean()),
                    "median_close_vs_floor_bp": float(pd.to_numeric(subset["close_vs_floor_bp"], errors="coerce").median()),
                    "median_low_extension_bp": float(pd.to_numeric(subset["path_low_extension_below_sweep_bp"], errors="coerce").median()),
                }
            )
    return pd.DataFrame(rows)


def state_feature_profile(checkpoints: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    metrics = [
        "close_vs_floor_bp",
        "close_vs_ceiling_bp",
        "path_low_extension_below_sweep_bp",
        "close_recovery_from_path_low_bp",
        "post_close_below_floor_share",
        "terminal_consecutive_closes_above_floor",
        "post_sell_notional_sum",
        "post_delta_notional_sum",
        "second_vs_first_sell_impact_ratio",
        "after_reclaim_delta_notional_sum",
        "after_reclaim_sell_notional_sum",
        "long_entry_delay_bp",
        "short_entry_delay_bp",
        "pre_entry_mfe_long_bp",
        "pre_entry_mae_long_bp",
    ]
    rows = []
    for (checkpoint, state), part in checkpoints.groupby(["checkpoint_minutes", "state"], sort=True):
        row = {"checkpoint_minutes": int(checkpoint), "state": str(state), "events": int(len(part))}
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{metric}_q25"] = float(values.quantile(0.25)) if values.notna().any() else np.nan
            row[f"{metric}_q75"] = float(values.quantile(0.75)) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_slices(frame: pd.DataFrame):
    yield "ALL", frame
    yield "HIGH_RELEASE", frame.loc[frame["high_release"].astype(bool)]
    yield "HIGH_TF_1H_PLUS", frame.loc[frame["high_timeframe_zone"].astype(bool)]
    yield "MULTITF", frame.loc[frame["multitimeframe_zone"].astype(bool)]
    yield "HIGH_RELEASE_HIGH_TF", frame.loc[frame["high_release"].astype(bool) & frame["high_timeframe_zone"].astype(bool)]


def _outcome_metrics(part: pd.DataFrame, token: str) -> dict[str, object]:
    net1 = pd.to_numeric(part[f"r{token}_net_1x_r"], errors="coerce")
    net2 = pd.to_numeric(part[f"r{token}_net_2x_r"], errors="coerce")
    gross = pd.to_numeric(part[f"r{token}_gross_r"], errors="coerce")
    return {
        "events": int(len(part)),
        "target_before_stop_rate": float(part[f"r{token}_target_before_stop"].mean()) if len(part) else np.nan,
        "stop_rate": float(part[f"r{token}_stopped"].mean()) if len(part) else np.nan,
        "gross_mean_r": float(gross.mean()) if gross.notna().any() else np.nan,
        "net_1x_mean_r": float(net1.mean()) if net1.notna().any() else np.nan,
        "net_2x_mean_r": float(net2.mean()) if net2.notna().any() else np.nan,
        "profit_factor_1x": float(_profit_factor(net1)),
        "profit_factor_2x": float(_profit_factor(net2)),
        "net_positive_rate": float(net1.gt(0).mean()) if net1.notna().any() else np.nan,
        "median_stop_distance_bp": float(pd.to_numeric(part["natural_stop_distance_bp"], errors="coerce").median()),
        "median_mfe_bp": float(pd.to_numeric(part["mfe_bp"], errors="coerce").median()),
        "median_mae_bp": float(pd.to_numeric(part["mae_bp"], errors="coerce").median()),
        "median_long_entry_delay_bp": float(pd.to_numeric(part["long_entry_delay_bp"], errors="coerce").median()),
        "median_short_entry_delay_bp": float(pd.to_numeric(part["short_entry_delay_bp"], errors="coerce").median()),
        "median_pre_entry_mfe_long_bp": float(pd.to_numeric(part["pre_entry_mfe_long_bp"], errors="coerce").median()),
    }


def direction_outcome_summary(outcomes: pd.DataFrame, config: PostSweepAcceptanceConfig, direction: str) -> pd.DataFrame:
    cfg = config.validate()
    if outcomes.empty:
        return pd.DataFrame()
    work = outcomes.loc[outcomes["trade_direction"].eq(direction)].copy()
    preferred_states = {"PRESSURE_TEST_REJECT", "STRONG_REJECT", "REJECT"} if direction == "LONG" else {"RECLAIM_FAILED", "PERSISTENT_ACCEPT"}
    work = work.loc[work["state"].isin(preferred_states)]
    rows = []
    for quality, subset in _quality_slices(work):
        for (checkpoint, state), part in subset.groupby(["checkpoint_minutes", "state"], sort=True):
            for r in cfg.target_r_multiples:
                token = str(float(r)).replace(".", "p")
                rows.append(
                    {
                        "quality_slice": quality,
                        "trade_direction": direction,
                        "checkpoint_minutes": int(checkpoint),
                        "state": str(state),
                        "target_r": float(r),
                        **_outcome_metrics(part, token),
                    }
                )
    return pd.DataFrame(rows)


def period_stability(outcomes: pd.DataFrame, config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    cfg = config.validate()
    if outcomes.empty:
        return pd.DataFrame()
    work = outcomes.loc[
        ((outcomes["trade_direction"].eq("LONG")) & outcomes["state"].isin(["PRESSURE_TEST_REJECT", "STRONG_REJECT", "REJECT"]))
        | ((outcomes["trade_direction"].eq("SHORT")) & outcomes["state"].isin(["RECLAIM_FAILED", "PERSISTENT_ACCEPT"]))
    ].copy()
    rows = []
    for keys, part in work.groupby(["trade_direction", "checkpoint_minutes", "state", "period"], sort=True):
        direction, checkpoint, state, period = keys
        for r in cfg.target_r_multiples:
            token = str(float(r)).replace(".", "p")
            rows.append(
                {
                    "trade_direction": str(direction),
                    "checkpoint_minutes": int(checkpoint),
                    "state": str(state),
                    "period": str(period),
                    "target_r": float(r),
                    **_outcome_metrics(part, token),
                }
            )
    return pd.DataFrame(rows)


def transition_matrix(checkpoints: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    wide = checkpoints.pivot(index="zone_event_id", columns="checkpoint_minutes", values="state")
    cps = sorted(wide.columns)
    rows = []
    for left, right in zip(cps[:-1], cps[1:]):
        pair = wide.loc[:, [left, right]].dropna()
        total = len(pair)
        for (from_state, to_state), part in pair.groupby([left, right], sort=True):
            rows.append(
                {
                    "from_checkpoint": int(left),
                    "to_checkpoint": int(right),
                    "from_state": str(from_state),
                    "to_state": str(to_state),
                    "events": int(len(part)),
                    "share_of_pair_events": float(len(part) / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def release_interaction(outcomes: pd.DataFrame, config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    cfg = config.validate()
    if outcomes.empty:
        return pd.DataFrame()
    rows = []
    work = outcomes.copy()
    work["release_bucket"] = np.where(work["high_release"].astype(bool), "HIGH_RELEASE", "LOW_RELEASE")
    work = work.loc[
        ((work["trade_direction"].eq("LONG")) & work["state"].isin(["PRESSURE_TEST_REJECT", "STRONG_REJECT", "REJECT"]))
        | ((work["trade_direction"].eq("SHORT")) & work["state"].isin(["RECLAIM_FAILED", "PERSISTENT_ACCEPT"]))
    ]
    for keys, part in work.groupby(["trade_direction", "checkpoint_minutes", "state", "release_bucket"], sort=True):
        direction, checkpoint, state, bucket = keys
        for r in cfg.target_r_multiples:
            token = str(float(r)).replace(".", "p")
            rows.append(
                {
                    "trade_direction": str(direction),
                    "checkpoint_minutes": int(checkpoint),
                    "state": str(state),
                    "release_bucket": str(bucket),
                    "target_r": float(r),
                    **_outcome_metrics(part, token),
                }
            )
    return pd.DataFrame(rows)


def family_timeframe_summary(outcomes: pd.DataFrame, config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    cfg = config.validate()
    if outcomes.empty:
        return pd.DataFrame()
    preferred = outcomes.loc[
        ((outcomes["trade_direction"].eq("LONG")) & outcomes["state"].isin(["PRESSURE_TEST_REJECT", "STRONG_REJECT", "REJECT"]))
        | ((outcomes["trade_direction"].eq("SHORT")) & outcomes["state"].isin(["RECLAIM_FAILED", "PERSISTENT_ACCEPT"]))
    ].copy()
    rows = []

    def emit(dimension_type: str, dimension_value: str, subset: pd.DataFrame) -> None:
        for keys, part in subset.groupby(["trade_direction", "checkpoint_minutes", "state"], sort=True):
            direction, checkpoint, state = keys
            for r in cfg.target_r_multiples:
                token = str(float(r)).replace(".", "p")
                rows.append(
                    {
                        "dimension_type": dimension_type,
                        "dimension_value": dimension_value,
                        "trade_direction": str(direction),
                        "checkpoint_minutes": int(checkpoint),
                        "state": str(state),
                        "target_r": float(r),
                        **_outcome_metrics(part, token),
                    }
                )

    for family in FAMILY_COLUMNS:
        if family in preferred.columns:
            emit("FAMILY", family, preferred.loc[preferred[family].fillna(False).astype(bool)])
    timeframe = pd.cut(
        pd.to_numeric(preferred["zone_max_timeframe_min"], errors="coerce"),
        bins=[0, 15, 30, 60, 240, np.inf],
        labels=["15m", "30m", "1H", "4H", "1D"],
        include_lowest=True,
    ).astype(str)
    for bucket in sorted(timeframe.dropna().unique()):
        emit("MAX_TIMEFRAME", str(bucket), preferred.loc[timeframe.eq(bucket)])
    emit("CONFLUENCE", "MULTITF", preferred.loc[preferred["multitimeframe_zone"].astype(bool)])
    emit("CONFLUENCE", "SINGLE_TF", preferred.loc[~preferred["multitimeframe_zone"].astype(bool)])
    return pd.DataFrame(rows)


def scorecard(long_summary: pd.DataFrame, short_summary: pd.DataFrame, periods: pd.DataFrame, config: PostSweepAcceptanceConfig) -> pd.DataFrame:
    cfg = config.validate()
    combined = pd.concat([long_summary, short_summary], ignore_index=True)
    if combined.empty:
        return combined
    base = combined.loc[combined["quality_slice"].eq("ALL")].copy()
    rows = []
    for row in base.itertuples(index=False):
        source = row._asdict()
        matching = periods.loc[
            periods["trade_direction"].eq(source["trade_direction"])
            & periods["checkpoint_minutes"].eq(source["checkpoint_minutes"])
            & periods["state"].eq(source["state"])
            & periods["target_r"].eq(source["target_r"])
        ]
        eligible_periods = matching.loc[matching["events"].ge(int(cfg.minimum_period_events))]
        positive_periods = int(pd.to_numeric(eligible_periods["net_1x_mean_r"], errors="coerce").gt(0).sum())
        stable_period_count = int(len(eligible_periods))
        research_ok = (
            int(source["events"]) >= int(cfg.minimum_spec_events)
            and float(source["net_1x_mean_r"]) > 0
            and float(source["profit_factor_1x"]) >= float(cfg.research_pf_gate)
            and float(source["net_2x_mean_r"]) > 0
            and positive_periods >= 2
        )
        promote_ok = (
            research_ok
            and int(source["events"]) >= int(cfg.minimum_promote_events)
            and float(source["profit_factor_1x"]) >= float(cfg.promote_pf_gate)
            and stable_period_count == 3
            and positive_periods == 3
        )
        source.update(
            {
                "stable_period_count": stable_period_count,
                "positive_period_count": positive_periods,
                "decision": "promote_to_backtest" if promote_ok else "research_continue" if research_ok else "rejected",
            }
        )
        rows.append(source)
    return pd.DataFrame(rows)


def research_brief(manifest: dict[str, object], score: pd.DataFrame, long_summary: pd.DataFrame, short_summary: pd.DataFrame) -> str:
    promoted = score.loc[score["decision"].eq("promote_to_backtest")] if not score.empty else pd.DataFrame()
    continued = score.loc[score["decision"].eq("research_continue")] if not score.empty else pd.DataFrame()
    best_long = long_summary.sort_values("net_1x_mean_r", ascending=False).head(10) if not long_summary.empty else pd.DataFrame()
    best_short = short_summary.sort_values("net_1x_mean_r", ascending=False).head(10) if not short_summary.empty else pd.DataFrame()
    return f"""# R12 Post-Sweep Rejection vs Acceptance Research Brief

## Research question

After a causally identified Swing-Low liquidity zone is first swept and stop-like
sell flow is released, can the first 1/3/5/10 closed minutes distinguish:

- rejection: lower prices fail to gain acceptance, supporting a long;
- acceptance: price remains below the zone or a reclaim fails, supporting a short?

This study does not predict the exact low and does not tune a single impact or
reclaim threshold.  It uses exclusive path states and enters only at the next
1m open after each checkpoint.

## Frozen design

- Checkpoints: `{manifest.get('checkpoints_minutes')}` minutes after the sweep bar.
- Rejection states: PRESSURE_TEST_REJECT, STRONG_REJECT and REJECT.
- Acceptance states: RECLAIM_FAILED and PERSISTENT_ACCEPT.
- Natural stop: all price information visible by the checkpoint, plus 5bp.
- Targets: `{manifest.get('target_r_multiples')}` R.
- Horizon: `{manifest.get('horizon_minutes')}` minutes after entry.
- Same-bar target and stop: conservative stop.
- 1x cost: 0.11% fees plus 2bp round-trip slippage; 2x stress doubles it.
- H1-H8, high release and higher-timeframe slices are reports, not mined filters.

## Decision summary

- Promote to backtest: `{len(promoted)}`
- Research continue: `{len(continued)}`
- Rejected: `{int(score['decision'].eq('rejected').sum()) if not score.empty else 0}`

## Best rejection-long rows

```text
{best_long.to_string(index=False) if not best_long.empty else '<none>'}
```

## Best acceptance-short rows

```text
{best_short.to_string(index=False) if not best_short.empty else '<none>'}
```

## Interpretation rule

A visually appealing reclaim is not enough.  Confirmation must leave positive
expectancy after costs, retain adequate events, and repeat across periods.  If
all checkpoints lose most MFE or fail costs, the full liquidity-sweep trading
branch should be retired rather than rescued with additional filters.
"""


def manifest_json(data: dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str, sort_keys=True)
