#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R04 multi-horizon liquidity opportunity research helpers.

R04 deliberately separates causal opportunity features from future path labels.
It does not decide at entry whether a setup is a short rebound or a multi-day
swing.  Instead it measures the full path after the same causal 5m episode
reclaim and studies which *entry-time* liquidity/context features predict:

* short rebound targets (0.3%--1.0%),
* medium targets (1.0%--2.0%),
* swing/major reversal targets (3.0%--5.0%+),
* continuation after the first frozen opposing 4H liquidity target.

No fixed time stop is introduced.  Horizons are label windows / censoring
windows only.  Same-bar target-vs-stop ambiguity is pessimistically resolved as
stop-first by requiring target_pos < stop_pos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex

EPS = 1e-12


@dataclass(frozen=True)
class R04Config:
    target_returns: tuple[float, ...] = (0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
    path_horizons_minutes: tuple[int, ...] = (60, 180, 360, 720, 1440, 2880, 4320, 7200, 10080, 20160)
    post_4h_horizons_minutes: tuple[int, ...] = (360, 720, 1440, 2880, 4320, 7200, 10080)
    market_roundtrip_cost: float = 0.0011
    max_horizon_minutes: int = 20160

    def validate(self) -> "R04Config":
        if not self.target_returns or any(x <= 0 for x in self.target_returns):
            raise ValueError("target_returns must be positive and non-empty")
        if tuple(sorted(self.target_returns)) != self.target_returns:
            raise ValueError("target_returns must be sorted")
        if not self.path_horizons_minutes or any(int(x) <= 0 for x in self.path_horizons_minutes):
            raise ValueError("path horizons must be positive")
        if int(self.max_horizon_minutes) < max(self.path_horizons_minutes):
            raise ValueError("max_horizon_minutes must cover all path horizons")
        if self.market_roundtrip_cost < 0:
            raise ValueError("market_roundtrip_cost cannot be negative")
        return self


def target_token(value: float) -> str:
    pct = float(value) * 100.0
    text = f"{pct:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_unique_opportunity_features(
    hierarchy_trade_rows: pd.DataFrame,
    hierarchy_stages: pd.DataFrame,
    r02_trade_features: pd.DataFrame,
    r02_trade_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build one causal feature row per concrete 5m long episode-reclaim trade.

    ``hierarchy_trade_rows`` contains repeated rows because one concrete trade
    can simultaneously be the first crossing for several named hierarchy
    cohorts.  We collapse those repetitions into boolean cohort flags while
    keeping the concrete R02 trade/stage as the analytical grain.
    """
    if hierarchy_trade_rows.empty:
        return pd.DataFrame()
    rows = hierarchy_trade_rows.copy()
    rows = rows.loc[
        pd.to_numeric(rows.get("trade_direction"), errors="coerce").eq(1)
        & pd.to_numeric(rows.get("execution_minutes"), errors="coerce").eq(5)
        & rows.get("trigger_type", pd.Series(index=rows.index, dtype=object)).astype(str).eq("episode_reclaim")
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["trade_event_id"] = rows["trade_event_id"].astype(str)
    rows["hierarchy_cohort"] = rows["hierarchy_cohort"].astype(str)

    cohort_flags = (
        rows.assign(_value=1)
        .pivot_table(index="trade_event_id", columns="hierarchy_cohort", values="_value", aggfunc="max", fill_value=0)
        .astype(np.int8)
    )
    cohort_flags.columns = [f"cohort_{str(c)}_flag" for c in cohort_flags.columns]
    base = rows.sort_values(["trade_event_id", "stage_id"], kind="stable").drop_duplicates("trade_event_id", keep="first")
    # R03.3 review rows also carry a future realized return for convenience.
    # It must never enter the R04 causal feature table.
    future_from_r033 = [
        c for c in base.columns
        if c.startswith("target_") and c != "target_htf240_price"
    ]
    base = base.drop(columns=["hierarchy_cohort", *future_from_r033], errors="ignore").merge(
        cohort_flags.reset_index(), on="trade_event_id", how="left", validate="one_to_one"
    )

    stage_cols = [
        "stage_id", "episode_stage_no", "sweep_pos_1m", "sweep_bar_time_1m", "episode_start_pos_1m",
        "episode_start_time_1m", "episode_elapsed_minutes", "levels_consumed_cum", "distinct_timeframes_cum",
        "max_source_timeframe_min_cum", "price_pools_10p0bp_cum", "pools_per_min_10p0bp_cum",
        "ict_price_pools_cum", "ict_st_only_pools_cum", "ict_it_plus_pools_cum", "ict_lt_pools_cum",
        "ict_htf240_pools_cum", "ict_multi_tf_pools_cum", "ict_external50_pools_cum", "ict_clean_pools_cum",
        "ict_structural_key_pools_cum", "ict_strongest_pool_rank_cum", "ict_max_pool_timeframes_cum",
        "ict_structural_key_pools_per_min_cum", "ict_it_plus_pools_per_min_cum",
    ]
    stage = hierarchy_stages[[c for c in stage_cols if c in hierarchy_stages.columns]].drop_duplicates("stage_id")
    # Prefer the hierarchy-stage copy for structural fields because it is the
    # canonical stage grain.  Avoid duplicate _x/_y columns.
    structural = [c for c in stage.columns if c != "stage_id" and c not in base.columns]
    base = base.merge(stage[["stage_id", *structural]], on="stage_id", how="left", validate="many_to_one")

    fcols = [
        "trade_event_id", "entry_pos_1m", "entry_time", "entry_price", "stop_price", "risk_bps",
        "signal_available_time", "episode_start_time_1m", "episode_start_pos_1m", "sweep_pos_1m",
        "year", "quarter", "month", "session_primary", "is_weekend_utc",
    ]
    f = r02_trade_features[[c for c in fcols if c in r02_trade_features.columns]].copy()
    f["trade_event_id"] = f["trade_event_id"].astype(str)
    f = f.drop_duplicates("trade_event_id")
    add_cols = [c for c in f.columns if c == "trade_event_id" or c not in base.columns]
    base = base.merge(f[add_cols], on="trade_event_id", how="left", validate="one_to_one")

    # Only the target *price* is causal and known at entry.  Outcome/gross
    # columns from R02 are future labels and are deliberately excluded here.
    lcols = ["trade_event_id", "target_htf240_price"]
    l = r02_trade_labels[[c for c in lcols if c in r02_trade_labels.columns]].copy()
    l["trade_event_id"] = l["trade_event_id"].astype(str)
    l = l.drop_duplicates("trade_event_id")
    base = base.merge(l, on="trade_event_id", how="left", validate="one_to_one")

    for c in ("entry_time", "signal_available_time", "sweep_bar_time_1m", "episode_start_time_1m"):
        if c in base.columns:
            base[c] = pd.to_datetime(base[c], errors="coerce")
    base["pool_n_bucket"] = np.select(
        [
            pd.to_numeric(base.get("ict_price_pools_cum"), errors="coerce").ge(4),
            pd.to_numeric(base.get("ict_price_pools_cum"), errors="coerce").eq(3),
            pd.to_numeric(base.get("ict_price_pools_cum"), errors="coerce").eq(2),
        ],
        ["4+", "3", "2"],
        default="1",
    )
    base["contains_4h_pool_flag"] = pd.to_numeric(base.get("ict_htf240_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    base["contains_lt_pool_flag"] = pd.to_numeric(base.get("ict_lt_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    base["contains_it_plus_pool_flag"] = pd.to_numeric(base.get("ict_it_plus_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    base["contains_multi_tf_pool_flag"] = pd.to_numeric(base.get("ict_multi_tf_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    base["contains_external50_pool_flag"] = pd.to_numeric(base.get("ict_external50_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    base["contains_clean_pool_flag"] = pd.to_numeric(base.get("ict_clean_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    base["contains_structural_key_pool_flag"] = pd.to_numeric(base.get("ict_structural_key_pools_cum"), errors="coerce").fillna(0).ge(1).astype(np.int8)
    rank = pd.to_numeric(base.get("ict_strongest_pool_rank_cum"), errors="coerce").fillna(0).astype(int)
    base["strongest_ict_class"] = np.select([rank.ge(3), rank.eq(2), rank.eq(1)], ["LT", "IT", "ST"], default="none")

    entry = pd.to_numeric(base["entry_price"], errors="coerce")
    stop = pd.to_numeric(base["stop_price"], errors="coerce")
    risk = (entry - stop) / entry
    base["structural_risk_return"] = risk.where(entry.gt(0) & stop.gt(0) & stop.lt(entry))
    return base.sort_values(["entry_time", "trade_event_id"], kind="stable").reset_index(drop=True)


def attach_tradebar_features(opportunities: pd.DataFrame, tradebar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach causal trade-bar features by concrete trade ID with explicit coverage."""
    if opportunities.empty:
        return opportunities.copy(), pd.DataFrame()
    out = opportunities.copy()
    if tradebar.empty:
        audit = pd.DataFrame([{"metric": "tradebar_join_coverage", "expected": len(out), "matched": 0, "missing": len(out), "coverage": 0.0}])
        return out, audit
    tb = tradebar.copy()
    tb["checkpoint_id"] = tb["checkpoint_id"].astype(str)
    if tb["checkpoint_id"].duplicated().any():
        raise RuntimeError("R04 tradebar features contain duplicate checkpoint_id")
    before = len(out)
    out = out.merge(tb, left_on="trade_event_id", right_on="checkpoint_id", how="left", validate="one_to_one")
    matched = int(out["checkpoint_id"].notna().sum())
    audit = pd.DataFrame([{
        "metric": "tradebar_join_coverage", "expected": before, "matched": matched,
        "missing": before - matched, "coverage": matched / before if before else np.nan,
    }])
    return out, audit


def _range_mfe_mae(high_idx: RangeMinMaxIndex, low_idx: RangeMinMaxIndex, start: int, end: int, entry: float) -> tuple[float, float]:
    if start > end or not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan
    low_v, _ = low_idx.query(start, end)
    _, high_v = high_idx.query(start, end)
    mfe = high_v / entry - 1.0 if np.isfinite(high_v) else np.nan
    mae = low_v / entry - 1.0 if np.isfinite(low_v) else np.nan
    return mfe, mae


def build_multi_horizon_path_labels(
    opportunities: pd.DataFrame,
    bars_1m: pd.DataFrame,
    config: R04Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build future-only opportunity labels from bare 1m K efficiently.

    Label arrays are accumulated in a dictionary and materialized once at the
    end.  This avoids repeated DataFrame column insertion / fragmentation on
    the 100+ column full-history label atlas.
    """
    cfg = (config or R04Config()).validate()
    if opportunities.empty:
        return pd.DataFrame(), pd.DataFrame()
    bars = bars_1m.copy().sort_index(kind="stable")
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
        bars.index = idx
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    high_thr = SegmentThresholdIndex(high)
    low_thr = SegmentThresholdIndex(low)
    high_rng = RangeMinMaxIndex(high)
    low_rng = RangeMinMaxIndex(low)

    n = len(opportunities)
    data: dict[str, object] = {"trade_event_id": opportunities["trade_event_id"].astype(str).to_numpy()}
    audit_rows: list[dict[str, object]] = []
    entry_pos = pd.to_numeric(opportunities["entry_pos_1m"], errors="coerce").fillna(-1).astype(int).to_numpy()
    entry_time = pd.to_datetime(opportunities["entry_time"], errors="coerce")
    entry_price = pd.to_numeric(opportunities["entry_price"], errors="coerce").to_numpy(dtype=float)
    stop_price = pd.to_numeric(opportunities["stop_price"], errors="coerce").to_numpy(dtype=float)
    htf240_price = pd.to_numeric(opportunities.get("target_htf240_price"), errors="coerce").to_numpy(dtype=float)

    aligned = np.zeros(n, dtype=bool)
    for i in range(n):
        p = int(entry_pos[i])
        if 0 <= p < len(idx) and pd.notna(entry_time.iloc[i]):
            aligned[i] = pd.Timestamp(idx[p]) == pd.Timestamp(entry_time.iloc[i])
    if not bool(aligned.all()):
        raise RuntimeError(f"R04 entry position/time alignment failed for {int((~aligned).sum())} opportunities")
    audit_rows.append({"check": "entry_pos_time_alignment", "rows": n, "violations": 0})

    stop_pos = np.full(n, -1, dtype=np.int64)
    max_end = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        start = int(entry_pos[i])
        end = min(len(idx) - 1, start + int(cfg.max_horizon_minutes) - 1)
        max_end[i] = end
        stop_pos[i] = low_thr.first_leq(start, end, float(stop_price[i])) if np.isfinite(stop_price[i]) else -1
    data["first_structural_stop_pos"] = stop_pos
    data["minutes_to_structural_stop"] = np.where(stop_pos >= 0, stop_pos - entry_pos + 1, np.nan)

    target_positions: dict[float, np.ndarray] = {}
    resolved_outcomes: dict[float, np.ndarray] = {}
    for target in cfg.target_returns:
        tok = target_token(target)
        pos = np.full(n, -1, dtype=np.int64)
        before_stop = np.zeros(n, dtype=np.int8)
        outcome = np.empty(n, dtype=object)
        gross = np.full(n, np.nan, dtype=float)
        for i in range(n):
            start, end = int(entry_pos[i]), int(max_end[i])
            p = high_thr.first_geq(start, end, float(entry_price[i]) * (1.0 + float(target)))
            pos[i] = p
            stop_p = int(stop_pos[i])
            if p >= 0 and (stop_p < 0 or p < stop_p):
                before_stop[i] = 1
                outcome[i] = "target"
                gross[i] = float(target)
            elif stop_p >= 0 and (p < 0 or stop_p <= p):
                outcome[i] = "stop"
                gross[i] = float(stop_price[i] / entry_price[i] - 1.0)
            else:
                outcome[i] = "censored"
        target_positions[target] = pos
        resolved_outcomes[target] = outcome
        data[f"first_tp_{tok}_pos"] = pos
        data[f"minutes_to_tp_{tok}"] = np.where(pos >= 0, pos - entry_pos + 1, np.nan)
        data[f"tp_{tok}_before_stop_14d_flag"] = before_stop
        data[f"tp_{tok}_vs_stop_outcome_14d"] = outcome
        data[f"tp_{tok}_gross_return_resolved"] = gross
        resolved = np.asarray(outcome, dtype=object) != "censored"
        for mult, name in ((1.0, "base"), (2.0, "cost2x"), (3.0, "cost3x")):
            data[f"tp_{tok}_net_return_{name}"] = np.where(
                resolved, gross - float(cfg.market_roundtrip_cost) * mult, np.nan
            )

    for horizon in cfg.path_horizons_minutes:
        h = int(horizon)
        complete = np.zeros(n, dtype=np.int8)
        close_ret = np.full(n, np.nan, dtype=float)
        mfe = np.full(n, np.nan, dtype=float)
        mae = np.full(n, np.nan, dtype=float)
        for i in range(n):
            start = int(entry_pos[i])
            end = start + h - 1
            if end >= len(idx):
                continue
            complete[i] = 1
            close_ret[i] = close[end] / entry_price[i] - 1.0
            mfe[i], mae[i] = _range_mfe_mae(high_rng, low_rng, start, end, float(entry_price[i]))
        data[f"label_complete_{h}m_flag"] = complete
        data[f"close_return_{h}m"] = close_ret
        data[f"mfe_{h}m"] = mfe
        data[f"mae_{h}m"] = mae
        for target in cfg.target_returns:
            tok = target_token(target)
            p = target_positions[target]
            success = (p >= 0) & (p < entry_pos + h) & ((stop_pos < 0) | (p < stop_pos))
            failure = (stop_pos >= 0) & (stop_pos < entry_pos + h) & ((p < 0) | (stop_pos <= p))
            resolved = complete.astype(bool) | success | failure
            flag = np.full(n, np.nan, dtype=float)
            flag[resolved] = success[resolved].astype(float)
            data[f"tp_{tok}_before_stop_within_{h}m_flag"] = flag
            data[f"tp_{tok}_label_resolved_within_{h}m_flag"] = resolved.astype(np.int8)

    ladder_specs = (
        ("short_0p5_6h", 0.005, 360),
        ("short_0p75_12h", 0.0075, 720),
        ("medium_1p5_1d", 0.015, 1440),
        ("medium_2p0_2d", 0.02, 2880),
        ("swing_3p0_3d", 0.03, 4320),
        ("major_5p0_7d", 0.05, 10080),
    )
    for name, target, horizon in ladder_specs:
        p = target_positions[target]
        complete = (entry_pos + int(horizon) - 1) < len(idx)
        success = (p >= 0) & (p < entry_pos + int(horizon)) & ((stop_pos < 0) | (p < stop_pos))
        failure = (stop_pos >= 0) & (stop_pos < entry_pos + int(horizon)) & ((p < 0) | (stop_pos <= p))
        resolved = complete | success | failure
        flag = np.full(n, np.nan, dtype=float)
        flag[resolved] = success[resolved].astype(float)
        data[f"{name}_flag"] = flag
        data[f"{name}_label_resolved_flag"] = resolved.astype(np.int8)

    max_tier = np.zeros(n, dtype=float)
    for target in cfg.target_returns:
        hit = np.asarray(data[f"tp_{target_token(target)}_before_stop_14d_flag"], dtype=bool)
        max_tier[hit] = float(target)
    top_target_hit = np.asarray(data[f"tp_{target_token(cfg.target_returns[-1])}_before_stop_14d_flag"], dtype=bool)
    max_complete = ((entry_pos + int(cfg.max_horizon_minutes) - 1) < len(idx)) | (stop_pos >= 0) | top_target_hit
    data["max_target_before_stop_14d"] = np.where(max_complete, max_tier, np.nan)
    data["max_target_label_complete_flag"] = max_complete.astype(np.int8)

    htf_pos = np.full(n, -1, dtype=np.int64)
    htf_before_stop = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if not np.isfinite(htf240_price[i]) or htf240_price[i] <= entry_price[i]:
            continue
        p = high_thr.first_geq(int(entry_pos[i]), int(max_end[i]), float(htf240_price[i]))
        htf_pos[i] = p
        htf_before_stop[i] = int(p >= 0 and (stop_pos[i] < 0 or p < stop_pos[i]))
    data["first_htf240_target_pos"] = htf_pos
    data["htf240_target_before_stop_flag"] = htf_before_stop
    data["minutes_to_htf240_target"] = np.where(htf_pos >= 0, htf_pos - entry_pos + 1, np.nan)
    htf_return = htf240_price / entry_price - 1.0
    data["htf240_target_return_from_entry"] = htf_return

    for horizon in cfg.post_4h_horizons_minutes:
        h = int(horizon)
        complete = np.zeros(n, dtype=np.int8)
        add_mfe = np.full(n, np.nan, dtype=float)
        add_mae = np.full(n, np.nan, dtype=float)
        mfe_from_entry = np.full(n, np.nan, dtype=float)
        for i in range(n):
            p = int(htf_pos[i])
            if not htf_before_stop[i] or p < 0:
                continue
            start = p + 1
            end = start + h - 1
            if start >= len(idx) or end >= len(idx):
                continue
            complete[i] = 1
            low_v, _ = low_rng.query(start, end)
            _, high_v = high_rng.query(start, end)
            if np.isfinite(high_v):
                add_mfe[i] = high_v / htf240_price[i] - 1.0
                mfe_from_entry[i] = high_v / entry_price[i] - 1.0
            if np.isfinite(low_v):
                add_mae[i] = low_v / htf240_price[i] - 1.0
        data[f"post4h_complete_{h}m_flag"] = complete
        data[f"post4h_additional_mfe_{h}m"] = add_mfe
        data[f"post4h_additional_mae_{h}m"] = add_mae
        data[f"post4h_mfe_from_entry_{h}m"] = mfe_from_entry
        data[f"htf240_capture_ratio_vs_post4h_mfe_{h}m"] = np.where(
            (htf_before_stop == 1) & np.isfinite(mfe_from_entry) & (mfe_from_entry > EPS) & np.isfinite(htf_return),
            htf_return / mfe_from_entry, np.nan,
        )

    risk = (entry_price - stop_price) / entry_price
    for target in (0.005, 0.0075, 0.01):
        tok = target_token(target)
        for mult, name in ((1.0, "base"), (2.0, "cost2x"), (3.0, "cost3x")):
            c = float(cfg.market_roundtrip_cost) * mult
            required = (risk + c) / (float(target) + risk)
            data[f"partial_fraction_at_{tok}_to_cover_original_stop_{name}"] = np.where(
                np.isfinite(risk) & (risk > 0), required, np.nan
            )

    out = pd.DataFrame(data)
    audit_rows.append({"check": "same_bar_target_requires_strict_before_stop", "rows": n, "violations": 0})
    audit_rows.append({"check": "post4h_continuation_starts_next_1m_bar", "rows": int(htf_before_stop.sum()), "violations": 0})
    return out, pd.DataFrame(audit_rows)


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= EPS:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _rule_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    pools = pd.to_numeric(features["ict_price_pools_cum"], errors="coerce").fillna(0)
    key = pd.to_numeric(features["ict_structural_key_pools_cum"], errors="coerce").fillna(0).ge(1)
    h4 = features["contains_4h_pool_flag"].astype(bool)
    lt = features["contains_lt_pool_flag"].astype(bool)
    it = features["contains_it_plus_pool_flag"].astype(bool)
    return {
        "any_reclaim": pd.Series(True, index=features.index),
        "it_plus": it,
        "lt": lt,
        "4h_plus": h4,
        "n2_plus_key": pools.ge(2) & key,
        "n3_plus_key": pools.ge(3) & key,
        "n4_plus_key": pools.ge(4) & key,
        "n3_plus_4h": pools.ge(3) & h4,
        "n4_plus_4h": pools.ge(4) & h4,
        "n3_plus_lt": pools.ge(3) & lt,
        "n4_plus_lt": pools.ge(4) & lt,
    }


def first_qualifying_opportunities(features: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    part = features.loc[mask.fillna(False)].copy()
    if part.empty:
        return part
    return (
        part.sort_values(["episode_id", "signal_available_time", "stage_id", "trade_event_id"], kind="stable")
        .drop_duplicates("episode_id", keep="first")
        .reset_index(drop=True)
    )


def build_rule_horizon_scoreboard(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    months: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty or labels.empty:
        return pd.DataFrame(), pd.DataFrame()
    outcome_cols = {
        "short_0p5_6h": "short_0p5_6h_flag",
        "short_0p75_12h": "short_0p75_12h_flag",
        "medium_1p5_1d": "medium_1p5_1d_flag",
        "medium_2p0_2d": "medium_2p0_2d_flag",
        "swing_3p0_3d": "swing_3p0_3d_flag",
        "major_5p0_7d": "major_5p0_7d_flag",
    }
    pf_targets = (0.005, 0.01, 0.02, 0.03, 0.05)
    # The future-label table is intentionally very wide. Project only the
    # columns needed by this scoreboard, then materialize one compact frame
    # before repeated rule/year slicing. This avoids Pandas block fragmentation
    # and needless copying of hundreds of unrelated future-label columns.
    needed = ["trade_event_id", *outcome_cols.values()]
    needed += [f"tp_{target_token(t)}_net_return_cost2x" for t in pf_targets]
    needed = [c for c in dict.fromkeys(needed) if c in labels.columns]
    lab = labels.loc[:, needed].copy().set_index("trade_event_id")
    rows: list[dict[str, object]] = []
    yearly: list[dict[str, object]] = []
    rules = _rule_masks(features)
    for rule, mask in rules.items():
        selected = first_qualifying_opportunities(features, mask)
        if selected.empty:
            continue
        ids = selected["trade_event_id"].astype(str)
        l = lab.loc[ids].reset_index()
        row: dict[str, object] = {
            "rule": rule, "opportunities": len(selected), "episodes": selected["episode_id"].nunique(),
            "trades_per_month": len(selected) / months if months > 0 else np.nan,
            "median_structural_risk_pct": float(pd.to_numeric(selected["structural_risk_return"], errors="coerce").median() * 100.0),
        }
        for name, c in outcome_cols.items():
            row[f"{name}_rate"] = float(pd.to_numeric(l[c], errors="coerce").mean()) if c in l else np.nan
        for target in pf_targets:
            tok = target_token(target)
            c = f"tp_{tok}_net_return_cost2x"
            x = pd.to_numeric(l[c], errors="coerce") if c in l else pd.Series(dtype=float)
            row[f"tp_{tok}_resolved"] = int(x.notna().sum())
            row[f"tp_{tok}_cost2x_pf"] = _profit_factor(x)
            row[f"tp_{tok}_cost2x_expectancy"] = float(x.mean()) if x.notna().any() else np.nan
        rows.append(row)

        selected_y = selected.assign(_year=pd.to_datetime(selected["entry_time"], errors="coerce").dt.year)
        for year, yp in selected_y.groupby("_year", dropna=True, sort=True):
            yids = yp["trade_event_id"].astype(str)
            yl = lab.loc[yids].reset_index()
            yr: dict[str, object] = {"rule": rule, "year": int(year), "opportunities": len(yp)}
            for name, c in outcome_cols.items():
                yr[f"{name}_rate"] = float(pd.to_numeric(yl[c], errors="coerce").mean()) if c in yl else np.nan
            for target in (0.005, 0.03, 0.05):
                tok = target_token(target)
                c = f"tp_{tok}_net_return_cost2x"
                x = pd.to_numeric(yl[c], errors="coerce") if c in yl else pd.Series(dtype=float)
                yr[f"tp_{tok}_cost2x_pf"] = _profit_factor(x)
                yr[f"tp_{tok}_cost2x_expectancy"] = float(x.mean()) if x.notna().any() else np.nan
            yearly.append(yr)
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def build_transition_ladder(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if features.empty or labels.empty:
        return pd.DataFrame()
    merged = features[["trade_event_id", "pool_n_bucket", "contains_4h_pool_flag", "contains_lt_pool_flag"]].merge(
        labels, on="trade_event_id", how="inner", validate="one_to_one"
    )
    pairs = ((0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05))
    groups: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=merged.index))]
    groups += [(f"pool_n_{n}", merged["pool_n_bucket"].astype(str).eq(n)) for n in ("1", "2", "3", "4+")]
    groups += [("contains_4h", merged["contains_4h_pool_flag"].astype(bool)), ("contains_lt", merged["contains_lt_pool_flag"].astype(bool))]
    rows = []
    for gname, gmask in groups:
        part = merged.loc[gmask]
        if part.empty:
            continue
        for a, b in pairs:
            at, bt = target_token(a), target_token(b)
            a_hit = pd.to_numeric(part[f"tp_{at}_before_stop_14d_flag"], errors="coerce").eq(1)
            b_hit = pd.to_numeric(part[f"tp_{bt}_before_stop_14d_flag"], errors="coerce").eq(1)
            eligible = part.loc[a_hit]
            conditional = float(b_hit[a_hit].mean()) if int(a_hit.sum()) else np.nan
            a_min = pd.to_numeric(eligible[f"minutes_to_tp_{at}"], errors="coerce")
            b_min = pd.to_numeric(eligible[f"minutes_to_tp_{bt}"], errors="coerce")
            inc = (b_min - a_min).where(b_hit[a_hit].to_numpy())
            rows.append({
                "group": gname, "from_target": a, "to_target": b, "from_hits": int(a_hit.sum()),
                "conditional_upgrade_rate": conditional,
                "median_additional_minutes_if_upgraded": float(inc.dropna().median()) if inc.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def build_4h_continuation_summary(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if features.empty or labels.empty:
        return pd.DataFrame()
    merged = features[["trade_event_id", "pool_n_bucket", "contains_4h_pool_flag", "contains_lt_pool_flag"]].merge(
        labels, on="trade_event_id", how="inner", validate="one_to_one"
    )
    groups = {
        "all": pd.Series(True, index=merged.index),
        "n3_plus": merged["pool_n_bucket"].isin(["3", "4+"]),
        "n4_plus": merged["pool_n_bucket"].eq("4+"),
        "contains_4h": merged["contains_4h_pool_flag"].astype(bool),
        "n4_plus_4h": merged["pool_n_bucket"].eq("4+") & merged["contains_4h_pool_flag"].astype(bool),
        "n4_plus_lt": merged["pool_n_bucket"].eq("4+") & merged["contains_lt_pool_flag"].astype(bool),
    }
    rows = []
    for gname, mask in groups.items():
        part = merged.loc[mask & merged["htf240_target_before_stop_flag"].eq(1)].copy()
        if part.empty:
            continue
        row: dict[str, object] = {"group": gname, "htf240_winners": len(part)}
        row["median_minutes_to_htf240"] = float(pd.to_numeric(part["minutes_to_htf240_target"], errors="coerce").median())
        for h in (1440, 2880, 4320, 7200, 10080):
            c = f"post4h_additional_mfe_{h}m"
            comp = part[f"post4h_complete_{h}m_flag"].eq(1)
            x = pd.to_numeric(part.loc[comp, c], errors="coerce")
            row[f"post4h_{h}m_complete"] = int(comp.sum())
            row[f"post4h_{h}m_median_additional_mfe"] = float(x.median()) if x.notna().any() else np.nan
            row[f"post4h_{h}m_prob_add_0p5"] = float(x.ge(0.005).mean()) if x.notna().any() else np.nan
            row[f"post4h_{h}m_prob_add_1p0"] = float(x.ge(0.01).mean()) if x.notna().any() else np.nan
            row[f"post4h_{h}m_prob_add_2p0"] = float(x.ge(0.02).mean()) if x.notna().any() else np.nan
            cap = pd.to_numeric(part.loc[comp, f"htf240_capture_ratio_vs_post4h_mfe_{h}m"], errors="coerce")
            row[f"post4h_{h}m_median_capture_ratio"] = float(cap.median()) if cap.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_partial_risk_coverage_summary(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if features.empty or labels.empty:
        return pd.DataFrame()
    merged = features[["trade_event_id", "pool_n_bucket", "contains_4h_pool_flag", "contains_lt_pool_flag"]].merge(
        labels, on="trade_event_id", how="inner", validate="one_to_one"
    )
    groups = {
        "all": pd.Series(True, index=merged.index),
        "n3_plus": merged["pool_n_bucket"].isin(["3", "4+"]),
        "n4_plus": merged["pool_n_bucket"].eq("4+"),
        "n4_plus_4h": merged["pool_n_bucket"].eq("4+") & merged["contains_4h_pool_flag"].astype(bool),
    }
    rows = []
    for gname, mask in groups.items():
        part = merged.loc[mask]
        for target in (0.005, 0.0075, 0.01):
            tok = target_token(target)
            req = pd.to_numeric(part[f"partial_fraction_at_{tok}_to_cover_original_stop_cost2x"], errors="coerce")
            hit = pd.to_numeric(part[f"tp_{tok}_before_stop_14d_flag"], errors="coerce").eq(1)
            rows.append({
                "group": gname, "short_target": target, "opportunities": len(part),
                "target_before_stop_rate": float(hit.mean()) if len(part) else np.nan,
                "median_required_partial_fraction_cost2x": float(req.median()) if req.notna().any() else np.nan,
                "pct_required_fraction_le_0p5": float(req.le(0.5).mean()) if req.notna().any() else np.nan,
                "pct_required_fraction_le_0p75": float(req.le(0.75).mean()) if req.notna().any() else np.nan,
                "pct_required_fraction_le_1p0": float(req.le(1.0).mean()) if req.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def build_tradebar_horizon_summary(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Descriptive, pre-defined order-flow flags only; no threshold optimization."""
    needed = {"tb_absorption_mechanism_flag", "tb_flow_recovery_flag", "tb_causal_valid"}
    if features.empty or labels.empty or not needed.issubset(features.columns):
        return pd.DataFrame()
    cols = ["trade_event_id", *sorted(needed)]
    merged = features[cols].merge(labels, on="trade_event_id", how="inner", validate="one_to_one")
    merged = merged.loc[merged["tb_causal_valid"].fillna(False).astype(bool)]
    rows = []
    for flag in ("tb_absorption_mechanism_flag", "tb_flow_recovery_flag"):
        for value in (0, 1):
            part = merged.loc[pd.to_numeric(merged[flag], errors="coerce").fillna(0).astype(int).eq(value)]
            if part.empty:
                continue
            rows.append({
                "feature": flag, "value": value, "rows": len(part),
                "short_0p5_6h_rate": float(pd.to_numeric(part["short_0p5_6h_flag"], errors="coerce").mean()),
                "medium_1p5_1d_rate": float(pd.to_numeric(part["medium_1p5_1d_flag"], errors="coerce").mean()),
                "swing_3p0_3d_rate": float(pd.to_numeric(part["swing_3p0_3d_flag"], errors="coerce").mean()),
                "major_5p0_7d_rate": float(pd.to_numeric(part["major_5p0_7d_flag"], errors="coerce").mean()),
            })
    return pd.DataFrame(rows)


def r04_causal_audit(features: pd.DataFrame, labels: pd.DataFrame, path_audit: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if features.empty:
        return pd.DataFrame([{"check": "features_nonempty", "rows": 0, "violations": 1}])
    ids = features["trade_event_id"].astype(str)
    rows.append({"check": "feature_trade_id_unique", "rows": len(features), "violations": int(ids.duplicated().sum())})
    sig = pd.to_datetime(features["signal_available_time"], errors="coerce")
    ent = pd.to_datetime(features["entry_time"], errors="coerce")
    rows.append({"check": "entry_not_before_signal", "rows": len(features), "violations": int((ent < sig).fillna(False).sum())})
    forbidden_tokens = ("mfe_", "mae_", "close_return_", "minutes_to_tp_", "_before_stop_", "post4h_", "max_target_")
    bad_feature_cols = [
        c for c in features.columns
        if any(tok in c for tok in forbidden_tokens)
        or (c.startswith("target_") and c != "target_htf240_price")
    ]
    rows.append({"check": "future_label_columns_absent_from_features", "rows": len(features.columns), "violations": len(bad_feature_cols), "detail": ",".join(bad_feature_cols)})
    if not labels.empty:
        rows.append({"check": "feature_label_trade_id_match", "rows": len(features), "violations": int(set(ids) != set(labels["trade_event_id"].astype(str)))})
    if path_audit is not None and not path_audit.empty:
        for item in path_audit.itertuples(index=False):
            rows.append({"check": f"path::{item.check}", "rows": getattr(item, "rows", np.nan), "violations": getattr(item, "violations", np.nan)})
    return pd.DataFrame(rows)
