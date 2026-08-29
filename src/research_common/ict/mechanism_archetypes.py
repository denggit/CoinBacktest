"""Mechanism archetype annotations for SOXL ICT semantic-gap research.

R09 does not turn R08's profitable buckets into entry rules.  Instead it learns
non-PnL distribution landmarks from the discovery period and tags every causal
setup with one or more interpretable path mechanisms.  Tags can overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .semantic_gap import summarize_outcome_group


@dataclass(frozen=True)
class MechanismArchetypeConfig:
    discovery_end_date: str = "2024-12-31"
    forward_start_date: str = "2025-01-01"
    forward_end_date: str = "2025-12-31"
    holdout_start_date: str = "2026-01-01"
    min_discovery_samples: int = 40


EDGE_FEATURES: tuple[str, ...] = (
    "maturity_terminal_extension_bp",
    "maturity_first_reclaim_after_final_terminal_minutes",
    "maturity_max_consecutive_outside_closes_terminal_to_signal",
    "maturity_penetration_area_bp_minutes_terminal_to_signal",
    "maturity_progressive_extreme_count",
    "terminal_to_mss_minutes",
    "mss_overshoot_pct",
    "reversal_path_efficiency",
    "semantic_reference_age_at_mss_minutes",
)


def _period(values: pd.Series, cfg: MechanismArchetypeConfig) -> pd.Series:
    d = pd.to_datetime(values, errors="coerce").dt.tz_localize(None).dt.normalize()
    disc_end = pd.Timestamp(cfg.discovery_end_date)
    fwd_start = pd.Timestamp(cfg.forward_start_date)
    fwd_end = pd.Timestamp(cfg.forward_end_date)
    hold = pd.Timestamp(cfg.holdout_start_date)
    return pd.Series(np.select(
        [d <= disc_end, (d >= fwd_start) & (d <= fwd_end), d >= hold],
        [f"discovery_through_{disc_end.date()}", "2025_forward", "2026_late_holdout"],
        default="other",
    ), index=values.index, dtype="object")


def fit_mechanism_distribution_edges(
    attempts: pd.DataFrame,
    *,
    config: MechanismArchetypeConfig = MechanismArchetypeConfig(),
    features: Iterable[str] = EDGE_FEATURES,
) -> pd.DataFrame:
    """Fit Q25/Q50/Q75 from discovery attempts only; PnL is never consulted."""
    if attempts.empty:
        return pd.DataFrame()
    work = attempts.copy()
    work["analysis_period"] = _period(work["ny_date"], config)
    disc_label = f"discovery_through_{pd.Timestamp(config.discovery_end_date).date()}"
    rows: list[dict[str, object]] = []
    for (tf, family), g in work.groupby(["execution_tf", "liquidity_family"], sort=True):
        d = g.loc[g["analysis_period"] == disc_label]
        for feature in features:
            if feature not in d:
                continue
            x = pd.to_numeric(d[feature], errors="coerce").dropna()
            if len(x) < int(config.min_discovery_samples):
                continue
            q = x.quantile([0.25, 0.50, 0.75])
            rows.append({
                "execution_tf": tf,
                "liquidity_family": family,
                "feature": feature,
                "discovery_samples": int(len(x)),
                "q25": float(q.iloc[0]), "q50": float(q.iloc[1]), "q75": float(q.iloc[2]),
            })
    return pd.DataFrame(rows)


def _edge_map(edges: pd.DataFrame) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    out: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    if edges.empty:
        return out
    for r in edges.itertuples(index=False):
        out[(str(r.execution_tf), str(r.liquidity_family), str(r.feature))] = (float(r.q25), float(r.q50), float(r.q75))
    return out


def attach_mechanism_archetypes(
    attempts: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Attach overlapping causal mechanism tags without filtering candidates."""
    if attempts.empty:
        return attempts.copy()
    work = attempts.copy()
    em = _edge_map(edges)
    tag_cols = [
        "arch_fast_rejection", "arch_sustained_consumption", "arch_deep_flush",
        "arch_progressive_flush", "arch_equal_pool_stop_run", "arch_moderate_mss_delivery",
        "arch_extended_mss_delivery", "arch_clean_reversal", "arch_mature_reference",
    ]
    for c in tag_cols:
        work[c] = False

    for i, row in work.iterrows():
        tf, fam = str(row.get("execution_tf")), str(row.get("liquidity_family"))
        def val(name: str) -> float:
            try:
                return float(row.get(name, np.nan))
            except (TypeError, ValueError):
                return np.nan
        def qs(name: str):
            return em.get((tf, fam, name))

        reclaim, outside, area = val("maturity_first_reclaim_after_final_terminal_minutes"), val("maturity_max_consecutive_outside_closes_terminal_to_signal"), val("maturity_penetration_area_bp_minutes_terminal_to_signal")
        ext, prog = val("maturity_terminal_extension_bp"), val("maturity_progressive_extreme_count")
        overshoot, eff, age = val("mss_overshoot_pct"), val("reversal_path_efficiency"), val("semantic_reference_age_at_mss_minutes")
        q_reclaim, q_outside, q_area = qs("maturity_first_reclaim_after_final_terminal_minutes"), qs("maturity_max_consecutive_outside_closes_terminal_to_signal"), qs("maturity_penetration_area_bp_minutes_terminal_to_signal")
        q_ext, q_prog = qs("maturity_terminal_extension_bp"), qs("maturity_progressive_extreme_count")
        q_over, q_eff, q_age = qs("mss_overshoot_pct"), qs("reversal_path_efficiency"), qs("semantic_reference_age_at_mss_minutes")

        if q_reclaim and q_outside and np.isfinite(reclaim) and np.isfinite(outside):
            work.at[i, "arch_fast_rejection"] = bool(reclaim <= q_reclaim[0] and outside <= q_outside[1])
        if q_outside and q_area and np.isfinite(outside) and np.isfinite(area):
            work.at[i, "arch_sustained_consumption"] = bool(outside >= q_outside[1] and area >= q_area[1])
        if q_ext and np.isfinite(ext):
            work.at[i, "arch_deep_flush"] = bool(ext >= q_ext[2])
        if q_prog and np.isfinite(prog):
            work.at[i, "arch_progressive_flush"] = bool(prog >= max(2.0, q_prog[2]))
        work.at[i, "arch_equal_pool_stop_run"] = bool(row.get("eq_context_present", False) or fam == "equal_liquidity_pool")
        if q_over and np.isfinite(overshoot):
            work.at[i, "arch_moderate_mss_delivery"] = bool(q_over[0] <= overshoot <= q_over[2])
            work.at[i, "arch_extended_mss_delivery"] = bool(overshoot > q_over[2])
        if q_eff and np.isfinite(eff):
            work.at[i, "arch_clean_reversal"] = bool(eff >= q_eff[1])
        if q_age and np.isfinite(age):
            work.at[i, "arch_mature_reference"] = bool(age >= q_age[1])

    work["mechanism_tag_count"] = work[tag_cols].sum(axis=1).astype(int)
    work["mechanism_tags"] = work[tag_cols].apply(
        lambda r: "|".join(c.removeprefix("arch_") for c, x in r.items() if bool(x)) or "unclassified",
        axis=1,
    )
    work["mechanism_feature_available_time"] = pd.to_datetime(work["signal_time"])
    return work


def build_mechanism_causal_audit(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame([{"check": "mechanism_attempts_non_empty", "passed": False, "violations": 0}])
    bad = int((pd.to_datetime(attempts["mechanism_feature_available_time"]) > pd.to_datetime(attempts["signal_time"])).fillna(True).sum()) if "mechanism_feature_available_time" in attempts else len(attempts)
    leakage = [c for c in attempts if c.startswith("outcome_")]
    return pd.DataFrame([
        {"check": "mechanism_features_available_by_signal", "passed": bad == 0, "violations": bad},
        {"check": "no_outcome_columns_in_mechanism_attempts", "passed": len(leakage) == 0, "violations": len(leakage)},
    ])


def build_mechanism_archetype_atlas(
    lifecycle: pd.DataFrame,
    *,
    config: MechanismArchetypeConfig = MechanismArchetypeConfig(),
) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"], config)
    tags = [c for c in work if c.startswith("arch_")]
    rows: list[dict[str, object]] = []
    for tag in tags:
        active = work.loc[work[tag].fillna(False).astype(bool)]
        for (tf, family, period), g in active.groupby(["execution_tf", "liquidity_family", "analysis_period"], sort=True):
            rows.append({"archetype": tag.removeprefix("arch_"), "execution_tf": tf, "liquidity_family": family, "analysis_period": period, **summarize_outcome_group(g)})
    return pd.DataFrame(rows)


def build_mechanism_combination_atlas(
    lifecycle: pd.DataFrame,
    *,
    config: MechanismArchetypeConfig = MechanismArchetypeConfig(),
    min_trades: int = 5,
) -> pd.DataFrame:
    if lifecycle.empty or "mechanism_tags" not in lifecycle:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"], config)
    rows: list[dict[str, object]] = []
    for (tf, family, period, tags), g in work.groupby(["execution_tf", "liquidity_family", "analysis_period", "mechanism_tags"], sort=True):
        if len(g) < int(min_trades):
            continue
        rows.append({"execution_tf": tf, "liquidity_family": family, "analysis_period": period, "mechanism_tags": tags, "tag_count": int(pd.to_numeric(g["mechanism_tag_count"], errors="coerce").median()), **summarize_outcome_group(g)})
    return pd.DataFrame(rows)


def build_equal_pool_performance_atlas(lifecycle: pd.DataFrame, *, config: MechanismArchetypeConfig = MechanismArchetypeConfig()) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"], config)
    rows: list[dict[str, object]] = []
    dims = ["eq_source_tf", "eq_context_present", "eq_context_source_tf"]
    for dim in dims:
        if dim not in work:
            continue
        for (tf, family, period, value), g in work.groupby(["execution_tf", "liquidity_family", "analysis_period", dim], dropna=False, sort=True):
            rows.append({"dimension": dim, "value": str(value), "execution_tf": tf, "liquidity_family": family, "analysis_period": period, **summarize_outcome_group(g)})
    return pd.DataFrame(rows)
