#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal probability research for ICT liquidity-delivery events.

R18 deliberately keeps the target simple:

    one side of the frozen 08:30 prominent-15m liquidity pair is raided
        -> will price reach the opposite frozen liquidity before session end?

A second, entry-conditioned target asks whether an actually filled order reaches
that same opposite liquidity before the terminal-extreme stop.  No 25/50/75
milestones enter either label or any predictor.

Only predeclared causal fields are admitted to models.  Full-day path labels and
post-entry MFE/MAE are explicitly excluded from predictors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_RANGE_MODEL = "prominent_15m_pair_0830"
VISIBLE_SWING_P = 0.50

EVENT_HYPOTHESIS_FEATURES: dict[str, list[str]] = {
    # H1: quality/geometry of the frozen external-liquidity pair.  This first
    # version intentionally does not claim to identify hidden institutions.
    "H1_liquidity_context": [
        "source_prominence_score",
        "target_prominence_score",
        "source_age_minutes_at_raid",
        "target_age_minutes_at_raid",
        "range_width_frac_source",
        "approach_efficiency",
        "approach_distance_contraction_ratio",
        "approach_recent_range_vs_prior",
        "approach_monotonic_swing_count",
        "approach_three_swing_contraction",
    ],
    # H2: sweep/terminal maturity observable by the snapshot time.
    "H2_terminal_maturity": [
        "first_raid_penetration_frac_range",
        "terminal_penetration_frac_range",
        "terminal_version",
        "source_reclaimed_before_snapshot",
        "raid_to_snapshot_minutes",
        "terminal_to_break_minutes",
        "minutes_since_terminal_extreme",
    ],
    # H3: whether the broken structure is meaningful rather than a micro pivot.
    "H3_mss_structure": [
        "causal_visibility_percentile",
        "two_sided_excursion_vs_prior_range",
        "local_prominence_vs_prior_range",
        "mss_reference_relation",
        "reference_is_latest_newly_broken",
        "reference_is_highest_visibility_newly_broken",
        "reference_is_outermost_barrier_newly_broken",
        "break_wick_cross",
        "break_close_cross",
        "mss_reference_age_minutes",
    ],
    # H4: displacement quality of the actual structure-breaking leg.
    "H4_displacement": [
        "directional_bar_fraction",
        "path_efficiency",
        "path_net_distance_frac_range",
        "break_overshoot_frac_range",
    ],
    # H6: cross-timeframe confirmation known *by this snapshot* only.
    "H6_cross_tf": [
        "other_tf_visible_by_snapshot",
        "other_tf_visibility_by_snapshot",
        "other_tf_close_break_by_snapshot",
    ],
}

ENTRY_HYPOTHESIS_FEATURES: dict[str, list[str]] = {
    **EVENT_HYPOTHESIS_FEATURES,
    # H5: mitigation / execution geometry, observed when the order actually fills.
    "H5_mitigation_entry": [
        "entry_archetype",
        "execution_tf",
        "entry_order_type",
        "initial_risk_frac_range",
        "entry_progress_fraction",
        "fvg_size_frac_range",
        "signal_minutes_from_raid",
        "raid_count_so_far_at_entry",
        "penetration_so_far_frac_range",
        "source_reclaimed_at_entry",
        "fill_wait_minutes",
    ],
}

FORBIDDEN_MODEL_COLUMNS = {
    "traversal_complete",
    "opposite_hit_time",
    "opposite_hit_minutes_from_raid",
    "max_same_side_penetration_abs",
    "max_same_side_penetration_frac_range",
    "same_side_raid_count",
    "max_progress_fraction",
    "milestone_25_time",
    "milestone_50_time",
    "milestone_75_time",
    "milestone_100_time",
    "mfe_r",
    "mae_r",
    "stop_hit",
    "stop_time",
    "milestone_100_before_stop",
    "net_return_exit_100",
    "exit_reason_100",
}

DEFAULT_ENTRY_ARCHETYPES = (
    "mss_first_visible_close_break_next_open_market",
    "mss_first_visible_break_fvg_near",
    "mss_first_visible_break_fvg_ce",
    "mss_first_visible_ob_open_limit",
    "mss_first_visible_ob_fvg_overlap_mid_limit",
    "mss_first_visible_2m_structure_1m_fvg_near_limit",
)


@dataclass(frozen=True)
class ProbabilityStudyConfig:
    range_model: str = DEFAULT_RANGE_MODEL
    visible_swing_percentile: float = VISIBLE_SWING_P
    discovery_end: str = "2024-12-31"
    validation_end: str = "2025-12-31"
    random_state: int = 7


def period_label(values: pd.Series | Sequence) -> pd.Series:
    d = pd.to_datetime(pd.Series(values), errors="coerce")
    out = pd.Series("", index=d.index, dtype="object")
    out.loc[d <= pd.Timestamp("2024-12-31")] = "discovery_2023H2_2024"
    out.loc[(d >= pd.Timestamp("2025-01-01")) & (d <= pd.Timestamp("2025-12-31"))] = "validation_2025"
    out.loc[d >= pd.Timestamp("2026-01-01")] = "forward_2026"
    return out


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    mapped = s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    return mapped.astype("boolean").fillna(False).astype(bool)


def _to_ny_naive(value: pd.Series) -> pd.Series:
    x = pd.to_datetime(value, errors="coerce", utc=True)
    try:
        return x.dt.tz_convert("America/New_York").dt.tz_localize(None)
    except Exception:
        return x.dt.tz_localize(None)


def _safe_minutes(later: pd.Series, earlier: pd.Series) -> pd.Series:
    a = _to_ny_naive(later)
    b = _to_ny_naive(earlier)
    return (a - b).dt.total_seconds() / 60.0


def _source_target_prominence(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    side = df.get("first_raid_side", pd.Series("", index=df.index)).astype(str)
    up = pd.to_numeric(df.get("upper_prominence_score"), errors="coerce")
    lo = pd.to_numeric(df.get("lower_prominence_score"), errors="coerce")
    source = pd.Series(np.where(side.eq("low"), lo, up), index=df.index, dtype=float)
    target = pd.Series(np.where(side.eq("low"), up, lo), index=df.index, dtype=float)
    return source, target


def _source_target_confirmation(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    side = df.get("first_raid_side", pd.Series("", index=df.index)).astype(str)
    up = df.get("upper_confirmation_time", pd.Series(pd.NaT, index=df.index))
    lo = df.get("lower_confirmation_time", pd.Series(pd.NaT, index=df.index))
    source = pd.Series(np.where(side.eq("low"), lo, up), index=df.index)
    target = pd.Series(np.where(side.eq("low"), up, lo), index=df.index)
    return source, target


def _terminal_penetration_frac(df: pd.DataFrame) -> pd.Series:
    side = df.get("first_raid_side", pd.Series("", index=df.index)).astype(str)
    terminal = pd.to_numeric(df.get("terminal_extreme_price"), errors="coerce")
    source = pd.to_numeric(df.get("source_level_price", df.get("level_price")), errors="coerce")
    width = pd.to_numeric(df.get("range_width_abs"), errors="coerce")
    raw = np.where(side.eq("low"), source - terminal, terminal - source)
    out = pd.Series(raw, index=df.index, dtype=float) / width.replace(0.0, np.nan)
    return out.clip(lower=0.0)


def _base_feature_engineering(df: pd.DataFrame, *, snapshot_col: str) -> pd.DataFrame:
    q = df.copy()
    source_prom, target_prom = _source_target_prominence(q)
    if "source_prominence_score" in q.columns:
        source_prom = source_prom.where(source_prom.notna(), pd.to_numeric(q["source_prominence_score"], errors="coerce"))
    if "target_prominence_score" in q.columns:
        target_prom = target_prom.where(target_prom.notna(), pd.to_numeric(q["target_prominence_score"], errors="coerce"))
    q["source_prominence_score"] = source_prom
    q["target_prominence_score"] = target_prom
    source_conf, target_conf = _source_target_confirmation(q)
    raid_time = q.get("first_raid_time", q.get("sweep_time"))
    source_age = _safe_minutes(raid_time, source_conf)
    target_age = _safe_minutes(raid_time, target_conf)
    if "source_age_minutes_at_raid" in q.columns:
        source_age = source_age.where(source_age.notna(), pd.to_numeric(q["source_age_minutes_at_raid"], errors="coerce"))
    if "target_age_minutes_at_raid" in q.columns:
        target_age = target_age.where(target_age.notna(), pd.to_numeric(q["target_age_minutes_at_raid"], errors="coerce"))
    q["source_age_minutes_at_raid"] = source_age
    q["target_age_minutes_at_raid"] = target_age
    width = pd.to_numeric(q.get("range_width_abs"), errors="coerce")
    source_price = pd.to_numeric(q.get("source_level_price", q.get("level_price")), errors="coerce").abs()
    q["range_width_frac_source"] = width / source_price.replace(0.0, np.nan)
    q["terminal_penetration_frac_range"] = _terminal_penetration_frac(q)
    q["raid_to_snapshot_minutes"] = _safe_minutes(q[snapshot_col], raid_time)
    q["minutes_since_terminal_extreme"] = _safe_minutes(q[snapshot_col], q.get("terminal_extreme_time"))
    q["mss_reference_age_minutes"] = _safe_minutes(q[snapshot_col], q.get("mss_reference_available_time"))
    reclaim = _to_ny_naive(q.get("first_reclaim_time", pd.Series(pd.NaT, index=q.index)))
    snap = _to_ny_naive(q[snapshot_col])
    q["source_reclaimed_before_snapshot"] = (reclaim.notna() & snap.notna() & reclaim.le(snap)).astype(int)
    q["path_net_distance_frac_range"] = pd.to_numeric(q.get("path_net_distance_abs"), errors="coerce") / width.replace(0.0, np.nan)
    q["break_overshoot_frac_range"] = pd.to_numeric(q.get("break_overshoot_abs"), errors="coerce") / width.replace(0.0, np.nan)
    for c in (
        "break_wick_cross", "break_close_cross", "reference_is_latest_newly_broken",
        "reference_is_highest_visibility_newly_broken", "reference_is_outermost_barrier_newly_broken",
        "approach_three_swing_contraction", "source_reclaimed_at_entry",
    ):
        if c in q.columns:
            q[c] = _to_bool(q[c]).astype(int)
    return q


def _first_visible_per_event_tf(narratives: pd.DataFrame, threshold: float) -> pd.DataFrame:
    q = narratives.copy()
    q["causal_visibility_percentile"] = pd.to_numeric(q["causal_visibility_percentile"], errors="coerce")
    q = q.loc[q["causal_visibility_percentile"].ge(float(threshold))].copy()
    q["break_available_time"] = pd.to_datetime(q["break_available_time"], errors="coerce", utc=True)
    q = q.sort_values(["path_event_id", "execution_tf", "break_available_time", "mss_reference_time"], kind="mergesort")
    return q.drop_duplicates(["path_event_id", "execution_tf"], keep="first")


def build_event_snapshot_dataset(
    paths: pd.DataFrame,
    narratives: pd.DataFrame,
    *,
    config: ProbabilityStudyConfig = ProbabilityStudyConfig(),
) -> pd.DataFrame:
    """Build one causal first-visible MSS snapshot per event and timeframe."""
    p = paths.loc[paths["range_model"].astype(str).eq(config.range_model)].copy()
    n = narratives.loc[narratives["range_model"].astype(str).eq(config.range_model)].copy()
    if p.empty or n.empty:
        return pd.DataFrame()
    first = _first_visible_per_event_tf(n, config.visible_swing_percentile)
    # Canonical path metadata fills fields that can be absent from narrow caches.
    path_cols = [
        "path_event_id", "ny_date", "range_model", "upper_price", "lower_price",
        "upper_confirmation_time", "lower_confirmation_time", "upper_prominence_score", "lower_prominence_score",
        "range_width_abs", "first_raid_side", "first_raid_time", "source_level_price", "target_price",
        "traversal_complete", "first_raid_penetration_frac_range", "first_reclaim_time",
        "approach_efficiency", "approach_distance_contraction_ratio", "approach_recent_range_vs_prior",
        "approach_monotonic_swing_count", "approach_three_swing_contraction",
    ]
    path_cols = [c for c in path_cols if c in p.columns]
    base = first.drop(columns=[c for c in path_cols if c != "path_event_id" and c in first.columns], errors="ignore")
    out = base.merge(p[path_cols].drop_duplicates("path_event_id"), on="path_event_id", how="left", validate="many_to_one")
    out = _base_feature_engineering(out, snapshot_col="break_available_time")

    # Cross-TF features are strictly as-of the current snapshot.  A later 2m
    # confirmation is never backfilled into an earlier 1m snapshot.
    lookup = first[["path_event_id", "execution_tf", "break_available_time", "causal_visibility_percentile", "break_close_cross"]].copy()
    lookup["break_close_cross"] = _to_bool(lookup["break_close_cross"]).astype(int)
    rows: list[dict[str, object]] = []
    by_event = {k: g.copy() for k, g in lookup.groupby("path_event_id", sort=False)}
    for rec in out.to_dict("records"):
        ev = rec["path_event_id"]
        tf = str(rec.get("execution_tf", ""))
        snap = pd.Timestamp(rec["break_available_time"])
        g = by_event.get(ev, pd.DataFrame())
        other = g.loc[~g["execution_tf"].astype(str).eq(tf)].copy() if not g.empty else pd.DataFrame()
        if not other.empty:
            other = other.loc[pd.to_datetime(other["break_available_time"], utc=True).le(snap)]
        if other.empty:
            rec["other_tf_visible_by_snapshot"] = 0
            rec["other_tf_visibility_by_snapshot"] = np.nan
            rec["other_tf_close_break_by_snapshot"] = 0
        else:
            z = other.sort_values("break_available_time", kind="mergesort").iloc[-1]
            rec["other_tf_visible_by_snapshot"] = 1
            rec["other_tf_visibility_by_snapshot"] = float(z["causal_visibility_percentile"])
            rec["other_tf_close_break_by_snapshot"] = int(z["break_close_cross"])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["stage"] = "visible_mss_" + out["execution_tf"].astype(str)
    out["target_opposite_by_eod"] = _to_bool(out["traversal_complete"]).astype(int)
    out["period"] = period_label(out["ny_date"]).to_numpy()
    return out


def build_entry_probability_dataset(
    lifecycle: pd.DataFrame,
    event_snapshots: pd.DataFrame,
    *,
    config: ProbabilityStudyConfig = ProbabilityStudyConfig(),
    entry_archetypes: Sequence[str] = DEFAULT_ENTRY_ARCHETYPES,
) -> pd.DataFrame:
    q = lifecycle.loc[
        lifecycle["range_model"].astype(str).eq(config.range_model)
        & lifecycle["entry_archetype"].astype(str).isin(list(entry_archetypes))
    ].copy()
    if q.empty:
        return pd.DataFrame()
    q["filled"] = _to_bool(q["filled"])
    q = q.loc[q["filled"]].copy()
    if q.empty:
        return q
    # Attach the earliest causal MSS snapshot on the same execution timeframe if
    # possible; otherwise the earliest visible snapshot for the event.
    es = event_snapshots.copy()
    es["_snap_time"] = pd.to_datetime(es["break_available_time"], errors="coerce", utc=True)
    exact = es.sort_values("_snap_time").drop_duplicates(["path_event_id", "execution_tf"], keep="first")
    fallback = es.sort_values("_snap_time").drop_duplicates("path_event_id", keep="first")
    event_features = sorted({c for fs in EVENT_HYPOTHESIS_FEATURES.values() for c in fs})
    keep = ["path_event_id", "execution_tf"] + [c for c in event_features if c in exact.columns]
    exact = exact[keep].copy()
    exact = exact.rename(columns={c: f"_event_{c}" for c in event_features if c in exact.columns})
    q = q.merge(exact, on=["path_event_id", "execution_tf"], how="left")
    fkeep = ["path_event_id"] + [c for c in event_features if c in fallback.columns]
    fb = fallback[fkeep].copy().rename(columns={c: f"_fallback_{c}" for c in event_features if c in fallback.columns})
    q = q.merge(fb, on="path_event_id", how="left")
    for c in event_features:
        ec, fc = f"_event_{c}", f"_fallback_{c}"
        if ec in q.columns or fc in q.columns:
            a = q[ec] if ec in q.columns else pd.Series(np.nan, index=q.index)
            b = q[fc] if fc in q.columns else pd.Series(np.nan, index=q.index)
            q[c] = a.where(a.notna(), b)
    q = q.drop(columns=[c for c in q.columns if c.startswith("_event_") or c.startswith("_fallback_")], errors="ignore")
    q = _base_feature_engineering(q, snapshot_col="fill_time")
    q["target_tp_before_terminal_sl"] = _to_bool(q["milestone_100_before_stop"]).astype(int)
    q["period"] = period_label(q["ny_date"]).to_numpy()
    return q


def feature_list(groups: Mapping[str, Sequence[str]], selected_groups: Sequence[str], columns: Iterable[str]) -> list[str]:
    cols = set(columns)
    out: list[str] = []
    for g in selected_groups:
        for c in groups[g]:
            if c in FORBIDDEN_MODEL_COLUMNS:
                raise AssertionError(f"forbidden future/outcome column requested as predictor: {c}")
            if c in cols and c not in out:
                out.append(c)
    return out


def _split_feature_types(df: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    cat: list[str] = []
    num: list[str] = []
    for c in features:
        if c not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def make_logistic_pipeline(df: pd.DataFrame, features: Sequence[str]) -> Pipeline:
    num, cat = _split_feature_types(df, features)
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ])


def _metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"n": 0, "observed_rate": np.nan, "mean_probability": np.nan, "brier": np.nan, "log_loss": np.nan, "auc": np.nan}
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    return {
        "n": int(len(y)),
        "observed_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": auc,
    }


def _constant_probability_metrics(df: pd.DataFrame, target: str, train_mask: pd.Series) -> pd.DataFrame:
    y_train = pd.to_numeric(df.loc[train_mask, target], errors="coerce").dropna().astype(int)
    base = float(y_train.mean()) if len(y_train) else np.nan
    rows = []
    for period, g in df.groupby("period", sort=True):
        y = pd.to_numeric(g[target], errors="coerce").dropna().astype(int).to_numpy()
        rec = _metric_row(y, np.full(len(y), base, dtype=float))
        rows.append({"period": period, "model_set": "baseline_constant", "features": "", **rec})
    return pd.DataFrame(rows)


def fit_cumulative_hypothesis_models(
    df: pd.DataFrame,
    *,
    target: str,
    groups: Mapping[str, Sequence[str]],
    group_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit discovery-only logistic models and evaluate unchanged later.

    Returns metrics, per-row predictions for the full model, and standardized
    logistic coefficients for the full model.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    q = df.copy()
    yall = pd.to_numeric(q[target], errors="coerce")
    q = q.loc[yall.isin([0, 1])].copy()
    q[target] = pd.to_numeric(q[target], errors="coerce").astype(int)
    train = q["period"].eq("discovery_2023H2_2024")
    if train.sum() < 30 or q.loc[train, target].nunique() < 2:
        raise RuntimeError("not enough discovery observations/classes for probability model")

    metrics = [_constant_probability_metrics(q, target, train)]
    full_pred = pd.DataFrame()
    full_coef = pd.DataFrame()
    selected: list[str] = []
    for group in group_order:
        selected.append(group)
        feats = feature_list(groups, selected, q.columns)
        # Drop features that are entirely missing in discovery.  This is a
        # deterministic availability check, not an outcome/PnL selection.
        feats = [c for c in feats if q.loc[train, c].notna().any()]
        if not feats:
            continue
        pipe = make_logistic_pipeline(q.loc[train], feats)
        pipe.fit(q.loc[train, feats], q.loc[train, target])
        rows = []
        for period, g in q.groupby("period", sort=True):
            p = pipe.predict_proba(g[feats])[:, 1]
            rec = _metric_row(g[target].to_numpy(int), p)
            rows.append({
                "period": period,
                "model_set": "+".join(selected),
                "features": "|".join(feats),
                **rec,
            })
        metrics.append(pd.DataFrame(rows))
        if group == group_order[-1]:
            p = pipe.predict_proba(q[feats])[:, 1]
            id_cols = [c for c in ["ny_date", "path_event_id", "stage", "execution_tf", "entry_archetype", target, "period"] if c in q.columns]
            full_pred = q[id_cols].copy()
            full_pred["predicted_probability"] = p
            full_pred["model_set"] = "+".join(selected)
            pre = pipe.named_steps["pre"]
            model = pipe.named_steps["model"]
            names = pre.get_feature_names_out()
            full_coef = pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})
            full_coef["abs_coefficient"] = full_coef["coefficient"].abs()
            full_coef = full_coef.sort_values("abs_coefficient", ascending=False, kind="mergesort")
    return pd.concat(metrics, ignore_index=True), full_pred, full_coef


def calibration_table(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    q = predictions.copy()
    bins = np.linspace(0.0, 1.0, 11)
    q["probability_bin"] = pd.cut(q["predicted_probability"], bins=bins, include_lowest=True, right=True)
    rows = []
    for (period, b), g in q.groupby(["period", "probability_bin"], observed=True, sort=True):
        rows.append({
            "period": period,
            "probability_bin": str(b),
            "n": int(len(g)),
            "mean_predicted_probability": float(g["predicted_probability"].mean()),
            "observed_rate": float(pd.to_numeric(g[target], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def stage_baseline_summary(paths: pd.DataFrame, event_snapshots: pd.DataFrame, *, range_model: str = DEFAULT_RANGE_MODEL) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    p = paths.loc[paths["range_model"].astype(str).eq(range_model)].copy()
    p = p.loc[p["first_raid_side"].astype(str).isin(["low", "high"])].copy()
    p["period"] = period_label(p["ny_date"]).to_numpy()
    for period, g in p.groupby("period", sort=True):
        rows.append({"stage": "sweep", "period": period, "events": int(g["path_event_id"].nunique()), "opposite_rate": float(_to_bool(g["traversal_complete"]).mean())})
    for stage, sg in event_snapshots.groupby("stage", sort=True):
        for period, g in sg.groupby("period", sort=True):
            rows.append({"stage": stage, "period": period, "events": int(g["path_event_id"].nunique()), "opposite_rate": float(g["target_opposite_by_eod"].mean())})
    return pd.DataFrame(rows)


def paired_entry_scorecard(entry_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if entry_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    score_rows = []
    for arch, g in entry_df.groupby("entry_archetype", sort=True):
        y = pd.to_numeric(g["target_tp_before_terminal_sl"], errors="coerce")
        rr = pd.to_numeric(g.get("rr_to_100"), errors="coerce") if "rr_to_100" in g else pd.Series(dtype=float)
        score_rows.append({
            "entry_archetype": arch,
            "filled_events": int(g["path_event_id"].nunique()),
            "tp_before_sl_rate": float(y.mean()),
            "median_rr_to_opposite": float(rr.median()) if len(rr) else np.nan,
            "mean_rr_to_opposite": float(rr.mean()) if len(rr) else np.nan,
        })
    pair_rows = []
    arches = sorted(entry_df["entry_archetype"].dropna().astype(str).unique())
    piv = entry_df.pivot_table(index="path_event_id", columns="entry_archetype", values="target_tp_before_terminal_sl", aggfunc="first")
    for i, a in enumerate(arches):
        for b in arches[i + 1:]:
            if a not in piv or b not in piv:
                continue
            g = piv[[a, b]].dropna()
            if g.empty:
                continue
            aa = g[a].astype(int); bb = g[b].astype(int)
            pair_rows.append({
                "entry_a": a,
                "entry_b": b,
                "common_filled_events": int(len(g)),
                "a_tp_before_sl_rate": float(aa.mean()),
                "b_tp_before_sl_rate": float(bb.mean()),
                "a_only_success": int(((aa == 1) & (bb == 0)).sum()),
                "b_only_success": int(((aa == 0) & (bb == 1)).sum()),
                "both_success": int(((aa == 1) & (bb == 1)).sum()),
                "both_fail": int(((aa == 0) & (bb == 0)).sum()),
            })
    return pd.DataFrame(score_rows), pd.DataFrame(pair_rows)


def assert_no_forbidden_features(groups: Mapping[str, Sequence[str]]) -> None:
    bad = sorted({c for cols in groups.values() for c in cols if c in FORBIDDEN_MODEL_COLUMNS})
    if bad:
        raise AssertionError(f"future/outcome leakage columns in model feature map: {bad}")
