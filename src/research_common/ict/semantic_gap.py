"""Causal semantic-gap diagnostics for discretionary ICT research.

This module does **not** define entry filters.  It describes the already-created
Sweep -> MSS -> FVG candidates with interpretable path features, then studies
how those features relate to outcomes.  Features prefixed ``semantic_`` are
computed only from information available no later than ``signal_time``.
Outcome columns are attached only after replay and must never be used by the
signal builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class SemanticGapConfig:
    reclaim_requires_close: bool = True
    terminal_retest_tolerance_bp: float = 10.0
    discovery_end_year: int = 2024
    forward_year: int = 2025
    holdout_year: int = 2026
    min_discovery_samples: int = 40
    quantile_bins: int = 5


SEMANTIC_CONTINUOUS_FEATURES: tuple[str, ...] = (
    "sweep_distance_pct",
    "semantic_terminal_extension_pct",
    "semantic_terminal_extension_vs_initial_sweep",
    "semantic_reclaim_minutes_from_sweep",
    "semantic_reclaim_minutes_from_terminal",
    "semantic_outside_liquidity_close_fraction_to_signal",
    "semantic_outside_liquidity_max_consecutive_closes",
    "semantic_terminal_retest_count_10bp",
    "semantic_reference_minus_terminal_minutes",
    "semantic_reference_age_at_mss_minutes",
    "sweep_to_terminal_minutes",
    "terminal_to_mss_minutes",
    "sweep_to_mss_minutes",
    "terminal_to_signal_minutes",
    "sweep_to_signal_minutes",
    "displacement_speed_ratio",
    "mss_outbound_speed_pct_per_min",
    "reversal_path_efficiency",
    "directional_body_share",
    "directional_bar_fraction",
    "max_directional_body_vs_pre20_median",
    "max_leg_range_vs_pre20_median",
    "mss_overshoot_pct",
    "fvg_size_pct",
    "fvg_size_vs_risk",
    "fvg_entry_depth_vs_mss_leg",
    "mss_to_fvg_minutes",
    "directional_fvg_count_at_mss",
    "directional_fvg_count_to_signal",
    "risk_pct",
    "planned_rr",
    "semantic_entry_progress_to_target",
)


def _as_ny_index(bars: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        raise ValueError("semantic gap bars must use a timezone-aware index")
    return idx


def _shape_from_source(source: object) -> str:
    text = str(source)
    if text == "post_terminal_dynamic":
        return "post_terminal_structure"
    if text == "post_sweep_pre_terminal_dynamic":
        return "post_sweep_pre_terminal_structure"
    if text == "pre_sweep_v_reference":
        return "direct_v_reference"
    return "other_reference"


def _analysis_period(values: pd.Series, cfg: SemanticGapConfig) -> pd.Series:
    years = pd.to_datetime(values, errors="coerce").dt.year
    return pd.Series(
        np.select(
            [years <= cfg.discovery_end_year, years == cfg.forward_year, years >= cfg.holdout_year],
            [f"discovery_through_{cfg.discovery_end_year}", str(cfg.forward_year), f"{cfg.holdout_year}_late_holdout"],
            default="other",
        ),
        index=values.index,
        dtype="object",
    )


def _max_consecutive(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def attach_causal_semantic_features(
    attempts: pd.DataFrame,
    bars_ny: pd.DataFrame,
    *,
    config: SemanticGapConfig = SemanticGapConfig(),
) -> pd.DataFrame:
    """Attach pre-signal semantic features without filtering any candidate."""
    if attempts.empty:
        return attempts.copy()
    idx = _as_ny_index(bars_ny)
    work = attempts.copy()

    # Cheap timestamp/geometry features first.
    sweep_time = pd.to_datetime(work["sweep_time"])
    terminal_time = pd.to_datetime(work["episode_terminal_extreme_time"])
    mss_time = pd.to_datetime(work["mss_time"])
    ref_time = pd.to_datetime(work["mss_reference_time"])
    level = pd.to_numeric(work["level_price"], errors="coerce")
    terminal = pd.to_numeric(work["episode_terminal_extreme_price"], errors="coerce")
    initial_extreme = pd.to_numeric(work["sweep_price_extreme_initial"], errors="coerce")
    entry = pd.to_numeric(work["fvg_near_edge_entry"], errors="coerce")
    target = pd.to_numeric(work["target_price"], errors="coerce")

    work["semantic_structure_shape"] = work["mss_reference_source"].map(_shape_from_source)
    work["semantic_reference_minus_terminal_minutes"] = (ref_time - terminal_time).dt.total_seconds() / 60.0
    work["semantic_reference_age_at_mss_minutes"] = (mss_time - ref_time).dt.total_seconds() / 60.0
    work["semantic_terminal_extension_pct"] = (terminal - level).abs() / level.abs().replace(0, np.nan)
    initial_sweep = (initial_extreme - level).abs()
    terminal_extension = (terminal - level).abs()
    work["semantic_terminal_extension_vs_initial_sweep"] = terminal_extension / initial_sweep.replace(0, np.nan)
    full_target_path = (target - terminal).abs()
    work["semantic_entry_progress_to_target"] = (entry - terminal).abs() / full_target_path.replace(0, np.nan)

    # Time-of-day is descriptive, not a gate.
    work["semantic_sweep_minute_ny"] = sweep_time.dt.hour * 60 + sweep_time.dt.minute
    work["semantic_mss_minute_ny"] = mss_time.dt.hour * 60 + mss_time.dt.minute
    work["semantic_signal_minute_ny"] = pd.to_datetime(work["signal_time"]).dt.hour * 60 + pd.to_datetime(work["signal_time"]).dt.minute

    # Per-day arrays make the 1m path diagnostics O(total bars + attempts*local window),
    # avoiding full-table scans for every setup.
    day_key = pd.Series(idx.date.astype(str), index=np.arange(len(idx)))
    day_bounds: dict[str, tuple[int, int]] = {}
    for day, positions in day_key.groupby(day_key, sort=False):
        arr = positions.index.to_numpy()
        day_bounds[str(day)] = (int(arr[0]), int(arr[-1]) + 1)

    closes_all = pd.to_numeric(bars_ny["close"], errors="coerce").to_numpy(float)
    highs_all = pd.to_numeric(bars_ny["high"], errors="coerce").to_numpy(float)
    lows_all = pd.to_numeric(bars_ny["low"], errors="coerce").to_numpy(float)
    available_1m = idx + pd.Timedelta(minutes=1)
    idx_ns = available_1m.as_unit("ns").asi8

    reclaim_from_sweep = np.full(len(work), np.nan)
    reclaim_from_terminal = np.full(len(work), np.nan)
    outside_fraction = np.full(len(work), np.nan)
    outside_max_run = np.full(len(work), np.nan)
    retest_count = np.full(len(work), np.nan)

    tol_frac = float(config.terminal_retest_tolerance_bp) / 10_000.0
    for out_i, row in enumerate(work.itertuples(index=False)):
        day = str(getattr(row, "ny_date"))
        bounds = day_bounds.get(day)
        if bounds is None:
            continue
        lo_bound, hi_bound = bounds
        sw = pd.Timestamp(getattr(row, "sweep_time"))
        sig = pd.Timestamp(getattr(row, "signal_time"))
        term = pd.Timestamp(getattr(row, "episode_terminal_extreme_time"))
        sw_pos = max(lo_bound, int(np.searchsorted(idx_ns, int(sw.value), side="left")))
        sig_pos = min(hi_bound - 1, int(np.searchsorted(idx_ns, int(sig.value), side="right") - 1))
        term_pos = max(lo_bound, int(np.searchsorted(idx_ns, int(term.value), side="left")))
        if sig_pos < sw_pos:
            continue
        level_px = float(getattr(row, "level_price"))
        terminal_px = float(getattr(row, "episode_terminal_extreme_price"))
        is_long = str(getattr(row, "trade_side")) == "LONG"
        closes = closes_all[sw_pos : sig_pos + 1]
        if is_long:
            outside = closes < level_px
            reclaimed = closes >= level_px
        else:
            outside = closes > level_px
            reclaimed = closes <= level_px
        finite = np.isfinite(closes)
        if finite.any():
            outside_fraction[out_i] = float(np.mean(outside[finite]))
            outside_max_run[out_i] = float(_max_consecutive(outside[finite]))
            where = np.flatnonzero(reclaimed & finite)
            if where.size:
                reclaim_ts = available_1m[sw_pos + int(where[0])]
                reclaim_from_sweep[out_i] = max(0.0, float((reclaim_ts - sw).total_seconds() / 60.0))
                reclaim_from_terminal[out_i] = float((reclaim_ts - term).total_seconds() / 60.0)

        # Diagnostic: how often price revisited within 10bp of the terminal after
        # the terminal print and before signal. This does not change eligibility.
        t0 = max(term_pos + 1, sw_pos)
        if t0 <= sig_pos and abs(terminal_px) > EPS:
            tol = abs(terminal_px) * tol_frac
            if is_long:
                near = lows_all[t0 : sig_pos + 1] <= terminal_px + tol
            else:
                near = highs_all[t0 : sig_pos + 1] >= terminal_px - tol
            retest_count[out_i] = float(np.sum(near & np.isfinite(near)))
        else:
            retest_count[out_i] = 0.0

    work["semantic_reclaim_minutes_from_sweep"] = reclaim_from_sweep
    work["semantic_reclaim_minutes_from_terminal"] = reclaim_from_terminal
    work["semantic_outside_liquidity_close_fraction_to_signal"] = outside_fraction
    work["semantic_outside_liquidity_max_consecutive_closes"] = outside_max_run
    work["semantic_terminal_retest_count_10bp"] = retest_count
    work["semantic_feature_available_time"] = pd.to_datetime(work["signal_time"])
    return work


def attach_outcome_path_labels(base_lifecycle: pd.DataFrame) -> pd.DataFrame:
    """Attach post-entry outcome labels for analysis only."""
    if base_lifecycle.empty:
        return base_lifecycle.copy()
    out = base_lifecycle.copy()
    mfe = pd.to_numeric(out.get("mfe_r"), errors="coerce")
    net_r = pd.to_numeric(out.get("net_r"), errors="coerce")
    gross_r = pd.to_numeric(out.get("gross_r"), errors="coerce")
    exit_reason = out.get("exit_reason", pd.Series(index=out.index, dtype="object")).astype(str)
    bars_held = pd.to_numeric(out.get("bars_held_1m"), errors="coerce")
    filled = out.get("filled", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    out["outcome_reached_0_5r"] = filled & (mfe >= 0.5)
    out["outcome_reached_1r"] = filled & (mfe >= 1.0)
    out["outcome_reached_2r"] = filled & (mfe >= 2.0)
    out["outcome_reached_3r"] = filled & (mfe >= 3.0)
    is_stop = exit_reason.str.contains("stop", case=False, na=False)
    is_target = exit_reason.str.contains("target", case=False, na=False)
    out["outcome_target_hit"] = filled & is_target
    out["outcome_stop_hit"] = filled & is_stop
    out["outcome_final_positive"] = filled & (net_r > 0)
    out["outcome_immediate_failure_15m"] = filled & is_stop & (bars_held <= 15) & (mfe < 0.5)
    out["outcome_favorable_then_failed_0_5r"] = filled & (mfe >= 0.5) & (net_r < 0)
    out["outcome_favorable_then_failed_1r"] = filled & (mfe >= 1.0) & (net_r < 0)
    out["outcome_favorable_then_failed_2r"] = filled & (mfe >= 2.0) & (net_r < 0)
    out["outcome_gross_positive_net_negative"] = filled & (gross_r > 0) & (net_r <= 0)
    return out


def _profit_factor(net_returns: pd.Series) -> float:
    x = pd.to_numeric(net_returns, errors="coerce").dropna()
    if x.empty:
        return np.nan
    wins = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= EPS:
        return np.inf if wins > 0 else np.nan
    return wins / losses


def summarize_outcome_group(group: pd.DataFrame) -> dict[str, object]:
    filled = group.loc[group.get("filled", False).fillna(False).astype(bool)].copy() if "filled" in group else group.copy()
    if filled.empty:
        return {"trades": 0}
    net = pd.to_numeric(filled["net_return"], errors="coerce")
    net_r = pd.to_numeric(filled.get("net_r"), errors="coerce")
    mfe = pd.to_numeric(filled.get("mfe_r"), errors="coerce")
    mae = pd.to_numeric(filled.get("mae_r"), errors="coerce")
    return {
        "trades": int(len(filled)),
        "win_rate": float((net > 0).mean()),
        "profit_factor": _profit_factor(net),
        "mean_net_return": float(net.mean()),
        "median_net_r": float(net_r.median()),
        "median_mfe_r": float(mfe.median()),
        "median_mae_r": float(mae.median()),
        "reached_0_5r_rate": float(filled["outcome_reached_0_5r"].mean()) if "outcome_reached_0_5r" in filled else np.nan,
        "reached_1r_rate": float(filled["outcome_reached_1r"].mean()) if "outcome_reached_1r" in filled else np.nan,
        "reached_2r_rate": float(filled["outcome_reached_2r"].mean()) if "outcome_reached_2r" in filled else np.nan,
        "reached_3r_rate": float(filled["outcome_reached_3r"].mean()) if "outcome_reached_3r" in filled else np.nan,
        "target_hit_rate": float(filled["outcome_target_hit"].mean()) if "outcome_target_hit" in filled else np.nan,
        "stop_hit_rate": float(filled["outcome_stop_hit"].mean()) if "outcome_stop_hit" in filled else np.nan,
        "immediate_failure_15m_rate": float(filled["outcome_immediate_failure_15m"].mean()) if "outcome_immediate_failure_15m" in filled else np.nan,
        "favorable_then_failed_1r_rate": float(filled["outcome_favorable_then_failed_1r"].mean()) if "outcome_favorable_then_failed_1r" in filled else np.nan,
        "favorable_then_failed_2r_rate": float(filled["outcome_favorable_then_failed_2r"].mean()) if "outcome_favorable_then_failed_2r" in filled else np.nan,
    }


def build_semantic_feature_atlas(
    lifecycle: pd.DataFrame,
    *,
    config: SemanticGapConfig = SemanticGapConfig(),
    features: Iterable[str] = SEMANTIC_CONTINUOUS_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build discovery-frozen quintile edges and out-of-sample performance."""
    if lifecycle.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    discovery_label = f"discovery_through_{config.discovery_end_year}"
    edge_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    probs = np.linspace(0.0, 1.0, int(config.quantile_bins) + 1)

    for (tf, family), fam in work.groupby(["execution_tf", "liquidity_family"], sort=True):
        disc = fam.loc[fam["analysis_period"] == discovery_label]
        for feature in features:
            if feature not in fam.columns:
                continue
            vals = pd.to_numeric(disc[feature], errors="coerce").dropna()
            if len(vals) < int(config.min_discovery_samples):
                continue
            edges = vals.quantile(probs).to_numpy(float)
            # Duplicate edges carry no information and make pd.cut unstable.
            unique = np.unique(edges[np.isfinite(edges)])
            if len(unique) < 3:
                continue
            internal = unique[1:-1]
            row: dict[str, object] = {
                "execution_tf": tf,
                "liquidity_family": family,
                "feature": feature,
                "discovery_samples": int(len(vals)),
                "bin_count": int(len(unique) - 1),
            }
            for i, x in enumerate(internal, start=1):
                row[f"edge_{i}"] = float(x)
            edge_rows.append(row)

            x = pd.to_numeric(fam[feature], errors="coerce")
            bins = np.concatenate(([-np.inf], internal, [np.inf]))
            labels = [f"B{i+1}" for i in range(len(bins) - 1)]
            bucket = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
            tmp = fam.assign(__bucket=bucket)
            for (period, b), g in tmp.dropna(subset=["__bucket"]).groupby(["analysis_period", "__bucket"], observed=True, sort=True):
                perf_rows.append({
                    "execution_tf": tf,
                    "liquidity_family": family,
                    "feature": feature,
                    "analysis_period": period,
                    "bucket": str(b),
                    "feature_median": float(pd.to_numeric(g[feature], errors="coerce").median()),
                    **summarize_outcome_group(g),
                })
    return pd.DataFrame(edge_rows), pd.DataFrame(perf_rows)


def build_semantic_category_atlas(
    lifecycle: pd.DataFrame,
    *,
    config: SemanticGapConfig = SemanticGapConfig(),
) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    dimensions = [
        "trade_side",
        "semantic_structure_shape",
        "mss_reference_source",
        "fvg_relation_to_mss",
        "htf_confluence_count",
    ]
    rows: list[dict[str, object]] = []
    for dim in dimensions:
        if dim not in work.columns:
            continue
        for (tf, family, period, value), g in work.groupby(
            ["execution_tf", "liquidity_family", "analysis_period", dim],
            dropna=False, sort=True,
        ):
            rows.append({
                "dimension": dim,
                "value": str(value),
                "execution_tf": tf,
                "liquidity_family": family,
                "analysis_period": period,
                **summarize_outcome_group(g),
            })
    return pd.DataFrame(rows)


def build_mfe_transition_table(lifecycle: pd.DataFrame, *, config: SemanticGapConfig = SemanticGapConfig()) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    thresholds = [(0.5, "0.5R"), (1.0, "1R"), (2.0, "2R"), (3.0, "3R")]
    rows: list[dict[str, object]] = []
    mfe = pd.to_numeric(work["mfe_r"], errors="coerce")
    for threshold, label in thresholds:
        reached = work.loc[mfe >= threshold].copy()
        for (tf, family, period), g in reached.groupby(["execution_tf", "liquidity_family", "analysis_period"], sort=True):
            rows.append({
                "execution_tf": tf,
                "liquidity_family": family,
                "analysis_period": period,
                "mfe_threshold": label,
                "reached_trades": int(len(g)),
                "final_positive_rate": float(g["outcome_final_positive"].mean()),
                "eventual_stop_rate": float(g["outcome_stop_hit"].mean()),
                "target_hit_rate": float(g["outcome_target_hit"].mean()),
                "median_final_net_r": float(pd.to_numeric(g["net_r"], errors="coerce").median()),
                "median_mfe_r": float(pd.to_numeric(g["mfe_r"], errors="coerce").median()),
            })
    return pd.DataFrame(rows)


def build_entry_failure_atlas(lifecycle: pd.DataFrame, *, config: SemanticGapConfig = SemanticGapConfig()) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    labels = np.select(
        [
            work["outcome_immediate_failure_15m"].fillna(False),
            work["outcome_favorable_then_failed_1r"].fillna(False),
            work["outcome_target_hit"].fillna(False),
            work["outcome_final_positive"].fillna(False),
        ],
        ["immediate_failure", "reached_1r_then_failed", "target_hit", "positive_other"],
        default="negative_other",
    )
    work["outcome_path_class"] = labels
    rows: list[dict[str, object]] = []
    for (tf, family, period, cls), g in work.groupby(
        ["execution_tf", "liquidity_family", "analysis_period", "outcome_path_class"], sort=True
    ):
        row = {
            "execution_tf": tf,
            "liquidity_family": family,
            "analysis_period": period,
            "outcome_path_class": cls,
            "trades": int(len(g)),
        }
        for feature in (
            "semantic_terminal_extension_pct",
            "semantic_reclaim_minutes_from_terminal",
            "semantic_outside_liquidity_close_fraction_to_signal",
            "semantic_terminal_retest_count_10bp",
            "terminal_to_mss_minutes",
            "reversal_path_efficiency",
            "mss_overshoot_pct",
            "fvg_entry_depth_vs_mss_leg",
            "planned_rr",
        ):
            if feature in g.columns:
                row[f"median_{feature}"] = float(pd.to_numeric(g[feature], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_semantic_causal_audit(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame([{"check": "semantic_attempts_non_empty", "passed": False, "violations": 0}])
    rows: list[dict[str, object]] = []
    signal = pd.to_datetime(attempts["signal_time"])
    feat_time = pd.to_datetime(attempts["semantic_feature_available_time"])
    bad = int((feat_time > signal).fillna(True).sum())
    rows.append({
        "check": "semantic_features_available_by_signal",
        "passed": bad == 0,
        "violations": bad,
        "detail": "all semantic entry features must use data available no later than signal_time",
    })
    # Outcome labels must not exist on the signal-attempt frame.
    leakage_cols = [c for c in attempts.columns if c.startswith("outcome_")]
    rows.append({
        "check": "no_outcome_columns_in_signal_attempts",
        "passed": len(leakage_cols) == 0,
        "violations": len(leakage_cols),
        "detail": ",".join(leakage_cols),
    })
    return pd.DataFrame(rows)
