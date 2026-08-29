"""Liquidity-consumption maturity diagnostics for ICT Sweep -> MSS -> FVG research.

R08 deliberately does *not* gate entries.  It describes how price consumes a
liquidity level before the MSS is confirmed so research can distinguish, for
example, a fast spike-and-reclaim, a shallow equal-high/low probe followed by a
sweep, and a progressive extension beyond the level.

All ``maturity_`` features are computed from completed 1-minute bars available
no later than ``signal_time``.  Frozen discovery bins are learned only from the
configured discovery period and are then reused unchanged in forward/holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict.semantic_gap import summarize_outcome_group

EPS = 1e-12


@dataclass(frozen=True)
class LiquidityMaturityConfig:
    discovery_end_date: str = "2024-12-31"
    forward_start_date: str = "2025-01-01"
    forward_end_date: str = "2025-12-31"
    holdout_start_date: str = "2026-01-01"
    presweep_lookback_minutes: int = 60
    near_touch_tolerance_bp: float = 10.0
    min_discovery_samples: int = 40
    quantile_bins: int = 5


MATURITY_CONTINUOUS_FEATURES: tuple[str, ...] = (
    "sweep_distance_pct",
    "maturity_initial_sweep_depth_bp",
    "maturity_terminal_extension_bp",
    "maturity_sweep_to_terminal_minutes",
    "maturity_progressive_extreme_count",
    "maturity_pre_sweep_near_touch_count",
    "maturity_minutes_since_last_pre_sweep_near_touch",
    "maturity_first_reclaim_after_initial_sweep_minutes",
    "maturity_first_reclaim_after_final_terminal_minutes",
    "maturity_outside_minutes_after_terminal_to_reclaim_or_signal",
    "maturity_outside_close_fraction_sweep_to_signal",
    "maturity_outside_close_fraction_terminal_to_signal",
    "maturity_max_consecutive_outside_closes_sweep_to_signal",
    "maturity_max_consecutive_outside_closes_terminal_to_signal",
    "maturity_penetration_area_bp_minutes_sweep_to_signal",
    "maturity_penetration_area_bp_minutes_terminal_to_signal",
    "maturity_max_penetration_bp_sweep_to_signal",
    "maturity_terminal_retest_count_10bp",
    "terminal_to_mss_minutes",
    "sweep_to_mss_minutes",
    "semantic_reference_age_at_mss_minutes",
    "mss_overshoot_pct",
    "fvg_entry_depth_vs_mss_leg",
    "reversal_path_efficiency",
    "displacement_speed_ratio",
)


def _as_ny_index(bars: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        raise ValueError("liquidity maturity bars require timezone-aware DatetimeIndex")
    return idx


def _max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask.astype(bool):
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _analysis_period(values: pd.Series, cfg: LiquidityMaturityConfig) -> pd.Series:
    d = pd.to_datetime(values, errors="coerce").dt.tz_localize(None).dt.normalize()
    disc_end = pd.Timestamp(cfg.discovery_end_date)
    fwd_start = pd.Timestamp(cfg.forward_start_date)
    fwd_end = pd.Timestamp(cfg.forward_end_date)
    hold_start = pd.Timestamp(cfg.holdout_start_date)
    return pd.Series(
        np.select(
            [d <= disc_end, (d >= fwd_start) & (d <= fwd_end), d >= hold_start],
            [f"discovery_through_{disc_end.date()}", "2025_forward", "2026_late_holdout"],
            default="other",
        ),
        index=values.index,
        dtype="object",
    )


def attach_liquidity_maturity_features(
    attempts: pd.DataFrame,
    bars_ny: pd.DataFrame,
    *,
    config: LiquidityMaturityConfig = LiquidityMaturityConfig(),
) -> pd.DataFrame:
    """Attach causal consumption-path features without filtering candidates."""
    if attempts.empty:
        return attempts.copy()
    idx = _as_ny_index(bars_ny)
    work = attempts.copy()

    day_key = pd.Series(idx.date.astype(str), index=np.arange(len(idx)))
    day_bounds: dict[str, tuple[int, int]] = {}
    for day, positions in day_key.groupby(day_key, sort=False):
        arr = positions.index.to_numpy()
        day_bounds[str(day)] = (int(arr[0]), int(arr[-1]) + 1)

    available = idx + pd.Timedelta(minutes=1)
    available_ns = available.as_unit("ns").asi8
    highs = pd.to_numeric(bars_ny["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(bars_ny["low"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(bars_ny["close"], errors="coerce").to_numpy(float)

    n = len(work)
    feature_arrays = {name: np.full(n, np.nan) for name in (
        "maturity_initial_sweep_depth_bp",
        "maturity_terminal_extension_bp",
        "maturity_sweep_to_terminal_minutes",
        "maturity_progressive_extreme_count",
        "maturity_pre_sweep_near_touch_count",
        "maturity_minutes_since_last_pre_sweep_near_touch",
        "maturity_first_reclaim_after_initial_sweep_minutes",
        "maturity_first_reclaim_after_final_terminal_minutes",
        "maturity_outside_minutes_after_terminal_to_reclaim_or_signal",
        "maturity_outside_close_fraction_sweep_to_signal",
        "maturity_outside_close_fraction_terminal_to_signal",
        "maturity_max_consecutive_outside_closes_sweep_to_signal",
        "maturity_max_consecutive_outside_closes_terminal_to_signal",
        "maturity_penetration_area_bp_minutes_sweep_to_signal",
        "maturity_penetration_area_bp_minutes_terminal_to_signal",
        "maturity_max_penetration_bp_sweep_to_signal",
        "maturity_terminal_retest_count_10bp",
    )}

    tol_frac = float(config.near_touch_tolerance_bp) / 10_000.0
    lookback = int(config.presweep_lookback_minutes)

    for out_i, row in enumerate(work.itertuples(index=False)):
        day = str(getattr(row, "ny_date"))
        bounds = day_bounds.get(day)
        if bounds is None:
            continue
        lo_bound, hi_bound = bounds
        sw = pd.Timestamp(getattr(row, "sweep_time"))
        term = pd.Timestamp(getattr(row, "episode_terminal_extreme_time"))
        sig = pd.Timestamp(getattr(row, "signal_time"))
        level = float(getattr(row, "level_price"))
        initial = float(getattr(row, "sweep_price_extreme_initial"))
        terminal = float(getattr(row, "episode_terminal_extreme_price"))
        if not np.isfinite(level) or abs(level) <= EPS:
            continue
        is_long = str(getattr(row, "trade_side")) == "LONG"

        sw_pos = max(lo_bound, int(np.searchsorted(available_ns, int(sw.value), side="left")))
        term_pos = max(lo_bound, int(np.searchsorted(available_ns, int(term.value), side="left")))
        sig_pos = min(hi_bound - 1, int(np.searchsorted(available_ns, int(sig.value), side="right") - 1))
        if sig_pos < sw_pos:
            continue
        term_pos = min(max(term_pos, sw_pos), sig_pos)

        feature_arrays["maturity_initial_sweep_depth_bp"][out_i] = abs(initial - level) / abs(level) * 10_000.0
        feature_arrays["maturity_terminal_extension_bp"][out_i] = abs(terminal - level) / abs(level) * 10_000.0
        feature_arrays["maturity_sweep_to_terminal_minutes"][out_i] = max(0.0, (term - sw).total_seconds() / 60.0)

        # Pre-sweep equal/probe context: bars that approached the level closely
        # from the unswept side without already trading through it.
        pre0 = max(lo_bound, sw_pos - lookback)
        if pre0 < sw_pos:
            if is_long:
                d = (lows[pre0:sw_pos] - level) / abs(level)
                near = (d >= 0.0) & (d <= tol_frac)
            else:
                d = (level - highs[pre0:sw_pos]) / abs(level)
                near = (d >= 0.0) & (d <= tol_frac)
            near &= np.isfinite(d)
            loc = np.flatnonzero(near)
            feature_arrays["maturity_pre_sweep_near_touch_count"][out_i] = float(loc.size)
            if loc.size:
                last_pos = pre0 + int(loc[-1])
                feature_arrays["maturity_minutes_since_last_pre_sweep_near_touch"][out_i] = max(
                    0.0, (sw - available[last_pos]).total_seconds() / 60.0
                )
            else:
                feature_arrays["maturity_minutes_since_last_pre_sweep_near_touch"][out_i] = np.nan
        else:
            feature_arrays["maturity_pre_sweep_near_touch_count"][out_i] = 0.0

        sl = slice(sw_pos, sig_pos + 1)
        c = closes[sl]
        h = highs[sl]
        l = lows[sl]
        if is_long:
            penetration_bp = np.maximum(0.0, (level - l) / abs(level) * 10_000.0)
            outside = c < level
            reclaimed = c >= level
            running = np.minimum.accumulate(np.where(np.isfinite(l), l, np.inf))
            raw_extreme = np.isfinite(l) & (l <= running + EPS)
        else:
            penetration_bp = np.maximum(0.0, (h - level) / abs(level) * 10_000.0)
            outside = c > level
            reclaimed = c <= level
            running = np.maximum.accumulate(np.where(np.isfinite(h), h, -np.inf))
            raw_extreme = np.isfinite(h) & (h >= running - EPS)
        finite_c = np.isfinite(c)
        penetration_bp = np.where(np.isfinite(penetration_bp), penetration_bp, 0.0)
        feature_arrays["maturity_penetration_area_bp_minutes_sweep_to_signal"][out_i] = float(penetration_bp.sum())
        feature_arrays["maturity_max_penetration_bp_sweep_to_signal"][out_i] = float(penetration_bp.max(initial=0.0))
        if finite_c.any():
            feature_arrays["maturity_outside_close_fraction_sweep_to_signal"][out_i] = float(np.mean(outside[finite_c]))
            feature_arrays["maturity_max_consecutive_outside_closes_sweep_to_signal"][out_i] = float(_max_consecutive(outside[finite_c]))
            where = np.flatnonzero(reclaimed & finite_c)
            if where.size:
                reclaim_ts = available[sw_pos + int(where[0])]
                feature_arrays["maturity_first_reclaim_after_initial_sweep_minutes"][out_i] = max(
                    0.0, (reclaim_ts - sw).total_seconds() / 60.0
                )

        # Count *new* progressive extrema after the initial sweep, excluding
        # equal prints. This is descriptive only; it is not an entry gate.
        extrema = l if is_long else h
        best = initial
        count = 0
        for value in extrema[1:]:
            if not np.isfinite(value):
                continue
            if (is_long and value < best - EPS) or ((not is_long) and value > best + EPS):
                count += 1
                best = float(value)
        feature_arrays["maturity_progressive_extreme_count"][out_i] = float(count)

        # Final-terminal semantics start strictly no earlier than the terminal
        # print. This fixes R07's ambiguous negative reclaim-time diagnostic.
        term_rel = term_pos - sw_pos
        c2 = c[term_rel:]
        p2 = penetration_bp[term_rel:]
        finite2 = np.isfinite(c2)
        if is_long:
            outside2 = c2 < level
            reclaimed2 = c2 >= level
        else:
            outside2 = c2 > level
            reclaimed2 = c2 <= level
        feature_arrays["maturity_penetration_area_bp_minutes_terminal_to_signal"][out_i] = float(p2.sum())
        if finite2.any():
            feature_arrays["maturity_outside_close_fraction_terminal_to_signal"][out_i] = float(np.mean(outside2[finite2]))
            feature_arrays["maturity_max_consecutive_outside_closes_terminal_to_signal"][out_i] = float(_max_consecutive(outside2[finite2]))
            where2 = np.flatnonzero(reclaimed2 & finite2)
            if where2.size:
                reclaim_ts2 = available[term_pos + int(where2[0])]
                mins = max(0.0, (reclaim_ts2 - term).total_seconds() / 60.0)
                feature_arrays["maturity_first_reclaim_after_final_terminal_minutes"][out_i] = mins
                feature_arrays["maturity_outside_minutes_after_terminal_to_reclaim_or_signal"][out_i] = mins
            else:
                feature_arrays["maturity_outside_minutes_after_terminal_to_reclaim_or_signal"][out_i] = max(
                    0.0, (sig - term).total_seconds() / 60.0
                )

        tol = abs(terminal) * tol_frac
        post_t0 = min(sig_pos + 1, term_pos + 1)
        if post_t0 <= sig_pos and np.isfinite(terminal):
            if is_long:
                near_terminal = lows[post_t0 : sig_pos + 1] <= terminal + tol
            else:
                near_terminal = highs[post_t0 : sig_pos + 1] >= terminal - tol
            feature_arrays["maturity_terminal_retest_count_10bp"][out_i] = float(np.sum(near_terminal & np.isfinite(near_terminal)))
        else:
            feature_arrays["maturity_terminal_retest_count_10bp"][out_i] = 0.0

    for name, values in feature_arrays.items():
        work[name] = values
    work["maturity_feature_available_time"] = pd.to_datetime(work["signal_time"])
    return work


def build_maturity_causal_audit(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame([{"check": "maturity_attempts_non_empty", "passed": False, "violations": 0}])
    sig = pd.to_datetime(attempts["signal_time"])
    feat = pd.to_datetime(attempts["maturity_feature_available_time"])
    bad = int((feat > sig).fillna(True).sum())
    neg_final = pd.to_numeric(attempts.get("maturity_first_reclaim_after_final_terminal_minutes"), errors="coerce")
    neg_count = int((neg_final < 0).fillna(False).sum())
    leakage_cols = [c for c in attempts.columns if c.startswith("outcome_")]
    return pd.DataFrame([
        {"check": "maturity_features_available_by_signal", "passed": bad == 0, "violations": bad},
        {"check": "final_terminal_reclaim_non_negative", "passed": neg_count == 0, "violations": neg_count},
        {"check": "no_outcome_columns_in_maturity_attempts", "passed": len(leakage_cols) == 0, "violations": len(leakage_cols)},
    ])


def build_maturity_feature_atlas(
    lifecycle: pd.DataFrame,
    *,
    config: LiquidityMaturityConfig = LiquidityMaturityConfig(),
    features: Iterable[str] = MATURITY_CONTINUOUS_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discovery-frozen one-dimensional response curves."""
    if lifecycle.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    disc_label = f"discovery_through_{pd.Timestamp(config.discovery_end_date).date()}"
    probs = np.linspace(0.0, 1.0, int(config.quantile_bins) + 1)
    edge_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    for (tf, family), fam in work.groupby(["execution_tf", "liquidity_family"], sort=True):
        disc = fam.loc[fam["analysis_period"] == disc_label]
        for feature in features:
            if feature not in fam.columns:
                continue
            vals = pd.to_numeric(disc[feature], errors="coerce").dropna()
            if len(vals) < int(config.min_discovery_samples):
                continue
            edges = vals.quantile(probs).to_numpy(float)
            unique = np.unique(edges[np.isfinite(edges)])
            if len(unique) < 3:
                continue
            internal = unique[1:-1]
            edge_row: dict[str, object] = {
                "execution_tf": tf, "liquidity_family": family, "feature": feature,
                "discovery_samples": int(len(vals)), "bin_count": int(len(unique) - 1),
            }
            for i, x in enumerate(internal, 1):
                edge_row[f"edge_{i}"] = float(x)
            edge_rows.append(edge_row)
            x = pd.to_numeric(fam[feature], errors="coerce")
            bins = np.concatenate(([-np.inf], internal, [np.inf]))
            labels = [f"B{i+1}" for i in range(len(bins) - 1)]
            tmp = fam.assign(__bucket=pd.cut(x, bins=bins, labels=labels, include_lowest=True))
            for (period, bucket), g in tmp.dropna(subset=["__bucket"]).groupby(["analysis_period", "__bucket"], observed=True, sort=True):
                perf_rows.append({
                    "execution_tf": tf, "liquidity_family": family, "feature": feature,
                    "analysis_period": period, "bucket": str(bucket),
                    "feature_median": float(pd.to_numeric(g[feature], errors="coerce").median()),
                    **summarize_outcome_group(g),
                })
    return pd.DataFrame(edge_rows), pd.DataFrame(perf_rows)


PAIR_FEATURES: tuple[tuple[str, str], ...] = (
    ("maturity_initial_sweep_depth_bp", "maturity_first_reclaim_after_final_terminal_minutes"),
    ("maturity_terminal_extension_bp", "maturity_first_reclaim_after_final_terminal_minutes"),
    ("maturity_pre_sweep_near_touch_count", "maturity_initial_sweep_depth_bp"),
    ("maturity_pre_sweep_near_touch_count", "maturity_first_reclaim_after_final_terminal_minutes"),
    ("maturity_penetration_area_bp_minutes_sweep_to_signal", "terminal_to_mss_minutes"),
    ("maturity_progressive_extreme_count", "maturity_first_reclaim_after_final_terminal_minutes"),
)


def build_maturity_pair_atlas(
    lifecycle: pd.DataFrame,
    *,
    config: LiquidityMaturityConfig = LiquidityMaturityConfig(),
    pairs: Iterable[tuple[str, str]] = PAIR_FEATURES,
) -> pd.DataFrame:
    """Frozen 2D response surfaces to reveal multiple profitable path families.

    Only 3 bins per axis are used to avoid exploding the sample into tiny cells.
    Edges are learned from discovery only and frozen for forward/holdout.
    """
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    disc_label = f"discovery_through_{pd.Timestamp(config.discovery_end_date).date()}"
    rows: list[dict[str, object]] = []
    for (tf, family), fam in work.groupby(["execution_tf", "liquidity_family"], sort=True):
        disc = fam.loc[fam["analysis_period"] == disc_label]
        for fx, fy in pairs:
            if fx not in fam or fy not in fam:
                continue
            dx = pd.to_numeric(disc[fx], errors="coerce")
            dy = pd.to_numeric(disc[fy], errors="coerce")
            valid = dx.notna() & dy.notna()
            if int(valid.sum()) < max(int(config.min_discovery_samples), 60):
                continue
            ex = np.unique(dx[valid].quantile([0.0, 1/3, 2/3, 1.0]).to_numpy(float))
            ey = np.unique(dy[valid].quantile([0.0, 1/3, 2/3, 1.0]).to_numpy(float))
            if len(ex) < 3 or len(ey) < 3:
                continue
            bx = np.concatenate(([-np.inf], ex[1:-1], [np.inf]))
            by = np.concatenate(([-np.inf], ey[1:-1], [np.inf]))
            tx = pd.cut(pd.to_numeric(fam[fx], errors="coerce"), bins=bx, labels=[f"X{i+1}" for i in range(len(bx)-1)], include_lowest=True)
            ty = pd.cut(pd.to_numeric(fam[fy], errors="coerce"), bins=by, labels=[f"Y{i+1}" for i in range(len(by)-1)], include_lowest=True)
            tmp = fam.assign(__x=tx, __y=ty)
            for (period, xb, yb), g in tmp.dropna(subset=["__x", "__y"]).groupby(["analysis_period", "__x", "__y"], observed=True, sort=True):
                rows.append({
                    "execution_tf": tf, "liquidity_family": family,
                    "feature_x": fx, "feature_y": fy,
                    "analysis_period": period, "x_bucket": str(xb), "y_bucket": str(yb),
                    "x_median": float(pd.to_numeric(g[fx], errors="coerce").median()),
                    "y_median": float(pd.to_numeric(g[fy], errors="coerce").median()),
                    **summarize_outcome_group(g),
                })
    return pd.DataFrame(rows)


def build_opportunity_frequency_atlas(
    lifecycle: pd.DataFrame,
    valid_days: Iterable[object],
    *,
    config: LiquidityMaturityConfig = LiquidityMaturityConfig(),
) -> pd.DataFrame:
    """Report how broad the actual base universe is, separate from profitable bins."""
    days = pd.Series([str(pd.Timestamp(d).date()) for d in valid_days], dtype="object")
    period_by_day = _analysis_period(days, config)
    sessions = pd.DataFrame({"ny_date": days, "analysis_period": period_by_day})
    session_counts = sessions.groupby("analysis_period", sort=True).size().to_dict()
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle.loc[lifecycle.get("filled", False).fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _analysis_period(work["ny_date"], config)
    rows: list[dict[str, object]] = []
    for (tf, family, period), g in work.groupby(["execution_tf", "liquidity_family", "analysis_period"], sort=True):
        n_sessions = int(session_counts.get(period, 0))
        active_days = int(pd.Series(g["ny_date"].astype(str).unique()).nunique())
        # attempt_id/event_id remain useful diagnostics even after single-lifecycle replay.
        unique_events = int(g["event_id"].astype(str).nunique()) if "event_id" in g else int(len(g))
        rows.append({
            "execution_tf": tf, "liquidity_family": family, "analysis_period": period,
            "sessions": n_sessions, "filled_trades": int(len(g)), "unique_liquidity_events": unique_events,
            "days_with_filled_trade": active_days,
            "trades_per_session": float(len(g) / n_sessions) if n_sessions else np.nan,
            "sessions_per_trade": float(n_sessions / len(g)) if len(g) else np.nan,
            "opportunity_day_rate": float(active_days / n_sessions) if n_sessions else np.nan,
        })
    return pd.DataFrame(rows)
