#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structural overlap audit helpers for SOXL spot vs perpetual proxies.

The audit is descriptive and causal-neutral: it compares already completed 1m
paths and independently generated ICT events from each source.  It never maps
future information from one source into the other source's signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg import (
    NY_TZ,
    PREMARKET_END,
    PREMARKET_START,
    TRADE_END,
    TRADE_START,
    slice_ny_day,
)


@dataclass(frozen=True)
class ProxyAuditThresholds:
    """Pre-declared engineering gates; not tuned on strategy PnL."""

    aligned_minute_coverage: float = 0.95
    return_correlation: float = 0.97
    median_daily_rebased_path_correlation: float = 0.99
    external_sweep_key_jaccard: float = 0.80
    base_setup_key_jaccard: float = 0.65


@dataclass(frozen=True)
class ProxyAuditCautionThresholds:
    aligned_minute_coverage: float = 0.90
    return_correlation: float = 0.95
    median_daily_rebased_path_correlation: float = 0.97
    external_sweep_key_jaccard: float = 0.65
    base_setup_key_jaccard: float = 0.50


def ensure_ny_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        raise ValueError("overlap audit expects timezone-aware indexes")
    out.index = idx.tz_convert(NY_TZ)
    out.index.name = "bar_start_ny"
    return out.sort_index()


def clip_equity_research_session(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_ny_index(frame)
    if out.empty:
        return out
    idx = pd.DatetimeIndex(out.index)
    mins = idx.hour * 60 + idx.minute
    weekday = idx.weekday < 5
    session = (mins >= 4 * 60) & (mins < 16 * 60 + 30)
    return out.loc[np.asarray(weekday & session)].copy()



def densify_equity_minutes_causally(frame: pd.DataFrame) -> pd.DataFrame:
    """Causally fill internal no-trade minutes with the last observed close.

    US-stock aggregate APIs can omit a minute when no eligible trade occurs.
    Treating those omissions as missing *time* breaks 2m/5m/15m aggregation.
    This function only carries the last already-observed close forward within
    the same New York day and only through the day's last observed bar. It never
    backfills before the first trade and never fills the tail after the last
    trade, so early closes/data truncation remain visible to quality checks.
    """
    src = clip_equity_research_session(frame)
    if src.empty:
        return src
    parts: list[pd.DataFrame] = []
    for day, grp in src.groupby([ts.date() for ts in src.index], sort=True):
        grp = grp.sort_index().copy()
        start = pd.Timestamp(day).tz_localize(NY_TZ) + pd.Timedelta(hours=4)
        end = pd.Timestamp(day).tz_localize(NY_TZ) + pd.Timedelta(hours=16, minutes=29)
        expected = pd.date_range(start, end, freq="1min", tz=NY_TZ)
        observed_index = pd.DatetimeIndex(grp.index)
        work = grp.reindex(expected)
        observed = work.index.isin(observed_index)
        work["is_observed_bar"] = observed
        previous_close = pd.to_numeric(work["close"], errors="coerce").ffill()
        last_observed = observed_index.max()
        internal_missing = (~observed) & previous_close.notna().to_numpy() & (work.index <= last_observed)
        for col in ("open", "high", "low", "close"):
            if col in work.columns:
                values = pd.to_numeric(work[col], errors="coerce")
                values.loc[internal_missing] = previous_close.loc[internal_missing]
                work[col] = values
        for col in ("volume", "trade_count"):
            if col in work.columns:
                values = pd.to_numeric(work[col], errors="coerce")
                values.loc[internal_missing] = 0.0
                work[col] = values
        if "vwap" in work.columns:
            values = pd.to_numeric(work["vwap"], errors="coerce")
            values.loc[internal_missing] = previous_close.loc[internal_missing]
            work["vwap"] = values
        work["is_synthetic_no_trade_bar"] = internal_missing
        work = work.dropna(subset=[c for c in ("open", "high", "low", "close") if c in work.columns])
        parts.append(work)
    out = pd.concat(parts).sort_index() if parts else pd.DataFrame()
    out.index.name = "bar_start_ny"
    return out


def build_equity_proxy_data_quality_table(
    bars_ny: pd.DataFrame,
    days: Sequence,
    *,
    max_trade_observed_gap_minutes: float = 10.0,
    min_last_observed_minute: int = 16 * 60 + 25,
) -> pd.DataFrame:
    """Quality gate aware of omitted zero-trade stock minutes.

    Premarket can legitimately be sparse. We therefore do not require 270 raw
    bars. Instead, at least one real premarket print must exist, the active
    trading window must remain observed through ~16:25, and no large (>10m) raw
    gap may occur after 08:30. Internal no-trade minutes can be causally carried
    by ``densify_equity_minutes_causally`` but remain auditable.
    """
    rows: list[dict[str, object]] = []
    for day in days:
        pre = slice_ny_day(bars_ny, day, PREMARKET_START, PREMARKET_END)
        trade = slice_ny_day(bars_ny, day, TRADE_START, TRADE_END)
        pre_obs = pre.loc[pre.get("is_observed_bar", pd.Series(True, index=pre.index)).fillna(False).astype(bool)].copy() if not pre.empty else pre
        trade_obs = trade.loc[trade.get("is_observed_bar", pd.Series(True, index=trade.index)).fillna(False).astype(bool)].copy() if not trade.empty else trade
        if len(trade_obs) >= 2:
            gaps = pd.Series(pd.DatetimeIndex(trade_obs.index)).diff().dt.total_seconds().div(60.0)
            max_gap = float(gaps.max())
        else:
            max_gap = float("inf")
        last_obs = pd.Timestamp(trade_obs.index.max()) if len(trade_obs) else pd.NaT
        last_minute = int(last_obs.hour * 60 + last_obs.minute) if pd.notna(last_obs) else -1
        coverage_pass = bool(
            len(pre_obs) > 0
            and len(trade_obs) > 0
            and max_gap <= float(max_trade_observed_gap_minutes)
            and last_minute >= int(min_last_observed_minute)
        )
        rows.append(
            {
                "ny_date": str(day),
                "weekday": day.strftime("%A"),
                "premarket_rows_dense": int(len(pre)),
                "premarket_observed_rows": int(len(pre_obs)),
                "premarket_synthetic_rows": int(pre.get("is_synthetic_no_trade_bar", pd.Series(False, index=pre.index)).fillna(False).sum()) if not pre.empty else 0,
                "trade_rows_dense": int(len(trade)),
                "trade_observed_rows": int(len(trade_obs)),
                "trade_synthetic_rows": int(trade.get("is_synthetic_no_trade_bar", pd.Series(False, index=trade.index)).fillna(False).sum()) if not trade.empty else 0,
                "max_trade_observed_gap_minutes": max_gap,
                "first_premarket_observed": pre_obs.index.min() if len(pre_obs) else pd.NaT,
                "last_trade_observed": last_obs,
                "coverage_pass": coverage_pass,
            }
        )
    return pd.DataFrame(rows)

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if float(pair.iloc[:, 0].std(ddof=0)) == 0 or float(pair.iloc[:, 1].std(ddof=0)) == 0:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def build_aligned_minute_paths(spot_ny: pd.DataFrame, perp_ny: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot = clip_equity_research_session(spot_ny)
    perp = clip_equity_research_session(perp_ny)
    cols = ["open", "high", "low", "close"]
    spot_px = spot[cols].rename(columns=lambda c: f"spot_{c}")
    perp_px = perp[cols].rename(columns=lambda c: f"perp_{c}")
    aligned = spot_px.join(perp_px, how="inner").sort_index()
    if aligned.empty:
        return aligned, pd.DataFrame()

    aligned["spot_ret_1m"] = np.log(pd.to_numeric(aligned["spot_close"], errors="coerce")).diff()
    aligned["perp_ret_1m"] = np.log(pd.to_numeric(aligned["perp_close"], errors="coerce")).diff()
    aligned["basis_ratio"] = pd.to_numeric(aligned["perp_close"], errors="coerce") / pd.to_numeric(aligned["spot_close"], errors="coerce")
    aligned["ny_date"] = [str(ts.date()) for ts in aligned.index]

    daily_rows: list[dict[str, object]] = []
    for day, grp in aligned.groupby("ny_date", sort=True):
        grp = grp.dropna(subset=["spot_close", "perp_close"])
        if grp.empty:
            continue
        s0 = float(grp["spot_close"].iloc[0])
        p0 = float(grp["perp_close"].iloc[0])
        spot_path = np.log(pd.to_numeric(grp["spot_close"], errors="coerce") / s0)
        perp_path = np.log(pd.to_numeric(grp["perp_close"], errors="coerce") / p0)
        daily_rows.append(
            {
                "ny_date": day,
                "aligned_rows": int(len(grp)),
                "rebased_path_corr": _safe_corr(spot_path, perp_path),
                "median_abs_rebased_path_diff_bps": float(np.nanmedian(np.abs((perp_path - spot_path).to_numpy(float))) * 10_000),
                "median_basis_ratio": float(pd.to_numeric(grp["basis_ratio"], errors="coerce").median()),
                "basis_ratio_iqr_bps": float(
                    (pd.to_numeric(grp["basis_ratio"], errors="coerce").quantile(0.75)
                     - pd.to_numeric(grp["basis_ratio"], errors="coerce").quantile(0.25)) * 10_000
                ),
            }
        )
    return aligned, pd.DataFrame(daily_rows)


def _key_set(frame: pd.DataFrame, cols: Sequence[str]) -> set[tuple[str, ...]]:
    if frame.empty:
        return set()
    work = frame.copy()
    for col in cols:
        if col not in work.columns:
            work[col] = ""
    return {tuple(str(row[col]) for col in cols) for _, row in work[list(cols)].drop_duplicates().iterrows()}


def jaccard_keys(left: pd.DataFrame, right: pd.DataFrame, cols: Sequence[str]) -> tuple[float, int, int, int]:
    a = _key_set(left, cols)
    b = _key_set(right, cols)
    union = a | b
    inter = a & b
    score = float(len(inter) / len(union)) if union else float("nan")
    return score, len(inter), len(a), len(b)


def pair_unique_events(
    spot: pd.DataFrame,
    perp: pd.DataFrame,
    *,
    keys: Sequence[str],
    spot_time_col: str,
    perp_time_col: str | None = None,
    spot_prefix: str = "spot_",
    perp_prefix: str = "perp_",
) -> pd.DataFrame:
    """Pair one-row-per-key event tables and retain time differences."""

    perp_time_col = perp_time_col or spot_time_col
    if spot.empty or perp.empty:
        return pd.DataFrame()
    s = spot.copy()
    p = perp.copy()
    for col in keys:
        s[col] = s[col].astype(str)
        p[col] = p[col].astype(str)
    s = s.sort_values(list(keys) + [spot_time_col]).drop_duplicates(list(keys), keep="first")
    p = p.sort_values(list(keys) + [perp_time_col]).drop_duplicates(list(keys), keep="first")
    s_keep = list(keys) + [spot_time_col]
    p_keep = list(keys) + [perp_time_col]
    out = s[s_keep].rename(columns={spot_time_col: f"{spot_prefix}{spot_time_col}"}).merge(
        p[p_keep].rename(columns={perp_time_col: f"{perp_prefix}{perp_time_col}"}),
        on=list(keys),
        how="inner",
    )
    if out.empty:
        return out
    st = pd.to_datetime(out[f"{spot_prefix}{spot_time_col}"])
    pt = pd.to_datetime(out[f"{perp_prefix}{perp_time_col}"])
    out["abs_time_diff_minutes"] = (st - pt).abs().dt.total_seconds() / 60.0
    return out


def summarize_proxy_audit(
    *,
    spot_ny: pd.DataFrame,
    perp_ny: pd.DataFrame,
    aligned: pd.DataFrame,
    daily_paths: pd.DataFrame,
    spot_sweeps: pd.DataFrame,
    perp_sweeps: pd.DataFrame,
    spot_attempts: pd.DataFrame,
    perp_attempts: pd.DataFrame,
    thresholds: ProxyAuditThresholds = ProxyAuditThresholds(),
    caution: ProxyAuditCautionThresholds = ProxyAuditCautionThresholds(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    spot_session = clip_equity_research_session(spot_ny)
    perp_session = clip_equity_research_session(perp_ny)
    denom = min(len(spot_session), len(perp_session))
    aligned_coverage = float(len(aligned) / denom) if denom else float("nan")
    return_corr = _safe_corr(aligned.get("spot_ret_1m", pd.Series(dtype=float)), aligned.get("perp_ret_1m", pd.Series(dtype=float)))
    median_daily_corr = float(pd.to_numeric(daily_paths.get("rebased_path_corr", pd.Series(dtype=float)), errors="coerce").median()) if not daily_paths.empty else float("nan")

    ss = spot_sweeps.loc[spot_sweeps.get("level_type", pd.Series(dtype=str)).astype(str).eq("premarket_extreme")].copy() if not spot_sweeps.empty else pd.DataFrame()
    ps = perp_sweeps.loc[perp_sweeps.get("level_type", pd.Series(dtype=str)).astype(str).eq("premarket_extreme")].copy() if not perp_sweeps.empty else pd.DataFrame()
    sweep_j, sweep_inter, sweep_spot, sweep_perp = jaccard_keys(ss, ps, ["ny_date", "trade_side"])

    sa = spot_attempts.loc[spot_attempts.get("level_type", pd.Series(dtype=str)).astype(str).eq("premarket_extreme")].copy() if not spot_attempts.empty else pd.DataFrame()
    pa = perp_attempts.loc[perp_attempts.get("level_type", pd.Series(dtype=str)).astype(str).eq("premarket_extreme")].copy() if not perp_attempts.empty else pd.DataFrame()
    setup_j, setup_inter, setup_spot, setup_perp = jaccard_keys(sa, pa, ["ny_date", "trade_side", "execution_tf"])

    rows = [
        {"metric": "aligned_minute_coverage", "value": aligned_coverage, "pass_threshold": thresholds.aligned_minute_coverage, "caution_threshold": caution.aligned_minute_coverage, "higher_is_better": True},
        {"metric": "return_correlation_1m", "value": return_corr, "pass_threshold": thresholds.return_correlation, "caution_threshold": caution.return_correlation, "higher_is_better": True},
        {"metric": "median_daily_rebased_path_correlation", "value": median_daily_corr, "pass_threshold": thresholds.median_daily_rebased_path_correlation, "caution_threshold": caution.median_daily_rebased_path_correlation, "higher_is_better": True},
        {"metric": "external_sweep_key_jaccard", "value": sweep_j, "pass_threshold": thresholds.external_sweep_key_jaccard, "caution_threshold": caution.external_sweep_key_jaccard, "higher_is_better": True},
        {"metric": "base_setup_key_jaccard", "value": setup_j, "pass_threshold": thresholds.base_setup_key_jaccard, "caution_threshold": caution.base_setup_key_jaccard, "higher_is_better": True},
    ]
    metrics = pd.DataFrame(rows)
    metrics["pass"] = pd.to_numeric(metrics["value"], errors="coerce") >= pd.to_numeric(metrics["pass_threshold"], errors="coerce")
    metrics["caution_pass"] = pd.to_numeric(metrics["value"], errors="coerce") >= pd.to_numeric(metrics["caution_threshold"], errors="coerce")

    if bool(metrics["pass"].fillna(False).all()):
        verdict = "PASS"
    elif bool(metrics["caution_pass"].fillna(False).all()):
        verdict = "CAUTION"
    else:
        verdict = "FAIL"

    detail = {
        "verdict": verdict,
        "aligned_rows": int(len(aligned)),
        "spot_session_rows": int(len(spot_session)),
        "perp_session_rows": int(len(perp_session)),
        "external_sweep_intersection": int(sweep_inter),
        "external_sweep_spot_keys": int(sweep_spot),
        "external_sweep_perp_keys": int(sweep_perp),
        "base_setup_intersection": int(setup_inter),
        "base_setup_spot_keys": int(setup_spot),
        "base_setup_perp_keys": int(setup_perp),
        "policy": "Thresholds are pre-declared structural proxy gates and are not optimized on strategy PnL.",
    }
    return metrics, detail
