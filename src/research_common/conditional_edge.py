#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Time-split conditional-edge discovery utilities.

The module intentionally separates hypothesis generation from holdout scoring:
thresholds and frozen candidate specifications are learned only from the
``discovery`` split. Validation and holdout rows are never used to choose a
feature, polarity, threshold, branch, horizon, or feature pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

_EPS = 1e-12


@dataclass(frozen=True)
class ConditionalEdgeConfig:
    """Hard anti-overfit and sample-size constraints for R02."""

    discovery_end: str = "2024-12-31 23:59:59"
    validation_end: str = "2025-09-30 23:59:59"
    minimum_total_events: int = 1000
    minimum_discovery_events: int = 500
    minimum_validation_events: int = 200
    minimum_holdout_events: int = 200
    minimum_year_events: int = 80
    minimum_full_profit_factor: float = 1.20
    minimum_split_profit_factor: float = 1.00
    minimum_positive_month_ratio: float = 0.65
    minimum_positive_years: int = 3
    minimum_active_date_ratio: float = 0.65
    target_monthly_events_low: float = 40.0
    target_monthly_events_high: float = 90.0
    maximum_top5_winner_share: float = 0.20
    discovery_fdr_alpha: float = 0.10
    max_pair_features: int = 4

    def validate(self) -> None:
        discovery_end = pd.Timestamp(self.discovery_end)
        validation_end = pd.Timestamp(self.validation_end)
        if validation_end <= discovery_end:
            raise ValueError("validation_end must be after discovery_end")
        for name in (
            "minimum_total_events",
            "minimum_discovery_events",
            "minimum_validation_events",
            "minimum_holdout_events",
            "minimum_year_events",
            "minimum_positive_years",
            "max_pair_features",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.minimum_positive_month_ratio) <= 1.0:
            raise ValueError("minimum_positive_month_ratio must be in (0, 1]")
        if not 0.0 < float(self.minimum_active_date_ratio) <= 1.0:
            raise ValueError("minimum_active_date_ratio must be in (0, 1]")
        if float(self.target_monthly_events_high) < float(self.target_monthly_events_low):
            raise ValueError("target monthly event range is invalid")


DEFAULT_FEATURE_POLARITIES: Mapping[str, tuple[str, ...]] = {
    "pressure_z": ("high",),
    "flow_ratio_aligned": ("high",),
    "trade_imbalance_aligned": ("high",),
    "large_flow_ratio_aligned": ("high",),
    "large_notional_share": ("high",),
    "large_trade_share": ("high",),
    "flow_concentration": ("high",),
    "flow_persistence": ("high",),
    "notional_ratio": ("high",),
    "avg_trade_notional_ratio": ("high",),
    "max_trade_notional_ratio": ("high",),
    "activity_z": ("high",),
    "price_response_norm": ("high", "low"),
    "pressure_effectiveness": ("high", "low"),
    "impact_bps_per_million": ("high", "low"),
    "direction_close_location": ("high", "low"),
}


def prepare_conditional_features(events: pd.DataFrame) -> pd.DataFrame:
    """Create direction-symmetric causal feature columns and split metadata."""
    required = {
        "side",
        "signal_bar_start",
        "pressure_window_bars",
        "event_cluster_id",
        "flow_ratio",
        "trade_imbalance",
        "large_flow_ratio",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise KeyError(f"events missing conditional feature fields: {missing}")

    out = events.copy().sort_values(
        ["signal_bar_start", "pressure_window_bars", "event_id"],
        kind="stable",
    )
    side = pd.to_numeric(out["side"], errors="coerce")
    for source, target in (
        ("flow_ratio", "flow_ratio_aligned"),
        ("trade_imbalance", "trade_imbalance_aligned"),
        ("large_flow_ratio", "large_flow_ratio_aligned"),
    ):
        out[target] = side * pd.to_numeric(out[source], errors="coerce")

    out["signal_bar_start"] = pd.to_datetime(out["signal_bar_start"])
    out["year"] = out["signal_bar_start"].dt.year.astype(int)
    out["month"] = out["signal_bar_start"].dt.to_period("M").astype(str)
    out["date"] = out["signal_bar_start"].dt.date.astype(str)
    minute = out["signal_bar_start"].dt.minute
    out["clock_phase"] = np.select(
        [minute.mod(15).eq(0), minute.mod(5).eq(0)],
        ["quarter_hour", "five_minute"],
        default="other_minute",
    )
    out["window_cluster_primary_flag"] = ~out.duplicated(
        ["pressure_window_bars", "event_cluster_id"],
        keep="first",
    )
    return out.reset_index(drop=True)


def assign_time_splits(events: pd.DataFrame, config: ConditionalEdgeConfig) -> pd.DataFrame:
    """Assign discovery/validation/holdout without consulting outcomes."""
    config.validate()
    out = events.copy()
    timestamp = pd.to_datetime(out["signal_bar_start"])
    discovery_end = pd.Timestamp(config.discovery_end)
    validation_end = pd.Timestamp(config.validation_end)
    out["research_split"] = np.select(
        [timestamp <= discovery_end, timestamp <= validation_end],
        ["discovery", "validation"],
        default="holdout",
    )
    counts = out["research_split"].value_counts()
    missing = [name for name in ("discovery", "validation", "holdout") if int(counts.get(name, 0)) == 0]
    if missing:
        raise RuntimeError(f"time split has no events for: {missing}")
    return out


def profit_factor(values: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gains = float(x[x > 0.0].sum())
    losses = float(-x[x <= 0.0].sum())
    if losses <= 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return gains / losses


def _normal_two_sided_pvalue(t_stat: float) -> float:
    if not math.isfinite(t_stat):
        return float("nan")
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def _months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max(1.0, (end - start + pd.Timedelta(days=1)).total_seconds() / (365.2425 / 12.0 * 86400.0))


def conditional_return_stats(
    part: pd.DataFrame,
    *,
    gross_column: str,
    net_column: str,
    split_start: pd.Timestamp | None = None,
    split_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Return event, month and frequency diagnostics for one frozen condition."""
    if part.empty:
        return {
            "events": 0,
            "gross_mean": np.nan,
            "net_mean": np.nan,
            "net_median": np.nan,
            "net_win_rate": np.nan,
            "net_profit_factor": np.nan,
            "net_p05": np.nan,
            "net_p95": np.nan,
            "top5_winner_share": np.nan,
            "positive_month_ratio": np.nan,
            "monthly_mean": np.nan,
            "monthly_tstat": np.nan,
            "monthly_pvalue": np.nan,
            "positive_years": 0,
            "years_present": 0,
            "min_year_events": 0,
            "events_per_month": np.nan,
            "active_date_ratio": np.nan,
            "longest_gap_days": np.nan,
        }

    valid = part.loc[
        pd.to_numeric(part[net_column], errors="coerce").notna()
        & pd.to_numeric(part[gross_column], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return {
            "events": 0,
            "gross_mean": np.nan,
            "net_mean": np.nan,
            "net_median": np.nan,
            "net_win_rate": np.nan,
            "net_profit_factor": np.nan,
            "net_p05": np.nan,
            "net_p95": np.nan,
            "top5_winner_share": np.nan,
            "positive_month_ratio": np.nan,
            "monthly_mean": np.nan,
            "monthly_tstat": np.nan,
            "monthly_pvalue": np.nan,
            "positive_years": 0,
            "years_present": 0,
            "min_year_events": 0,
            "events_per_month": np.nan,
            "active_date_ratio": np.nan,
            "longest_gap_days": np.nan,
        }
    gross = pd.to_numeric(valid[gross_column], errors="coerce")
    net = pd.to_numeric(valid[net_column], errors="coerce")
    winners = net[net > 0.0].sort_values(ascending=False)
    top5_share = float(winners.head(5).sum() / winners.sum()) if float(winners.sum()) > 0.0 else np.nan

    monthly = valid.assign(_net=net).groupby("month", observed=False)["_net"].agg(["mean", "size"])
    monthly_means = pd.to_numeric(monthly["mean"], errors="coerce").dropna()
    if len(monthly_means) >= 2 and float(monthly_means.std(ddof=1)) > _EPS:
        monthly_tstat = float(monthly_means.mean() / (monthly_means.std(ddof=1) / math.sqrt(len(monthly_means))))
        monthly_pvalue = _normal_two_sided_pvalue(monthly_tstat)
    else:
        monthly_tstat = np.nan
        monthly_pvalue = np.nan

    yearly = valid.assign(_net=net).groupby("year", observed=False)["_net"].agg(["mean", "size"])
    dates = pd.to_datetime(valid["signal_bar_start"]).dt.normalize().drop_duplicates().sort_values()
    if split_start is None:
        split_start = pd.to_datetime(valid["signal_bar_start"]).min().normalize()
    if split_end is None:
        split_end = pd.to_datetime(valid["signal_bar_start"]).max().normalize()
    split_start = pd.Timestamp(split_start).normalize()
    split_end = pd.Timestamp(split_end).normalize()
    calendar_days = max(1, int((split_end - split_start).days + 1))
    gaps = dates.diff().dt.total_seconds().div(86400.0).dropna()

    return {
        "events": int(len(valid)),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "net_median": float(net.median()),
        "net_win_rate": float((net > 0.0).mean()),
        "net_profit_factor": profit_factor(net),
        "net_p05": float(net.quantile(0.05)),
        "net_p95": float(net.quantile(0.95)),
        "top5_winner_share": top5_share,
        "positive_month_ratio": float((monthly_means > 0.0).mean()) if len(monthly_means) else np.nan,
        "monthly_mean": float(monthly_means.mean()) if len(monthly_means) else np.nan,
        "monthly_tstat": monthly_tstat,
        "monthly_pvalue": monthly_pvalue,
        "positive_years": int((yearly["mean"] > 0.0).sum()),
        "years_present": int(len(yearly)),
        "min_year_events": int(yearly["size"].min()) if len(yearly) else 0,
        "events_per_month": float(len(valid) / _months_in_range(split_start, split_end)),
        "active_date_ratio": float(len(dates) / calendar_days),
        "longest_gap_days": float(gaps.max()) if len(gaps) else np.nan,
    }


def fit_discovery_quantiles(
    events: pd.DataFrame,
    *,
    features: Sequence[str],
    tail_quantiles: Sequence[float],
) -> pd.DataFrame:
    """Fit every threshold on discovery rows only, separately by pressure window."""
    discovery = events.loc[events["research_split"].eq("discovery")].copy()
    quantiles = sorted(set([0.5, *[float(q) for q in tail_quantiles], *[1.0 - float(q) for q in tail_quantiles]]))
    quantiles = [q for q in quantiles if 0.0 < q < 1.0]
    rows: list[dict[str, Any]] = []
    for window, part in discovery.groupby("pressure_window_bars", observed=False):
        for feature in features:
            values = pd.to_numeric(part[feature], errors="coerce").dropna()
            for q in quantiles:
                rows.append(
                    {
                        "pressure_window_bars": int(window),
                        "feature": feature,
                        "quantile": float(q),
                        "threshold": float(values.quantile(q)) if len(values) else np.nan,
                        "discovery_non_null_events": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def build_tail_specs(
    thresholds: pd.DataFrame,
    *,
    feature_polarities: Mapping[str, Sequence[str]],
    tail_quantiles: Sequence[float],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Build predeclared cumulative-tail specifications."""
    lookup = {
        (int(row.pressure_window_bars), str(row.feature), round(float(row.quantile), 10)): float(row.threshold)
        for row in thresholds.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    spec_id = 0
    windows = sorted(int(v) for v in thresholds["pressure_window_bars"].dropna().unique())
    for window in windows:
        for feature, polarities in feature_polarities.items():
            for polarity in polarities:
                for selectivity_q in sorted(set(float(q) for q in tail_quantiles)):
                    threshold_q = selectivity_q if polarity == "high" else 1.0 - selectivity_q
                    threshold = lookup.get((window, feature, round(threshold_q, 10)), np.nan)
                    if not math.isfinite(threshold):
                        continue
                    for branch in ("continuation", "reversal"):
                        for horizon in horizons:
                            spec_id += 1
                            rows.append(
                                {
                                    "spec_id": f"U{spec_id:06d}",
                                    "spec_type": "univariate_tail",
                                    "pressure_window_bars": int(window),
                                    "branch": branch,
                                    "horizon_bars": int(horizon),
                                    "feature_1": feature,
                                    "polarity_1": polarity,
                                    "selectivity_q_1": float(selectivity_q),
                                    "threshold_1": float(threshold),
                                    "feature_2": "",
                                    "polarity_2": "",
                                    "selectivity_q_2": np.nan,
                                    "threshold_2": np.nan,
                                }
                            )
    return pd.DataFrame(rows)


def _condition_mask(events: pd.DataFrame, spec: Mapping[str, Any]) -> np.ndarray:
    values = pd.to_numeric(events[str(spec["feature_1"])], errors="coerce")
    mask = values.ge(float(spec["threshold_1"])) if spec["polarity_1"] == "high" else values.le(float(spec["threshold_1"]))
    feature_2 = str(spec.get("feature_2", "") or "")
    if feature_2:
        values_2 = pd.to_numeric(events[feature_2], errors="coerce")
        mask_2 = values_2.ge(float(spec["threshold_2"])) if spec["polarity_2"] == "high" else values_2.le(float(spec["threshold_2"]))
        mask &= mask_2
    return mask.fillna(False).to_numpy(dtype=bool)


def _split_bounds(events: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    result: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, part in events.groupby("research_split", observed=False):
        ts = pd.to_datetime(part["signal_bar_start"])
        result[str(split)] = (ts.min().normalize(), ts.max().normalize())
    all_ts = pd.to_datetime(events["signal_bar_start"])
    result["full"] = (all_ts.min().normalize(), all_ts.max().normalize())
    return result


def _fast_stats_from_arrays(
    *,
    gross: np.ndarray,
    net: np.ndarray,
    mask: np.ndarray,
    month_codes: np.ndarray,
    year_values: np.ndarray,
    date_ordinals: np.ndarray,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
) -> dict[str, Any]:
    valid = mask & np.isfinite(gross) & np.isfinite(net)
    idx = np.flatnonzero(valid)
    if not len(idx):
        return {
            "events": 0,
            "gross_mean": np.nan,
            "net_mean": np.nan,
            "net_median": np.nan,
            "net_win_rate": np.nan,
            "net_profit_factor": np.nan,
            "net_p05": np.nan,
            "net_p95": np.nan,
            "top5_winner_share": np.nan,
            "positive_month_ratio": np.nan,
            "monthly_mean": np.nan,
            "monthly_tstat": np.nan,
            "monthly_pvalue": np.nan,
            "positive_years": 0,
            "years_present": 0,
            "min_year_events": 0,
            "events_per_month": np.nan,
            "active_date_ratio": np.nan,
            "longest_gap_days": np.nan,
        }
    x = net[idx]
    g = gross[idx]
    gains = x[x > 0.0]
    losses = -x[x <= 0.0]
    gain_sum = float(gains.sum())
    loss_sum = float(losses.sum())
    pf = gain_sum / loss_sum if loss_sum > 0.0 else (float("inf") if gain_sum > 0.0 else np.nan)
    top5_share = float(np.sort(gains)[-5:].sum() / gain_sum) if gain_sum > 0.0 else np.nan

    selected_month = month_codes[idx]
    month_unique, month_inv = np.unique(selected_month, return_inverse=True)
    month_counts = np.bincount(month_inv)
    month_sums = np.bincount(month_inv, weights=x)
    month_means = month_sums / np.maximum(month_counts, 1)
    if len(month_means) >= 2 and float(np.std(month_means, ddof=1)) > _EPS:
        monthly_tstat = float(np.mean(month_means) / (np.std(month_means, ddof=1) / math.sqrt(len(month_means))))
        monthly_pvalue = _normal_two_sided_pvalue(monthly_tstat)
    else:
        monthly_tstat = np.nan
        monthly_pvalue = np.nan

    selected_year = year_values[idx]
    year_unique, year_inv = np.unique(selected_year, return_inverse=True)
    year_counts = np.bincount(year_inv)
    year_sums = np.bincount(year_inv, weights=x)
    year_means = year_sums / np.maximum(year_counts, 1)

    unique_dates = np.unique(date_ordinals[idx])
    calendar_days = max(1, int((pd.Timestamp(split_end).normalize() - pd.Timestamp(split_start).normalize()).days + 1))
    gaps = np.diff(unique_dates) if len(unique_dates) >= 2 else np.array([], dtype=float)
    return {
        "events": int(len(idx)),
        "gross_mean": float(np.mean(g)),
        "net_mean": float(np.mean(x)),
        "net_median": float(np.median(x)),
        "net_win_rate": float(np.mean(x > 0.0)),
        "net_profit_factor": pf,
        "net_p05": float(np.quantile(x, 0.05)),
        "net_p95": float(np.quantile(x, 0.95)),
        "top5_winner_share": top5_share,
        "positive_month_ratio": float(np.mean(month_means > 0.0)),
        "monthly_mean": float(np.mean(month_means)),
        "monthly_tstat": monthly_tstat,
        "monthly_pvalue": monthly_pvalue,
        "positive_years": int(np.sum(year_means > 0.0)),
        "years_present": int(len(year_unique)),
        "min_year_events": int(np.min(year_counts)) if len(year_counts) else 0,
        "events_per_month": float(len(idx) / _months_in_range(pd.Timestamp(split_start), pd.Timestamp(split_end))),
        "active_date_ratio": float(len(unique_dates) / calendar_days),
        "longest_gap_days": float(np.max(gaps)) if len(gaps) else np.nan,
    }


def evaluate_specs(
    events: pd.DataFrame,
    specs: pd.DataFrame,
    *,
    splits: Sequence[str] = ("discovery", "validation", "holdout", "full"),
    progress_enabled: bool = False,
    progress_label: str = "[conditional-scan]",
) -> pd.DataFrame:
    """Evaluate frozen specs quickly; thresholds are never refit."""
    if specs.empty:
        return pd.DataFrame()
    source = events.loc[events["window_cluster_primary_flag"].astype(bool)].copy().reset_index(drop=True)
    bounds = _split_bounds(source)
    timestamps = pd.to_datetime(source["signal_bar_start"])
    month_codes = pd.factorize(source["month"], sort=True)[0].astype(np.int32)
    year_values = pd.to_numeric(source["year"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
    date_ordinals = (timestamps.dt.normalize().astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.int64)
    split_values = source["research_split"].astype(str).to_numpy(dtype=object)
    window_values = pd.to_numeric(source["pressure_window_bars"], errors="coerce").to_numpy(dtype=np.int16)

    records = specs.to_dict(orient="records")
    needed_features = {str(row["feature_1"]) for row in records}
    needed_features.update(str(row.get("feature_2", "") or "") for row in records)
    needed_features.discard("")
    feature_arrays = {
        feature: pd.to_numeric(source[feature], errors="coerce").to_numpy(dtype=float)
        for feature in needed_features
    }
    needed_outcomes: set[str] = set()
    for row in records:
        horizon = int(row["horizon_bars"])
        branch = str(row["branch"])
        needed_outcomes.add(f"{branch}_gross_h{horizon}")
        needed_outcomes.add(f"{branch}_net_h{horizon}")
    outcome_arrays = {
        column: pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
        for column in needed_outcomes
    }
    split_masks = {
        split: np.ones(len(source), dtype=bool) if split == "full" else (split_values == split)
        for split in splits
    }

    rows: list[dict[str, Any]] = []
    reporter = ProgressReporter(
        progress_label,
        len(records),
        every=max(1, len(records) // 100),
        enabled=progress_enabled,
    )
    for done, spec in enumerate(records, start=1):
        window_mask = window_values == int(spec["pressure_window_bars"])
        values = feature_arrays[str(spec["feature_1"])]
        condition = values >= float(spec["threshold_1"]) if spec["polarity_1"] == "high" else values <= float(spec["threshold_1"])
        feature_2 = str(spec.get("feature_2", "") or "")
        if feature_2:
            values_2 = feature_arrays[feature_2]
            condition_2 = values_2 >= float(spec["threshold_2"]) if spec["polarity_2"] == "high" else values_2 <= float(spec["threshold_2"])
            condition &= condition_2
        condition &= window_mask & np.isfinite(values)
        horizon = int(spec["horizon_bars"])
        branch = str(spec["branch"])
        gross = outcome_arrays[f"{branch}_gross_h{horizon}"]
        net = outcome_arrays[f"{branch}_net_h{horizon}"]
        base = dict(spec)
        for split in splits:
            start, end = bounds[split]
            stats = _fast_stats_from_arrays(
                gross=gross,
                net=net,
                mask=condition & split_masks[split],
                month_codes=month_codes,
                year_values=year_values,
                date_ordinals=date_ordinals,
                split_start=start,
                split_end=end,
            )
            rows.append({**base, "research_split": split, **stats})
        reporter.update(done)
    reporter.close()
    return pd.DataFrame(rows)

def evaluate_base_universes(events: pd.DataFrame, *, horizons: Sequence[int]) -> pd.DataFrame:
    specs: list[dict[str, Any]] = []
    for window in sorted(events["pressure_window_bars"].unique()):
        for branch in ("continuation", "reversal"):
            for horizon in horizons:
                specs.append(
                    {
                        "spec_id": f"BASE_w{int(window)}_{branch}_h{int(horizon)}",
                        "spec_type": "base_unfiltered",
                        "pressure_window_bars": int(window),
                        "branch": branch,
                        "horizon_bars": int(horizon),
                        "feature_1": "pressure_z",
                        "polarity_1": "high",
                        "selectivity_q_1": 0.0,
                        "threshold_1": -np.inf,
                        "feature_2": "",
                        "polarity_2": "",
                        "selectivity_q_2": np.nan,
                        "threshold_2": np.nan,
                    }
                )
    return evaluate_specs(events, pd.DataFrame(specs))


def pivot_split_results(results: pd.DataFrame) -> pd.DataFrame:
    """Create one row per spec with discovery/validation/holdout/full metrics."""
    if results.empty:
        return pd.DataFrame()
    id_cols = [
        "spec_id",
        "spec_type",
        "pressure_window_bars",
        "branch",
        "horizon_bars",
        "feature_1",
        "polarity_1",
        "selectivity_q_1",
        "threshold_1",
        "feature_2",
        "polarity_2",
        "selectivity_q_2",
        "threshold_2",
    ]
    metric_cols = [
        "events",
        "gross_mean",
        "net_mean",
        "net_median",
        "net_win_rate",
        "net_profit_factor",
        "net_p05",
        "net_p95",
        "top5_winner_share",
        "positive_month_ratio",
        "monthly_mean",
        "monthly_tstat",
        "monthly_pvalue",
        "positive_years",
        "years_present",
        "min_year_events",
        "events_per_month",
        "active_date_ratio",
        "longest_gap_days",
    ]
    wide_parts: list[pd.DataFrame] = []
    base = results[id_cols].drop_duplicates("spec_id").set_index("spec_id")
    wide_parts.append(base)
    for split in ("discovery", "validation", "holdout", "full"):
        part = results.loc[results["research_split"].eq(split), ["spec_id", *metric_cols]].copy()
        part = part.set_index("spec_id").add_prefix(f"{split}_")
        wide_parts.append(part)
    return pd.concat(wide_parts, axis=1).reset_index()


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values, preserving original order and NaNs."""
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.dropna().clip(0.0, 1.0)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    if valid.empty:
        return out
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def add_base_uplift(wide: pd.DataFrame, base_wide: pd.DataFrame) -> pd.DataFrame:
    """Compare every condition with its same-window/branch/horizon universe."""
    if wide.empty:
        return wide.copy()
    keys = ["pressure_window_bars", "branch", "horizon_bars"]
    base_cols = keys + [
        "discovery_net_mean",
        "validation_net_mean",
        "holdout_net_mean",
        "full_net_mean",
    ]
    base = base_wide[base_cols].rename(
        columns={column: f"base_{column}" for column in base_cols if column not in keys}
    )
    out = wide.merge(base, on=keys, how="left", validate="many_to_one")
    for split in ("discovery", "validation", "holdout", "full"):
        out[f"{split}_uplift_vs_base"] = out[f"{split}_net_mean"] - out[f"base_{split}_net_mean"]
    out["discovery_fdr_q"] = benjamini_hochberg(out["discovery_monthly_pvalue"])
    return out


def feature_monotonicity(wide: pd.DataFrame) -> pd.DataFrame:
    """Measure whether more-selective cumulative tails improve expectancy."""
    if wide.empty:
        return pd.DataFrame()
    group_cols = ["pressure_window_bars", "branch", "horizon_bars", "feature_1", "polarity_1"]
    rows: list[dict[str, Any]] = []
    for key, part in wide.groupby(group_cols, dropna=False, observed=False):
        ordered = part.sort_values("selectivity_q_1")
        valid_d = ordered[["selectivity_q_1", "discovery_net_mean"]].dropna()
        valid_v = ordered[["selectivity_q_1", "validation_net_mean"]].dropna()
        rows.append(
            {
                **dict(zip(group_cols, key, strict=False)),
                "thresholds": int(len(ordered)),
                "discovery_spearman": float(valid_d.corr(method="spearman").iloc[0, 1]) if len(valid_d) >= 3 else np.nan,
                "validation_spearman": float(valid_v.corr(method="spearman").iloc[0, 1]) if len(valid_v) >= 3 else np.nan,
                "discovery_positive_thresholds": int((ordered["discovery_net_mean"] > 0.0).sum()),
                "validation_positive_thresholds": int((ordered["validation_net_mean"] > 0.0).sum()),
                "holdout_positive_thresholds": int((ordered["holdout_net_mean"] > 0.0).sum()),
                "broadest_positive_selectivity_q": float(ordered.loc[ordered["discovery_net_mean"] > 0.0, "selectivity_q_1"].min()) if (ordered["discovery_net_mean"] > 0.0).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def freeze_discovery_candidates(
    wide: pd.DataFrame,
    monotonicity: pd.DataFrame,
    config: ConditionalEdgeConfig,
) -> pd.DataFrame:
    """Freeze single-feature candidates using discovery metrics only."""
    if wide.empty:
        return pd.DataFrame()
    keys = ["pressure_window_bars", "branch", "horizon_bars", "feature_1", "polarity_1"]
    out = wide.merge(monotonicity, on=keys, how="left", validate="many_to_one")
    out["discovery_sample_gate"] = out["discovery_events"] >= int(config.minimum_discovery_events)
    out["discovery_expectancy_gate"] = (
        out["discovery_net_mean"].gt(0.0)
        & out["discovery_net_profit_factor"].gt(1.0)
        & out["discovery_monthly_mean"].gt(0.0)
    )
    out["discovery_frequency_gate"] = out["discovery_events_per_month"].between(
        float(config.target_monthly_events_low) * 0.50,
        float(config.target_monthly_events_high) * 1.50,
        inclusive="both",
    )
    out["discovery_stability_gate"] = (
        out["discovery_positive_month_ratio"].ge(0.55)
        & out["discovery_min_year_events"].ge(max(40, int(config.minimum_year_events // 2)))
        & out["discovery_top5_winner_share"].le(float(config.maximum_top5_winner_share))
    )
    out["discovery_multiple_test_gate"] = out["discovery_fdr_q"].le(float(config.discovery_fdr_alpha))
    out["discovery_monotonicity_gate"] = out["discovery_spearman"].ge(0.30)
    out["frozen_discovery_flag"] = (
        out["discovery_sample_gate"]
        & out["discovery_expectancy_gate"]
        & out["discovery_frequency_gate"]
        & out["discovery_stability_gate"]
        & out["discovery_multiple_test_gate"]
        & out["discovery_monotonicity_gate"]
    )
    return out.sort_values(
        ["frozen_discovery_flag", "discovery_monthly_tstat", "discovery_net_mean", "discovery_events"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_pair_specs(
    frozen_singles: pd.DataFrame,
    *,
    config: ConditionalEdgeConfig,
) -> pd.DataFrame:
    """Pair only discovery-frozen broad conditions; validation/holdout stay untouched."""
    selected = frozen_singles.loc[frozen_singles["frozen_discovery_flag"].astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    spec_id = 0
    group_cols = ["pressure_window_bars", "branch", "horizon_bars"]
    for key, part in selected.groupby(group_cols, observed=False):
        # One broadest passing threshold per feature/polarity, selected without
        # looking at validation or holdout results.
        part = part.sort_values(
            ["feature_1", "polarity_1", "selectivity_q_1", "discovery_monthly_tstat"],
            ascending=[True, True, True, False],
        ).drop_duplicates(["feature_1", "polarity_1"], keep="first")
        part = part.sort_values(
            ["discovery_monthly_tstat", "discovery_net_mean", "discovery_events"],
            ascending=[False, False, False],
        ).head(int(config.max_pair_features))
        records = part.to_dict(orient="records")
        for left, right in combinations(records, 2):
            if left["feature_1"] == right["feature_1"]:
                continue
            spec_id += 1
            rows.append(
                {
                    "spec_id": f"P{spec_id:06d}",
                    "spec_type": "pairwise_frozen_discovery",
                    "pressure_window_bars": int(key[0]),
                    "branch": str(key[1]),
                    "horizon_bars": int(key[2]),
                    "feature_1": left["feature_1"],
                    "polarity_1": left["polarity_1"],
                    "selectivity_q_1": float(left["selectivity_q_1"]),
                    "threshold_1": float(left["threshold_1"]),
                    "feature_2": right["feature_1"],
                    "polarity_2": right["polarity_1"],
                    "selectivity_q_2": float(right["selectivity_q_1"]),
                    "threshold_2": float(right["threshold_1"]),
                }
            )
    return pd.DataFrame(rows)


def final_qualification(wide: pd.DataFrame, config: ConditionalEdgeConfig) -> pd.DataFrame:
    """Apply the user's hard sample/frequency/robustness gates after holdout."""
    if wide.empty:
        return wide.copy()
    out = wide.copy()
    out["sample_gate"] = (
        out["full_events"].ge(int(config.minimum_total_events))
        & out["discovery_events"].ge(int(config.minimum_discovery_events))
        & out["validation_events"].ge(int(config.minimum_validation_events))
        & out["holdout_events"].ge(int(config.minimum_holdout_events))
        & out["full_min_year_events"].ge(int(config.minimum_year_events))
    )
    out["all_split_expectancy_gate"] = (
        out["discovery_net_mean"].gt(0.0)
        & out["validation_net_mean"].gt(0.0)
        & out["holdout_net_mean"].gt(0.0)
        & out["full_net_mean"].gt(0.0)
    )
    out["profit_factor_gate"] = (
        out["full_net_profit_factor"].ge(float(config.minimum_full_profit_factor))
        & out["discovery_net_profit_factor"].ge(float(config.minimum_split_profit_factor))
        & out["validation_net_profit_factor"].ge(float(config.minimum_split_profit_factor))
        & out["holdout_net_profit_factor"].ge(float(config.minimum_split_profit_factor))
    )
    out["calendar_stability_gate"] = (
        out["full_positive_month_ratio"].ge(float(config.minimum_positive_month_ratio))
        & out["full_positive_years"].ge(int(config.minimum_positive_years))
        & out["full_active_date_ratio"].ge(float(config.minimum_active_date_ratio))
    )
    out["main_frequency_gate"] = out["full_events_per_month"].between(
        float(config.target_monthly_events_low),
        float(config.target_monthly_events_high),
        inclusive="both",
    )
    out["concentration_gate"] = out["full_top5_winner_share"].le(float(config.maximum_top5_winner_share))
    out["qualified_edge_flag"] = (
        out["sample_gate"]
        & out["all_split_expectancy_gate"]
        & out["profit_factor_gate"]
        & out["calendar_stability_gate"]
        & out["main_frequency_gate"]
        & out["concentration_gate"]
    )
    out["passed_gate_count"] = out[
        [
            "sample_gate",
            "all_split_expectancy_gate",
            "profit_factor_gate",
            "calendar_stability_gate",
            "main_frequency_gate",
            "concentration_gate",
        ]
    ].sum(axis=1)
    return out.sort_values(
        ["qualified_edge_flag", "passed_gate_count", "full_net_profit_factor", "full_net_mean", "full_events"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def candidate_time_stability(
    events: pd.DataFrame,
    specs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detailed yearly/monthly results for frozen or qualified specs."""
    if specs.empty:
        return pd.DataFrame(), pd.DataFrame()
    source = events.loc[events["window_cluster_primary_flag"].astype(bool)].copy()
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for spec in specs.to_dict(orient="records"):
        part = source.loc[source["pressure_window_bars"].eq(int(spec["pressure_window_bars"]))].copy()
        part = part.loc[_condition_mask(part, spec)].copy()
        gross_col = f"{spec['branch']}_gross_h{int(spec['horizon_bars'])}"
        net_col = f"{spec['branch']}_net_h{int(spec['horizon_bars'])}"
        for year, group in part.groupby("year", observed=False):
            stats = conditional_return_stats(group, gross_column=gross_col, net_column=net_col)
            yearly_rows.append({"spec_id": spec["spec_id"], "year": int(year), **stats})
        for month, group in part.groupby("month", observed=False):
            stats = conditional_return_stats(group, gross_column=gross_col, net_column=net_col)
            monthly_rows.append({"spec_id": spec["spec_id"], "month": str(month), **stats})
    return pd.DataFrame(yearly_rows), pd.DataFrame(monthly_rows)


def clock_phase_diagnostic(events: pd.DataFrame, *, horizons: Sequence[int]) -> pd.DataFrame:
    """External-research diagnostic only; never used to freeze candidates."""
    rows: list[dict[str, Any]] = []
    source = events.loc[events["window_cluster_primary_flag"].astype(bool)].copy()
    for key, part in source.groupby(["pressure_window_bars", "clock_phase"], observed=False):
        for branch in ("continuation", "reversal"):
            for horizon in horizons:
                gross_col = f"{branch}_gross_h{int(horizon)}"
                net_col = f"{branch}_net_h{int(horizon)}"
                stats = conditional_return_stats(part, gross_column=gross_col, net_column=net_col)
                rows.append(
                    {
                        "pressure_window_bars": int(key[0]),
                        "clock_phase": str(key[1]),
                        "branch": branch,
                        "horizon_bars": int(horizon),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def freeze_pair_candidates(wide: pd.DataFrame, config: ConditionalEdgeConfig) -> pd.DataFrame:
    """Freeze pairwise specs from discovery metrics only."""
    if wide.empty:
        return wide.copy()
    out = wide.copy()
    out["discovery_sample_gate"] = out["discovery_events"].ge(int(config.minimum_discovery_events))
    out["discovery_expectancy_gate"] = (
        out["discovery_net_mean"].gt(0.0)
        & out["discovery_net_profit_factor"].gt(1.0)
        & out["discovery_monthly_mean"].gt(0.0)
    )
    out["discovery_frequency_gate"] = out["discovery_events_per_month"].between(
        float(config.target_monthly_events_low) * 0.50,
        float(config.target_monthly_events_high) * 1.50,
        inclusive="both",
    )
    out["discovery_stability_gate"] = (
        out["discovery_positive_month_ratio"].ge(0.55)
        & out["discovery_min_year_events"].ge(max(40, int(config.minimum_year_events // 2)))
        & out["discovery_top5_winner_share"].le(float(config.maximum_top5_winner_share))
    )
    out["discovery_multiple_test_gate"] = out["discovery_fdr_q"].le(float(config.discovery_fdr_alpha))
    out["frozen_discovery_flag"] = (
        out["discovery_sample_gate"]
        & out["discovery_expectancy_gate"]
        & out["discovery_frequency_gate"]
        & out["discovery_stability_gate"]
        & out["discovery_multiple_test_gate"]
    )
    return out.sort_values(
        ["frozen_discovery_flag", "discovery_monthly_tstat", "discovery_net_mean", "discovery_events"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
