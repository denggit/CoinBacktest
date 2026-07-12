#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rich causal sequence features for second-stage C3 swing-low typology.

The parent C3 label is produced by research 01.  This module only inspects the
extreme bar and older trade bars.  Future confirmation fields remain metadata
and are never used as clustering features.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12


METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "confirmation_time",
        "confirmation_available_time",
        "completion_bars",
        "realized_confirmation_move_pct",
        "parent_cluster_id",
        "parent_distance_to_centroid",
        "parent_split",
        "year",
    }
)


def _safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= EPS:
        return float(default)
    return float(num / den)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _median(values: np.ndarray) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if arr.size else float("nan")


def _mean(values: np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: np.ndarray) -> float:
    arr = _finite(values)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def _linear_slope_r2(values: np.ndarray) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    x_center = x - x.mean()
    y_center = y - y.mean()
    den = float(np.dot(x_center, x_center))
    if den <= EPS:
        return 0.0, 0.0
    slope = float(np.dot(x_center, y_center) / den)
    fitted = y.mean() + slope * x_center
    ss_tot = float(np.dot(y_center, y_center))
    ss_res = float(np.dot(y - fitted, y - fitted))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    return slope, float(max(-1.0, min(1.0, r2)))


def _sign_changes(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    signs = np.sign(arr)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _hhi(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    total = float(arr.sum())
    if total <= EPS:
        return float("nan")
    share = arr / total
    return float(np.dot(share, share))


def _phase_slices(length: int, bins: int) -> list[slice]:
    edges = np.linspace(0, length, bins + 1, dtype=int)
    out: list[slice] = []
    for i in range(bins):
        start, end = int(edges[i]), int(edges[i + 1])
        if end <= start:
            end = min(length, start + 1)
        out.append(slice(start, end))
    return out


def _family_for_feature(feature: str) -> str:
    if feature.startswith("phase_price_") or feature.startswith("phase_low_"):
        return "sequence_price"
    if feature.startswith("phase_cvd_") or feature.startswith("phase_large_cvd_"):
        return "sequence_flow"
    if feature.startswith("phase_activity_") or feature.startswith("phase_trades_"):
        return "sequence_activity"
    if feature.startswith("current_"):
        return "extreme_bar"
    if any(token in feature for token in ("delta", "cvd", "buy_share", "sell_flow")):
        if any(token in feature for token in ("response", "dislocation", "correlation", "no_up", "no_down", "absorption")):
            return "price_flow_response"
        return "orderflow_path"
    if any(token in feature for token in ("notional", "trades", "activity", "large_trade", "max_trade", "hhi")):
        return "activity_path"
    return "price_structure"


def build_feature_dictionary(feature_columns: Sequence[str]) -> pd.DataFrame:
    labels: dict[str, str] = {
        "current_bar_return": "低点K线收益",
        "current_range_pct": "低点K线振幅",
        "current_body_pct": "低点K线实体",
        "current_lower_wick_share": "低点K线下影占比",
        "current_close_position": "低点K线收盘位置",
        "current_delta_ratio": "低点K线主动Delta占比",
        "current_large_delta_ratio": "低点K线大单Delta占比",
        "current_notional_intensity": "低点K线成交额倍率",
        "current_trades_intensity": "低点K线成交笔数倍率",
        "current_max_trade_share": "低点K线最大单占比",
    }
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        label = labels.get(feature, feature.replace("_", " "))
        rows.append(
            {
                "feature": feature,
                "family": _family_for_feature(feature),
                "label": label,
                "causal_cutoff": "extreme bar close or older",
            }
        )
    return pd.DataFrame(rows)


def _numeric_arrays(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    required = (
        "open",
        "high",
        "low",
        "close",
        "notional",
        "trades_count",
        "buy_notional",
        "sell_notional",
        "delta_notional",
        "avg_trade_size",
        "large_buy_notional",
        "large_sell_notional",
        "large_delta_notional",
        "max_trade_notional",
        "vwap",
    )
    return {
        col: pd.to_numeric(bars[col], errors="coerce").to_numpy(dtype=float, copy=False)
        for col in required
    }


def build_c3_sequence_features(
    bars: pd.DataFrame,
    parent_assignments: pd.DataFrame,
    *,
    windows: Sequence[int] = (15, 30, 60, 120, 240),
    phase_lookback: int = 240,
    phase_bins: int = 12,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build family-rich features using only data visible by the extreme close."""

    windows = tuple(sorted({int(x) for x in windows}))
    if not windows or min(windows) < 5:
        raise ValueError("windows must contain values >= 5")
    if phase_lookback < max(windows):
        raise ValueError("phase_lookback must be >= max(windows)")
    if phase_bins < 4:
        raise ValueError("phase_bins must be >= 4")

    numeric = _numeric_arrays(bars)
    open_ = numeric["open"]
    high = numeric["high"]
    low = numeric["low"]
    close = numeric["close"]
    notional = numeric["notional"]
    trades = numeric["trades_count"]
    buy = numeric["buy_notional"]
    sell = numeric["sell_notional"]
    delta = numeric["delta_notional"]
    avg_trade = numeric["avg_trade_size"]
    large_buy = numeric["large_buy_notional"]
    large_sell = numeric["large_sell_notional"]
    large_delta = numeric["large_delta_notional"]
    max_trade = numeric["max_trade_notional"]
    vwap = numeric["vwap"]
    large_gross = large_buy + large_sell
    timestamps = pd.DatetimeIndex(bars.index)

    events = parent_assignments.copy()
    events["extreme_pos"] = pd.to_numeric(events["extreme_pos"], errors="raise").astype(int)
    events = events[events["extreme_pos"] >= phase_lookback].sort_values("extreme_time").reset_index(drop=True)
    reporter = (
        ProgressReporter("[features] C3 sequence", total=len(events), every=max(1, int(progress_every)))
        if ProgressReporter is not None
        else None
    )

    rows: list[dict[str, object]] = []
    for idx, event in enumerate(events.itertuples(index=False)):
        pos = int(event.extreme_pos)
        ts = timestamps[pos]
        prev_close = close[pos - 1]
        bar_range = high[pos] - low[pos]
        lower_wick = min(open_[pos], close[pos]) - low[pos]
        prev30 = slice(max(0, pos - 30), pos)
        current_large_gross = large_gross[pos]
        row: dict[str, object] = {
            "event_id": event.event_id,
            "extreme_time": pd.Timestamp(event.extreme_time),
            "feature_available_time": pd.Timestamp(event.feature_available_time),
            "extreme_pos": pos,
            "extreme_price": float(event.extreme_price),
            "confirmation_time": pd.Timestamp(event.confirmation_time),
            "confirmation_available_time": pd.Timestamp(event.confirmation_available_time),
            "completion_bars": int(event.completion_bars),
            "realized_confirmation_move_pct": float(event.realized_confirmation_move_pct),
            "parent_cluster_id": str(event.cluster_id),
            "parent_distance_to_centroid": float(event.distance_to_train_centroid),
            "parent_split": str(event.split),
            "year": int(pd.Timestamp(event.extreme_time).year),
            "current_bar_return": _safe_ratio(close[pos], open_[pos]) - 1.0,
            "current_range_pct": _safe_ratio(bar_range, prev_close),
            "current_body_pct": _safe_ratio(abs(close[pos] - open_[pos]), prev_close),
            "current_lower_wick_share": _safe_ratio(lower_wick, bar_range),
            "current_close_position": _safe_ratio(close[pos] - low[pos], bar_range),
            "current_low_vs_prev_close": _safe_ratio(low[pos], prev_close) - 1.0,
            "current_delta_ratio": _safe_ratio(delta[pos], notional[pos]),
            "current_large_delta_ratio": _safe_ratio(large_delta[pos], current_large_gross),
            "current_buy_share": _safe_ratio(buy[pos], notional[pos]),
            "current_notional_intensity": _safe_ratio(notional[pos], _median(notional[prev30])),
            "current_trades_intensity": _safe_ratio(trades[pos], _median(trades[prev30])),
            "current_avg_trade_size_intensity": _safe_ratio(avg_trade[pos], _median(avg_trade[prev30])),
            "current_max_trade_share": _safe_ratio(max_trade[pos], notional[pos]),
            "current_large_trade_share": _safe_ratio(current_large_gross, notional[pos]),
            "current_close_vs_vwap": _safe_ratio(close[pos], vwap[pos]) - 1.0,
            "hour_sin": float(np.sin(2.0 * np.pi * ts.hour / 24.0)),
            "hour_cos": float(np.cos(2.0 * np.pi * ts.hour / 24.0)),
            "is_weekend": float(ts.dayofweek >= 5),
        }

        for window in windows:
            start = pos - window + 1
            sl = slice(start, pos + 1)
            c = close[sl]
            o = open_[sl]
            h = high[sl]
            l = low[sl]
            n = notional[sl]
            t = trades[sl]
            d = delta[sl]
            ld = large_delta[sl]
            lg = large_gross[sl]
            mt = max_trade[sl]
            at = avg_trade[sl]

            returns = np.divide(c, o, out=np.full_like(c, np.nan), where=np.abs(o) > EPS) - 1.0
            valid_c = c[np.isfinite(c) & (c > 0)]
            log_c = np.log(valid_c) if valid_c.size else np.array([], dtype=float)
            log_ret = np.diff(log_c) if log_c.size >= 2 else np.array([], dtype=float)
            price_slope, price_r2 = _linear_slope_r2(log_c)
            cum_delta = np.nancumsum(np.where(np.isfinite(d), d, 0.0))
            total_notional = float(np.nansum(n))
            cum_delta_norm = cum_delta / total_notional if total_notional > EPS else np.full_like(cum_delta, np.nan)
            cvd_slope, cvd_r2 = _linear_slope_r2(cum_delta_norm)
            log_activity = np.log1p(np.where(np.isfinite(n) & (n >= 0), n, np.nan))
            activity_slope, activity_r2 = _linear_slope_r2(log_activity)
            log_trades = np.log1p(np.where(np.isfinite(t) & (t >= 0), t, np.nan))
            trades_slope, _ = _linear_slope_r2(log_trades)
            log_avg_trade = np.log1p(np.where(np.isfinite(at) & (at >= 0), at, np.nan))
            avg_trade_slope, _ = _linear_slope_r2(log_avg_trade)

            max_high_idx = int(np.nanargmax(h)) if np.isfinite(h).any() else 0
            max_high = float(np.nanmax(h))
            min_low = float(np.nanmin(l))
            close_start = c[0]
            abs_path = float(np.nansum(np.abs(np.diff(c))))
            direction_change = _sign_changes(np.diff(c))
            down_mask = c < o
            red_streak = _longest_true_run(down_mask)
            lower_low_share = _mean((l[1:] < l[:-1]).astype(float)) if len(l) > 1 else np.nan
            prior_l = l[:-1]
            test_10bp = float(np.sum(prior_l <= low[pos] * 1.0010)) if len(prior_l) else 0.0
            test_25bp = float(np.sum(prior_l <= low[pos] * 1.0025)) if len(prior_l) else 0.0
            dwell_25bp = _mean((prior_l <= low[pos] * 1.0025).astype(float)) if len(prior_l) else np.nan
            running_min = np.minimum.accumulate(l)
            rebound = np.divide(h, running_min, out=np.full_like(h, np.nan), where=np.abs(running_min) > EPS) - 1.0
            rebound_above = rebound >= 0.0015
            rebound_attempts = int(np.sum(rebound_above & np.r_[True, ~rebound_above[:-1]]))

            thirds = np.array_split(np.arange(len(c)), 3)
            first_idx, middle_idx, last_idx = thirds
            first_delta = _safe_ratio(float(np.nansum(d[first_idx])), float(np.nansum(n[first_idx])))
            last_delta = _safe_ratio(float(np.nansum(d[last_idx])), float(np.nansum(n[last_idx])))
            first_large = _safe_ratio(float(np.nansum(ld[first_idx])), float(np.nansum(lg[first_idx])))
            last_large = _safe_ratio(float(np.nansum(ld[last_idx])), float(np.nansum(lg[last_idx])))
            first_activity = _median(n[first_idx])
            last_activity = _median(n[last_idx])

            delta_ratio_bar = np.divide(d, n, out=np.full_like(d, np.nan), where=np.abs(n) > EPS)
            bar_ret = returns
            valid_pair = np.isfinite(delta_ratio_bar) & np.isfinite(bar_ret)
            corr = float(np.corrcoef(delta_ratio_bar[valid_pair], bar_ret[valid_pair])[0, 1]) if valid_pair.sum() >= 4 else np.nan
            positive_delta_no_up = _mean(((delta_ratio_bar > 0) & (bar_ret <= 0)).astype(float))
            negative_delta_no_down = _mean(((delta_ratio_bar < 0) & (bar_ret >= 0)).astype(float))
            previous_running_min = np.r_[l[0], np.minimum.accumulate(l[:-1])]
            no_new_low = l >= previous_running_min * 0.9998
            sell_flow_without_new_low = _mean(((delta_ratio_bar < 0) & no_new_low).astype(float))
            negative_delta_sum = float(np.nansum(np.minimum(d, 0.0)))
            negative_return_sum = float(np.nansum(np.minimum(bar_ret, 0.0)))
            positive_delta_sum = float(np.nansum(np.maximum(d, 0.0)))
            positive_return_sum = float(np.nansum(np.maximum(bar_ret, 0.0)))

            row.update(
                {
                    f"close_return_{window}": _safe_ratio(c[-1], close_start) - 1.0,
                    f"low_return_{window}": _safe_ratio(low[pos], close_start) - 1.0,
                    f"drawdown_from_high_{window}": _safe_ratio(low[pos], max_high) - 1.0,
                    f"bars_since_high_ratio_{window}": float((len(c) - 1 - max_high_idx) / max(1, window - 1)),
                    f"close_position_{window}": _safe_ratio(c[-1] - min_low, max_high - min_low),
                    f"price_trend_slope_{window}": price_slope,
                    f"price_trend_r2_{window}": price_r2,
                    f"realized_vol_{window}": _std(log_ret),
                    f"downside_vol_{window}": _std(log_ret[log_ret < 0]) if log_ret.size else np.nan,
                    f"path_efficiency_{window}": _safe_ratio(abs(c[-1] - c[0]), abs_path),
                    f"direction_change_rate_{window}": direction_change,
                    f"down_bar_share_{window}": _mean(down_mask.astype(float)),
                    f"longest_down_streak_ratio_{window}": float(red_streak / window),
                    f"lower_low_share_{window}": lower_low_share,
                    f"prior_low_test_count_10bp_{window}": test_10bp,
                    f"prior_low_test_count_25bp_{window}": test_25bp,
                    f"near_floor_dwell_share_25bp_{window}": dwell_25bp,
                    f"max_rebound_before_low_{window}": float(np.nanmax(rebound)) if np.isfinite(rebound).any() else np.nan,
                    f"rebound_attempt_count_{window}": float(rebound_attempts),
                    f"cvd_ratio_{window}": _safe_ratio(float(np.nansum(d)), total_notional),
                    f"large_cvd_ratio_{window}": _safe_ratio(float(np.nansum(ld)), float(np.nansum(lg))),
                    f"cvd_slope_{window}": cvd_slope,
                    f"cvd_r2_{window}": cvd_r2,
                    f"delta_positive_share_{window}": _mean((d > 0).astype(float)),
                    f"delta_sign_change_rate_{window}": _sign_changes(d),
                    f"delta_acceleration_{window}": last_delta - first_delta,
                    f"large_delta_acceleration_{window}": last_large - first_large,
                    f"delta_hhi_{window}": _hhi(np.abs(d)),
                    f"notional_trend_slope_{window}": activity_slope,
                    f"notional_trend_r2_{window}": activity_r2,
                    f"trades_trend_slope_{window}": trades_slope,
                    f"avg_trade_size_trend_slope_{window}": avg_trade_slope,
                    f"activity_acceleration_{window}": _safe_ratio(last_activity, first_activity) - 1.0,
                    f"activity_burst_share_{window}": _mean((n > 2.0 * _median(n)).astype(float)),
                    f"notional_hhi_{window}": _hhi(n),
                    f"large_trade_share_{window}": _safe_ratio(float(np.nansum(lg)), total_notional),
                    f"max_trade_concentration_{window}": _safe_ratio(float(np.nanmax(mt)), total_notional),
                    f"return_delta_correlation_{window}": corr,
                    f"positive_delta_no_up_share_{window}": positive_delta_no_up,
                    f"negative_delta_no_down_share_{window}": negative_delta_no_down,
                    f"sell_flow_without_new_low_share_{window}": sell_flow_without_new_low,
                    f"buy_price_efficiency_{window}": _safe_ratio(positive_return_sum, positive_delta_sum / total_notional if total_notional > EPS else np.nan),
                    f"sell_price_efficiency_{window}": _safe_ratio(-negative_return_sum, -negative_delta_sum / total_notional if total_notional > EPS else np.nan),
                    f"price_cvd_dislocation_{window}": (_safe_ratio(c[-1], c[0]) - 1.0) - _safe_ratio(float(np.nansum(d)), total_notional),
                    f"absorption_score_{window}": sell_flow_without_new_low * max(0.0, -_safe_ratio(float(np.nansum(d)), total_notional)),
                }
            )

        phase_start = pos - phase_lookback + 1
        phase_sl = slice(phase_start, pos + 1)
        pc = close[phase_sl]
        pl = low[phase_sl]
        pn = notional[phase_sl]
        pt = trades[phase_sl]
        pdlt = delta[phase_sl]
        pld = large_delta[phase_sl]
        plg = large_gross[phase_sl]
        phase_activity_base = _median(pn)
        phase_trade_base = _median(pt)
        for phase_idx, phase_slice in enumerate(_phase_slices(len(pc), phase_bins), start=1):
            c_bin = pc[phase_slice]
            l_bin = pl[phase_slice]
            n_bin = pn[phase_slice]
            t_bin = pt[phase_slice]
            d_bin = pdlt[phase_slice]
            ld_bin = pld[phase_slice]
            lg_bin = plg[phase_slice]
            row[f"phase_price_{phase_idx:02d}"] = _safe_ratio(_median(c_bin), low[pos]) - 1.0
            row[f"phase_low_{phase_idx:02d}"] = _safe_ratio(float(np.nanmin(l_bin)), low[pos]) - 1.0
            row[f"phase_cvd_{phase_idx:02d}"] = _safe_ratio(float(np.nansum(d_bin)), float(np.nansum(n_bin)))
            row[f"phase_large_cvd_{phase_idx:02d}"] = _safe_ratio(float(np.nansum(ld_bin)), float(np.nansum(lg_bin)))
            row[f"phase_activity_{phase_idx:02d}"] = _safe_ratio(_median(n_bin), phase_activity_base)
            row[f"phase_trades_{phase_idx:02d}"] = _safe_ratio(_median(t_bin), phase_trade_base)

        rows.append(row)
        if reporter is not None and idx + 1 < len(events):
            reporter.update(idx + 1)

    if reporter is not None:
        reporter.close()
    features = pd.DataFrame(rows).sort_values("extreme_time").reset_index(drop=True)
    feature_columns = [c for c in features.columns if c not in METADATA_COLUMNS]
    dictionary = build_feature_dictionary(feature_columns)
    return features, dictionary


def build_sequence_profiles(features: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Convert phase features into compact cluster path profiles."""

    merged = assignments[["event_id", "split", "subcluster_id"]].merge(features, on="event_id", how="inner")
    rows: list[dict[str, object]] = []
    prefixes = {
        "price": "phase_price_",
        "low": "phase_low_",
        "cvd": "phase_cvd_",
        "large_cvd": "phase_large_cvd_",
        "activity": "phase_activity_",
        "trades": "phase_trades_",
    }
    for (subcluster, split), group in merged.groupby(["subcluster_id", "split"], sort=True):
        for metric, prefix in prefixes.items():
            cols = sorted(c for c in features.columns if c.startswith(prefix))
            for phase, col in enumerate(cols, start=1):
                values = pd.to_numeric(group[col], errors="coerce")
                rows.append(
                    {
                        "subcluster_id": subcluster,
                        "split": split,
                        "metric": metric,
                        "phase": phase,
                        "count": int(values.notna().sum()),
                        "median": float(values.median()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                    }
                )
    return pd.DataFrame(rows)


def build_causal_audit(features: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    forbidden_tokens = ("future", "post_", "forward", "confirmation", "completion", "mfe", "mae")
    forbidden = [c for c in feature_columns if any(token in c.lower() for token in forbidden_tokens)]
    feature_time = pd.to_datetime(features["feature_available_time"])
    extreme_time = pd.to_datetime(features["extreme_time"])
    confirmation_time = pd.to_datetime(features["confirmation_available_time"])
    return pd.DataFrame(
        [
            {
                "check": "features_end_at_extreme_close",
                "passed": bool((feature_time > extreme_time).all()),
                "detail": "left-labelled extreme bar features become available after bar close",
            },
            {
                "check": "confirmation_after_feature_cutoff",
                "passed": bool((confirmation_time > feature_time).all()),
                "detail": "future confirmation remains metadata only",
            },
            {
                "check": "no_future_named_features",
                "passed": not forbidden,
                "detail": ",".join(forbidden),
            },
            {
                "check": "parent_cluster_is_c3_only",
                "passed": bool((features["parent_cluster_id"].astype(str) == "C3").all()),
                "detail": f"rows={len(features):,}",
            },
            {
                "check": "rich_feature_count",
                "passed": len(feature_columns) >= 100,
                "detail": str(len(feature_columns)),
            },
        ]
    )
