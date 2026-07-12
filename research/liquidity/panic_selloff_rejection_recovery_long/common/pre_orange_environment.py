#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal pre-orange feature extraction and winner/loser diagnostics.

This module is specific to the panic selloff -> recovery research family.
It deliberately lives under the research directory rather than ``src``.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return aa / bb


def _rolling_ratio(delta: pd.Series, total: pd.Series, window: int, *, shift: int = 1) -> pd.Series:
    min_periods = max(2, min(window, window // 3))
    d = pd.to_numeric(delta, errors="coerce").shift(shift).rolling(window, min_periods=min_periods).sum()
    t = pd.to_numeric(total, errors="coerce").shift(shift).rolling(window, min_periods=min_periods).sum()
    return _safe_divide(d, t)


def _sample_series(series: pd.Series, positions: np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full(len(positions), np.nan, dtype=float)
    valid = (positions >= 0) & (positions < len(arr))
    if np.any(valid):
        out[valid] = arr[positions[valid]]
    return out


def feature_id(text: str) -> str:
    raw = str(text)
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    if len(cleaned) <= 100:
        return cleaned
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:88]}__{digest}"


def _add_sampled_feature(
    feature_values: dict[str, np.ndarray],
    meta: list[dict[str, str]],
    *,
    name: str,
    series: pd.Series,
    positions: np.ndarray,
    family: str,
    scope: str,
    description: str,
) -> None:
    feature_values[name] = _sample_series(series, positions)
    meta.append(
        {
            "feature": name,
            "family": family,
            "scope": scope,
            "description": description,
        }
    )


def build_pre_orange_features(
    bars: pd.DataFrame,
    orderflow: pd.DataFrame,
    starts: pd.DataFrame,
    *,
    windows: list[int],
    progress_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build event-level features without materializing another huge frame.

    Every ``pre_*`` series shifts by one bar before rolling, so the latest input
    is the bar immediately before orange. ``orange_*`` is sampled from the
    closed orange bar and kept in a separate scope.
    """
    starts = starts.sort_values("event_time").drop_duplicates("episode_id").reset_index(drop=True)
    positions = bars.index.get_indexer(pd.DatetimeIndex(starts["event_time"]))
    base_values: dict[str, Any] = {
        "episode_id": starts["episode_id"].to_numpy(),
        "orange_time": pd.to_datetime(starts["event_time"]).to_numpy(),
        "orange_pos": positions,
        "has_green_signal": starts["has_green_signal"].fillna(False).astype(bool).to_numpy(),
        "episode_status": starts["episode_status"].astype(str).to_numpy(),
        # Forward diagnostics are carried only for reporting and explicitly
        # excluded from the feature metadata/candidate list.
        "diagnostic_start_to_final_low": numeric_series(starts, "diagnostic_start_to_final_low").to_numpy(),
        "diagnostic_start_was_near_final_low": starts[
            "diagnostic_start_was_near_final_low"
        ].fillna(False).astype(bool).to_numpy(),
    }
    feature_values: dict[str, np.ndarray] = {}
    meta: list[dict[str, str]] = []

    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    ret = close.pct_change()
    prior_close = close.shift(1)

    delta = pd.to_numeric(orderflow["delta_notional"], errors="coerce")
    notional = pd.to_numeric(orderflow["notional"], errors="coerce")
    large_delta = pd.to_numeric(orderflow["large_delta_notional"], errors="coerce")
    large_notional = pd.to_numeric(orderflow["large_notional"], errors="coerce")
    buy_notional = pd.to_numeric(orderflow["buy_notional"], errors="coerce")
    sell_notional = pd.to_numeric(orderflow["sell_notional"], errors="coerce")
    sell_dominant = (orderflow["delta_ratio"] < 0).astype(float)
    extreme_sell = (orderflow["delta_ratio"] <= -0.25).astype(float)
    rejection = (
        (orderflow["sell_notional_ratio_base"] >= 1.50)
        & (orderflow["close_pos"] >= 0.55)
    ).astype(float)
    progress = ProgressReporter(
        label="[orange-pre] feature windows",
        total=len(windows) + 2,
        every=1,
        enabled=progress_enabled,
    )

    for window_i, window in enumerate(windows, start=1):
        min_periods = max(2, min(window, window // 3))
        pre_ret = prior_close / close.shift(window + 1) - 1.0
        path = ret.abs().shift(1).rolling(window, min_periods=min_periods).sum()
        _add_sampled_feature(
            feature_values, meta, name=f"pre_price_return_{window}", series=pre_ret, positions=positions,
            family="price_path", scope="pre_orange",
            description=f"橙灯前{window} bars收益",
        )
        _add_sampled_feature(
            feature_values, meta, name=f"pre_price_efficiency_{window}", series=_safe_divide(pre_ret.abs(), path),
            positions=positions, family="price_path", scope="pre_orange",
            description=f"橙灯前{window} bars方向效率",
        )
        _add_sampled_feature(
            feature_values, meta, name=f"pre_red_fraction_{window}",
            series=(ret < 0).astype(float).shift(1).rolling(window, min_periods=min_periods).mean(),
            positions=positions, family="price_sequence", scope="pre_orange",
            description=f"橙灯前{window} bars阴线占比",
        )
        prior_high = high.shift(1).rolling(window, min_periods=min_periods).max()
        prior_low = low.shift(1).rolling(window, min_periods=min_periods).min()
        _add_sampled_feature(
            feature_values, meta, name=f"pre_range_position_{window}",
            series=_safe_divide(prior_close - prior_low, prior_high - prior_low).clip(0.0, 1.0),
            positions=positions, family="price_location", scope="pre_orange",
            description=f"橙灯前价格位于{window} bars区间的位置",
        )
        _add_sampled_feature(
            feature_values, meta, name=f"pre_drawdown_{window}", series=prior_close / prior_high - 1.0,
            positions=positions, family="price_location", scope="pre_orange",
            description=f"橙灯前距离{window} bars高点回撤",
        )

        flow_delta = _rolling_ratio(delta, notional, window, shift=1)
        flow_large_delta = _rolling_ratio(large_delta, large_notional, window, shift=1)
        flow_taker = _rolling_ratio(buy_notional, notional, window, shift=1)
        _add_sampled_feature(
            feature_values, meta, name=f"pre_delta_ratio_{window}", series=flow_delta,
            positions=positions, family="active_flow", scope="pre_orange",
            description=f"橙灯前{window} bars主动买卖成交额差占比",
        )
        _add_sampled_feature(
            feature_values, meta, name=f"pre_large_delta_ratio_{window}", series=flow_large_delta,
            positions=positions, family="large_flow", scope="pre_orange",
            description=f"橙灯前{window} bars大单主动买卖差占比",
        )
        _add_sampled_feature(
            feature_values, meta, name=f"pre_taker_buy_ratio_{window}", series=flow_taker,
            positions=positions, family="active_flow", scope="pre_orange",
            description=f"橙灯前{window} bars主动买入成交额占比",
        )

        # Detailed microstructure rolling distributions are most useful on
        # short/medium windows. For 120/240 bars we keep only price and flow
        # aggregates above, avoiding dozens of expensive full-history rolls.
        if window <= 60:
            rolling_specs = [
                ("sell_intensity_mean", orderflow["sell_notional_ratio_base"].shift(1).rolling(window, min_periods=min_periods).mean(), "activity", "卖出成交额强度均值"),
                ("sell_intensity_max", orderflow["sell_notional_ratio_base"].shift(1).rolling(window, min_periods=min_periods).max(), "activity", "卖出成交额强度峰值"),
                ("trades_intensity_max", orderflow["trades_ratio_base"].shift(1).rolling(window, min_periods=min_periods).max(), "activity", "成交笔数强度峰值"),
                ("notional_intensity_max", orderflow["notional_ratio_base"].shift(1).rolling(window, min_periods=min_periods).max(), "activity", "总成交额强度峰值"),
                ("large_sell_share_mean", orderflow["large_sell_share_of_sell"].shift(1).rolling(window, min_periods=min_periods).mean(), "large_participation", "大卖单占卖出成交额均值"),
                ("large_sell_share_max", orderflow["large_sell_share_of_sell"].shift(1).rolling(window, min_periods=min_periods).max(), "large_participation", "大卖单占卖出成交额峰值"),
                ("max_trade_share_max", orderflow["max_trade_share"].shift(1).rolling(window, min_periods=min_periods).max(), "trade_concentration", "最大单笔成交额占比峰值"),
                ("absorption_mean", orderflow["absorption_score"].shift(1).rolling(window, min_periods=min_periods).mean(), "absorption", "吸收评分均值"),
                ("absorption_max", orderflow["absorption_score"].shift(1).rolling(window, min_periods=min_periods).max(), "absorption", "吸收评分峰值"),
                ("sell_impact_mean", orderflow["sell_impact_per_intensity"].shift(1).rolling(window, min_periods=min_periods).mean(), "price_impact", "单位卖压价格冲击均值"),
                ("close_position_mean", orderflow["close_pos"].shift(1).rolling(window, min_periods=min_periods).mean(), "rejection", "收盘在bar区间位置均值"),
                ("lower_wick_max", orderflow["lower_wick_frac"].shift(1).rolling(window, min_periods=min_periods).max(), "rejection", "下影线占比峰值"),
            ]
            for suffix, series, family, desc in rolling_specs:
                _add_sampled_feature(
                    feature_values, meta, name=f"pre_{suffix}_{window}", series=series, positions=positions,
                    family=family, scope="pre_orange", description=f"橙灯前{window} bars{desc}",
                )

            _add_sampled_feature(
                feature_values, meta, name=f"pre_sell_dominant_fraction_{window}",
                series=sell_dominant.shift(1).rolling(window, min_periods=min_periods).mean(),
                positions=positions, family="flow_sequence", scope="pre_orange",
                description=f"橙灯前{window} bars主动卖方占优比例",
            )
            _add_sampled_feature(
                feature_values, meta, name=f"pre_extreme_sell_fraction_{window}",
                series=extreme_sell.shift(1).rolling(window, min_periods=min_periods).mean(),
                positions=positions, family="flow_sequence", scope="pre_orange",
                description=f"橙灯前{window} bars极端主动卖压比例",
            )
            _add_sampled_feature(
                feature_values, meta, name=f"pre_rejection_fraction_{window}",
                series=rejection.shift(1).rolling(window, min_periods=min_periods).mean(),
                positions=positions, family="rejection", scope="pre_orange",
                description=f"橙灯前{window} bars卖压放大但价格拒绝比例",
            )
        progress.update(window_i)

    # Explicit recent-vs-prior acceleration, all ending before orange.
    recent = 5
    prior = 15
    recent_delta = _rolling_ratio(delta, notional, recent, shift=1)
    prior_delta = _rolling_ratio(delta, notional, prior, shift=recent + 1)
    recent_large = _rolling_ratio(large_delta, large_notional, recent, shift=1)
    prior_large = _rolling_ratio(large_delta, large_notional, prior, shift=recent + 1)
    recent_sell = orderflow["sell_notional_ratio_base"].shift(1).rolling(recent, min_periods=2).mean()
    prior_sell = orderflow["sell_notional_ratio_base"].shift(recent + 1).rolling(prior, min_periods=5).mean()
    recent_price = close.shift(1) / close.shift(recent + 1) - 1.0
    prior_price = close.shift(recent + 1) / close.shift(recent + prior + 1) - 1.0
    acceleration_specs = [
        ("pre_delta_acceleration_5_vs_prior15", recent_delta - prior_delta, "flow_acceleration", "近期主动流相对前段变化"),
        ("pre_large_delta_acceleration_5_vs_prior15", recent_large - prior_large, "flow_acceleration", "近期大单流相对前段变化"),
        ("pre_sell_intensity_acceleration_5_vs_prior15", recent_sell - prior_sell, "activity_acceleration", "近期卖出强度相对前段变化"),
        ("pre_price_acceleration_5_vs_prior15", recent_price - prior_price * (recent / prior), "price_acceleration", "近期下跌速度相对前段变化"),
    ]
    for name, series, family, desc in acceleration_specs:
        _add_sampled_feature(
            feature_values, meta, name=name, series=series, positions=positions,
            family=family, scope="pre_orange", description=desc,
        )
    progress.update(len(windows) + 1)

    # Orange closed-bar fields: secondary analysis only, kept separate from pre.
    orange_specs = [
        ("node_delta_ratio", "orange_delta_ratio", "orange_flow", "橙灯bar主动买卖差占比"),
        ("node_large_delta_ratio", "orange_large_delta_ratio", "orange_large_flow", "橙灯bar大单主动买卖差占比"),
        ("node_taker_buy_ratio", "orange_taker_buy_ratio", "orange_flow", "橙灯bar主动买入占比"),
        ("node_sell_notional_ratio_base", "orange_sell_intensity", "orange_activity", "橙灯bar卖出成交额强度"),
        ("node_trades_ratio_base", "orange_trades_intensity", "orange_activity", "橙灯bar成交笔数强度"),
        ("node_large_trade_share", "orange_large_trade_share", "orange_large_flow", "橙灯bar大单成交额占比"),
        ("node_large_sell_share_of_sell", "orange_large_sell_share", "orange_large_flow", "橙灯bar大卖单占卖出成交额"),
        ("node_max_trade_share", "orange_max_trade_share", "orange_concentration", "橙灯bar最大单笔占比"),
        ("node_absorption_score", "orange_absorption_score", "orange_absorption", "橙灯bar吸收评分"),
        ("node_sell_impact_per_intensity", "orange_sell_impact", "orange_price_impact", "橙灯bar单位卖压价格冲击"),
        ("close_pos", "orange_close_position", "orange_rejection", "橙灯bar收盘位置"),
        ("lower_wick_frac", "orange_lower_wick_fraction", "orange_rejection", "橙灯bar下影线占比"),
        ("node_delta_reversal_short", "orange_delta_reversal", "orange_reversal", "橙灯bar短周期主动流改善"),
        ("node_large_delta_reversal_short", "orange_large_delta_reversal", "orange_reversal", "橙灯bar短周期大单流改善"),
        ("selloff_return", "orange_trigger_selloff_return", "orange_trigger", "橙灯触发窗口跌幅"),
        ("drop_vol_mult", "orange_trigger_drop_vol_mult", "orange_trigger", "橙灯跌幅波动倍数"),
        ("red_count", "orange_trigger_red_count", "orange_trigger", "橙灯触发窗口阴线数"),
        ("window_volume_ratio", "orange_trigger_volume_ratio", "orange_trigger", "橙灯触发窗口成交量倍数"),
    ]
    for source, name, family, desc in orange_specs:
        feature_values[name] = numeric_series(starts, source).to_numpy(dtype=float)
        meta.append({"feature": name, "family": family, "scope": "orange_closed", "description": desc})

    # Existing context columns from 01 are strictly pre-node and useful as slow context.
    for source in (
        "pre_ret_15", "pre_ret_60", "pre_ret_240", "pre_efficiency_60", "pre_efficiency_240",
        "pre_atr_pct_60", "pre_atr_regime_ratio", "pre_range_pos_240", "pre_drawdown_240", "pre_volume_ratio",
    ):
        if source not in starts.columns:
            continue
        name = f"context_{source}"
        feature_values[name] = numeric_series(starts, source).to_numpy(dtype=float)
        meta.append(
            {
                "feature": name,
                "family": "slow_context",
                "scope": "pre_orange",
                "description": f"01既有严格前置上下文: {source}",
            }
        )

    progress.close()
    table = pd.concat(
        [pd.DataFrame(base_values), pd.DataFrame(feature_values)],
        axis=1,
        copy=False,
    )
    feature_meta = pd.DataFrame(meta).drop_duplicates("feature").reset_index(drop=True)
    return table.replace([np.inf, -np.inf], np.nan), feature_meta


def join_orange_features_to_green(
    enriched: pd.DataFrame,
    orange_features: pd.DataFrame,
    *,
    candidate_horizon: int,
    winner_threshold: float,
    strong_winner_threshold: float,
    strong_loser_threshold: float,
) -> pd.DataFrame:
    signals = enriched[enriched["stage"] == "signal"].copy().sort_values("event_time").reset_index(drop=True)
    signals = signals.merge(orange_features, on="episode_id", how="left", validate="one_to_one")
    return_col = f"ret_h{int(candidate_horizon)}_net"
    if return_col not in signals.columns:
        raise RuntimeError(f"missing candidate outcome: {return_col}")
    outcome = numeric_series(signals, return_col)
    signals["label_winner"] = outcome > float(winner_threshold)
    signals["label_strong_winner"] = outcome >= float(strong_winner_threshold)
    signals["label_strong_loser"] = outcome <= float(strong_loser_threshold)
    signals["label_strong_contrast"] = np.where(
        signals["label_strong_winner"], 1,
        np.where(signals["label_strong_loser"], 0, np.nan),
    )
    signals["orange_to_green_bars"] = numeric_series(signals, "event_pos") - numeric_series(signals, "orange_pos")
    return signals


def _binary_auc(values: pd.Series, labels: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(values, errors="coerce"), "y": labels}).dropna()
    if frame.empty:
        return np.nan
    y = frame["y"].astype(int)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = frame["x"].rank(method="average")
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _contrast_stats(part: pd.DataFrame, feature: str, label_col: str) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "x": numeric_series(part, feature),
            "y": pd.to_numeric(part[label_col], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return {
            "count": 0, "winner_count": 0, "loser_count": 0,
            "winner_mean": np.nan, "loser_mean": np.nan,
            "winner_median": np.nan, "loser_median": np.nan,
            "mean_diff": np.nan, "standardized_diff": np.nan,
            "auc": np.nan, "directional_strength": np.nan,
        }
    winners = frame.loc[frame["y"] == 1, "x"]
    losers = frame.loc[frame["y"] == 0, "x"]
    if winners.empty or losers.empty:
        return {
            "count": int(len(frame)), "winner_count": int(len(winners)), "loser_count": int(len(losers)),
            "winner_mean": np.nan, "loser_mean": np.nan,
            "winner_median": np.nan, "loser_median": np.nan,
            "mean_diff": np.nan, "standardized_diff": np.nan,
            "auc": np.nan, "directional_strength": np.nan,
        }
    mean_diff = float(winners.mean() - losers.mean())
    pooled = math.sqrt(max(0.0, (float(winners.var(ddof=1)) + float(losers.var(ddof=1))) / 2.0))
    auc = _binary_auc(frame["x"], frame["y"])
    return {
        "count": int(len(frame)),
        "winner_count": int(len(winners)),
        "loser_count": int(len(losers)),
        "winner_mean": float(winners.mean()),
        "loser_mean": float(losers.mean()),
        "winner_median": float(winners.median()),
        "loser_median": float(losers.median()),
        "mean_diff": mean_diff,
        "standardized_diff": mean_diff / pooled if pooled > 0 else np.nan,
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "directional_strength": float(abs(auc - 0.5) * 2.0) if np.isfinite(auc) else np.nan,
    }


def winner_loser_contrast(
    signals: pd.DataFrame,
    feature_meta: pd.DataFrame,
    *,
    label_col: str,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    train_mask = pd.to_datetime(signals["event_time"]) <= train_end
    rows: list[dict[str, Any]] = []
    meta_by_feature = feature_meta.set_index("feature").to_dict(orient="index")
    for feature in feature_meta["feature"].tolist():
        row: dict[str, Any] = {"feature": feature, **meta_by_feature[feature]}
        for prefix, part in (
            ("all", signals),
            ("train", signals[train_mask]),
            ("holdout", signals[~train_mask]),
        ):
            stats = _contrast_stats(part, feature, label_col)
            row.update({f"{prefix}_{k}": v for k, v in stats.items()})
        train_auc = row.get("train_auc", np.nan)
        holdout_auc = row.get("holdout_auc", np.nan)
        train_dir = np.sign(float(train_auc) - 0.5) if np.isfinite(train_auc) else 0.0
        if np.isfinite(train_auc) and np.isfinite(holdout_auc):
            holdout_dir = np.sign(float(holdout_auc) - 0.5)
            row["direction_stable"] = bool(train_dir != 0 and train_dir == holdout_dir)
        else:
            row["direction_stable"] = False

        year_parts: list[str] = []
        valid_years = 0
        same_direction_years = 0
        for year, yp in signals.groupby(pd.to_datetime(signals["event_time"]).dt.year):
            stats = _contrast_stats(yp, feature, label_col)
            auc = stats.get("auc", np.nan)
            count = int(stats.get("count", 0) or 0)
            if count < 20 or not np.isfinite(auc):
                continue
            valid_years += 1
            year_dir = np.sign(float(auc) - 0.5)
            if train_dir != 0 and year_dir == train_dir:
                same_direction_years += 1
            year_parts.append(f"{int(year)}:{count}:{float(auc):.4f}")
        row["valid_years"] = valid_years
        row["same_direction_years"] = same_direction_years
        row["year_auc_detail"] = ";".join(year_parts)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["direction_stable", "same_direction_years", "train_directional_strength", "holdout_directional_strength"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


