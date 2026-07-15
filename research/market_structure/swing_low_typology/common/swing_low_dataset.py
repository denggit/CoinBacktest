#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build retrospectively labelled swing-low events and causal pre-low features.

The future path is used only to decide whether an historical extreme qualifies
as a swing low. Every clustering feature is computed from the extreme bar or
older bars. The module is research-local because the swing definition and its
feature vocabulary belong to this experiment, not to the shared runtime API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover - standalone fallback
    ProgressReporter = None  # type: ignore[assignment]


EPS = 1e-12


@dataclass(frozen=True)
class ConfirmedSwingLow:
    extreme_pos: int
    entry_pos: int
    confirmation_pos: int
    extreme_price: float
    entry_price: float
    confirmation_price: float
    realized_move: float

    @property
    def completion_bars(self) -> int:
        """Number of future closed bars observed after the extreme bar."""

        return int(self.confirmation_pos - self.extreme_pos)


def _safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= EPS:
        return float(default)
    return float(num / den)


def _finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _finite_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def iter_confirmed_swing_lows(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    threshold: float,
) -> Iterable[ConfirmedSwingLow]:
    """Yield tradably confirmed swing lows.

    The structural extreme remains the historical bar ``low``.  Whether that
    low delivered the requested rebound is evaluated as a hypothetical causal
    trade: enter at the following bar ``open`` and require a later *closed-bar*
    ``close`` to reach the target.  Future intrabar highs and lows never decide
    target completion, MFE, or MAE.

    High/low are still allowed for the directional-change pivot structure.  A
    newly updated low cannot confirm on the same bar because its next-bar entry
    price does not exist yet.
    """

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    n = min(len(high), len(low), len(open_), len(close))
    if n < 2:
        return

    def low_ready(candidate_pos: int, observation_pos: int) -> bool:
        entry_pos = candidate_pos + 1
        if entry_pos >= n or observation_pos < entry_pos:
            return False
        entry = float(open_[entry_pos])
        observed_close = float(close[observation_pos])
        return bool(
            np.isfinite(entry)
            and entry > 0
            and np.isfinite(observed_close)
            and observed_close >= entry * (1.0 + threshold)
        )

    def make_event(candidate_pos: int, confirmation_pos: int) -> ConfirmedSwingLow:
        entry_pos = candidate_pos + 1
        extreme = float(low[candidate_pos])
        entry = float(open_[entry_pos])
        confirmation = float(close[confirmation_pos])
        return ConfirmedSwingLow(
            extreme_pos=candidate_pos,
            entry_pos=entry_pos,
            confirmation_pos=confirmation_pos,
            extreme_price=extreme,
            entry_price=entry,
            confirmation_price=confirmation,
            realized_move=confirmation / entry - 1.0,
        )

    candidate_high = 0
    candidate_low = 0
    mode: str | None = None  # high = seek/track high; low = seek/track low.

    for i in range(1, n):
        hi = float(high[i])
        lo = float(low[i])
        observed_close = float(close[i])
        if (
            not np.isfinite(hi)
            or not np.isfinite(lo)
            or not np.isfinite(observed_close)
            or hi <= 0
            or lo <= 0
        ):
            continue

        if mode is None:
            if hi > float(high[candidate_high]):
                candidate_high = i
            if lo < float(low[candidate_low]):
                candidate_low = i

            rebound_ready = low_ready(candidate_low, i)
            high_ready = i > candidate_high and lo <= float(high[candidate_high]) * (1.0 - threshold)

            if rebound_ready and high_ready:
                if candidate_low < candidate_high:
                    high_ready = False
                elif candidate_high < candidate_low:
                    rebound_ready = False
                else:
                    continue

            if rebound_ready:
                yield make_event(candidate_low, i)
                mode = "high"
                candidate_high = i
                continue

            if high_ready:
                mode = "low"
                candidate_low = i
                continue

        elif mode == "high":
            updated = False
            if hi > float(high[candidate_high]):
                candidate_high = i
                updated = True
            if not updated and i > candidate_high and lo <= float(high[candidate_high]) * (1.0 - threshold):
                mode = "low"
                candidate_low = i

        else:  # mode == "low"
            updated = False
            if lo < float(low[candidate_low]):
                candidate_low = i
                updated = True
            if not updated and low_ready(candidate_low, i):
                yield make_event(candidate_low, i)
                mode = "high"
                candidate_high = i


def detect_swing_lows(
    bars: pd.DataFrame,
    *,
    target_move_pct: float,
    max_completion_bars: int,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    minimum_history_bars: int,
) -> pd.DataFrame:
    """Detect low-anchored swing lows with tradable close confirmation.

    Structural low: current bar ``low``.
    Entry reference: following bar ``open``.
    Target observation: subsequent closed-bar ``close`` values only.
    """

    threshold = float(target_move_pct) / 100.0
    if not 0 < threshold < 1:
        raise ValueError("target_move_pct must be between 0 and 100")
    if max_completion_bars < 1:
        raise ValueError("max_completion_bars must be >= 1")

    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float, copy=True)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float, copy=True)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float, copy=True)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float, copy=True)
    timestamps = pd.DatetimeIndex(bars.index)
    diffs = timestamps.to_series().diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    bar_delta = positive_diffs.median() if not positive_diffs.empty else pd.Timedelta(minutes=1)

    rows: list[dict[str, object]] = []
    for swing in iter_confirmed_swing_lows(high, low, open_, close, threshold):
        if swing.completion_bars > max_completion_bars:
            continue
        if swing.extreme_pos < minimum_history_bars:
            continue
        extreme_time = timestamps[swing.extreme_pos]
        if not (research_start <= extreme_time < research_end_exclusive):
            continue
        entry_time = timestamps[swing.entry_pos]
        confirmation_time = timestamps[swing.confirmation_pos]
        rows.append(
            {
                "event_id": f"SL_{extreme_time.strftime('%Y%m%d_%H%M%S')}_{swing.extreme_pos}",
                "extreme_pos": int(swing.extreme_pos),
                "entry_pos": int(swing.entry_pos),
                "confirmation_pos": int(swing.confirmation_pos),
                "extreme_time": extreme_time,
                "feature_available_time": extreme_time + bar_delta,
                "entry_time": entry_time,
                "confirmation_time": confirmation_time,
                "confirmation_available_time": confirmation_time + bar_delta,
                "extreme_price": float(swing.extreme_price),
                "entry_price": float(swing.entry_price),
                "confirmation_price": float(swing.confirmation_price),
                "completion_bars": int(swing.completion_bars),
                "target_move_pct": float(target_move_pct),
                "realized_confirmation_move_pct": float(swing.realized_move * 100.0),
                "retrospective_label": True,
                "swing_extreme_price_source": "low",
                "swing_entry_price_source": "next_bar_open",
                "swing_target_observation_source": "future_closed_bar_close",
            }
        )

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values("extreme_time").reset_index(drop=True)


CORE_TRADE_BAR_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades_count",
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "taker_buy_ratio",
    "avg_trade_size",
    "vwap",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "large_trades_count",
    "max_trade_notional",
)


def validate_trade_bar_fields(bars: pd.DataFrame) -> pd.DataFrame:
    """Return field coverage and fail if rich trade-bar data is unavailable."""

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    unusable: list[str] = []
    for col in CORE_TRADE_BAR_FIELDS:
        exists = col in bars.columns
        if not exists:
            missing.append(col)
            rows.append({"field": col, "exists": False, "non_null_ratio": 0.0, "nunique": 0, "usable": False})
            continue
        values = pd.to_numeric(bars[col], errors="coerce")
        ratio = float(values.notna().mean())
        nunique = int(values.nunique(dropna=True))
        usable = ratio >= 0.95 and nunique > 1
        if not usable:
            unusable.append(col)
        rows.append({"field": col, "exists": True, "non_null_ratio": ratio, "nunique": nunique, "usable": usable})

    coverage = pd.DataFrame(rows)
    if missing or unusable:
        raise RuntimeError(
            "Swing-low typology requires rich trade-bar fields; "
            f"missing={missing}, unusable={unusable}. It will not silently fall back to OHLCV."
        )
    return coverage


def _feature_dictionary(windows: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        ("current_bar_return", "current", "当前K线收益", "close/open-1"),
        ("current_range_pct", "current", "当前振幅", "(high-low)/prev_close"),
        ("current_body_pct", "current", "当前实体幅度", "abs(close-open)/prev_close"),
        ("current_lower_wick_share", "current", "当前下影线占比", "lower_wick/range"),
        ("current_close_position", "current", "当前收盘位置", "(close-low)/(high-low)"),
        ("current_low_vs_prev_close", "current", "低点相对前收跌幅", "low/prev_close-1"),
        ("current_delta_ratio", "current_orderflow", "当前主动Delta占比", "delta_notional/notional"),
        ("current_large_delta_ratio", "current_orderflow", "当前大单Delta占比", "large_delta/large_gross"),
        ("current_buy_share", "current_orderflow", "当前主动买入占比", "buy_notional/notional"),
        ("current_notional_vs_prev30_median", "current_activity", "当前成交额倍率", "notional/过去30根中位数"),
        ("current_trades_vs_prev30_median", "current_activity", "当前成交笔数倍率", "trades/过去30根中位数"),
        ("current_avg_trade_size_vs_prev30_median", "current_activity", "当前平均单笔倍率", "avg_trade_size/过去30根中位数"),
        ("current_max_trade_share", "current_orderflow", "当前最大单占比", "max_trade_notional/notional"),
        ("current_large_trade_share", "current_orderflow", "当前大单成交占比", "large_gross/notional"),
        ("current_close_vs_vwap", "current", "收盘相对VWAP", "close/vwap-1"),
        ("hour_sin", "time", "小时周期正弦", "sin(hour/24)"),
        ("hour_cos", "time", "小时周期余弦", "cos(hour/24)"),
        ("is_weekend", "time", "是否周末", "Saturday/Sunday"),
    ]
    out = [dict(zip(("feature", "family", "label", "definition"), row)) for row in rows]
    templates = [
        ("close_return", "price_path", "窗口收盘收益", "close_t/close_start-1"),
        ("low_return", "price_path", "窗口低点跌幅", "low_t/close_start-1"),
        ("drawdown_from_high", "price_structure", "距窗口最高点回撤", "low_t/max(high)-1"),
        ("close_position", "price_structure", "窗口收盘位置", "(close_t-min_low)/(max_high-min_low)"),
        ("realized_vol", "volatility", "窗口已实现波动", "std(log_return)"),
        ("path_efficiency", "price_path", "价格路径效率", "abs(net_move)/sum(abs(bar_moves))"),
        ("down_bar_share", "price_path", "阴线占比", "mean(close<open)"),
        ("lower_low_share", "price_structure", "连续创新低占比", "mean(low_t<low_t-1)"),
        ("near_low_test_share", "price_structure", "历史近低点测试占比", "prior_low <= extreme_low*1.15‰"),
        ("max_rebound_before_low", "price_structure", "下跌途中最大反弹", "max(high/running_min_low-1)"),
        ("range_expansion", "volatility", "当前振幅扩张倍数", "current_range/median(prior_range)"),
        ("notional_intensity", "activity", "当前成交额相对窗口", "current_notional/median(window_notional)"),
        ("trades_intensity", "activity", "当前成交笔数相对窗口", "current_trades/median(window_trades)"),
        ("delta_ratio", "orderflow", "窗口主动Delta占比", "sum(delta)/sum(notional)"),
        ("large_delta_ratio", "orderflow", "窗口大单Delta占比", "sum(large_delta)/sum(large_gross)"),
        ("negative_delta_share", "orderflow", "主动卖出持续占比", "mean(delta<0)"),
        ("delta_acceleration", "orderflow_path", "Delta前后半段变化", "second_half_delta_ratio-first_half"),
        ("large_delta_acceleration", "orderflow_path", "大单Delta前后半段变化", "second_half_large_delta-first_half"),
        ("large_trade_share", "orderflow", "窗口大单成交占比", "sum(large_gross)/sum(notional)"),
        ("max_trade_concentration", "orderflow", "窗口最大单集中度", "max(max_trade_notional)/sum(notional)"),
        ("price_delta_dislocation", "absorption", "价格与Delta错位", "close_return-delta_ratio"),
    ]
    for window in windows:
        for prefix, family, label, definition in templates:
            out.append(
                {
                    "feature": f"{prefix}_{window}",
                    "family": family,
                    "label": f"{window} bars {label}",
                    "definition": definition,
                }
            )
    return pd.DataFrame(out)


def build_causal_features(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Sequence[int],
    progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build multi-scale features using only bars up to each extreme position."""

    if events.empty:
        return events.copy(), _feature_dictionary(windows)
    windows = tuple(sorted({int(w) for w in windows if int(w) >= 2}))
    if not windows:
        raise ValueError("windows must contain at least one integer >= 2")
    max_window = max(windows)

    numeric: dict[str, np.ndarray] = {}
    for col in CORE_TRADE_BAR_FIELDS:
        numeric[col] = pd.to_numeric(bars[col], errors="coerce").to_numpy(dtype=float, copy=False)
    large_gross = numeric["large_buy_notional"] + numeric["large_sell_notional"]
    bar_range = numeric["high"] - numeric["low"]

    rows: list[dict[str, object]] = []
    total = len(events)
    reporter = ProgressReporter("[features] causal swing-low windows", total=total, every=max(1, progress_every)) if ProgressReporter else None

    for idx, event in events.iterrows():
        pos = int(event["extreme_pos"])
        if pos < max_window:
            continue
        ts = pd.Timestamp(event["extreme_time"])
        prev_close = numeric["close"][pos - 1]
        rng = bar_range[pos]
        lower_wick = min(numeric["open"][pos], numeric["close"][pos]) - numeric["low"][pos]
        prev30_start = max(0, pos - 30)
        prev30 = slice(prev30_start, pos)

        row: dict[str, object] = {
            "event_id": event["event_id"],
            "extreme_time": ts,
            "feature_available_time": pd.Timestamp(event["feature_available_time"]),
            "extreme_pos": pos,
            "extreme_price": float(event["extreme_price"]),
            "confirmation_time": pd.Timestamp(event["confirmation_time"]),
            "confirmation_available_time": pd.Timestamp(event["confirmation_available_time"]),
            "completion_bars": int(event["completion_bars"]),
            "realized_confirmation_move_pct": float(event["realized_confirmation_move_pct"]),
            "current_bar_return": _safe_ratio(numeric["close"][pos], numeric["open"][pos]) - 1.0,
            "current_range_pct": _safe_ratio(rng, prev_close),
            "current_body_pct": _safe_ratio(abs(numeric["close"][pos] - numeric["open"][pos]), prev_close),
            "current_lower_wick_share": _safe_ratio(lower_wick, rng),
            "current_close_position": _safe_ratio(numeric["close"][pos] - numeric["low"][pos], rng),
            "current_low_vs_prev_close": _safe_ratio(numeric["low"][pos], prev_close) - 1.0,
            "current_delta_ratio": _safe_ratio(numeric["delta_notional"][pos], numeric["notional"][pos]),
            "current_large_delta_ratio": _safe_ratio(numeric["large_delta_notional"][pos], large_gross[pos]),
            "current_buy_share": _safe_ratio(numeric["buy_notional"][pos], numeric["notional"][pos]),
            "current_notional_vs_prev30_median": _safe_ratio(numeric["notional"][pos], _finite_median(numeric["notional"][prev30])),
            "current_trades_vs_prev30_median": _safe_ratio(numeric["trades_count"][pos], _finite_median(numeric["trades_count"][prev30])),
            "current_avg_trade_size_vs_prev30_median": _safe_ratio(numeric["avg_trade_size"][pos], _finite_median(numeric["avg_trade_size"][prev30])),
            "current_max_trade_share": _safe_ratio(numeric["max_trade_notional"][pos], numeric["notional"][pos]),
            "current_large_trade_share": _safe_ratio(large_gross[pos], numeric["notional"][pos]),
            "current_close_vs_vwap": _safe_ratio(numeric["close"][pos], numeric["vwap"][pos]) - 1.0,
            "hour_sin": float(np.sin(2.0 * np.pi * ts.hour / 24.0)),
            "hour_cos": float(np.cos(2.0 * np.pi * ts.hour / 24.0)),
            "is_weekend": float(ts.dayofweek >= 5),
        }

        for window in windows:
            start = pos - window + 1
            sl = slice(start, pos + 1)
            close = numeric["close"][sl]
            open_ = numeric["open"][sl]
            high = numeric["high"][sl]
            low = numeric["low"][sl]
            notional = numeric["notional"][sl]
            trades = numeric["trades_count"][sl]
            delta = numeric["delta_notional"][sl]
            large_delta = numeric["large_delta_notional"][sl]
            large = large_gross[sl]
            max_trade = numeric["max_trade_notional"][sl]
            ranges = bar_range[sl]

            close_start = close[0]
            close_return = _safe_ratio(close[-1], close_start) - 1.0
            low_return = _safe_ratio(numeric["low"][pos], close_start) - 1.0
            max_high = float(np.nanmax(high))
            min_low = float(np.nanmin(low))
            drawdown = _safe_ratio(numeric["low"][pos], max_high) - 1.0
            close_position = _safe_ratio(close[-1] - min_low, max_high - min_low)
            valid_close = close[np.isfinite(close) & (close > 0)]
            log_ret = np.diff(np.log(valid_close)) if valid_close.size >= 2 else np.array([], dtype=float)
            abs_path = float(np.nansum(np.abs(np.diff(close))))
            efficiency = _safe_ratio(abs(close[-1] - close[0]), abs_path)
            down_share = _finite_mean((close < open_).astype(float))
            lower_low_share = _finite_mean((low[1:] < low[:-1]).astype(float)) if len(low) > 1 else np.nan
            prior_low = low[:-1]
            near_low_share = _finite_mean((prior_low <= numeric["low"][pos] * 1.0015).astype(float)) if len(prior_low) else np.nan
            running_min = np.minimum.accumulate(low)
            rebound = np.divide(high, running_min, out=np.full_like(high, np.nan), where=np.abs(running_min) > EPS) - 1.0
            max_rebound = float(np.nanmax(rebound)) if np.isfinite(rebound).any() else np.nan
            prior_ranges = ranges[:-1]
            range_expansion = _safe_ratio(ranges[-1], _finite_median(prior_ranges))
            notional_intensity = _safe_ratio(notional[-1], _finite_median(notional))
            trades_intensity = _safe_ratio(trades[-1], _finite_median(trades))
            delta_ratio = _safe_ratio(float(np.nansum(delta)), float(np.nansum(notional)))
            large_delta_ratio = _safe_ratio(float(np.nansum(large_delta)), float(np.nansum(large)))
            negative_delta_share = _finite_mean((delta < 0).astype(float))

            half = max(1, len(delta) // 2)
            first_delta_ratio = _safe_ratio(float(np.nansum(delta[:half])), float(np.nansum(notional[:half])))
            second_delta_ratio = _safe_ratio(float(np.nansum(delta[half:])), float(np.nansum(notional[half:])))
            first_large_ratio = _safe_ratio(float(np.nansum(large_delta[:half])), float(np.nansum(large[:half])))
            second_large_ratio = _safe_ratio(float(np.nansum(large_delta[half:])), float(np.nansum(large[half:])))

            row.update(
                {
                    f"close_return_{window}": close_return,
                    f"low_return_{window}": low_return,
                    f"drawdown_from_high_{window}": drawdown,
                    f"close_position_{window}": close_position,
                    f"realized_vol_{window}": _finite_std(log_ret),
                    f"path_efficiency_{window}": efficiency,
                    f"down_bar_share_{window}": down_share,
                    f"lower_low_share_{window}": lower_low_share,
                    f"near_low_test_share_{window}": near_low_share,
                    f"max_rebound_before_low_{window}": max_rebound,
                    f"range_expansion_{window}": range_expansion,
                    f"notional_intensity_{window}": notional_intensity,
                    f"trades_intensity_{window}": trades_intensity,
                    f"delta_ratio_{window}": delta_ratio,
                    f"large_delta_ratio_{window}": large_delta_ratio,
                    f"negative_delta_share_{window}": negative_delta_share,
                    f"delta_acceleration_{window}": second_delta_ratio - first_delta_ratio,
                    f"large_delta_acceleration_{window}": second_large_ratio - first_large_ratio,
                    f"large_trade_share_{window}": _safe_ratio(float(np.nansum(large)), float(np.nansum(notional))),
                    f"max_trade_concentration_{window}": _safe_ratio(float(np.nanmax(max_trade)), float(np.nansum(notional))),
                    f"price_delta_dislocation_{window}": close_return - delta_ratio,
                }
            )

        rows.append(row)
        if reporter and idx + 1 < total:
            reporter.update(idx + 1)

    if reporter:
        reporter.close()
    features = pd.DataFrame(rows).sort_values("extreme_time").reset_index(drop=True)
    return features, _feature_dictionary(windows)


def build_pre_low_path_profiles(
    bars: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    lookback_bars: int = 120,
    max_samples_per_cluster_split: int = 800,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aggregate only the historical path ending at the labelled low."""

    if assignments.empty:
        return pd.DataFrame()
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float, copy=False)
    notional = pd.to_numeric(bars["notional"], errors="coerce").to_numpy(dtype=float, copy=False)
    delta = pd.to_numeric(bars["delta_notional"], errors="coerce").to_numpy(dtype=float, copy=False)
    rng = np.random.default_rng(random_state)
    out: list[dict[str, object]] = []

    for (cluster_id, split), group in assignments.groupby(["cluster_id", "split"], sort=True):
        valid = group[group["extreme_pos"] >= lookback_bars]
        if len(valid) > max_samples_per_cluster_split:
            chosen = rng.choice(valid.index.to_numpy(), size=max_samples_per_cluster_split, replace=False)
            valid = valid.loc[chosen]
        price_paths: list[np.ndarray] = []
        delta_paths: list[np.ndarray] = []
        activity_paths: list[np.ndarray] = []
        for row in valid.itertuples(index=False):
            pos = int(row.extreme_pos)
            sl = slice(pos - lookback_bars, pos + 1)
            extreme = low[pos]
            prices = close[sl] / extreme - 1.0
            notionals = notional[sl]
            deltas = np.divide(delta[sl], notionals, out=np.full(lookback_bars + 1, np.nan), where=np.abs(notionals) > EPS)
            med = _finite_median(notionals[:-1])
            activity = notionals / med if np.isfinite(med) and abs(med) > EPS else np.full(lookback_bars + 1, np.nan)
            price_paths.append(prices)
            delta_paths.append(deltas)
            activity_paths.append(activity)
        if not price_paths:
            continue
        p = np.asarray(price_paths, dtype=float)
        d = np.asarray(delta_paths, dtype=float)
        a = np.asarray(activity_paths, dtype=float)
        for j, offset in enumerate(range(-lookback_bars, 1)):
            out.append(
                {
                    "cluster_id": cluster_id,
                    "split": split,
                    "offset_bars": offset,
                    "sample_count": int(len(p)),
                    "median_close_vs_extreme": float(np.nanmedian(p[:, j])),
                    "q25_close_vs_extreme": float(np.nanquantile(p[:, j], 0.25)),
                    "q75_close_vs_extreme": float(np.nanquantile(p[:, j], 0.75)),
                    "median_bar_delta_ratio": float(np.nanmedian(d[:, j])),
                    "median_notional_ratio": float(np.nanmedian(a[:, j])),
                }
            )
    return pd.DataFrame(out)


def build_causal_audit(features: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    forbidden_prefixes = ("future_", "post_", "forward_", "confirmation_", "completion_")
    forbidden_exact = {"confirmation_time", "confirmation_price", "completion_bars", "realized_confirmation_move_pct", "mfe", "mae"}
    forbidden_features = [
        c for c in feature_columns
        if c.lower() in forbidden_exact or c.lower().startswith(forbidden_prefixes)
    ]
    if features.empty:
        return pd.DataFrame(
            [{"check": "non_empty_features", "passed": False, "detail": "no events/features"}]
        )
    feature_time = pd.to_datetime(features["feature_available_time"])
    extreme_time = pd.to_datetime(features["extreme_time"])
    confirmation_available = pd.to_datetime(features["confirmation_available_time"])
    cutoff_ok = bool((feature_time > extreme_time).all())
    confirmation_after = bool((confirmation_available > feature_time).all())
    return pd.DataFrame(
        [
            {"check": "extreme_bar_features_available_after_close", "passed": cutoff_ok, "detail": "feature_available_time is after left-labelled extreme bar start"},
            {"check": "future_confirmation_available_after_features", "passed": confirmation_after, "detail": "future confirmation is label only"},
            {"check": "no_future_named_features", "passed": not forbidden_features, "detail": ",".join(forbidden_features)},
            {"check": "feature_count", "passed": len(feature_columns) > 10, "detail": str(len(feature_columns))},
        ]
    )
