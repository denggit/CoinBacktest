"""Causal orange-to-green path features for panic recovery research.

This module is specific to the panic selloff/rejection/recovery research family.
Every candidate feature ends at the closed green signal bar. Post-green path
statistics are produced separately and must never be used by a signal filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover - compatibility with older project baselines
    ProgressReporter = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PathFeatureDef:
    feature: str
    family: str
    description: str
    scope: str = "orange_to_green_path"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-12:
        return np.nan
    return float(num / den)


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _segment_ratio(frame: pd.DataFrame, num_col: str, den_col: str) -> float:
    num = _series(frame, num_col).sum(min_count=1)
    den = _series(frame, den_col).sum(min_count=1)
    return _safe_ratio(_finite(num), _finite(den))


def _mean(frame: pd.DataFrame, name: str) -> float:
    s = _series(frame, name)
    return _finite(s.mean()) if s.notna().any() else np.nan


def _max(frame: pd.DataFrame, name: str) -> float:
    s = _series(frame, name)
    return _finite(s.max()) if s.notna().any() else np.nan


def _min(frame: pd.DataFrame, name: str) -> float:
    s = _series(frame, name)
    return _finite(s.min()) if s.notna().any() else np.nan


def _last(frame: pd.DataFrame, name: str) -> float:
    s = _series(frame, name).dropna()
    return _finite(s.iloc[-1]) if not s.empty else np.nan


def _slope(values: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)[valid]
    y = y[valid]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def _efficiency(close: pd.Series) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 2:
        return np.nan
    path = float(c.diff().abs().sum())
    return _safe_ratio(float(c.iloc[-1] - c.iloc[0]), path)


def _split_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, frame
    cut = max(1, int(np.ceil(len(frame) / 2)))
    return frame.iloc[:cut], frame.iloc[cut:]


def _quarter_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return [frame.copy() for _ in range(4)]
    positions = np.array_split(np.arange(len(frame)), 4)
    return [frame.iloc[pos] if len(pos) else frame.iloc[0:0] for pos in positions]


def _strict_new_low_count(lows: pd.Series) -> int:
    arr = pd.to_numeric(lows, errors="coerce").to_numpy(dtype=float)
    count = 0
    running = np.inf
    for value in arr:
        if np.isfinite(value) and value < running - max(abs(running) if np.isfinite(running) else 0.0, 1.0) * 1e-12:
            count += 1
            running = value
    return count


def _failed_bounce_count(close: pd.Series, *, min_bounce_pct: float = 0.0010) -> int:
    arr = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    if len(arr) < 3:
        return 0
    running_low = np.inf
    in_bounce = False
    count = 0
    for value in arr:
        if not np.isfinite(value):
            continue
        if value < running_low:
            if in_bounce:
                count += 1
            running_low = value
            in_bounce = False
        elif running_low > 0 and value / running_low - 1.0 >= min_bounce_pct:
            in_bounce = True
    return count


def _longest_true_run(mask: Iterable[bool]) -> int:
    best = current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _register(defs: list[PathFeatureDef], feature: str, family: str, description: str) -> None:
    defs.append(PathFeatureDef(feature=feature, family=family, description=description))


def build_orange_to_green_path_features(
    bars: pd.DataFrame,
    orderflow: pd.DataFrame,
    stage_events: pd.DataFrame,
    *,
    low_retest_tolerance_pct: float = 0.0008,
    progress_enabled: bool = True,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one causal path row per green signal.

    The slice is inclusive of the orange/start closed bar and green/signal
    closed bar. No value after ``green_time`` is read.
    """
    if stage_events.empty:
        return pd.DataFrame(), pd.DataFrame()
    bars = bars.sort_index().copy()
    orderflow = orderflow.sort_index().reindex(bars.index)
    signals = stage_events[stage_events["stage"] == "signal"].copy()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    exhaustion_map = (
        stage_events[stage_events["stage"] == "exhaustion"]
        .sort_values("event_time")
        .groupby("episode_id")["event_time"]
        .first()
        .to_dict()
    )
    pos = pd.Series(np.arange(len(bars), dtype=int), index=bars.index)
    rows: list[dict[str, Any]] = []
    defs: list[PathFeatureDef] = []

    reporter = None
    if ProgressReporter is not None:
        reporter = ProgressReporter(
            "[path] orange-to-green features",
            len(signals),
            every=max(1, int(progress_every)),
            enabled=progress_enabled,
        )

    for done, signal in enumerate(signals.itertuples(index=False), start=1):
        episode_id = int(signal.episode_id)
        orange_time = pd.Timestamp(signal.episode_start_time)
        green_time = pd.Timestamp(signal.event_time)
        if orange_time not in bars.index or green_time not in bars.index or green_time < orange_time:
            if reporter is not None:
                reporter.update(done)
            continue
        orange_pos = int(pos.at[orange_time])
        green_pos = int(pos.at[green_time])
        price_path = bars.iloc[orange_pos : green_pos + 1]
        flow_path = orderflow.iloc[orange_pos : green_pos + 1]
        if price_path.empty:
            if reporter is not None:
                reporter.update(done)
            continue

        low_time = pd.Timestamp(price_path["low"].idxmin())
        low_pos = int(pos.at[low_time])
        sell_price = bars.iloc[orange_pos : low_pos + 1]
        sell_flow = orderflow.iloc[orange_pos : low_pos + 1]
        recovery_price = bars.iloc[low_pos : green_pos + 1]
        recovery_flow = orderflow.iloc[low_pos : green_pos + 1]
        rec_early_price, rec_late_price = _split_halves(recovery_price)
        rec_early_flow, rec_late_flow = _split_halves(recovery_flow)

        orange_close = _finite(bars.at[orange_time, "close"])
        low_price = _finite(price_path["low"].min())
        green_close = _finite(bars.at[green_time, "close"])
        sell_drop = _safe_ratio(low_price - orange_close, orange_close)
        recovery = _safe_ratio(green_close - low_price, low_price)
        recovery_fraction = _safe_ratio(green_close - low_price, orange_close - low_price)

        lows_after = pd.to_numeric(recovery_price["low"], errors="coerce")
        retest_level = low_price * (1.0 + float(low_retest_tolerance_pct))
        low_retests = int((lows_after <= retest_level).sum()) if np.isfinite(low_price) else 0
        closes_rec = pd.to_numeric(recovery_price["close"], errors="coerce")
        highs_rec = pd.to_numeric(recovery_price["high"], errors="coerce")
        positive_frac = _finite((closes_rec.diff() > 0).mean())
        higher_low_frac = _finite((lows_after.diff() > 0).mean())
        close_above_prev_high_frac = _finite((closes_rec > highs_rec.shift(1)).mean())

        early_return = (
            _safe_ratio(_finite(rec_early_price["close"].iloc[-1]) - low_price, low_price)
            if not rec_early_price.empty else np.nan
        )
        late_start = _finite(rec_late_price["close"].iloc[0]) if not rec_late_price.empty else np.nan
        late_end = _finite(rec_late_price["close"].iloc[-1]) if not rec_late_price.empty else np.nan
        late_return = _safe_ratio(late_end - late_start, late_start)

        exhaustion_time_raw = exhaustion_map.get(episode_id)
        exhaustion_time = pd.Timestamp(exhaustion_time_raw) if exhaustion_time_raw is not None else pd.NaT
        exhaustion_to_green = (
            green_pos - int(pos.at[exhaustion_time])
            if pd.notna(exhaustion_time) and exhaustion_time in pos.index else np.nan
        )

        row: dict[str, Any] = {
            "episode_id": episode_id,
            "orange_time": orange_time,
            "low_time": low_time,
            "exhaustion_time": exhaustion_time,
            "green_time": green_time,
            "path_window_end": green_time,
            "path_bars_total": green_pos - orange_pos,
            "path_bars_orange_to_low": low_pos - orange_pos,
            "path_bars_low_to_green": green_pos - low_pos,
            "path_bars_exhaustion_to_green": exhaustion_to_green,
            "path_low_position_fraction": _safe_ratio(low_pos - orange_pos, max(1, green_pos - orange_pos)),
            "path_orange_to_low_return": sell_drop,
            "path_low_to_green_rebound": recovery,
            "path_recovery_fraction": recovery_fraction,
            "path_green_vs_orange_return": _safe_ratio(green_close - orange_close, orange_close),
            "path_signal_close_risk_pct": _safe_ratio(green_close - low_price, green_close),
            "path_sell_price_efficiency": abs(_efficiency(sell_price["close"])),
            "path_recovery_price_efficiency": _efficiency(recovery_price["close"]),
            "path_new_low_count": _strict_new_low_count(price_path["low"]),
            "path_failed_bounce_count": _failed_bounce_count(sell_price["close"]),
            "path_low_retest_count": low_retests,
            "path_recovery_positive_bar_fraction": positive_frac,
            "path_recovery_higher_low_fraction": higher_low_frac,
            "path_recovery_close_above_prev_high_fraction": close_above_prev_high_frac,
            "path_recovery_early_return": early_return,
            "path_recovery_late_return": late_return,
            "path_recovery_acceleration": late_return - early_return if np.isfinite(late_return) and np.isfinite(early_return) else np.nan,
            "path_recovery_speed_per_bar": _safe_ratio(recovery, max(1, green_pos - low_pos)),
            "path_sell_delta_ratio": _segment_ratio(sell_flow, "delta_notional", "notional"),
            "path_recovery_delta_ratio": _segment_ratio(recovery_flow, "delta_notional", "notional"),
            "path_recovery_early_delta_ratio": _segment_ratio(rec_early_flow, "delta_notional", "notional"),
            "path_recovery_late_delta_ratio": _segment_ratio(rec_late_flow, "delta_notional", "notional"),
            "path_sell_large_delta_ratio": _segment_ratio(sell_flow, "large_delta_notional", "large_notional"),
            "path_recovery_large_delta_ratio": _segment_ratio(recovery_flow, "large_delta_notional", "large_notional"),
            "path_recovery_early_large_delta_ratio": _segment_ratio(rec_early_flow, "large_delta_notional", "large_notional"),
            "path_recovery_late_large_delta_ratio": _segment_ratio(rec_late_flow, "large_delta_notional", "large_notional"),
            "path_delta_recovery": _segment_ratio(recovery_flow, "delta_notional", "notional") - _segment_ratio(sell_flow, "delta_notional", "notional"),
            "path_large_delta_recovery": _segment_ratio(recovery_flow, "large_delta_notional", "large_notional") - _segment_ratio(sell_flow, "large_delta_notional", "large_notional"),
            "path_delta_late_vs_early": _segment_ratio(rec_late_flow, "delta_notional", "notional") - _segment_ratio(rec_early_flow, "delta_notional", "notional"),
            "path_large_delta_late_vs_early": _segment_ratio(rec_late_flow, "large_delta_notional", "large_notional") - _segment_ratio(rec_early_flow, "large_delta_notional", "large_notional"),
            "path_sell_taker_buy_mean": _mean(sell_flow, "taker_buy_ratio"),
            "path_recovery_taker_buy_mean": _mean(recovery_flow, "taker_buy_ratio"),
            "path_recovery_taker_buy_last": _last(recovery_flow, "taker_buy_ratio_2"),
            "path_recovery_positive_delta_fraction": _finite((_series(recovery_flow, "delta_ratio") > 0).mean()),
            "path_recovery_positive_large_delta_fraction": _finite((_series(recovery_flow, "large_delta_ratio") > 0).mean()),
            "path_recovery_positive_delta_longest_run": _longest_true_run(_series(recovery_flow, "delta_ratio").fillna(-1) > 0),
            "path_sell_intensity_peak": _max(sell_flow, "sell_notional_ratio_base"),
            "path_recovery_sell_intensity_mean": _mean(recovery_flow, "sell_notional_ratio_base"),
            "path_recovery_sell_intensity_last": _last(recovery_flow, "sell_notional_ratio_base"),
            "path_sell_intensity_decay": _safe_ratio(_last(recovery_flow, "sell_notional_ratio_base"), _max(sell_flow, "sell_notional_ratio_base")),
            "path_recovery_trades_intensity_mean": _mean(recovery_flow, "trades_ratio_base"),
            "path_recovery_notional_intensity_mean": _mean(recovery_flow, "notional_ratio_base"),
            "path_recovery_max_trade_share": _max(recovery_flow, "max_trade_share"),
            "path_recovery_large_trade_share": _mean(recovery_flow, "large_trade_share"),
            "path_low_absorption_score": _finite(orderflow.at[low_time, "absorption_score"]) if "absorption_score" in orderflow else np.nan,
            "path_low_close_position": _finite(orderflow.at[low_time, "close_pos"]) if "close_pos" in orderflow else np.nan,
            "path_low_lower_wick_fraction": _finite(orderflow.at[low_time, "lower_wick_frac"]) if "lower_wick_frac" in orderflow else np.nan,
            "path_low_delta_ratio": _finite(orderflow.at[low_time, "delta_ratio"]) if "delta_ratio" in orderflow else np.nan,
            "path_low_large_delta_ratio": _finite(orderflow.at[low_time, "large_delta_ratio"]) if "large_delta_ratio" in orderflow else np.nan,
            "path_low_delta_vs_sell_min": _finite(orderflow.at[low_time, "delta_ratio"]) - _min(sell_flow.iloc[:-1], "delta_ratio"),
            "path_low_large_delta_vs_sell_min": _finite(orderflow.at[low_time, "large_delta_ratio"]) - _min(sell_flow.iloc[:-1], "large_delta_ratio"),
            "path_recovery_delta_slope": _slope(_series(recovery_flow, "delta_ratio")),
            "path_recovery_large_delta_slope": _slope(_series(recovery_flow, "large_delta_ratio")),
            "path_recovery_taker_buy_slope": _slope(_series(recovery_flow, "taker_buy_ratio")),
            "path_recovery_sell_intensity_slope": _slope(_series(recovery_flow, "sell_notional_ratio_base")),
            "path_recovery_price_slope_pct": _safe_ratio(_slope(recovery_price["close"]), low_price),
            "path_recovery_flow_price_divergence": _segment_ratio(recovery_flow, "delta_notional", "notional") - recovery,
            "path_recovery_cvd_from_min": np.nan,
            "path_recovery_large_cvd_from_min": np.nan,
        }

        delta = _series(flow_path, "delta_notional").fillna(0.0)
        notion = _series(flow_path, "notional").fillna(0.0)
        cvd = delta.cumsum()
        large_delta = _series(flow_path, "large_delta_notional").fillna(0.0)
        large_notional = _series(flow_path, "large_notional").fillna(0.0)
        large_cvd = large_delta.cumsum()
        row["path_recovery_cvd_from_min"] = _safe_ratio(float(cvd.iloc[-1] - cvd.min()), float(notion.sum()))
        row["path_recovery_large_cvd_from_min"] = _safe_ratio(float(large_cvd.iloc[-1] - large_cvd.min()), float(large_notional.sum()))

        for q, (price_q, flow_q) in enumerate(zip(_quarter_segments(price_path), _quarter_segments(flow_path)), start=1):
            q_open = _finite(price_q["open"].iloc[0]) if not price_q.empty else np.nan
            q_close = _finite(price_q["close"].iloc[-1]) if not price_q.empty else np.nan
            row[f"path_q{q}_price_return"] = _safe_ratio(q_close - q_open, q_open)
            row[f"path_q{q}_delta_ratio"] = _segment_ratio(flow_q, "delta_notional", "notional")
            row[f"path_q{q}_large_delta_ratio"] = _segment_ratio(flow_q, "large_delta_notional", "large_notional")
            row[f"path_q{q}_sell_intensity"] = _mean(flow_q, "sell_notional_ratio_base")
            row[f"path_q{q}_absorption"] = _mean(flow_q, "absorption_score")

        rows.append(row)
        if reporter is not None and done < len(signals):
            reporter.update(done)

    if reporter is not None:
        reporter.close()

    # Metadata is deliberately built from a fixed allow-list. Timestamp fields
    # and post-green outcomes can never silently become candidate features.
    duration = {
        "path_bars_total": "橙灯到绿灯持续bars",
        "path_bars_orange_to_low": "橙灯到已知最低点持续bars",
        "path_bars_low_to_green": "最低点到绿灯持续bars",
        "path_bars_exhaustion_to_green": "黄灯到绿灯持续bars",
        "path_low_position_fraction": "最低点在橙灯到绿灯路径中的相对位置",
    }
    price = {
        "path_orange_to_low_return": "橙灯收盘到最低点跌幅",
        "path_low_to_green_rebound": "最低点到绿灯收盘反弹",
        "path_recovery_fraction": "绿灯收复橙灯到低点跌幅比例",
        "path_green_vs_orange_return": "绿灯相对橙灯收盘收益",
        "path_signal_close_risk_pct": "绿灯收盘到紫灯低点距离",
        "path_sell_price_efficiency": "下跌路径方向效率",
        "path_recovery_price_efficiency": "恢复路径方向效率",
        "path_new_low_count": "路径内严格创新低次数",
        "path_failed_bounce_count": "最终低点前失败反弹次数",
        "path_low_retest_count": "最低点附近重复测试次数",
        "path_recovery_positive_bar_fraction": "恢复段上涨bar比例",
        "path_recovery_higher_low_fraction": "恢复段抬高低点比例",
        "path_recovery_close_above_prev_high_fraction": "恢复段收盘突破前高比例",
        "path_recovery_early_return": "恢复段前半收益",
        "path_recovery_late_return": "恢复段后半收益",
        "path_recovery_acceleration": "恢复段后半相对前半加速",
        "path_recovery_speed_per_bar": "最低点到绿灯每bar恢复速度",
        "path_recovery_price_slope_pct": "恢复段价格归一化斜率",
    }
    flow = {
        "path_sell_delta_ratio": "下探段主动成交净额比例",
        "path_recovery_delta_ratio": "恢复段主动成交净额比例",
        "path_recovery_early_delta_ratio": "恢复前半主动成交净额比例",
        "path_recovery_late_delta_ratio": "恢复后半主动成交净额比例",
        "path_sell_large_delta_ratio": "下探段大单净额比例",
        "path_recovery_large_delta_ratio": "恢复段大单净额比例",
        "path_recovery_early_large_delta_ratio": "恢复前半大单净额比例",
        "path_recovery_late_large_delta_ratio": "恢复后半大单净额比例",
        "path_delta_recovery": "恢复段相对下探段delta改善",
        "path_large_delta_recovery": "恢复段相对下探段大单delta改善",
        "path_delta_late_vs_early": "恢复后半相对前半delta改善",
        "path_large_delta_late_vs_early": "恢复后半相对前半大单delta改善",
        "path_sell_taker_buy_mean": "下探段主动买入比例",
        "path_recovery_taker_buy_mean": "恢复段主动买入比例",
        "path_recovery_taker_buy_last": "绿灯附近主动买入比例",
        "path_recovery_positive_delta_fraction": "恢复段正delta bar比例",
        "path_recovery_positive_large_delta_fraction": "恢复段正大单delta bar比例",
        "path_recovery_positive_delta_longest_run": "恢复段连续正delta最长长度",
        "path_recovery_delta_slope": "恢复段delta路径斜率",
        "path_recovery_large_delta_slope": "恢复段大单delta路径斜率",
        "path_recovery_taker_buy_slope": "恢复段主动买入比例斜率",
        "path_recovery_flow_price_divergence": "恢复段delta与价格恢复差",
        "path_recovery_cvd_from_min": "路径CVD从最低值恢复幅度",
        "path_recovery_large_cvd_from_min": "路径大单CVD从最低值恢复幅度",
    }
    activity = {
        "path_sell_intensity_peak": "下探段卖出成交额强度峰值",
        "path_recovery_sell_intensity_mean": "恢复段卖出成交额强度均值",
        "path_recovery_sell_intensity_last": "绿灯附近卖出成交额强度",
        "path_sell_intensity_decay": "绿灯卖压相对下探峰值衰减比例",
        "path_recovery_trades_intensity_mean": "恢复段成交笔数强度",
        "path_recovery_notional_intensity_mean": "恢复段成交额强度",
        "path_recovery_max_trade_share": "恢复段最大单笔占比",
        "path_recovery_large_trade_share": "恢复段大单成交占比",
        "path_recovery_sell_intensity_slope": "恢复段卖压强度斜率",
    }
    absorption = {
        "path_low_absorption_score": "最低点吸收分数",
        "path_low_close_position": "最低点bar收盘位置",
        "path_low_lower_wick_fraction": "最低点bar下影比例",
        "path_low_delta_ratio": "最低点主动成交净额比例",
        "path_low_large_delta_ratio": "最低点大单净额比例",
        "path_low_delta_vs_sell_min": "最低点delta相对前序最差值改善",
        "path_low_large_delta_vs_sell_min": "最低点大单delta相对前序最差值改善",
    }
    for feature, description in duration.items():
        _register(defs, feature, "path_timing", description)
    for feature, description in price.items():
        _register(defs, feature, "price_path", description)
    for feature, description in flow.items():
        _register(defs, feature, "flow_path", description)
    for feature, description in activity.items():
        _register(defs, feature, "activity_path", description)
    for feature, description in absorption.items():
        _register(defs, feature, "low_absorption", description)
    for q in range(1, 5):
        _register(defs, f"path_q{q}_price_return", "normalized_phase", f"标准化路径第{q}段价格收益")
        _register(defs, f"path_q{q}_delta_ratio", "normalized_phase", f"标准化路径第{q}段delta比例")
        _register(defs, f"path_q{q}_large_delta_ratio", "normalized_phase", f"标准化路径第{q}段大单delta比例")
        _register(defs, f"path_q{q}_sell_intensity", "normalized_phase", f"标准化路径第{q}段卖压强度")
        _register(defs, f"path_q{q}_absorption", "normalized_phase", f"标准化路径第{q}段吸收分数")

    frame = pd.DataFrame(rows)
    meta = pd.DataFrame([d.__dict__ for d in defs]).drop_duplicates("feature")
    return frame, meta


def attach_post_green_path_diagnostics(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizon: int,
    entry_delay_bars: int,
    stop_buffer_pct: float,
    entry_fee_rate: float,
    exit_fee_rate: float,
    entry_slippage_pct: float,
    exit_slippage_pct: float,
) -> pd.DataFrame:
    """Attach future path labels for diagnostics only.

    These columns are prefixed ``post_`` and must never be present in feature
    metadata or candidate filter expressions.
    """
    out = signals.copy().reset_index(drop=True)
    positions = bars.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(out["event_time"])))
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for i, signal_pos in enumerate(positions):
        if signal_pos < 0:
            records.append({})
            continue
        entry_pos = int(signal_pos) + int(entry_delay_bars)
        if entry_pos >= len(bars):
            records.append({})
            continue
        end = min(len(bars) - 1, entry_pos + int(horizon) - 1)
        entry_raw = float(opens[entry_pos])
        entry_cash = entry_raw * (1.0 + entry_slippage_pct) * (1.0 + entry_fee_rate)
        stop = _finite(out.at[i, "episode_low"]) * (1.0 - stop_buffer_pct)
        risk = entry_raw - stop
        if not np.isfinite(risk) or risk <= 0:
            records.append({})
            continue
        mfe_idx = entry_pos + int(np.nanargmax(highs[entry_pos : end + 1]))
        mae_idx = entry_pos + int(np.nanargmin(lows[entry_pos : end + 1]))
        mfe_r = (highs[mfe_idx] - entry_raw) / risk
        mae_r = (lows[mae_idx] - entry_raw) / risk
        rec: dict[str, Any] = {
            "post_entry_time": bars.index[entry_pos],
            "post_horizon_end_time": bars.index[end],
            "post_entry_raw": entry_raw,
            "post_stop_price": stop,
            "post_risk_pct": risk / entry_raw,
            "post_mfe_r": mfe_r,
            "post_mae_r": mae_r,
            "post_time_to_mfe_bars": mfe_idx - entry_pos,
            "post_time_to_mae_bars": mae_idx - entry_pos,
        }
        stop_bar = None
        target_bars: dict[float, int | None] = {0.5: None, 1.0: None, 1.5: None}
        for p in range(entry_pos, end + 1):
            stop_hit = lows[p] <= stop
            if stop_hit and stop_bar is None:
                stop_bar = p
            for r in target_bars:
                if target_bars[r] is None and highs[p] >= entry_raw + r * risk:
                    target_bars[r] = p
            if stop_bar is not None and all(v is not None for v in target_bars.values()):
                break
        for r, target_bar in target_bars.items():
            key = str(r).replace(".", "_")
            rec[f"post_target_{key}R_before_stop"] = bool(
                target_bar is not None and (stop_bar is None or target_bar < stop_bar)
            )
            rec[f"post_time_to_{key}R_bars"] = target_bar - entry_pos if target_bar is not None else np.nan
        rec["post_stop_hit"] = stop_bar is not None
        rec["post_time_to_stop_bars"] = stop_bar - entry_pos if stop_bar is not None else np.nan
        exit_cash = closes[end] * (1.0 - exit_slippage_pct) * (1.0 - exit_fee_rate)
        rec["post_horizon_net"] = exit_cash / entry_cash - 1.0
        max_close_pos = entry_pos + int(np.nanargmax(closes[entry_pos : end + 1]))
        peak_close = closes[max_close_pos]
        rec["post_close_peak_giveback"] = closes[end] / peak_close - 1.0 if peak_close > 0 else np.nan
        target_1r = bool(rec["post_target_1_0R_before_stop"])
        target_half = bool(rec["post_target_0_5R_before_stop"])
        if target_1r and rec["post_time_to_1_0R_bars"] <= 5:
            outcome_class = "immediate_continuation"
        elif target_1r:
            outcome_class = "dip_or_slow_then_rally"
        elif target_half and (rec["post_horizon_net"] <= 0 or rec["post_stop_hit"]):
            outcome_class = "short_bounce_fade"
        elif rec["post_stop_hit"] and not target_half:
            outcome_class = "direct_failure"
        else:
            outcome_class = "drift_or_incomplete"
        rec["post_outcome_class"] = outcome_class
        records.append(rec)
    return pd.concat([out, pd.DataFrame(records, index=out.index)], axis=1)
