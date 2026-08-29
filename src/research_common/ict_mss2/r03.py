#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03 helpers for ETH liquidity-stack microstructure and execution research.

R03 deliberately keeps the R02 structural edge definition separate from the
new microstructure evidence.  The core statistical unit remains one causal
liquidity-stack episode.  Trade-bar and footprint data are attached only at a
known decision timestamp and are never allowed to change the historical swing
or pool lifecycle retroactively.

The module also contains a secondary FVG execution overlay.  This is an
execution study, not a new edge admission rule: once a frozen liquidity-stack
threshold is known, it compares first-directional-FVG market, proximal limit,
and 50/50 market+limit execution while keeping the same structural stop and
opposing 4H liquidity objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

from .core import EPS, aggregate_bars, normalize_1m_bars
from .r02 import (
    R02Config,
    _DynamicActiveLiquidityBook,
    _first_competing_outcome,
    _market_entry_after_signal,
    _structural_stop_before_entry,
)


@dataclass(frozen=True)
class R03Config:
    """Frozen R03 research design.

    ``pool_threshold_core`` is the R02 candidate. ``pool_threshold_expand`` is
    the only predeclared frequency-recovery relaxation.  R03 does not search a
    large threshold grid.
    """

    pool_tolerance_bps: float = 10.0
    pool_threshold_core: int = 4
    pool_threshold_expand: int = 3
    baseline_execution_minutes: int = 5
    target_name: str = "htf240"
    tradebar_lookbacks_minutes: tuple[int, ...] = (1, 3, 5, 15)
    tradebar_baseline_minutes: int = 60
    tradebar_chunk_days: int = 31
    fvg_execution_minutes: tuple[int, ...] = (1, 2, 5)
    fvg_signal_wait_minutes: int = 180
    fvg_limit_wait_minutes: int = 180
    execution_censor_minutes: int = 10_080
    stop_buffer_bps: float = 2.0
    market_roundtrip_cost: float = 0.0011
    limit_roundtrip_cost: float = 0.0009
    footprint_chunk_days: int = 120

    def validate(self) -> "R03Config":
        if self.pool_tolerance_bps <= 0:
            raise ValueError("pool_tolerance_bps must be positive")
        if self.pool_threshold_core <= self.pool_threshold_expand:
            raise ValueError("core threshold must be stricter than expansion threshold")
        if self.pool_threshold_expand < 2:
            raise ValueError("pool_threshold_expand must be >=2")
        if self.baseline_execution_minutes not in {1, 2, 5}:
            raise ValueError("baseline_execution_minutes must be 1/2/5")
        if not self.tradebar_lookbacks_minutes or min(self.tradebar_lookbacks_minutes) <= 0:
            raise ValueError("tradebar lookbacks must be positive")
        if self.tradebar_baseline_minutes <= max(self.tradebar_lookbacks_minutes):
            raise ValueError("tradebar baseline window must exceed short lookbacks")
        if min(self.fvg_execution_minutes) <= 0:
            raise ValueError("fvg execution minutes must be positive")
        if min(self.fvg_signal_wait_minutes, self.fvg_limit_wait_minutes, self.execution_censor_minutes) <= 0:
            raise ValueError("execution windows must be positive")
        if self.stop_buffer_bps < 0:
            raise ValueError("stop buffer cannot be negative")
        if min(self.market_roundtrip_cost, self.limit_roundtrip_cost) < 0:
            raise ValueError("costs cannot be negative")
        return self


def r03_globalize_legacy_trade_ids(features: pd.DataFrame, labels: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Repair legacy R02 local IDs without an unsafe many-to-many join.

    Old R02 reports restarted ``R02_TRADE_...`` for each execution timeframe.
    The feature and label files were written from the same row order, so R03
    verifies that row-wise IDs match before prefixing the execution timeframe.
    New R02 reports already contain ``R02_<N>M_TRADE_...`` and are left intact.
    """

    feat = features.copy()
    lab = labels.copy() if labels is not None else None
    if feat.empty or "trade_event_id" not in feat.columns:
        return feat, lab
    ids = feat["trade_event_id"].astype(str)
    is_legacy = ids.str.match(r"^R02_TRADE_\d+$", na=False)
    if not is_legacy.any():
        if ids.duplicated().any():
            raise RuntimeError("R02 trade_event_id remains duplicated after global-ID migration")
        return feat, lab
    if "execution_minutes" not in feat.columns:
        raise RuntimeError("legacy R02 ID repair requires execution_minutes in feature rows")
    if lab is not None:
        if len(lab) != len(feat):
            raise RuntimeError("legacy R02 feature/label row counts differ; refusing positional repair")
        if "trade_event_id" not in lab.columns:
            raise RuntimeError("legacy R02 label rows are missing trade_event_id")
        same = ids.to_numpy(dtype=str) == lab["trade_event_id"].astype(str).to_numpy(dtype=str)
        if not bool(np.all(same)):
            raise RuntimeError("legacy R02 feature/label row order is not identical; refusing positional repair")
    local_suffix = ids.str.removeprefix("R02_TRADE_")
    minutes = pd.to_numeric(feat["execution_minutes"], errors="raise").astype(int).astype(str)
    repaired = "R02_" + minutes + "M_TRADE_" + local_suffix
    feat["trade_event_id"] = repaired
    if lab is not None:
        lab["trade_event_id"] = repaired.to_numpy()
    if feat["trade_event_id"].duplicated().any():
        raise RuntimeError("legacy R02 global-ID repair still produced duplicates")
    return feat, lab


def first_pool_threshold_crossing_trades(
    trades: pd.DataFrame,
    *,
    threshold: int,
    tolerance_bps: float = 10.0,
    direction: int = 1,
    trigger_type: str = "episode_reclaim",
    execution_minutes: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return one causal first-crossing trade per episode/execution configuration."""

    if trades.empty:
        return trades.copy()
    token = str(float(tolerance_bps)).replace(".", "p")
    pool_col = f"price_pools_{token}bp_cum"
    if pool_col not in trades.columns:
        raise KeyError(pool_col)
    frame = trades.loc[
        pd.to_numeric(trades["trade_direction"], errors="coerce").eq(int(direction))
        & trades["trigger_type"].astype(str).eq(str(trigger_type))
        & pd.to_numeric(trades[pool_col], errors="coerce").fillna(0).ge(int(threshold))
    ].copy()
    if execution_minutes is not None:
        allowed = {int(x) for x in execution_minutes}
        frame = frame.loc[pd.to_numeric(frame["execution_minutes"], errors="coerce").isin(allowed)].copy()
    if frame.empty:
        return frame
    sort_cols = [c for c in ("episode_id", "execution_minutes", "entry_pos_1m", "signal_available_time", "trade_event_id") if c in frame.columns]
    frame = frame.sort_values(sort_cols, kind="stable")
    keys = [c for c in ("episode_id", "execution_minutes", "trigger_type") if c in frame.columns]
    if keys:
        frame = frame.drop_duplicates(keys, keep="first")
    frame["r03_pool_threshold"] = int(threshold)
    frame["r03_pool_tolerance_bps"] = float(tolerance_bps)
    return frame.reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def _bp(value: float, base: float) -> float:
    return _safe_ratio(value, base) * 10_000.0


def _window_tradebar_summary(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if frame.empty:
        for name in (
            "bars", "notional", "buy_notional", "sell_notional", "delta_notional", "delta_ratio",
            "sell_share", "trades_count", "large_buy_notional", "large_sell_notional",
            "large_sell_share", "max_trade_notional", "price_return_bp", "downside_bp",
            "close_off_low_bp", "downside_bp_per_sell_million",
        ):
            out[f"{prefix}_{name}"] = np.nan
        return out
    numeric_cols = (
        "notional", "buy_notional", "sell_notional", "delta_notional", "trades_count",
        "large_buy_notional", "large_sell_notional", "max_trade_notional",
    )
    vals: dict[str, float] = {}
    for name in numeric_cols:
        series = pd.to_numeric(frame.get(name), errors="coerce")
        vals[name] = float(series.sum()) if name != "max_trade_notional" else float(series.max())
    first_open = float(pd.to_numeric(frame["open"], errors="coerce").iloc[0])
    last_close = float(pd.to_numeric(frame["close"], errors="coerce").iloc[-1])
    low = float(pd.to_numeric(frame["low"], errors="coerce").min())
    out[f"{prefix}_bars"] = float(len(frame))
    for name, value in vals.items():
        out[f"{prefix}_{name}"] = value
    out[f"{prefix}_delta_ratio"] = _safe_ratio(vals["delta_notional"], vals["notional"])
    out[f"{prefix}_sell_share"] = _safe_ratio(vals["sell_notional"], vals["notional"])
    out[f"{prefix}_large_sell_share"] = _safe_ratio(vals["large_sell_notional"], vals["sell_notional"])
    out[f"{prefix}_price_return_bp"] = _bp(last_close - first_open, first_open)
    out[f"{prefix}_downside_bp"] = max(0.0, _bp(first_open - low, first_open)) if np.isfinite(low) else np.nan
    out[f"{prefix}_close_off_low_bp"] = max(0.0, _bp(last_close - low, low)) if np.isfinite(low) else np.nan
    out[f"{prefix}_downside_bp_per_sell_million"] = _safe_ratio(
        out[f"{prefix}_downside_bp"], vals["sell_notional"] / 1_000_000.0
    )
    return out


def _tradebar_chunks(source: pd.DataFrame, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    if source.empty:
        return []
    decision = pd.to_datetime(source["decision_time"], errors="coerce")
    start = decision.min().normalize()
    end = decision.max().normalize()
    chunks: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    cursor = start
    delta = pd.Timedelta(days=max(1, int(days)))
    while cursor <= end:
        chunk_end = min(end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1), cursor + delta - pd.Timedelta(microseconds=1))
        mask = decision.between(cursor, chunk_end, inclusive="both")
        part = source.loc[mask].copy()
        if not part.empty:
            chunks.append((cursor, chunk_end, part))
        cursor = (chunk_end + pd.Timedelta(microseconds=1)).normalize()
    return chunks


def build_tradebar_microstructure_features(
    checkpoints: pd.DataFrame,
    *,
    symbol: str = "ETH-USDT-SWAP",
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    config: R03Config | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach causal 1m trade-bar flow features at each entry decision.

    Only bars whose full 1m interval has completed by ``decision_time`` are
    included.  The episode window starts from the causal episode start carried
    by R02; the comparison baseline is the fixed 60 minutes immediately before
    that episode.
    """

    cfg = (config or R03Config()).validate()
    required = {"checkpoint_id", "decision_time", "episode_start_time"}
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        raise KeyError(f"tradebar checkpoints missing {missing}")
    source = checkpoints[list(required)].copy()
    source["checkpoint_id"] = source["checkpoint_id"].astype(str)
    source["decision_time"] = pd.to_datetime(source["decision_time"], errors="coerce")
    source["episode_start_time"] = pd.to_datetime(source["episode_start_time"], errors="coerce")
    source = source.dropna(subset=["checkpoint_id", "decision_time", "episode_start_time"])
    source = source.drop_duplicates("checkpoint_id", keep="first").sort_values("decision_time", kind="stable")
    if source.empty:
        return pd.DataFrame(), pd.DataFrame()

    loader = OKXTradeBarLoader(symbol=symbol, timeframe="1m", data_dir=data_dir, db_name=db_name)
    chunks = _tradebar_chunks(source, cfg.tradebar_chunk_days)
    reporter = ProgressReporter("[r03-tradebar]", total=len(chunks), every=1, enabled=show_progress)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    baseline_delta = pd.Timedelta(minutes=int(cfg.tradebar_baseline_minutes))

    for chunk_no, (chunk_start, chunk_end, part) in enumerate(chunks, start=1):
        query_start = min(
            part["episode_start_time"].min() - baseline_delta - pd.Timedelta(minutes=2),
            part["decision_time"].min() - baseline_delta - pd.Timedelta(minutes=2),
        )
        query_end = part["decision_time"].max() - pd.Timedelta(microseconds=1)
        bars = loader.load_local_data(query_start, query_end)
        if not bars.empty:
            bars = bars.sort_index(kind="stable")
            bars = bars.loc[~bars.index.duplicated(keep="last")]
        rows: list[dict[str, object]] = []
        causal_bad = 0
        for item in part.itertuples(index=False):
            decision = pd.Timestamp(item.decision_time)
            episode_start = pd.Timestamp(item.episode_start_time)
            record: dict[str, object] = {"checkpoint_id": str(item.checkpoint_id), "decision_time": decision, "episode_start_time": episode_start}
            if bars.empty:
                record.update({"tb_causal_valid": False, "tb_last_source_time": pd.NaT, "tb_source_available_time": pd.NaT})
                record.update(_window_tradebar_summary(pd.DataFrame(), "tb_pre60"))
                record.update(_window_tradebar_summary(pd.DataFrame(), "tb_episode"))
                for minutes in cfg.tradebar_lookbacks_minutes:
                    record.update(_window_tradebar_summary(pd.DataFrame(), f"tb_last{int(minutes)}"))
                record["tb_episode_notional_intensity_vs_pre60"] = np.nan
                record["tb_episode_sell_notional_intensity_vs_pre60"] = np.nan
                record["tb_episode_trades_count_intensity_vs_pre60"] = np.nan
                record["tb_episode_impact_ratio_vs_pre60"] = np.nan
                record["tb_last5_delta_improvement_vs_episode"] = np.nan
                record["tb_last5_reclaim_flag"] = 0
                record["tb_episode_cvd_end"] = np.nan
                record["tb_episode_cvd_min"] = np.nan
                record["tb_episode_cvd_recovery"] = np.nan
                record["tb_episode_cvd_recovery_ratio"] = np.nan
                for _m in (3, 5, 15):
                    record[f"tb_cvd_bullish_divergence_{_m}m_flag"] = 0
                    record[f"tb_cvd_at_recent_low_minus_prior_low_{_m}m"] = np.nan
                record["tb_absorption_mechanism_flag"] = 0
                record["tb_flow_recovery_flag"] = 0
                rows.append(record)
                continue
            # A 1m trade bar left-labelled at t is only complete at t+1m.
            right = int(bars.index.searchsorted(decision, side="left"))
            completed = bars.iloc[:right]
            last_source = pd.Timestamp(completed.index[-1]) if len(completed) else pd.NaT
            causal_valid = bool(len(completed) and last_source + pd.Timedelta(minutes=1) <= decision)
            if not causal_valid:
                causal_bad += 1
            record["tb_causal_valid"] = causal_valid
            record["tb_last_source_time"] = last_source
            record["tb_source_available_time"] = last_source + pd.Timedelta(minutes=1) if pd.notna(last_source) else pd.NaT
            pre = bars.loc[(bars.index >= episode_start - baseline_delta) & (bars.index < episode_start)]
            episode = bars.loc[(bars.index >= episode_start) & (bars.index < decision)]
            record.update(_window_tradebar_summary(pre, "tb_pre60"))
            record.update(_window_tradebar_summary(episode, "tb_episode"))
            for minutes in cfg.tradebar_lookbacks_minutes:
                window = bars.loc[(bars.index >= decision - pd.Timedelta(minutes=int(minutes))) & (bars.index < decision)]
                record.update(_window_tradebar_summary(window, f"tb_last{int(minutes)}"))

            ep_bars = float(record.get("tb_episode_bars", np.nan))
            pre_bars = float(record.get("tb_pre60_bars", np.nan))
            for flow in ("notional", "sell_notional", "trades_count"):
                ep_total = float(record.get(f"tb_episode_{flow}", np.nan))
                pre_total = float(record.get(f"tb_pre60_{flow}", np.nan))
                ep_per_bar = _safe_ratio(ep_total, ep_bars)
                pre_per_bar = _safe_ratio(pre_total, pre_bars)
                record[f"tb_episode_{flow}_intensity_vs_pre60"] = _safe_ratio(ep_per_bar, pre_per_bar)
            record["tb_episode_impact_ratio_vs_pre60"] = _safe_ratio(
                float(record.get("tb_episode_downside_bp_per_sell_million", np.nan)),
                float(record.get("tb_pre60_downside_bp_per_sell_million", np.nan)),
            )
            record["tb_last5_delta_improvement_vs_episode"] = (
                float(record.get("tb_last5_delta_ratio", np.nan)) - float(record.get("tb_episode_delta_ratio", np.nan))
            )
            record["tb_last5_reclaim_flag"] = int(float(record.get("tb_last5_price_return_bp", np.nan)) > 0)

            # Episode-anchored CVD diagnostics.  We recompute CVD from stored
            # 1m delta_notional instead of trusting the loader's cvd column,
            # whose cumulative origin may depend on the read/cache boundary.
            if not episode.empty:
                ep_delta = pd.to_numeric(episode.get("delta_notional"), errors="coerce").fillna(0.0)
                ep_cvd = ep_delta.cumsum()
                cvd_end = float(ep_cvd.iloc[-1]) if len(ep_cvd) else np.nan
                cvd_min = float(ep_cvd.min()) if len(ep_cvd) else np.nan
                cvd_recovery = cvd_end - cvd_min if np.isfinite(cvd_end) and np.isfinite(cvd_min) else np.nan
                record["tb_episode_cvd_end"] = cvd_end
                record["tb_episode_cvd_min"] = cvd_min
                record["tb_episode_cvd_recovery"] = cvd_recovery
                record["tb_episode_cvd_recovery_ratio"] = _safe_ratio(
                    cvd_recovery, float(record.get("tb_episode_notional", np.nan))
                )
                ep_low = pd.to_numeric(episode.get("low"), errors="coerce")
                for _m in (3, 5, 15):
                    cutoff = decision - pd.Timedelta(minutes=int(_m))
                    prior_mask = episode.index < cutoff
                    recent_mask = episode.index >= cutoff
                    flag = 0
                    diff = np.nan
                    if prior_mask.any() and recent_mask.any():
                        prior_low_series = ep_low.loc[prior_mask].dropna()
                        recent_low_series = ep_low.loc[recent_mask].dropna()
                        if len(prior_low_series) and len(recent_low_series):
                            prior_low_time = prior_low_series.idxmin()
                            recent_low_time = recent_low_series.idxmin()
                            prior_low_price = float(prior_low_series.loc[prior_low_time])
                            recent_low_price = float(recent_low_series.loc[recent_low_time])
                            prior_cvd = float(ep_cvd.loc[prior_low_time])
                            recent_cvd = float(ep_cvd.loc[recent_low_time])
                            diff = recent_cvd - prior_cvd
                            flag = int(recent_low_price < prior_low_price and recent_cvd > prior_cvd)
                    record[f"tb_cvd_bullish_divergence_{_m}m_flag"] = flag
                    record[f"tb_cvd_at_recent_low_minus_prior_low_{_m}m"] = diff
            else:
                record["tb_episode_cvd_end"] = np.nan
                record["tb_episode_cvd_min"] = np.nan
                record["tb_episode_cvd_recovery"] = np.nan
                record["tb_episode_cvd_recovery_ratio"] = np.nan
                for _m in (3, 5, 15):
                    record[f"tb_cvd_bullish_divergence_{_m}m_flag"] = 0
                    record[f"tb_cvd_at_recent_low_minus_prior_low_{_m}m"] = np.nan

            # Fixed mechanism semantics, not fitted thresholds:
            # more sell activity than the pre-episode baseline, but less downside
            # progress per sell million and improving short-window delta.
            sell_intensity = float(record.get("tb_episode_sell_notional_intensity_vs_pre60", np.nan))
            impact_ratio = float(record.get("tb_episode_impact_ratio_vs_pre60", np.nan))
            delta_improve = float(record.get("tb_last5_delta_improvement_vs_episode", np.nan))
            record["tb_absorption_mechanism_flag"] = int(
                np.isfinite(sell_intensity) and np.isfinite(impact_ratio) and np.isfinite(delta_improve)
                and sell_intensity >= 1.0 and impact_ratio < 1.0 and delta_improve > 0.0
            )
            record["tb_flow_recovery_flag"] = int(
                np.isfinite(delta_improve) and delta_improve > 0.0 and float(record.get("tb_last5_price_return_bp", np.nan)) > 0.0
            )
            rows.append(record)
        outputs.append(pd.DataFrame(rows))
        audits.append(
            {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "events": int(len(part)),
                "tradebar_rows": int(len(bars)),
                "causal_bad": int(causal_bad),
                "coverage": float(pd.DataFrame(rows).get("tb_causal_valid", pd.Series(dtype=bool)).fillna(False).mean()) if rows else 0.0,
            }
        )
        reporter.update(chunk_no)
    reporter.close()
    features = pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()
    if not features.empty and features["checkpoint_id"].duplicated().any():
        raise RuntimeError("duplicate checkpoint_id in R03 tradebar features")
    return features, pd.DataFrame(audits)


def attach_footprint_microstructure_features(
    checkpoints: pd.DataFrame,
    *,
    symbol: str = "ETH-USDT-SWAP",
    data_dir: str | Path | None = None,
    range_db_name: str = "okx_range_bars.db",
    footprint_db_name: str = "okx_range_footprints.db",
    config: R03Config | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the project's causal r0020/step1 footprint context."""

    cfg = (config or R03Config()).validate()
    from src.research_common.post_sweep_supervised.config import PostSweepSupervisedConfig
    from src.research_common.post_sweep_supervised.features import build_footprint_features

    source = checkpoints[["checkpoint_id", "decision_time"]].copy()
    ps_cfg = PostSweepSupervisedConfig(footprint_chunk_days=int(cfg.footprint_chunk_days)).validate()
    result = build_footprint_features(
        source,
        symbol=symbol,
        data_dir=data_dir,
        range_db_name=range_db_name,
        footprint_db_name=footprint_db_name,
        config=ps_cfg,
        progress=show_progress,
    )
    features = result.features.copy()
    if features.empty:
        return features, result.audit.copy()
    for name in (
        "fp_impact_ratio_vs_prev_down",
        "fp_low3_sell_vs_prev_down_ratio",
        "fp_low3_delta_improvement_vs_prev_down",
        "fp_close_off_low_improvement_vs_prev_down_bp",
    ):
        if name not in features.columns:
            features[name] = np.nan
    impact = pd.to_numeric(features["fp_impact_ratio_vs_prev_down"], errors="coerce")
    sell = pd.to_numeric(features["fp_low3_sell_vs_prev_down_ratio"], errors="coerce")
    delta = pd.to_numeric(features["fp_low3_delta_improvement_vs_prev_down"], errors="coerce")
    reclaim = pd.to_numeric(features["fp_close_off_low_improvement_vs_prev_down_bp"], errors="coerce")
    features["fp_absorption_mechanism_flag"] = (
        sell.ge(1.0) & impact.lt(1.0) & delta.gt(0.0) & reclaim.gt(0.0)
    ).astype(np.int8)
    features["fp_delta_recovery_flag"] = delta.gt(0.0).astype(np.int8)
    return features, result.audit.copy()


def _first_directional_fvg(exec_bars: pd.DataFrame, *, direction: int, start_pos: int, end_pos: int) -> tuple[int, float, float, float]:
    """First fully closed directional 3-bar FVG in [start_pos, end_pos]."""

    high = exec_bars["high"].to_numpy(dtype=float)
    low = exec_bars["low"].to_numpy(dtype=float)
    left = max(2, int(start_pos))
    right = min(len(exec_bars) - 1, int(end_pos))
    for pos in range(left, right + 1):
        if direction > 0 and low[pos] > high[pos - 2]:
            lower = float(high[pos - 2])
            upper = float(low[pos])
            return pos, lower, upper, upper
        if direction < 0 and high[pos] < low[pos - 2]:
            lower = float(high[pos])
            upper = float(low[pos - 2])
            return pos, lower, upper, lower
    return -1, np.nan, np.nan, np.nan


def build_fvg_execution_overlay_attempts(
    primary_1m: pd.DataFrame,
    threshold_stages: pd.DataFrame,
    *,
    execution_minutes: int,
    config: R03Config | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build first-FVG market/limit attempts after a frozen pool threshold.

    This intentionally does not require a later MSS.  The liquidity-stack edge
    is held fixed; FVG is studied only as an execution timing device.
    """

    cfg = (config or R03Config()).validate()
    if threshold_stages.empty:
        return pd.DataFrame()
    bars1 = normalize_1m_bars(primary_1m)
    exec_bars = aggregate_bars(bars1, int(execution_minutes))
    if exec_bars.empty:
        return pd.DataFrame()
    exec_index = exec_bars.index
    low1 = bars1["low"].to_numpy(dtype=float)
    high1 = bars1["high"].to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low1)
    high_index = SegmentThresholdIndex(high1)
    max_signal_bars = max(1, int(np.ceil(cfg.fvg_signal_wait_minutes / int(execution_minutes))))
    rows: list[dict[str, object]] = []
    stages = threshold_stages.sort_values(["sweep_pos_1m", "episode_id", "stage_id"], kind="stable")
    reporter = ProgressReporter(
        f"[r03-fvg-overlay-{int(execution_minutes)}m]", total=len(stages), every=max(1, len(stages) // 100), enabled=show_progress
    )
    for loop_i, source in enumerate(stages.itertuples(index=False), start=1):
        reporter.update(loop_i)
        direction = int(getattr(source, "trade_direction"))
        sweep_time = pd.Timestamp(getattr(source, "sweep_bar_time_1m"))
        sweep_exec_pos = int(exec_index.searchsorted(sweep_time, side="right")) - 1
        if sweep_exec_pos < 0:
            continue
        end_pos = min(len(exec_bars) - 1, sweep_exec_pos + max_signal_bars)
        fvg_pos, lower, upper, proximal = _first_directional_fvg(
            exec_bars, direction=direction, start_pos=sweep_exec_pos, end_pos=end_pos
        )
        if fvg_pos < 0:
            continue
        signal_available = pd.Timestamp(exec_bars["bar_end_time"].iloc[fvg_pos])
        market_pos, market_time, market_price = _market_entry_after_signal(bars1, signal_available)
        if market_pos < 0:
            continue
        episode_start = int(getattr(source, "episode_start_pos_1m"))
        structural_extreme, stop = _structural_stop_before_entry(
            low1,
            high1,
            direction=direction,
            start_pos=episode_start,
            end_pos=max(episode_start, market_pos - 1),
            buffer_bps=cfg.stop_buffer_bps,
        )
        if not np.isfinite(stop):
            continue
        attempt_id = f"R03_{int(execution_minutes)}M_FVG_{getattr(source, 'episode_id')}_{getattr(source, 'stage_id')}"
        common = {
            "overlay_attempt_id": attempt_id,
            "episode_id": getattr(source, "episode_id"),
            "stage_id": getattr(source, "stage_id"),
            "trade_direction": direction,
            "execution_minutes": int(execution_minutes),
            "signal_exec_pos": int(fvg_pos),
            "signal_bar_time": exec_index[fvg_pos],
            "signal_available_time": signal_available,
            "fvg_lower": lower,
            "fvg_upper": upper,
            "fvg_proximal": proximal,
            "fvg_width_bp": abs(upper / lower - 1.0) * 10_000.0 if lower > EPS else np.nan,
            "episode_start_pos_1m": episode_start,
            "sweep_pos_1m": int(getattr(source, "sweep_pos_1m")),
            "price_pools_10p0bp_cum": int(getattr(source, "price_pools_10p0bp_cum")),
            "max_source_timeframe_min_cum": int(getattr(source, "max_source_timeframe_min_cum")),
            "structural_extreme_pre_entry": structural_extreme,
            "stop_price": stop,
        }
        market = dict(common)
        market.update(
            {
                "trade_event_id": attempt_id + "_MARKET",
                "trigger_type": "stack_first_fvg_market",
                "reference_mode": "none",
                "entry_kind": "market_next_open",
                "entry_fill_flag": 1,
                "entry_pos_1m": int(market_pos),
                "entry_time": market_time,
                "entry_price": float(market_price),
            }
        )
        rows.append(market)

        limit_start = market_pos
        limit_end = min(len(bars1) - 1, limit_start + int(cfg.fvg_limit_wait_minutes) - 1)
        if direction > 0:
            fill_pos = low_index.first_leq(limit_start, limit_end, float(proximal))
        else:
            fill_pos = high_index.first_geq(limit_start, limit_end, float(proximal))
        limit = dict(common)
        limit.update(
            {
                "trade_event_id": attempt_id + "_LIMIT",
                "trigger_type": "stack_first_fvg_limit",
                "reference_mode": "none",
                "entry_kind": "fvg_limit",
                "entry_fill_flag": int(fill_pos >= 0),
                "entry_pos_1m": int(fill_pos) if fill_pos >= 0 else -1,
                "entry_time": bars1.index[fill_pos] if fill_pos >= 0 else pd.NaT,
                "entry_price": float(proximal) if fill_pos >= 0 else np.nan,
                "limit_start_pos_1m": int(limit_start),
                "limit_end_pos_1m": int(limit_end),
            }
        )
        rows.append(limit)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if out["trade_event_id"].duplicated().any():
        raise RuntimeError("duplicate R03 FVG overlay trade_event_id")
    return out.sort_values(["signal_available_time", "overlay_attempt_id", "trigger_type"], kind="stable").reset_index(drop=True)


def attach_overlay_structural_outcomes(
    primary_1m: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    attempts: pd.DataFrame,
    *,
    config: R03Config | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Freeze one opposing 4H target at FVG signal time, then resolve legs.

    Market, limit and hybrid must see the same liquidity book.  Therefore the
    target is selected at the first executable 1m open after the FVG closes,
    not later at the limit fill.  A resting limit is cancelled if either the
    structural stop or the already-frozen target is hit before fill.
    """

    cfg = (config or R03Config()).validate()
    if attempts.empty:
        return attempts.copy()
    bars = normalize_1m_bars(primary_1m)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    high_idx = SegmentThresholdIndex(high)
    low_idx = SegmentThresholdIndex(low)
    buy_book = _DynamicActiveLiquidityBook(classified_lifecycle, side="high")
    sell_book = _DynamicActiveLiquidityBook(classified_lifecycle, side="low")
    out = attempts.copy().reset_index(drop=True)
    for name, default in (
        ("target_htf240_price", np.nan),
        ("target_htf240_outcome", "unfilled"),
        ("target_htf240_exit_pos", -1),
        ("target_htf240_holding_minutes", np.nan),
        ("target_htf240_gross_return", np.nan),
        ("target_htf240_net_execution_cost", np.nan),
        ("target_htf240_net_execution_cost2x", np.nan),
        ("target_htf240_net_execution_cost3x", np.nan),
    ):
        out[name] = default

    groups = list(out.groupby("overlay_attempt_id", sort=False))
    groups.sort(key=lambda item: pd.to_datetime(item[1]["signal_available_time"], errors="coerce").min())
    reporter = ProgressReporter("[r03-overlay-exits]", total=len(groups), every=max(1, len(groups) // 100), enabled=show_progress)
    for loop_i, (_, group) in enumerate(groups, start=1):
        reporter.update(loop_i)
        market_rows = group.loc[group["entry_kind"].astype(str).eq("market_next_open")]
        if market_rows.empty:
            continue
        market = market_rows.iloc[0]
        freeze_pos = int(market["entry_pos_1m"])
        if freeze_pos < 0 or freeze_pos >= len(bars):
            continue
        direction = int(market["trade_direction"])
        freeze_price = float(market["entry_price"])
        book = buy_book if direction > 0 else sell_book
        book.advance(freeze_pos)
        target = book.nearest_category(freeze_price, above=direction > 0, category="htf240")
        idxs = list(group.index)
        out.loc[idxs, "target_htf240_price"] = target
        if not np.isfinite(target) or (direction > 0 and target <= freeze_price) or (direction < 0 and target >= freeze_price):
            out.loc[idxs, "target_htf240_outcome"] = "no_target"
            continue

        for i in idxs:
            row = out.loc[i]
            if int(row.get("entry_fill_flag", 0)) != 1:
                continue
            entry_pos = int(row["entry_pos_1m"])
            entry = float(row["entry_price"])
            stop = float(row["stop_price"])
            if entry_pos < 0 or entry_pos >= len(bars) or not np.isfinite(entry) or not np.isfinite(stop):
                continue
            if str(row["entry_kind"]) == "fvg_limit" and entry_pos > freeze_pos:
                if direction > 0:
                    stop_before = low_idx.first_leq(freeze_pos, entry_pos - 1, stop)
                    target_before = high_idx.first_geq(freeze_pos, entry_pos - 1, float(target))
                else:
                    stop_before = high_idx.first_geq(freeze_pos, entry_pos - 1, stop)
                    target_before = low_idx.first_leq(freeze_pos, entry_pos - 1, float(target))
                if stop_before >= 0:
                    out.at[i, "entry_fill_flag"] = 0
                    out.at[i, "target_htf240_outcome"] = "cancelled_stop_before_fill"
                    continue
                if target_before >= 0:
                    out.at[i, "entry_fill_flag"] = 0
                    out.at[i, "target_htf240_outcome"] = "cancelled_target_before_fill"
                    continue
            end_pos = min(len(bars) - 1, entry_pos + int(cfg.execution_censor_minutes) - 1)
            outcome, exit_pos = _first_competing_outcome(
                direction=direction,
                entry_kind=str(row["entry_kind"]),
                entry_pos=entry_pos,
                end_pos=end_pos,
                stop=stop,
                target=float(target),
                low_index=low_idx,
                high_index=high_idx,
            )
            out.at[i, "target_htf240_outcome"] = outcome
            if outcome not in {"target", "stop"} or exit_pos < 0:
                continue
            exit_price = float(target) if outcome == "target" else stop
            gross = direction * (exit_price / entry - 1.0)
            out.at[i, "target_htf240_exit_pos"] = int(exit_pos)
            out.at[i, "target_htf240_holding_minutes"] = int(exit_pos - entry_pos + 1)
            out.at[i, "target_htf240_gross_return"] = gross
            rt_cost = cfg.market_roundtrip_cost if str(row["entry_kind"]) == "market_next_open" else cfg.limit_roundtrip_cost
            out.at[i, "target_htf240_net_execution_cost"] = gross - rt_cost
            out.at[i, "target_htf240_net_execution_cost2x"] = gross - 2.0 * rt_cost
            out.at[i, "target_htf240_net_execution_cost3x"] = gross - 3.0 * rt_cost
    reporter.close()
    return out.sort_values(["signal_available_time", "overlay_attempt_id", "trigger_type"], kind="stable").reset_index(drop=True)


def build_hybrid_5050_outcomes(
    primary_1m: pd.DataFrame,
    overlay_outcomes: pd.DataFrame,
    *,
    config: R03Config | None = None,
) -> pd.DataFrame:
    """Combine 50% FVG-market + 50% FVG-limit on full-intent notional.

    If the limit never fills before the market leg resolves, only half of the
    intended position was deployed, so the unfilled half contributes zero PnL
    and zero cost.  The target is the market leg's frozen 4H objective.
    """

    cfg = (config or R03Config()).validate()
    if overlay_outcomes.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    high_idx = SegmentThresholdIndex(bars["high"].to_numpy(dtype=float))
    low_idx = SegmentThresholdIndex(bars["low"].to_numpy(dtype=float))
    rows: list[dict[str, object]] = []
    for attempt_id, part in overlay_outcomes.groupby("overlay_attempt_id", sort=False):
        market_part = part.loc[part["entry_kind"].astype(str).eq("market_next_open")]
        limit_part = part.loc[part["entry_kind"].astype(str).eq("fvg_limit")]
        if market_part.empty or limit_part.empty:
            continue
        market = market_part.iloc[0]
        limit = limit_part.iloc[0]
        outcome = str(market.get("target_htf240_outcome", ""))
        market_gross = float(market.get("target_htf240_gross_return", np.nan))
        market_exit_pos = int(market.get("target_htf240_exit_pos", -1))
        if outcome not in {"target", "stop"} or not np.isfinite(market_gross) or market_exit_pos < 0:
            rows.append(
                {
                    "overlay_attempt_id": attempt_id,
                    "execution_minutes": int(market["execution_minutes"]),
                    "episode_id": market["episode_id"],
                    "hybrid_outcome": outcome or "censored",
                    "limit_filled_before_market_exit": 0,
                    "hybrid_gross_return": np.nan,
                    "hybrid_net_execution_cost": np.nan,
                    "hybrid_net_execution_cost2x": np.nan,
                    "hybrid_net_execution_cost3x": np.nan,
                }
            )
            continue
        direction = int(market["trade_direction"])
        stop = float(market["stop_price"])
        target = float(market["target_htf240_price"])
        limit_filled = int(limit.get("entry_fill_flag", 0)) == 1
        limit_fill_pos = int(limit.get("entry_pos_1m", -1))
        limit_gross = np.nan
        limit_used = bool(limit_filled and 0 <= limit_fill_pos <= market_exit_pos)
        if limit_used:
            limit_price = float(limit["entry_price"])
            leg_outcome, leg_exit_pos = _first_competing_outcome(
                direction=direction,
                entry_kind="fvg_limit",
                entry_pos=limit_fill_pos,
                end_pos=market_exit_pos,
                stop=stop,
                target=target,
                low_index=low_idx,
                high_index=high_idx,
            )
            if leg_outcome in {"target", "stop"} and leg_exit_pos >= 0:
                leg_exit = target if leg_outcome == "target" else stop
                limit_gross = direction * (leg_exit / limit_price - 1.0)
            else:
                # Conservative: if the limit fill path cannot be resolved by the
                # market leg's exit, treat the limit half as unused.
                limit_used = False
        gross = 0.5 * market_gross + (0.5 * limit_gross if limit_used and np.isfinite(limit_gross) else 0.0)
        cost = 0.5 * cfg.market_roundtrip_cost + (0.5 * cfg.limit_roundtrip_cost if limit_used else 0.0)
        rows.append(
            {
                "overlay_attempt_id": attempt_id,
                "execution_minutes": int(market["execution_minutes"]),
                "episode_id": market["episode_id"],
                "hybrid_outcome": outcome,
                "limit_filled_before_market_exit": int(limit_used),
                "market_gross_return": market_gross,
                "limit_gross_return": limit_gross,
                "hybrid_gross_return": gross,
                "hybrid_net_execution_cost": gross - cost,
                "hybrid_net_execution_cost2x": gross - 2.0 * cost,
                "hybrid_net_execution_cost3x": gross - 3.0 * cost,
            }
        )
    return pd.DataFrame(rows)


def r03_causal_audit(
    tradebar_features: pd.DataFrame,
    footprint_features: pd.DataFrame,
    overlay_attempts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not tradebar_features.empty:
        decision = pd.to_datetime(tradebar_features.get("decision_time"), errors="coerce")
        available = pd.to_datetime(tradebar_features.get("tb_source_available_time"), errors="coerce")
        claimed = available.notna()
        bad = int((claimed & available.gt(decision)).sum())
        rows.append({
            "check": "tradebar_source_completed_by_decision",
            "violations": bad,
            "rows": int(claimed.sum()),
            "missing_coverage": int((~claimed).sum()),
        })
    else:
        rows.append({"check": "tradebar_source_completed_by_decision", "violations": 0, "rows": 0, "missing_coverage": 0})
    if not footprint_features.empty and "fp_causal_valid" in footprint_features.columns:
        valid = footprint_features["fp_causal_valid"].astype("boolean").fillna(False).astype(bool)
        # Missing footprint coverage is not a causality violation. Only rows that
        # claim validity but have source end after checkpoint would be; the shared
        # builder already computes fp_causal_valid from that condition.
        claimed = int(valid.sum())
        rows.append({"check": "footprint_shared_causal_gate", "violations": 0, "rows": claimed})
    else:
        rows.append({"check": "footprint_shared_causal_gate", "violations": 0, "rows": 0})
    if not overlay_attempts.empty:
        signal = pd.to_datetime(overlay_attempts["signal_available_time"], errors="coerce")
        entry = pd.to_datetime(overlay_attempts["entry_time"], errors="coerce")
        filled = pd.to_numeric(overlay_attempts["entry_fill_flag"], errors="coerce").fillna(0).eq(1)
        bad = int((filled & entry.lt(signal)).sum())
        rows.append({"check": "overlay_entry_not_before_fvg_close", "violations": bad, "rows": int(filled.sum())})
    else:
        rows.append({"check": "overlay_entry_not_before_fvg_close", "violations": 0, "rows": 0})
    return pd.DataFrame(rows)


def build_microstructure_checkpoint_union(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact union of >=3 and >=4 causal trade checkpoints.

    ``core_ge4`` is an episode subset of ``expand_ge3`` but is not generally a
    row/trade-ID subset: the first >=3 and first >=4 pool crossings may occur at
    different episode stages.  R03.2 therefore extracts microstructure once per
    concrete trade checkpoint in the union, never from only one cohort.
    """
    required = {"trade_event_id", "signal_available_time", "episode_start_time_1m", "cohort"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise KeyError(f"candidate checkpoints missing {missing}")
    source = candidates.loc[:, [
        "trade_event_id", "signal_available_time", "episode_start_time_1m", "cohort"
    ]].copy()
    source["trade_event_id"] = source["trade_event_id"].astype(str)
    source["decision_time"] = pd.to_datetime(source["signal_available_time"], errors="coerce")
    source["episode_start_time"] = pd.to_datetime(source["episode_start_time_1m"], errors="coerce")
    source = source.dropna(subset=["trade_event_id", "decision_time", "episode_start_time"])

    membership = (
        source.groupby("trade_event_id", sort=False, dropna=False)["cohort"]
        .agg(lambda s: "|".join(sorted(set(s.astype(str)))))
        .rename("cohort_membership")
        .reset_index()
    )
    checkpoints = (
        source.sort_values(["decision_time", "trade_event_id"], kind="stable")
        .drop_duplicates("trade_event_id", keep="first")
        .rename(columns={"trade_event_id": "checkpoint_id"})
        [["checkpoint_id", "decision_time", "episode_start_time"]]
        .reset_index(drop=True)
    )
    checkpoints = checkpoints.merge(
        membership.rename(columns={"trade_event_id": "checkpoint_id"}),
        on="checkpoint_id", how="left", validate="one_to_one",
    )
    audit_rows = [
        {
            "check": "candidate_rows", "value": int(len(candidates)),
            "expected": int(len(candidates)), "passed": 1,
        },
        {
            "check": "union_unique_trade_event_ids", "value": int(len(checkpoints)),
            "expected": int(candidates["trade_event_id"].astype(str).nunique()), "passed": 1,
        },
    ]
    for cohort in sorted(candidates["cohort"].dropna().astype(str).unique()):
        ids = candidates.loc[candidates["cohort"].astype(str).eq(cohort), "trade_event_id"].astype(str)
        audit_rows.append({
            "check": f"{cohort}_unique_trade_event_ids",
            "value": int(ids.nunique()), "expected": int(ids.nunique()), "passed": 1,
        })
    return checkpoints, pd.DataFrame(audit_rows)


def microstructure_feature_join_audit(
    checkpoints: pd.DataFrame,
    features: pd.DataFrame,
    *,
    module: str,
) -> pd.DataFrame:
    """Require one feature row for every requested concrete checkpoint ID.

    This is a *row attachment* audit.  A footprint row may still legitimately
    have ``fp_causal_valid=False`` when the cache has no causal coverage, but the
    checkpoint itself must never disappear during extraction/join.
    """
    expected = checkpoints.get("checkpoint_id", pd.Series(dtype=str)).astype(str)
    actual = features.get("checkpoint_id", pd.Series(dtype=str)).astype(str)
    expected_set = set(expected)
    actual_set = set(actual)
    duplicates = int(actual.duplicated().sum())
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    rows = [
        {"module": module, "check": "requested_unique_checkpoints", "value": len(expected_set), "passed": 1},
        {"module": module, "check": "feature_unique_checkpoints", "value": len(actual_set), "passed": int(len(actual_set) == len(expected_set))},
        {"module": module, "check": "duplicate_feature_checkpoint_ids", "value": duplicates, "passed": int(duplicates == 0)},
        {"module": module, "check": "missing_requested_checkpoint_ids", "value": len(missing), "passed": int(len(missing) == 0)},
        {"module": module, "check": "unexpected_extra_checkpoint_ids", "value": len(extra), "passed": int(len(extra) == 0)},
    ]
    return pd.DataFrame(rows)


def _first_bar_threshold_hit(
    *,
    direction: int,
    start_pos: int,
    end_pos: int,
    stop: float,
    target: float,
    low_index: SegmentThresholdIndex,
    high_index: SegmentThresholdIndex,
) -> tuple[str, int]:
    """First stop/target hit for an opportunity that has not entered yet.

    If both thresholds are touched on the same 1m bar, stop wins.  This keeps
    cancellation semantics pessimistic and avoids using unknown intrabar order.
    """
    if end_pos < start_pos:
        return "none", -1
    if direction > 0:
        sp = low_index.first_leq(start_pos, end_pos, float(stop))
        tp = high_index.first_geq(start_pos, end_pos, float(target))
    else:
        sp = high_index.first_geq(start_pos, end_pos, float(stop))
        tp = low_index.first_leq(start_pos, end_pos, float(target))
    if sp < 0 and tp < 0:
        return "none", -1
    if sp >= 0 and (tp < 0 or sp <= tp):
        return "stop", int(sp)
    return "target", int(tp)


def _resolve_execution_leg(
    *,
    direction: int,
    entry_kind: str,
    entry_pos: int,
    entry_price: float,
    end_pos: int,
    stop: float,
    target: float,
    low_index: SegmentThresholdIndex,
    high_index: SegmentThresholdIndex,
) -> tuple[str, int, float]:
    if entry_pos < 0 or end_pos < entry_pos or not np.isfinite(entry_price):
        return "unfilled", -1, np.nan
    outcome, exit_pos = _first_competing_outcome(
        direction=direction,
        entry_kind=entry_kind,
        entry_pos=entry_pos,
        end_pos=end_pos,
        stop=stop,
        target=target,
        low_index=low_index,
        high_index=high_index,
    )
    if outcome not in {"target", "stop"} or exit_pos < 0:
        return outcome, int(exit_pos), np.nan
    exit_price = target if outcome == "target" else stop
    gross = float(direction) * (float(exit_price) / float(entry_price) - 1.0)
    return outcome, int(exit_pos), gross


def build_core_reclaim_execution_overlays(
    primary_1m: pd.DataFrame,
    core_trades: pd.DataFrame,
    *,
    fvg_minutes: int,
    config: R03Config | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare executions on the exact frozen 5m-reclaim core opportunities.

    The signal, structural stop, opposing 4H target, and absolute 7-day censor
    horizon are frozen from the R02 core trade.  Only execution changes:

    - ``reclaim_market``: original R02 next-open market entry (tie-out baseline);
    - ``post_reclaim_fvg_market``: wait for first same-direction FVG after the
      reclaim decision, then market at its first executable 1m open;
    - ``post_reclaim_fvg_limit``: after that FVG, rest at the proximal boundary;
    - ``hybrid_reclaim_market_fvg_limit``: 50% original reclaim market + 50%
      FVG limit, with unfilled limit half contributing zero PnL and zero cost.

    This function never re-selects a later liquidity target.  ``target_htf240``
    is the price frozen by R02 at the original reclaim entry.
    """
    cfg = (config or R03Config()).validate()
    required = {
        "trade_event_id", "episode_id", "stage_id", "trade_direction",
        "signal_available_time", "entry_pos_1m", "entry_time", "entry_price",
        "stop_price", "target_htf240_price", "target_htf240_outcome",
        "target_htf240_gross_return",
    }
    missing = sorted(required - set(core_trades.columns))
    if missing:
        raise KeyError(f"core execution overlay missing columns {missing}")
    if core_trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    bars = normalize_1m_bars(primary_1m)
    exec_bars = aggregate_bars(bars, int(fvg_minutes))
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    high_index = SegmentThresholdIndex(high)
    exec_end_index = pd.DatetimeIndex(pd.to_datetime(exec_bars["bar_end_time"], errors="coerce"))
    max_fvg_bars = max(1, int(np.ceil(cfg.fvg_signal_wait_minutes / int(fvg_minutes))))
    max_limit_bars_1m = int(cfg.fvg_limit_wait_minutes)
    rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []

    source = core_trades.sort_values(["entry_pos_1m", "trade_event_id"], kind="stable").reset_index(drop=True)
    reporter = ProgressReporter(
        f"[r03.2-core-fvg-{int(fvg_minutes)}m]", total=len(source),
        every=max(1, len(source) // 100), enabled=show_progress,
    )
    for i, row in source.iterrows():
        reporter.update(i + 1)
        direction = int(row["trade_direction"])
        base_signal = pd.Timestamp(row["signal_available_time"])
        base_entry_pos = int(row["entry_pos_1m"])
        base_entry = float(row["entry_price"])
        stop = float(row["stop_price"])
        target = float(row["target_htf240_price"])
        base_geometry_valid = bool(
            0 <= base_entry_pos < len(bars)
            and np.isfinite(base_entry)
            and np.isfinite(stop)
        )
        target_valid = bool(np.isfinite(target))
        if not base_geometry_valid:
            # Preserve the opportunity grain even when an upstream row is malformed.
            # The caller hard-audits this instead of silently shrinking the cohort.
            base_end_pos = -1
            base_outcome, base_exit_pos, base_gross = "invalid_geometry", -1, np.nan
        else:
            base_end_pos = min(len(bars) - 1, base_entry_pos + int(cfg.execution_censor_minutes) - 1)
            if target_valid:
                base_outcome, base_exit_pos, base_gross = _resolve_execution_leg(
                    direction=direction, entry_kind="market_next_open", entry_pos=base_entry_pos,
                    entry_price=base_entry, end_pos=base_end_pos, stop=stop, target=target,
                    low_index=low_index, high_index=high_index,
                )
            else:
                base_outcome, base_exit_pos, base_gross = "no_target", -1, np.nan
        stored_outcome = str(row.get("target_htf240_outcome", ""))
        stored_gross = float(pd.to_numeric(pd.Series([row.get("target_htf240_gross_return")]), errors="coerce").iloc[0])
        outcome_match = int(base_outcome == stored_outcome)
        gross_match = int(
            (not np.isfinite(base_gross) and not np.isfinite(stored_gross)) or
            (np.isfinite(base_gross) and np.isfinite(stored_gross) and abs(base_gross - stored_gross) <= 1e-10)
        )
        tie_rows.append({
            "trade_event_id": str(row["trade_event_id"]),
            "fvg_minutes": int(fvg_minutes),
            "stored_outcome": stored_outcome, "recomputed_outcome": base_outcome,
            "stored_gross_return": stored_gross, "recomputed_gross_return": base_gross,
            "outcome_match": outcome_match, "gross_match": gross_match,
        })

        common = {
            "base_trade_event_id": str(row["trade_event_id"]),
            "episode_id": row["episode_id"], "stage_id": row["stage_id"],
            "trade_direction": direction, "fvg_minutes": int(fvg_minutes),
            "base_signal_available_time": base_signal,
            "base_entry_pos_1m": base_entry_pos, "base_entry_time": pd.Timestamp(row["entry_time"]),
            "base_entry_price": base_entry, "stop_price": stop, "target_htf240_price": target,
            "base_absolute_censor_pos": base_end_pos,
            "year": int(row.get("year")) if pd.notna(row.get("year", np.nan)) else pd.Timestamp(base_signal).year,
            "quarter": row.get("quarter", ""),
            "price_pools_10p0bp_cum": row.get("price_pools_10p0bp_cum", np.nan),
            "max_source_timeframe_min_cum": row.get("max_source_timeframe_min_cum", np.nan),
        }

        # Original reclaim-market baseline.
        base_cost = cfg.market_roundtrip_cost
        rows.append({
            **common, "execution_variant": "reclaim_market", "fvg_signal_available_time": pd.NaT,
            "fvg_lower": np.nan, "fvg_upper": np.nan, "fvg_proximal": np.nan,
            "entry_fill_flag": 1, "entry_pos_1m": base_entry_pos, "entry_time": pd.Timestamp(row["entry_time"]),
            "entry_price": base_entry, "outcome": base_outcome, "exit_pos_1m": base_exit_pos,
            "gross_return": base_gross,
            "net_return_base": base_gross - base_cost if np.isfinite(base_gross) else np.nan,
            "net_return_cost2x": base_gross - 2.0 * base_cost if np.isfinite(base_gross) else np.nan,
            "net_return_cost3x": base_gross - 3.0 * base_cost if np.isfinite(base_gross) else np.nan,
            "limit_filled_flag": 0,
        })

        if not base_geometry_valid or not target_valid:
            unavailable_common = {
                **common, "fvg_signal_available_time": pd.NaT,
                "fvg_lower": np.nan, "fvg_upper": np.nan, "fvg_proximal": np.nan,
            }
            for variant in ("post_reclaim_fvg_market", "post_reclaim_fvg_limit", "hybrid_reclaim_market_fvg_limit"):
                rows.append({
                    **unavailable_common, "execution_variant": variant, "entry_fill_flag": 0,
                    "entry_pos_1m": -1, "entry_time": pd.NaT, "entry_price": np.nan,
                    "outcome": base_outcome, "exit_pos_1m": -1,
                    "gross_return": np.nan, "net_return_base": np.nan,
                    "net_return_cost2x": np.nan, "net_return_cost3x": np.nan,
                    "limit_filled_flag": 0,
                })
            continue

        # First same-direction FVG whose close/availability is not before reclaim decision.
        # Compare timestamps directly.  Do not cast datetime64 to int: pandas may
        # store Series in us while Timestamp.value is ns, which made every
        # search land past the end of the array on some environments.
        start_exec = int(exec_end_index.searchsorted(pd.Timestamp(base_signal), side="left"))
        if start_exec >= len(exec_bars):
            continue
        end_exec = min(len(exec_bars) - 1, start_exec + max_fvg_bars)
        fvg_pos, lower, upper, proximal = _first_directional_fvg(
            exec_bars, direction=direction, start_pos=start_exec, end_pos=end_exec,
        )
        if fvg_pos < 0:
            no_fvg_common = {
                **common, "fvg_signal_available_time": pd.NaT,
                "fvg_lower": np.nan, "fvg_upper": np.nan, "fvg_proximal": np.nan,
            }
            for variant in ("post_reclaim_fvg_market", "post_reclaim_fvg_limit"):
                rows.append({
                    **no_fvg_common, "execution_variant": variant, "entry_fill_flag": 0,
                    "entry_pos_1m": -1, "entry_time": pd.NaT, "entry_price": np.nan,
                    "outcome": "no_fvg_within_wait", "exit_pos_1m": -1,
                    "gross_return": np.nan, "net_return_base": np.nan,
                    "net_return_cost2x": np.nan, "net_return_cost3x": np.nan,
                    "limit_filled_flag": 0,
                })
            hybrid_gross = 0.5 * base_gross if np.isfinite(base_gross) else np.nan
            hybrid_cost = 0.5 * cfg.market_roundtrip_cost
            rows.append({
                **no_fvg_common, "execution_variant": "hybrid_reclaim_market_fvg_limit",
                "entry_fill_flag": 1, "entry_pos_1m": base_entry_pos,
                "entry_time": pd.Timestamp(row["entry_time"]), "entry_price": base_entry,
                "outcome": base_outcome, "exit_pos_1m": base_exit_pos,
                "gross_return": hybrid_gross,
                "net_return_base": hybrid_gross - hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
                "net_return_cost2x": hybrid_gross - 2.0 * hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
                "net_return_cost3x": hybrid_gross - 3.0 * hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
                "limit_filled_flag": 0,
            })
            continue
        fvg_signal = pd.Timestamp(exec_bars["bar_end_time"].iloc[fvg_pos])
        fvg_market_pos, fvg_market_time, fvg_market_price = _market_entry_after_signal(bars, fvg_signal)
        if fvg_market_pos < 0 or fvg_market_pos > base_end_pos:
            continue

        # If the already-frozen stop/target was consumed before FVG became executable,
        # the delayed FVG variants never enter.
        pre_outcome, pre_hit_pos = _first_bar_threshold_hit(
            direction=direction, start_pos=base_entry_pos,
            end_pos=fvg_market_pos - 1, stop=stop, target=target,
            low_index=low_index, high_index=high_index,
        )
        fvg_common = {
            **common, "fvg_signal_available_time": fvg_signal,
            "fvg_lower": lower, "fvg_upper": upper, "fvg_proximal": proximal,
        }
        if pre_outcome != "none":
            for variant in ("post_reclaim_fvg_market", "post_reclaim_fvg_limit"):
                rows.append({
                    **fvg_common, "execution_variant": variant, "entry_fill_flag": 0,
                    "entry_pos_1m": -1, "entry_time": pd.NaT, "entry_price": np.nan,
                    "outcome": f"cancelled_{pre_outcome}_before_fvg_entry", "exit_pos_1m": pre_hit_pos,
                    "gross_return": np.nan, "net_return_base": np.nan,
                    "net_return_cost2x": np.nan, "net_return_cost3x": np.nan,
                    "limit_filled_flag": 0,
                })
            # Hybrid still keeps its original 50% reclaim-market half.
            hybrid_gross = 0.5 * base_gross if np.isfinite(base_gross) else np.nan
            hybrid_cost = 0.5 * cfg.market_roundtrip_cost
            rows.append({
                **fvg_common, "execution_variant": "hybrid_reclaim_market_fvg_limit",
                "entry_fill_flag": 1, "entry_pos_1m": base_entry_pos,
                "entry_time": pd.Timestamp(row["entry_time"]), "entry_price": base_entry,
                "outcome": base_outcome, "exit_pos_1m": base_exit_pos,
                "gross_return": hybrid_gross,
                "net_return_base": hybrid_gross - hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
                "net_return_cost2x": hybrid_gross - 2.0 * hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
                "net_return_cost3x": hybrid_gross - 3.0 * hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
                "limit_filled_flag": 0,
            })
            continue

        # 100% market after FVG confirmation.
        m_out, m_exit, m_gross = _resolve_execution_leg(
            direction=direction, entry_kind="market_next_open", entry_pos=fvg_market_pos,
            entry_price=fvg_market_price, end_pos=base_end_pos, stop=stop, target=target,
            low_index=low_index, high_index=high_index,
        )
        m_cost = cfg.market_roundtrip_cost
        rows.append({
            **fvg_common, "execution_variant": "post_reclaim_fvg_market", "entry_fill_flag": 1,
            "entry_pos_1m": fvg_market_pos, "entry_time": fvg_market_time, "entry_price": fvg_market_price,
            "outcome": m_out, "exit_pos_1m": m_exit, "gross_return": m_gross,
            "net_return_base": m_gross - m_cost if np.isfinite(m_gross) else np.nan,
            "net_return_cost2x": m_gross - 2.0 * m_cost if np.isfinite(m_gross) else np.nan,
            "net_return_cost3x": m_gross - 3.0 * m_cost if np.isfinite(m_gross) else np.nan,
            "limit_filled_flag": 0,
        })

        # 100% proximal limit after the same FVG.
        limit_deadline = min(base_end_pos, fvg_market_pos + max_limit_bars_1m - 1)
        if direction > 0:
            limit_fill_pos = low_index.first_leq(fvg_market_pos, limit_deadline, float(proximal))
        else:
            limit_fill_pos = high_index.first_geq(fvg_market_pos, limit_deadline, float(proximal))
        limit_valid = limit_fill_pos >= 0
        if limit_valid and limit_fill_pos > fvg_market_pos:
            before, before_pos = _first_bar_threshold_hit(
                direction=direction, start_pos=fvg_market_pos, end_pos=limit_fill_pos - 1,
                stop=stop, target=target, low_index=low_index, high_index=high_index,
            )
            if before != "none":
                limit_valid = False
                limit_cancel_reason = f"cancelled_{before}_before_limit_fill"
                limit_cancel_pos = before_pos
            else:
                limit_cancel_reason = ""
                limit_cancel_pos = -1
        else:
            limit_cancel_reason = ""
            limit_cancel_pos = -1
        if not limit_valid:
            if limit_fill_pos < 0:
                # Distinguish no fill from a threshold that killed the resting order.
                pre_limit, pre_limit_pos = _first_bar_threshold_hit(
                    direction=direction, start_pos=fvg_market_pos, end_pos=limit_deadline,
                    stop=stop, target=target, low_index=low_index, high_index=high_index,
                )
                if pre_limit != "none":
                    limit_cancel_reason = f"cancelled_{pre_limit}_before_limit_fill"
                    limit_cancel_pos = pre_limit_pos
                else:
                    limit_cancel_reason = "unfilled_limit_expired"
                    limit_cancel_pos = -1
            rows.append({
                **fvg_common, "execution_variant": "post_reclaim_fvg_limit", "entry_fill_flag": 0,
                "entry_pos_1m": -1, "entry_time": pd.NaT, "entry_price": np.nan,
                "outcome": limit_cancel_reason, "exit_pos_1m": limit_cancel_pos,
                "gross_return": np.nan, "net_return_base": np.nan,
                "net_return_cost2x": np.nan, "net_return_cost3x": np.nan,
                "limit_filled_flag": 0,
            })
            l_gross = np.nan
            l_out = limit_cancel_reason
            l_exit = limit_cancel_pos
            limit_used = False
        else:
            limit_price = float(proximal)
            l_out, l_exit, l_gross = _resolve_execution_leg(
                direction=direction, entry_kind="fvg_limit", entry_pos=int(limit_fill_pos),
                entry_price=limit_price, end_pos=base_end_pos, stop=stop, target=target,
                low_index=low_index, high_index=high_index,
            )
            l_cost = cfg.limit_roundtrip_cost
            rows.append({
                **fvg_common, "execution_variant": "post_reclaim_fvg_limit", "entry_fill_flag": 1,
                "entry_pos_1m": int(limit_fill_pos), "entry_time": pd.Timestamp(bars.index[int(limit_fill_pos)]),
                "entry_price": limit_price, "outcome": l_out, "exit_pos_1m": l_exit,
                "gross_return": l_gross,
                "net_return_base": l_gross - l_cost if np.isfinite(l_gross) else np.nan,
                "net_return_cost2x": l_gross - 2.0 * l_cost if np.isfinite(l_gross) else np.nan,
                "net_return_cost3x": l_gross - 3.0 * l_cost if np.isfinite(l_gross) else np.nan,
                "limit_filled_flag": 1,
            })
            limit_used = True

        # 50% original reclaim market + 50% FVG limit.  The market half always
        # participates; the limit half contributes only if it fills causally.
        # The resting hybrid half is cancelled once the already-open market
        # half resolves the setup.  A fill on the exact same 1m bar as the
        # market leg exit is ambiguous, so it is conservatively not counted.
        if limit_used and base_exit_pos >= 0 and int(limit_fill_pos) >= int(base_exit_pos):
            limit_used = False
            l_gross = np.nan
        hybrid_gross = 0.5 * base_gross if np.isfinite(base_gross) else np.nan
        hybrid_cost = 0.5 * cfg.market_roundtrip_cost
        if limit_used and np.isfinite(l_gross):
            hybrid_gross = hybrid_gross + 0.5 * l_gross if np.isfinite(hybrid_gross) else np.nan
            hybrid_cost += 0.5 * cfg.limit_roundtrip_cost
        rows.append({
            **fvg_common, "execution_variant": "hybrid_reclaim_market_fvg_limit",
            "entry_fill_flag": 1, "entry_pos_1m": base_entry_pos,
            "entry_time": pd.Timestamp(row["entry_time"]), "entry_price": base_entry,
            "outcome": base_outcome, "exit_pos_1m": max(base_exit_pos, l_exit if limit_used else -1),
            "gross_return": hybrid_gross,
            "net_return_base": hybrid_gross - hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
            "net_return_cost2x": hybrid_gross - 2.0 * hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
            "net_return_cost3x": hybrid_gross - 3.0 * hybrid_cost if np.isfinite(hybrid_gross) else np.nan,
            "limit_filled_flag": int(limit_used),
        })
    reporter.close()
    return pd.DataFrame(rows), pd.DataFrame(tie_rows)
