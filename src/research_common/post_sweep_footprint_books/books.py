#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optional compact Books + Trades liquidity-map context for R07.

The layer reads already prebuilt causal liquidity-map features.  It never scans
raw 5000-level archives event by event, and it never downloads missing history.
Books are an incremental mechanism check over their actual local coverage, not a
hard requirement for the full-history Footprint study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader
from src.research_common.progress import ProgressReporter

from .config import PostSweepFootprintBooksConfig


BOOK_SNAPSHOT_COLUMNS = (
    "bid_depth_5bps_base",
    "ask_depth_5bps_base",
    "bid_depth_10bps_base",
    "ask_depth_10bps_base",
    "bid_depth_25bps_base",
    "ask_depth_25bps_base",
    "depth_imbalance_25bps",
    "top_bid_wall_depth_base",
    "top_bid_wall_ratio",
    "top_bid_wall_distance_bps",
    "top_ask_wall_depth_base",
    "top_ask_wall_ratio",
    "top_ask_wall_distance_bps",
)

BOOK_FLOW_COLUMNS = (
    "aggressive_buy_base",
    "aggressive_sell_base",
    "book_added_bid_base",
    "book_added_ask_base",
    "book_removed_bid_base",
    "book_removed_ask_base",
    "estimated_bid_cancel_base",
    "estimated_ask_cancel_base",
    "estimated_bid_consumed_base",
    "estimated_ask_consumed_base",
    "estimated_bid_replenished_base",
    "estimated_ask_replenished_base",
)

BOOK_FEATURE_COLUMNS = (
    "book_metric_time",
    "book_age_seconds",
    "book_window_rows",
    "book_valid_fraction",
    "book_trade_attribution_valid_fraction",
    "book_bid_depth_5bps_base",
    "book_ask_depth_5bps_base",
    "book_bid_depth_25bps_base",
    "book_ask_depth_25bps_base",
    "book_depth_imbalance_25bps",
    "book_top_bid_wall_ratio",
    "book_top_bid_wall_distance_bps",
    "book_bid_depth_5bps_change",
    "book_bid_depth_5bps_recovery_fraction",
    "book_ask_depth_5bps_change",
    "book_depth_imbalance_change",
    "book_aggressive_sell_base_lookback",
    "book_aggressive_buy_base_lookback",
    "book_added_bid_base_lookback",
    "book_removed_bid_base_lookback",
    "book_bid_cancel_base_lookback",
    "book_bid_consumed_base_lookback",
    "book_bid_replenished_base_lookback",
    "book_bid_replenished_to_consumed",
    "book_bid_cancel_share_of_removal",
    "book_bid_added_to_removed",
    "book_bid_replenished_per_aggressive_sell",
    "book_aggressive_sell_to_mean_bid_depth_5bps",
    "book_flow_valid",
    "books_causal_valid",
)


@dataclass(frozen=True)
class BooksBuildResult:
    context: pd.DataFrame
    audit: pd.DataFrame


def _ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-12:
        return np.nan
    return float(num / den)


def _project_timezone_offset() -> pd.Timedelta:
    text = str(TIMEZONE).strip()
    if text.startswith("+"):
        return pd.Timedelta(hours=float(text[1:] or 0))
    if text.startswith("-"):
        return -pd.Timedelta(hours=float(text[1:] or 0))
    return pd.Timedelta(0)


def books_coverage_table(loader: OKXLiquidityMapLoader) -> pd.DataFrame:
    rows = []
    for item in loader.coverage():
        rows.append(
            {
                "day": item.day,
                "features": int(item.features),
                "heatmap_cells": int(item.heatmap_cells),
                "metadata": item.metadata,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["day"] = pd.to_datetime(out["day"], errors="coerce")
        out = out.sort_values("day", kind="mergesort").reset_index(drop=True)
    return out


def _event_book_features(
    frame: pd.DataFrame,
    event_time: pd.Timestamp,
    config: PostSweepFootprintBooksConfig,
) -> dict[str, object]:
    cfg = config
    empty = {name: np.nan for name in BOOK_FEATURE_COLUMNS}
    empty["book_window_rows"] = 0
    empty["book_valid_fraction"] = 0.0
    empty["book_trade_attribution_valid_fraction"] = 0.0
    empty["book_flow_valid"] = False
    empty["books_causal_valid"] = False
    if frame.empty or pd.isna(event_time):
        return empty
    times = pd.to_datetime(frame["available_time"], errors="coerce")
    # Force an explicit nanosecond NumPy representation.  Pandas may store
    # datetimes internally as us/ms in newer versions; using raw ``astype
    # (int64)`` would then mismatch ``Timestamp.value`` by 1,000x/1,000,000x.
    values_ns = times.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    event_ns = pd.Timestamp(event_time).to_datetime64().astype("datetime64[ns]").astype("int64")
    start_time = pd.Timestamp(event_time) - pd.Timedelta(seconds=cfg.books_lookback_seconds)
    start_ns = start_time.to_datetime64().astype("datetime64[ns]").astype("int64")
    left = int(np.searchsorted(values_ns, start_ns, side="left"))
    right = int(np.searchsorted(values_ns, event_ns, side="right"))
    if right <= left:
        return empty
    window = frame.iloc[left:right]
    latest = window.iloc[-1]
    metric_time = pd.Timestamp(latest["available_time"])
    age = float((pd.Timestamp(event_time) - metric_time).total_seconds())
    if age < -1e-9 or age > cfg.books_max_staleness_seconds:
        empty["book_metric_time"] = metric_time
        empty["book_age_seconds"] = age
        empty["book_window_rows"] = len(window)
        return empty

    book_valid = pd.to_numeric(window.get("book_valid"), errors="coerce").fillna(0).astype(bool)
    flow_valid = pd.to_numeric(window.get("trade_attribution_valid"), errors="coerce").fillna(0).astype(bool)
    valid_fraction = float(book_valid.mean()) if len(window) else 0.0
    flow_fraction = float(flow_valid.mean()) if len(window) else 0.0
    first = window.iloc[0]

    def n(row: pd.Series, name: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else np.nan

    def s(name: str) -> float:
        if name not in window.columns:
            return 0.0
        return float(pd.to_numeric(window[name], errors="coerce").fillna(0.0).sum())

    bid_start = n(first, "bid_depth_5bps_base")
    bid_end = n(latest, "bid_depth_5bps_base")
    ask_start = n(first, "ask_depth_5bps_base")
    ask_end = n(latest, "ask_depth_5bps_base")
    imb_start = n(first, "depth_imbalance_25bps")
    imb_end = n(latest, "depth_imbalance_25bps")
    bid_series = pd.to_numeric(window.get("bid_depth_5bps_base"), errors="coerce")
    bid_min = float(bid_series.min()) if bid_series.notna().any() else np.nan
    recovery_den = bid_start - bid_min if np.isfinite(bid_start) and np.isfinite(bid_min) else np.nan
    recovery = _ratio(bid_end - bid_min, recovery_den) if np.isfinite(recovery_den) and recovery_den > 0 else np.nan

    aggressive_sell = s("aggressive_sell_base")
    aggressive_buy = s("aggressive_buy_base")
    bid_added = s("book_added_bid_base")
    bid_removed = s("book_removed_bid_base")
    bid_cancel = s("estimated_bid_cancel_base")
    bid_consumed = s("estimated_bid_consumed_base")
    bid_replenished = s("estimated_bid_replenished_base")
    mean_bid = float(bid_series.mean()) if bid_series.notna().any() else np.nan

    result: dict[str, object] = {
        "book_metric_time": metric_time,
        "book_age_seconds": age,
        "book_window_rows": len(window),
        "book_valid_fraction": valid_fraction,
        "book_trade_attribution_valid_fraction": flow_fraction,
        "book_bid_depth_5bps_base": bid_end,
        "book_ask_depth_5bps_base": ask_end,
        "book_bid_depth_25bps_base": n(latest, "bid_depth_25bps_base"),
        "book_ask_depth_25bps_base": n(latest, "ask_depth_25bps_base"),
        "book_depth_imbalance_25bps": imb_end,
        "book_top_bid_wall_ratio": n(latest, "top_bid_wall_ratio"),
        "book_top_bid_wall_distance_bps": n(latest, "top_bid_wall_distance_bps"),
        "book_bid_depth_5bps_change": _ratio(bid_end - bid_start, bid_start),
        "book_bid_depth_5bps_recovery_fraction": recovery,
        "book_ask_depth_5bps_change": _ratio(ask_end - ask_start, ask_start),
        "book_depth_imbalance_change": imb_end - imb_start if np.isfinite(imb_end) and np.isfinite(imb_start) else np.nan,
        "book_aggressive_sell_base_lookback": aggressive_sell,
        "book_aggressive_buy_base_lookback": aggressive_buy,
        "book_added_bid_base_lookback": bid_added,
        "book_removed_bid_base_lookback": bid_removed,
        "book_bid_cancel_base_lookback": bid_cancel,
        "book_bid_consumed_base_lookback": bid_consumed,
        "book_bid_replenished_base_lookback": bid_replenished,
        "book_bid_replenished_to_consumed": _ratio(bid_replenished, bid_consumed),
        "book_bid_cancel_share_of_removal": _ratio(bid_cancel, bid_removed),
        "book_bid_added_to_removed": _ratio(bid_added, bid_removed),
        "book_bid_replenished_per_aggressive_sell": _ratio(bid_replenished, aggressive_sell),
        "book_aggressive_sell_to_mean_bid_depth_5bps": _ratio(aggressive_sell, mean_bid),
        "book_flow_valid": bool(flow_fraction >= cfg.books_min_valid_fraction),
        "books_causal_valid": bool(
            valid_fraction >= cfg.books_min_valid_fraction
            and age >= -1e-9
            and age <= cfg.books_max_staleness_seconds
        ),
    }
    return result


def _chunks(start: pd.Timestamp, end: pd.Timestamp, days: int = 30):
    cursor = start.normalize()
    while cursor <= end:
        chunk_end = min(end, cursor + pd.Timedelta(days=days) - pd.Timedelta(microseconds=1))
        yield cursor, chunk_end
        cursor = chunk_end.normalize() + pd.Timedelta(days=1)


def attach_books_context(
    events: pd.DataFrame,
    *,
    loader: OKXLiquidityMapLoader,
    config: PostSweepFootprintBooksConfig,
    progress: bool = True,
) -> BooksBuildResult:
    """Attach causal compact Books flow to matched events over actual coverage."""

    cfg = config.validate()
    if events.empty:
        return BooksBuildResult(pd.DataFrame(), pd.DataFrame())
    base = events[["checkpoint_id", "checkpoint_available_time"]].drop_duplicates("checkpoint_id").copy()
    base["checkpoint_available_time"] = pd.to_datetime(base["checkpoint_available_time"], errors="coerce")
    base = base.dropna(subset=["checkpoint_available_time"]).sort_values("checkpoint_available_time", kind="mergesort")
    coverage = books_coverage_table(loader)
    if coverage.empty:
        context = base.copy()
        for name in BOOK_FEATURE_COLUMNS:
            context[name] = False if name in {"book_flow_valid", "books_causal_valid"} else np.nan
        context["book_window_rows"] = 0
        return BooksBuildResult(
            context=context,
            audit=pd.DataFrame([{"status": "no_compact_books_coverage", "events": len(base)}]),
        )

    # Coverage metadata is partitioned by UTC day while event timestamps are
    # project-local naive time.  Convert only the broad coverage envelope to
    # project time so years without Books are filled immediately instead of
    # iterating event-by-event.  Gaps inside the envelope remain explicit NaNs.
    offset = _project_timezone_offset()
    first_utc_day = pd.Timestamp(coverage["day"].min())
    last_utc_day = pd.Timestamp(coverage["day"].max())
    coverage_start = first_utc_day + offset
    coverage_end = last_utc_day + pd.Timedelta(days=1) + offset
    eligible_mask = base["checkpoint_available_time"].between(
        coverage_start - pd.Timedelta(seconds=cfg.books_lookback_seconds),
        coverage_end + pd.Timedelta(seconds=cfg.books_max_staleness_seconds),
        inclusive="both",
    )
    eligible = base.loc[eligible_mask].copy()
    ineligible = base.loc[~eligible_mask].copy()

    if eligible.empty:
        context = base[["checkpoint_id"]].copy()
        for name in BOOK_FEATURE_COLUMNS:
            context[name] = False if name in {"book_flow_valid", "books_causal_valid"} else np.nan
        context["book_window_rows"] = 0
        return BooksBuildResult(
            context=context,
            audit=pd.DataFrame(
                [
                    {
                        "status": "events_outside_compact_books_coverage",
                        "events": len(base),
                        "coverage_start_project": coverage_start,
                        "coverage_end_project": coverage_end,
                    }
                ]
            ),
        )

    chunks = list(_chunks(eligible["checkpoint_available_time"].min(), eligible["checkpoint_available_time"].max()))
    reporter = ProgressReporter("[books] causal chunks", len(chunks), every=1, enabled=progress)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        mask = eligible["checkpoint_available_time"].between(chunk_start, chunk_end, inclusive="both")
        chunk_events = eligible.loc[mask].copy()
        if chunk_events.empty:
            reporter.update(index)
            continue
        load_start = chunk_start - pd.Timedelta(seconds=cfg.books_lookback_seconds + 10)
        features = loader.load_features(
            load_start,
            chunk_end,
            project_time=True,
            index_mode="none",
            valid_only=False,
        )
        if not features.empty:
            features = features.sort_values("available_time", kind="mergesort").reset_index(drop=True)
        causal_count = 0
        for event in chunk_events.itertuples(index=False):
            extracted = _event_book_features(features, pd.Timestamp(event.checkpoint_available_time), cfg)
            causal_count += int(bool(extracted.get("books_causal_valid", False)))
            rows.append({"checkpoint_id": event.checkpoint_id, **extracted})
        audits.append(
            {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "events": len(chunk_events),
                "source_rows": len(features),
                "causal_attached": causal_count,
                "status": "complete" if len(features) else "missing_compact_books_rows",
            }
        )
        reporter.update(index)
    reporter.close()

    context = pd.DataFrame(rows)
    if not ineligible.empty:
        missing = ineligible[["checkpoint_id"]].copy()
        for name in BOOK_FEATURE_COLUMNS:
            missing[name] = False if name in {"book_flow_valid", "books_causal_valid"} else np.nan
        missing["book_window_rows"] = 0
        context = pd.concat([context, missing], ignore_index=True, sort=False)
    if context["checkpoint_id"].duplicated().any():
        raise RuntimeError("duplicate Books context checkpoints")
    return BooksBuildResult(context=context, audit=pd.DataFrame(audits))
