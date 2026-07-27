#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline Books + Raw Trades liquidity feature builder.

The builder is intentionally event-driven and day-partitioned.  It never loads a
month of L2 updates into memory.  Raw trades are reduced to one-day, one-second
price buckets first; the reconstructed order book is then sampled on a fixed
clock.  Every strategy-facing row has an explicit ``available_time_ms`` later
than the sampled interval.
"""

from __future__ import annotations

import heapq
from array import array
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from .models import BookEvent, LiquidityBuildStats, LiquidityMapConfig
from .replay import OrderBookReplay
from .schemas import FEATURE_DTYPES, HEATMAP_DTYPES

UTC = timezone.utc
FlowBucketMap = dict[int, dict[tuple[str, int], float]]

_ARRAY_TYPECODES = {
    ("u", 1): "B",
    ("i", 1): "b",
    ("i", 2): "h",
    ("i", 4): "i",
    ("i", 8): "q",
    ("f", 4): "f",
    ("f", 8): "d",
}


class _TypedColumnBuffer:
    """Append-only compact typed columns for variable-size heatmap output."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.columns: dict[str, array] = {}
        for name, dtype in schema.items():
            dt = np.dtype(dtype)
            try:
                code = _ARRAY_TYPECODES[(dt.kind, dt.itemsize)]
            except KeyError as exc:  # pragma: no cover - schema is static
                raise TypeError(f"unsupported column dtype: {name}={dt}") from exc
            self.columns[name] = array(code)

    def extend_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            for name, dtype in self.schema.items():
                default = math.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
                self.columns[name].append(row.get(name, default))

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            name: np.frombuffer(column, dtype=np.dtype(self.schema[name]))
            for name, column in self.columns.items()
        }


def _allocate_fixed_columns(schema: dict[str, Any], rows: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, dtype in schema.items():
        dt = np.dtype(dtype)
        fill = np.nan if np.issubdtype(dt, np.floating) else 0
        out[name] = np.full(rows, fill, dtype=dt)
    return out


def _write_fixed_row(columns: dict[str, np.ndarray], position: int, row: dict[str, Any]) -> None:
    for name, values in columns.items():
        if name in row:
            values[position] = row[name]



class OfflineLiquidityMapBuilder:
    def __init__(self, config: LiquidityMapConfig):
        config.validate()
        self.config = config

    def aggregate_trades(
        self,
        trade_chunks: Iterable[pd.DataFrame],
        *,
        stats: LiquidityBuildStats | None = None,
    ) -> tuple[dict[int, dict[int, list[float]]], dict[int, list[float]]]:
        """Reduce raw trades to one-day causal time/price buckets.

        Each input chunk is vectorized with pandas and released before the next
        chunk is read.  The returned dictionaries contain only one UTC day of
        compact aggregates, never the original tick rows.
        """

        by_price: dict[int, dict[int, list[float]]] = {}
        by_time: dict[int, list[float]] = {}
        feature_ms = self.config.feature_ms
        step = self.config.price_step
        cv = self.config.contract_value_base
        total_rows = 0

        for chunk in trade_chunks:
            if chunk is None or chunk.empty:
                continue
            required = {"ts_ms", "price", "size", "side"}
            missing = required.difference(chunk.columns)
            if missing:
                raise ValueError(f"raw trade chunk missing columns: {sorted(missing)}")
            work = pd.DataFrame(
                {
                    "ts_ms": pd.to_numeric(chunk["ts_ms"], errors="coerce"),
                    "price": pd.to_numeric(chunk["price"], errors="coerce"),
                    "size": pd.to_numeric(chunk["size"], errors="coerce"),
                    "side": chunk["side"].astype(str).str.lower(),
                }
            )
            work = work.dropna(subset=["ts_ms", "price", "size"])
            work = work.loc[work["size"] >= 0]
            work = work.loc[work["side"].isin(["buy", "sell"])]
            if work.empty:
                continue
            total_rows += int(len(work))
            work["bucket"] = (work["ts_ms"].astype("int64") // feature_ms) * feature_ms
            work["price_index"] = np.floor(work["price"].astype(float) / step + 1e-12).astype("int64")
            work["base"] = work["size"].astype(float) * cv
            grouped = (
                work.groupby(["bucket", "price_index", "side"], sort=False, observed=True)["base"]
                .sum()
                .unstack("side", fill_value=0.0)
            )
            for (bucket, price_index), row in grouped.iterrows():
                bucket_map = by_price.setdefault(int(bucket), {})
                pair = bucket_map.setdefault(int(price_index), [0.0, 0.0])
                pair[0] += float(row.get("buy", 0.0))
                pair[1] += float(row.get("sell", 0.0))
            totals = work.groupby(["bucket", "side"], sort=False, observed=True)["base"].sum().unstack("side", fill_value=0.0)
            for bucket, row in totals.iterrows():
                pair = by_time.setdefault(int(bucket), [0.0, 0.0])
                pair[0] += float(row.get("buy", 0.0))
                pair[1] += float(row.get("sell", 0.0))

        if stats is not None:
            stats.raw_trade_rows = total_rows
            stats.trade_buckets = sum(len(bucket) for bucket in by_price.values())
        return by_price, by_time

    def build_day(
        self,
        day: str | date,
        *,
        book_events: Iterable[BookEvent],
        trade_by_price: dict[int, dict[int, list[float]]] | None = None,
        trade_by_time: dict[int, list[float]] | None = None,
        stats: LiquidityBuildStats | None = None,
        progress_every_events: int = 250_000,
        trade_attribution_valid: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], LiquidityBuildStats, set[str]]:
        d = self._parse_day(day)
        stat = stats or LiquidityBuildStats(day=d.isoformat())
        cfg = self.config
        replay = OrderBookReplay(price_step=cfg.price_step, strict_sequence=cfg.strict_sequence)
        trade_price = trade_by_price or {}
        trade_time = trade_by_time or {}
        day_start_ms = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
        day_end_ms = day_start_ms + 86_400_000
        next_feature_end = day_start_ms + cfg.feature_ms
        next_heatmap_end = day_start_ms + cfg.heatmap_ms
        feature_row_count = 86_400_000 // cfg.feature_ms
        feature_columns = _allocate_fixed_columns(FEATURE_DTYPES, feature_row_count)
        feature_position = 0
        heatmap_columns = _TypedColumnBuffer(HEATMAP_DTYPES)
        sources: set[str] = set()

        # Per-feature-bucket exact book changes keyed by side and price bin.
        # Bucket-first nesting avoids repeatedly scanning every outstanding
        # flow key when emitting one second of features.
        added: FlowBucketMap = {}
        removed: FlowBucketMap = {}
        heatmap_accum: dict[tuple[str, int], list[float]] = {}
        heatmap_sample_count = 0
        last_event_ts: int | None = None
        cached_revision = -1
        cached_book_values: dict[str, Any] | None = None
        cached_retained_bins: dict[str, list[tuple[int, float, int]]] | None = None

        def current_book_view() -> tuple[dict[str, Any], dict[str, list[tuple[int, float, int]]]]:
            nonlocal cached_revision, cached_book_values, cached_retained_bins
            if (
                cached_book_values is None
                or cached_retained_bins is None
                or cached_revision != replay.revision
            ):
                cached_book_values, cached_retained_bins = self._build_book_view(replay)
                cached_revision = replay.revision
            return cached_book_values, cached_retained_bins

        def emit_feature(bucket_end_ms: int) -> None:
            nonlocal heatmap_sample_count, next_heatmap_end, feature_position
            bucket_start_ms = bucket_end_ms - cfg.feature_ms
            book_usable = self._book_is_usable(replay, bucket_end_ms)
            book_values = None
            retained_bins = None
            if book_usable:
                book_values, retained_bins = current_book_view()
            snapshot = self._snapshot_feature_row(
                replay,
                book_usable=book_usable,
                book_values=book_values,
                bucket_start_ms=bucket_start_ms,
                bucket_end_ms=bucket_end_ms,
                trade_time=trade_time,
                added=added,
                removed=removed,
                trade_attribution_valid=trade_attribution_valid,
            )
            _write_fixed_row(feature_columns, feature_position, snapshot)
            feature_position += 1
            self._accumulate_heatmap_snapshot(
                replay,
                heatmap_accum,
                book_usable=book_usable,
                retained_bins=retained_bins,
                bucket_start_ms=bucket_start_ms,
                added=added,
                removed=removed,
                trade_by_price=trade_price,
                trade_attribution_valid=trade_attribution_valid,
            )
            heatmap_sample_count += 1
            if bucket_end_ms >= next_heatmap_end:
                heatmap_columns.extend_rows(
                    self._flush_heatmap(
                        replay,
                        heatmap_accum,
                        start_ms=next_heatmap_end - cfg.heatmap_ms,
                        end_ms=next_heatmap_end,
                        sample_count=heatmap_sample_count,
                        trade_attribution_valid=trade_attribution_valid,
                        end_retained_bins=retained_bins,
                        end_book_usable=book_usable,
                    )
                )
                heatmap_accum.clear()
                heatmap_sample_count = 0
                while next_heatmap_end <= bucket_end_ms:
                    next_heatmap_end += cfg.heatmap_ms
            self._drop_bucket_metrics(bucket_start_ms, added, removed, trade_price, trade_time)

        for event in book_events:
            stat.book_events += 1
            if event.is_snapshot:
                stat.snapshots += 1
            else:
                stat.updates += 1
            if event.source_file:
                sources.add(event.source_file)
            stat.first_event_ms = event.ts_ms if stat.first_event_ms is None else min(stat.first_event_ms, event.ts_ms)
            stat.last_event_ms = event.ts_ms if stat.last_event_ms is None else max(stat.last_event_ms, event.ts_ms)
            if last_event_ts is not None and event.ts_ms < last_event_ts:
                stat.invalid_events += 1
                if not stat.warnings or "out-of-order" not in stat.warnings[-1]:
                    stat.warnings.append(
                        f"out-of-order book event skipped: {event.ts_ms} < {last_event_ts} at {event.source_file}:{event.source_line}"
                    )
                continue
            last_event_ts = event.ts_ms
            if event.ts_ms < day_start_ms:
                # Earlier snapshot/update can seed the day state, so apply it but
                # do not assign its deltas to the target day.
                replay.apply(event)
                continue
            if event.ts_ms >= day_end_ms:
                break
            while next_feature_end <= event.ts_ms and next_feature_end <= day_end_ms:
                emit_feature(next_feature_end)
                next_feature_end += cfg.feature_ms
            deltas, gap = replay.apply(event)
            if gap:
                stat.sequence_gaps += 1
            if replay.valid and deltas:
                bucket = int(event.ts_ms // cfg.feature_ms * cfg.feature_ms)
                for delta in deltas:
                    idx = replay.price_index(delta.price)
                    if delta.added_contracts > 0:
                        bucket_map = added.setdefault(bucket, {})
                        key = (delta.side, idx)
                        bucket_map[key] = bucket_map.get(key, 0.0) + delta.added_contracts * cfg.contract_value_base
                    if delta.removed_contracts > 0:
                        bucket_map = removed.setdefault(bucket, {})
                        key = (delta.side, idx)
                        bucket_map[key] = bucket_map.get(key, 0.0) + delta.removed_contracts * cfg.contract_value_base
            if progress_every_events > 0 and stat.book_events % progress_every_events == 0:
                print(
                    f"[books] events={stat.book_events:,} snapshots={stat.snapshots:,} "
                    f"updates={stat.updates:,} feature_rows={feature_position:,} "
                    f"heatmap_cells={len(heatmap_columns):,}",
                    flush=True,
                )

        while next_feature_end <= day_end_ms:
            emit_feature(next_feature_end)
            next_feature_end += cfg.feature_ms

        # Flush a partial heatmap interval only when it contains samples.
        if heatmap_accum and heatmap_sample_count:
            partial_end = min(day_end_ms, next_heatmap_end)
            partial_start = partial_end - heatmap_sample_count * cfg.feature_ms
            heatmap_columns.extend_rows(
                self._flush_heatmap(
                    replay,
                    heatmap_accum,
                    start_ms=partial_start,
                    end_ms=partial_end,
                    sample_count=heatmap_sample_count,
                    trade_attribution_valid=trade_attribution_valid,
                    end_retained_bins=(current_book_view()[1] if self._book_is_usable(replay, partial_end) else None),
                    end_book_usable=self._book_is_usable(replay, partial_end),
                )
            )

        if feature_position != feature_row_count:
            feature_columns = {name: values[:feature_position] for name, values in feature_columns.items()}
        stat.book_feature_rows = feature_position
        stat.heatmap_cells = len(heatmap_columns)
        return feature_columns, heatmap_columns.to_numpy(), stat, sources

    def _book_is_usable(self, replay: OrderBookReplay, sample_end_ms: int) -> bool:
        if not replay.valid or replay.last_ts_ms is None:
            return False
        return 0 <= sample_end_ms - replay.last_ts_ms <= self.config.max_book_staleness_ms

    def _build_book_view(
        self,
        replay: OrderBookReplay,
    ) -> tuple[dict[str, Any], dict[str, list[tuple[int, float, int]]]]:
        """Build all static book-state metrics once per replay revision.

        A 5000-level historical snapshot is commonly reused by several 1-second
        feature rows before the next book event arrives.  Recomputing depth
        windows, wall candidates and retained heatmap bins for every clock tick
        was the dominant CPU cost.  This view is immutable for one replay
        revision and can therefore be reused without changing causal timing.
        """

        cfg = self.config
        nan = float("nan")
        best_bid = replay.best_bid
        best_ask = replay.best_ask
        if best_bid is None or best_ask is None:
            values = {name: nan for name in self._float_feature_names()}
            values.update({"large_bid_bins": 0, "large_ask_bins": 0})
            return values, {"bid": [], "ask": []}

        best_bid = float(best_bid)
        best_ask = float(best_ask)
        mid = (best_bid + best_ask) / 2.0
        values: dict[str, Any] = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": (best_ask - best_bid) / mid * 10_000 if mid > 0 else nan,
        }

        side_data: dict[str, list[tuple[int, float, int, float]]] = {"bid": [], "ask": []}
        retained: dict[str, list[tuple[int, float, int]]] = {"bid": [], "ask": []}
        for side in ("bid", "ask"):
            current = side_data[side]
            for idx, contracts, orders in replay.iter_binned_depth(side):
                price = replay.price_for_index(idx)
                distance_pct = abs(price - mid) / mid if mid > 0 else math.inf
                if distance_pct <= cfg.max_distance_pct:
                    current.append(
                        (idx, contracts * cfg.contract_value_base, orders, distance_pct * 10_000)
                    )

            if current:
                max_depth = max(item[1] for item in current)
                threshold = max(cfg.min_store_depth_base, max_depth * cfg.min_store_ratio)
                selected = [(item[0], item[1], item[2]) for item in current if item[1] >= threshold]
                if cfg.max_levels_per_side > 0 and len(selected) > cfg.max_levels_per_side:
                    selected = heapq.nlargest(
                        cfg.max_levels_per_side,
                        selected,
                        key=lambda item: item[1],
                    )
                retained[side] = selected

        for bps in (5, 10, 25, 50):
            values[f"bid_depth_{bps}bps_base"] = sum(
                item[1] for item in side_data["bid"] if item[3] <= bps
            )
            values[f"ask_depth_{bps}bps_base"] = sum(
                item[1] for item in side_data["ask"] if item[3] <= bps
            )
        bid25 = values["bid_depth_25bps_base"]
        ask25 = values["ask_depth_25bps_base"]
        denom = bid25 + ask25
        values["depth_imbalance_25bps"] = (bid25 - ask25) / denom if denom > 0 else 0.0

        for side in ("bid", "ask"):
            current = side_data[side]
            if current:
                top = max(current, key=lambda item: item[1])
                max_depth = top[1]
                large = [item for item in current if max_depth > 0 and item[1] / max_depth >= cfg.large_depth_ratio]
                top_price = replay.price_for_index(top[0])
                nearest_large = (
                    max(large, key=lambda item: item[0])
                    if side == "bid"
                    else min(large, key=lambda item: item[0])
                )
                values[f"top_{side}_wall_price"] = top_price
                values[f"top_{side}_wall_depth_base"] = max_depth
                values[f"top_{side}_wall_ratio"] = 1.0
                values[f"top_{side}_wall_distance_bps"] = (
                    abs(top_price - mid) / mid * 10_000 if mid > 0 else nan
                )
                values[f"nearest_large_{side}_price"] = replay.price_for_index(nearest_large[0])
                values[f"nearest_large_{side}_depth_base"] = nearest_large[1]
                values[f"large_{side}_depth_base"] = sum(item[1] for item in large)
                values[f"large_{side}_bins"] = len(large)
            else:
                values[f"top_{side}_wall_price"] = nan
                values[f"top_{side}_wall_depth_base"] = 0.0
                values[f"top_{side}_wall_ratio"] = 0.0
                values[f"top_{side}_wall_distance_bps"] = nan
                values[f"nearest_large_{side}_price"] = nan
                values[f"nearest_large_{side}_depth_base"] = 0.0
                values[f"large_{side}_depth_base"] = 0.0
                values[f"large_{side}_bins"] = 0
        return values, retained

    def _snapshot_feature_row(
        self,
        replay: OrderBookReplay,
        *,
        book_usable: bool,
        book_values: dict[str, Any] | None,
        bucket_start_ms: int,
        bucket_end_ms: int,
        trade_time: dict[int, list[float]],
        added: FlowBucketMap,
        removed: FlowBucketMap,
        trade_attribution_valid: bool,
    ) -> dict[str, Any]:
        cfg = self.config
        row: dict[str, Any] = {
            "bucket_start_ms": bucket_start_ms,
            "bucket_end_ms": bucket_end_ms,
            "available_time_ms": bucket_end_ms + cfg.decision_delay_ms,
            "book_valid": int(book_usable),
            "trade_attribution_valid": int(trade_attribution_valid),
        }
        nan = float("nan")
        if not book_usable or not book_values:
            row.update({name: nan for name in self._float_feature_names()})
            row.update({"large_bid_bins": 0, "large_ask_bins": 0})
            buy, sell = trade_time.get(bucket_start_ms, [0.0, 0.0])
            row["aggressive_buy_base"] = buy
            row["aggressive_sell_base"] = sell
            row["trade_delta_base"] = buy - sell
            for name in self._book_flow_feature_names():
                row[name] = 0.0
            return row

        row.update(book_values)

        buy, sell = trade_time.get(bucket_start_ms, [0.0, 0.0])
        row["aggressive_buy_base"] = buy
        row["aggressive_sell_base"] = sell
        row["trade_delta_base"] = buy - sell
        self._fill_book_flow_summary(
            row, bucket_start_ms, added, removed, buy=buy, sell=sell,
            trade_attribution_valid=trade_attribution_valid,
        )
        return row

    def _fill_book_flow_summary(
        self,
        row: dict[str, Any],
        bucket_start_ms: int,
        added: FlowBucketMap,
        removed: FlowBucketMap,
        *,
        buy: float,
        sell: float,
        trade_attribution_valid: bool,
    ) -> None:
        added_bucket = added.get(bucket_start_ms, {})
        removed_bucket = removed.get(bucket_start_ms, {})
        bid_add = sum(value for (side, _), value in added_bucket.items() if side == "bid")
        ask_add = sum(value for (side, _), value in added_bucket.items() if side == "ask")
        bid_remove = sum(value for (side, _), value in removed_bucket.items() if side == "bid")
        ask_remove = sum(value for (side, _), value in removed_bucket.items() if side == "ask")
        bid_consumed = min(bid_remove, sell) if trade_attribution_valid else 0.0
        ask_consumed = min(ask_remove, buy) if trade_attribution_valid else 0.0
        row.update(
            {
                "book_added_bid_base": bid_add,
                "book_added_ask_base": ask_add,
                "book_removed_bid_base": bid_remove,
                "book_removed_ask_base": ask_remove,
                "estimated_bid_cancel_base": max(bid_remove - sell, 0.0) if trade_attribution_valid else 0.0,
                "estimated_ask_cancel_base": max(ask_remove - buy, 0.0) if trade_attribution_valid else 0.0,
                "estimated_bid_consumed_base": bid_consumed,
                "estimated_ask_consumed_base": ask_consumed,
                "estimated_bid_replenished_base": min(bid_add, sell) if trade_attribution_valid else 0.0,
                "estimated_ask_replenished_base": min(ask_add, buy) if trade_attribution_valid else 0.0,
            }
        )

    def _retained_snapshot_bins(
        self,
        replay: OrderBookReplay,
        side: str,
    ) -> list[tuple[int, float, int]]:
        """Return the currently retained price bins for one book side.

        The same retention rule is used for both the time-averaged heatmap and
        the exact end-of-bucket snapshot fields.  This prevents the display
        layer from comparing two subtly different book universes.
        """

        if replay.mid_price is None:
            return []
        cfg = self.config
        mid = float(replay.mid_price)
        filtered: list[tuple[int, float, int]] = []
        for idx, contracts, orders in replay.iter_binned_depth(side):
            price = replay.price_for_index(idx)
            if mid > 0 and abs(price - mid) / mid <= cfg.max_distance_pct:
                filtered.append((idx, contracts * cfg.contract_value_base, orders))
        if not filtered:
            return []
        max_depth = max(item[1] for item in filtered)
        threshold = max(cfg.min_store_depth_base, max_depth * cfg.min_store_ratio)
        selected = [item for item in filtered if item[1] >= threshold]
        if cfg.max_levels_per_side > 0 and len(selected) > cfg.max_levels_per_side:
            selected = heapq.nlargest(cfg.max_levels_per_side, selected, key=lambda item: item[1])
        return selected

    def _accumulate_heatmap_snapshot(
        self,
        replay: OrderBookReplay,
        accum: dict[tuple[str, int], list[float]],
        *,
        book_usable: bool,
        retained_bins: dict[str, list[tuple[int, float, int]]] | None,
        bucket_start_ms: int,
        added: FlowBucketMap,
        removed: FlowBucketMap,
        trade_by_price: dict[int, dict[int, list[float]]],
        trade_attribution_valid: bool,
    ) -> None:
        if not book_usable or replay.mid_price is None or retained_bins is None:
            return
        cfg = self.config
        mid = float(replay.mid_price)
        trade_bucket = trade_by_price.get(bucket_start_ms, {})
        added_bucket = added.get(bucket_start_ms, {})
        removed_bucket = removed.get(bucket_start_ms, {})

        def values_for(side: str, idx: int) -> list[float]:
            key = (side, idx)
            values = accum.get(key)
            if values is None:
                # depth_sum, max_depth, last_depth, order_sum, sample_count,
                # added, removed, executed
                values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                accum[key] = values
            return values

        for side in ("bid", "ask"):
            selected = retained_bins[side]
            touched: set[int] = set()
            for idx, depth_base, orders in selected:
                trade = trade_bucket.get(idx, [0.0, 0.0])
                executed = (trade[0] if side == "ask" else trade[1]) if trade_attribution_valid else 0.0
                add = added_bucket.get((side, idx), 0.0)
                remove = removed_bucket.get((side, idx), 0.0)
                values = values_for(side, idx)
                values[0] += depth_base
                values[1] = max(values[1], depth_base)
                values[2] = depth_base
                values[3] += orders
                values[4] += 1.0
                values[5] += add
                values[6] += remove
                values[7] += executed
                touched.add(idx)

            # Keep price-level flow facts even when the level was fully removed
            # before the fixed-clock sample.  These rows have zero display depth
            # and are normally hidden by the UI, but remain available to event
            # studies and backtests.
            flow_indices = {idx for (flow_side, idx) in added_bucket if flow_side == side}
            flow_indices.update(
                idx for (flow_side, idx) in removed_bucket if flow_side == side
            )
            flow_indices.update(
                idx for idx, pair in trade_bucket.items()
                if (pair[0] if side == "ask" else pair[1]) > 0
            )
            for idx in flow_indices.difference(touched):
                price = replay.price_for_index(idx)
                if mid <= 0 or abs(price - mid) / mid > cfg.max_distance_pct:
                    continue
                trade = trade_bucket.get(idx, [0.0, 0.0])
                executed = (trade[0] if side == "ask" else trade[1]) if trade_attribution_valid else 0.0
                add = added_bucket.get((side, idx), 0.0)
                remove = removed_bucket.get((side, idx), 0.0)
                if add <= 0 and remove <= 0 and executed <= 0:
                    continue
                values = values_for(side, idx)
                values[5] += add
                values[6] += remove
                values[7] += executed

    def _flush_heatmap(
        self,
        replay: OrderBookReplay,
        accum: dict[tuple[str, int], list[float]],
        *,
        start_ms: int,
        end_ms: int,
        sample_count: int,
        trade_attribution_valid: bool,
        end_retained_bins: dict[str, list[tuple[int, float, int]]] | None,
        end_book_usable: bool,
    ) -> list[dict[str, Any]]:
        if sample_count <= 0:
            return []
        cfg = self.config
        expected_samples = max(float(sample_count), 1.0)

        # Existing fields remain time-weighted averages for research/backtest
        # compatibility.  The new end_* fields are the exact reconstructed
        # book state at the bucket boundary (events with ts < end_ms).
        averaged: dict[tuple[str, int], tuple[float, list[float]]] = {}
        average_side_max: dict[str, float] = {"bid": 0.0, "ask": 0.0}
        for key, values in accum.items():
            avg_depth = values[0] / expected_samples
            averaged[key] = (avg_depth, values)
            average_side_max[key[0]] = max(average_side_max[key[0]], avg_depth)

        end_snapshot: dict[tuple[str, int], tuple[float, int]] = {}
        end_side_max: dict[str, float] = {"bid": 0.0, "ask": 0.0}
        if replay.mid_price is not None and end_book_usable and end_retained_bins is not None:
            for side in ("bid", "ask"):
                for idx, depth_base, orders in end_retained_bins[side]:
                    end_snapshot[(side, idx)] = (float(depth_base), int(orders))
                    end_side_max[side] = max(end_side_max[side], float(depth_base))

        keys = set(averaged).union(end_snapshot)
        rows: list[dict[str, Any]] = []
        empty_values = [0.0] * 8
        for side, idx in sorted(keys, key=lambda item: (item[0], item[1])):
            avg_depth, values = averaged.get((side, idx), (0.0, empty_values))
            end_depth, end_orders = end_snapshot.get((side, idx), (0.0, 0))
            activity = values[5] + values[6] + values[7]
            if avg_depth < cfg.min_store_depth_base and end_depth < cfg.min_store_depth_base and activity <= 0:
                continue
            average_local_ratio = (
                avg_depth / average_side_max[side] if average_side_max[side] > 0 else 0.0
            )
            end_local_ratio = (
                end_depth / end_side_max[side] if end_side_max[side] > 0 else 0.0
            )
            added_base = values[5]
            removed_base = values[6]
            executed_base = values[7]
            consumed = min(removed_base, executed_base)
            price = replay.price_for_index(idx)
            rows.append(
                {
                    "bucket_start_ms": start_ms,
                    "bucket_end_ms": end_ms,
                    "price_index": idx,
                    "side_code": 1 if side == "bid" else -1,
                    "flow_valid": int(trade_attribution_valid),
                    "depth_base": avg_depth,
                    "depth_usd": avg_depth * price,
                    "order_count": int(round(values[3] / expected_samples)),
                    "local_depth_ratio": average_local_ratio,
                    "end_depth_base": end_depth,
                    "end_depth_usd": end_depth * price,
                    "end_order_count": int(end_orders),
                    "end_local_depth_ratio": end_local_ratio,
                    "added_base": added_base,
                    "removed_base": removed_base,
                    "executed_base": executed_base,
                    "cancelled_base": max(removed_base - executed_base, 0.0) if trade_attribution_valid else 0.0,
                    "consumed_base": consumed if trade_attribution_valid else 0.0,
                    "replenished_base": min(added_base, executed_base) if trade_attribution_valid else 0.0,
                }
            )
        return rows

    def _drop_bucket_metrics(
        self,
        bucket_start_ms: int,
        added: FlowBucketMap,
        removed: FlowBucketMap,
        trade_price: dict[int, dict[int, list[float]]],
        trade_time: dict[int, list[float]],
    ) -> None:
        added.pop(bucket_start_ms, None)
        removed.pop(bucket_start_ms, None)
        trade_price.pop(bucket_start_ms, None)
        trade_time.pop(bucket_start_ms, None)

    @staticmethod
    def _float_feature_names() -> tuple[str, ...]:
        return (
            "best_bid",
            "best_ask",
            "mid_price",
            "spread_bps",
            "bid_depth_5bps_base",
            "ask_depth_5bps_base",
            "bid_depth_10bps_base",
            "ask_depth_10bps_base",
            "bid_depth_25bps_base",
            "ask_depth_25bps_base",
            "bid_depth_50bps_base",
            "ask_depth_50bps_base",
            "depth_imbalance_25bps",
            "top_bid_wall_price",
            "top_ask_wall_price",
            "top_bid_wall_depth_base",
            "top_ask_wall_depth_base",
            "top_bid_wall_ratio",
            "top_ask_wall_ratio",
            "top_bid_wall_distance_bps",
            "top_ask_wall_distance_bps",
            "nearest_large_bid_price",
            "nearest_large_ask_price",
            "nearest_large_bid_depth_base",
            "nearest_large_ask_depth_base",
            "large_bid_depth_base",
            "large_ask_depth_base",
        )

    @staticmethod
    def _book_flow_feature_names() -> tuple[str, ...]:
        return (
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

    @staticmethod
    def _parse_day(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()
