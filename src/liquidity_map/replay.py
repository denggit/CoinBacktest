#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Incremental order-book replay state used by offline backtests."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import BookEvent, BookLevel


# A $1 ETH heatmap bin contains roughly 100 native $0.01 book levels.
# Repeated float additions/removals can leave ~1e-11 contract residues after
# every exact level has disappeared.  Broad-map mode deliberately has no
# economic depth filter, so numerical dust must be removed at the replay layer.
_DEPTH_ABS_EPSILON = 1e-9
_DEPTH_REL_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class LevelDelta:
    side: str
    price: float
    old_size_contracts: float
    new_size_contracts: float
    added_contracts: float
    removed_contracts: float


class OrderBookReplay:
    """Reconstruct a full book from snapshots and incremental updates.

    The replay keeps both exact price levels and price-binned aggregates.  The
    latter make fixed-clock feature extraction much cheaper than rescanning raw
    event payloads.  A sequence discontinuity invalidates the book until the
    next snapshot; no future snapshot is used to repair the gap.
    """

    def __init__(self, *, price_step: float, strict_sequence: bool = True):
        if price_step <= 0:
            raise ValueError("price_step must be > 0")
        self.price_step = float(price_step)
        self.strict_sequence = bool(strict_sequence)
        self.bids: dict[float, BookLevel] = {}
        self.asks: dict[float, BookLevel] = {}
        self.bid_bins: dict[int, float] = defaultdict(float)
        self.ask_bins: dict[int, float] = defaultdict(float)
        self.bid_order_bins: dict[int, int] = defaultdict(int)
        self.ask_order_bins: dict[int, int] = defaultdict(int)
        self.valid = False
        self.last_seq_id: int | None = None
        self.last_ts_ms: int | None = None
        self.sequence_gaps = 0
        self.revision = 0
        self._best_bid: float | None = None
        self._best_ask: float | None = None

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.bid_bins.clear()
        self.ask_bins.clear()
        self.bid_order_bins.clear()
        self.ask_order_bins.clear()
        self.valid = False
        self.last_seq_id = None
        self.last_ts_ms = None
        self._best_bid = None
        self._best_ask = None
        self.revision += 1

    def price_index(self, price: float) -> int:
        # Rounding rather than floor prevents binary-float noise around exact
        # integer/tick boundaries while still making bins deterministic.
        return int(math.floor(float(price) / self.price_step + 1e-12))

    def price_for_index(self, index: int) -> float:
        return float(index) * self.price_step

    def apply(self, event: BookEvent) -> tuple[list[LevelDelta], bool]:
        """Apply one event and return (level_deltas, sequence_gap)."""

        gap = False
        if event.is_snapshot:
            # Some historical exports store repeated full snapshots rather than
            # WebSocket deltas.  Diff consecutive valid snapshots so removal,
            # addition and trade-consumption attribution remain available.
            deltas = self._replace_snapshot(event, with_deltas=self.valid)
            self.last_seq_id = event.seq_id
            self.last_ts_ms = event.ts_ms
            self.valid = True
            self.revision += 1
            return deltas, False

        if not self.valid:
            return [], False

        if self.strict_sequence and self._is_sequence_gap(event):
            self.sequence_gaps += 1
            self.clear()
            return [], True

        deltas: list[LevelDelta] = []
        deltas.extend(self._apply_levels("bid", event.bids))
        deltas.extend(self._apply_levels("ask", event.asks))
        if event.seq_id is not None:
            self.last_seq_id = event.seq_id
        self.last_ts_ms = event.ts_ms
        if deltas:
            self.revision += 1
        return deltas, gap

    def _is_sequence_gap(self, event: BookEvent) -> bool:
        if event.prev_seq_id is None or self.last_seq_id is None:
            return False
        # OKX heartbeat/no-change update can have prevSeqId == seqId == last.
        if event.prev_seq_id == self.last_seq_id:
            return False
        return True

    def _diff_snapshot(self, event: BookEvent) -> list[LevelDelta]:
        out: list[LevelDelta] = []
        for side, old_levels, new_sequence in (
            ("bid", self.bids, event.bids),
            ("ask", self.asks, event.asks),
        ):
            new_levels = {level.price: level for level in new_sequence if level.size_contracts > 0}
            for price in old_levels.keys() | new_levels.keys():
                old_size = old_levels.get(price).size_contracts if price in old_levels else 0.0
                new_size = new_levels.get(price).size_contracts if price in new_levels else 0.0
                diff = new_size - old_size
                if abs(diff) <= 1e-15:
                    continue
                out.append(
                    LevelDelta(
                        side=side,
                        price=float(price),
                        old_size_contracts=float(old_size),
                        new_size_contracts=float(new_size),
                        added_contracts=max(float(diff), 0.0),
                        removed_contracts=max(float(-diff), 0.0),
                    )
                )
        return out

    def _replace_snapshot(self, event: BookEvent, *, with_deltas: bool = False) -> list[LevelDelta]:
        """Replace the full book in one pass.

        Historical 400/5000-level exports are dominated by repeated full
        snapshots.  The previous implementation constructed the new snapshot
        once for diffing and then iterated it a second time to rebuild the
        replay state.  Building exact levels and aggregate bins together cuts
        that duplicated work while preserving identical level deltas.
        """

        new_bids = {level.price: level for level in event.bids if level.size_contracts > 0}
        new_asks = {level.price: level for level in event.asks if level.size_contracts > 0}
        deltas: list[LevelDelta] = []
        if with_deltas:
            for side, old_levels, new_levels in (
                ("bid", self.bids, new_bids),
                ("ask", self.asks, new_asks),
            ):
                for price in old_levels.keys() | new_levels.keys():
                    old_size = old_levels.get(price).size_contracts if price in old_levels else 0.0
                    new_size = new_levels.get(price).size_contracts if price in new_levels else 0.0
                    diff = float(new_size) - float(old_size)
                    if abs(diff) <= 1e-15:
                        continue
                    deltas.append(
                        LevelDelta(
                            side=side,
                            price=float(price),
                            old_size_contracts=float(old_size),
                            new_size_contracts=float(new_size),
                            added_contracts=max(diff, 0.0),
                            removed_contracts=max(-diff, 0.0),
                        )
                    )

        self.bids = new_bids
        self.asks = new_asks
        self.bid_bins = defaultdict(float)
        self.ask_bins = defaultdict(float)
        self.bid_order_bins = defaultdict(int)
        self.ask_order_bins = defaultdict(int)
        self._seed_levels("bid", new_bids.values(), update_exact=False)
        self._seed_levels("ask", new_asks.values(), update_exact=False)
        self._best_bid = max(new_bids) if new_bids else None
        self._best_ask = min(new_asks) if new_asks else None
        return deltas

    def _seed_levels(
        self,
        side: str,
        levels: Iterable[BookLevel],
        *,
        update_exact: bool = True,
    ) -> None:
        exact = self.bids if side == "bid" else self.asks
        bins = self.bid_bins if side == "bid" else self.ask_bins
        order_bins = self.bid_order_bins if side == "bid" else self.ask_order_bins
        for level in levels:
            if level.size_contracts <= 0:
                continue
            if update_exact:
                exact[level.price] = level
            idx = self.price_index(level.price)
            bins[idx] += level.size_contracts
            order_bins[idx] += level.order_count

    def _apply_levels(self, side: str, levels: Iterable[BookLevel]) -> list[LevelDelta]:
        exact = self.bids if side == "bid" else self.asks
        bins = self.bid_bins if side == "bid" else self.ask_bins
        order_bins = self.bid_order_bins if side == "bid" else self.ask_order_bins
        out: list[LevelDelta] = []
        best_dirty = False
        for level in levels:
            old = exact.get(level.price)
            old_size = old.size_contracts if old else 0.0
            old_orders = old.order_count if old else 0
            new_size = max(0.0, float(level.size_contracts))
            idx = self.price_index(level.price)
            size_diff = new_size - old_size
            order_diff = int(level.order_count) - old_orders

            # Update the exact book first; it is the source of truth used by
            # best_bid/best_ask and lets us recover safely if aggregate float
            # arithmetic ever produces an impossible negative bin.
            if new_size <= 0:
                exact.pop(level.price, None)
                if side == "bid" and self._best_bid == level.price:
                    best_dirty = True
                elif side == "ask" and self._best_ask == level.price:
                    best_dirty = True
            else:
                exact[level.price] = BookLevel(level.price, new_size, level.order_count)
                if side == "bid" and (self._best_bid is None or level.price > self._best_bid):
                    self._best_bid = float(level.price)
                elif side == "ask" and (self._best_ask is None or level.price < self._best_ask):
                    self._best_ask = float(level.price)

            if abs(size_diff) > 1e-15:
                current_bin = float(bins.get(idx, 0.0))
                updated_bin = current_bin + size_diff
                tolerance = max(
                    _DEPTH_ABS_EPSILON,
                    _DEPTH_REL_EPSILON * (abs(current_bin) + abs(size_diff) + 1.0),
                )
                if abs(updated_bin) <= tolerance:
                    bins.pop(idx, None)
                elif updated_bin < 0:
                    # This should only happen through accumulated round-off.
                    # Rebuild the affected bin from exact levels instead of
                    # allowing a negative or crossed ghost level to survive.
                    rebuilt = math.fsum(
                        item.size_contracts
                        for price, item in exact.items()
                        if self.price_index(price) == idx
                    )
                    if rebuilt <= tolerance:
                        bins.pop(idx, None)
                    else:
                        bins[idx] = rebuilt
                else:
                    bins[idx] = updated_bin
            if order_diff:
                order_bins[idx] += order_diff
                if order_bins[idx] <= 0:
                    order_bins.pop(idx, None)
            out.append(
                LevelDelta(
                    side=side,
                    price=level.price,
                    old_size_contracts=old_size,
                    new_size_contracts=new_size,
                    added_contracts=max(size_diff, 0.0),
                    removed_contracts=max(-size_diff, 0.0),
                )
            )
        if best_dirty:
            if side == "bid":
                self._best_bid = max(exact) if exact else None
            else:
                self._best_ask = min(exact) if exact else None
        return out

    @property
    def best_bid(self) -> float | None:
        return self._best_bid

    @property
    def best_ask(self) -> float | None:
        return self._best_ask

    @property
    def mid_price(self) -> float | None:
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def iter_binned_depth(self, side: str):
        bins = self.bid_bins if side == "bid" else self.ask_bins
        order_bins = self.bid_order_bins if side == "bid" else self.ask_order_bins
        for idx, size in bins.items():
            if size > _DEPTH_ABS_EPSILON:
                yield idx, float(size), int(order_bins.get(idx, 0))
