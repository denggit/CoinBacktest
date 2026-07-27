#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event-driven martingale execution engine for bar and raw-trade replay."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import EngineConfig, MartingaleVariant

EPS = 1e-12


class MartingaleEngine:
    """Single-variant event-driven long martingale simulator."""

    def __init__(self, variant: MartingaleVariant, config: EngineConfig):
        variant.validate()
        config.validate()
        self.variant = variant
        self.config = config

        self.initial_capital = float(config.initial_capital)
        self.capital = float(config.initial_capital)
        self.realized_peak = float(config.initial_capital)
        self.mtm_peak = float(config.initial_capital)
        self.max_mtm_drawdown = 0.0
        self.min_mtm_equity = float(config.initial_capital)
        self.last_price: float | None = None
        self.last_time: pd.Timestamp | None = None
        self.bankrupt = False

        self.trades: list[dict[str, Any]] = []
        self.fills: list[dict[str, Any]] = []
        self._daily_snapshots: dict[str, dict[str, Any]] = {}

        self.cycle_id = 0
        self.anchor_price: float | None = None
        self.order_prices: list[float] = []
        self.order_notionals: list[float] = []
        self.order_weights: list[float] = []
        self.next_order_index = 0
        self.cycle_start_capital = self.capital
        self.cycle_planned_notional = 0.0

        self.qty = 0.0
        self.cost = 0.0
        self.avg_entry = 0.0
        self.entry_fees = 0.0
        self.entry_time: pd.Timestamp | None = None
        self.first_entry = 0.0
        self.target_price: float | None = None
        self.max_fav_price: float | None = None
        self.max_adv_price: float | None = None

    # ------------------------------------------------------------------
    # Order-ladder construction
    # ------------------------------------------------------------------
    def start_cycle(self, anchor_price: float, ts: Any) -> None:
        if self.bankrupt:
            return
        anchor = float(anchor_price)
        if not math.isfinite(anchor) or anchor <= 0:
            raise ValueError(f"invalid anchor price: {anchor_price!r}")

        self.cycle_id += 1
        self.anchor_price = anchor
        self.cycle_start_capital = float(self.capital)
        self.cycle_planned_notional = (
            self.cycle_start_capital
            * float(self.variant.leverage)
            * float(self.config.capital_utilization)
        )

        prices: list[float] = []
        current = anchor
        for order_index in range(self.variant.total_orders):
            gap = float(self.variant.entry_drop_pct) * (
                float(self.variant.spacing_multiplier) ** order_index
            )
            if not 0 < gap < 1:
                raise ValueError(
                    f"invalid ladder gap={gap} at order_index={order_index}; "
                    "entry_drop_pct * spacing_multiplier**index must stay in (0, 1)"
                )
            current *= 1.0 - gap
            prices.append(current)

        weights = [1.0]
        for add_index in range(1, self.variant.max_additions + 1):
            weights.append(
                float(self.variant.initial_add_ratio)
                * (float(self.variant.amount_multiplier) ** (add_index - 1))
            )
        weight_sum = float(sum(weights))
        notionals = [self.cycle_planned_notional * weight / weight_sum for weight in weights]

        self.order_prices = prices
        self.order_weights = weights
        self.order_notionals = notionals
        self.next_order_index = 0

        self.qty = 0.0
        self.cost = 0.0
        self.avg_entry = 0.0
        self.entry_fees = 0.0
        self.entry_time = None
        self.first_entry = 0.0
        self.target_price = None
        self.max_fav_price = None
        self.max_adv_price = None
        self.last_time = pd.Timestamp(ts)
        self.last_price = anchor

    def ensure_started(self, price: float, ts: Any) -> None:
        if self.anchor_price is None and not self.bankrupt:
            self.start_cycle(price, ts)

    @property
    def in_position(self) -> bool:
        return self.qty > EPS

    @property
    def next_order_price(self) -> float | None:
        if self.next_order_index >= len(self.order_prices):
            return None
        return float(self.order_prices[self.next_order_index])

    @property
    def additions_filled(self) -> int:
        return max(0, int(self.next_order_index) - 1)

    @property
    def total_filled_orders(self) -> int:
        return int(self.next_order_index)

    def liquidation_price(self) -> float | None:
        """Approximate cross-margin liquidation trigger for the current long.

        Trigger equation includes entry fees, estimated exit fee, and configured
        maintenance margin.  It deliberately avoids pretending to reproduce OKX
        tiered mark-price liquidation exactly.
        """
        if not self.in_position:
            return None
        available_equity = float(self.capital) - float(self.entry_fees)
        numerator = float(self.avg_entry) * float(self.qty) - available_equity
        denominator = float(self.qty) * (
            1.0 - float(self.config.fee_rate) - float(self.config.maintenance_margin_rate)
        )
        if denominator <= EPS or numerator <= 0:
            return None
        price = numerator / denominator
        if not math.isfinite(price) or price <= 0:
            return None
        return float(price)

    def _fill_next_order(self, ts: Any, source: str) -> None:
        if self.next_order_index >= len(self.order_prices):
            return
        order_index = int(self.next_order_index)
        price = float(self.order_prices[order_index])
        notional = float(self.order_notionals[order_index])
        qty = notional / price
        fee = notional * float(self.config.fee_rate)

        if not self.in_position:
            self.entry_time = pd.Timestamp(ts)
            self.first_entry = price
            self.max_fav_price = price
            self.max_adv_price = price

        self.qty += qty
        self.cost += price * qty
        self.entry_fees += fee
        self.avg_entry = self.cost / self.qty
        self.next_order_index += 1
        self.target_price = self.avg_entry * (1.0 + float(self.variant.take_profit_pct))

        self.fills.append(
            {
                "variant": self.variant.key,
                "cycle_id": self.cycle_id,
                "fill_time": pd.Timestamp(ts),
                "source": source,
                "order_index": order_index,
                "order_role": "INITIAL" if order_index == 0 else f"ADD_{order_index}",
                "fill_price": price,
                "notional": notional,
                "qty": qty,
                "fee": fee,
                "total_qty_after": self.qty,
                "avg_entry_after": self.avg_entry,
                "target_after": self.target_price,
                "liquidation_price_after": self.liquidation_price(),
                "capital_before_cycle": self.cycle_start_capital,
            }
        )

    # ------------------------------------------------------------------
    # Equity / close handling
    # ------------------------------------------------------------------
    def _equity_at_price(self, price: float) -> float:
        """Return net liquidation-value equity at ``price``."""
        p = float(price)
        if not self.in_position:
            return max(0.0, float(self.capital))
        estimated_exit_fee = p * float(self.qty) * float(self.config.fee_rate)
        equity = (
            float(self.capital)
            - float(self.entry_fees)
            + (p - float(self.avg_entry)) * float(self.qty)
            - estimated_exit_fee
        )
        return max(0.0, float(equity))

    def mark_equity(self, price: float) -> float:
        equity = self._equity_at_price(price)
        self.mtm_peak = max(self.mtm_peak, equity)
        self.min_mtm_equity = min(self.min_mtm_equity, equity)
        if self.mtm_peak > 0:
            self.max_mtm_drawdown = max(
                self.max_mtm_drawdown,
                (self.mtm_peak - equity) / self.mtm_peak,
            )
        return equity

    def _mark_equity_segment(self, prices: np.ndarray) -> None:
        """Update MTM statistics for an ordered fixed-position price segment.

        Raw-trade replay skips prints that cannot touch an order, TP, or
        liquidation threshold.  Position size is therefore constant inside the
        skipped segment, allowing exact vectorized equity/drawdown accounting.
        """
        if not self.in_position or len(prices) == 0:
            return
        values = np.asarray(prices, dtype="float64")
        if values.size == 0:
            return
        equities = (
            float(self.capital)
            - float(self.entry_fees)
            + (values - float(self.avg_entry)) * float(self.qty)
            - values * float(self.qty) * float(self.config.fee_rate)
        )
        np.maximum(equities, 0.0, out=equities)

        initial_peak = float(self.mtm_peak)
        running_peaks = np.maximum.accumulate(equities)
        np.maximum(running_peaks, initial_peak, out=running_peaks)
        valid = running_peaks > 0.0
        if np.any(valid):
            segment_drawdown = np.max(
                (running_peaks[valid] - equities[valid]) / running_peaks[valid]
            )
            self.max_mtm_drawdown = max(
                float(self.max_mtm_drawdown),
                float(segment_drawdown),
            )
        self.mtm_peak = max(initial_peak, float(np.max(equities)))
        self.min_mtm_equity = min(
            float(self.min_mtm_equity),
            float(np.min(equities)),
        )

    def _update_extrema(self, low: float, high: float) -> None:
        if not self.in_position:
            return
        if self.max_adv_price is None:
            self.max_adv_price = float(low)
        else:
            self.max_adv_price = min(float(self.max_adv_price), float(low))
        if self.max_fav_price is None:
            self.max_fav_price = float(high)
        else:
            self.max_fav_price = max(float(self.max_fav_price), float(high))

    def _close_position(
        self,
        ts: Any,
        exit_price: float,
        reason: str,
        *,
        restart_cycle: bool = True,
    ) -> None:
        if not self.in_position:
            return
        exit_ts = pd.Timestamp(ts)
        px = float(exit_price)
        self.mark_equity(px)
        qty = float(self.qty)
        avg = float(self.avg_entry)
        entry_fee = float(self.entry_fees)
        exit_fee = qty * px * float(self.config.fee_rate)
        gross_pnl = (px - avg) * qty
        raw_pnl = gross_pnl - entry_fee - exit_fee

        cap_before = float(self.capital)
        cap_after = max(0.0, cap_before + raw_pnl)
        pnl = cap_after - cap_before
        self.capital = cap_after
        self.realized_peak = max(self.realized_peak, self.capital)
        self.mtm_peak = max(self.mtm_peak, self.capital)
        self.min_mtm_equity = min(self.min_mtm_equity, self.capital)

        max_fav = float(self.max_fav_price if self.max_fav_price is not None else avg)
        max_adv = float(self.max_adv_price if self.max_adv_price is not None else avg)
        mfe_pct = (max_fav / avg - 1.0) if avg > 0 else 0.0
        mae_pct = (max_adv / avg - 1.0) if avg > 0 else 0.0
        holding_seconds = (
            max(0.0, (exit_ts - self.entry_time).total_seconds())
            if self.entry_time is not None
            else 0.0
        )
        liq = reason == "LIQUIDATION"

        self.trades.append(
            {
                "variant": self.variant.key,
                "cycle_id": self.cycle_id,
                "entry_time": self.entry_time,
                "exit_time": exit_ts,
                "type": "LONG",
                "entry": avg,
                "first_entry": self.first_entry,
                "avg_entry": avg,
                "exit": px,
                "exit_price": px,
                "target": self.target_price,
                "qty": qty,
                "filled_orders": self.total_filled_orders,
                "additions": self.additions_filled,
                "max_additions": self.variant.max_additions,
                "leverage": self.variant.leverage,
                "cycle_planned_notional": self.cycle_planned_notional,
                "actual_entry_notional": self.cost,
                "gross_pnl": gross_pnl,
                "pnl": pnl,
                "fee": entry_fee + exit_fee,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "capital": self.capital,
                "return_pct": pnl / max(cap_before, EPS),
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "mfe_r": mfe_pct,
                "mae_r": abs(mae_pct),
                "holding_seconds": holding_seconds,
                "holding_hours": holding_seconds / 3600.0,
                "note": reason,
                "liquidated": liq,
            }
        )

        self.qty = 0.0
        self.cost = 0.0
        self.avg_entry = 0.0
        self.entry_fees = 0.0
        self.entry_time = None
        self.first_entry = 0.0
        self.target_price = None
        self.max_fav_price = None
        self.max_adv_price = None

        self.last_time = exit_ts
        self.last_price = px
        if self.capital <= EPS:
            self.bankrupt = True
            self.anchor_price = None
            self.order_prices = []
            self.order_notionals = []
            self.order_weights = []
            self.next_order_index = 0
        elif restart_cycle:
            self.start_cycle(px, exit_ts)
        else:
            self.anchor_price = None
            self.order_prices = []
            self.order_notionals = []
            self.order_weights = []
            self.next_order_index = 0

    def snapshot(self, ts: Any, price: float) -> None:
        stamp = pd.Timestamp(ts)
        p = float(price)
        equity = self.mark_equity(p)
        key = stamp.strftime("%Y-%m-%d")
        self._daily_snapshots[key] = {
            "time": stamp,
            "price": p,
            "capital_realized": self.capital,
            "equity_mtm": equity,
            "drawdown_pct": (
                (self.mtm_peak - equity) / self.mtm_peak if self.mtm_peak > 0 else 0.0
            ),
            "in_position": self.in_position,
            "avg_entry": self.avg_entry if self.in_position else np.nan,
            "qty": self.qty,
            "filled_orders": self.total_filled_orders if self.in_position else 0,
            "next_order_price": self.next_order_price,
            "target_price": self.target_price,
            "liquidation_price": self.liquidation_price(),
        }
        self.last_time = stamp
        self.last_price = p

    # ------------------------------------------------------------------
    # Conservative OHLC/range-bar replay
    # ------------------------------------------------------------------
    def _process_downward_touch(self, ts: Any, low: float, source: str) -> tuple[int, bool]:
        """Replay the downward path to ``low`` in threshold order.

        Returns ``(fills_count, liquidated)``.
        """
        fills_before = self.total_filled_orders
        low_px = float(low)

        while not self.bankrupt:
            if not self.in_position:
                next_buy = self.next_order_price
                if next_buy is not None and low_px <= next_buy:
                    self._fill_next_order(ts, source)
                    continue
                break

            next_buy = self.next_order_price
            liq = self.liquidation_price()

            # As price descends, the higher threshold occurs first. A pending
            # add below liquidation cannot be filled before liquidation.
            if next_buy is not None and (liq is None or next_buy > liq):
                if low_px <= next_buy:
                    self._fill_next_order(ts, source)
                    continue

            liq = self.liquidation_price()
            if liq is not None and low_px <= liq:
                self._update_extrema(low_px, self.avg_entry)
                self.mark_equity(liq)
                self._close_position(ts, liq, "LIQUIDATION")
                return self.total_filled_orders - fills_before, True
            break

        return self.total_filled_orders - fills_before, False

    def process_bar(
        self,
        ts: Any,
        open_price: float,
        high: float,
        low: float,
        close: float,
        *,
        source: str,
    ) -> None:
        if self.bankrupt:
            return
        stamp = pd.Timestamp(ts)
        o = float(open_price)
        h = float(high)
        lo = float(low)
        c = float(close)
        if not all(math.isfinite(x) and x > 0 for x in (o, h, lo, c)):
            return
        if lo > h:
            lo, h = h, lo

        self.ensure_started(o, stamp)
        had_position = self.in_position
        fills_count, liquidated = self._process_downward_touch(stamp, lo, source)
        if liquidated or self.bankrupt:
            self.last_time = stamp
            self.last_price = c
            return

        if self.in_position:
            # The low is always causally reachable under the conservative path.
            if self.max_adv_price is None:
                self.max_adv_price = lo
            else:
                self.max_adv_price = min(float(self.max_adv_price), lo)
            self.mark_equity(lo)

            # Only use the bar high and allow TP if no size was added in this bar.
            # Otherwise high may have occurred before the low/add fills.
            if fills_count == 0:
                if had_position:
                    if self.max_fav_price is None:
                        self.max_fav_price = h
                    else:
                        self.max_fav_price = max(float(self.max_fav_price), h)
                if self.target_price is not None and h >= self.target_price:
                    target = float(self.target_price)
                    self._close_position(stamp, target, "TAKE_PROFIT")
                    self.last_time = stamp
                    self.last_price = c
                    return
                self.mark_equity(h)

        self.mark_equity(c)
        self.last_time = stamp
        self.last_price = c

    # ------------------------------------------------------------------
    # Exact-sequence raw-trade replay
    # ------------------------------------------------------------------
    def _next_down_threshold(self) -> float | None:
        if not self.in_position:
            return self.next_order_price
        next_buy = self.next_order_price
        liq = self.liquidation_price()
        if next_buy is None:
            return liq
        if liq is None:
            return next_buy
        return max(next_buy, liq)

    def process_tick(self, ts: Any, price: float) -> None:
        if self.bankrupt:
            return
        stamp = pd.Timestamp(ts)
        p = float(price)
        if not math.isfinite(p) or p <= 0:
            return
        self.ensure_started(p, stamp)

        fills_before = self.total_filled_orders
        while not self.bankrupt:
            if not self.in_position:
                next_buy = self.next_order_price
                if next_buy is not None and p <= next_buy:
                    self._fill_next_order(stamp, "raw_trade")
                    continue
                break

            next_buy = self.next_order_price
            liq = self.liquidation_price()
            if next_buy is not None and (liq is None or next_buy > liq) and p <= next_buy:
                self._fill_next_order(stamp, "raw_trade")
                continue
            liq = self.liquidation_price()
            if liq is not None and p <= liq:
                self._update_extrema(p, self.avg_entry)
                self.mark_equity(liq)
                self._close_position(stamp, liq, "LIQUIDATION")
                self.last_time = stamp
                self.last_price = p
                return
            break

        fills_count = self.total_filled_orders - fills_before
        if self.in_position:
            self._update_extrema(p, p)
            self.mark_equity(p)
            # A limit fill and TP cannot both use the exact same trade print.
            if fills_count == 0 and self.target_price is not None and p >= self.target_price:
                target = float(self.target_price)
                self._close_position(stamp, target, "TAKE_PROFIT")
        self.last_time = stamp
        self.last_price = p

    def process_tick_chunk(self, timestamps: np.ndarray, prices: np.ndarray) -> None:
        """Event-skip a normalized raw-trade chunk using vectorized searches."""
        n = int(len(prices))
        if n == 0 or self.bankrupt:
            return
        pos = 0
        while pos < n and not self.bankrupt:
            if self.anchor_price is None:
                self.ensure_started(float(prices[pos]), pd.Timestamp(timestamps[pos]))
                pos += 1
                continue

            down = self._next_down_threshold()
            tp = self.target_price if self.in_position else None
            segment = prices[pos:]
            cond = np.zeros(len(segment), dtype=bool)
            if down is not None:
                cond |= segment <= float(down)
            if tp is not None:
                cond |= segment >= float(tp)

            hit = np.flatnonzero(cond)
            if hit.size == 0:
                if self.in_position:
                    self._update_extrema(float(np.nanmin(segment)), float(np.nanmax(segment)))
                    self._mark_equity_segment(segment)
                self.last_time = pd.Timestamp(timestamps[-1])
                self.last_price = float(prices[-1])
                break

            rel = int(hit[0])
            event_pos = pos + rel
            if self.in_position and rel > 0:
                before = prices[pos:event_pos]
                self._update_extrema(float(np.nanmin(before)), float(np.nanmax(before)))
                self._mark_equity_segment(before)
            self.process_tick(pd.Timestamp(timestamps[event_pos]), float(prices[event_pos]))
            pos = event_pos + 1

    def finalize(self, *, force_close: bool | None = None) -> None:
        if force_close is None:
            force_close = self.config.force_close_end
        if self.last_time is None or self.last_price is None:
            return
        if self.in_position and bool(force_close):
            self._update_extrema(self.last_price, self.last_price)
            self._close_position(
                self.last_time,
                self.last_price,
                "FORCE_CLOSE_END",
                restart_cycle=False,
            )
        self.snapshot(self.last_time, self.last_price)

    def equity_frame(self) -> pd.DataFrame:
        if not self._daily_snapshots:
            return pd.DataFrame(
                columns=[
                    "price",
                    "capital_realized",
                    "equity_mtm",
                    "drawdown_pct",
                    "in_position",
                ]
            )
        out = pd.DataFrame(self._daily_snapshots.values()).sort_values("time")
        return out.set_index("time")

    def open_position_snapshot(self) -> dict[str, Any]:
        return {
            "variant": self.variant.key,
            "as_of": self.last_time,
            "last_price": self.last_price,
            "capital_realized": self.capital,
            "equity_mtm": self.mark_equity(self.last_price) if self.last_price else self.capital,
            "in_position": self.in_position,
            "cycle_id": self.cycle_id,
            "entry_time": self.entry_time,
            "first_entry": self.first_entry if self.in_position else None,
            "avg_entry": self.avg_entry if self.in_position else None,
            "qty": self.qty,
            "filled_orders": self.total_filled_orders if self.in_position else 0,
            "additions": self.additions_filled if self.in_position else 0,
            "next_order_price": self.next_order_price,
            "target_price": self.target_price,
            "liquidation_price": self.liquidation_price(),
            "entry_fees_paid": self.entry_fees,
            "bankrupt": self.bankrupt,
        }


