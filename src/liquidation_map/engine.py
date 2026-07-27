#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal estimated liquidation heatmap engine.

This is deliberately an *estimated* map.  It never claims access to account
positions.  Open-interest changes create probabilistic long/short cohorts,
funding/order flow tilt their relative weights, and transparent leverage buckets
project those cohorts to approximate liquidation prices.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.liquidation_map.models import (
    HeatmapCell,
    LiquidationMapConfig,
    LiquidationMapResult,
    LiquidationZone,
)


@dataclass
class _Level:
    notional: float = 0.0
    confidence_weighted: float = 0.0

    def add(self, notional: float, confidence: float) -> None:
        amount = max(0.0, float(notional))
        self.notional += amount
        self.confidence_weighted += amount * max(0.0, min(1.0, float(confidence)))

    @property
    def confidence(self) -> float:
        return self.confidence_weighted / self.notional if self.notional > 0 else 0.0

    def scale(self, factor: float) -> None:
        factor = max(0.0, min(1.0, float(factor)))
        self.notional *= factor
        self.confidence_weighted *= factor


class EstimatedLiquidationMapEngine:
    def __init__(self, config: LiquidationMapConfig | None = None) -> None:
        self.config = config or LiquidationMapConfig()
        self.config.validate()

    def compute(
        self,
        bars: pd.DataFrame,
        *,
        open_interest: pd.DataFrame,
        funding_rates: pd.DataFrame | None = None,
        mark_prices: pd.DataFrame | None = None,
        liquidations: pd.DataFrame | None = None,
    ) -> LiquidationMapResult:
        frame = self._prepare_frame(bars, open_interest, funding_rates, mark_prices, liquidations)
        if frame.empty:
            return LiquidationMapResult([], pd.DataFrame(), [], {"ready": False, "reason": "empty_input"})
        if frame["oi_usd"].notna().sum() < 2:
            return LiquidationMapResult(
                [],
                self._empty_row_frame(frame.index),
                [],
                {"ready": False, "reason": "open_interest_missing", "oi_rows": int(frame["oi_usd"].notna().sum())},
            )

        median_price = float(pd.to_numeric(frame["model_price"], errors="coerce").dropna().median())
        bucket_step = max(1e-8, median_price * self.config.price_bucket_pct)
        levels: dict[tuple[str, int], _Level] = defaultdict(_Level)
        active_buckets: dict[str, list[int]] = {"long": [], "short": []}
        active_sets: dict[str, set[int]] = {"long": set(), "short": set()}
        leverage_weights = np.asarray(self.config.leverage_weights, dtype=float)
        leverage_weights = leverage_weights / leverage_weights.sum()
        cells_raw: list[dict[str, Any]] = []
        snapshots: list[tuple[pd.Timestamp, dict[tuple[str, int], tuple[float, float]], float]] = []
        causal_intensity_max = 0.0

        summary_rows: list[dict[str, float]] = []
        previous_ts: pd.Timestamp | None = None
        last_decay_ts: pd.Timestamp | None = None
        previous_oi = float("nan")
        created_count = 0
        crossed_count = 0
        oi_update_count = 0

        effective_snapshot_every = max(self.config.snapshot_every_bars, int(math.ceil(len(frame) / 1500.0)))

        for i, (timestamp, values) in enumerate(frame.iterrows()):
            ts = pd.Timestamp(timestamp)
            price = float(values["model_price"])
            high = float(values["high"])
            low = float(values["low"])
            previous_ts = ts
            if last_decay_ts is None:
                last_decay_ts = ts
            if i % 5 == 0:
                elapsed_hours = max(0.0, (ts - last_decay_ts).total_seconds() / 3600.0)
                if elapsed_hours > 0:
                    decay = math.exp(-math.log(2.0) * elapsed_hours / self.config.cohort_half_life_hours)
                    self._scale_all(levels, decay)
                last_decay_ts = ts

            oi_value = float(values["oi_usd"]) if pd.notna(values["oi_usd"]) else float("nan")
            oi_delta = float(values["oi_delta_usd"]) if pd.notna(values["oi_delta_usd"]) else 0.0
            if bool(values["oi_update"]):
                oi_update_count += 1
                if oi_delta > self.config.minimum_oi_delta_usd:
                    long_share, confidence = self._directional_shares(values)
                    short_share = 1.0 - long_share
                    for leverage, weight in zip(self.config.leverage_buckets, leverage_weights):
                        long_liq = self._liquidation_price(price, "long", leverage)
                        short_liq = self._liquidation_price(price, "short", leverage)
                        self._add_level(levels, active_buckets, active_sets, "long", long_liq, oi_delta * long_share * weight, confidence, bucket_step)
                        self._add_level(levels, active_buckets, active_sets, "short", short_liq, oi_delta * short_share * weight, confidence, bucket_step)
                        created_count += 2
                elif oi_delta < 0 and math.isfinite(previous_oi) and previous_oi > 0:
                    reduction = min(self.config.oi_reduction_cap, abs(oi_delta) / previous_oi)
                    self._scale_all(levels, 1.0 - reduction)
                if math.isfinite(oi_value):
                    previous_oi = oi_value

            crossed_count += self._consume_crossed_levels(levels, active_buckets, active_sets, low, high, bucket_step)
            self._apply_observed_liquidations(levels, values, bucket_step)
            self._drop_dust(levels)

            if i % effective_snapshot_every == 0:
                snapshot = self._snapshot(levels, price, bucket_step)
                snapshot_max = max((amount for amount, _ in snapshot.values()), default=0.0)
                causal_intensity_max = max(causal_intensity_max, snapshot_max)
                snapshots.append((ts, snapshot, causal_intensity_max))
                last_summary = self._nearest_summary(levels, price, bucket_step)
            elif i == 0:
                last_summary = ((float("nan"), float("nan")), (float("nan"), float("nan")), 0.0, 0.0)

            above, below, balance, confidence = last_summary
            summary_rows.append({
                "nearest_short_liq_price": above[0],
                "nearest_short_liq_distance_pct": above[1],
                "nearest_long_liq_price": below[0],
                "nearest_long_liq_distance_pct": below[1],
                "liquidation_balance": balance,
                "model_confidence": confidence,
                "active_level_count": float(len(levels)),
                "oi_usd": oi_value,
                "oi_delta_usd": oi_delta,
            })

        row = pd.DataFrame(summary_rows, index=frame.index)
        final_ts = pd.Timestamp(frame.index[-1])
        if not snapshots or snapshots[-1][0] != final_ts:
            final_snapshot = self._snapshot(levels, float(frame["model_price"].iloc[-1]), bucket_step)
            final_max = max((amount for amount, _ in final_snapshot.values()), default=0.0)
            causal_intensity_max = max(causal_intensity_max, final_max)
            snapshots.append((final_ts, final_snapshot, causal_intensity_max))
        for snap_index, (start_ts, snapshot, causal_max) in enumerate(snapshots):
            end_ts = snapshots[snap_index + 1][0] if snap_index + 1 < len(snapshots) else frame.index[-1]
            for (side, bucket), (amount, confidence) in snapshot.items():
                if amount <= 0 or causal_max <= 0:
                    continue
                intensity = math.log1p(amount) / math.log1p(causal_max)
                center = bucket * bucket_step
                cells_raw.append(
                    {
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "price_low": center - bucket_step / 2.0,
                        "price_high": center + bucket_step / 2.0,
                        "intensity": intensity,
                        "raw_notional": amount,
                        "side": side,
                        "confidence": confidence,
                    }
                )
        cells = [
            HeatmapCell(
                **item,
                label="多头潜在清算" if item["side"] == "long" else "空头潜在清算",
                fields={"estimated": True},
            )
            for item in cells_raw
        ]
        zones = self._zones_from_last_snapshot(
            snapshots[-1][1] if snapshots else {},
            float(frame["model_price"].iloc[-1]),
            bucket_step,
            snapshots[-1][2] if snapshots else 0.0,
        )
        positive_deltas = pd.to_numeric(frame.loc[frame["oi_update"].eq(True), "oi_delta_usd"], errors="coerce")
        positive_deltas = positive_deltas[positive_deltas > 0]
        oi_usd_direct_rows = int(pd.to_numeric(frame.get("oi_usd_raw"), errors="coerce").notna().sum())
        oi_ccy_rows = int(pd.to_numeric(frame.get("oi_ccy"), errors="coerce").notna().sum())
        if oi_usd_direct_rows:
            oi_value_source = "oi_usd"
        elif oi_ccy_rows:
            oi_value_source = "oi_ccy_x_mark"
        else:
            oi_value_source = "unavailable"
        diagnostics = {
            "ready": True,
            "estimated": True,
            "input_rows": int(len(frame)),
            "oi_rows": int(frame["oi_update"].sum()),
            "oi_value_source": oi_value_source,
            "positive_oi_updates": int(len(positive_deltas)),
            "max_positive_oi_delta_usd": float(positive_deltas.max()) if len(positive_deltas) else 0.0,
            "minimum_oi_delta_usd": float(self.config.minimum_oi_delta_usd),
            "funding_rows": int(frame["funding_rate"].notna().sum()),
            "observed_liquidation_rows": int((frame["long_liq_notional"] + frame["short_liq_notional"] > 0).sum()),
            "cohort_additions": int(created_count),
            "crossed_levels": int(crossed_count),
            "snapshots": int(len(snapshots)),
            "effective_snapshot_every_bars": int(effective_snapshot_every),
            "cells": int(len(cells)),
            "bucket_step": float(bucket_step),
            "model_note": "estimated_from_public_oi_funding_orderflow_not_account_positions",
        }
        if not cells:
            if oi_value_source == "unavailable":
                diagnostics["empty_reason"] = "oi_value_units_unavailable"
            elif not len(positive_deltas):
                diagnostics["empty_reason"] = "no_positive_oi_updates"
            elif diagnostics["max_positive_oi_delta_usd"] <= self.config.minimum_oi_delta_usd:
                diagnostics["empty_reason"] = "oi_delta_below_threshold"
            else:
                diagnostics["empty_reason"] = "all_levels_consumed_or_outside_display_range"
        return LiquidationMapResult(cells, row, zones, diagnostics)

    def _prepare_frame(
        self,
        bars: pd.DataFrame,
        open_interest: pd.DataFrame,
        funding_rates: pd.DataFrame | None,
        mark_prices: pd.DataFrame | None,
        liquidations: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame()
        frame = bars.copy().sort_index()
        frame.index = pd.to_datetime(frame.index)
        for col in ("open", "high", "low", "close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        oi = (open_interest.copy() if open_interest is not None else pd.DataFrame()).sort_index()
        if oi.empty:
            frame["oi_usd_raw"] = np.nan
            frame["oi_ccy"] = np.nan
            frame["oi_contracts"] = np.nan
            frame["oi_update"] = False
        else:
            oi.index = pd.to_datetime(oi.index)
            oi_events = pd.DataFrame(index=oi.index)
            oi_events["oi_usd_raw"] = pd.to_numeric(oi.get("oi_usd"), errors="coerce")
            oi_events["oi_ccy"] = pd.to_numeric(oi.get("oi_ccy"), errors="coerce")
            oi_events["oi_contracts"] = pd.to_numeric(oi.get("oi_contracts"), errors="coerce")
            oi_events["oi_update"] = True
            frame = self._causal_merge_events(
                frame,
                oi_events,
                ffill_columns=("oi_usd_raw", "oi_ccy", "oi_contracts"),
                event_columns=("oi_update",),
            )
            frame["oi_update"] = frame["oi_update"].eq(True)

        funding = funding_rates.copy() if funding_rates is not None else pd.DataFrame()
        if funding.empty:
            frame["funding_rate"] = 0.0
        else:
            funding.index = pd.to_datetime(funding.index)
            series = pd.to_numeric(funding.get("funding_rate"), errors="coerce")
            frame["funding_rate"] = series.reindex(frame.index, method="ffill").fillna(0.0)

        mark = mark_prices.copy() if mark_prices is not None else pd.DataFrame()
        if mark.empty or "close" not in mark:
            frame["model_price"] = frame["close"]
        else:
            mark.index = pd.to_datetime(mark.index)
            frame["model_price"] = pd.to_numeric(mark["close"], errors="coerce").reindex(frame.index, method="ffill").fillna(frame["close"])

        # Prefer OKX-provided USD OI.  Some deployments only return base-asset
        # OI (oiCcy); convert it causally with the mark price available at the
        # OI update.  Do not use contract counts without instrument ctVal.
        raw_usd = pd.to_numeric(frame.get("oi_usd_raw"), errors="coerce")
        oi_ccy = pd.to_numeric(frame.get("oi_ccy"), errors="coerce")
        converted_usd = oi_ccy * pd.to_numeric(frame["model_price"], errors="coerce")
        frame["oi_usd"] = raw_usd.where(raw_usd.notna(), converted_usd)
        frame["oi_delta_usd"] = np.nan
        update_mask = frame["oi_update"].eq(True)
        if update_mask.any():
            update_index = frame.index[update_mask]
            update_values = frame.loc[update_index, "oi_usd"]
            # When oiCcy exists, use its change times the current mark price so
            # price-only moves do not masquerade as newly opened interest.
            update_ccy = oi_ccy.loc[update_index]
            delta_from_ccy = update_ccy.diff() * frame.loc[update_index, "model_price"]
            delta_from_usd = update_values.diff()
            effective_delta = delta_from_ccy.where(update_ccy.notna(), delta_from_usd)
            frame.loc[update_index, "oi_delta_usd"] = effective_delta.to_numpy()

        total = pd.to_numeric(frame.get("buy_notional", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(frame.get("sell_notional", 0.0), errors="coerce").fillna(0.0)
        delta = pd.to_numeric(frame.get("delta_notional", 0.0), errors="coerce").fillna(0.0)
        frame["flow_imbalance"] = (delta / total.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
        frame["momentum"] = np.log(frame["close"].replace(0.0, np.nan)).diff(5).fillna(0.0).clip(-0.10, 0.10)

        frame["long_liq_notional"] = 0.0
        frame["short_liq_notional"] = 0.0
        if liquidations is not None and not liquidations.empty:
            liq = liquidations.copy()
            liq.index = pd.to_datetime(liq.index)
            for ts, item in liq.iterrows():
                pos = int(frame.index.searchsorted(ts, side="left"))
                if pos >= len(frame):
                    continue
                idx = frame.index[pos]
                amount = float(item.get("notional", 0.0) or 0.0)
                side = str(item.get("side", item.get("position_side", ""))).lower()
                if side == "long":
                    frame.loc[idx, "long_liq_notional"] += amount
                elif side == "short":
                    frame.loc[idx, "short_liq_notional"] += amount
        return frame.dropna(subset=["model_price", "high", "low"])

    @staticmethod
    def _causal_merge_events(
        frame: pd.DataFrame,
        events: pd.DataFrame,
        *,
        ffill_columns: tuple[str, ...],
        event_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        out = frame.copy()
        mapped = pd.DataFrame(index=out.index)
        for ts, row in events.iterrows():
            pos = int(out.index.searchsorted(pd.Timestamp(ts), side="left"))
            if pos >= len(out):
                continue
            target = out.index[pos]
            for col in events.columns:
                mapped.loc[target, col] = row[col]
        for col in ffill_columns:
            out[col] = mapped.get(col, pd.Series(index=out.index, dtype=float)).ffill()
        for col in event_columns:
            out[col] = mapped.get(col, pd.Series(index=out.index, dtype=float))
        return out

    def _directional_shares(self, row: pd.Series) -> tuple[float, float]:
        flow = float(row.get("flow_imbalance", 0.0) or 0.0)
        momentum = float(row.get("momentum", 0.0) or 0.0)
        funding = float(row.get("funding_rate", 0.0) or 0.0)
        score = 1.35 * flow + 8.0 * momentum + 120.0 * funding
        long_share = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, score))))
        long_share = max(0.20, min(0.80, long_share))
        confidence = min(0.90, 0.35 + 0.35 * abs(flow) + 2.5 * abs(momentum) + min(0.15, abs(funding) * 500.0))
        return long_share, confidence

    def _liquidation_price(self, entry: float, side: str, leverage: int) -> float:
        margin = 1.0 / float(leverage)
        buffer = self.config.maintenance_margin_rate + self.config.liquidation_fee_buffer
        if side == "long":
            return max(1e-8, entry * (1.0 - margin + buffer))
        return entry * (1.0 + margin - buffer)

    @staticmethod
    def _bucket(price: float, step: float) -> int:
        return int(round(float(price) / step))

    def _add_level(
        self,
        levels: dict[tuple[str, int], _Level],
        active_buckets: dict[str, list[int]],
        active_sets: dict[str, set[int]],
        side: str,
        price: float,
        amount: float,
        confidence: float,
        step: float,
    ) -> None:
        bucket = self._bucket(price, step)
        levels[(side, bucket)].add(amount, confidence)
        if bucket not in active_sets[side]:
            insort(active_buckets[side], bucket)
            active_sets[side].add(bucket)

    @staticmethod
    def _scale_all(levels: dict[tuple[str, int], _Level], factor: float) -> None:
        for level in levels.values():
            level.scale(factor)

    def _consume_crossed_levels(
        self,
        levels: dict[tuple[str, int], _Level],
        active_buckets: dict[str, list[int]],
        active_sets: dict[str, set[int]],
        low: float,
        high: float,
        step: float,
    ) -> int:
        long_threshold = int(math.ceil(low / step))
        long_list = active_buckets["long"]
        long_pos = bisect_left(long_list, long_threshold)
        crossed_long = long_list[long_pos:]
        if crossed_long:
            active_buckets["long"] = long_list[:long_pos]
            active_sets["long"].difference_update(crossed_long)

        short_threshold = int(math.floor(high / step))
        short_list = active_buckets["short"]
        short_pos = bisect_right(short_list, short_threshold)
        crossed_short = short_list[:short_pos]
        if crossed_short:
            active_buckets["short"] = short_list[short_pos:]
            active_sets["short"].difference_update(crossed_short)

        crossed = 0
        for side, buckets in (("long", crossed_long), ("short", crossed_short)):
            for bucket in buckets:
                level = levels.get((side, bucket))
                if level is not None:
                    level.scale(self.config.crossed_level_survival)
                    crossed += 1
        return crossed

    def _apply_observed_liquidations(self, levels: dict[tuple[str, int], _Level], row: pd.Series, step: float) -> None:
        for side, column in (("long", "long_liq_notional"), ("short", "short_liq_notional")):
            observed = float(row.get(column, 0.0) or 0.0)
            if observed <= 0:
                continue
            candidates = [(key, value) for key, value in levels.items() if key[0] == side]
            total = sum(value.notional for _, value in candidates)
            if total <= 0:
                continue
            reduction = min(0.90, observed / total)
            for _, value in candidates:
                value.scale(1.0 - reduction)

    @staticmethod
    def _drop_dust(levels: dict[tuple[str, int], _Level]) -> None:
        for key in [key for key, value in levels.items() if value.notional < 1.0]:
            levels.pop(key, None)

    def _snapshot(self, levels: dict[tuple[str, int], _Level], price: float, step: float) -> dict[tuple[str, int], tuple[float, float]]:
        within = []
        for key, level in levels.items():
            center = key[1] * step
            distance = abs(center / price - 1.0)
            if distance <= self.config.max_distance_pct and level.notional > 0:
                within.append((key, level.notional, level.confidence))
        within.sort(key=lambda item: item[1], reverse=True)
        return {key: (amount, confidence) for key, amount, confidence in within[: self.config.max_cells_per_snapshot]}

    def _nearest_summary(self, levels: dict[tuple[str, int], _Level], price: float, step: float) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        above: list[tuple[float, _Level]] = []
        below: list[tuple[float, _Level]] = []
        long_total = 0.0
        short_total = 0.0
        conf_num = 0.0
        total = 0.0
        for (side, bucket), level in levels.items():
            center = bucket * step
            if side == "short" and center > price:
                above.append((center, level))
                short_total += level.notional
            if side == "long" and center < price:
                below.append((center, level))
                long_total += level.notional
            total += level.notional
            conf_num += level.notional * level.confidence
        nearest_above = min(above, key=lambda item: item[0], default=(float("nan"), _Level()))
        nearest_below = max(below, key=lambda item: item[0], default=(float("nan"), _Level()))
        above_pair = (nearest_above[0], nearest_above[0] / price - 1.0 if math.isfinite(nearest_above[0]) else float("nan"))
        below_pair = (nearest_below[0], nearest_below[0] / price - 1.0 if math.isfinite(nearest_below[0]) else float("nan"))
        balance = (short_total - long_total) / (short_total + long_total) if short_total + long_total > 0 else 0.0
        confidence = conf_num / total if total > 0 else 0.0
        return above_pair, below_pair, balance, confidence

    def _zones_from_last_snapshot(
        self,
        snapshot: dict[tuple[str, int], tuple[float, float]],
        price: float,
        step: float,
        global_max: float,
    ) -> list[LiquidationZone]:
        items = []
        for (side, bucket), (amount, confidence) in snapshot.items():
            center = bucket * step
            intensity = math.log1p(amount) / math.log1p(global_max) if global_max > 0 else 0.0
            items.append(
                LiquidationZone(
                    side=side,
                    price_low=center - step / 2,
                    price_high=center + step / 2,
                    center_price=center,
                    raw_notional=amount,
                    intensity=intensity,
                    distance_pct=center / price - 1.0,
                    confidence=confidence,
                )
            )
        items.sort(key=lambda item: item.raw_notional, reverse=True)
        return items[: self.config.top_zone_count * 2]

    @staticmethod
    def _empty_row_frame(index: pd.Index) -> pd.DataFrame:
        columns = [
            "nearest_short_liq_price",
            "nearest_short_liq_distance_pct",
            "nearest_long_liq_price",
            "nearest_long_liq_distance_pct",
            "liquidation_balance",
            "model_confidence",
            "active_level_count",
            "oi_usd",
            "oi_delta_usd",
        ]
        return pd.DataFrame(index=index, columns=columns, dtype=float)
