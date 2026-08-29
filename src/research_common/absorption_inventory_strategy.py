#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-scale market-control votes and cross-margin inventory simulator.

This module implements a strategy, not an event atlas.  Each closed market bar can
create a *new evidence vote* describing who currently controls price:

- failed aggressive pressure / repeated defense / spring -> vote against pressure;
- efficient aggressive pressure that actually moves price -> vote with pressure.

The signal layer never reads account position, PnL, average entry, TP, SL or time in
position.  Account inventory is only the mechanical sum of executed votes.

Higher-timeframe features are left-labelled and become visible only at
``bar_start + timeframe``.  The 1m executor then trades at the next 1m open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.research_common.multiscale_absorption import (
    AbsorptionFeatureConfig,
    build_absorption_features,
    resample_trade_bars,
)

_EPS = 1e-12


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    rule: str
    available_delta: pd.Timedelta
    config: AbsorptionFeatureConfig


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen semantic thresholds; intentionally no parameter search grid."""

    pressure_z: float = 1.5
    persistence: float = 0.60
    rejection_response_max: float = 0.0
    effective_response_min: float = 0.75
    decay_pressure_z: float = 1.0
    decay_persistence: float = 0.55
    decay_pressure_retention_min: float = 0.80
    decay_response_max: float = 0.50
    decay_response_retention_max: float = 0.50
    defense_count_min: float = 2.0
    defense_hold_ratio_min: float = 0.70
    defense_stability_atr_max: float = 1.50
    # A lower scale can veto a higher-scale reversal only when it shows clearly
    # effective pressure in the opposite direction.  Neutral lower scales do not veto.
    micro_veto_response_min: float = 0.90
    micro_veto_pressure_z: float = 1.5
    # 15m events are allowed only when the 1H context is not strongly opposite.
    require_1h_context_for_15m: bool = True


@dataclass(frozen=True)
class AccountConfig:
    initial_equity: float = 10_000.0
    leverage: float = 10.0
    vote_margin_fraction: float = 0.01
    fee_rate_per_fill: float = 0.00055
    slippage_bps_per_fill: float = 0.0
    maintenance_margin_rate: float = 0.005

    def validate(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")
        if not (0 < self.vote_margin_fraction <= 1):
            raise ValueError("vote_margin_fraction must be in (0, 1]")
        if self.fee_rate_per_fill < 0 or self.slippage_bps_per_fill < 0:
            raise ValueError("costs must be non-negative")
        if not (0 <= self.maintenance_margin_rate < 1):
            raise ValueError("maintenance_margin_rate must be in [0,1)")


def default_scale_specs() -> tuple[ScaleSpec, ...]:
    """One fixed strategy hierarchy chosen from semantic time horizons, not returns."""
    return (
        ScaleSpec(
            "5m",
            "5min",
            pd.Timedelta(minutes=5),
            AbsorptionFeatureConfig(
                process_window=3,
                baseline_bars=288,      # 24h
                baseline_min_periods=144,
                floor_lookback=72,       # 6h
                defense_lookback=144,    # 12h
                reclaim_bars=3,
                atr_lookback=48,
            ),
        ),
        ScaleSpec(
            "15m",
            "15min",
            pd.Timedelta(minutes=15),
            AbsorptionFeatureConfig(
                process_window=3,
                baseline_bars=192,       # 48h
                baseline_min_periods=96,
                floor_lookback=64,        # 16h
                defense_lookback=96,      # 24h
                reclaim_bars=3,
                atr_lookback=48,
            ),
        ),
        ScaleSpec(
            "1H",
            "1h",
            pd.Timedelta(hours=1),
            AbsorptionFeatureConfig(
                process_window=3,
                baseline_bars=168,       # 7d
                baseline_min_periods=84,
                floor_lookback=72,        # 3d
                defense_lookback=96,      # 4d
                reclaim_bars=3,
                atr_lookback=48,
            ),
        ),
        ScaleSpec(
            "4H",
            "4h",
            pd.Timedelta(hours=4),
            AbsorptionFeatureConfig(
                process_window=3,
                baseline_bars=126,       # 21d
                baseline_min_periods=63,
                floor_lookback=42,        # 7d
                defense_lookback=84,      # 14d
                reclaim_bars=3,
                atr_lookback=42,
            ),
        ),
    )


def _num(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def _rising(mask: pd.Series) -> pd.Series:
    x = mask.fillna(False).astype(bool)
    return x & ~x.shift(1, fill_value=False)


def build_scale_states(features: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Classify causal control state and *fresh* vote events on one bar scale."""
    ready = features["feature_ready"].fillna(False).astype(bool)
    side = np.sign(_num(features, "flow_side").fillna(0.0)).astype(np.int8)
    z = _num(features, "pressure_z")
    persistence = _num(features, "flow_persistence")
    response = _num(features, "price_response_norm")

    strong = ready & z.ge(config.pressure_z) & persistence.ge(config.persistence) & side.ne(0)
    rejected = strong & response.le(config.rejection_response_max)
    effective = strong & response.ge(config.effective_response_min)

    same_side = features["same_side_adjacent_window"].fillna(False).astype(bool)
    decay = (
        ready
        & z.ge(config.decay_pressure_z)
        & persistence.ge(config.decay_persistence)
        & same_side
        & _num(features, "pressure_retention").ge(config.decay_pressure_retention_min)
        & response.le(config.decay_response_max)
        & _num(features, "response_retention").le(config.decay_response_retention_max)
        & side.ne(0)
    )

    defense_long = (
        _num(features, "prior_defense_count_long").ge(config.defense_count_min)
        & _num(features, "hold_ratio_long").ge(config.defense_hold_ratio_min)
        & _num(features, "floor_stability_atr").le(config.defense_stability_atr_max)
    )
    defense_short = (
        _num(features, "prior_defense_count_short").ge(config.defense_count_min)
        & _num(features, "hold_ratio_short").ge(config.defense_hold_ratio_min)
        & _num(features, "ceiling_stability_atr").le(config.defense_stability_atr_max)
    )
    spring_long = features["spring_reclaim_long"].fillna(False).astype(bool) & defense_long
    spring_short = features["spring_reclaim_short"].fillna(False).astype(bool) & defense_short

    # Persistent state: useful as a causal context/veto after alignment.
    state = pd.Series(0, index=features.index, dtype=np.int8)
    # Efficient pressure means the aggressor currently controls price.
    state.loc[effective & side.eq(1)] = 1
    state.loc[effective & side.eq(-1)] = -1
    # Rejected/decaying pressure means the passive side currently controls price.
    state.loc[(rejected | decay) & side.eq(-1)] = 1
    state.loc[(rejected | decay) & side.eq(1)] = -1
    state.loc[spring_long] = 1
    state.loc[spring_short] = -1

    # Fresh event votes.  A persistent condition does not spam one vote every bar.
    reversal_event = _rising(rejected | decay)
    continuation_event = _rising(effective)
    spring_long_event = _rising(spring_long)
    spring_short_event = _rising(spring_short)

    vote = pd.Series(0, index=features.index, dtype=np.int8)
    family = pd.Series("", index=features.index, dtype=object)

    rev_long = reversal_event & side.eq(-1)
    rev_short = reversal_event & side.eq(1)
    cont_long = continuation_event & side.eq(1)
    cont_short = continuation_event & side.eq(-1)

    vote.loc[rev_long] = 1
    family.loc[rev_long] = "pressure_failed"
    vote.loc[rev_short] = -1
    family.loc[rev_short] = "pressure_failed"
    vote.loc[cont_long] = 1
    family.loc[cont_long] = "pressure_effective"
    vote.loc[cont_short] = -1
    family.loc[cont_short] = "pressure_effective"

    # A qualified spring/upthrust overrides another same-bar family because it
    # contains the richer repeated-defense path.
    vote.loc[spring_long_event] = 1
    family.loc[spring_long_event] = "defense_spring"
    vote.loc[spring_short_event] = -1
    family.loc[spring_short_event] = "defense_spring"

    return pd.DataFrame(
        {
            "state": state,
            "vote": vote,
            "family": family,
            "pressure_z": z,
            "response_norm": response,
            "flow_side": side,
            "effective_pressure": effective.astype(np.int8),
            "failed_pressure": (rejected | decay).astype(np.int8),
            "spring_long": spring_long.astype(np.int8),
            "spring_short": spring_short.astype(np.int8),
        },
        index=features.index,
    )


def build_multiscale_votes(
    trade_bars_1m: pd.DataFrame,
    *,
    scale_specs: tuple[ScaleSpec, ...] | None = None,
    config: StrategyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 1m signal frame and fresh-vote audit, causally aligned.

    The returned ``signal`` is position-blind.  Multiple scale events on one
    minute are deliberately collapsed to one +1/-1 vote if they agree; if they
    conflict, the higher scale wins (4H > 1H > 15m).  5m is confirmation/veto
    only and never creates inventory by itself.
    """
    if trade_bars_1m.empty:
        raise ValueError("trade_bars_1m is empty")
    specs = scale_specs or default_scale_specs()
    cfg = config or StrategyConfig()
    base = trade_bars_1m.sort_index().copy()
    base.index = pd.to_datetime(base.index)
    base = base[~base.index.duplicated(keep="last")]

    aligned: dict[str, pd.DataFrame] = {}
    raw_events: list[pd.DataFrame] = []

    for spec in specs:
        bars = resample_trade_bars(base, spec.rule)
        features = build_absorption_features(bars, spec.config)
        states = build_scale_states(features, cfg)
        states = states.copy()
        states["bar_start_time"] = states.index
        states["available_time"] = states.index + spec.available_delta
        states["scale"] = spec.name

        ev = states[states["vote"].ne(0)].copy()
        if not ev.empty:
            raw_events.append(ev.reset_index(drop=True))

        ctx = states.set_index("available_time").drop(columns=["bar_start_time"], errors="ignore")
        ctx = ctx[~ctx.index.duplicated(keep="last")].sort_index()
        # asof onto every closed 1m signal time; only already-available HTF bars exist.
        left = pd.DataFrame(index=base.index)
        merged = pd.merge_asof(
            left.sort_index(),
            ctx.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            allow_exact_matches=True,
        )
        # Persistent context can be carried forward, but a fresh event vote must
        # exist only at its exact causal available_time.  Forward-filling `vote`
        # would turn one HTF event into dozens/hundreds of 1m orders.
        event_exact = states.set_index("available_time")[["vote", "family"]].copy()
        event_exact = event_exact[~event_exact.index.duplicated(keep="last")]
        exact = event_exact.reindex(base.index)
        merged["vote"] = pd.to_numeric(exact["vote"], errors="coerce").fillna(0).astype(np.int8)
        merged["family"] = exact["family"].fillna("").astype(str)
        aligned[spec.name] = merged

    signal = pd.Series(0, index=base.index, dtype=np.int8)
    chosen_scale = pd.Series("", index=base.index, dtype=object)
    chosen_family = pd.Series("", index=base.index, dtype=object)
    veto_reason = pd.Series("", index=base.index, dtype=object)

    # 5m is a veto only: clearly effective opposite micro pressure blocks a
    # reversal vote but does not block a same-direction continuation vote.
    micro = aligned["5m"]
    micro_side = np.sign(_num(micro, "flow_side").fillna(0.0)).astype(np.int8)
    micro_veto_long = (
        _num(micro, "pressure_z").ge(cfg.micro_veto_pressure_z)
        & _num(micro, "response_norm").ge(cfg.micro_veto_response_min)
        & micro_side.eq(-1)
    )
    micro_veto_short = (
        _num(micro, "pressure_z").ge(cfg.micro_veto_pressure_z)
        & _num(micro, "response_norm").ge(cfg.micro_veto_response_min)
        & micro_side.eq(1)
    )

    # Prefer higher-scale fresh events when more than one becomes available on
    # the same minute.  5m never reaches this loop as an independent vote.
    for scale in ("4H", "1H", "15m"):
        a = aligned[scale]
        fresh_vote = pd.to_numeric(a["vote"], errors="coerce").fillna(0).astype(np.int8)
        family = a["family"].fillna("").astype(str)

        candidate = signal.eq(0) & fresh_vote.ne(0)
        if scale == "15m" and cfg.require_1h_context_for_15m:
            h1_state = pd.to_numeric(aligned["1H"]["state"], errors="coerce").fillna(0).astype(np.int8)
            candidate &= ~((fresh_vote.eq(1) & h1_state.eq(-1)) | (fresh_vote.eq(-1) & h1_state.eq(1)))

        reversal = family.eq("pressure_failed") | family.eq("defense_spring")
        block_long = candidate & fresh_vote.eq(1) & reversal & micro_veto_long
        block_short = candidate & fresh_vote.eq(-1) & reversal & micro_veto_short
        blocked = block_long | block_short
        veto_reason.loc[blocked] = "5m_effective_opposite_pressure"
        candidate &= ~blocked

        signal.loc[candidate] = fresh_vote.loc[candidate]
        chosen_scale.loc[candidate] = scale
        chosen_family.loc[candidate] = family.loc[candidate]

    frame = base[["open", "high", "low", "close"]].copy()
    frame["signal"] = signal
    frame["signal_scale"] = chosen_scale
    frame["signal_family"] = chosen_family
    frame["veto_reason"] = veto_reason
    for scale, a in aligned.items():
        frame[f"state_{scale}"] = pd.to_numeric(a["state"], errors="coerce").fillna(0).astype(np.int8)

    event_audit = pd.concat(raw_events, ignore_index=True) if raw_events else pd.DataFrame()
    return frame, event_audit


def simulate_cross_inventory(
    signal_frame: pd.DataFrame,
    *,
    account: AccountConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | bool]]:
    """Simulate linear-perpetual cross inventory with next-open execution.

    Important invariant: a LONG vote can only execute a non-negative notional
    delta; a SHORT vote can only execute a non-positive delta.  If equity falls
    and existing exposure is already above the leverage cap, same-side votes are
    blocked rather than transformed into forced opposite trades.
    """
    cfg = account or AccountConfig()
    cfg.validate()
    required = {"open", "high", "low", "close", "signal"}
    missing = required.difference(signal_frame.columns)
    if missing:
        raise ValueError(f"signal_frame missing {sorted(missing)}")

    f = signal_frame.sort_index().copy()
    idx = pd.to_datetime(f.index)
    opens = pd.to_numeric(f["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(f["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(f["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype=float)
    signals = pd.to_numeric(f["signal"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    equity = float(cfg.initial_equity)
    qty = 0.0
    prev_close = np.nan
    liquidated = False
    blocked = 0
    clipped = 0
    total_fees = 0.0
    total_slippage = 0.0
    orders: list[dict[str, object]] = []
    path: list[dict[str, object]] = []
    slip_rate = cfg.slippage_bps_per_fill * 1e-4

    for i, ts in enumerate(idx):
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        if not all(np.isfinite(x) and x > 0 for x in (op, hi, lo, cl)):
            path.append({"time": ts, "equity": equity, "net_qty": qty, "net_notional": np.nan, "net_exposure_x": np.nan, "signal": int(signals[i])})
            continue

        # Position held from previous close through this bar open.
        if np.isfinite(prev_close):
            equity += qty * (op - prev_close)

        # Cross account can liquidate on the gap/open before a new vote executes.
        open_maintenance = abs(qty * op) * cfg.maintenance_margin_rate
        if equity <= max(0.0, open_maintenance):
            liquidated = True
            equity = 0.0
            path.append(
                {
                    "time": ts,
                    "equity": equity,
                    "net_qty": qty,
                    "net_notional": qty * op,
                    "net_exposure_x": np.inf,
                    "signal": int(signals[i]),
                    "executed_signal": 0,
                }
            )
            break

        executed_signal = 0
        delta_notional = 0.0
        source_i = i - 1
        if source_i >= 0 and signals[source_i] != 0 and equity > 0:
            sig = int(signals[source_i])
            vote_notional = cfg.vote_margin_fraction * equity * cfg.leverage
            current_notional = qty * op
            cap = cfg.leverage * equity
            desired_delta = sig * vote_notional
            proposed = current_notional + desired_delta

            # If the vote reduces absolute exposure, allow it fully.  If it adds
            # risk beyond the cap, clip or block but NEVER reverse its sign.
            if abs(proposed) <= cap + 1e-9 or abs(proposed) < abs(current_notional) - 1e-9:
                delta_notional = desired_delta
            else:
                target = sig * cap
                permitted = target - current_notional
                if permitted * sig <= 1e-9:
                    permitted = 0.0
                if abs(permitted) < abs(desired_delta) - 1e-9 and abs(permitted) > 0:
                    clipped += 1
                delta_notional = permitted

            # Explicit sign invariant guards the R01 bug.
            if sig > 0:
                delta_notional = max(0.0, delta_notional)
            else:
                delta_notional = min(0.0, delta_notional)

            if abs(delta_notional) <= 1e-12:
                blocked += 1
            else:
                fill_price = op * (1.0 + np.sign(delta_notional) * slip_rate)
                delta_qty = delta_notional / fill_price
                fee = abs(delta_qty * fill_price) * cfg.fee_rate_per_fill
                slippage_cost = abs(delta_qty) * abs(fill_price - op)
                equity -= fee + slippage_cost
                total_fees += fee
                total_slippage += slippage_cost
                qty += delta_qty
                executed_signal = sig
                orders.append(
                    {
                        "source_signal_time": idx[source_i],
                        "execution_time": ts,
                        "signal": sig,
                        "scale": str(f.iloc[source_i].get("signal_scale", "")),
                        "family": str(f.iloc[source_i].get("signal_family", "")),
                        "market_open": op,
                        "fill_price": fill_price,
                        "delta_notional": delta_notional,
                        "delta_qty": delta_qty,
                        "fee": fee,
                        "equity_before_bar_close": equity,
                    }
                )

        # Conservative intrabar liquidation check after the open execution.
        # Long inventory is stressed at the bar low; short inventory at the high.
        adverse = lo if qty > 0 else hi if qty < 0 else op
        adverse_equity = equity + qty * (adverse - op)
        adverse_maintenance = abs(qty * adverse) * cfg.maintenance_margin_rate
        if adverse_equity <= max(0.0, adverse_maintenance):
            liquidated = True
            equity = 0.0
            notional = qty * adverse
            exposure = np.inf
        else:
            equity += qty * (cl - op)
            prev_close = cl
            notional = qty * cl
            abs_notional = abs(notional)
            exposure = abs_notional / max(equity, _EPS) if equity > 0 else np.inf

        path.append(
            {
                "time": ts,
                "equity": equity,
                "net_qty": qty,
                "net_notional": notional,
                "net_exposure_x": exposure,
                "signal": int(signals[i]),
                "executed_signal": executed_signal,
            }
        )
        if liquidated:
            # Keep a deterministic terminal row and stop; no imaginary trading
            # after account bankruptcy.
            break

    path_df = pd.DataFrame(path).set_index("time") if path else pd.DataFrame()
    orders_df = pd.DataFrame(orders)
    if path_df.empty:
        raise RuntimeError("empty simulation path")

    eq = pd.to_numeric(path_df["equity"], errors="coerce")
    peak = eq.cummax()
    dd = eq / peak.replace(0.0, np.nan) - 1.0
    years = max((path_df.index[-1] - path_df.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final_equity = float(eq.iloc[-1])
    total_return = final_equity / cfg.initial_equity - 1.0
    cagr = (final_equity / cfg.initial_equity) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
    summary: dict[str, float | int | bool] = {
        "initial_equity": cfg.initial_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": float(dd.min()) if dd.notna().any() else 0.0,
        "orders": int(len(orders_df)),
        "long_orders": int((orders_df.get("signal", pd.Series(dtype=int)) > 0).sum()) if len(orders_df) else 0,
        "short_orders": int((orders_df.get("signal", pd.Series(dtype=int)) < 0).sum()) if len(orders_df) else 0,
        "blocked_votes": int(blocked),
        "clipped_votes": int(clipped),
        "total_fees": float(total_fees),
        "total_slippage": float(total_slippage),
        "mean_abs_exposure_x": float(pd.to_numeric(path_df["net_exposure_x"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        "max_abs_exposure_x": float(pd.to_numeric(path_df["net_exposure_x"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()),
        "liquidated": bool(liquidated),
    }
    return path_df, orders_df, summary


def period_returns(path: pd.DataFrame, freq: str) -> pd.DataFrame:
    if path.empty:
        return pd.DataFrame()
    eq = pd.to_numeric(path["equity"], errors="coerce")
    sampled = eq.resample(freq).last().dropna()
    start = eq.resample(freq).first().reindex(sampled.index)
    ret = sampled / start - 1.0
    return pd.DataFrame({"start_equity": start, "end_equity": sampled, "return": ret})
