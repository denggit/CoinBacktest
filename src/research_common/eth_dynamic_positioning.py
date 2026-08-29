"""Price-only ETH dynamic positioning primitives.

This module intentionally studies a *position state* rather than entry/exit
setups.  Two independent sleeves (medium and slow) continuously estimate:

    trend state -> current location -> volatility-scaled desired exposure

Desired exposure is only reconsidered on a fixed 4-hour decision clock.  A
no-trade band and bounded position step suppress economically meaningless
micro-adjustments.  Sleeves remain separate all the way through accounting so
simultaneous long/short holdings are preserved and fees are charged on gross
sleeve turnover rather than on prematurely netted exposure.

No machine learning, parameter search, discretionary chart pattern, TP or SL is
used in V1.  The purpose is to test whether location-aware position management
can improve on the repository's prior trend+volatility continuous portfolio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 365.25 * 24.0


@dataclass(frozen=True)
class DynamicPositionConfig:
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    trade_start: str = "2023-01-01 00:00:00"
    trade_end: str = "2026-06-30 23:59:59"

    # Frozen horizons; they are state measurements, not separate trade setups.
    medium_trend_hours: tuple[int, int, int] = (24, 72, 168)       # 1d / 3d / 7d
    slow_trend_hours: tuple[int, int, int] = (168, 336, 720)       # 7d / 14d / 30d
    medium_anchor_hours: int = 72
    slow_anchor_hours: int = 240
    medium_range_hours: int = 168
    slow_range_hours: int = 720
    volatility_hours: int = 168

    # Continuous sizing.
    target_annual_volatility: float = 0.60
    sleeve_notional_cap: float = 1.0
    gross_notional_cap: float = 2.0
    net_notional_cap: float = 1.5
    location_strength: float = 0.25

    # Position management.  Forecast/state may update hourly, but positions may
    # only be reconsidered once per four hours and only if the gap is meaningful.
    decision_hours: int = 4
    no_trade_band: float = 0.20
    max_step_per_decision: float = 0.50
    execution_delay_hours: int = 0  # extra delay beyond mandatory next-hour open

    # Economic assumptions.  Default fee is the user's 0.11% round-trip rule.
    fee_rate_per_side: float = 0.00055
    slippage_rate_per_side: float = 0.00010
    fallback_annual_carry_drag: float = 0.0

    def validate(self) -> None:
        for name, values in (("medium_trend_hours", self.medium_trend_hours), ("slow_trend_hours", self.slow_trend_hours)):
            if len(values) != 3 or tuple(sorted(values)) != values or min(values) <= 0:
                raise ValueError(f"{name} must contain three increasing positive horizons")
        if min(self.medium_anchor_hours, self.slow_anchor_hours, self.volatility_hours) <= 1:
            raise ValueError("anchor/volatility horizons must exceed one hour")
        if not 0 < self.target_annual_volatility < 3:
            raise ValueError("target_annual_volatility must be positive and bounded")
        if not 0 < self.sleeve_notional_cap <= self.gross_notional_cap:
            raise ValueError("invalid sleeve/gross notional cap")
        if not 0 < self.net_notional_cap <= self.gross_notional_cap:
            raise ValueError("invalid net notional cap")
        if not 0 <= self.location_strength <= 1:
            raise ValueError("location_strength must be in [0, 1]")
        if self.decision_hours <= 0 or self.execution_delay_hours < 0:
            raise ValueError("invalid decision/execution delay")
        if self.no_trade_band < 0 or self.max_step_per_decision <= 0:
            raise ValueError("invalid execution controls")
        if min(self.fee_rate_per_side, self.slippage_rate_per_side, self.fallback_annual_carry_drag) < 0:
            raise ValueError("cost assumptions must be non-negative")
        if pd.Timestamp(self.trade_start) < pd.Timestamp(self.warmup_start):
            raise ValueError("trade_start cannot precede warmup_start")
        if pd.Timestamp(self.trade_end) <= pd.Timestamp(self.trade_start):
            raise ValueError("trade_end must be after trade_start")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["medium_trend_hours"] = list(self.medium_trend_hours)
        payload["slow_trend_hours"] = list(self.slow_trend_hours)
        return payload


def validate_hourly_ohlcv(bars: pd.DataFrame) -> dict[str, object]:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars index must be DatetimeIndex")
    if bars.empty:
        raise ValueError("bars are empty")
    numeric = bars[list(sorted(required))].apply(pd.to_numeric, errors="coerce")
    duplicate = int(bars.index.duplicated().sum())
    non_monotonic = not bars.index.is_monotonic_increasing
    null_cells = int(numeric.isna().sum().sum())
    invalid = int(
        (
            (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
            | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ).sum()
    )
    expected = pd.date_range(bars.index.min(), bars.index.max(), freq="1h")
    missing_ts = int(len(expected.difference(bars.index)))
    return {
        "rows": int(len(bars)),
        "start": str(bars.index.min()),
        "end": str(bars.index.max()),
        "duplicate_timestamps": duplicate,
        "non_monotonic_index": bool(non_monotonic),
        "null_numeric_cells": null_cells,
        "invalid_ohlc_rows": invalid,
        "missing_timestamps": missing_ts,
        "missing_timestamp_rate": float(missing_ts / len(expected)) if len(expected) else np.nan,
        "ready": bool(not any((duplicate, non_monotonic, null_cells, invalid))),
    }


def _ewm_anchor(close: pd.Series, hours: int) -> pd.Series:
    # span is chosen directly from clock hours; current bar close is known at
    # signal time, and execution is delayed to the next hourly open.
    return close.ewm(span=hours, adjust=False, min_periods=hours).mean()


def _normalised_trend(log_close: pd.Series, hourly_vol: pd.Series, horizon: int) -> pd.Series:
    denom = hourly_vol * np.sqrt(float(horizon))
    z = log_close.diff(horizon) / denom.replace(0.0, np.nan)
    # Smoothly bound extreme trend observations; no threshold-based event gate.
    return np.tanh(z / 2.0)


def _range_location(bars: pd.DataFrame, hours: int) -> pd.Series:
    hi = bars["high"].rolling(hours, min_periods=hours).max()
    lo = bars["low"].rolling(hours, min_periods=hours).min()
    width = (hi - lo).replace(0.0, np.nan)
    return (2.0 * (bars["close"] - lo) / width - 1.0).clip(-1.0, 1.0)


def _location_multiplier(
    *,
    trend_score: pd.Series,
    log_close: pd.Series,
    anchor: pd.Series,
    hourly_vol: pd.Series,
    anchor_hours: int,
    strength: float,
) -> tuple[pd.Series, pd.Series]:
    anchor_extension_z = np.log(np.exp(log_close) / anchor) / (
        hourly_vol * np.sqrt(float(anchor_hours))
    ).replace(0.0, np.nan)
    extension_score = np.tanh(anchor_extension_z)
    # Positive aligned_extension means price is extended in the current trend
    # direction; V1 reduces size there.  Negative means a pullback while the
    # broader trend remains intact; V1 allows modestly more size.  This changes
    # *position size only* and never flips direction by itself.
    aligned_extension = np.sign(trend_score) * extension_score
    multiplier = (1.0 - strength * aligned_extension).clip(0.50, 1.50)
    return multiplier, extension_score


def build_state_frame(bars: pd.DataFrame, cfg: DynamicPositionConfig) -> pd.DataFrame:
    """Build causal close-known state and desired sleeve exposures.

    Row timestamp is the *start* of the hourly candle.  ``available_time`` is
    therefore timestamp + 1 hour.  Desired exposure from a row can only execute
    at or after that next hourly open.
    """
    cfg.validate()
    ordered = bars.sort_index(kind="stable").copy()
    ordered = ordered[~ordered.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
    ordered = ordered.dropna(subset=["open", "high", "low", "close", "volume"])

    close = ordered["close"]
    log_close = np.log(close)
    hourly_return = log_close.diff()
    hourly_vol = hourly_return.rolling(cfg.volatility_hours, min_periods=cfg.volatility_hours).std(ddof=0)
    annual_vol = hourly_vol * np.sqrt(HOURS_PER_YEAR)

    medium_components = pd.concat(
        [
            _normalised_trend(log_close, hourly_vol, horizon).rename(f"medium_trend_{horizon}h")
            for horizon in cfg.medium_trend_hours
        ],
        axis=1,
    )
    slow_components = pd.concat(
        [
            _normalised_trend(log_close, hourly_vol, horizon).rename(f"slow_trend_{horizon}h")
            for horizon in cfg.slow_trend_hours
        ],
        axis=1,
    )
    medium_trend = medium_components.mean(axis=1, skipna=False).clip(-1.0, 1.0)
    slow_trend = slow_components.mean(axis=1, skipna=False).clip(-1.0, 1.0)

    medium_anchor = _ewm_anchor(close, cfg.medium_anchor_hours)
    slow_anchor = _ewm_anchor(close, cfg.slow_anchor_hours)
    medium_loc_mult, medium_extension = _location_multiplier(
        trend_score=medium_trend,
        log_close=log_close,
        anchor=medium_anchor,
        hourly_vol=hourly_vol,
        anchor_hours=cfg.medium_anchor_hours,
        strength=cfg.location_strength,
    )
    slow_loc_mult, slow_extension = _location_multiplier(
        trend_score=slow_trend,
        log_close=log_close,
        anchor=slow_anchor,
        hourly_vol=hourly_vol,
        anchor_hours=cfg.slow_anchor_hours,
        strength=cfg.location_strength,
    )

    risk_scalar = (cfg.target_annual_volatility / annual_vol.replace(0.0, np.nan)).clip(lower=0.0, upper=1.5)
    medium_desired = (medium_trend * medium_loc_mult * risk_scalar).clip(
        -cfg.sleeve_notional_cap, cfg.sleeve_notional_cap
    )
    slow_desired = (slow_trend * slow_loc_mult * risk_scalar).clip(
        -cfg.sleeve_notional_cap, cfg.sleeve_notional_cap
    )

    frame = ordered.copy()
    for column in medium_components.columns:
        frame[column] = medium_components[column]
    for column in slow_components.columns:
        frame[column] = slow_components[column]
    frame["hourly_vol"] = hourly_vol
    frame["annual_vol"] = annual_vol
    frame["medium_trend"] = medium_trend
    frame["slow_trend"] = slow_trend
    frame["medium_anchor"] = medium_anchor
    frame["slow_anchor"] = slow_anchor
    frame["medium_extension"] = medium_extension
    frame["slow_extension"] = slow_extension
    frame["medium_range_location"] = _range_location(ordered, cfg.medium_range_hours)
    frame["slow_range_location"] = _range_location(ordered, cfg.slow_range_hours)
    frame["medium_location_multiplier"] = medium_loc_mult
    frame["slow_location_multiplier"] = slow_loc_mult
    frame["risk_scalar"] = risk_scalar
    frame["medium_desired_close"] = medium_desired
    frame["slow_desired_close"] = slow_desired
    frame["available_time"] = frame.index + pd.Timedelta(hours=1)

    # Decide on a wall-clock 4H cadence using the time at which the just-closed
    # 1H bar truly becomes available.  This remains stable even if warmup starts
    # on another hour.
    available_hour = pd.DatetimeIndex(frame["available_time"]).hour
    frame["decision_close"] = (available_hour % cfg.decision_hours) == 0
    ready_cols = [
        "medium_trend",
        "slow_trend",
        "annual_vol",
        "medium_location_multiplier",
        "slow_location_multiplier",
        "medium_desired_close",
        "slow_desired_close",
    ]
    frame["state_ready"] = frame[ready_cols].notna().all(axis=1)
    return frame


def _apply_caps(targets: dict[str, float], cfg: DynamicPositionConfig) -> dict[str, float]:
    values = {name: float(np.clip(value, -cfg.sleeve_notional_cap, cfg.sleeve_notional_cap)) for name, value in targets.items()}
    gross = sum(abs(v) for v in values.values())
    if gross > cfg.gross_notional_cap and gross > 0:
        scale = cfg.gross_notional_cap / gross
        values = {k: v * scale for k, v in values.items()}
    net = sum(values.values())
    if abs(net) > cfg.net_notional_cap and abs(net) > 0:
        scale = cfg.net_notional_cap / abs(net)
        values = {k: v * scale for k, v in values.items()}
    return values


def _step_position(current: float, desired: float, *, band: float, max_step: float) -> float:
    gap = desired - current
    if abs(gap) < band:
        return float(current)
    step = float(np.clip(gap, -max_step, max_step))
    return float(current + step)


def prepare_execution_targets(state: pd.DataFrame, cfg: DynamicPositionConfig) -> pd.DataFrame:
    """Create next-open executable desired targets without doing account replay."""
    desired = state[["medium_desired_close", "slow_desired_close"]].copy()
    decision = state["decision_close"] & state["state_ready"]
    held = desired.where(decision, axis=0).ffill().fillna(0.0)
    # Mandatory one-hour next-open shift plus optional extra stress delay.
    shift_hours = 1 + int(cfg.execution_delay_hours)
    executable = held.shift(shift_hours).fillna(0.0)
    executable.columns = ["medium_raw_target", "slow_raw_target"]
    executable["execution_decision"] = decision.shift(shift_hours, fill_value=False).astype(bool)
    return executable


def _funding_interval_rate(funding: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if funding is None or funding.empty or "funding_rate" not in funding.columns:
        return 0.0
    # Funding at exactly ``end`` belongs to the position held during (start,end];
    # a target executed at ``end`` should not retroactively avoid that settlement.
    rates = pd.to_numeric(
        funding.loc[(funding.index > start) & (funding.index <= end), "funding_rate"],
        errors="coerce",
    ).dropna()
    return float(rates.sum()) if not rates.empty else 0.0


def simulate_dynamic_positioning(
    state: pd.DataFrame,
    cfg: DynamicPositionConfig,
    *,
    funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replay independent medium/slow sleeves on an hourly marked account."""
    cfg.validate()
    targets = prepare_execution_targets(state, cfg)
    start = pd.Timestamp(cfg.trade_start)
    end = pd.Timestamp(cfg.trade_end)
    bars = state.loc[(state.index >= start) & (state.index <= end)].copy()
    targets = targets.reindex(bars.index).fillna(0.0)
    if len(bars) < 2:
        return pd.DataFrame()

    current = {"medium": 0.0, "slow": 0.0}
    equity = 1.0
    peak = 1.0
    rows: list[dict[str, object]] = []
    total_cost_rate = cfg.fee_rate_per_side + cfg.slippage_rate_per_side

    for i in range(len(bars) - 1):
        ts = pd.Timestamp(bars.index[i])
        next_ts = pd.Timestamp(bars.index[i + 1])
        raw = {
            "medium": float(targets["medium_raw_target"].iloc[i]),
            "slow": float(targets["slow_raw_target"].iloc[i]),
        }
        execution_decision = bool(targets["execution_decision"].iloc[i])
        if execution_decision:
            stepped = {
                name: _step_position(
                    current[name], raw[name], band=cfg.no_trade_band, max_step=cfg.max_step_per_decision
                )
                for name in ("medium", "slow")
            }
            stepped = _apply_caps(stepped, cfg)
        else:
            stepped = dict(current)
        turnover_medium = abs(stepped["medium"] - current["medium"])
        turnover_slow = abs(stepped["slow"] - current["slow"])
        turnover = turnover_medium + turnover_slow
        trading_cost = turnover * total_cost_rate

        net_exposure = stepped["medium"] + stepped["slow"]
        gross_exposure = abs(stepped["medium"]) + abs(stepped["slow"])
        long_gross = max(stepped["medium"], 0.0) + max(stepped["slow"], 0.0)
        short_gross = max(-stepped["medium"], 0.0) + max(-stepped["slow"], 0.0)

        open_px = float(bars["open"].iloc[i])
        next_open_px = float(bars["open"].iloc[i + 1])
        price_return = next_open_px / open_px - 1.0
        gross_return = net_exposure * price_return

        funding_rate = _funding_interval_rate(funding, ts, next_ts)
        funding_return = -net_exposure * funding_rate
        carry_drag = gross_exposure * cfg.fallback_annual_carry_drag / HOURS_PER_YEAR
        net_return = gross_return + funding_return - trading_cost - carry_drag
        equity_before = equity
        equity *= max(0.0, 1.0 + net_return)
        peak = max(peak, equity)

        rows.append(
            {
                "timestamp": ts,
                "next_timestamp": next_ts,
                "open": open_px,
                "next_open": next_open_px,
                "medium_raw_target": raw["medium"],
                "slow_raw_target": raw["slow"],
                "execution_decision": execution_decision,
                "medium_position": stepped["medium"],
                "slow_position": stepped["slow"],
                "net_exposure": net_exposure,
                "gross_exposure": gross_exposure,
                "long_gross_exposure": long_gross,
                "short_gross_exposure": short_gross,
                "long_short_overlap": bool(long_gross > 1e-12 and short_gross > 1e-12),
                "turnover_medium": turnover_medium,
                "turnover_slow": turnover_slow,
                "turnover": turnover,
                "price_return": price_return,
                "gross_return": gross_return,
                "funding_rate": funding_rate,
                "funding_return": funding_return,
                "trading_cost": trading_cost,
                "carry_drag": carry_drag,
                "net_return": net_return,
                "equity_before": equity_before,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "medium_trend": float(bars["medium_trend"].iloc[i]) if np.isfinite(bars["medium_trend"].iloc[i]) else np.nan,
                "slow_trend": float(bars["slow_trend"].iloc[i]) if np.isfinite(bars["slow_trend"].iloc[i]) else np.nan,
                "medium_extension": float(bars["medium_extension"].iloc[i]) if np.isfinite(bars["medium_extension"].iloc[i]) else np.nan,
                "slow_extension": float(bars["slow_extension"].iloc[i]) if np.isfinite(bars["slow_extension"].iloc[i]) else np.nan,
                "annual_vol": float(bars["annual_vol"].iloc[i]) if np.isfinite(bars["annual_vol"].iloc[i]) else np.nan,
            }
        )
        current = stepped

    return pd.DataFrame(rows).set_index("timestamp")


def _profit_factor(returns: pd.Series) -> float:
    x = pd.to_numeric(returns, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)


def _max_consecutive_loss_days(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    best = cur = 0
    for value in daily:
        cur = cur + 1 if value < 0 else 0
        best = max(best, cur)
    return int(best)


def _max_flat_days(frame: pd.DataFrame, threshold: float = 0.10) -> float:
    if frame.empty:
        return 0.0
    flat = frame["gross_exposure"] < threshold
    best = cur = 0
    for value in flat.astype(bool):
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return float(best / 24.0)


def summarize_account(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    years = max((frame["next_timestamp"].iloc[-1] - frame.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final_equity = float(frame["equity"].iloc[-1])
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
    max_dd = float(frame["drawdown"].min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    monthly = (1.0 + frame["net_return"]).groupby(frame.index.to_period("M")).prod() - 1.0
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    adjustment = frame["turnover"] > 1e-12
    gross_positive = float(frame.loc[frame["gross_return"] > 0, "gross_return"].sum())
    cost_total = float(frame["trading_cost"].sum())
    return {
        "start": str(frame.index.min()),
        "end": str(frame["next_timestamp"].max()),
        "hours": int(len(frame)),
        "total_return": final_equity - 1.0,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "profit_factor_hourly": _profit_factor(frame["net_return"]),
        "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "positive_day_rate": float((daily > 0).mean()) if len(daily) else np.nan,
        "max_consecutive_loss_days": _max_consecutive_loss_days(frame),
        "max_flat_days_below_0_1x": _max_flat_days(frame),
        "mean_abs_net_exposure": float(frame["net_exposure"].abs().mean()),
        "max_abs_net_exposure": float(frame["net_exposure"].abs().max()),
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "max_gross_exposure": float(frame["gross_exposure"].max()),
        "long_short_overlap_rate": float(frame["long_short_overlap"].mean()),
        "annual_turnover": float(frame["turnover"].sum() / years),
        "position_adjustments": int(adjustment.sum()),
        "adjustments_per_day": float(adjustment.sum() / max(len(frame) / 24.0, 1e-12)),
        "total_trading_cost_return": cost_total,
        "total_funding_return": float(frame["funding_return"].sum()),
        "total_fallback_carry_drag": float(frame["carry_drag"].sum()),
        "trading_cost_share_of_positive_gross": float(cost_total / gross_positive) if gross_positive > 0 else np.nan,
    }


def period_summary(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()
    for period, group in frame.groupby(frame.index.to_period(frequency)):
        ret = (1.0 + group["net_return"]).cumprod()
        dd = ret / ret.cummax() - 1.0
        rows.append(
            {
                "period": str(period),
                "return": float(ret.iloc[-1] - 1.0),
                "max_drawdown": float(dd.min()),
                "profit_factor_hourly": _profit_factor(group["net_return"]),
                "mean_abs_net_exposure": float(group["net_exposure"].abs().mean()),
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
                "turnover": float(group["turnover"].sum()),
                "trading_cost": float(group["trading_cost"].sum()),
                "funding_return": float(group["funding_return"].sum()),
                "position_adjustments": int((group["turnover"] > 1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def extract_sleeve_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()
    for sleeve in ("medium", "slow"):
        col = f"{sleeve}_position"
        sign = np.sign(frame[col]).astype(int)
        change = sign.ne(sign.shift(fill_value=0))
        episode_id = change.cumsum()
        active = sign.ne(0)
        for _, group in frame.loc[active].groupby(episode_id.loc[active]):
            side = "LONG" if group[col].iloc[0] > 0 else "SHORT"
            rows.append(
                {
                    "sleeve": sleeve,
                    "side": side,
                    "start": str(group.index.min()),
                    "end": str(group["next_timestamp"].max()),
                    "hours": int(len(group)),
                    "days": float(len(group) / 24.0),
                    "mean_abs_exposure": float(group[col].abs().mean()),
                    "max_abs_exposure": float(group[col].abs().max()),
                    "gross_price_contribution": float((group[col] * group["price_return"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def scenario_configs(base: DynamicPositionConfig) -> tuple[tuple[str, DynamicPositionConfig], ...]:
    """Pre-specified diagnostics; never select the best row as a tuned model."""
    return (
        ("base_location", base),
        ("trend_only_no_location", replace(base, location_strength=0.0)),
        ("no_trade_band_off", replace(base, no_trade_band=0.0, max_step_per_decision=10.0)),
        (
            "cost_2x",
            replace(
                base,
                fee_rate_per_side=base.fee_rate_per_side * 2.0,
                slippage_rate_per_side=base.slippage_rate_per_side * 2.0,
            ),
        ),
        (
            "cost_3x",
            replace(
                base,
                fee_rate_per_side=base.fee_rate_per_side * 3.0,
                slippage_rate_per_side=base.slippage_rate_per_side * 3.0,
            ),
        ),
        ("delay_plus_4h", replace(base, execution_delay_hours=base.execution_delay_hours + 4)),
        ("carry_stress_5pct", replace(base, fallback_annual_carry_drag=0.05)),
        ("carry_stress_10pct", replace(base, fallback_annual_carry_drag=0.10)),
    )


def top_day_removal(frame: pd.DataFrame, count: int = 10) -> dict[str, object]:
    if frame.empty:
        return {"removed_best_days": count, "remaining_total_return": np.nan}
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    top = daily.nlargest(min(count, len(daily)))
    remaining = daily.drop(top.index)
    return {
        "removed_best_days": int(count),
        "original_total_return": float((1.0 + daily).prod() - 1.0),
        "remaining_total_return": float((1.0 + remaining).prod() - 1.0),
        "largest_removed_day": float(top.iloc[0]) if len(top) else np.nan,
        "sum_removed_days": float(top.sum()) if len(top) else 0.0,
    }


def live_candidate_verdict(summary: dict[str, object], yearly: pd.DataFrame, *, funding_complete: bool) -> dict[str, object]:
    if not summary:
        return {"pass": False, "reason": "empty_result"}
    positive_years = int((pd.to_numeric(yearly.get("return", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not yearly.empty else 0
    checks = {
        "cagr_gt_abs_mdd": float(summary["cagr"]) > abs(float(summary["max_drawdown"])),
        "calmar_ge_1": float(summary["calmar"]) >= 1.0 if np.isfinite(float(summary["calmar"])) else False,
        "positive_years_ge_3": positive_years >= 3,
        "funding_coverage_complete": bool(funding_complete),
        "cost_share_lt_35pct": (
            float(summary["trading_cost_share_of_positive_gross"]) < 0.35
            if np.isfinite(float(summary["trading_cost_share_of_positive_gross"]))
            else False
        ),
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "positive_years": positive_years}


__all__ = [
    "DynamicPositionConfig",
    "build_state_frame",
    "extract_sleeve_episodes",
    "live_candidate_verdict",
    "period_summary",
    "prepare_execution_targets",
    "scenario_configs",
    "simulate_dynamic_positioning",
    "summarize_account",
    "top_day_removal",
    "validate_hourly_ohlcv",
]
