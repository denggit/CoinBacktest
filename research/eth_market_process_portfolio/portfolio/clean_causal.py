"""Causal, low-complexity ETH cross-margin portfolio research utilities.

The module intentionally uses price-only, literature-style time-series momentum
with volatility targeting.  It contains no parameter optimiser.  Signals are
formed from completed 4H bars and become positions only at a later bar open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd


BARS_PER_DAY = 6
BARS_PER_YEAR = 365 * BARS_PER_DAY


@dataclass(frozen=True)
class PortfolioConfig:
    symbol: str = "ETH-USDT-SWAP"
    start: str = "2022-01-01 00:00:00"
    end: str = "2026-08-15 23:59:59"
    momentum_days: tuple[int, int, int] = (7, 30, 90)
    volatility_days: int = 30
    target_annual_volatility: float = 0.12
    strategy_notional_cap: float = 1.50
    exchange_leverage_cap: float = 15.0
    one_way_cost: float = 0.00050
    annual_carry_drag: float = 0.05
    rebalance_bars: int = BARS_PER_DAY
    execution_delay_bars: int = 0
    maintenance_margin_rate: float = 0.005
    drawdown_half_speed: float = -0.10
    drawdown_quarter_speed: float = -0.15

    def validate(self) -> None:
        if len(self.momentum_days) != 3 or tuple(sorted(self.momentum_days)) != self.momentum_days:
            raise ValueError("momentum_days must contain three increasing horizons")
        if min(self.momentum_days) <= 0 or self.volatility_days <= 1:
            raise ValueError("lookbacks must be positive")
        if not 0 < self.target_annual_volatility < 1:
            raise ValueError("target_annual_volatility must be in (0, 1)")
        if not 0 < self.strategy_notional_cap <= self.exchange_leverage_cap:
            raise ValueError("strategy notional cap must not exceed exchange leverage cap")
        if self.one_way_cost < 0 or self.annual_carry_drag < 0:
            raise ValueError("cost assumptions must be non-negative")
        if self.rebalance_bars <= 0 or self.execution_delay_bars < 0:
            raise ValueError("execution settings must be non-negative")
        if not 0 <= self.maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["momentum_days"] = list(self.momentum_days)
        return payload


def validate_ohlcv(bars: pd.DataFrame, *, expected_frequency: str) -> dict[str, object]:
    """Return compact data-quality evidence without altering the input."""
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must use a DatetimeIndex")
    numeric = bars[list(sorted(required))].apply(pd.to_numeric, errors="coerce")
    duplicate_count = int(bars.index.duplicated().sum())
    non_monotonic = not bars.index.is_monotonic_increasing
    null_cells = int(numeric.isna().sum().sum())
    invalid_ohlc = int(
        (
            (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
            | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ).sum()
    )
    negative_volume = int((numeric["volume"] < 0).sum())
    expected = pd.date_range(bars.index.min(), bars.index.max(), freq=expected_frequency)
    missing_timestamps = int(len(expected.difference(bars.index)))
    expected_count = int(len(expected))
    return {
        "rows": int(len(bars)),
        "start": str(bars.index.min()),
        "end": str(bars.index.max()),
        "duplicate_timestamps": duplicate_count,
        "non_monotonic_index": bool(non_monotonic),
        "null_numeric_cells": null_cells,
        "invalid_ohlc_rows": invalid_ohlc,
        "negative_volume_rows": negative_volume,
        "expected_timestamps": expected_count,
        "missing_timestamps": missing_timestamps,
        "missing_timestamp_rate": float(missing_timestamps / expected_count) if expected_count else np.nan,
        "ready": not any((duplicate_count, non_monotonic, null_cells, invalid_ohlc, negative_volume)),
    }


def resample_to_4h(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate left-closed local candles into completed 4H bars."""
    ordered = minute_bars.sort_index(kind="stable")
    ordered = ordered[~ordered.index.duplicated(keep="last")]
    out = ordered.resample("4h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_minutes=("close", "count"),
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def build_desired_exposure(bars: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    """Build close-known desired exposure before execution delay is applied."""
    cfg.validate()
    close = pd.to_numeric(bars["close"], errors="coerce")
    log_close = np.log(close)
    horizons = [days * BARS_PER_DAY for days in cfg.momentum_days]
    votes = pd.concat(
        [np.sign(log_close.diff(horizon)).rename(f"vote_{days}d") for days, horizon in zip(cfg.momentum_days, horizons)],
        axis=1,
    )
    direction_score = votes.mean(axis=1, skipna=False)
    bar_returns = log_close.diff()
    realised_vol = (
        bar_returns.rolling(cfg.volatility_days * BARS_PER_DAY, min_periods=cfg.volatility_days * BARS_PER_DAY)
        .std(ddof=0)
        * np.sqrt(BARS_PER_YEAR)
    )
    volatility_multiplier = (cfg.target_annual_volatility / realised_vol.replace(0, np.nan)).clip(
        upper=cfg.strategy_notional_cap
    )
    update_mask = pd.Series(np.arange(len(bars)) % cfg.rebalance_bars == 0, index=bars.index)
    sleeve_columns: dict[str, pd.Series] = {}
    for days, vote_column in zip(cfg.momentum_days, votes.columns):
        desired_sleeve = (
            votes[vote_column] * volatility_multiplier / len(cfg.momentum_days)
        ).where(direction_score.notna() & volatility_multiplier.notna())
        sleeve_columns[f"desired_{days}d_close"] = desired_sleeve
        sleeve_columns[f"rebalanced_{days}d_close"] = desired_sleeve.where(update_mask).ffill()
    desired = sum(sleeve_columns[f"desired_{days}d_close"] for days in cfg.momentum_days)
    rebalanced = sum(sleeve_columns[f"rebalanced_{days}d_close"] for days in cfg.momentum_days)
    return pd.DataFrame(
        {
            **{column: votes[column] for column in votes.columns},
            **sleeve_columns,
            "direction_score": direction_score,
            "realised_vol_annual": realised_vol,
            "volatility_multiplier": volatility_multiplier,
            "desired_exposure_close": desired,
            "rebalanced_exposure_close": rebalanced,
        },
        index=bars.index,
    )


def simulate_portfolio(bars: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    """Replay a single-net-position cross-margin account at 4H resolution.

    A signal from close ``t`` can first become exposure at open ``t+1``.  An
    execution-delay stress shifts it one or more additional opens.  Drawdown
    throttles use only equity known before the current open.
    """
    cfg.validate()
    features = build_desired_exposure(bars, cfg)
    delay = 1 + int(cfg.execution_delay_bars)
    sleeve_names = [f"{days}d" for days in cfg.momentum_days]
    scheduled = {
        name: features[f"rebalanced_{name}_close"].shift(delay).fillna(0.0)
        for name in sleeve_names
    }
    n = max(0, len(bars) - 1)
    records: list[dict[str, object]] = []
    equity = 1.0
    peak = 1.0
    previous_sleeves = {name: 0.0 for name in sleeve_names}
    for i in range(n):
        ts = bars.index[i]
        next_ts = bars.index[i + 1]
        drawdown_before = equity / peak - 1.0
        if drawdown_before <= cfg.drawdown_quarter_speed:
            risk_speed = 0.25
        elif drawdown_before <= cfg.drawdown_half_speed:
            risk_speed = 0.50
        else:
            risk_speed = 1.0
        raw_sleeves = {
            name: float(series.iloc[i]) if np.isfinite(series.iloc[i]) else 0.0
            for name, series in scheduled.items()
        }
        sleeve_positions = {name: value * risk_speed for name, value in raw_sleeves.items()}
        gross_exposure = float(sum(abs(value) for value in sleeve_positions.values()))
        if gross_exposure > cfg.strategy_notional_cap and gross_exposure > 0:
            scale = cfg.strategy_notional_cap / gross_exposure
            sleeve_positions = {name: value * scale for name, value in sleeve_positions.items()}
            gross_exposure = cfg.strategy_notional_cap
        net_position = float(sum(sleeve_positions.values()))
        long_gross = float(sum(max(value, 0.0) for value in sleeve_positions.values()))
        short_gross = float(sum(max(-value, 0.0) for value in sleeve_positions.values()))
        turnover = float(sum(abs(sleeve_positions[name] - previous_sleeves[name]) for name in sleeve_names))
        trading_cost = turnover * cfg.one_way_cost
        carry_cost = gross_exposure * cfg.annual_carry_drag / BARS_PER_YEAR
        open_price = float(bars["open"].iloc[i])
        next_open = float(bars["open"].iloc[i + 1])
        interval_return = next_open / open_price - 1.0
        gross_return = net_position * interval_return
        net_return = gross_return - trading_cost - carry_cost
        if net_position > 0:
            adverse_price_return = float(bars["low"].iloc[i]) / open_price - 1.0
        elif net_position < 0:
            adverse_price_return = -(float(bars["high"].iloc[i]) / open_price - 1.0)
        else:
            adverse_price_return = 0.0
        intrabar_equity_ratio = 1.0 - trading_cost - carry_cost + abs(net_position) * adverse_price_return
        maintenance_required = cfg.maintenance_margin_rate * gross_exposure
        maintenance_headroom = intrabar_equity_ratio - maintenance_required
        liquidated = bool(maintenance_headroom <= 0.0 or 1.0 + net_return <= 0.0)
        equity_before = equity
        equity *= max(0.0, 1.0 + net_return)
        peak = max(peak, equity)
        records.append(
            {
                "timestamp": ts,
                "next_timestamp": next_ts,
                "open": open_price,
                "next_open": next_open,
                "direction_score": features["direction_score"].iloc[i],
                "realised_vol_annual": features["realised_vol_annual"].iloc[i],
                **{f"raw_{name}_position": raw_sleeves[name] for name in sleeve_names},
                **{f"{name}_position": sleeve_positions[name] for name in sleeve_names},
                "raw_position": float(sum(raw_sleeves.values())),
                "risk_speed": risk_speed,
                "position": net_position,
                "net_exposure": net_position,
                "long_gross_exposure": long_gross,
                "short_gross_exposure": short_gross,
                "gross_exposure": gross_exposure,
                "long_short_overlap": bool(long_gross > 1e-12 and short_gross > 1e-12),
                "turnover": turnover,
                "price_return": interval_return,
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "carry_cost": carry_cost,
                "net_return": net_return,
                "equity_before": equity_before,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "intrabar_equity_ratio": intrabar_equity_ratio,
                "maintenance_required": maintenance_required,
                "maintenance_headroom": maintenance_headroom,
                "liquidated": liquidated,
            }
        )
        previous_sleeves = sleeve_positions
    return pd.DataFrame.from_records(records).set_index("timestamp") if records else pd.DataFrame()


def profit_factor(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)


def summarize_equity(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    returns = pd.to_numeric(frame["net_return"], errors="coerce").fillna(0.0)
    elapsed_years = max((frame.index[-1] - frame.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final_equity = float(frame["equity"].iloc[-1])
    annual_return = final_equity ** (1.0 / elapsed_years) - 1.0 if final_equity > 0 else -1.0
    annual_vol = float(returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR)) if returns.std(ddof=0) > 0 else np.nan
    max_drawdown = float(frame["drawdown"].min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    monthly = (1.0 + returns).groupby(frame.index.to_period("M")).prod() - 1.0
    active = frame["gross_exposure"] > 1e-12
    return {
        "start": str(frame.index.min()),
        "end": str(frame["next_timestamp"].max()),
        "bars": int(len(frame)),
        "total_return": final_equity - 1.0,
        "final_equity": final_equity,
        "cagr": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_zero_rf": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "profit_factor_4h": profit_factor(returns),
        "positive_month_rate": float((monthly > 0).mean()),
        "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()),
        "active_bar_rate": float(active.mean()),
        "long_bar_rate": float((frame["position"] > 0).mean()),
        "short_bar_rate": float((frame["position"] < 0).mean()),
        "mean_abs_exposure": float(frame["position"].abs().mean()),
        "max_abs_exposure": float(frame["position"].abs().max()),
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "max_gross_exposure": float(frame["gross_exposure"].max()),
        "long_short_overlap_rate": float(frame["long_short_overlap"].mean()),
        "annual_turnover": float(frame["turnover"].sum() / elapsed_years),
        "total_trading_cost": float(frame["trading_cost"].sum()),
        "total_carry_cost": float(frame["carry_cost"].sum()),
        "min_maintenance_headroom": float(frame["maintenance_headroom"].min()),
        "liquidation_events": int(frame["liquidated"].sum()),
    }


def period_summary(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, group in frame.groupby(frame.index.to_period(frequency)):
        returns = pd.to_numeric(group["net_return"], errors="coerce").fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "period": str(period),
                "return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(drawdown.min()),
                "profit_factor_4h": profit_factor(returns),
                "active_bar_rate": float((group["gross_exposure"] > 1e-12).mean()),
                "mean_abs_exposure": float(group["position"].abs().mean()),
                "max_abs_exposure": float(group["position"].abs().max()),
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
                "max_gross_exposure": float(group["gross_exposure"].max()),
                "long_short_overlap_rate": float(group["long_short_overlap"].mean()),
                "trading_cost": float(group["trading_cost"].sum()),
                "carry_cost": float(group["carry_cost"].sum()),
                "liquidation_events": int(group["liquidated"].sum()),
            }
        )
    return pd.DataFrame(rows)


def extract_position_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    sleeve_columns = [
        column for column in frame.columns
        if column.endswith("d_position") and not column.startswith("raw_")
    ]
    for sleeve_column in sleeve_columns:
        sign = np.sign(frame[sleeve_column]).astype(int)
        episode_id = (sign.ne(sign.shift(fill_value=0)) | sign.eq(0)).cumsum()
        for _, group in frame[sign.ne(0)].groupby(episode_id[sign.ne(0)]):
            side = "LONG" if group[sleeve_column].iloc[0] > 0 else "SHORT"
            contribution = group[sleeve_column] * group["price_return"]
            rows.append(
                {
                    "sleeve": sleeve_column.removesuffix("_position"),
                    "entry_time": str(group.index.min()),
                    "exit_time": str(group["next_timestamp"].max()),
                    "side": side,
                    "bars": int(len(group)),
                    "days": float(len(group) / BARS_PER_DAY),
                    "gross_price_contribution": float(contribution.sum()),
                    "max_abs_exposure": float(group[sleeve_column].abs().max()),
                    "min_maintenance_headroom": float(group["maintenance_headroom"].min()),
                }
            )
    return pd.DataFrame(rows)


def shock_survival_table(exposures: Iterable[float], cfg: PortfolioConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shock in (0.20, 0.35, 0.50):
        for exposure in exposures:
            absolute = abs(float(exposure))
            equity_ratio = 1.0 - absolute * shock
            maintenance = cfg.maintenance_margin_rate * absolute
            rows.append(
                {
                    "adverse_instantaneous_move": shock,
                    "absolute_exposure": absolute,
                    "equity_ratio_after_shock": equity_ratio,
                    "maintenance_required": maintenance,
                    "maintenance_headroom": equity_ratio - maintenance,
                    "survives_assumed_liquidation_rule": bool(equity_ratio > maintenance),
                }
            )
    return pd.DataFrame(rows)


def scenario_configs(base: PortfolioConfig) -> tuple[tuple[str, PortfolioConfig], ...]:
    return (
        ("base", base),
        ("cost_2x", replace(base, one_way_cost=base.one_way_cost * 2.0)),
        ("delay_1bar", replace(base, execution_delay_bars=base.execution_delay_bars + 1)),
        ("carry_2x", replace(base, annual_carry_drag=base.annual_carry_drag * 2.0)),
        (
            "no_drawdown_throttle",
            replace(base, drawdown_half_speed=-1.0, drawdown_quarter_speed=-1.0),
        ),
    )


__all__ = [
    "BARS_PER_DAY",
    "PortfolioConfig",
    "build_desired_exposure",
    "extract_position_episodes",
    "period_summary",
    "resample_to_4h",
    "scenario_configs",
    "shock_survival_table",
    "simulate_portfolio",
    "summarize_equity",
    "validate_ohlcv",
]
