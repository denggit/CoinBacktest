#!/usr/bin/env python
"""Screen pre-declared PA mechanisms without selecting parameter optima.

This is a mechanism comparison, not a parameter grid.  All variants use the
same costs, volatility sizing, next-open execution, and risk caps.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio.ict_pa_model import (  # noqa: E402
    BARS_PER_DAY,
    IctPaConfig,
    build_open_positions,
    confirmed_pivots,
    resample_ohlcv,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402


OUT = Path(__file__).resolve().parent / "ict_pa_v1" / "results"


def load_bars() -> pd.DataFrame:
    cache = Path(__file__).resolve().parent / "bars_15m.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    minute = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data()
    minute = minute.loc["2020-01-01":"2026-08-15 23:59:59", ["open", "high", "low", "close", "volume"]]
    return resample_ohlcv(minute, "15min")


def structure_state(bars: pd.DataFrame, frequency: str, known_delay: str) -> pd.Series:
    context = resample_ohlcv(bars, frequency)
    pivots = confirmed_pivots(context, 2, 2)
    high = pivots["confirmed_high"].shift(1)
    low = pivots["confirmed_low"].shift(1)
    long_event = (context["close"] > high) & (context["close"].shift(1) <= high.shift(1))
    short_event = (context["close"] < low) & (context["close"].shift(1) >= low.shift(1))
    values: list[float] = []
    current = 0.0
    for go_long, go_short in zip(long_event.fillna(False), short_event.fillna(False)):
        if bool(go_long) and not bool(go_short):
            current = 1.0
        elif bool(go_short) and not bool(go_long):
            current = -1.0
        values.append(current)
    known = pd.Series(values, index=context.index + pd.Timedelta(known_delay))
    return known.reindex(bars.index, method="ffill").fillna(0.0)


def volatility_size(bars: pd.DataFrame, cfg: IctPaConfig) -> pd.Series:
    daily = resample_ohlcv(bars, "1D")
    vol = np.log(daily["close"]).diff().shift(1).rolling(
        cfg.core_volatility_days, min_periods=cfg.core_volatility_days
    ).std(ddof=0) * np.sqrt(365)
    size = (cfg.core_target_volatility / vol.replace(0.0, np.nan)).clip(upper=cfg.core_notional_cap)
    size.index = size.index + pd.Timedelta(days=1)
    return size.reindex(bars.index, method="ffill").fillna(0.0)


def replay(bars: pd.DataFrame, positions: pd.DataFrame, cfg: IctPaConfig) -> tuple[dict[str, object], dict[int, float]]:
    # Callers pass positions already scheduled for the current open.
    raw = positions.fillna(0.0)
    gross = raw.abs().sum(axis=1)
    scale = (cfg.gross_notional_cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    position = raw.mul(scale, axis=0)
    gross = position.abs().sum(axis=1)
    turnover = position.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(position.iloc[0].abs().sum())
    price_return = bars["open"].shift(-1) / bars["open"] - 1.0
    ret = position.sum(axis=1) * price_return - turnover * cfg.one_way_cost - gross * cfg.annual_carry_drag / (365 * BARS_PER_DAY)
    mask = (ret.index >= pd.Timestamp(cfg.start)) & (ret.index <= pd.Timestamp(cfg.end)) & price_return.notna()
    ret = ret.loc[mask]
    gross = gross.loc[mask]
    equity = (1.0 + ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    yearly = (1.0 + ret).groupby(ret.index.year).prod() - 1.0
    elapsed = max((ret.index[-1] - ret.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final = float(equity.iloc[-1])
    summary = {
        "total_return": final - 1.0,
        "cagr": final ** (1 / elapsed) - 1.0,
        "max_drawdown": float(drawdown.min()),
        "positive_years": int((yearly > 0).sum()),
        "worst_year": float(yearly.min()),
        "mean_gross_exposure": float(gross.mean()),
        "turnover": float(turnover.loc[mask].sum()),
    }
    return summary, {int(year): float(value) for year, value in yearly.items()}


def main() -> int:
    # Unfiltered tactical features are required so the screen can apply each
    # deployment rule independently instead of inheriting the final choice.
    cfg = replace(IctPaConfig(), core_mode="daily", tactical_mode="independent")
    bars = load_bars()
    feature = build_open_positions(bars, cfg)
    daily = structure_state(bars, "1D", "1D")
    # Monday-to-Monday is the pre-declared crypto weekly candle convention.
    weekly = structure_state(bars, "W-MON", "7D")
    size = volatility_size(bars, cfg)
    core_variants = {
        "daily_structure": daily * size,
        "weekly_structure": weekly * size,
        "daily_weekly_consensus": daily.where(daily == weekly, 0.0) * size,
    }
    tactical_variants = {
        "no_tactical": (feature["swing_long_position"] * 0.0, feature["swing_short_position"] * 0.0),
        "both_sides": (feature["swing_long_position"], feature["swing_short_position"]),
        "counter_structure_hedge": (
            feature["swing_long_position"].where(daily < 0, 0.0),
            feature["swing_short_position"].where(daily > 0, 0.0),
        ),
    }
    rows: list[dict[str, object]] = []
    for core_name, core_position in core_variants.items():
        for tactical_name, (long_position, short_position) in tactical_variants.items():
            positions = pd.DataFrame(
                {
                    "core": core_position.shift(1 + cfg.execution_delay_bars).fillna(0.0),
                    "swing_long": long_position,
                    "swing_short": short_position,
                },
                index=bars.index,
            )
            summary, years = replay(bars, positions, cfg)
            rows.append(
                {
                    "core_mechanism": core_name,
                    "tactical_mechanism": tactical_name,
                    **summary,
                    **{f"return_{year}": value for year, value in years.items()},
                }
            )
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "frozen_mechanism_screen.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
