#!/usr/bin/env python
"""Research a stable, low-turnover ETH price-action portfolio.

This is a mechanism screen, not a parameter optimiser.  The mechanisms and
lookbacks are fixed before evaluation and are combined with equal risk.  Every
daily signal is available only at the following natural-day open; every 4H
signal is available only after that 4H candle closes and trades at the next
15-minute open.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio.ict_pa_model import resample_ohlcv
from src.data_feed.okx_loader import OKXDataLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2026-08-15 23:59:59")
WARMUP = pd.Timestamp("2020-01-01")
ONE_WAY_COST = 0.0005
RESULTS = Path(__file__).resolve().parent / "ict_pa_v2" / "results"


def _state_from_events(long_event: pd.Series, short_event: pd.Series) -> pd.Series:
    out = np.zeros(len(long_event), dtype=float)
    state = 0.0
    for i, (go_long, go_short) in enumerate(zip(long_event.fillna(False), short_event.fillna(False))):
        if bool(go_long) and not bool(go_short):
            state = 1.0
        elif bool(go_short) and not bool(go_long):
            state = -1.0
        out[i] = state
    return pd.Series(out, index=long_event.index)


def _ewm(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _donchian_state(close: pd.Series, high: pd.Series, low: pd.Series, entry: int, exit_: int) -> pd.Series:
    upper = high.shift(1).rolling(entry, min_periods=entry).max()
    lower = low.shift(1).rolling(entry, min_periods=entry).min()
    exit_low = low.shift(1).rolling(exit_, min_periods=exit_).min()
    exit_high = high.shift(1).rolling(exit_, min_periods=exit_).max()
    state = np.zeros(len(close), dtype=float)
    current = 0.0
    for i in range(len(close)):
        px = float(close.iloc[i])
        if current >= 0 and np.isfinite(lower.iloc[i]) and px < float(lower.iloc[i]):
            current = -1.0
        elif current <= 0 and np.isfinite(upper.iloc[i]) and px > float(upper.iloc[i]):
            current = 1.0
        elif current > 0 and np.isfinite(exit_low.iloc[i]) and px < float(exit_low.iloc[i]):
            current = 0.0
        elif current < 0 and np.isfinite(exit_high.iloc[i]) and px > float(exit_high.iloc[i]):
            current = 0.0
        state[i] = current
    return pd.Series(state, index=close.index)


def build_daily_core(bars: pd.DataFrame) -> pd.DataFrame:
    daily = resample_ohlcv(bars, "1D")
    close, high, low = daily["close"], daily["high"], daily["low"]

    # Three independent and widely used price-action descriptions.  Adjacent
    # horizons are averaged; no historical winner is selected.
    momentum = pd.concat(
        [np.sign(np.log(close).diff(days)) for days in (21, 63, 126, 252)], axis=1
    ).mean(axis=1)
    ma_trend = pd.concat(
        [np.sign(_ewm(close, fast) - _ewm(close, slow)) for fast, slow in ((16, 64), (32, 128), (64, 256))],
        axis=1,
    ).mean(axis=1)
    channel = pd.concat(
        [_donchian_state(close, high, low, entry, entry // 2) for entry in (60, 120, 240)], axis=1
    ).mean(axis=1)

    # Equal mechanism weight prevents the full sample from choosing a family.
    forecast = pd.concat([momentum, ma_trend, channel], axis=1).mean(axis=1).clip(-1.0, 1.0)
    # The continuity anchor is only used when independent mechanisms cancel to
    # exactly zero.  It uses the slowest predeclared momentum direction at 5%
    # of the normal forecast, so natural-day flatness is not manufactured by a
    # transient vote tie while its risk contribution remains negligible.
    continuity = np.sign(np.log(close).diff(252)) * 0.05
    forecast = forecast.where(forecast.abs() > 1e-12, continuity).fillna(0.0)
    ret = np.log(close).diff()
    vol30 = ret.rolling(30, min_periods=30).std(ddof=0) * np.sqrt(365.25)
    vol90 = ret.rolling(90, min_periods=90).std(ddof=0) * np.sqrt(365.25)
    robust_vol = pd.concat([vol30, vol90], axis=1).max(axis=1).clip(lower=0.25)
    # A 15% risk target and 0.65 notional ceiling leave risk capacity for
    # tactical hedge sleeves while keeping gross exposure below 1x.
    size = (0.15 / robust_vol).clip(upper=0.65).fillna(0.0)
    desired = forecast * size

    # Drawdown-independent volatility brake: only observable price volatility
    # is used.  This is frozen ex ante and does not inspect strategy PnL.
    stress = (vol30 / vol90.replace(0.0, np.nan)).fillna(1.0)
    brake = pd.Series(np.where(stress > 1.5, 0.50, np.where(stress > 1.2, 0.75, 1.0)), index=daily.index)
    desired *= brake

    out = pd.DataFrame(
        {
            # The shifted availability index must be positional.  Timestamp
            # alignment would map day D+1 values to the D+1 open and leak that
            # day's close into execution.
            "momentum": momentum.to_numpy(),
            "ma_trend": ma_trend.to_numpy(),
            "channel": channel.to_numpy(),
            "forecast": forecast.to_numpy(),
            "annual_vol": robust_vol.to_numpy(),
            "position": desired.to_numpy(),
        },
        index=daily.index + pd.Timedelta(days=1),
    )
    return out


def _hold_events(long_event: pd.Series, short_event: pd.Series, bars: int, exposure: float) -> tuple[pd.Series, pd.Series]:
    long_pos = np.zeros(len(long_event), dtype=float)
    short_pos = np.zeros(len(short_event), dtype=float)
    long_left = short_left = 0
    for i, (go_long, go_short) in enumerate(zip(long_event.fillna(False), short_event.fillna(False))):
        if bool(go_long):
            long_left = bars
        if bool(go_short):
            short_left = bars
        if long_left > 0:
            long_pos[i] = exposure
            long_left -= 1
        if short_left > 0:
            short_pos[i] = -exposure
            short_left -= 1
    return pd.Series(long_pos, index=long_event.index), pd.Series(short_pos, index=short_event.index)


def build_micro_sleeves(trade_15m: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    numeric = trade_15m.copy()
    four = numeric.resample("4h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        notional=("notional", "sum"), delta_notional=("delta_notional", "sum"),
        large_delta_notional=("large_delta_notional", "sum"), trades=("trades_count", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    candle_range = (four["high"] - four["low"]).replace(0.0, np.nan)
    close_location = (four["close"] - four["low"]) / candle_range
    delta_ratio = four["delta_notional"] / four["notional"].replace(0.0, np.nan)
    large_ratio = four["large_delta_notional"] / four["notional"].replace(0.0, np.nan)
    prior_low = four["low"].shift(1).rolling(30, min_periods=30).min()
    prior_high = four["high"].shift(1).rolling(30, min_periods=30).max()
    q20 = delta_ratio.shift(1).rolling(180, min_periods=60).quantile(0.20)
    q80 = delta_ratio.shift(1).rolling(180, min_periods=60).quantile(0.80)
    large_q20 = large_ratio.shift(1).rolling(180, min_periods=60).quantile(0.20)
    large_q80 = large_ratio.shift(1).rolling(180, min_periods=60).quantile(0.80)

    # Liquidity sweep plus aggressive-flow absorption.  Negative delta with a
    # high close after a downside sweep indicates selling failed to move price;
    # the short rule is the exact mirror.
    absorption_long = (four["low"] < prior_low) & (four["close"] > prior_low) & (close_location >= 0.65) & (delta_ratio <= q20) & (large_ratio <= large_q20)
    absorption_short = (four["high"] > prior_high) & (four["close"] < prior_high) & (close_location <= 0.35) & (delta_ratio >= q80) & (large_ratio >= large_q80)
    absorb_l, absorb_s = _hold_events(absorption_long, absorption_short, bars=3, exposure=0.12)

    # A broader order-flow absorption definition does not require an external
    # sweep: aggressive sellers failing to prevent a bullish, high-location
    # close (and its exact mirror) is the market-microstructure mechanism.
    flow_reversal_long = (delta_ratio <= q20) & (close_location >= 0.65) & (four["close"] > four["open"])
    flow_reversal_short = (delta_ratio >= q80) & (close_location <= 0.35) & (four["close"] < four["open"])
    flow_l, flow_s = _hold_events(flow_reversal_long, flow_reversal_short, bars=2, exposure=0.08)

    # Pure PA failed auction, independent of trade classification.  It is a
    # mirrored sweep/reclaim and receives the same modest tactical risk.
    failed_sweep_long = (four["low"] < prior_low) & (four["close"] > prior_low) & (close_location >= 0.65)
    failed_sweep_short = (four["high"] > prior_high) & (four["close"] < prior_high) & (close_location <= 0.35)
    failed_l, failed_s = _hold_events(failed_sweep_long, failed_sweep_short, bars=3, exposure=0.08)

    # Breakout continuation requires both PA expansion and taker-flow
    # confirmation.  It diversifies the reversal sleeve and holds for one day.
    true_range = pd.concat(
        [four["high"] - four["low"], (four["high"] - four["close"].shift(1)).abs(), (four["low"] - four["close"].shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = true_range.shift(1).rolling(42, min_periods=42).median()
    breakout_long = (four["close"] > prior_high) & (close_location >= 0.70) & (true_range >= 1.25 * atr) & (delta_ratio >= q80)
    breakout_short = (four["close"] < prior_low) & (close_location <= 0.30) & (true_range >= 1.25 * atr) & (delta_ratio <= q20)
    break_l, break_s = _hold_events(breakout_long, breakout_short, bars=6, exposure=0.10)

    available = pd.DataFrame(
        {
            # Use arrays deliberately.  Supplying timestamp-indexed Series
            # together with a shifted index would align by label and leak the
            # following 4H observation into the current availability time.
            "absorption_long": absorb_l.to_numpy(),
            "absorption_short": absorb_s.to_numpy(),
            "breakout_long": break_l.to_numpy(),
            "breakout_short": break_s.to_numpy(),
            "flow_reversal_long": flow_l.to_numpy(),
            "flow_reversal_short": flow_s.to_numpy(),
            "failed_sweep_long": failed_l.to_numpy(),
            "failed_sweep_short": failed_s.to_numpy(),
            "absorption_long_event": absorption_long.to_numpy(),
            "absorption_short_event": absorption_short.to_numpy(),
            "breakout_long_event": breakout_long.to_numpy(),
            "breakout_short_event": breakout_short.to_numpy(),
            "flow_reversal_long_event": flow_reversal_long.to_numpy(),
            "flow_reversal_short_event": flow_reversal_short.to_numpy(),
            "failed_sweep_long_event": failed_sweep_long.to_numpy(),
            "failed_sweep_short_event": failed_sweep_short.to_numpy(),
        },
        index=four.index + pd.Timedelta(hours=4),
    )
    # 4H candle [T,T+4H) first becomes known at T+4H.  Reindexing at that
    # timestamp therefore represents a fill at the next 15m open.
    aligned = available.reindex(target_index, method="ffill").fillna(0.0)
    # Never carry the last cached microstructure state past source coverage.
    # The final completed 4H state may only live until the next source bar.
    if len(available):
        aligned.loc[aligned.index >= available.index.max() + pd.Timedelta(hours=4), :] = 0.0
    return aligned


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    minute = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data()
    minute = minute.loc[WARMUP:END.floor("min"), ["open", "high", "low", "close", "volume"]].copy()
    bars = resample_ohlcv(minute, "15min")
    trade = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="15m").load_local_data(
        "2022-01-01", "2026-07-13"
    )
    return bars, trade


def build_positions(bars: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    core_daily = build_daily_core(bars)
    core = core_daily["position"].reindex(bars.index, method="ffill").fillna(0.0)
    micro = build_micro_sleeves(trade, bars.index)
    sleeve_columns = [
        "absorption_long", "absorption_short", "breakout_long", "breakout_short",
        "flow_reversal_long", "flow_reversal_short", "failed_sweep_long", "failed_sweep_short",
    ]
    pos = pd.concat([core.rename("core"), micro[sleeve_columns]], axis=1).fillna(0.0)
    gross = pos.abs().sum(axis=1)
    scale = (0.95 / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    return pos.mul(scale, axis=0)


def simulate(bars: pd.DataFrame, positions: pd.DataFrame, *, cost: float = ONE_WAY_COST, delay_bars: int = 0) -> pd.DataFrame:
    pos = positions.shift(delay_bars).fillna(0.0)
    pos = pos.loc[START:END]
    px = bars["open"].reindex(pos.index)
    next_px = bars["open"].shift(-1).reindex(pos.index)
    price_return = next_px / px - 1.0
    valid = price_return.notna()
    pos, price_return = pos.loc[valid], price_return.loc[valid]
    turnover = pos.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = pos.iloc[0].abs().sum()
    net_exposure = pos.sum(axis=1)
    gross = pos.abs().sum(axis=1)
    gross_return = net_exposure * price_return
    trading_cost = turnover * cost
    net_return = gross_return - trading_cost
    equity = (1.0 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.concat(
        [pos.add_prefix("position_"), price_return.rename("price_return"), turnover.rename("turnover"), gross.rename("gross_exposure"), net_exposure.rename("net_exposure"), gross_return.rename("gross_return"), trading_cost.rename("trading_cost"), net_return.rename("net_return"), equity.rename("equity"), drawdown.rename("drawdown")],
        axis=1,
    )


def _streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def metrics(frame: pd.DataFrame, name: str) -> dict[str, object]:
    daily_return = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    daily_gross = frame["gross_exposure"].groupby(frame.index.floor("D")).max()
    total = float(frame["equity"].iloc[-1] - 1.0)
    years = (frame.index[-1] - frame.index[0]).total_seconds() / (365.25 * 86400)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    max_dd = abs(float(frame["drawdown"].min()))
    return {
        "candidate": name,
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": cagr / max_dd if max_dd > 0 else np.nan,
        "max_consecutive_flat_days": _streak(daily_gross <= 1e-12),
        "max_consecutive_losing_days": _streak(daily_return < 0),
        "positive_month_rate": float(((1.0 + frame["net_return"]).groupby(frame.index.to_period("M")).prod() - 1.0 > 0).mean()),
        "annual_volatility": float(frame["net_return"].std(ddof=0) * np.sqrt(365.25 * 96)),
        "max_gross_exposure": float(frame["gross_exposure"].max()),
        "total_cost": float(frame["trading_cost"].sum()),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bars, trade = load_inputs()
    positions = build_positions(bars, trade)
    zero = positions * 0.0
    no_micro = {column: 0.0 for column in positions.columns if column != "core"}
    variants = {
        "equal_mechanism_core": positions.assign(**no_micro),
        "absorption_only": zero.assign(absorption_long=positions["absorption_long"], absorption_short=positions["absorption_short"]),
        "breakout_only": zero.assign(breakout_long=positions["breakout_long"], breakout_short=positions["breakout_short"]),
        "breakout_long_only": zero.assign(breakout_long=positions["breakout_long"]),
        "breakout_short_only": zero.assign(breakout_short=positions["breakout_short"]),
        "flow_reversal_only": zero.assign(flow_reversal_long=positions["flow_reversal_long"], flow_reversal_short=positions["flow_reversal_short"]),
        "failed_sweep_only": zero.assign(failed_sweep_long=positions["failed_sweep_long"], failed_sweep_short=positions["failed_sweep_short"]),
        "core_plus_absorption": positions.assign(**{column: 0.0 for column in positions if column not in {"core", "absorption_long", "absorption_short"}}),
        "core_plus_breakout": positions.assign(**{column: 0.0 for column in positions if column not in {"core", "breakout_long", "breakout_short"}}),
        "core_plus_flow_reversal": positions.assign(**{column: 0.0 for column in positions if column not in {"core", "flow_reversal_long", "flow_reversal_short"}}),
        "core_plus_failed_sweep": positions.assign(**{column: 0.0 for column in positions if column not in {"core", "failed_sweep_long", "failed_sweep_short"}}),
        "stable_pa_micro_portfolio": positions,
    }
    rows: list[dict[str, object]] = []
    selected: pd.DataFrame | None = None
    for name, candidate_positions in variants.items():
        frame = simulate(bars, candidate_positions)
        rows.append(metrics(frame, name))
        if name == "stable_pa_micro_portfolio":
            selected = frame
    assert selected is not None
    for name, cost, delay in (("base", ONE_WAY_COST, 0), ("double_cost", 2 * ONE_WAY_COST, 0), ("delay_15m", ONE_WAY_COST, 1)):
        stress = simulate(bars, positions, cost=cost, delay_bars=delay)
        row = metrics(stress, name)
        row["scenario"] = name
        pd.DataFrame([row]).to_csv(RESULTS / f"stress_{name}.csv", index=False)

    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "mechanism_screen.csv", index=False)
    daily = selected.groupby(selected.index.floor("D")).agg(
        equity=("equity", "last"), drawdown=("drawdown", "last"), max_gross_exposure=("gross_exposure", "max"),
        end_net_exposure=("net_exposure", "last"), turnover=("turnover", "sum")
    )
    daily["net_return"] = (1.0 + selected["net_return"]).groupby(selected.index.floor("D")).prod() - 1.0
    daily.to_csv(RESULTS / "daily_equity.csv")

    contribution_rows = []
    for sleeve in positions.columns:
        sleeve_pos = positions[[sleeve]].copy()
        sleeve_frame = simulate(bars, sleeve_pos)
        for year, group in sleeve_frame.groupby(sleeve_frame.index.year):
            contribution_rows.append(
                {
                    "sleeve": sleeve,
                    "period": str(year),
                    "gross_arithmetic_contribution": float(group["gross_return"].sum()),
                    "trading_cost": float(group["trading_cost"].sum()),
                    "compounded_return": float((1.0 + group["net_return"]).prod() - 1.0),
                    "active_bar_rate": float((group["gross_exposure"] > 0).mean()),
                    "mean_signed_exposure": float(group["net_exposure"].mean()),
                }
            )
    pd.DataFrame(contribution_rows).to_csv(RESULTS / "sleeve_contribution_by_year.csv", index=False)

    daily_return = (1.0 + selected["net_return"]).groupby(selected.index.floor("D")).prod() - 1.0
    top_days = daily_return.sort_values(ascending=False).head(20).rename("return").reset_index()
    top_days.columns = ["date", "return"]
    top_days.to_csv(RESULTS / "top_20_days.csv", index=False)
    removed_rows = []
    for count in (5, 10, 20):
        kept = daily_return.drop(daily_return.nlargest(count).index)
        removed_rows.append({"removed_best_days": count, "remaining_total_return": float((1.0 + kept).prod() - 1.0)})
    pd.DataFrame(removed_rows).to_csv(RESULTS / "top_day_removal.csv", index=False)
    yearly_rows = []
    for year, group in selected.groupby(selected.index.year):
        local = group.copy()
        local["equity"] = (1.0 + local["net_return"]).cumprod()
        local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
        yearly_rows.append(metrics(local, str(year)))
    pd.DataFrame(yearly_rows).to_csv(RESULTS / "yearly.csv", index=False)

    core = build_daily_core(bars)
    micro = build_micro_sleeves(trade, bars.index)
    event_cols = [column for column in micro if column.endswith("_event")]
    audit = {
        "price_start": str(bars.index.min()), "price_end": str(bars.index.max()), "price_rows_15m": int(len(bars)),
        "trade_start": str(trade.index.min()), "trade_end": str(trade.index.max()), "trade_rows_15m": int(len(trade)),
        "one_way_cost": ONE_WAY_COST, "round_trip_cost": 2 * ONE_WAY_COST,
        "exchange_leverage_cap": 15.0, "strategy_gross_cap": 0.95,
        "daily_signal_availability": "completed day D -> D+1 natural-day open",
        "micro_signal_availability": "completed 4H candle -> next 15m open",
        "selection_rule": "equal weight across all predeclared mechanisms; no winner selection",
        "micro_events": {column: int(pd.to_numeric(micro[column]).sum()) for column in event_cols},
        "core_rows": int(len(core)),
    }
    (RESULTS / "run_config.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(screen.to_string(index=False))
    print("\nYEARLY\n", pd.DataFrame(yearly_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
