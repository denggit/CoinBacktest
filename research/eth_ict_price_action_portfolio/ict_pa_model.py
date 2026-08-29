"""Causal ETH ICT + Price Action hedge-mode portfolio.

The module deliberately uses a small, frozen rule set:

* a daily confirmed-market-structure (BOS) core that can hold for months;
* 4H external-liquidity sweeps used only as macro context;
* independent 15m Price Action / market-structure-shift execution sleeves;
* account-level cross-margin accounting with both sides kept gross.

Every signal is formed from completed candles.  The earliest fill is the next
15m open.  No parameter optimiser or same-bar signal fill is implemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd


BARS_PER_DAY = 96
BARS_PER_YEAR = 365 * BARS_PER_DAY


@dataclass(frozen=True)
class IctPaConfig:
    start: str = "2022-01-01 00:00:00"
    end: str = "2026-08-15 23:59:59"
    one_way_cost: float = 0.00050
    annual_carry_drag: float = 0.03
    exchange_leverage_cap: float = 15.0
    gross_notional_cap: float = 1.00
    maintenance_margin_rate: float = 0.005
    core_target_volatility: float = 0.10
    core_notional_cap: float = 0.60
    core_volatility_days: int = 30
    core_mode: str = "daily_12m_blend"
    daily_pivot_left: int = 2
    daily_pivot_right: int = 2
    sweep_pivot_left: int = 3
    sweep_pivot_right: int = 3
    sweep_atr_bars: int = 14
    sweep_close_location: float = 0.60
    sweep_wick_fraction: float = 0.35
    displacement_body_fraction: float = 0.60
    displacement_atr_multiple: float = 0.80
    micro_pivot_left: int = 2
    micro_pivot_right: int = 2
    micro_atr_bars: int = 32
    context_valid_bars: int = 32
    require_volume_confirmation: bool = True
    swing_risk_budget: float = 0.0030
    swing_notional_cap: float = 0.25
    swing_reward_risk: float = 2.0
    swing_max_hold_bars: int = 288
    tactical_mode: str = "counter"
    stop_atr_buffer: float = 0.10
    execution_delay_bars: int = 0
    drawdown_half_speed: float = -0.10
    drawdown_quarter_speed: float = -0.15

    def validate(self) -> None:
        if self.one_way_cost < 0:
            raise ValueError("trading cost must be non-negative")
        if not 0 < self.gross_notional_cap <= self.exchange_leverage_cap:
            raise ValueError("gross notional cap must be positive and <= exchange cap")
        if not 0 < self.core_notional_cap <= self.gross_notional_cap:
            raise ValueError("core cap must fit inside gross cap")
        if not 0 < self.swing_notional_cap <= self.gross_notional_cap:
            raise ValueError("swing cap must fit inside gross cap")
        if min(
            self.daily_pivot_left,
            self.daily_pivot_right,
            self.sweep_pivot_left,
            self.sweep_pivot_right,
            self.micro_pivot_left,
            self.micro_pivot_right,
        ) < 1:
            raise ValueError("pivot confirmation requires positive left/right bars")
        if self.execution_delay_bars < 0 or self.swing_max_hold_bars < 1:
            raise ValueError("execution delay/holding period is invalid")
        if self.core_mode not in {"daily", "daily_12m_blend", "daily_weekly_consensus"}:
            raise ValueError("unsupported core_mode")
        if self.tactical_mode not in {"counter", "independent", "none"}:
            raise ValueError("unsupported tactical_mode")
        if not 0 <= self.maintenance_margin_rate < 1:
            raise ValueError("maintenance margin rate must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resample_ohlcv(bars: pd.DataFrame, frequency: str) -> pd.DataFrame:
    ordered = bars.sort_index(kind="stable")
    ordered = ordered[~ordered.index.duplicated(keep="last")]
    out = ordered.resample(frequency, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_bars=("close", "count"),
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def true_range(bars: pd.DataFrame) -> pd.Series:
    previous = bars["close"].shift(1)
    return pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous).abs(),
            (bars["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def confirmed_pivots(bars: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """Return pivot levels only from their close-time confirmation onward.

    A pivot at position ``p`` requires bars through ``p + right``.  Therefore
    it is shifted by ``right`` bars before being forward-filled.  A separate
    one-bar lag is applied by signal builders when the level is tested.
    """
    low = pd.to_numeric(bars["low"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    left_low = low.shift(1).rolling(left, min_periods=left).min()
    left_high = high.shift(1).rolling(left, min_periods=left).max()
    right_low = low.iloc[::-1].shift(1).rolling(right, min_periods=right).min().iloc[::-1]
    right_high = high.iloc[::-1].shift(1).rolling(right, min_periods=right).max().iloc[::-1]
    pivot_low = low.where((low < left_low) & (low <= right_low))
    pivot_high = high.where((high > left_high) & (high >= right_high))
    return pd.DataFrame(
        {
            "confirmed_low": pivot_low.shift(right).ffill(),
            "confirmed_high": pivot_high.shift(right).ffill(),
            "new_low_confirmation": pivot_low.shift(right).notna(),
            "new_high_confirmation": pivot_high.shift(right).notna(),
        },
        index=bars.index,
    )


def _persistent_state(long_event: pd.Series, short_event: pd.Series) -> pd.Series:
    state = np.zeros(len(long_event), dtype=float)
    current = 0.0
    for i, (go_long, go_short) in enumerate(zip(long_event.fillna(False), short_event.fillna(False))):
        if bool(go_long) and not bool(go_short):
            current = 1.0
        elif bool(go_short) and not bool(go_long):
            current = -1.0
        state[i] = current
    return pd.Series(state, index=long_event.index)


def build_daily_structure_core(bars_15m: pd.DataFrame, cfg: IctPaConfig) -> pd.DataFrame:
    daily = resample_ohlcv(bars_15m, "1D")
    pivots = confirmed_pivots(daily, cfg.daily_pivot_left, cfg.daily_pivot_right)
    prior_high = pivots["confirmed_high"].shift(1)
    prior_low = pivots["confirmed_low"].shift(1)
    bos_long = (daily["close"] > prior_high) & (daily["close"].shift(1) <= prior_high.shift(1))
    bos_short = (daily["close"] < prior_low) & (daily["close"].shift(1) >= prior_low.shift(1))
    daily_regime = _persistent_state(bos_long, bos_short)
    log_return = np.log(daily["close"]).diff()
    vol = log_return.shift(1).rolling(cfg.core_volatility_days, min_periods=cfg.core_volatility_days).std(ddof=0) * np.sqrt(365)
    size = (cfg.core_target_volatility / vol.replace(0.0, np.nan)).clip(upper=cfg.core_notional_cap)
    # A daily candle labelled D is only complete at D+1.
    daily_frame = pd.DataFrame(
        {
            "daily_regime": daily_regime.to_numpy(),
            "momentum_12m_regime": np.sign(np.log(daily["close"]).diff(365)).to_numpy(),
            "core_volatility": vol.to_numpy(),
            "core_size": size.to_numpy(),
            "daily_bos_long": bos_long.to_numpy(),
            "daily_bos_short": bos_short.to_numpy(),
        },
        index=daily.index + pd.Timedelta(days=1),
    ).reindex(bars_15m.index, method="ffill")

    # The weekly layer is not a second trade signal.  It is a slow external
    # structure veto: the long-hold core is active only when completed daily
    # and Monday-to-Monday weekly BOS states agree.  This reduces structural
    # whipsaw without optimising a numerical trend threshold.
    weekly = resample_ohlcv(bars_15m, "W-MON")
    weekly_pivots = confirmed_pivots(weekly, cfg.daily_pivot_left, cfg.daily_pivot_right)
    weekly_high = weekly_pivots["confirmed_high"].shift(1)
    weekly_low = weekly_pivots["confirmed_low"].shift(1)
    weekly_long = (weekly["close"] > weekly_high) & (weekly["close"].shift(1) <= weekly_high.shift(1))
    weekly_short = (weekly["close"] < weekly_low) & (weekly["close"].shift(1) >= weekly_low.shift(1))
    weekly_regime = _persistent_state(weekly_long, weekly_short)
    weekly_aligned = pd.Series(
        weekly_regime.to_numpy(), index=weekly.index + pd.Timedelta(days=7), name="weekly_regime"
    ).reindex(bars_15m.index, method="ffill")

    aligned = daily_frame.copy()
    aligned["weekly_regime"] = weekly_aligned
    aligned = aligned.fillna(
        {
            "daily_regime": 0.0,
            "momentum_12m_regime": 0.0,
            "weekly_regime": 0.0,
            "core_size": 0.0,
            "daily_bos_long": False,
            "daily_bos_short": False,
        }
    )
    aligned["consensus_regime"] = aligned["daily_regime"].where(
        aligned["daily_regime"] == aligned["weekly_regime"], 0.0
    )
    if cfg.core_mode == "daily":
        aligned["core_regime"] = aligned["daily_regime"]
    elif cfg.core_mode == "daily_12m_blend":
        slow = aligned["momentum_12m_regime"].where(
            aligned["momentum_12m_regime"].ne(0.0), aligned["daily_regime"]
        )
        aligned["core_regime"] = (2.0 * aligned["daily_regime"] + slow) / 3.0
    else:
        aligned["core_regime"] = aligned["consensus_regime"]
    aligned["core_desired_close"] = aligned["core_regime"] * aligned["core_size"].fillna(0.0)
    return aligned


def build_sweep_signals(bars: pd.DataFrame, cfg: IctPaConfig) -> pd.DataFrame:
    """Combine frozen 4H ICT context with causal 15m PA/MSS execution.

    The 4H layer defines only an external-liquidity sweep.  A trade is not
    allowed until a later completed 15m candle breaks a previously confirmed
    local pivot with directional displacement.  This deliberately avoids
    tuning an inventory of ICT pattern parameters.
    """
    macro = resample_ohlcv(bars, "4h")
    pivots = confirmed_pivots(macro, cfg.sweep_pivot_left, cfg.sweep_pivot_right)
    known_low = pivots["confirmed_low"].shift(1)
    known_high = pivots["confirmed_high"].shift(1)
    candle_range = (macro["high"] - macro["low"]).replace(0.0, np.nan)
    close_location = (macro["close"] - macro["low"]) / candle_range
    lower_wick = (macro[["open", "close"]].min(axis=1) - macro["low"]) / candle_range
    upper_wick = (macro["high"] - macro[["open", "close"]].max(axis=1)) / candle_range
    sweep_long = (
        (macro["low"] < known_low)
        & (macro["close"] > known_low)
        & (close_location >= cfg.sweep_close_location)
        & (lower_wick >= cfg.sweep_wick_fraction)
    )
    sweep_short = (
        (macro["high"] > known_high)
        & (macro["close"] < known_high)
        & (close_location <= 1.0 - cfg.sweep_close_location)
        & (upper_wick >= cfg.sweep_wick_fraction)
    )
    macro_tr = true_range(macro)
    macro_atr = macro_tr.shift(1).rolling(cfg.sweep_atr_bars, min_periods=cfg.sweep_atr_bars).mean()

    # A 4H candle labelled T is complete at T+4H.  Only then may its sweep and
    # extreme become 15m context.
    macro_known = pd.DataFrame(
        {
            "macro_sweep_long": sweep_long.to_numpy(),
            "macro_sweep_short": sweep_short.to_numpy(),
            "macro_long_extreme": macro["low"].where(sweep_long).to_numpy(),
            "macro_short_extreme": macro["high"].where(sweep_short).to_numpy(),
            "macro_atr": macro_atr.to_numpy(),
        },
        index=macro.index + pd.Timedelta(hours=4),
    ).reindex(bars.index)

    micro_pivots = confirmed_pivots(bars, cfg.micro_pivot_left, cfg.micro_pivot_right)
    local_high = micro_pivots["confirmed_high"].shift(1)
    local_low = micro_pivots["confirmed_low"].shift(1)
    micro_range = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    body_fraction = (bars["close"] - bars["open"]).abs() / micro_range
    micro_tr = true_range(bars)
    micro_atr = micro_tr.shift(1).rolling(cfg.micro_atr_bars, min_periods=cfg.micro_atr_bars).mean()
    historical_volume_median = bars["volume"].shift(1).rolling(
        cfg.micro_atr_bars, min_periods=cfg.micro_atr_bars
    ).median()
    volume_confirmation = bars["volume"] >= historical_volume_median
    displacement_long = (
        (bars["close"] > bars["open"])
        & (bars["close"] > local_high)
        & (bars["close"].shift(1) <= local_high.shift(1))
        & (body_fraction >= cfg.displacement_body_fraction)
        & (micro_tr >= cfg.displacement_atr_multiple * micro_atr)
        & (volume_confirmation if cfg.require_volume_confirmation else True)
    )
    displacement_short = (
        (bars["close"] < bars["open"])
        & (bars["close"] < local_low)
        & (bars["close"].shift(1) >= local_low.shift(1))
        & (body_fraction >= cfg.displacement_body_fraction)
        & (micro_tr >= cfg.displacement_atr_multiple * micro_atr)
        & (volume_confirmation if cfg.require_volume_confirmation else True)
    )

    recent_long_context = macro_known["macro_sweep_long"].fillna(False).rolling(
        cfg.context_valid_bars, min_periods=1
    ).max().astype(bool)
    recent_short_context = macro_known["macro_sweep_short"].fillna(False).rolling(
        cfg.context_valid_bars, min_periods=1
    ).max().astype(bool)
    long_signal = recent_long_context & displacement_long
    short_signal = recent_short_context & displacement_short

    long_extreme = macro_known["macro_long_extreme"].ffill(limit=cfg.context_valid_bars - 1)
    short_extreme = macro_known["macro_short_extreme"].ffill(limit=cfg.context_valid_bars - 1)
    mapped_macro_atr = macro_known["macro_atr"].ffill(limit=cfg.context_valid_bars - 1)
    return pd.DataFrame(
        {
            "known_swing_low": local_low,
            "known_swing_high": local_high,
            "atr": mapped_macro_atr,
            "sweep_long": macro_known["macro_sweep_long"].fillna(False),
            "sweep_short": macro_known["macro_sweep_short"].fillna(False),
            "displacement_long": displacement_long,
            "displacement_short": displacement_short,
            "long_signal": long_signal,
            "short_signal": short_signal,
            "long_sweep_extreme": long_extreme.where(long_signal),
            "short_sweep_extreme": short_extreme.where(short_signal),
        },
        index=bars.index,
    )


def _swing_position(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: IctPaConfig,
    side: int,
) -> pd.DataFrame:
    signal_col = "long_signal" if side > 0 else "short_signal"
    extreme_col = "long_sweep_extreme" if side > 0 else "short_sweep_extreme"
    position = np.zeros(len(bars), dtype=float)
    entry_flag = np.zeros(len(bars), dtype=bool)
    exit_flag = np.zeros(len(bars), dtype=bool)
    stop_levels = np.full(len(bars), np.nan)
    target_levels = np.full(len(bars), np.nan)
    active = False
    exposure = 0.0
    stop = target = np.nan
    held = 0
    delay = 1 + int(cfg.execution_delay_bars)

    for i in range(len(bars)):
        # Positions are decided at the current open using a signal that was
        # already known at least one completed bar earlier.
        source = i - delay
        if not active and source >= 0 and bool(signals[signal_col].iloc[source]):
            entry = float(bars["open"].iloc[i])
            extreme = float(signals[extreme_col].iloc[source])
            atr = float(signals["atr"].iloc[source])
            if np.isfinite(entry) and np.isfinite(extreme) and np.isfinite(atr):
                stop = extreme - cfg.stop_atr_buffer * atr if side > 0 else extreme + cfg.stop_atr_buffer * atr
                risk_distance = (entry - stop) / entry if side > 0 else (stop - entry) / entry
                if risk_distance > 0:
                    exposure = min(cfg.swing_notional_cap, cfg.swing_risk_budget / risk_distance)
                    target = entry * (1.0 + cfg.swing_reward_risk * risk_distance) if side > 0 else entry * (1.0 - cfg.swing_reward_risk * risk_distance)
                    active = exposure > 0
                    held = 0
                    entry_flag[i] = active

        if active:
            position[i] = side * exposure
            stop_levels[i] = stop
            target_levels[i] = target
            held += 1
            stop_hit = float(bars["low"].iloc[i]) <= stop if side > 0 else float(bars["high"].iloc[i]) >= stop
            target_hit = float(bars["high"].iloc[i]) >= target if side > 0 else float(bars["low"].iloc[i]) <= target
            # Exit at the next open after an observed bar.  If stop and target
            # coexist intrabar, the stop-first convention is conservative.
            if stop_hit or target_hit or held >= cfg.swing_max_hold_bars:
                active = False
                exit_flag[min(i + 1, len(bars) - 1)] = True

    return pd.DataFrame(
        {
            "position": position,
            "entry": entry_flag,
            "exit": exit_flag,
            "stop": stop_levels,
            "target": target_levels,
        },
        index=bars.index,
    )


def build_open_positions(bars: pd.DataFrame, cfg: IctPaConfig) -> pd.DataFrame:
    cfg.validate()
    core = build_daily_structure_core(bars, cfg)
    signals = build_sweep_signals(bars, cfg)
    delay = 1 + int(cfg.execution_delay_bars)
    core_position = core["core_desired_close"].shift(delay).fillna(0.0)
    long_swing = _swing_position(bars, signals, cfg, side=1)
    short_swing = _swing_position(bars, signals, cfg, side=-1)
    # Tactical positions are hedges, not a second trend engine.  A long sweep
    # may execute only while the daily core is short; a short sweep may execute
    # only while the daily core is long.  This preserves independent hedge-mode
    # legs and prevents the tactical layer from pyramiding the macro position.
    if cfg.tactical_mode == "counter":
        long_swing.loc[core_position >= 0.0, "position"] = 0.0
        short_swing.loc[core_position <= 0.0, "position"] = 0.0
    elif cfg.tactical_mode == "none":
        long_swing.loc[:, "position"] = 0.0
        short_swing.loc[:, "position"] = 0.0
    return pd.concat(
        [
            core,
            signals,
            core_position.rename("core_position"),
            long_swing.add_prefix("swing_long_"),
            short_swing.add_prefix("swing_short_"),
        ],
        axis=1,
    )


def _simulate_feature(bars: pd.DataFrame, cfg: IctPaConfig, feature: pd.DataFrame) -> pd.DataFrame:
    cfg.validate()
    sleeves = ("core", "swing_long", "swing_short")
    equity = peak = 1.0
    previous = {name: 0.0 for name in sleeves}
    records: list[dict[str, object]] = []

    for i in range(len(bars) - 1):
        ts = bars.index[i]
        if ts < pd.Timestamp(cfg.start) or ts > pd.Timestamp(cfg.end):
            continue
        drawdown_before = equity / peak - 1.0
        speed = 0.25 if drawdown_before <= cfg.drawdown_quarter_speed else 0.50 if drawdown_before <= cfg.drawdown_half_speed else 1.0
        raw = {
            "core": float(feature["core_position"].iloc[i]),
            "swing_long": float(feature["swing_long_position"].iloc[i]),
            "swing_short": float(feature["swing_short_position"].iloc[i]),
        }
        target = {name: value * speed for name, value in raw.items()}
        gross = sum(abs(value) for value in target.values())
        if gross > cfg.gross_notional_cap:
            scale = cfg.gross_notional_cap / gross
            target = {name: value * scale for name, value in target.items()}
            gross = cfg.gross_notional_cap
        long_gross = sum(max(value, 0.0) for value in target.values())
        short_gross = sum(max(-value, 0.0) for value in target.values())
        net = sum(target.values())
        turnover = sum(abs(target[name] - previous[name]) for name in sleeves)
        trading_cost = turnover * cfg.one_way_cost
        carry_cost = gross * cfg.annual_carry_drag / BARS_PER_YEAR
        open_price = float(bars["open"].iloc[i])
        next_open = float(bars["open"].iloc[i + 1])
        price_return = next_open / open_price - 1.0
        gross_return = net * price_return
        net_return = gross_return - trading_cost - carry_cost
        low_return = float(bars["low"].iloc[i]) / open_price - 1.0
        high_return = float(bars["high"].iloc[i]) / open_price - 1.0
        worst_intrabar_pnl = long_gross * low_return - short_gross * high_return
        intrabar_equity_ratio = 1.0 - trading_cost - carry_cost + worst_intrabar_pnl
        maintenance = cfg.maintenance_margin_rate * gross
        liquidated = intrabar_equity_ratio <= maintenance or 1.0 + net_return <= 0
        before = equity
        equity *= max(0.0, 1.0 + net_return)
        peak = max(peak, equity)
        records.append(
            {
                "timestamp": ts,
                "next_timestamp": bars.index[i + 1],
                **{f"raw_{name}_position": raw[name] for name in sleeves},
                **{f"{name}_position": target[name] for name in sleeves},
                "long_gross_exposure": long_gross,
                "short_gross_exposure": short_gross,
                "gross_exposure": gross,
                "net_exposure": net,
                "hedged": bool(long_gross > 1e-12 and short_gross > 1e-12),
                "risk_speed": speed,
                "turnover": turnover,
                "price_return": price_return,
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "carry_cost": carry_cost,
                "net_return": net_return,
                "equity_before": before,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "intrabar_equity_ratio": intrabar_equity_ratio,
                "maintenance_required": maintenance,
                "maintenance_headroom": intrabar_equity_ratio - maintenance,
                "liquidated": liquidated,
                "sweep_long_signal": bool(feature["long_signal"].iloc[i]),
                "sweep_short_signal": bool(feature["short_signal"].iloc[i]),
                "daily_core_regime": float(feature["core_regime"].iloc[i]),
            }
        )
        previous = target
    return pd.DataFrame.from_records(records).set_index("timestamp")


def simulate_portfolio(bars: pd.DataFrame, cfg: IctPaConfig) -> pd.DataFrame:
    """Replay one frozen confirmation definition."""
    return _simulate_feature(bars, cfg, build_open_positions(bars, cfg))


def ensemble_configs(base: IctPaConfig) -> tuple[IctPaConfig, IctPaConfig, IctPaConfig]:
    """Equal-weight adjacent definitions; never select the historical winner."""
    return (
        replace(
            base,
            daily_pivot_left=1,
            daily_pivot_right=1,
            sweep_pivot_left=2,
            sweep_pivot_right=2,
            micro_pivot_left=1,
            micro_pivot_right=1,
        ),
        base,
        replace(
            base,
            daily_pivot_left=3,
            daily_pivot_right=3,
            sweep_pivot_left=4,
            sweep_pivot_right=4,
            micro_pivot_left=3,
            micro_pivot_right=3,
        ),
    )


def simulate_ensemble(bars: pd.DataFrame, cfg: IctPaConfig) -> pd.DataFrame:
    """Replay an equal-capital ensemble of fast/base/slow confirmations.

    Equal weighting is fixed and intentionally ignores which neighbour earned
    the most historically.  Each component is fully causal and independently
    maintains long/short tactical sleeves; the account cap is applied only
    after their gross positions are combined.
    """
    parts = [build_open_positions(bars, candidate) for candidate in ensemble_configs(cfg)]
    feature = parts[0].copy()
    for column in ("core_position", "swing_long_position", "swing_short_position"):
        feature[column] = sum(part[column] for part in parts) / len(parts)
    feature["long_signal"] = pd.concat([part["long_signal"] for part in parts], axis=1).any(axis=1)
    feature["short_signal"] = pd.concat([part["short_signal"] for part in parts], axis=1).any(axis=1)
    feature["core_regime"] = sum(part["core_regime"] for part in parts) / len(parts)
    return _simulate_feature(bars, cfg, feature)


def profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    returns = frame["net_return"].astype(float)
    elapsed = max((frame.index[-1] - frame.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final = float(frame["equity"].iloc[-1])
    cagr = final ** (1 / elapsed) - 1 if final > 0 else -1.0
    vol = float(returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR))
    monthly = (1.0 + returns).groupby(frame.index.to_period("M")).prod() - 1.0
    return {
        "start": str(frame.index.min()),
        "end": str(frame["next_timestamp"].max()),
        "bars": int(len(frame)),
        "total_return": final - 1.0,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe_zero_rf": float(returns.mean() / returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR)) if returns.std(ddof=0) > 0 else np.nan,
        "max_drawdown": float(frame["drawdown"].min()),
        "calmar": cagr / abs(float(frame["drawdown"].min())) if frame["drawdown"].min() < 0 else np.nan,
        "profit_factor_15m": profit_factor(returns),
        "positive_month_rate": float((monthly > 0).mean()),
        "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()),
        "max_gross_exposure": float(frame["gross_exposure"].max()),
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "hedged_bar_rate": float(frame["hedged"].mean()),
        "long_bar_rate": float((frame["net_exposure"] > 0).mean()),
        "short_bar_rate": float((frame["net_exposure"] < 0).mean()),
        "total_trading_cost": float(frame["trading_cost"].sum()),
        "total_carry_cost": float(frame["carry_cost"].sum()),
        "liquidation_events": int(frame["liquidated"].sum()),
        "min_maintenance_headroom": float(frame["maintenance_headroom"].min()),
    }


def period_summary(frame: pd.DataFrame, frequency: str = "Y") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, group in frame.groupby(frame.index.to_period(frequency)):
        returns = group["net_return"].astype(float)
        equity = (1.0 + returns).cumprod()
        dd = equity / equity.cummax() - 1.0
        rows.append(
            {
                "period": str(period),
                "return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(dd.min()),
                "profit_factor_15m": profit_factor(returns),
                "positive_bar_rate": float((returns > 0).mean()),
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
                "hedged_bar_rate": float(group["hedged"].mean()),
                "trading_cost": float(group["trading_cost"].sum()),
                "carry_cost": float(group["carry_cost"].sum()),
            }
        )
    return pd.DataFrame(rows)


def scenario_configs(base: IctPaConfig) -> tuple[tuple[str, IctPaConfig], ...]:
    return (
        ("base", base),
        ("cost_2x", replace(base, one_way_cost=base.one_way_cost * 2.0)),
        ("delay_plus_15m", replace(base, execution_delay_bars=base.execution_delay_bars + 1)),
        ("carry_5pct", replace(base, annual_carry_drag=0.05)),
        ("gross_cap_075", replace(base, gross_notional_cap=0.75, core_notional_cap=0.50)),
    )


def shock_survival(exposures: Iterable[float], cfg: IctPaConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shock in (0.20, 0.35, 0.50, 0.70):
        for exposure in exposures:
            gross = abs(float(exposure))
            equity_after = 1.0 - gross * shock
            maintenance = cfg.maintenance_margin_rate * gross
            rows.append(
                {
                    "adverse_move": shock,
                    "gross_exposure": gross,
                    "equity_after": equity_after,
                    "maintenance_required": maintenance,
                    "headroom": equity_after - maintenance,
                    "survives_assumption": equity_after > maintenance,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "BARS_PER_DAY",
    "IctPaConfig",
    "build_daily_structure_core",
    "build_open_positions",
    "build_sweep_signals",
    "confirmed_pivots",
    "period_summary",
    "resample_ohlcv",
    "scenario_configs",
    "shock_survival",
    "simulate_portfolio",
    "simulate_ensemble",
    "summarize",
]
