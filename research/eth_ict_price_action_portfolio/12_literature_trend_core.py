#!/usr/bin/env python
"""Literature-defined monthly TSMOM plus confirmed daily PA structure core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _stable_portfolio_bridge as stable
from research.eth_ict_price_action_portfolio.ict_pa_model import IctPaConfig, build_daily_structure_core


RESULTS = Path(__file__).resolve().parent / "ict_pa_v3" / "results"


def monthly_tsmom_positions(bars: pd.DataFrame, mode: str) -> pd.Series:
    daily = bars.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna()
    monthly = daily["close"].resample("ME").last()
    monthly_signals = []
    for lookback, holding in ((1, 1), (3, 1), (6, 1), (12, 1), (12, 3)):
        signal = np.sign(np.log(monthly).diff(lookback))
        active = pd.concat([signal.shift(age) for age in range(holding)], axis=1).mean(axis=1)
        monthly_signals.append(active)
    forecast = pd.concat(monthly_signals, axis=1).mean(axis=1).clip(-1.0, 1.0)

    daily_return = np.log(daily["close"]).diff()
    # TSMOM paper: exponentially weighted lagged squared daily returns with a
    # 60-day center of mass, always applied with a lag.
    alpha = 1.0 / 61.0
    variance = daily_return.pow(2).ewm(alpha=alpha, adjust=False, min_periods=60).mean().shift(1) * 365.25
    monthly_variance = variance.reindex(monthly.index, method="ffill")
    monthly_vol = np.sqrt(monthly_variance)
    if mode == "inverse_volatility":
        size = (0.15 / monthly_vol.replace(0.0, np.nan)).clip(upper=0.70)
    elif mode == "inverse_variance":
        # Moreira-Muir mean-variance scaling: target annual variance divided
        # by last known annualized realized variance.  No fitted normalizer.
        size = ((0.15 ** 2) / monthly_variance.replace(0.0, np.nan)).clip(upper=0.70)
    else:
        raise ValueError(mode)
    desired = (forecast * size).fillna(0.0)

    # Month-end close is observable at the following natural-day boundary;
    # wait one additional 15m interval before execution.
    available_index = monthly.index + pd.Timedelta(days=1, minutes=15)
    available = pd.Series(desired.to_numpy(), index=available_index)
    return available.reindex(bars.index, method="ffill").fillna(0.0)


def daily_bos_positions(bars: pd.DataFrame) -> pd.Series:
    cfg = IctPaConfig(core_mode="daily_12m_blend")
    structure = build_daily_structure_core(bars, cfg)
    # Structure becomes known at the daily boundary; wait one 15m open.
    return structure["core_desired_close"].shift(1).fillna(0.0)


def _run(bars: pd.DataFrame, columns: dict[str, pd.Series]) -> pd.DataFrame:
    return stable.simulate(bars, pd.DataFrame(columns, index=bars.index))


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bars, _ = stable.load_inputs()
    inv_vol = monthly_tsmom_positions(bars, "inverse_volatility")
    inv_var = monthly_tsmom_positions(bars, "inverse_variance")
    bos = daily_bos_positions(bars)
    candidates = {
        "monthly_tsmom_inverse_volatility": {"tsmom": inv_vol},
        "monthly_tsmom_inverse_variance": {"tsmom": inv_var},
        "confirmed_daily_bos": {"bos": bos},
        "equal_tsmom_vol_and_variance": {"tsmom_vol": inv_vol * 0.5, "tsmom_var": inv_var * 0.5},
        "equal_monthly_tsmom_and_daily_bos": {"tsmom_vol": inv_vol * 0.25, "tsmom_var": inv_var * 0.25, "bos": bos * 0.5},
    }
    rows = []
    period_rows = []
    for name, positions in candidates.items():
        frame = _run(bars, positions)
        rows.append(stable.metrics(frame, name))
        for year, group in frame.groupby(frame.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = stable.metrics(local, f"{name}:{year}")
            row["model"] = name
            row["year"] = year
            period_rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "03_literature_trend_core_screen.csv", index=False)
    pd.DataFrame(period_rows).to_csv(RESULTS / "04_literature_trend_core_years.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
