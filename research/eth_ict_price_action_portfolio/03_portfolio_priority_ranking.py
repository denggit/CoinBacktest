#!/usr/bin/env python
"""Rank frozen portfolio mechanisms by the user's explicit capability order."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio.ict_pa_model import (  # noqa: E402
    IctPaConfig,
    resample_ohlcv,
    simulate_portfolio,
    summarize,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402


OUT = Path(__file__).resolve().parent / "ict_pa_v1" / "results"


def max_true_streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def priority_metrics(name: str, frame: pd.DataFrame, dual_sleeve_eligible: bool) -> dict[str, object]:
    daily_return = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    daily_max_gross = frame["gross_exposure"].groupby(frame.index.floor("D")).max()
    flat = daily_max_gross <= 1e-12
    losing = daily_return < 0.0
    base = summarize(frame)
    return {
        "candidate": name,
        "dual_sleeve_eligible": dual_sleeve_eligible,
        "max_consecutive_flat_days": max_true_streak(flat),
        "total_flat_days": int(flat.sum()),
        "max_consecutive_losing_days": max_true_streak(losing),
        "total_losing_days": int(losing.sum()),
        "max_drawdown_abs": abs(float(base["max_drawdown"])),
        "cagr": float(base["cagr"]),
        "total_return": float(base["total_return"]),
        "positive_month_rate": float(base["positive_month_rate"]),
        "max_gross_exposure": float(base["max_gross_exposure"]),
        "hedged_bar_rate": float(base["hedged_bar_rate"]),
        "liquidation_events": int(base["liquidation_events"]),
    }


def load_bars() -> pd.DataFrame:
    cache = Path(__file__).resolve().parent / "bars_15m.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    minute = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data()
    minute = minute.loc["2020-01-01":"2026-08-15 23:59:59", ["open", "high", "low", "close", "volume"]]
    return resample_ohlcv(minute, "15min")


def main() -> int:
    bars = load_bars()
    base = IctPaConfig()
    candidates = (
        ("daily_12m_blend_counter_hedge", base, True),
        ("daily_core_counter_hedge", replace(base, core_mode="daily"), True),
        ("daily_core_only", replace(base, core_mode="daily", tactical_mode="none"), False),
        ("daily_core_independent_tactical", replace(base, core_mode="daily", tactical_mode="independent"), True),
        ("daily_weekly_consensus_counter", replace(base, core_mode="daily_weekly_consensus"), True),
    )
    rows: list[dict[str, object]] = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cfg, eligible in candidates:
        print(f"[rank] {name}", flush=True)
        frame = simulate_portfolio(bars, cfg)
        rows.append(priority_metrics(name, frame, eligible))
        frame.resample("1D").last().reset_index().to_csv(OUT / f"daily_{name}.csv", index=False)
    ranking = pd.DataFrame(rows).sort_values(
        [
            "max_consecutive_flat_days",
            "max_consecutive_losing_days",
            "max_drawdown_abs",
            "cagr",
            "total_return",
        ],
        ascending=[True, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "all_candidate_rank", range(1, len(ranking) + 1))
    ranking.insert(1, "eligible_priority_rank", pd.Series(pd.NA, index=ranking.index, dtype="Int64"))
    eligible_index = ranking.index[ranking["dual_sleeve_eligible"]]
    ranking.loc[eligible_index, "eligible_priority_rank"] = range(1, len(eligible_index) + 1)
    ranking.to_csv(OUT / "portfolio_priority_ranking.csv", index=False)
    print(ranking.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
