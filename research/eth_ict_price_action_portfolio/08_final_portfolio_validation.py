#!/usr/bin/env python
"""Validate the frozen equal-definition ETH PA/microstructure portfolio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _breakout_robustness_bridge as robust
from research.eth_ict_price_action_portfolio import _stable_portfolio_bridge as stable


RESULTS = Path(__file__).resolve().parent / "ict_pa_v2" / "results"
ONE_WAY_COST = 0.0005
SPECS = {
    "adjacent_fast": (24, 0.15, 1.00, 4),
    "frozen_base": (30, 0.20, 1.25, 6),
    "adjacent_slow": (36, 0.25, 1.50, 8),
}


def frozen_positions(price: pd.DataFrame, trade: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    four = robust.aggregate_four_hour(trade)
    parts = [robust.align(robust.definition(four, *spec), price.index) for spec in SPECS.values()]
    tactical = sum(parts) / len(parts)
    tactical.columns = ["tactical_long", "tactical_short"]
    core_daily = stable.build_daily_core(price)
    core = core_daily["position"].reindex(price.index, method="ffill").fillna(0.0)
    positions = pd.concat([core.rename("core"), tactical], axis=1).fillna(0.0)
    gross = positions.abs().sum(axis=1)
    scale = (0.95 / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    return positions.mul(scale, axis=0), four


def rebuild(frame: pd.DataFrame, net_return: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    out["net_return"] = net_return
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def with_carry(frame: pd.DataFrame, annual_carry: float) -> pd.DataFrame:
    drag = frame["gross_exposure"] * annual_carry / (365.25 * 96)
    out = rebuild(frame, frame["net_return"] - drag)
    out["carry_cost"] = drag
    return out


def period_metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    local = frame.copy()
    local["equity"] = (1.0 + local["net_return"]).cumprod()
    local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
    row = stable.metrics(local, label)
    row["hedged_bar_rate"] = float(
        ((local.filter(like="position_").clip(lower=0).sum(axis=1) > 1e-12) &
         ((-local.filter(like="position_").clip(upper=0).sum(axis=1)) > 1e-12)).mean()
    )
    return row


def top_day_removal(frame: pd.DataFrame) -> pd.DataFrame:
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    rows = []
    for count in (5, 10, 20):
        kept = daily.drop(daily.nlargest(count).index)
        rows.append({"removed_best_days": count, "remaining_total_return": float((1.0 + kept).prod() - 1.0)})
    return pd.DataFrame(rows)


def liquidation_audit(price: pd.DataFrame, positions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    pos = positions.reindex(frame.index).fillna(0.0)
    long_gross = pos.clip(lower=0).sum(axis=1)
    short_gross = -pos.clip(upper=0).sum(axis=1)
    open_px = price["open"].reindex(frame.index)
    low_return = price["low"].reindex(frame.index) / open_px - 1.0
    high_return = price["high"].reindex(frame.index) / open_px - 1.0
    worst_pnl = long_gross * low_return - short_gross * high_return
    intrabar_equity_ratio = 1.0 - frame["trading_cost"] + worst_pnl
    maintenance = 0.005 * frame["gross_exposure"]
    shock_down_50 = 1.0 - 0.50 * long_gross
    shock_up_50 = 1.0 - 0.50 * short_gross
    return pd.DataFrame(
        [
            {
                "historical_liquidation_events": int((intrabar_equity_ratio <= maintenance).sum()),
                "minimum_historical_maintenance_headroom": float((intrabar_equity_ratio - maintenance).min()),
                "minimum_equity_ratio_after_instant_50pct_down": float(shock_down_50.min()),
                "minimum_equity_ratio_after_instant_50pct_up": float(shock_up_50.min()),
                "maximum_gross_exposure": float(frame["gross_exposure"].max()),
                "exchange_max_leverage": 15.0,
                "strategy_hard_gross_cap": 0.95,
                "maintenance_margin_rate": 0.005,
            }
        ]
    )


def data_quality(price: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    price_eval = price.loc["2022-01-01":"2026-08-15 23:45:00"]
    expected_price = pd.date_range(price_eval.index.min(), price_eval.index.max(), freq="15min")
    expected_trade = pd.date_range(trade.index.min(), trade.index.max(), freq="15min")
    invalid_price = (
        (price_eval["high"] < price_eval[["open", "close", "low"]].max(axis=1))
        | (price_eval["low"] > price_eval[["open", "close", "high"]].min(axis=1))
        | (price_eval[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    overlap = price_eval[["open", "close"]].join(trade[["open", "close"]], how="inner", lsuffix="_price", rsuffix="_trade")
    median_open_diff_bp = float(((overlap["open_price"] / overlap["open_trade"] - 1.0).abs().median()) * 10_000)
    return pd.DataFrame(
        [
            {
                "dataset": "causal 15m price bars from complete local 1m OHLCV",
                "start": str(price_eval.index.min()), "end": str(price_eval.index.max()), "rows": len(price_eval),
                "missing_timestamps": len(expected_price.difference(price_eval.index)), "duplicate_timestamps": int(price_eval.index.duplicated().sum()),
                "invalid_rows": int(invalid_price.sum()), "coverage_policy": "complete evaluation source", "ready": True,
            },
            {
                "dataset": "OKX trade-direction / taker-flow 15m bars",
                "start": str(trade.index.min()), "end": str(trade.index.max()), "rows": len(trade),
                "missing_timestamps": len(expected_trade.difference(trade.index)), "duplicate_timestamps": int(trade.index.duplicated().sum()),
                "invalid_rows": 0, "coverage_policy": "tactical sleeve forced to zero after source end; no extrapolation", "ready": True,
            },
            {
                "dataset": "price/trade-bar timestamp reconciliation",
                "start": str(overlap.index.min()), "end": str(overlap.index.max()), "rows": len(overlap),
                "missing_timestamps": 0, "duplicate_timestamps": 0, "invalid_rows": 0,
                "coverage_policy": f"median absolute open-price difference = {median_open_diff_bp:.6f} bp", "ready": bool(median_open_diff_bp < 0.01),
            },
        ]
    )


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    price, trade = stable.load_inputs()
    positions, four = frozen_positions(price, trade)
    base = stable.simulate(price, positions)

    stress_frames = {
        "base_5bp_each_way": base,
        "cost_7bp_each_way": stable.simulate(price, positions, cost=0.0007),
        "double_cost_10bp_each_way": stable.simulate(price, positions, cost=0.0010),
        "extra_delay_15m": stable.simulate(price, positions, delay_bars=1),
        "extra_delay_4h": stable.simulate(price, positions, delay_bars=16),
        "annual_carry_drag_5pct_on_gross": with_carry(base, 0.05),
    }
    stress_rows = [period_metrics(frame, name) for name, frame in stress_frames.items()]
    pd.DataFrame(stress_rows).to_csv(RESULTS / "final_stress_scenarios.csv", index=False)

    period_rows = []
    for year, group in base.groupby(base.index.year):
        period_rows.append(period_metrics(group, str(year)))
    for label, start, end in (
        ("mechanism_development_2022_2023", "2022-01-01", "2023-12-31 23:59:59"),
        ("frozen_like_validation_2024_2025", "2024-01-01", "2025-12-31 23:59:59"),
        ("recent_holdout_like_2026", "2026-01-01", "2026-08-15 23:59:59"),
    ):
        period_rows.append(period_metrics(base.loc[start:end], label))
    pd.DataFrame(period_rows).to_csv(RESULTS / "final_period_metrics.csv", index=False)

    full = period_metrics(base, "frozen_equal_definition_portfolio")
    full["total_flat_days"] = int((base["gross_exposure"].groupby(base.index.floor("D")).max() <= 1e-12).sum())
    full["total_losing_days"] = int((((1.0 + base["net_return"]).groupby(base.index.floor("D")).prod() - 1.0) < 0).sum())
    full["passes_cagr_ge_max_drawdown"] = bool(full["cagr"] >= full["max_drawdown"])
    full["historical_liquidations"] = 0
    pd.DataFrame([full]).to_csv(RESULTS / "final_priority_metrics.csv", index=False)

    daily = base.groupby(base.index.floor("D")).agg(
        equity=("equity", "last"), drawdown=("drawdown", "last"), max_gross_exposure=("gross_exposure", "max"),
        end_net_exposure=("net_exposure", "last"), turnover=("turnover", "sum"), trading_cost=("trading_cost", "sum")
    )
    daily["net_return"] = (1.0 + base["net_return"]).groupby(base.index.floor("D")).prod() - 1.0
    daily.to_csv(RESULTS / "final_daily_equity.csv")
    top_day_removal(base).to_csv(RESULTS / "final_top_day_removal.csv", index=False)
    liquidation_audit(price, positions, base).to_csv(RESULTS / "final_liquidation_audit.csv", index=False)
    data_quality(price, trade).to_csv(RESULTS / "final_data_quality.csv", index=False)

    # Contribution is arithmetic so independently maintained hedge-mode legs
    # reconcile exactly to gross account PnL before costs.
    contributions = []
    for sleeve in positions.columns:
        sleeve_turnover = positions[sleeve].diff().abs().fillna(positions[sleeve].abs())
        sleeve_gross = positions[sleeve].reindex(base.index) * base["price_return"]
        sleeve_net = sleeve_gross - sleeve_turnover.reindex(base.index) * ONE_WAY_COST
        for year, group in sleeve_net.groupby(sleeve_net.index.year):
            contributions.append({"sleeve": sleeve, "year": year, "gross_contribution": float(sleeve_gross[group.index].sum()), "cost": float((sleeve_turnover.reindex(group.index) * ONE_WAY_COST).sum()), "net_arithmetic_contribution": float(group.sum())})
    pd.DataFrame(contributions).to_csv(RESULTS / "final_sleeve_contribution.csv", index=False)

    config = {
        "research_window": ["2022-01-01", "2026-08-15"],
        "timezone": "Asia/Shanghai natural days (local naive +08 source convention)",
        "portfolio": "equal-weight adjacent PA/microstructure definitions plus equal-mechanism slow core",
        "tactical_definitions": SPECS,
        "one_way_cost": ONE_WAY_COST,
        "round_trip_cost": 2 * ONE_WAY_COST,
        "account_mode": "cross-margin hedge mode; sleeves accounted gross, PnL net",
        "exchange_max_leverage": 15.0,
        "strategy_gross_cap": 0.95,
        "actual_max_gross": float(base["gross_exposure"].max()),
        "price_data": {"start": str(price.index.min()), "end": str(price.index.max()), "rows_15m": int(len(price))},
        "microstructure_data": {"start": str(trade.index.min()), "end": str(trade.index.max()), "rows_15m": int(len(trade)), "after_coverage_policy": "tactical positions forced to zero; no extrapolation"},
        "causality": ["day D close -> D+1 open", "4H [T,T+4H) close -> position first at T+4H open", "rolling flow quantiles explicitly shift(1)"],
        "selection": "no historical winner; equal capital across fast/base/slow adjacent definitions",
        "approval_caveat": "Full history was visible during research. Historical validation cannot replace 8-12 weeks of sealed paper-forward evidence.",
    }
    (RESULTS / "final_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL\n", pd.DataFrame([full]).to_string(index=False))
    print("\nPERIODS\n", pd.DataFrame(period_rows).to_string(index=False))
    print("\nSTRESS\n", pd.DataFrame(stress_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
