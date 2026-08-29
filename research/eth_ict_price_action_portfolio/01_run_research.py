#!/usr/bin/env python
"""Run the frozen ETH ICT-macro / PA-microstructure portfolio research."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio.ict_pa_model import (  # noqa: E402
    BARS_PER_YEAR,
    IctPaConfig,
    period_summary,
    resample_ohlcv,
    scenario_configs,
    shock_survival,
    simulate_portfolio,
    summarize,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent / "ict_pa_v1" / "results"
WARMUP_START = pd.Timestamp("2020-01-01 00:00:00")
NEIGHBOURHOODS = (
    ("faster_confirmation", {"daily_pivot_left": 1, "daily_pivot_right": 1, "sweep_pivot_left": 2, "sweep_pivot_right": 2, "micro_pivot_left": 1, "micro_pivot_right": 1}),
    ("frozen_base", {}),
    ("slower_confirmation", {"daily_pivot_left": 3, "daily_pivot_right": 3, "sweep_pivot_left": 4, "sweep_pivot_right": 4, "micro_pivot_left": 3, "micro_pivot_right": 3}),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-neighbourhood", action="store_true")
    return parser.parse_args(argv)


def load_local_minute(cfg: IctPaConfig) -> pd.DataFrame:
    source = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data()
    if source.empty:
        raise RuntimeError("local ETH-USDT-SWAP 1m data is unavailable")
    source = source.sort_index(kind="stable")
    source = source[~source.index.duplicated(keep="last")]
    selected = source.loc[WARMUP_START:pd.Timestamp(cfg.end).floor("min"), ["open", "high", "low", "close", "volume"]].copy()
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna()
    if selected.index.min() > WARMUP_START or selected.index.max() < pd.Timestamp(cfg.end).floor("min"):
        raise RuntimeError(f"minute coverage gap at boundary: {selected.index.min()} -> {selected.index.max()}")
    return selected


def quality_rows(minute: pd.DataFrame, bars: pd.DataFrame, cfg: IctPaConfig) -> list[dict[str, object]]:
    minute_eval = minute.loc[pd.Timestamp(cfg.start):pd.Timestamp(cfg.end).floor("min")]
    expected_minute = pd.date_range(minute_eval.index.min(), minute_eval.index.max(), freq="1min")
    expected_15m = pd.date_range(bars.index.min(), bars.index.max(), freq="15min")
    invalid_minute = (
        (minute_eval["high"] < minute_eval[["open", "close", "low"]].max(axis=1))
        | (minute_eval["low"] > minute_eval[["open", "close", "high"]].min(axis=1))
        | (minute_eval[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    return [
        {
            "dataset": "ETH-USDT-SWAP local 1m OHLCV",
            "start": str(minute_eval.index.min()),
            "end": str(minute_eval.index.max()),
            "rows": int(len(minute_eval)),
            "missing_timestamps": int(len(expected_minute.difference(minute_eval.index))),
            "duplicate_timestamps": int(minute_eval.index.duplicated().sum()),
            "invalid_ohlc_rows": int(invalid_minute.sum()),
            "ready": bool(len(expected_minute.difference(minute_eval.index)) == 0 and not invalid_minute.any()),
        },
        {
            "dataset": "causal 15m aggregate",
            "start": str(bars.index.min()),
            "end": str(bars.index.max()),
            "rows": int(len(bars)),
            "missing_timestamps": int(len(expected_15m.difference(bars.index))),
            "duplicate_timestamps": int(bars.index.duplicated().sum()),
            "invalid_ohlc_rows": 0,
            "ready": bool(len(expected_15m.difference(bars.index)) == 0),
        },
    ]


def segment_summary(frame: pd.DataFrame, name: str, start: str, end: str) -> dict[str, object]:
    group = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    ret = group["net_return"].astype(float)
    equity = (1.0 + ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    elapsed = max((group.index[-1] - group.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final = float(equity.iloc[-1])
    monthly = (1.0 + ret).groupby(ret.index.to_period("M")).prod() - 1.0
    return {
        "segment": name,
        "start": str(group.index.min()),
        "end": str(group.index.max()),
        "total_return": final - 1.0,
        "cagr": final ** (1 / elapsed) - 1.0,
        "max_drawdown": float(drawdown.min()),
        "sharpe_zero_rf": float(ret.mean() / ret.std(ddof=0) * np.sqrt(BARS_PER_YEAR)) if ret.std(ddof=0) > 0 else np.nan,
        "positive_month_rate": float((monthly > 0).mean()),
        "months": int(len(monthly)),
    }


def position_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sleeve in ("core", "swing_long", "swing_short"):
        position = frame[f"{sleeve}_position"]
        sign = np.sign(position).astype(int)
        changed = sign.ne(sign.shift(fill_value=0))
        episode_id = changed.cumsum()
        for _, group in frame[sign.ne(0)].groupby(episode_id[sign.ne(0)]):
            contribution = group[f"{sleeve}_position"] * group["price_return"]
            rows.append(
                {
                    "sleeve": sleeve,
                    "side": "LONG" if group[f"{sleeve}_position"].iloc[0] > 0 else "SHORT",
                    "entry_time": str(group.index.min()),
                    "exit_time": str(group["next_timestamp"].max()),
                    "bars": int(len(group)),
                    "days": float(len(group) / 96),
                    "gross_contribution": float(contribution.sum()),
                    "max_notional": float(group[f"{sleeve}_position"].abs().max()),
                }
            )
    return pd.DataFrame(rows)


def top_day_removal(frame: pd.DataFrame, count: int = 10) -> dict[str, object]:
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    removed = daily.drop(daily.nlargest(count).index)
    return {
        "removed_best_days": count,
        "original_total_return": float((1.0 + daily).prod() - 1.0),
        "remaining_total_return": float((1.0 + removed).prod() - 1.0),
        "largest_removed_day": float(daily.max()),
    }


def daily_account_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = frame.groupby(frame.index.floor("D"))
    out = pd.DataFrame(
        {
            "date": groups.size().index,
            "net_return": groups["net_return"].apply(lambda x: float((1.0 + x).prod() - 1.0)).to_numpy(),
            "equity": groups["equity"].last().to_numpy(),
            "drawdown": groups["drawdown"].last().to_numpy(),
            "max_gross_exposure": groups["gross_exposure"].max().to_numpy(),
            "end_net_exposure": groups["net_exposure"].last().to_numpy(),
            "hedged_bar_rate": groups["hedged"].mean().to_numpy(),
            "trading_cost": groups["trading_cost"].sum().to_numpy(),
            "carry_cost": groups["carry_cost"].sum().to_numpy(),
        }
    )
    out["flat_day"] = out["max_gross_exposure"] <= 1e-12
    out["losing_day"] = out["net_return"] < 0.0
    return out


def max_true_streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = IctPaConfig()
    cfg.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[load] local 1m OHLCV", flush=True)
    minute = load_local_minute(cfg)
    bars = resample_ohlcv(minute, "15min")
    print(f"[data] minute={len(minute):,} 15m={len(bars):,}", flush=True)

    scenario_rows: list[dict[str, object]] = []
    base: pd.DataFrame | None = None
    for name, scenario_cfg in scenario_configs(cfg):
        print(f"[scenario] {name}", flush=True)
        frame = simulate_portfolio(bars, scenario_cfg)
        if name == "base":
            base = frame
        scenario_rows.append({"scenario": name, **summarize(frame)})
    assert base is not None

    neighbourhood_rows: list[dict[str, object]] = []
    if not args.skip_neighbourhood:
        for name, changes in NEIGHBOURHOODS:
            print(f"[neighbourhood] {name}", flush=True)
            candidate = replace(cfg, **changes)
            frame = base if name == "frozen_base" else simulate_portfolio(bars, candidate)
            neighbourhood_rows.append({"variant": name, **summarize(frame)})

    segments = [
        segment_summary(base, "mechanism-development (not untouched OOS)", "2022-01-01", "2023-12-31 23:59:59"),
        segment_summary(base, "post-development validation", "2024-01-01", "2025-12-31 23:59:59"),
        segment_summary(base, "recent holdout-like window", "2026-01-01", "2026-08-15 23:59:59"),
    ]
    summary = pd.DataFrame(scenario_rows)
    yearly = period_summary(base, "Y")
    monthly = period_summary(base, "M")
    daily = daily_account_summary(base)
    base_summary = summarize(base)
    priority = pd.DataFrame(
        [
            {
                "candidate": "daily_12m_blend_counter_hedge",
                "max_consecutive_flat_days": max_true_streak(daily["flat_day"]),
                "total_flat_days": int(daily["flat_day"].sum()),
                "max_consecutive_losing_days": max_true_streak(daily["losing_day"]),
                "total_losing_days": int(daily["losing_day"].sum()),
                "max_drawdown_abs": abs(float(base_summary["max_drawdown"])),
                "cagr": float(base_summary["cagr"]),
                "total_return": float(base_summary["total_return"]),
                "calmar": float(base_summary["calmar"]),
                "passes_cagr_ge_drawdown": bool(base_summary["cagr"] >= abs(base_summary["max_drawdown"])),
                "dual_sleeve_eligible": bool(base_summary["hedged_bar_rate"] > 0),
            }
        ]
    )
    episodes = position_episodes(base)
    data_quality = pd.DataFrame(quality_rows(minute, bars, cfg))
    shock = shock_survival([0.50, cfg.gross_notional_cap, cfg.exchange_leverage_cap], cfg)
    signal_audit = pd.DataFrame(
        [
            {
                "long_mss_signals": int(base["sweep_long_signal"].sum()),
                "short_mss_signals": int(base["sweep_short_signal"].sum()),
                "hedged_bars": int(base["hedged"].sum()),
                "hedged_bar_rate": float(base["hedged"].mean()),
                "max_gross_exposure": float(base["gross_exposure"].max()),
                "max_exchange_leverage": cfg.exchange_leverage_cap,
                "one_way_cost": cfg.one_way_cost,
                "round_trip_cost": cfg.one_way_cost * 2,
            }
        ]
    )

    outputs = {
        "summary.csv": summary,
        "yearly.csv": yearly,
        "monthly.csv": monthly,
        "daily_equity.csv": daily,
        "position_episodes.csv": episodes,
        "data_quality.csv": data_quality,
        "shock_survival.csv": shock,
        "validation_segments.csv": pd.DataFrame(segments),
        "signal_audit.csv": signal_audit,
        "parameter_neighbourhood.csv": pd.DataFrame(neighbourhood_rows),
        "top_day_removal.csv": pd.DataFrame([top_day_removal(base)]),
        "final_priority_metrics.csv": priority,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "source": "local SQLite ETH_USDT_SWAP_1m",
                "warmup_start": str(WARMUP_START),
                "research_window": [cfg.start, cfg.end],
                "causality": "completed candles -> next 15m open or later",
                "selection_warning": "The full historical window was visible during this research; 2026 is holdout-like, not a pristine untouched OOS test.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(summary[["scenario", "total_return", "cagr", "max_drawdown", "sharpe_zero_rf", "liquidation_events"]].to_string(index=False))
    print("[done]", args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
