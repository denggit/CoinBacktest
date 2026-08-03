#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R01 pipeline: preflight -> samples -> models -> trades -> decision."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.utils.report import print_full_report

from .backtest import FoldEvaluation, evaluate_model_on_fold, select_validation_champion
from .config import TradesBaselineConfig
from .dataset import (
    build_monthly_sample_cache,
    create_loader,
    feature_columns,
    list_cached_months,
    run_public_loader_preflight,
)
from .modeling import (
    calibration_thresholds,
    default_folds,
    feature_importance_frame,
    fit_model_set,
    save_model_bundle,
    validate_model_dependencies,
)


@dataclass(frozen=True)
class PipelineResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None
    sealed_result: dict[str, object] | None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _trade_history_for_full_report(trades: pd.DataFrame, initial_capital: float) -> tuple[list[dict[str, object]], float]:
    if trades.empty:
        return [], initial_capital
    ordered = trades.sort_values("exit_time", kind="stable")
    history: list[dict[str, object]] = []
    prev_capital = float(initial_capital)
    for row in ordered.itertuples(index=False):
        capital = float(row.capital)
        pnl = capital - prev_capital
        gross = float(row.gross_return)
        direction = str(row.direction)
        exit_price = 1.0 + gross if direction == "long" else max(1e-9, 1.0 - gross)
        history.append(
            {
                "entry_time": pd.Timestamp(row.entry_time),
                "exit_time": pd.Timestamp(row.exit_time),
                "type": direction,
                "entry": 1.0,
                "exit": exit_price,
                "pnl": pnl,
                "fee": prev_capital * float(row.cost_rate),
                "capital": capital,
                "mfe_r": np.nan,
                "mae_r": np.nan,
            }
        )
        prev_capital = capital
    return history, prev_capital


def _write_full_report(
    trades: pd.DataFrame,
    *,
    strategy_name: str,
    report_dir: Path,
    initial_capital: float,
) -> None:
    history, final_capital = _trade_history_for_full_report(trades, initial_capital)
    if trades.empty:
        idx = pd.date_range("2023-01-01", periods=2, freq="D")
    else:
        idx = pd.DatetimeIndex([pd.Timestamp(trades["decision_time"].min()), pd.Timestamp(trades["exit_time"].max())])
    dummy = pd.DataFrame(index=idx, data={"close": [1.0] * len(idx)})
    total_days = max(1.0, (idx[-1] - idx[0]).total_seconds() / 86400.0)
    print_full_report(
        history,
        dummy,
        initial_capital,
        final_capital,
        strategy_name,
        total_days,
        ai_enabled=True,
        symbol="ETH-USDT-SWAP",
        report_dir=str(report_dir),
    )




def _period_breakdown(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"])
    rows: list[dict[str, object]] = []
    for frequency, period_values in (
        ("year", frame["exit_time"].dt.to_period("Y")),
        ("quarter", frame["exit_time"].dt.to_period("Q")),
        ("month", frame["exit_time"].dt.to_period("M")),
    ):
        for period, group in frame.groupby(period_values, sort=True):
            returns = group["net_return"].to_numpy(dtype=float)
            wins = returns[returns > 0]
            losses = returns[returns <= 0]
            pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("inf")
            rows.append(
                {
                    "dataset": label,
                    "frequency": frequency,
                    "period": str(period),
                    "trades": int(len(group)),
                    "win_rate": float((returns > 0).mean()),
                    "mean_net_return": float(returns.mean()),
                    "profit_factor": pf,
                    "compound_return": float(np.prod(1.0 + returns) - 1.0),
                    "long_trades": int((group["direction"] == "long").sum()),
                    "short_trades": int((group["direction"] == "short").sum()),
                }
            )
    return pd.DataFrame(rows)


def _markdown_summary(
    *,
    config: TradesBaselineConfig,
    decision: str,
    champion: dict[str, object] | None,
    sealed: dict[str, object] | None,
    preflight_status: str,
) -> str:
    lines = [
        "# ETH AI Trading R01 — Trades-only supervised baseline",
        "",
        f"- Decision: **{decision}**",
        f"- Public-loader preflight: `{preflight_status}`",
        "- Input: existing `OKXTradeBarLoader(timeframe=\"1s\")` only",
        f"- Decision cadence: `{config.decision_interval_seconds}s`",
        f"- Full market-order fee: `{config.round_trip_fee_rate:.3%}`",
        f"- Slippage assumption: `{config.slippage_rate_per_side:.3%}` per side",
        "- Model selection period: `WF_2025` only",
        "- Sealed period: `2026-01-01` to `2026-06-30`",
        "",
        "## Validation champion",
        "",
    ]
    if champion is None:
        lines.append("No candidate passed the frozen validation and robustness gates.")
    else:
        lines.extend(
            [
                f"- Model: `{champion['model']}`",
                f"- Horizon: `{int(champion['horizon_seconds'])}s`",
                f"- Signal quantile: `{float(champion['quantile']):.3f}`",
                f"- Validation trades: `{int(champion['trades'])}`",
                f"- Validation PF: `{float(champion['profit_factor']):.3f}`",
                f"- Validation mean net/trade: `{float(champion['mean_net_return']):.4%}`",
                f"- Validation total return: `{float(champion['total_return']):.2%}`",
                f"- Validation 2x-cost return: `{float(champion['return_2x']):.2%}`",
            ]
        )
    lines.extend(["", "## Sealed 2026 result", ""])
    if sealed is None:
        lines.append("Not evaluated because no validation champion passed.")
    else:
        lines.extend(
            [
                f"- Trades: `{int(sealed['trades'])}`",
                f"- Win rate: `{float(sealed['win_rate']):.2%}`",
                f"- PF: `{float(sealed['profit_factor']):.3f}`",
                f"- Mean net/trade: `{float(sealed['mean_net_return']):.4%}`",
                f"- Total return: `{float(sealed['total_return']):.2%}`",
                f"- MDD: `{float(sealed['max_drawdown']):.2%}`",
                f"- Positive month ratio: `{float(sealed['positive_month_ratio']):.2%}`",
                f"- Top-10 removed return: `{float(sealed['top10_removed_total_return']):.2%}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "R01 is a fast go/no-go gate. It does not claim a final live strategy and its fixed-horizon exit is only a baseline label conversion. If it fails, do not hide the failure with a larger neural network. If it passes, the next stage adds market state and unified signal scoring.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pipeline(
    config: TradesBaselineConfig,
    *,
    data_dir: str | Path | None = None,
    force_rebuild_cache: bool = False,
    models: tuple[str, ...] = ("ridge", "lightgbm"),
    progress: bool = True,
) -> PipelineResult:
    config.validate()
    # Fail before loader access or the potentially expensive monthly cache build.
    model_dependencies = validate_model_dependencies(models)
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "models").mkdir(exist_ok=True)
    (report_dir / "full_reports").mkdir(exist_ok=True)

    _write_json(
        report_dir / "00_runtime_and_config.json",
        {
            "config": config.to_dict(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "model_dependencies": model_dependencies,
        },
    )

    loader = create_loader(config, data_dir=data_dir)
    preflight = run_public_loader_preflight(loader, config)
    _write_json(report_dir / "01_public_loader_preflight.json", preflight.to_dict())
    if preflight.status != "PASS":
        summary = _markdown_summary(
            config=config,
            decision="BLOCKED_PUBLIC_LOADER",
            champion=None,
            sealed=None,
            preflight_status=preflight.status,
        )
        (report_dir / "99_decision.md").write_text(summary, encoding="utf-8")
        return PipelineResult("BLOCKED_PUBLIC_LOADER", report_dir, None, None)

    start = pd.Timestamp(config.research_start)
    end = pd.Timestamp(config.research_end)
    cache_paths = build_monthly_sample_cache(
        loader,
        config,
        start=start,
        end=end,
        force_rebuild=force_rebuild_cache,
        progress=progress,
    )
    if not cache_paths:
        cache_paths = list_cached_months(config)
    _write_json(report_dir / "02_sample_cache_manifest.json", {"files": [str(path) for path in cache_paths]})

    folds = default_folds(config)
    _write_json(report_dir / "03_walk_forward_folds.json", [fold.to_dict() for fold in folds])
    total_jobs = len(folds) * len(models) * len(config.horizons_seconds)
    reporter = ProgressReporter("[R01 models]", total_jobs, every=1, enabled=progress)
    prediction_rows: list[dict[str, object]] = []
    scenario_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    job_index = 0

    for fold in folds:
        for horizon in config.horizons_seconds:
            fitted = fit_model_set(models, cache_paths, fold, horizon, config)
            for model_name in models:
                model, metadata = fitted[model_name]
                thresholds = calibration_thresholds(model, cache_paths, fold, horizon, config)
                model_dir = report_dir / "models" / fold.fold_id / model_name / f"h{horizon}"
                save_model_bundle(model, metadata, thresholds, model_dir)
                importance = feature_importance_frame(model, feature_columns(config))
                importance.to_csv(model_dir / "feature_importance.csv", index=False)

                evaluation: FoldEvaluation = evaluate_model_on_fold(
                    model=model,
                    model_name=model_name,
                    thresholds=thresholds,
                    paths=cache_paths,
                    fold=fold,
                    horizon=horizon,
                    config=config,
                )
                prediction_row = {
                    "fold_id": fold.fold_id,
                    "sealed": fold.sealed,
                    "model": model_name,
                    "horizon_seconds": horizon,
                    **evaluation.prediction_metrics.to_dict(),
                }
                prediction_rows.append(prediction_row)
                scenario_frames.append(evaluation.scenario_summaries)
                if not evaluation.base_trades.empty:
                    trade_frames.append(evaluation.base_trades)
                job_index += 1
                reporter.update(job_index)
    reporter.close()

    prediction_df = pd.DataFrame(prediction_rows)
    scenario_df = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    prediction_df.to_csv(report_dir / "04_prediction_metrics.csv", index=False)
    scenario_df.to_csv(report_dir / "05_trade_stress_matrix.csv", index=False)
    if not trades_df.empty:
        trades_df.to_csv(report_dir / "06_base_scenario_trades.csv.gz", index=False, compression="gzip")

    champion = select_validation_champion(scenario_df, config)
    sealed_result: dict[str, object] | None = None
    decision = "FAIL_NO_VALIDATION_EDGE"
    if champion is not None:
        sealed_rows = scenario_df.loc[
            (scenario_df["fold_id"] == "WF_2026")
            & (scenario_df["model"] == champion["model"])
            & (scenario_df["horizon_seconds"] == champion["horizon_seconds"])
            & (scenario_df["quantile"] == champion["quantile"])
            & (scenario_df["latency_seconds"] == config.base_latency_seconds)
            & (scenario_df["cost_multiplier"] == 1.0)
        ]
        if not sealed_rows.empty:
            sealed_result = {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in sealed_rows.iloc[0].to_dict().items()
            }
            sealed_2x = scenario_df.loc[
                (scenario_df["fold_id"] == "WF_2026")
                & (scenario_df["model"] == champion["model"])
                & (scenario_df["horizon_seconds"] == champion["horizon_seconds"])
                & (scenario_df["quantile"] == champion["quantile"])
                & (scenario_df["latency_seconds"] == config.base_latency_seconds)
                & (scenario_df["cost_multiplier"] == 2.0)
            ]
            sealed_1s = scenario_df.loc[
                (scenario_df["fold_id"] == "WF_2026")
                & (scenario_df["model"] == champion["model"])
                & (scenario_df["horizon_seconds"] == champion["horizon_seconds"])
                & (scenario_df["quantile"] == champion["quantile"])
                & (scenario_df["latency_seconds"] == 1.0)
                & (scenario_df["cost_multiplier"] == 1.0)
            ]
            sealed_pass = (
                int(sealed_result["trades"]) >= 150
                and float(sealed_result["mean_net_return"]) > 0
                and float(sealed_result["profit_factor"]) > 1.05
                and float(sealed_result["top10_removed_total_return"]) > 0
                and not sealed_2x.empty
                and float(sealed_2x.iloc[0]["total_return"]) > 0
                and not sealed_1s.empty
                and float(sealed_1s.iloc[0]["total_return"]) > 0
            )
            decision = "PASS_TRADES_ONLY_EDGE" if sealed_pass else "FAIL_SEALED_HOLDOUT"

    _write_json(report_dir / "07_validation_champion.json", champion)
    _write_json(report_dir / "08_sealed_result.json", sealed_result)
    summary = _markdown_summary(
        config=config,
        decision=decision,
        champion=champion,
        sealed=sealed_result,
        preflight_status=preflight.status,
    )
    (report_dir / "99_decision.md").write_text(summary, encoding="utf-8")

    if champion is not None and not trades_df.empty:
        validation_trades = trades_df.loc[
            (trades_df["fold_id"] == "WF_2025")
            & (trades_df["model"] == champion["model"])
            & (trades_df["horizon_seconds"] == champion["horizon_seconds"])
            & (trades_df["quantile"] == champion["quantile"])
            & (trades_df["latency_seconds"] == config.base_latency_seconds)
            & (trades_df["cost_multiplier"] == 1.0)
        ]
        sealed_trades = trades_df.loc[
            (trades_df["fold_id"] == "WF_2026")
            & (trades_df["model"] == champion["model"])
            & (trades_df["horizon_seconds"] == champion["horizon_seconds"])
            & (trades_df["quantile"] == champion["quantile"])
            & (trades_df["latency_seconds"] == config.base_latency_seconds)
            & (trades_df["cost_multiplier"] == 1.0)
        ]
        breakdown = pd.concat(
            [
                _period_breakdown(validation_trades, "WF_2025_validation"),
                _period_breakdown(sealed_trades, "WF_2026_sealed"),
            ],
            ignore_index=True,
        )
        if not breakdown.empty:
            breakdown.to_csv(report_dir / "09_champion_period_breakdown.csv", index=False)
        _write_full_report(
            validation_trades,
            strategy_name="ETH_AI_R01_VALIDATION_CHAMPION",
            report_dir=report_dir / "full_reports",
            initial_capital=config.initial_capital,
        )
        _write_full_report(
            sealed_trades,
            strategy_name="ETH_AI_R01_SEALED_2026",
            report_dir=report_dir / "full_reports",
            initial_capital=config.initial_capital,
        )

    return PipelineResult(decision, report_dir, champion, sealed_result)
