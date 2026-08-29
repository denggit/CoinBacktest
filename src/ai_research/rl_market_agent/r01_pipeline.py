#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01: direct opportunity-model -> executable ETH strategy walk-forward."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

from .config import PROJECT_ROOT
from .dataset import DatasetCatalog
from .opportunity import (
    FixedOpportunityRegressor,
    conservative_template_returns,
    feature_groups,
    prediction_quality,
    required_label_names,
)
from .r01_config import R01Config, WalkForwardFold
from .sources import SourceRepository
from .splits import LoadedWindow, load_purged_window, make_purged_window
from .strategy import evaluate_trades, replay_strategy, selection_key


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _matrix_columns(window: LoadedWindow, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    lookup = {name: i for i, name in enumerate(window.feature_names)}
    idx = np.asarray([lookup[name] for name in names], dtype=np.int64)
    x = np.asarray(window.features[:, idx], dtype=np.float32)
    # Training-only hygiene: discard nearly absent or constant features.  This
    # prevents optional-source calendar missingness from becoming an accidental
    # year classifier. The same retained columns are then frozen for cal/OOS.
    finite = np.isfinite(x)
    coverage = finite.mean(axis=0)
    with np.errstate(all="ignore"):
        std = np.nanstd(x, axis=0)
    keep = (coverage >= 0.95) & np.isfinite(std) & (std > 1e-12)
    kept_names = tuple(name for name, flag in zip(names, keep) if flag)
    if not kept_names:
        raise ValueError("feature group has no sufficiently covered non-constant training features")
    return x[:, keep], kept_names


def _select_columns(window: LoadedWindow, names: tuple[str, ...]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(window.feature_names)}
    return np.asarray(window.features[:, [lookup[name] for name in names]], dtype=np.float32)


def _threshold(pred: np.ndarray, quantile: float) -> float:
    finite = np.asarray(pred, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("inf")
    return float(max(0.0, np.nanquantile(finite, float(quantile))))


def _candidate_pass(metrics: dict[str, float | int], config: R01Config) -> bool:
    return bool(
        int(metrics["trades"]) >= int(config.min_calibration_trades)
        and float(metrics["total_return_pct"]) > 0.0
        and float(metrics["cagr_pct"]) > 0.0
        and float(metrics["profit_factor"]) > 1.0
        and float(metrics["max_drawdown_pct"]) <= float(config.calibration_max_drawdown_pct)
    )


def _monthly_summary(trades: pd.DataFrame, cost: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["month", "trades", "return_pct", "win_rate"])
    frame = trades.copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"])
    frame["equity_return"] = (
        pd.to_numeric(frame["gross_price_return"], errors="coerce") - float(cost)
    ) * pd.to_numeric(frame["notional_multiple"], errors="coerce")
    frame["month"] = frame["exit_time"].dt.to_period("M").astype(str)
    rows = []
    for month, g in frame.groupby("month", sort=True):
        r = g["equity_return"].fillna(0.0).to_numpy(dtype=float)
        rows.append({
            "month": month,
            "trades": int(len(g)),
            "return_pct": float((np.prod(1.0 + r) - 1.0) * 100.0),
            "win_rate": float((r > 0).mean()),
        })
    return pd.DataFrame(rows)


def _load_fold_path(repo: SourceRepository, fold: WalkForwardFold) -> pd.DataFrame:
    start = pd.Timestamp(fold.calibration_start)
    end = pd.Timestamp(fold.oos_end_exclusive) - pd.Timedelta(minutes=1)
    return repo.load_trade_bars("1m", start, end)


def _fold_windows(fold: WalkForwardFold, horizon: int, seal: str) -> tuple:
    train = make_purged_window(f"{fold.name}_TRAIN", fold.train_start, fold.train_end_exclusive, horizon)
    cal = make_purged_window(f"{fold.name}_CAL", fold.calibration_start, fold.calibration_end_exclusive, horizon)
    oos = make_purged_window(f"{fold.name}_OOS", fold.oos_start, fold.oos_end_exclusive, horizon)
    if pd.Timestamp(oos.end_exclusive) > pd.Timestamp(seal):
        raise PermissionError(f"{fold.name} reaches sealed holdout")
    return train, cal, oos



def _zero_top_winners(trades: pd.DataFrame, *, cost: float, top_n: int) -> pd.DataFrame:
    if trades is None or trades.empty or top_n <= 0:
        return pd.DataFrame() if trades is None else trades.copy()
    frame = trades.copy()
    net_equity = (
        pd.to_numeric(frame["gross_price_return"], errors="coerce") - float(cost)
    ) * pd.to_numeric(frame["notional_multiple"], errors="coerce")
    winners = net_equity[net_equity > 0].nlargest(int(top_n)).index
    if len(winners):
        # Preserve occupancy/timing but neutralize the selected winner PnL.
        frame.loc[winners, "gross_price_return"] = float(cost)
    return frame


def _trade_breakdown(trades: pd.DataFrame, *, cost: float) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["dimension", "value", "trades", "mean_equity_return", "win_rate"])
    frame = trades.copy()
    frame["equity_return"] = (
        pd.to_numeric(frame["gross_price_return"], errors="coerce") - float(cost)
    ) * pd.to_numeric(frame["notional_multiple"], errors="coerce")
    rows = []
    for dim in ("side", "exit_reason", "fold"):
        if dim not in frame.columns:
            continue
        for value, g in frame.groupby(dim, sort=True):
            r = g["equity_return"].to_numpy(dtype=float)
            rows.append({
                "dimension": dim, "value": value, "trades": int(len(g)),
                "mean_equity_return": float(np.nanmean(r)) if len(r) else np.nan,
                "win_rate": float(np.mean(r > 0)) if len(r) else np.nan,
            })
    return pd.DataFrame(rows)

def run_r01(
    config: R01Config,
    *,
    data_dir: str | Path | None = None,
    repository: SourceRepository | None = None,
    finalize_report: bool = True,
) -> dict[str, Any]:
    config.validate()
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir = report_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    catalog = DatasetCatalog(config.cache_path, project_root=PROJECT_ROOT, allow_sealed=False)
    shard_ids = catalog.shard_ids()
    if not shard_ids:
        raise FileNotFoundError(f"R00 cache has no shards: {config.cache_path}")
    first = catalog.load(shard_ids[0])
    groups = feature_groups(first.feature_names)
    repo = repository or SourceRepository(symbol=config.symbol, data_dir=data_dir)

    fold_manifest: list[dict[str, Any]] = []
    model_quality_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    all_oos_trades: list[pd.DataFrame] = []
    delayed_oos_trades: dict[int, list[pd.DataFrame]] = {int(x): [] for x in config.delay_stress_minutes}

    total_jobs = len(config.folds) * len(config.trade_templates) * len(groups)
    progress = ProgressReporter("[R01 models]", total=total_jobs, every=1)
    job = 0

    for fold in config.folds:
        print(f"[fold] {fold.name} load 1m replay path", flush=True)
        path = _load_fold_path(repo, fold)
        fold_candidates: list[dict[str, Any]] = []
        runtime_models: dict[str, tuple[Any, Any, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]] = {}

        for template in config.trade_templates:
            train_w, cal_w, oos_w = _fold_windows(fold, template.horizon_minutes, config.sealed_holdout_start)
            needed_labels = required_label_names(template)
            train = load_purged_window(catalog, train_w, label_names=needed_labels, sealed_holdout_start=config.sealed_holdout_start)
            cal = load_purged_window(catalog, cal_w, label_names=needed_labels, sealed_holdout_start=config.sealed_holdout_start)
            oos = load_purged_window(catalog, oos_w, label_names=needed_labels, sealed_holdout_start=config.sealed_holdout_start)
            for w, loaded in ((train_w, train), (cal_w, cal), (oos_w, oos)):
                fold_manifest.append({
                    "fold": fold.name,
                    "template": template.name,
                    "window": loaded.name,
                    "horizon_minutes": template.horizon_minutes,
                    "start": str(w.start),
                    "end_exclusive": str(w.end_exclusive),
                    "last_safe_decision": str(w.last_safe_decision),
                    "rows_before_purge": loaded.rows_before_purge,
                    "rows_after_purge": loaded.rows_after_purge,
                    "purged_rows": loaded.rows_before_purge - loaded.rows_after_purge,
                })
            train_long, train_short = conservative_template_returns(
                train.labels, train.label_names, template, round_trip_cost=config.round_trip_cost
            )
            cal_long_actual, cal_short_actual = conservative_template_returns(
                cal.labels, cal.label_names, template, round_trip_cost=config.round_trip_cost
            )
            oos_long_actual, oos_short_actual = conservative_template_returns(
                oos.labels, oos.label_names, template, round_trip_cost=config.round_trip_cost
            )

            for group_name, raw_group_names in groups.items():
                job += 1
                x_train, kept_names = _matrix_columns(train, raw_group_names)
                x_cal = _select_columns(cal, kept_names)
                x_oos = _select_columns(oos, kept_names)
                long_model = FixedOpportunityRegressor(random_state=config.random_state).fit(x_train, train_long)
                short_model = FixedOpportunityRegressor(random_state=config.random_state + 1).fit(x_train, train_short)
                cal_long_pred = long_model.predict(x_cal)
                cal_short_pred = short_model.predict(x_cal)
                oos_long_pred = long_model.predict(x_oos)
                oos_short_pred = short_model.predict(x_oos)
                model_id = f"{fold.name}__{template.name}__{group_name}"
                runtime_models[model_id] = (
                    long_model, short_model, kept_names,
                    np.asarray(oos.timestamps_ns, dtype=np.int64), oos_long_pred, oos_short_pred,
                )

                for split_name, lp, sp, la, sa in (
                    ("CAL", cal_long_pred, cal_short_pred, cal_long_actual, cal_short_actual),
                    ("OOS", oos_long_pred, oos_short_pred, oos_long_actual, oos_short_actual),
                ):
                    for side, pred, actual in (("LONG", lp, la), ("SHORT", sp, sa)):
                        q = prediction_quality(pred, actual)
                        model_quality_rows.append({
                            "fold": fold.name, "model_id": model_id, "template": template.name,
                            "feature_group": group_name, "backend": long_model.backend,
                            "feature_count": len(kept_names), "split": split_name, "side": side, **q,
                        })

                for quantile in config.threshold_quantiles:
                    lthr = _threshold(cal_long_pred, quantile)
                    sthr = _threshold(cal_short_pred, quantile)
                    cal_trades = replay_strategy(
                        decision_times_ns=cal.timestamps_ns,
                        long_scores=cal_long_pred,
                        short_scores=cal_short_pred,
                        path_1m=path,
                        template=template,
                        long_threshold=lthr,
                        short_threshold=sthr,
                        round_trip_cost=config.round_trip_cost,
                        risk_per_trade=config.risk_per_trade,
                        max_notional_multiple=config.max_notional_multiple,
                    )
                    metrics = evaluate_trades(
                        cal_trades,
                        start=fold.calibration_start,
                        end_exclusive=fold.calibration_end_exclusive,
                        round_trip_cost=config.round_trip_cost,
                    )
                    passed = _candidate_pass(metrics, config)
                    candidate_id = f"{model_id}__Q{int(round(quantile * 100)):02d}"
                    row = {
                        "fold": fold.name, "candidate_id": candidate_id, "model_id": model_id,
                        "template": template.name, "horizon_minutes": template.horizon_minutes,
                        "take_profit": template.take_profit, "stop_loss": template.stop_loss,
                        "feature_group": group_name, "feature_count": len(kept_names),
                        "threshold_quantile": quantile, "long_threshold": lthr, "short_threshold": sthr,
                        "calibration_pass": passed, **metrics,
                    }
                    calibration_rows.append(row)
                    fold_candidates.append(row)
                progress.update(job)

        passing = [row for row in fold_candidates if row["calibration_pass"]]
        if passing:
            selected = min(passing, key=lambda row: selection_key(row))
            selection_status = "PASSING_CALIBRATION_SELECTION"
        else:
            # Diagnostic fallback remains selected exclusively on pre-OOS calibration.
            # It is never relabelled as a valid candidate.
            selected = max(
                fold_candidates,
                key=lambda row: (float(row["total_return_pct"]), float(row["profit_factor"]), int(row["trades"])),
            )
            selection_status = "FALLBACK_NO_CALIBRATION_PASS"

        model_id = selected["model_id"]
        long_model, short_model, kept_names, oos_timestamps_ns, oos_lp, oos_sp = runtime_models[model_id]
        template = next(t for t in config.trade_templates if t.name == selected["template"])
        oos_trades = pd.DataFrame()
        for delay in config.delay_stress_minutes:
            delayed = replay_strategy(
                decision_times_ns=oos_timestamps_ns,
                long_scores=oos_lp,
                short_scores=oos_sp,
                path_1m=path,
                template=template,
                long_threshold=float(selected["long_threshold"]),
                short_threshold=float(selected["short_threshold"]),
                round_trip_cost=config.round_trip_cost,
                risk_per_trade=config.risk_per_trade,
                max_notional_multiple=config.max_notional_multiple,
                entry_delay_minutes=int(delay),
            )
            if not delayed.empty:
                delayed.insert(0, "fold", fold.name)
                delayed.insert(1, "candidate_id", selected["candidate_id"])
                delayed.insert(2, "entry_delay_minutes", int(delay))
                delayed_oos_trades[int(delay)].append(delayed)
            if int(delay) == 0:
                oos_trades = delayed
                if not oos_trades.empty:
                    all_oos_trades.append(oos_trades)
        metrics = evaluate_trades(
            oos_trades,
            start=fold.oos_start,
            end_exclusive=fold.oos_end_exclusive,
            round_trip_cost=config.round_trip_cost,
        )
        oos_rows.append({
            "fold": fold.name,
            "selection_status": selection_status,
            "candidate_id": selected["candidate_id"],
            "template": selected["template"],
            "feature_group": selected["feature_group"],
            "threshold_quantile": selected["threshold_quantile"],
            **metrics,
        })
        long_path = long_model.save(model_dir / f"{fold.name}_long")
        short_path = short_model.save(model_dir / f"{fold.name}_short")
        selected_rows.append({
            "fold": fold.name,
            "selection_status": selection_status,
            **{k: selected[k] for k in (
                "candidate_id", "model_id", "template", "feature_group", "feature_count",
                "threshold_quantile", "long_threshold", "short_threshold",
            )},
            "long_model_path": long_path,
            "short_model_path": short_path,
        })
        for side, model in (("LONG", long_model), ("SHORT", short_model)):
            imp = model.feature_importance(kept_names).head(50)
            if not imp.empty:
                imp.insert(0, "fold", fold.name)
                imp.insert(1, "side", side)
                imp.insert(2, "candidate_id", selected["candidate_id"])
                importance_rows.extend(imp.to_dict("records"))
    progress.close()

    trades = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    combined_start = min(pd.Timestamp(f.oos_start) for f in config.folds)
    combined_end = max(pd.Timestamp(f.oos_end_exclusive) for f in config.folds)
    cost_rows = []
    for mult in config.cost_stress_multipliers:
        metrics = evaluate_trades(
            trades,
            start=combined_start,
            end_exclusive=combined_end,
            round_trip_cost=config.round_trip_cost * float(mult),
        )
        cost_rows.append({"cost_multiplier": float(mult), "round_trip_cost": config.round_trip_cost * float(mult), **metrics})
    delay_rows = []
    for delay in config.delay_stress_minutes:
        delayed_trades = pd.concat(delayed_oos_trades[int(delay)], ignore_index=True) if delayed_oos_trades[int(delay)] else pd.DataFrame()
        metrics = evaluate_trades(
            delayed_trades, start=combined_start, end_exclusive=combined_end,
            round_trip_cost=config.round_trip_cost,
        )
        delay_rows.append({"entry_delay_minutes": int(delay), **metrics})

    top_removal_rows = []
    for top_n in config.top_trade_removal_counts:
        stressed = _zero_top_winners(trades, cost=config.round_trip_cost, top_n=int(top_n))
        metrics = evaluate_trades(
            stressed, start=combined_start, end_exclusive=combined_end,
            round_trip_cost=config.round_trip_cost,
        )
        top_removal_rows.append({"top_winners_neutralized": int(top_n), **metrics})

    base = cost_rows[0]
    cost2 = next((x for x in cost_rows if x["cost_multiplier"] == 2.0), None)
    fold_df = pd.DataFrame(oos_rows)
    every_year_positive = bool(len(fold_df) and (fold_df["total_return_pct"] > 0).all())
    all_folds_passing_calibration = bool(selected_rows and all(x["selection_status"] == "PASSING_CALIBRATION_SELECTION" for x in selected_rows))
    robust_2x = bool(cost2 is not None and float(cost2["total_return_pct"]) > 0)
    if (
        all_folds_passing_calibration
        and every_year_positive
        and float(base["total_return_pct"]) > 0
        and float(base["max_drawdown_pct"]) <= 20.0
        and int(base["trades"]) >= 100
        and robust_2x
    ):
        decision = "PASS_R01_STRATEGY_CANDIDATE"
    elif float(base["total_return_pct"]) > 0:
        decision = "PROMISING_BUT_NOT_R01_PASS"
    else:
        decision = "NO_TRADABLE_STRATEGY_R01"

    _write_json(report_dir / "00_config.json", config.to_dict())
    pd.DataFrame(fold_manifest).to_csv(report_dir / "01_purged_fold_manifest.csv", index=False)
    pd.DataFrame(model_quality_rows).to_csv(report_dir / "02_model_quality_secondary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(report_dir / "03_calibration_strategy_candidates.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(report_dir / "04_selected_fold_strategies.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(report_dir / "05_oos_fold_results.csv", index=False)
    trades.to_csv(report_dir / "06_oos_trades.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(report_dir / "07_cost_stress.csv", index=False)
    _monthly_summary(trades, config.round_trip_cost).to_csv(report_dir / "08_oos_monthly.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(report_dir / "09_selected_feature_importance.csv", index=False)
    pd.DataFrame(delay_rows).to_csv(report_dir / "10_delay_stress.csv", index=False)
    pd.DataFrame(top_removal_rows).to_csv(report_dir / "11_top_trade_removal_stress.csv", index=False)
    _trade_breakdown(trades, cost=config.round_trip_cost).to_csv(report_dir / "12_trade_breakdown.csv", index=False)
    _write_json(report_dir / "13_environment.json", {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "sealed_holdout_opened": False,
    })
    summary = f"""# R01 Opportunity Model + Executable Strategy Backtest\n\n- Decision: **{decision}**\n- 2026 sealed holdout opened: **NO**\n- Walk-forward OOS: 2024 and 2025 only.\n- Every model/threshold/template is selected only from its pre-OOS training + calibration history.\n- Horizon-aware purge is applied at train/calibration/OOS right boundaries, including the 2025 -> 2026 seal.\n- Base round-trip cost: {config.round_trip_cost:.3%}; 2x/3x cost stress is reported without changing signals.\n- Strategy is one non-overlapping ETH sleeve; entry is decision-time 1m open; TP/SL replay uses exact 1m OHLC; same-minute TP+SL is adverse-first.\n- Position size targets {config.risk_per_trade:.2%} equity risk including base cost and is capped at {config.max_notional_multiple:.2f}x notional.\n\n## Combined OOS base metrics\n- Trades: {int(base['trades'])}\n- Max flat days: {float(base['max_flat_days']):.3f}\n- Max consecutive losing days: {int(base['max_consecutive_losing_days'])}\n- Max drawdown: {float(base['max_drawdown_pct']):.2f}%\n- CAGR: {float(base['cagr_pct']):.2f}%\n- Total return: {float(base['total_return_pct']):.2f}%\n- Profit factor: {float(base['profit_factor']):.3f}\n\n## Champion priority\nAfter profitability/risk feasibility gates: max flat days -> max consecutive losing days -> MDD -> CAGR -> total return.\n\nModel-quality IC/lift is intentionally secondary; R01 succeeds only through the strategy backtest.\n"""
    (report_dir / "99_decision.md").write_text(summary, encoding="utf-8")
    if finalize_report:
        finalize_research_report(report_dir, title="ETH RL Market Agent V1 - R01 Opportunity Strategy")
    return {
        "decision": decision,
        "selected": selected_rows,
        "oos_results": oos_rows,
        "cost_stress": cost_rows,
        "delay_stress": delay_rows,
        "top_trade_removal_stress": top_removal_rows,
        "report_dir": str(report_dir),
    }


def config_with_overrides(config: R01Config, **kwargs: Any) -> R01Config:
    return replace(config, **{k: v for k, v in kwargs.items() if v is not None})
