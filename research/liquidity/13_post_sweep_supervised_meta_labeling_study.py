#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R13 post-sweep supervised meta-labeling study.

The study treats R09's 18k structured liquidity sweeps as the independent event
key and combines R09 static/release context with R12 causal post-sweep dynamics.
It then adds 1-second Trade, r0020 Range, Range Footprint, and Binance OI modules
one at a time.  Models may choose Long, Short, or Skip; the final holdout is
never used to fit a model or score threshold.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.post_sweep_supervised import (  # noqa: E402
    FeatureModuleResult,
    PostSweepSupervisedConfig,
    ablation_delta,
    build_footprint_features,
    build_oi_features,
    build_range_features,
    build_trade_1s_features,
    cache_module,
    causal_audit,
    checkpoint_index,
    data_quality_report,
    decision_counts,
    load_cached_module,
    load_r13_data,
    manifest,
    module_coverage_report,
    research_brief,
    run_supervised_ablation,
    write_csv,
    write_manifest,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "13_post_sweep_supervised_meta_labeling_study"
SCRIPT_VERSION = "1.0.2"
EXPERIMENT_ID = "ETH_POST_SWEEP_SUPERVISED_META_LABELING_R13"
EDGE_ID = "RESEARCH_ONLY_POST_SWEEP_LONG_SHORT_SKIP_MODEL"
TITLE = "ETH Post-Sweep Supervised Meta-Labeling R13"
DEFAULT_R09_DIR = "data/reports/research/liquidity/structured_swing_stop_pool_hypotheses_r09"
DEFAULT_R12_DIR = "data/reports/research/liquidity/12_post_sweep_rejection_acceptance_r12"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/13_post_sweep_supervised_meta_labeling_r13"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--r09-dir", default=DEFAULT_R09_DIR)
    p.add_argument("--r12-dir", default=DEFAULT_R12_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--trade-db-name", default="okx_trade_bars.db")
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--footprint-db-name", default="okx_range_footprints.db")
    p.add_argument("--oi-symbol", default="ETHUSDT")
    p.add_argument("--oi-data-dir", default=None)
    p.add_argument("--oi-db-name", default="binance_futures_metrics.db")
    p.add_argument("--max-events", type=int, default=0, help="Time-stratified smoke cap; 0 uses all R09 sweeps.")
    p.add_argument("--disable-trade-1s", action="store_true")
    p.add_argument("--disable-range", action="store_true")
    p.add_argument("--disable-footprint", action="store_true")
    p.add_argument("--disable-oi", action="store_true")
    p.add_argument("--rebuild-feature-cache", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _empty_module(name: str, checkpoints: pd.DataFrame, present_column: str, reason: str) -> FeatureModuleResult:
    features = pd.DataFrame({"checkpoint_id": checkpoints["checkpoint_id"].astype(str), present_column: False})
    audit = pd.DataFrame([{"events": len(checkpoints), "coverage": 0.0, "status": "disabled", "reason": reason}])
    return FeatureModuleResult(name, features, audit)


def _cache_matches(module: FeatureModuleResult, checkpoints: pd.DataFrame) -> bool:
    if module.features.empty or "checkpoint_id" not in module.features.columns:
        return False
    expected = set(checkpoints["checkpoint_id"].astype(str))
    actual = set(module.features["checkpoint_id"].astype(str))
    return expected == actual and not module.features["checkpoint_id"].duplicated().any()


def _load_or_build(
    name: str,
    *,
    cache_dir: Path,
    checkpoints: pd.DataFrame,
    rebuild: bool,
    builder,
) -> FeatureModuleResult:
    if not rebuild:
        cached = load_cached_module(name, cache_dir)
        if cached is not None and _cache_matches(cached, checkpoints):
            print(f"[cache] {name} rows={len(cached.features):,}", flush=True)
            return cached
    result = builder()
    cache_module(result, cache_dir)
    return result


def _module_audit_table(modules: dict[str, FeatureModuleResult]) -> pd.DataFrame:
    parts = []
    for name, module in modules.items():
        audit = module.audit.copy()
        audit.insert(0, "module", name)
        parts.append(audit)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _design_table(config: PostSweepSupervisedConfig) -> pd.DataFrame:
    rows = [
        {"section": "sample", "item": "independent_group", "value": "R09 zone_event_id"},
        {"section": "decision", "item": "checkpoints_minutes", "value": str(config.checkpoints_minutes)},
        {"section": "label", "item": "primary_path", "value": "natural structural stop vs 2R; 180m censoring window"},
        {"section": "label", "item": "unresolved_path", "value": "TIME/INVALID rows unlabeled; no forced time-exit learning"},
        {"section": "label", "item": "positive", "value": f"net_1x_r >= {config.profitable_net_r_threshold:.2f}R"},
        {"section": "split", "item": "train_end_exclusive", "value": config.train_end_exclusive},
        {"section": "split", "item": "validation_end_exclusive", "value": config.validation_end_exclusive},
        {"section": "split", "item": "holdout", "value": "2025-10-01 through requested end"},
        {"section": "models", "item": "families", "value": "regularized Logistic + shallow HistGradientBoosting"},
        {"section": "selection", "item": "primary_quantile", "value": config.primary_score_quantile},
        {"section": "selection", "item": "action", "value": "LONG / SHORT / SKIP"},
        {"section": "features", "item": "ablation", "value": "A R09; B +R12; C +1s Trade; D +r0020 Range; E +Footprint; F +OI"},
        {"section": "features", "item": "books", "value": "excluded from main model because coverage starts 2025-11"},
        {"section": "cost", "item": "labels", "value": "R12 13bp 1x and 26bp 2x"},
        {"section": "anti_overfit", "item": "forbidden", "value": "random split, holdout tuning, model grid, checkpoint mixing, feature-combination mining"},
    ]
    return pd.DataFrame(rows)


def _synthetic_reports(root: Path, events: int = 900) -> tuple[Path, Path]:
    r09 = root / "r09"
    r12 = root / "r12"
    r09.mkdir(parents=True, exist_ok=True)
    r12.mkdir(parents=True, exist_ok=True)
    # Time-stratified events ensure every split is represented.
    dates = pd.to_datetime(np.concatenate([
        pd.date_range("2023-01-01", "2024-12-30", periods=events // 2).to_numpy(),
        pd.date_range("2025-01-01", "2025-09-29", periods=events // 4).to_numpy(),
        pd.date_range("2025-10-01", "2026-06-20", periods=events - 3 * (events // 4)).to_numpy(),
    ])).sort_values()
    n = len(dates)
    ids = pd.Series([f"S{i:06d}" for i in range(n)])
    signal = np.sin(np.arange(n) / 11.0) + (np.arange(n) % 7) / 10.0
    feature = pd.DataFrame({
        "zone_event_id": ids,
        "event_kind": "swing_zone_sweep",
        "event_bar_time": dates - pd.Timedelta(minutes=1),
        "event_available_time": dates,
        "zone_primary_timeframe": np.where(np.arange(n) % 2, "15m", "1H"),
        "zone_timeframe_count": 1 + (np.arange(n) % 3),
        "zone_width_bp": 2.0 + np.abs(signal),
        "sweep_depth_below_floor_bp": 3.0 + signal,
        "pre_return_60m": signal / 100.0,
        "current_delta_ratio": -signal / 5.0,
    })
    long_win = signal > 0.45
    gross = np.where(long_win, 0.0015, -0.0015)
    r09_label = pd.DataFrame({
        "zone_event_id": ids,
        "r09_entry_time": dates,
        "tp15_sl15_outcome": np.where(long_win, "TP", "SL"),
        "tp15_sl15_same_bar_both_flag": False,
        "tp15_sl15_gross_return": gross,
        "tp15_sl15_net_return_1x_cost": gross - 0.0013,
        "tp15_sl15_net_return_2x_cost": gross - 0.0026,
        "release_sell_share_1m": 0.5 + signal / 20.0,
        "release_negative_delta_ratio_1m": np.clip(signal / 4.0, 0, None),
        "stop_release_score": signal,
        "high_stop_release_label": signal > 0.5,
    })
    feature.to_csv(r09 / "16_zone_feature_table.csv.gz", index=False, compression="gzip")
    r09_label.to_csv(r09 / "17_zone_label_table.csv.gz", index=False, compression="gzip")

    checkpoint_rows = []
    outcome_rows = []
    for minutes in (1, 3, 5, 10):
        decision = dates + pd.Timedelta(minutes=minutes)
        dyn = signal + minutes / 20.0
        cp = feature.copy()
        cp["checkpoint_minutes"] = minutes
        cp["checkpoint_available_time"] = decision
        cp["entry_time"] = decision
        cp["state"] = np.where(dyn > 0.5, "PRESSURE_TEST_REJECT", "PERSISTENT_ACCEPT")
        cp["state_direction"] = np.where(dyn > 0.5, "LONG", "SHORT")
        cp["close_vs_floor_bp"] = dyn * 5
        cp["post_close_below_floor_share"] = np.clip(0.5 - dyn / 10.0, 0, 1)
        cp["post_delta_notional_sum"] = -dyn * 1_000_000
        cp["post_notional_sum"] = 5_000_000
        cp["checkpoint_close"] = 100 + dyn
        cp["sweep_bar_close"] = 100.0
        checkpoint_rows.append(cp)
        for direction in ("LONG", "SHORT"):
            is_long = direction == "LONG"
            favorable = (dyn > 0.5) if is_long else (dyn <= 0.5)
            net = np.where(favorable, 0.8, -1.4)
            outcome_rows.append(pd.DataFrame({
                "zone_event_id": ids,
                "checkpoint_minutes": minutes,
                "checkpoint_available_time": decision,
                "entry_time": decision,
                "trade_direction": direction,
                "natural_stop_distance_bp": 30.0,
                "r2p0_outcome": np.where(favorable, "TARGET", "STOP"),
                "r2p0_target_before_stop": favorable,
                "r2p0_gross_r": np.where(favorable, 2.0, -1.0),
                "r2p0_net_1x_r": net,
                "r2p0_net_2x_r": net - 0.4,
            }))
    pd.concat(checkpoint_rows, ignore_index=True).to_csv(r12 / "14_checkpoint_feature_table.csv.gz", index=False, compression="gzip")
    pd.concat(outcome_rows, ignore_index=True).to_csv(r12 / "15_outcome_label_table.csv.gz", index=False, compression="gzip")
    return r09, r12


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="r13_selftest_") as temp:
        root = Path(temp)
        r09, r12 = _synthetic_reports(root)
        cfg = replace(
            PostSweepSupervisedConfig(),
            minimum_train_events=100,
            minimum_validation_events=50,
            minimum_holdout_events=50,
            minimum_holdout_trades=5,
            hgb_min_samples_leaf=20,
            hgb_max_iter=60,
        ).validate()
        bundle = load_r13_data(r09, r12, cfg)
        checkpoints = checkpoint_index(bundle.datasets)
        modules = {
            "trade_1s": _empty_module("trade_1s", checkpoints, "trade1s_causal_valid", "selftest"),
            "range_r0020": _empty_module("range_r0020", checkpoints, "range_causal_valid", "selftest"),
            "footprint": _empty_module("footprint", checkpoints, "fp_causal_valid", "selftest"),
            "oi": _empty_module("oi", checkpoints, "oi_context_present", "selftest"),
        }
        result = run_supervised_ablation(bundle.datasets, bundle.base_columns, bundle.dynamic_columns, modules, cfg)
        complete = result.model_summary.loc[result.model_summary.get("status", pd.Series(dtype=str)).eq("COMPLETE")]
        if complete.empty:
            raise RuntimeError("R13 self-test produced no complete models")
        if result.selection_summary.empty:
            raise RuntimeError("R13 self-test produced no selection summary")
        quality = data_quality_report(bundle.audit, checkpoints, modules)
        if quality["status"].eq("FAIL").any():
            raise RuntimeError(f"R13 self-test quality failure:\n{quality.to_string(index=False)}")
    print("[self-test] PASS", flush=True)


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.self_test:
        run_self_test()
        return Path(args.out_dir)
    smoke_only = int(args.max_events) > 0
    cfg = PostSweepSupervisedConfig().validate()
    if smoke_only:
        per_split = max(20, int(args.max_events) // 3)
        cfg = replace(
            cfg,
            minimum_train_events=max(20, per_split // 3),
            minimum_validation_events=max(20, per_split // 3),
            minimum_holdout_events=max(20, per_split // 3),
            minimum_holdout_trades=max(5, int(args.max_events) // 100),
            hgb_min_samples_leaf=max(10, min(50, int(args.max_events) // 30)),
            hgb_max_iter=80,
        ).validate()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache" / (f"max_{int(args.max_events)}" if smoke_only else "full")
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[design] R09 event key + R12 causal path + module ablation; chronological split; LONG/SHORT/SKIP; holdout untouched", flush=True)
    print(f"[window] research={args.start_date}->{args.end_date} train<2025-01-01 validation<2025-10-01 holdout>=2025-10-01", flush=True)

    print("[stage] load R09/R12 report caches and build independent checkpoint datasets", flush=True)
    bundle = load_r13_data(
        PROJECT_ROOT / args.r09_dir,
        PROJECT_ROOT / args.r12_dir,
        cfg,
        max_events=int(args.max_events),
    )
    checkpoints = checkpoint_index(bundle.datasets)
    print(f"[universe] checkpoints={len(checkpoints):,} sweeps={checkpoints['zone_event_id'].nunique():,} by_M={checkpoints['checkpoint_minutes'].value_counts().sort_index().to_dict()}", flush=True)

    progress = not bool(args.no_progress)
    modules: dict[str, FeatureModuleResult] = {}
    if args.disable_trade_1s:
        modules["trade_1s"] = _empty_module("trade_1s", checkpoints, "trade1s_causal_valid", "disabled_by_cli")
    else:
        modules["trade_1s"] = _load_or_build(
            "trade_1s", cache_dir=cache_dir, checkpoints=checkpoints, rebuild=args.rebuild_feature_cache,
            builder=lambda: build_trade_1s_features(
                checkpoints, symbol=args.symbol, data_dir=args.data_dir, db_name=args.trade_db_name,
                config=cfg, progress=progress,
            ),
        )
    if args.disable_range:
        modules["range_r0020"] = _empty_module("range_r0020", checkpoints, "range_causal_valid", "disabled_by_cli")
    else:
        modules["range_r0020"] = _load_or_build(
            "range_r0020", cache_dir=cache_dir, checkpoints=checkpoints, rebuild=args.rebuild_feature_cache,
            builder=lambda: build_range_features(
                checkpoints, symbol=args.symbol, data_dir=args.data_dir, db_name=args.range_db_name,
                config=cfg, progress=progress,
            ),
        )
    if args.disable_footprint:
        modules["footprint"] = _empty_module("footprint", checkpoints, "fp_causal_valid", "disabled_by_cli")
    else:
        modules["footprint"] = _load_or_build(
            "footprint", cache_dir=cache_dir, checkpoints=checkpoints, rebuild=args.rebuild_feature_cache,
            builder=lambda: build_footprint_features(
                checkpoints, symbol=args.symbol, data_dir=args.data_dir,
                range_db_name=args.range_db_name, footprint_db_name=args.footprint_db_name,
                config=cfg, progress=progress,
            ),
        )
    if args.disable_oi:
        modules["oi"] = _empty_module("oi", checkpoints, "oi_context_present", "disabled_by_cli")
    else:
        lookup = pd.concat(bundle.datasets.values(), ignore_index=True, sort=False).drop_duplicates("checkpoint_id")
        modules["oi"] = _load_or_build(
            "oi", cache_dir=cache_dir, checkpoints=checkpoints, rebuild=args.rebuild_feature_cache,
            builder=lambda: build_oi_features(
                checkpoints, lookup, symbol=args.oi_symbol,
                data_dir=args.oi_data_dir or args.data_dir, db_name=args.oi_db_name,
            ),
        )

    quality = data_quality_report(bundle.audit, checkpoints, modules)
    failures = quality.loc[quality["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(f"R13 data-quality fail-fast gate failed:\n{failures.to_string(index=False)}")
    coverage = module_coverage_report(checkpoints, modules)
    print("[stage] train fixed Logistic/HGB models and validation-frozen score thresholds", flush=True)
    model_reporter = ProgressReporter("[r13] model ablation", total=46, every=1, enabled=progress)
    def model_progress(done: int, total: int, minutes: int, ablation: str, model: str) -> None:
        model_reporter.total = total
        model_reporter.update(done)
    modeling = run_supervised_ablation(
        bundle.datasets, bundle.base_columns, bundle.dynamic_columns, modules, cfg,
        progress_callback=model_progress,
    )
    model_reporter.close()
    print("[stage] causal audit and leakage gates", flush=True)
    causal = causal_audit(checkpoints, modules, modeling)
    causal_failures = causal.loc[causal["status"].eq("FAIL")]
    if not causal_failures.empty:
        raise RuntimeError(f"R13 causal fail-fast gate failed:\n{causal_failures.to_string(index=False)}")

    print("[stage] write compact reports and review artifacts", flush=True)
    write_manifest(out_dir / "00_manifest.json", manifest(
        experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE,
        script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
        symbol=args.symbol, start_date=args.start_date, end_date=args.end_date,
        r09_dir=args.r09_dir, r12_dir=args.r12_dir, config=cfg, modules=modules,
        smoke_only=smoke_only,
    ))
    write_csv(quality, out_dir / "01_data_quality.csv")
    write_csv(_design_table(cfg), out_dir / "02_frozen_design.csv")
    write_csv(bundle.audit, out_dir / "03_source_dataset_audit.csv")
    write_csv(coverage, out_dir / "04_module_coverage.csv")
    write_csv(_module_audit_table(modules), out_dir / "05_module_build_audit.csv")
    write_csv(modeling.model_summary, out_dir / "06_model_classification_summary.csv")
    write_csv(modeling.selection_summary, out_dir / "07_high_score_trade_selection.csv")
    write_csv(ablation_delta(modeling.selection_summary, quantile=cfg.primary_score_quantile), out_dir / "08_ablation_incremental_value.csv")
    write_csv(modeling.score_deciles, out_dir / "09_score_decile_monotonicity.csv")
    write_csv(modeling.decision_summary, out_dir / "10_candidate_scorecard.csv")
    write_csv(modeling.feature_contract, out_dir / "11_feature_contract.csv")
    write_csv(causal, out_dir / "12_causal_audit.csv")
    write_csv(modeling.prediction_sample, out_dir / "13_prediction_sample.csv.gz", compression="gzip")
    brief = research_brief(modeling, coverage, causal, cfg, smoke_only=smoke_only)
    (out_dir / "15_research_brief.md").write_text(brief, encoding="utf-8")
    sample_columns = list(dict.fromkeys([
        "checkpoint_id", "zone_event_id", "checkpoint_minutes", "decision_time", "split",
        "long_profitable_label", "short_profitable_label",
        "long_gross_r", "long_net_1x_r", "long_net_2x_r",
        "short_gross_r", "short_net_1x_r", "short_net_2x_r",
        *[name for columns in bundle.base_columns.values() for name in columns],
        *[name for columns in bundle.dynamic_columns.values() for name in columns],
    ]))
    sample_parts = []
    per_checkpoint = max(1, int(cfg.sample_rows) // max(1, len(bundle.datasets)))
    for minutes, frame in sorted(bundle.datasets.items()):
        available = [name for name in sample_columns if name in frame.columns]
        sample_parts.append(frame.loc[:, available].head(per_checkpoint))
    dataset_sample = pd.concat(sample_parts, ignore_index=True, sort=False).head(cfg.sample_rows)
    write_csv(dataset_sample, out_dir / "14_model_dataset_sample.csv.gz", compression="gzip")
    del dataset_sample, sample_parts

    review_path = None
    if not args.skip_review_pack:
        review = finalize_research_report(
            out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE,
        )
        review_path = review.zip_path
        print(f"[done] review_pack={review_path}", flush=True)
    counts = decision_counts(modeling.decision_summary)
    print(f"[decision-summary] {counts.to_dict(orient='records')}", flush=True)
    print(f"[done] report={out_dir} elapsed={time.perf_counter()-started:.1f}s smoke_only={smoke_only}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
