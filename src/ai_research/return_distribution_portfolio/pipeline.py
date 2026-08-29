from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

from .config import DEFAULT_CONFIG, ReturnDistributionConfig
from .dataset import feature_columns, load_or_build_shards
from .modeling import evaluate_bundle, fit_quantile_bundle


@dataclass(frozen=True)
class PipelineResult:
    report_dir: Path
    metrics: pd.DataFrame


def _folds(shards: dict[int, pd.DataFrame]) -> list[tuple[str, list[int], int]]:
    years = sorted(shards)
    out = []
    for test_year in years:
        train_years = [year for year in years if year < test_year]
        if train_years and test_year >= 2024:
            out.append((f"WF_{test_year}", train_years, test_year))
    return out


def _write_target_summary(shards: dict[int, pd.DataFrame], config: ReturnDistributionConfig, path: Path) -> None:
    rows = []
    for year, frame in sorted(shards.items()):
        for horizon in config.horizons_minutes:
            col = f"ret_h{horizon}"
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            rows.append(
                {
                    "year": year,
                    "horizon_minutes": horizon,
                    "rows": len(s),
                    "mean": s.mean(),
                    "std": s.std(ddof=0),
                    "q10": s.quantile(0.10),
                    "q25": s.quantile(0.25),
                    "q50": s.quantile(0.50),
                    "q75": s.quantile(0.75),
                    "q90": s.quantile(0.90),
                    "positive_rate": (s > 0).mean(),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_pipeline(
    *,
    config: ReturnDistributionConfig = DEFAULT_CONFIG,
    data_dir: str | None = None,
    force_rebuild_shards: bool = False,
    progress: bool = True,
) -> PipelineResult:
    config.validate()
    report = config.report_path
    report.mkdir(parents=True, exist_ok=True)
    (report / "00_config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    shards = load_or_build_shards(config, data_dir=data_dir, force=force_rebuild_shards, progress=progress)
    _write_target_summary(shards, config, report / "01_target_distribution.csv")
    feat = feature_columns(next(iter(shards.values())))
    pd.DataFrame({"feature": feat}).to_csv(report / "02_feature_manifest.csv", index=False)

    fold_specs = _folds(shards)
    total_models = len(fold_specs) * len(config.horizons_minutes)
    reporter = ProgressReporter(label="[RDP][forecast]", total=total_models, every=1, enabled=progress)
    metric_rows: list[dict[str, object]] = []
    decile_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []

    for fold_id, train_years, test_year in fold_specs:
        train = pd.concat([shards[y] for y in train_years], axis=0).sort_index()
        test = shards[test_year]
        for horizon in config.horizons_minutes:
            bundle = fit_quantile_bundle(train, horizon, config)
            metrics, deciles, sample = evaluate_bundle(bundle, train, test, fold_id, config)
            metrics["train_years"] = ",".join(str(y) for y in train_years)
            metrics["test_year"] = test_year
            metric_rows.append(metrics)
            decile_parts.append(deciles)
            # Keep a bounded deterministic sample for human inspection; do not
            # write millions of prediction rows into the review pack.
            sample_parts.append(sample.iloc[:: max(1, len(sample) // 4000 + 1)])
            reporter.step()
    reporter.close()

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(report / "03_oos_distribution_metrics.csv", index=False)
    pd.concat(decile_parts, ignore_index=True).to_csv(report / "04_oos_decile_curve.csv", index=False)
    pd.concat(sample_parts, axis=0).to_csv(report / "05_oos_prediction_sample.csv")

    # Summary is descriptive only. No q70/q90 threshold is produced anywhere.
    summary_rows = []
    for horizon, grp in metrics_df.groupby("horizon_minutes", sort=True):
        summary_rows.append(
            {
                "horizon_minutes": int(horizon),
                "folds": int(len(grp)),
                "rank_ic_min": float(grp["rank_ic_q50"].min()),
                "rank_ic_median": float(grp["rank_ic_q50"].median()),
                "top_bottom_spread_min": float(grp["top_bottom_decile_spread"].min()),
                "top_bottom_spread_median": float(grp["top_bottom_decile_spread"].median()),
                "interval_coverage_median": float(grp["q10_q90_coverage"].median()),
                "crossing_rate_max": float(grp["quantile_crossing_rate"].max()),
                "q50_pinball_skill_min": float(grp["pinball_skill_q50"].min()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(report / "06_horizon_summary.csv", index=False)

    decision_text = """# RDP V1 Stage 01 decision\n\nThis stage does **not** create a trading strategy or an entry threshold.\n\nInterpretation order:\n1. Check whether directional q50 ranking is stable across chronological OOS folds.\n2. Check whether top-vs-bottom forecast deciles separate realized returns consistently.\n3. Check quantile calibration/coverage and pinball skill versus unconditional train distributions.\n4. Only horizons with persistent OOS information may proceed to continuous target-exposure mapping.\n5. Do not convert scores into q70/q90 event gates; that would recreate the archived failure mode.\n\nWF_2026 is chronological OOS for this code path, but it is **not** an untouched project-level sealed holdout because 2026 has already been inspected in earlier research. A new future paper/live shadow period is required before deployment.\n"""
    (report / "99_DECISION_GUIDE.md").write_text(decision_text, encoding="utf-8")
    return PipelineResult(report_dir=report, metrics=metrics_df)
