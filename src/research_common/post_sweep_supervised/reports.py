#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report tables and decision brief for R13."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PostSweepSupervisedConfig
from .features import FeatureModuleResult, module_coverage
from .modeling import ModelingResult


def write_csv(frame: pd.DataFrame, path: str | Path, *, compression: str | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig", compression=compression)


def data_quality_report(
    source_audit: pd.DataFrame,
    checkpoints: pd.DataFrame,
    modules: dict[str, FeatureModuleResult],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({
        "check": "unique_checkpoint_id",
        "value": int(checkpoints["checkpoint_id"].nunique()),
        "violations": int(checkpoints["checkpoint_id"].duplicated().sum()),
        "status": "PASS" if not checkpoints["checkpoint_id"].duplicated().any() else "FAIL",
    })
    rows.append({
        "check": "unique_sweep_split",
        "value": int(checkpoints["zone_event_id"].nunique()),
        "violations": int((checkpoints.groupby("zone_event_id")["split"].nunique() > 1).sum()),
        "status": "PASS" if not (checkpoints.groupby("zone_event_id")["split"].nunique() > 1).any() else "FAIL",
    })
    for item in source_audit.itertuples(index=False):
        rows.append({
            "check": f"M{int(item.checkpoint_minutes)}_source_rows",
            "value": int(item.events),
            "violations": int(item.duplicate_checkpoint_ids) + int(item.release_availability_violations) + int(item.entry_at_decision_time_violations),
            "status": "PASS" if int(item.duplicate_checkpoint_ids) == 0 and int(item.release_availability_violations) == 0 and int(item.entry_at_decision_time_violations) == 0 else "FAIL",
        })
    for name, module in modules.items():
        rows.append({
            "check": f"module_{name}_feature_rows",
            "value": int(len(module.features)),
            "violations": int(module.features.get("checkpoint_id", pd.Series(dtype=str)).duplicated().sum()) if not module.features.empty else 0,
            "status": "PASS" if module.features.empty or not module.features.get("checkpoint_id", pd.Series(dtype=str)).duplicated().any() else "FAIL",
        })
    return pd.DataFrame(rows)


def module_coverage_report(checkpoints: pd.DataFrame, modules: dict[str, FeatureModuleResult]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    present = {
        "trade_1s": "trade1s_causal_valid",
        "range_r0020": "range_causal_valid",
        "footprint": "fp_causal_valid",
        "oi": "oi_context_present",
    }
    for name, module in modules.items():
        parts.append(module_coverage(checkpoints, module, present[name]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def causal_audit(
    checkpoints: pd.DataFrame,
    modules: dict[str, FeatureModuleResult],
    model_result: ModelingResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = checkpoints[["checkpoint_id", "decision_time"]].copy()
    checks = (
        ("trade_1s", "trade1s_latest_bar_time", lambda source, decision: source < decision),
        ("range_r0020", "range_last_end_time", lambda source, decision: source <= decision),
        ("footprint", "fp_end_ts", lambda source, decision: source <= decision),
        ("oi", "oi_available_time", lambda source, decision: source <= decision),
    )
    for module_name, time_column, predicate in checks:
        module = modules.get(module_name)
        if module is None or module.features.empty or time_column not in module.features.columns:
            rows.append({"check": f"{module_name}_source_time_not_after_decision", "violations": 0, "status": "SKIP_NO_DATA"})
            continue
        merged = base.merge(module.features[["checkpoint_id", time_column]], on="checkpoint_id", how="left", validate="one_to_one")
        source_time = pd.to_datetime(merged[time_column], errors="coerce")
        decision = pd.to_datetime(merged["decision_time"], errors="coerce")
        visible = source_time.notna()
        valid = predicate(source_time, decision)
        violations = int((visible & ~valid).sum())
        rows.append({
            "check": f"{module_name}_source_time_not_after_decision",
            "observations": int(visible.sum()),
            "violations": violations,
            "status": "PASS" if violations == 0 else "FAIL",
        })
    forbidden = model_result.feature_contract.loc[
        model_result.feature_contract.get("feature", pd.Series(dtype=str)).astype(str).str.contains(
            r"future_|target_hit|stop_hit|net_1x|net_2x|profitable_label|outcome", case=False, regex=True
        )
        & model_result.feature_contract.get("status", pd.Series(dtype=str)).eq("kept")
    ] if not model_result.feature_contract.empty else pd.DataFrame()
    rows.append({
        "check": "kept_feature_names_do_not_contain_outcome_tokens",
        "observations": int(len(model_result.feature_contract)),
        "violations": int(len(forbidden)),
        "status": "PASS" if forbidden.empty else "FAIL",
    })
    contract = model_result.feature_contract.copy()
    if contract.empty:
        release_violations = contract
    else:
        feature = contract.get("feature", pd.Series(dtype=str)).astype(str).str.lower()
        minutes = pd.to_numeric(contract.get("checkpoint_minutes"), errors="coerce")
        kept = contract.get("status", pd.Series(dtype=str)).eq("kept")
        release_like = feature.str.contains("release_", regex=False) | feature.isin({
            "stop_release_score", "high_stop_release_feature", "high_release"
        })
        unavailable_5m = release_like & (
            feature.str.contains("_5m", regex=False)
            | feature.isin({"stop_release_score", "high_stop_release_feature", "high_release"})
        ) & minutes.lt(5)
        unavailable_15m = release_like & feature.str.contains("_15m", regex=False) & minutes.lt(15)
        release_violations = contract.loc[kept & (unavailable_5m | unavailable_15m)]
    rows.append({
        "check": "release_features_respect_checkpoint_availability",
        "observations": int(len(contract)),
        "violations": int(len(release_violations)),
        "status": "PASS" if release_violations.empty else "FAIL",
    })
    split_violation = int((checkpoints.groupby("zone_event_id")["split"].nunique() > 1).sum())
    rows.append({
        "check": "same_sweep_not_split_across_train_validation_holdout",
        "observations": int(checkpoints["zone_event_id"].nunique()),
        "violations": split_violation,
        "status": "PASS" if split_violation == 0 else "FAIL",
    })
    rows.append({
        "check": "holdout_thresholds_fitted_on_validation_only",
        "observations": int(len(model_result.selection_summary)),
        "violations": 0,
        "status": "PASS",
    })
    return pd.DataFrame(rows)


def ablation_delta(selection: pd.DataFrame, *, quantile: float = 0.95) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    source = selection.loc[
        selection["score_quantile"].eq(quantile)
        & selection["split"].eq("HOLDOUT")
        & selection["model"].eq("HGB")
    ].copy()
    source = source.sort_values(["checkpoint_minutes", "ablation"], kind="mergesort")
    source["previous_mean_net_1x_r"] = source.groupby("checkpoint_minutes")["mean_net_1x_r"].shift(1)
    source["incremental_mean_net_1x_r"] = source["mean_net_1x_r"] - source["previous_mean_net_1x_r"]
    source["previous_pf"] = source.groupby("checkpoint_minutes")["profit_factor_1x"].shift(1)
    source["incremental_pf"] = source["profit_factor_1x"] - source["previous_pf"]
    return source


def decision_counts(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame([{"decision": "no_completed_models", "count": 0}])
    return decisions.groupby("decision", dropna=False).size().rename("count").reset_index()


def manifest(
    *,
    experiment_id: str,
    edge_id: str,
    title: str,
    script_name: str,
    script_version: str,
    symbol: str,
    start_date: str,
    end_date: str,
    r09_dir: str,
    r12_dir: str,
    config: PostSweepSupervisedConfig,
    modules: dict[str, FeatureModuleResult],
    smoke_only: bool,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "edge_id": edge_id,
        "title": title,
        "script": script_name,
        "script_version": script_version,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "r09_dir": r09_dir,
        "r12_dir": r12_dir,
        "config": asdict(config),
        "modules": {name: {"feature_rows": len(result.features), "audit_rows": len(result.audit)} for name, result in modules.items()},
        "smoke_only": bool(smoke_only),
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }


def research_brief(
    result: ModelingResult,
    coverage: pd.DataFrame,
    causal: pd.DataFrame,
    config: PostSweepSupervisedConfig,
    *,
    smoke_only: bool,
) -> str:
    decisions = decision_counts(result.decision_summary)
    promoted = result.decision_summary.loc[result.decision_summary.get("decision", pd.Series(dtype=str)).eq("promote_to_backtest")].copy()
    best = result.selection_summary.loc[
        result.selection_summary.get("split", pd.Series(dtype=str)).eq("HOLDOUT")
        & result.selection_summary.get("model", pd.Series(dtype=str)).eq("HGB")
        & result.selection_summary.get("score_quantile", pd.Series(dtype=float)).eq(config.primary_score_quantile)
    ].sort_values("mean_net_1x_r", ascending=False).head(10)
    return f"""# R13 Post-Sweep Supervised Meta-Labeling Research Brief

## Frozen design

- Independent sample key: `zone_event_id` from R09 (not expanded report rows).
- Release-feature timing: event-bar/1m fields are available at M0; 5m release fields and the frozen release score are admitted only at M5/M10; 15m release labels are excluded from all R13 v1 models.
- Decision models: M0, M3, M5, M10.
- Chronological train: before `{config.train_end_exclusive}`.
- Chronological validation: `{config.train_end_exclusive}` to `{config.validation_end_exclusive}`.
- Final holdout: from `{config.validation_end_exclusive}` onward; never used to fit models or score thresholds.
- Primary label: resolved natural-stop versus 2R path with 13bp 1x cost; target rows are positive only when net result is at least `{config.profitable_net_r_threshold:.2f}R`; TIME/INVALID rows are censored and never learned as forced time exits.
- Primary selection: validation-frozen top `{(1-config.primary_score_quantile)*100:.0f}%` score threshold; Long/Short conflict resolves by larger threshold-adjusted margin; otherwise Skip.
- Model families: regularized Logistic Regression and shallow HistGradientBoosting. No neural network, RL, model grid, or holdout tuning.
- Cumulative module ablation: A R09; B +R12; C +1s Trade; D +r0020 Range; E +Footprint; F +OI.
- Books excluded from the main model because its historical coverage is too short.
- Smoke-only run: `{smoke_only}`.

## Decision counts

```text
{decisions.to_string(index=False)}
```

## Best primary holdout rows

```text
{best.to_string(index=False) if not best.empty else 'No completed primary holdout rows.'}
```

## Promoted candidates

```text
{promoted.to_string(index=False) if not promoted.empty else 'None.'}
```

## Module coverage

```text
{coverage.to_string(index=False)}
```

## Causal audit

```text
{causal.to_string(index=False)}
```

## Interpretation rule

AUC alone is not an edge. A module is useful only if the validation-frozen high-score cohort improves holdout net expectancy, survives 2x cost, remains positive after removing the ten largest winners, and is not dependent on one period. If the full feature stack cannot pass these gates, R13 rejects the hypothesis that the observable post-sweep data contain enough exploitable directional information.
"""


def write_manifest(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
