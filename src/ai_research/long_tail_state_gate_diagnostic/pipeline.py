#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.17 descriptive state/regime attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_forward_extension.config import DEFAULT_FORWARD_EXTENSION_CONFIG
from src.ai_research.long_tail_sealed_holdout import pipeline as sealed_pipeline
from src.ai_research.swing_baseline.dataset import build_yearly_cache, create_loader, run_public_loader_preflight
from src.ai_research.swing_long_context.config import FEATURE_PROFILE, LONG_CONTEXT_BASE_CONFIG
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches

from . import reports
from .analysis import (
    align_state,
    build_attribution_findings,
    build_state_timeline,
    counterfactual_gate_summary,
    monthly_market_vs_c2,
    summarize_c2_by_state,
    summarize_fixed6h_by_state,
    summarize_score_state,
)
from .config import DEFAULT_STATE_GATE_DIAGNOSTIC_CONFIG, STAGE_ID, STAGE_NAME, StateGateDiagnosticConfig


@dataclass(frozen=True)
class StateGateDiagnosticResult:
    decision: str
    report_dir: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _decision_contains(path: Path, token: str) -> bool:
    return token in path.read_text(encoding="utf-8")


def _validate_sources(config: StateGateDiagnosticConfig) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    contracts = (
        ("r03_4_2_15_pass", config.source_2_15_path / "99_decision.md", "PASS_FINAL_ACCOUNT_LIVE_READINESS"),
        ("r03_4_2_16_failed_seal", config.source_2_16_path / "99_decision.md", "FAIL_2026_SEALED_HOLDOUT"),
        ("r03_4_2_16_1_july_completed", config.source_2_16_1_path / "99_decision.md", "JULY_FORWARD_"),
    )
    for name, path, token in contracts:
        passed = path.exists() and _decision_contains(path, token)
        checks.append({"check": name, "pass": passed, "value": str(path), "required": token})
        if not passed:
            raise RuntimeError(f"source contract failed: {name}")
    for label, source in (("h1", config.source_2_16_path), ("july", config.source_2_16_1_path)):
        seal = _read_json(source / "18_post_run_seal_check.json")
        passed = str(seal.get("status")) == "PASS" and bool(seal.get("unchanged", False))
        checks.append({"check": f"{label}_seal_unchanged", "pass": passed, "value": seal.get("status"), "required": "PASS/unchanged"})
        if not passed:
            raise RuntimeError(f"{label} seal is not unchanged")
        failures = _read_csv(source / "20_failures.csv")
        passed = failures.empty
        checks.append({"check": f"{label}_runtime_failures_empty", "pass": passed, "value": int(len(failures)), "required": 0})
        if not passed:
            raise RuntimeError(f"{label} source contains runtime failures")
    h1_score = _read_csv(config.source_2_16_path / "03_model_threshold_audit.csv")
    july_score = _read_csv(config.source_2_16_1_path / "03_model_threshold_audit.csv")
    if len(h1_score) != 1 or len(july_score) != 1:
        raise RuntimeError("source threshold audits must each contain one row")
    for column in ("calibration_threshold", "fit_rows", "calibration_rows", "feature_schema_hash"):
        a, b = h1_score.iloc[0][column], july_score.iloc[0][column]
        if column == "calibration_threshold":
            passed = abs(float(a) - float(b)) <= 1e-12
        else:
            passed = str(a) == str(b)
        checks.append({"check": f"h1_july_{column}_match", "pass": passed, "value": a, "required": b})
        if not passed:
            raise RuntimeError(f"H1/July source drift in {column}")
    return pd.DataFrame(checks)


def _base_config(config: StateGateDiagnosticConfig):
    return replace(
        LONG_CONTEXT_BASE_CONFIG,
        research_end=config.analysis_end,
        cache_dir=config.source_2_16_base_cache_dir,
    )


def _frozen_scores(
    config: StateGateDiagnosticConfig,
    *,
    data_dir: str | Path | None,
    force_rebuild_base: bool,
    force_rebuild_outcomes: bool,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_config = DEFAULT_FORWARD_EXTENSION_CONFIG
    base_config = _base_config(config)
    loader = create_loader(base_config, data_dir=data_dir)
    base_paths = build_yearly_cache(
        loader,
        base_config,
        force_rebuild=force_rebuild_base,
        progress=progress,
        feature_profile=FEATURE_PROFILE,
    )
    outcome_config = sealed_pipeline._outcome_config(source_config)
    outcome_paths = build_outcome_caches(
        base_paths,
        outcome_config,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )
    fit = collect_base_period_data(
        base_paths, outcome_paths,
        start=pd.Timestamp(config.fit_start), end=config.fit_end, outcome_config=outcome_config,
    )
    calibration = collect_base_period_data(
        base_paths, outcome_paths,
        start=pd.Timestamp(config.calibration_start), end=pd.Timestamp(config.calibration_end), outcome_config=outcome_config,
    )
    test = collect_base_period_data(
        base_paths, outcome_paths,
        start=pd.Timestamp("2026-01-01 00:00:00"), end=pd.Timestamp(config.analysis_end), outcome_config=outcome_config,
    )
    exit_config = sealed_pipeline._exit_config(source_config)
    bundle = fit_frozen_base_scores(fit, calibration, test, exit_config)
    threshold = float(bundle.timeline.threshold(config.q70_quantile))
    source_audit = _read_csv(config.source_2_16_path / "03_model_threshold_audit.csv").iloc[0]
    if abs(threshold - float(source_audit["calibration_threshold"])) > 1e-12:
        raise RuntimeError("diagnostic threshold differs from sealed source")
    if int(len(fit.timestamps_ns)) != int(source_audit["fit_rows"]):
        raise RuntimeError("diagnostic fit rows differ from sealed source")
    if int(len(calibration.timestamps_ns)) != int(source_audit["calibration_rows"]):
        raise RuntimeError("diagnostic calibration rows differ from sealed source")
    if bundle.feature_schema_hash != str(source_audit["feature_schema_hash"]):
        raise RuntimeError("diagnostic feature schema differs from sealed source")
    calibration_frame = pd.DataFrame({
        "decision_time": pd.to_datetime(np.asarray(calibration.timestamps_ns, dtype=np.int64)),
        "score": np.asarray(bundle.calibration_score, dtype=float),
        "analysis_period": "CAL_Q4_2025",
    })
    test_times = pd.to_datetime(np.asarray(test.timestamps_ns, dtype=np.int64))
    test_frame = pd.DataFrame({"decision_time": test_times, "score": np.asarray(bundle.test_score, dtype=float)})
    test_frame["analysis_period"] = np.select(
        [
            test_frame["decision_time"] < pd.Timestamp("2026-04-01"),
            test_frame["decision_time"] < pd.Timestamp("2026-07-01"),
        ],
        ["2026_Q1", "2026_Q2"],
        default="2026_JULY",
    )
    scores = pd.concat([calibration_frame, test_frame], ignore_index=True)
    audit = pd.DataFrame([
        {
            "fit_rows": int(len(fit.timestamps_ns)),
            "calibration_rows": int(len(calibration.timestamps_ns)),
            "test_rows": int(len(test.timestamps_ns)),
            "calibration_threshold": threshold,
            "feature_schema_hash": bundle.feature_schema_hash,
            "source_threshold": float(source_audit["calibration_threshold"]),
            "source_feature_schema_hash": str(source_audit["feature_schema_hash"]),
            "exact_source_match": True,
        }
    ])
    return scores, audit


def _minute_frame(path) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": path.open, "high": path.high, "low": path.low, "close": path.close},
        index=path.index,
    )


def _load_anchor_cycles(config: StateGateDiagnosticConfig) -> pd.DataFrame:
    historical = _read_csv(config.source_2_15_path / "03_continuous_account_cycles.csv")
    historical = historical.loc[
        historical["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
        & np.isclose(historical["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
    ].copy()
    historical["analysis_period"] = historical["fold_id"].astype(str).replace({"WF_2024": "2024", "WF_2025": "2025"})
    h1 = _read_csv(config.source_2_16_path / "08_account_cycles.csv")
    h1 = h1.loc[
        h1["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
        & np.isclose(h1["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
    ].copy()
    h1["analysis_period"] = "2026_H1"
    july = _read_csv(config.source_2_16_1_path / "08_account_cycles.csv")
    july = july.loc[
        july["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
        & np.isclose(july["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
    ].copy()
    july["analysis_period"] = "2026_JULY"
    out = pd.concat([historical, h1, july], ignore_index=True, sort=False)
    out["decision_time"] = pd.to_datetime(out["decision_time"])
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["exit_time"] = pd.to_datetime(out["exit_time"])
    return out.sort_values("entry_time").reset_index(drop=True)




def _load_anchor_monthly_returns(config: StateGateDiagnosticConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    historical = _read_csv(config.source_2_15_path / "06_monthly_returns.csv")
    if not historical.empty:
        historical = historical.loc[
            historical["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
            & np.isclose(historical["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
        ].copy()
        historical = historical.rename(columns={"return": "c2_account_return"})
        frames.append(historical[["month", "c2_account_return"]])

    for source in (config.source_2_16_path, config.source_2_16_1_path):
        monthly = _read_csv(source / "13_monthly_returns.csv")
        if monthly.empty:
            continue
        monthly = monthly.loc[
            monthly["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
            & np.isclose(monthly["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
            & monthly["period_kind"].astype(str).eq("month")
        ].copy()
        monthly = monthly.rename(columns={"period": "month", "return": "c2_account_return"})
        frames.append(monthly[["month", "c2_account_return"]])

    if not frames:
        return pd.DataFrame(columns=["month", "c2_account_return"])
    out = pd.concat(frames, ignore_index=True)
    out["month"] = out["month"].astype(str)
    return out.drop_duplicates("month", keep="last").sort_values("month").reset_index(drop=True)


def _load_fixed6h(config: StateGateDiagnosticConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, period in ((config.source_2_16_path, "2026_H1"), (config.source_2_16_1_path, "2026_JULY")):
        frame = _read_csv(source / "04_fixed_6h_trades.csv")
        frame = frame.loc[frame["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)].copy()
        frame["analysis_period"] = period
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["decision_time"] = pd.to_datetime(out["decision_time"])
    return out




def _state_report_view(state: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decision_time", "tf4h_available_time", "tf1d_available_time",
        "trend_4h", "trend_1d", "combined_state", "drawdown_state", "vol_state",
        "above_1d_ema50", "tf4h_close_rel_ema20", "tf4h_ema20_rel_ema50",
        "tf4h_ema20_slope3", "tf4h_ret42", "tf4h_vol_ratio",
        "tf1d_close_rel_ema20", "tf1d_close_rel_ema50", "tf1d_ema20_rel_ema50",
        "tf1d_ema20_slope3", "tf1d_ret20", "tf1d_drawdown_90d",
        "state_ready", "context_available_time_flag",
    ]
    return state.loc[:, [column for column in columns if column in state.columns]].copy()


def _score_report_view(scores: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decision_time", "analysis_period", "score", "combined_state", "trend_1d",
        "trend_4h", "drawdown_state", "vol_state", "above_1d_ema50",
        "tf1d_close_rel_ema50", "tf1d_ret20", "tf1d_drawdown_90d",
        "tf4h_ret42", "tf4h_vol_ratio", "context_available_time_flag",
    ]
    return scores.loc[:, [column for column in columns if column in scores.columns]].copy()

def _causal_audit(state: pd.DataFrame, config: StateGateDiagnosticConfig) -> pd.DataFrame:
    return pd.DataFrame([
        {"check": "diagnostic_only", "status": "PASS", "detail": "R03.4.2.17 cannot promote a gate or overwrite the failed V1 seal"},
        {"check": "frozen_strategy_unchanged", "status": "PASS", "detail": "C2 trades are read from frozen source reports; no trade rule is resimulated or changed"},
        {"check": "completed_4h_context", "status": "PASS" if bool(state["context_available_time_flag"].all()) else "FAIL", "detail": "4h bars become available at bar_start+4h"},
        {"check": "completed_1d_context", "status": "PASS" if bool(state["context_available_time_flag"].all()) else "FAIL", "detail": "1d bars become available at bar_start+1d"},
        {"check": "score_recipe_exact_match", "status": "PASS", "detail": "fit/calibration rows, threshold and feature schema must exactly match R03.4.2.16"},
        {"check": "counterfactual_disclosure", "status": "PASS", "detail": "all gate tables are descriptive development evidence and require future untouched validation"},
        {"check": "no_2026_parameter_search", "status": "PASS", "detail": "only predeclared economic trend/drawdown/volatility states are reported; no grid search"},
        {"check": "v1_not_live", "status": "PASS", "detail": "FAIL_2026_SEALED_HOLDOUT remains binding regardless of diagnostic result"},
    ])


def _write_empty(
    config: StateGateDiagnosticConfig,
    *,
    decision: str,
    reason: str,
    preflight: dict[str, object],
    source_integrity: pd.DataFrame,
    failures: pd.DataFrame,
) -> StateGateDiagnosticResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_integrity=source_integrity,
        state_definition=reports.state_definition_frame(config),
        state_timeline=pd.DataFrame(),
        cycles=pd.DataFrame(),
        c2_state_summary=pd.DataFrame(),
        fixed_trades=pd.DataFrame(),
        fixed_state_summary=pd.DataFrame(),
        scores=pd.DataFrame(),
        score_summary=pd.DataFrame(),
        monthly=pd.DataFrame(),
        gate_summary=pd.DataFrame(),
        findings=pd.DataFrame(),
        model_audit=pd.DataFrame(),
        causal=pd.DataFrame(),
        failures=failures,
        decision=decision,
        reason=reason,
    )
    return StateGateDiagnosticResult(decision, config.report_path)


def run_state_gate_diagnostic(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_base: bool = False,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: StateGateDiagnosticConfig = DEFAULT_STATE_GATE_DIAGNOSTIC_CONFIG,
) -> StateGateDiagnosticResult:
    config.validate()
    failures: list[dict[str, object]] = []
    source_integrity = pd.DataFrame()
    try:
        source_integrity = _validate_sources(config)
    except Exception as exc:
        failures.append({"stage": "source", "error": f"{type(exc).__name__}: {exc}"})
        return _write_empty(
            config, decision="BLOCKED_SOURCE", reason="Required sealed source reports are missing or inconsistent.",
            preflight={}, source_integrity=source_integrity, failures=pd.DataFrame(failures),
        )

    base_config = _base_config(config)
    loader = create_loader(base_config, data_dir=data_dir)
    preflight_result = run_public_loader_preflight(
        loader, base_config, sample_dates=("2024-01-15", "2026-06-15", "2026-07-15"),
    )
    preflight = preflight_result.to_dict()
    if preflight_result.status != "PASS":
        return _write_empty(
            config, decision="BLOCKED_DATA", reason="Public 1m Trade Bar preflight failed; no regime result is interpreted.",
            preflight=preflight, source_integrity=source_integrity, failures=pd.DataFrame(),
        )

    try:
        exit_config = sealed_pipeline._exit_config(DEFAULT_FORWARD_EXTENSION_CONFIG)
        path = load_minute_path_data(
            start=pd.Timestamp(config.state_warmup_start),
            end=pd.Timestamp(config.analysis_end),
            data_dir=data_dir,
            config=exit_config,
            progress=progress,
        )
        minute = _minute_frame(path)
        state = build_state_timeline(minute, config)
        if state.empty or not bool(state["context_available_time_flag"].all()):
            raise RuntimeError("causal state timeline is empty or has unavailable context")

        cycles = align_state(_load_anchor_cycles(config), state)
        if cycles["combined_state"].isna().any():
            raise RuntimeError("some C2 cycles could not be aligned to market state")
        c2_state_summary = summarize_c2_by_state(cycles)

        fixed_trades = align_state(_load_fixed6h(config), state)
        fixed_state_summary = summarize_fixed6h_by_state(fixed_trades, config)

        score_frame, model_audit = _frozen_scores(
            config, data_dir=data_dir, force_rebuild_base=force_rebuild_base,
            force_rebuild_outcomes=force_rebuild_outcomes, progress=progress,
        )
        scores = align_state(score_frame, state)
        threshold = float(model_audit.iloc[0]["calibration_threshold"])
        score_summary = summarize_score_state(scores, threshold)
        monthly = monthly_market_vs_c2(
            minute,
            cycles,
            scores,
            state,
            threshold,
            account_monthly_returns=_load_anchor_monthly_returns(config),
        )
        gate_summary = counterfactual_gate_summary(cycles)
        findings, decision, reason = build_attribution_findings(
            c2_state_summary, fixed_state_summary, score_summary, gate_summary,
        )
        causal = _causal_audit(state, config)
        if not causal["status"].astype(str).eq("PASS").all():
            raise RuntimeError("causal audit failed")
    except Exception as exc:
        failures.append({"stage": "diagnostic", "error": f"{type(exc).__name__}: {exc}"})
        return _write_empty(
            config, decision="FAIL_RUNTIME", reason="State-gate diagnostic failed; incomplete outputs must not be interpreted.",
            preflight=preflight, source_integrity=source_integrity, failures=pd.DataFrame(failures),
        )

    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_integrity=source_integrity,
        state_definition=reports.state_definition_frame(config),
        state_timeline=_state_report_view(state),
        cycles=cycles,
        c2_state_summary=c2_state_summary,
        fixed_trades=fixed_trades,
        fixed_state_summary=fixed_state_summary,
        scores=_score_report_view(scores),
        score_summary=score_summary,
        monthly=monthly,
        gate_summary=gate_summary,
        findings=findings,
        model_audit=model_audit,
        causal=causal,
        failures=pd.DataFrame(),
        decision=decision,
        reason=reason,
    )
    return StateGateDiagnosticResult(decision, config.report_path)
