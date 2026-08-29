#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R00 causal state/opportunity dataset builder."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.market_state.causal_alignment import timeframe_to_timedelta

try:
    from config.loader import TIMEZONE
except ImportError:
    TIMEZONE = "+8"
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

from .causal import align_available_events, align_left_labeled_bars, causal_audit, make_decision_index
from .config import PROJECT_ROOT, RLMarketAgentConfig
from .contracts import CausalAuditRecord, SourceCoverageRecord, rows_to_dicts
from .features import (
    add_time_since_event_feature,
    build_fixed_bar_features,
    resample_ohlcv_from_1m_bars,
    build_footprint_event_features,
    build_range_event_features,
    build_trade_bar_features,
    summarize_footprint_bars,
)
from .labels import build_forward_path_labels, label_names, label_specs
from .schema import build_feature_specs, range_code
from .shards import ShardStore
from .sources import SourceRepository


_RANGE_COLUMNS = [
    "bar_id", "start_ts", "end_ts", "duration_seconds", "direction", "notional",
    "delta_notional", "large_delta_notional", "trades_count", "taker_buy_ratio", "max_trade_notional",
]
_FOOTPRINT_COLUMNS = [
    "bar_id", "start_ts", "end_ts", "price_bucket", "notional", "delta_notional",
    "buy_notional", "sell_notional", "large_delta_notional", "max_trade_notional",
]


def _local_timezone_offset() -> pd.Timedelta:
    raw = str(TIMEZONE).strip()
    try:
        if raw.startswith("+"):
            return pd.Timedelta(hours=float(raw[1:]))
        if raw.startswith("-"):
            return -pd.Timedelta(hours=float(raw[1:]))
        return pd.Timedelta(hours=float(raw))
    except (TypeError, ValueError):
        return pd.Timedelta(0)


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    first = pd.Timestamp(start).to_period("M")
    last = pd.Timestamp(end).to_period("M")
    out = []
    for period in pd.period_range(first, last, freq="M"):
        a = max(pd.Timestamp(start), period.start_time)
        b = min(pd.Timestamp(end), period.end_time)
        out.append((a, b, period.strftime("%Y-%m")))
    return out


def _required(config: RLMarketAgentConfig, source: str) -> bool:
    if source == "trade_1m":
        return config.require_core_trade_bars
    if source.startswith("kline_"):
        return config.require_all_kline_context
    if source == "trade_5s":
        return config.require_micro_trade_bars
    if source.startswith("range_"):
        return config.require_range_bars
    if source.startswith("footprint_"):
        return config.require_footprint
    return False


def _coverage(shard_id: str, source: str, available: pd.Series, expected: int, required: bool) -> SourceCoverageRecord:
    valid = pd.to_datetime(available, errors="coerce").dropna()
    ratio = float(len(valid) / expected) if expected else 0.0
    status = "OK" if ratio >= 0.99 else ("PARTIAL" if len(valid) else "MISSING")
    return SourceCoverageRecord(
        shard_id=shard_id,
        source=source,
        expected_rows=int(expected),
        available_rows=int(len(valid)),
        coverage_ratio=ratio,
        first_available_time=None if valid.empty else str(valid.min()),
        last_available_time=None if valid.empty else str(valid.max()),
        required=required,
        status=status,
        note="required source below 99%" if required and ratio < 0.99 else "",
    )


def _audit_record(shard_id: str, source: str, decision: pd.DatetimeIndex, available: pd.Series) -> CausalAuditRecord:
    result = causal_audit(decision, available)
    return CausalAuditRecord(shard_id=shard_id, source=source, **result)


def _safe_source(load, empty: pd.DataFrame | None = None):
    try:
        return load(), ""
    except Exception as exc:  # source absence is reported, never auto-built
        return (pd.DataFrame() if empty is None else empty), f"{type(exc).__name__}: {exc}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_reports(
    *, report_dir: Path, config: RLMarketAgentConfig, feature_specs, records: list[dict[str, Any]],
    coverages: list[dict[str, Any]], audits: list[dict[str, Any]], source_errors: list[dict[str, str]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "00_config.json", config.to_dict())
    _write_json(report_dir / "01_feature_schema.json", [x.to_dict() for x in feature_specs])
    _write_json(report_dir / "02_label_schema.json", [x.to_dict() for x in label_specs(config.label_horizons_minutes)])
    pd.DataFrame(records).to_csv(report_dir / "03_shard_manifest.csv", index=False)
    pd.DataFrame(coverages).to_csv(report_dir / "04_source_coverage.csv", index=False)
    pd.DataFrame(audits).to_csv(report_dir / "05_causal_audit.csv", index=False)
    pd.DataFrame(source_errors).to_csv(report_dir / "06_source_errors.csv", index=False)

    audit_df = pd.DataFrame(audits)
    cov_df = pd.DataFrame(coverages)
    failed_audit = int((~audit_df["passed"].astype(bool)).sum()) if not audit_df.empty else 0
    required_bad = int(((cov_df.get("required", False) == True) & (cov_df.get("coverage_ratio", 0.0) < 0.99)).sum()) if not cov_df.empty else 0
    decision = "PASS_R00" if failed_audit == 0 and required_bad == 0 else "BLOCKED_DATA_OR_CAUSAL"
    summary = f"""# R00 Causal State Dataset Audit\n\n- Decision: **{decision}**\n- Shards: {len(records)}\n- Feature columns: {len(feature_specs)}\n- Label columns: {len(label_names(config.label_horizons_minutes))}\n- Causal-audit failures: {failed_audit}\n- Required source coverage failures (<99%): {required_bad}\n- 2026-01-01 onward is sealed holdout; R00 may materialize outcomes but later training/tuning must not consume it.\n- Raw data end: {config.research_end}; last decision time is label-safe at {config.decision_end}.\n- R00 trains no model and places no trades.\n\n## Frozen selection priority for later policy comparison\n1. max flat days (lower)\n2. max consecutive losing days (lower)\n3. max drawdown (lower)\n4. CAGR (higher)\n5. total return (higher)\n\n## Cost contract\nRound-trip base cost: {config.round_trip_fee_rate:.4%}; later policy validation must also test {', '.join(f'{x:g}x' for x in config.cost_stress_multipliers)}.\n"""
    (report_dir / "99_decision.md").write_text(summary, encoding="utf-8")


def run_r00(
    config: RLMarketAgentConfig,
    *,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
    audit_only: bool = False,
    max_shards: int | None = None,
    repository: SourceRepository | None = None,
    finalize_report: bool = True,
) -> dict[str, Any]:
    config.validate()
    report_dir = config.report_path
    cache_dir = config.cache_path
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo = repository or SourceRepository(symbol=config.symbol, data_dir=data_dir)
    store = ShardStore(cache_dir, project_root=PROJECT_ROOT)
    feature_specs = build_feature_specs(config)
    feature_names = [x.name for x in feature_specs]
    # The final max-label-horizon tail is data-only context, not a decision row.
    # This guarantees every persisted row has a fully observable 360m label.
    months = _months(pd.Timestamp(config.research_start), config.decision_end)
    if max_shards is not None:
        months = months[: max(0, int(max_shards))]

    records: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_errors: list[dict[str, str]] = []

    progress = ProgressReporter("[R00 shards]", total=len(months), every=1)
    for idx, (month_start, month_end, shard_id) in enumerate(months, start=1):
        if not overwrite and not audit_only and store.exists(shard_id):
            meta = store.load_metadata(shard_id)
            records.append(meta["record"])
            coverage_rows.extend(meta.get("coverage", []))
            audit_rows.extend(meta.get("causal_audits", []))
            source_errors.extend(meta.get("source_errors", []))
            progress.update(idx)
            continue

        decision = make_decision_index(month_start, month_end, config.decision_interval)
        query_start = max(pd.Timestamp(config.warmup_start), month_start - config.micro_context)
        # Fixed-bar features use up to 48 bars of history.  The 1D context is
        # therefore given a conservative 60-day local 1m-Kline history window.
        fixed_context = max(timeframe_to_timedelta(tf) * 60 for tf in config.kline_timeframes)
        fixed_query_start = max(pd.Timestamp(config.warmup_start), month_start - fixed_context)
        core_query_start = min(query_start, fixed_query_start)
        query_end = min(pd.Timestamp(config.research_end), month_end + config.max_label_horizon)
        pieces: list[pd.DataFrame] = []
        flags = pd.DataFrame(index=decision)
        shard_cov: list[dict[str, Any]] = []
        shard_audits: list[dict[str, Any]] = []
        shard_errors: list[dict[str, str]] = []

        def register(source: str, aligned, required: bool, *, note: str = "") -> None:
            cov = _coverage(shard_id, source, aligned.source_available_time, len(decision), required)
            if note:
                payload = cov.to_dict()
                payload["note"] = note if not payload.get("note") else f"{payload['note']}; {note}"
            else:
                payload = cov.to_dict()
            audit = _audit_record(shard_id, source, decision, aligned.source_available_time)
            shard_cov.append(payload)
            shard_audits.append(audit.to_dict())
            flags[f"available__{source}"] = aligned.source_available_time.notna().astype(np.uint8)
            f = aligned.features.copy()
            f[f"availability__{source}"] = aligned.source_available_time.notna().astype(float).to_numpy()
            pieces.append(f)

        # 1m tick-derived core is mandatory for flow state and forward-path labels.
        # It stays independent from the official K-line context. Read enough left
        # context for state features and through the largest label horizon.
        core_raw, error = _safe_source(lambda: repo.load_trade_bars(config.trade_bar_timeframe, core_query_start, query_end))
        if error:
            shard_errors.append({"shard_id": shard_id, "source": "trade_1m", "error": error})
        state_core = core_raw.loc[core_raw.index <= month_end] if not core_raw.empty else core_raw

        # Fixed K-line context is derived from one locally prebuilt official 1m
        # K-line source.  Do not read independent HTF K-line tables here: they can
        # have different refresh dates even after the 1m prebuild is current.
        # Trade bars remain an independent tick-derived source and are never used
        # to patch K-line context.
        official_1m, kline_error = _safe_source(
            lambda: repo.load_kline("1m", fixed_query_start, month_end)
        )
        if kline_error:
            shard_errors.append({"shard_id": shard_id, "source": "kline_1m_base", "error": kline_error})
        for tf in config.kline_timeframes:
            source = f"kline_{tf.lower()}"
            raw = (
                resample_ohlcv_from_1m_bars(
                    official_1m, timeframe=tf, daily_offset=_local_timezone_offset()
                )
                if not official_1m.empty else pd.DataFrame()
            )
            features = build_fixed_bar_features(raw, prefix=source) if not raw.empty else pd.DataFrame()
            aligned = align_left_labeled_bars(
                decision, features, bar_duration=timeframe_to_timedelta(tf),
                tolerance=timeframe_to_timedelta(tf) * 3,
            )
            register(source, aligned, _required(config, source), note="official_1m_resampled")

        # 1m tick-derived order-flow state and forward path labels.
        trade_windows = [pd.Timedelta(minutes=x) for x in config.trade_windows_minutes]
        trade_features = build_trade_bar_features(state_core, prefix="trade_1m", windows=trade_windows) if not state_core.empty else pd.DataFrame()
        aligned_core = align_left_labeled_bars(decision, trade_features, bar_duration="1min", tolerance="3min")
        register("trade_1m", aligned_core, _required(config, "trade_1m"))
        labels = build_forward_path_labels(core_raw, decision, config.label_horizons_minutes) if not core_raw.empty else pd.DataFrame(index=decision, columns=label_names(config.label_horizons_minutes))

        # 5s microstructure enrichment.
        micro_raw, error = _safe_source(lambda: repo.load_trade_bars(config.micro_trade_bar_timeframe, query_start, month_end))
        if error:
            shard_errors.append({"shard_id": shard_id, "source": "trade_5s", "error": error})
        micro_windows = [pd.Timedelta(seconds=x) for x in config.micro_windows_seconds]
        micro_features = build_trade_bar_features(micro_raw, prefix="trade_5s", windows=micro_windows) if not micro_raw.empty else pd.DataFrame()
        aligned_micro = align_left_labeled_bars(decision, micro_features, bar_duration="5s", tolerance="30s")
        register("trade_5s", aligned_micro, _required(config, "trade_5s"))

        # Closed range-bar event streams.
        range_windows = [pd.Timedelta(minutes=x) for x in config.range_windows_minutes]
        for rp in config.range_pcts:
            source = f"range_{range_code(rp)}"
            raw, error = _safe_source(lambda rp=rp: repo.load_range_bars(rp, query_start, month_end, columns=_RANGE_COLUMNS))
            if error:
                shard_errors.append({"shard_id": shard_id, "source": source, "error": error})
            event = build_range_event_features(raw, prefix=source, windows=range_windows) if not raw.empty else pd.DataFrame()
            aligned = align_available_events(decision, event)
            aligned = type(aligned)(
                add_time_since_event_feature(aligned.features, decision_index=decision, source_available_time=aligned.source_available_time, name=f"{source}__time_since_last_event_log_seconds"),
                aligned.source_available_time,
            )
            register(source, aligned, _required(config, source))

        # Closed r0.20 footprint summaries.
        fp_source = f"footprint_{range_code(config.footprint_range_pct)}"
        fp_raw, error = _safe_source(lambda: repo.load_footprint(config.footprint_range_pct, config.footprint_price_step, query_start, month_end, columns=_FOOTPRINT_COLUMNS))
        if error:
            shard_errors.append({"shard_id": shard_id, "source": fp_source, "error": error})
        fp_summary = summarize_footprint_bars(fp_raw, prefix=fp_source) if not fp_raw.empty else pd.DataFrame()
        fp_event = build_footprint_event_features(fp_summary, prefix=fp_source) if not fp_summary.empty else pd.DataFrame()
        fp_aligned = align_available_events(decision, fp_event)
        fp_aligned = type(fp_aligned)(
            add_time_since_event_feature(fp_aligned.features, decision_index=decision, source_available_time=fp_aligned.source_available_time, name=f"{fp_source}__time_since_last_event_log_seconds"),
            fp_aligned.source_available_time,
        )
        register(fp_source, fp_aligned, _required(config, fp_source))

        features = pd.concat(pieces, axis=1).reindex(index=decision, columns=feature_names)
        labels = labels.reindex(index=decision, columns=label_names(config.label_horizons_minutes))
        flags["sealed_holdout"] = (decision >= pd.Timestamp(config.sealed_holdout_start)).astype(np.uint8)
        flags["core_valid"] = (flags.get("available__trade_1m", 0).astype(bool) & labels["entry_price"].notna()).astype(np.uint8)

        # Required data and causal violations are hard gates for R00 validity.
        bad_required = [x for x in shard_cov if x["required"] and x["coverage_ratio"] < 0.99]
        bad_causal = [x for x in shard_audits if not x["passed"]]
        if bad_causal:
            raise RuntimeError(f"causal audit failed for {shard_id}: {bad_causal}")
        if bad_required:
            raise RuntimeError(f"required source coverage failed for {shard_id}: {bad_required}")

        core_mask = flags["core_valid"].astype(bool)
        features = features.loc[core_mask]
        labels = labels.loc[core_mask]
        flags = flags.loc[core_mask]
        record = {
            "shard_id": shard_id,
            "start_time": str(features.index.min()) if len(features) else "",
            "end_time": str(features.index.max()) if len(features) else "",
            "rows": int(len(features)),
            "feature_count": len(feature_names),
            "label_count": len(labels.columns),
            "sealed_holdout": bool(len(features) and features.index.min() >= pd.Timestamp(config.sealed_holdout_start)),
            "audit_only": bool(audit_only),
        }
        if not audit_only:
            saved = store.write(
                shard_id=shard_id,
                features=features,
                labels=labels,
                flags=flags,
                sealed_holdout=record["sealed_holdout"],
                extra_metadata={
                    "feature_names": feature_names,
                    "label_names": list(labels.columns),
                    "flag_names": list(flags.columns),
                    "coverage": shard_cov,
                    "causal_audits": shard_audits,
                    "source_errors": shard_errors,
                },
            )
            record = saved.to_dict()
        records.append(record)
        coverage_rows.extend(shard_cov)
        audit_rows.extend(shard_audits)
        source_errors.extend(shard_errors)
        progress.update(idx)
    progress.close()

    _write_reports(
        report_dir=report_dir,
        config=config,
        feature_specs=feature_specs,
        records=records,
        coverages=coverage_rows,
        audits=audit_rows,
        source_errors=source_errors,
    )
    _write_json(report_dir / "07_environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "project_root": str(PROJECT_ROOT),
        "cache_format": "npy float32/int64/uint8; mmap-compatible; no pyarrow dependency",
    })
    if finalize_report:
        finalize_research_report(report_dir, title="ETH RL Market Agent V1 - R00 Causal State Dataset")
    return {
        "records": records,
        "coverage": coverage_rows,
        "causal_audits": audit_rows,
        "source_errors": source_errors,
        "report_dir": str(report_dir),
        "cache_dir": str(cache_dir),
    }


def config_with_overrides(config: RLMarketAgentConfig, **kwargs: Any) -> RLMarketAgentConfig:
    return replace(config, **{k: v for k, v in kwargs.items() if v is not None})
