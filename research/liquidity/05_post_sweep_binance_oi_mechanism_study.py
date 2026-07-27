#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R05 Binance OI enrichment for post-Swing-zone-sweep state transitions.

This is a mechanism study, not a final strategy backtest. It causally aligns
Binance USD-M ETHUSDT 5-minute metrics to the R04 post-sweep checkpoints and
asks whether position building, position release, or subsequent OI contraction
helps distinguish continuation, absorption, short-covering-compatible rebounds,
and large-MFE opportunities.

Binance OI is an external cross-exchange positioning proxy. It is never labelled
as OKX-local OI. Future OI paths are physically separated from causal features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.binance_futures_metrics_loader import BinanceFuturesMetricsLoader  # noqa: E402
from src.research_common.post_sweep_oi import (  # noqa: E402
    PostSweepOIConfig,
    add_attempt_pair_features,
    attempt_mechanism_summary,
    build_future_oi_labels,
    causal_align_oi,
    causal_audit,
    compact_event_sample,
    coverage_by_period,
    data_quality_report,
    fixed_oi_bin_summary,
    large_mfe_oi_profile,
    load_r04_tables,
    new_low_attempt_oi_summary,
    oracle_pair_summary,
    oracle_turning_points,
    pair_oracle_with_prior_attempt,
    position_flow_state_summary,
    rebound_oi_path_summary,
    split_features_labels,
    taker_ratio_summary,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "05_post_sweep_binance_oi_mechanism_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_POST_SWEEP_BINANCE_OI_MECHANISM_R05"
EDGE_ID = "RESEARCH_ONLY_SWING_ZONE_POST_SWEEP_OI"
TITLE = "ETH Post-Sweep Binance OI Positioning Mechanism Study R05"
DEFAULT_R04_DIR = "data/reports/research/liquidity/post_sweep_continuation_exhaustion_r04"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/post_sweep_binance_oi_mechanism_r05"


def _comma_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return values


def _comma_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in str(value).split(",") if item.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causally align Binance 5m OI/positioning metrics to R04 post-sweep checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--r04-dir", default=DEFAULT_R04_DIR)
    parser.add_argument("--binance-data-dir", default=None)
    parser.add_argument("--binance-db-name", default="binance_futures_metrics.db")
    parser.add_argument("--oi-windows", type=_comma_strings, default=("5m", "15m", "30m", "1h", "4h", "1d"))
    parser.add_argument("--future-oi-horizons", type=_comma_ints, default=(15, 30, 60, 180))
    parser.add_argument("--publication-lag", default="1min")
    parser.add_argument("--baseline-tolerance", default="1min")
    parser.add_argument("--alignment-tolerance", default="10min")
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> PostSweepOIConfig:
    return PostSweepOIConfig(
        oi_windows=tuple(args.oi_windows),
        future_oi_horizons=tuple(args.future_oi_horizons),
        publication_lag=str(args.publication_lag),
        baseline_tolerance=str(args.baseline_tolerance),
        alignment_tolerance=str(args.alignment_tolerance),
        sample_rows=int(args.sample_rows),
    ).validate()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["empty_result"])
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", compression={"method": "gzip", "compresslevel": 1})


def _brief(cfg: PostSweepOIConfig, features: pd.DataFrame, oracle: pd.DataFrame, audit: pd.DataFrame) -> str:
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) if len(audit) else 0
    aligned = int(features["oi_context_present"].eq(True).sum()) if len(features) else 0
    rate = aligned / len(features) if len(features) else 0.0
    return f"""# Post-Sweep Binance OI Positioning Mechanism Study R05

## Decision supported

Determine whether external ETH perpetual positioning changes help distinguish:

1. post-sweep position building compatible with chase-short entry,
2. position release compatible with long liquidation/stop-out,
3. selling-impact failure while OI is still rising,
4. subsequent price-up/OI-down paths compatible with short covering,
5. large-MFE events with a stable causal OI profile.

## Source semantics

- Price and order flow: OKX ETH-USDT-SWAP R04 causal checkpoints.
- OI and positioning: Binance USD-M ETHUSDT official 5-minute metrics.
- Binance OI is an external cross-exchange proxy, not OKX-local OI.
- OI windows: {cfg.oi_windows}.
- Publication lag: {cfg.publication_lag}.
- Maximum causal staleness: {cfg.alignment_tolerance}.

## Sample

- Checkpoints: {len(features):,}
- Independent Sweep Zone events: {features['zone_event_id'].nunique() if len(features) else 0:,}
- OI-aligned checkpoints: {aligned:,} ({rate:.2%})
- Future-labelled oracle turning points: {len(oracle):,}

## Key safeguards

- OI is joined by ``oi_available_time <= checkpoint_available_time``.
- Future OI changes are stored only in the label table.
- State summaries use one first occurrence per event/state to avoid repeated-checkpoint weighting.
- Fixed OI bins are rounded and predeclared; no irregular fitted threshold is promoted to a rule.
- Oracle turning points use future knowledge and are descriptive only.
- No entry, stop, position size, fee PnL, or final strategy is produced in R05.

## Causal audit

Total violations: {violations}. Any non-zero value invalidates the report.
"""


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoints = pd.date_range("2025-01-01 00:06:00", periods=30, freq="1min")
    features = pd.DataFrame({
        "checkpoint_id": [f"E1_C{i:03d}" for i in range(1, 31)],
        "zone_event_id": ["E1"] * 30,
        "event_kind": ["swing_zone_sweep"] * 30,
        "period": ["TEST"] * 30,
        "event_pos": [100] * 30,
        "event_available_time": [pd.Timestamp("2025-01-01 00:05:00")] * 30,
        "checkpoint_pos": np.arange(101, 131),
        "checkpoint_time": checkpoints - pd.Timedelta(minutes=1),
        "checkpoint_available_time": checkpoints,
        "elapsed_bars": np.arange(1, 31),
        "zone_floor_price": 100.0, "zone_ceiling_price": 100.2, "zone_center_price": 100.1,
        "sweep_low": 99.8, "checkpoint_open": 99.9, "checkpoint_high": 100.0,
        "checkpoint_low": 99.7 - np.arange(30) * 0.01,
        "checkpoint_close": 99.8 - np.arange(30) * 0.005,
        "running_low_since_sweep": 99.7 - np.arange(30) * 0.01,
        "new_low_attempt_flag": [True] * 30, "new_low_attempt_index": np.arange(1, 31),
        "bars_since_new_low_attempt": 0, "new_low_extension_bp": 1.0,
        "new_low_extension_to_pre_atr_240m": 0.1, "attempt_delta_notional": -100.0,
        "attempt_sell_notional": 500.0, "attempt_extension_vs_previous": 0.8,
        "close_vs_zone_floor_bp": -10.0, "close_vs_running_low_bp": 10.0,
        "running_low_vs_zone_floor_bp": -20.0, "zone_floor_reclaimed": False,
        "zone_ceiling_reclaimed": False, "cum_delta_since_sweep": -1000.0,
        "cum_delta_ratio_since_sweep": -0.2, "cvd_new_low_flag": True,
        "cvd_new_low_without_price_new_low": False,
        "negative_delta_without_price_new_low": False,
        "no_new_low_3bars": False, "no_new_low_5bars": False, "no_new_low_10bars": False,
        "micro_high_break_3bars": False, "micro_high_break_5bars": False,
        "micro_high_break_10bars": False,
        "delta_ratio_1m": -0.2, "sell_share_1m": 0.6, "large_delta_ratio_1m": -0.3,
        "price_change_1m_bp": -2.0,
        "downside_bp_per_sell_million_1m": np.linspace(2.0, 0.3, 30),
        "downside_bp_per_abs_negative_delta_million_1m": np.linspace(4.0, 0.8, 30),
        "delta_ratio_5m": -0.2, "sell_share_5m": 0.6, "large_delta_ratio_5m": -0.3,
        "price_change_5m_bp": -5.0, "downside_bp_per_sell_million_5m": 1.0,
        "downside_bp_per_abs_negative_delta_million_5m": 2.0,
        "delta_ratio_15m": -0.1, "sell_share_15m": 0.55, "price_change_15m_bp": -8.0,
        "delta_ratio_30m": -0.1, "sell_share_30m": 0.55, "price_change_30m_bp": -10.0,
    })
    labels = pd.DataFrame({
        "checkpoint_id": features["checkpoint_id"], "zone_event_id": "E1", "period": "TEST",
        "elapsed_bars": features["elapsed_bars"], "entry_reference_time": checkpoints,
        "entry_reference_price": 100.0,
    })
    for h in (15, 30, 60, 180):
        labels[f"future_label_complete_{h}m"] = True
        labels[f"future_mfe_{h}m"] = 0.01
        labels[f"future_mae_{h}m"] = -0.002
        labels[f"future_close_return_{h}m"] = 0.005
        labels[f"future_no_lower_low_{h}m"] = True
    labels["future_reversal_dominant_60m"] = True
    labels["future_continuation_dominant_60m"] = False
    labels["future_reversal_dominant_180m"] = True
    labels["future_continuation_dominant_180m"] = False
    for tag in ("0p5", "1", "2"):
        labels[f"future_large_mfe_{tag}_180m"] = tag != "2"
    static = pd.DataFrame({"zone_event_id": ["E1"], "zone_prior_touch_median": [0.0],
                           "zone_left_high_range_20_bp_max": [300.0],
                           "zone_confirmation_reaction_close_bp_max": [100.0],
                           "sweep_depth_to_pre_atr_240m": [1.0]})
    oi_time = pd.date_range("2024-12-31 23:45:00", periods=60, freq="5min")
    oi = pd.DataFrame({
        "timestamp": oi_time,
        "available_time": oi_time + pd.Timedelta(minutes=1),
        "sum_open_interest": 1_000_000.0 + np.arange(60) * 100.0,
        "sum_open_interest_value": 3_000_000_000.0 + np.arange(60) * 1_000_000.0,
        "taker_volume_imbalance": -0.1,
        "top_trader_account_long_share": 0.6,
        "top_trader_position_long_share": 0.65,
        "global_account_long_share": 0.55,
    })
    for tag, shift in (("5m", 1), ("15m", 3), ("30m", 6), ("1h", 12), ("4h", 48), ("1d", 288)):
        oi[f"oi_base_change_{tag}"] = oi["sum_open_interest"].pct_change(shift)
        oi[f"oi_usd_change_{tag}"] = oi["sum_open_interest_value"].pct_change(shift)
        oi[f"oi_baseline_age_seconds_{tag}"] = pd.Timedelta(tag).total_seconds()
    return features, labels, static, oi


def run_self_test() -> None:
    from src.research_common.post_sweep_oi import (
        add_attempt_pair_features, build_future_oi_labels, causal_align_oi,
        causal_audit, split_features_labels,
    )

    cfg = PostSweepOIConfig(future_oi_horizons=(15, 30, 60), sample_rows=100).validate()
    r04_features, r04_labels, static, oi = _synthetic_tables()
    aligned = causal_align_oi(r04_features.merge(static, on="zone_event_id", how="left"), oi, cfg)
    aligned = add_attempt_pair_features(aligned)
    future = build_future_oi_labels(aligned, oi, cfg)
    features, labels = split_features_labels(aligned, r04_labels, future)
    audit = causal_audit(features, labels)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if violations or not features["oi_context_present"].all():
        raise RuntimeError(f"R05 self-test failed:\n{audit}")
    print(f"[self-test] passed rows={len(features)}", flush=True)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir

    cfg = _config(args)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    r04_dir = PROJECT_ROOT / args.r04_dir
    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] {start} -> {end}", flush=True)
    print(f"[source] R04={r04_dir}", flush=True)
    print(f"[oi] Binance {args.symbol} windows={cfg.oi_windows} lag={cfg.publication_lag}", flush=True)

    print("[stage] load selected R04 checkpoint/features/labels", flush=True)
    r04_features, r04_labels, static = load_r04_tables(r04_dir)
    mask = (r04_features["checkpoint_available_time"] >= start) & (r04_features["checkpoint_available_time"] <= end)
    ids = set(r04_features.loc[mask, "checkpoint_id"].astype(str))
    r04_features = r04_features.loc[mask].reset_index(drop=True)
    r04_labels = r04_labels.loc[r04_labels["checkpoint_id"].astype(str).isin(ids)].reset_index(drop=True)
    active_events = set(r04_features["zone_event_id"].astype(str))
    static = static.loc[static["zone_event_id"].astype(str).isin(active_events)].reset_index(drop=True)
    if r04_features.empty:
        raise RuntimeError("No R04 checkpoints in requested window")
    print(f"[r04] checkpoints={len(r04_features):,} events={len(active_events):,}", flush=True)

    print("[stage] load causal Binance metrics features", flush=True)
    loader = BinanceFuturesMetricsLoader(
        args.symbol,
        data_dir=args.binance_data_dir,
        db_name=args.binance_db_name,
    )
    max_future = max(cfg.future_oi_horizons)
    oi_start = r04_features["checkpoint_available_time"].min() - pd.Timedelta("1D")
    oi_end = r04_features["checkpoint_available_time"].max() + pd.Timedelta(minutes=max_future) + pd.Timedelta(cfg.alignment_tolerance)
    oi = loader.load_relative_features(
        oi_start,
        oi_end,
        windows=cfg.oi_windows,
        publication_lag=cfg.publication_lag,
        baseline_tolerance=cfg.baseline_tolerance,
        index_mode="none",
    )
    if oi.empty:
        raise RuntimeError(
            "Binance futures metrics database returned zero rows. Run tools/prebuild_binance_futures_metrics.py first."
        )
    print(f"[oi] rows={len(oi):,} available={oi['available_time'].min()}->{oi['available_time'].max()}", flush=True)

    print("[stage] causal as-of alignment and attempt-pair features", flush=True)
    checkpoint_base = r04_features.merge(static, on="zone_event_id", how="left", validate="many_to_one")
    aligned = causal_align_oi(checkpoint_base, oi, cfg)
    aligned = add_attempt_pair_features(aligned)
    aligned_ids = set(aligned["checkpoint_id"].astype(str))
    r04_labels = r04_labels.loc[r04_labels["checkpoint_id"].astype(str).isin(aligned_ids)].copy()
    future_oi = build_future_oi_labels(aligned, oi, cfg)
    feature_table, label_table = split_features_labels(aligned, r04_labels, future_oi)
    full = feature_table.merge(
        label_table.drop(columns=["zone_event_id", "period", "elapsed_bars"], errors="ignore"),
        on="checkpoint_id", how="left", validate="one_to_one",
    )

    audit = causal_audit(feature_table, label_table)
    violations = int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"R05 causal audit failed with {violations} violations:\n{audit.to_string(index=False)}")
    aligned_rate = float(feature_table["oi_context_present"].mean())
    print(f"[align] rows={len(feature_table):,} aligned={aligned_rate:.2%}", flush=True)
    if aligned_rate < 0.90:
        raise RuntimeError(f"OI alignment coverage too low: {aligned_rate:.2%}; inspect database coverage/gaps")

    print("[stage] OI positioning and future-path reports", flush=True)
    coverage = loader.coverage()
    coverage_days = loader.coverage_by_day(start.date(), end.date())
    oracle = oracle_turning_points(full)
    oracle_pairs = pair_oracle_with_prior_attempt(full, oracle)
    reports = {
        "01_data_quality.csv": data_quality_report(feature_table, label_table, coverage, coverage_days),
        "02_oi_alignment_coverage_by_period.csv": coverage_by_period(feature_table),
        "03_position_flow_state_summary.csv": position_flow_state_summary(full),
        "04_fixed_oi_change_bin_summary.csv": fixed_oi_bin_summary(full, cfg),
        "05_new_low_attempt_oi_summary.csv": new_low_attempt_oi_summary(full),
        "06_attempt_impact_oi_mechanism_summary.csv": attempt_mechanism_summary(full),
        "07_oracle_turning_point_oi_pair_summary.csv": oracle_pair_summary(oracle_pairs),
        "08_future_price_oi_path_summary.csv": rebound_oi_path_summary(full),
        "09_large_mfe_oi_feature_profile.csv": large_mfe_oi_profile(full, cfg),
        "10_taker_ratio_summary.csv": taker_ratio_summary(full),
        "11_oracle_turning_point_oi_sample.csv": oracle.head(cfg.sample_rows),
        "12_large_opportunity_event_sample.csv": compact_event_sample(full, cfg.sample_rows),
        "13_causal_audit.csv": audit,
        "14_binance_coverage_by_day.csv": coverage_days,
    }

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": SCRIPT_NAME, "version": SCRIPT_VERSION, "title": TITLE,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID,
        "start_date": str(start), "end_date": str(end),
        "r04_dir": str(r04_dir), "binance_symbol": args.symbol,
        "binance_db": str(loader.db_path), "config": asdict(cfg),
        "checkpoints": len(feature_table), "events": int(feature_table["zone_event_id"].nunique()),
        "oi_alignment_rate": aligned_rate,
        "notes": [
            "Binance OI is an external cross-exchange proxy, not OKX-local OI.",
            "Oracle and future OI fields are future labels only.",
            "This is mechanism research, not a PnL backtest.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, frame in reports.items():
        _write_csv(frame, out_dir / name)
    _write_gzip(feature_table, out_dir / "15_checkpoint_oi_feature_table.csv.gz")
    _write_gzip(label_table, out_dir / "16_checkpoint_oi_label_table.csv.gz")
    _write_csv(oracle_pairs.head(cfg.sample_rows), out_dir / "17_oracle_vs_prior_attempt_sample.csv")
    (out_dir / "18_research_brief.md").write_text(_brief(cfg, feature_table, oracle, audit), encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
