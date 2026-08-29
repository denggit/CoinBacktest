#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03 - ETH liquidity-stack order-flow uplift + execution overlays.

R03 freezes the R02 long-side liquidity-stack hypothesis and asks two separate
questions:

1. Can causal 1m trade-bar flow or r0020/step1 Range Footprint improve quality
   without simply destroying the already-small sample?  A >=3-pool cohort is
   included as the only frequency-expansion layer; >=4 pools remains the core.
2. Once the stack edge is known, does FVG execution work better as immediate
   market, proximal limit, or 50/50 market+limit?  This is explicitly an
   execution overlay and is not allowed to redefine the core liquidity event.

No NY-open gate is used.  No time-profit exit is used.  The primary objective
remains the opposing active 4H+ liquidity target with a structural sweep stop.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2 import (  # noqa: E402
    R03Config,
    attach_footprint_microstructure_features,
    attach_overlay_structural_outcomes,
    build_fvg_execution_overlay_attempts,
    build_hybrid_5050_outcomes,
    build_tradebar_microstructure_features,
    first_pool_threshold_crossing_trades,
    r03_causal_audit,
    r03_globalize_legacy_trade_ids,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "3.0.1"
EXPERIMENT_ID = "ETH_ICT_MSS2_STACK_ORDERFLOW_EXECUTION_R03"
EDGE_ID = "RESEARCH_ONLY_ETH_HTF_MULTI_LIQUIDITY_STACK_EXHAUSTION_LONG"
TITLE = "ETH ICT MSS2 Liquidity Stack Order-Flow Uplift + Execution R03"
DEFAULT_R02_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r03_liquidity_stack_orderflow_execution"


def _read_r02_manifest(r02_dir: Path) -> dict[str, object]:
    path = r02_dir / "00_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse R02 manifest: {path}: {exc}") from exc


def _position_alignment_audit(
    bars: pd.DataFrame,
    threshold_stages: pd.DataFrame,
    lifecycle: pd.DataFrame,
) -> pd.DataFrame:
    """Verify persisted R02 integer positions still refer to the same 1m bars.

    R02 stores positions relative to the naked-1m frame loaded from its warmup
    start.  R03 must use that same origin for any positional lifecycle logic.
    The timestamp/position redundancy is used here as an explicit guard against
    silently shifted stop or target books.
    """
    idx = bars.index
    checks: list[dict[str, object]] = []

    def _check(frame: pd.DataFrame, pos_col: str, time_col: str, name: str) -> None:
        if frame.empty or pos_col not in frame.columns or time_col not in frame.columns:
            checks.append({"check": name, "rows": 0, "violations": 0, "missing": 0})
            return
        pos = pd.to_numeric(frame[pos_col], errors="coerce")
        ts = pd.to_datetime(frame[time_col], errors="coerce")
        valid = pos.notna() & pos.ge(0) & ts.notna()
        rows = int(valid.sum())
        bad = 0
        missing = 0
        for p0, t0 in zip(pos.loc[valid].astype(int), ts.loc[valid]):
            if p0 < 0 or p0 >= len(idx):
                bad += 1
                continue
            if pd.Timestamp(idx[p0]) != pd.Timestamp(t0):
                bad += 1
        missing = int((~valid).sum())
        checks.append({"check": name, "rows": rows, "violations": int(bad), "missing": missing})

    _check(threshold_stages, "sweep_pos_1m", "sweep_bar_time_1m", "stage_sweep_pos_matches_time")
    _check(threshold_stages, "episode_start_pos_1m", "episode_start_time_1m", "episode_start_pos_matches_time")
    _check(lifecycle, "active_pos_1m", "initial_available_time", "lifecycle_active_pos_matches_time")
    _check(lifecycle, "sweep_pos_1m", "sweep_bar_time_1m", "lifecycle_sweep_pos_matches_time")
    return pd.DataFrame(checks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tradebar-db-name", default="okx_trade_bars.db")
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--footprint-db-name", default="okx_range_footprints.db")
    p.add_argument("--r02-report-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-footprint", action="store_true")
    p.add_argument("--skip-execution-overlay", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--market-roundtrip-cost", type=float, default=0.0011)
    p.add_argument("--limit-roundtrip-cost", type=float, default=0.0009)
    p.add_argument("--fvg-signal-wait-minutes", type=int, default=180)
    p.add_argument("--fvg-limit-wait-minutes", type=int, default=180)
    p.add_argument("--footprint-chunk-days", type=int, default=120)
    return p.parse_args(argv)


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x.loc[x > 0].sum())
    losses = float(-x.loc[x < 0].sum())
    if losses <= 1e-12:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _metric(frame: pd.DataFrame, net_col: str) -> dict[str, object]:
    row: dict[str, object] = {"trades_total": int(len(frame))}
    if frame.empty or net_col not in frame.columns:
        return row
    net = pd.to_numeric(frame[net_col], errors="coerce")
    resolved = frame.loc[net.notna()].copy()
    x = pd.to_numeric(resolved[net_col], errors="coerce").dropna()
    row["resolved"] = int(len(x))
    row["censored_or_missing"] = int(len(frame) - len(x))
    row["mean_net"] = float(x.mean()) if len(x) else np.nan
    row["median_net"] = float(x.median()) if len(x) else np.nan
    row["win_rate"] = float((x > 0).mean()) if len(x) else np.nan
    row["pf"] = _pf(x)
    row["sum_net"] = float(x.sum()) if len(x) else np.nan
    return row


def _group_metric(frame: pd.DataFrame, group_cols: Sequence[str], net_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in frame.columns]
    grouped = [((), frame)] if not cols else frame.groupby(cols, dropna=False, observed=False, sort=True)
    rows = []
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(cols, keys)}
        row.update(_metric(part, net_col))
        rows.append(row)
    return pd.DataFrame(rows)


def _read_r02_report(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = report_dir / "10_trade_features_causal.csv"
    label_path = report_dir / "11_trade_structural_exit_labels.csv"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"R03 requires the completed R02 report: {feature_path} and {label_path}. Run R02 first."
        )
    feature_cols = [
        "trade_event_id", "stage_id", "episode_id", "trade_direction", "execution_minutes", "trigger_type",
        "sweep_pos_1m", "sweep_bar_time_1m", "episode_start_pos_1m", "episode_start_time_1m",
        "episode_elapsed_minutes", "levels_consumed_cum", "distinct_timeframes_cum",
        "max_source_timeframe_min_cum", "htf_240m_plus_levels_cum", "htf_1440m_plus_levels_cum",
        "episode_consumption_depth_bp", "levels_consumed_per_min_cum", "price_pools_5p0bp_cum",
        "pools_per_min_5p0bp_cum", "price_pools_10p0bp_cum", "pools_per_min_10p0bp_cum",
        "price_pools_20p0bp_cum", "pools_per_min_20p0bp_cum", "signal_available_time", "signal_bar_time",
        "entry_pos_1m", "entry_time", "entry_price", "entry_kind", "stop_price", "risk_bps",
        "mss_reference_confirmed_order", "displacement_atr", "path_efficiency", "fvg_width_atr",
        "has_displacement_fvg", "session_primary", "is_weekend_utc", "year", "quarter", "month",
    ]
    features = pd.read_csv(feature_path, usecols=lambda c: c in set(feature_cols), low_memory=False)
    target_names = ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r2p0", "r3p0", "r5p0")
    label_cols = ["trade_event_id", "stage_id", "episode_id"]
    for name in target_names:
        label_cols += [
            f"target_{name}_outcome", f"target_{name}_r_multiple", f"target_{name}_holding_minutes",
            f"target_{name}_gross_return", f"target_{name}_net_return_base",
            f"target_{name}_net_return_cost2x", f"target_{name}_net_return_cost3x",
        ]
    labels = pd.read_csv(label_path, usecols=lambda c: c in set(label_cols), low_memory=False)
    if len(features) != len(labels):
        raise RuntimeError("R02 feature/label row counts differ")
    features, labels = r03_globalize_legacy_trade_ids(features, labels)
    assert labels is not None
    if not np.array_equal(features["trade_event_id"].astype(str).to_numpy(), labels["trade_event_id"].astype(str).to_numpy()):
        raise RuntimeError("R02 feature/label rows are not aligned after ID repair")
    merged = pd.concat(
        [features.reset_index(drop=True), labels.drop(columns=["trade_event_id", "stage_id", "episode_id"]).reset_index(drop=True)],
        axis=1,
    )
    for name in ("sweep_bar_time_1m", "episode_start_time_1m", "signal_available_time", "signal_bar_time", "entry_time"):
        if name in merged.columns:
            merged[name] = pd.to_datetime(merged[name], errors="coerce")
    return merged, labels


def _candidate_sets(r02: pd.DataFrame, cfg: R03Config) -> pd.DataFrame:
    frames = []
    for threshold, name in ((cfg.pool_threshold_expand, "expand_ge3"), (cfg.pool_threshold_core, "core_ge4")):
        part = first_pool_threshold_crossing_trades(
            r02,
            threshold=threshold,
            tolerance_bps=cfg.pool_tolerance_bps,
            direction=1,
            trigger_type="episode_reclaim",
            execution_minutes=(1, 2, 5),
        )
        part["cohort"] = name
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _candidate_performance(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    target = "target_htf240_net_return_cost2x"
    overall = _group_metric(candidates, ["cohort", "execution_minutes"], target)
    year = _group_metric(candidates, ["cohort", "execution_minutes", "year"], target)
    htf = candidates.copy()
    htf["stack_has_4h_plus"] = pd.to_numeric(htf["max_source_timeframe_min_cum"], errors="coerce").ge(240)
    htf_summary = _group_metric(htf, ["cohort", "execution_minutes", "stack_has_4h_plus"], target)
    return overall, year, htf_summary


def _split_name(ts: pd.Series) -> pd.Series:
    x = pd.to_datetime(ts, errors="coerce")
    out = pd.Series("forward_2025_2026", index=x.index, dtype="string")
    out.loc[x < pd.Timestamp("2025-01-01")] = "train_2023_2024"
    out.loc[x >= pd.Timestamp("2025-10-01")] = "late_2025Q4_2026H1"
    return out


def _tradebar_summaries(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    frame = joined.copy()
    frame["split"] = _split_name(frame["entry_time"])
    net = "target_htf240_net_return_cost2x"
    flags = {
        "all": pd.Series(True, index=frame.index),
        "tb_absorption": frame.get("tb_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1),
        "tb_flow_recovery": frame.get("tb_flow_recovery_flag", 0).fillna(0).astype(int).eq(1),
    }
    flags["tb_absorption_and_recovery"] = flags["tb_absorption"] & flags["tb_flow_recovery"]
    rows = []
    for flag_name, mask in flags.items():
        part = frame.loc[mask].copy()
        summary = _group_metric(part, ["cohort", "execution_minutes", "split"], net)
        if not summary.empty:
            summary.insert(0, "mechanism", flag_name)
            rows.append(summary)
    mechanism = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()

    # Frozen quartiles: thresholds are computed on 2023-2024 only, then reused
    # on all later rows. They are descriptive feature-shape checks, not a search
    # over combinations or a promoted gate.
    feature_names = [
        "tb_episode_sell_notional_intensity_vs_pre60",
        "tb_episode_impact_ratio_vs_pre60",
        "tb_last5_delta_improvement_vs_episode",
        "tb_episode_close_off_low_bp",
        "tb_episode_large_sell_share",
    ]
    strata_rows = []
    threshold_rows = []
    for cohort in sorted(frame["cohort"].dropna().astype(str).unique()):
        for minutes in sorted(pd.to_numeric(frame["execution_minutes"], errors="coerce").dropna().astype(int).unique()):
            base = frame.loc[frame["cohort"].astype(str).eq(cohort) & pd.to_numeric(frame["execution_minutes"], errors="coerce").eq(minutes)]
            train = base.loc[pd.to_datetime(base["entry_time"], errors="coerce") < pd.Timestamp("2025-01-01")]
            for feature in feature_names:
                if feature not in base.columns:
                    continue
                train_values = pd.to_numeric(train[feature], errors="coerce").dropna()
                if len(train_values) < 20:
                    continue
                q = train_values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
                threshold_rows.append(
                    {
                        "cohort": cohort,
                        "execution_minutes": minutes,
                        "feature": feature,
                        "train_rows": int(len(train_values)),
                        "q25": q[0], "q50": q[1], "q75": q[2],
                    }
                )
                values = pd.to_numeric(base[feature], errors="coerce")
                bins = pd.cut(values, [-np.inf, q[0], q[1], q[2], np.inf], labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
                temp = base.copy()
                temp["feature_bin"] = bins.astype("string")
                temp["split"] = _split_name(temp["entry_time"])
                summ = _group_metric(temp.dropna(subset=["feature_bin"]), ["feature_bin", "split"], net)
                if not summ.empty:
                    summ.insert(0, "feature", feature)
                    summ.insert(0, "execution_minutes", minutes)
                    summ.insert(0, "cohort", cohort)
                    strata_rows.append(summ)
    strata = pd.concat(strata_rows, ignore_index=True, sort=False) if strata_rows else pd.DataFrame()
    thresholds = pd.DataFrame(threshold_rows)

    # Frequency-recovery scorecard: compare the broad >=3 cohort with fixed
    # microstructure semantics against the core >=4 baseline. No best-bin pick.
    freq_rows = []
    for minutes in (1, 2, 5):
        core = frame.loc[frame["cohort"].eq("core_ge4") & pd.to_numeric(frame["execution_minutes"], errors="coerce").eq(minutes)]
        broad = frame.loc[frame["cohort"].eq("expand_ge3") & pd.to_numeric(frame["execution_minutes"], errors="coerce").eq(minutes)]
        for name, part in (
            ("core_ge4_all", core),
            ("expand_ge3_all", broad),
            ("expand_ge3_tb_absorption", broad.loc[broad.get("tb_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1)]),
            ("expand_ge3_tb_flow_recovery", broad.loc[broad.get("tb_flow_recovery_flag", 0).fillna(0).astype(int).eq(1)]),
            ("expand_ge3_tb_both", broad.loc[
                broad.get("tb_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1)
                & broad.get("tb_flow_recovery_flag", 0).fillna(0).astype(int).eq(1)
            ]),
        ):
            row = {"execution_minutes": minutes, "selection": name}
            row.update(_metric(part, net))
            row["years_present"] = int(pd.to_datetime(part.get("entry_time"), errors="coerce").dt.year.nunique()) if len(part) else 0
            freq_rows.append(row)
    return mechanism, strata, pd.DataFrame(freq_rows), thresholds


def _footprint_summaries(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = joined.copy()
    frame["split"] = _split_name(frame["entry_time"])
    valid = frame.get("fp_causal_valid", pd.Series(False, index=frame.index, dtype="boolean")).astype("boolean").fillna(False).astype(bool)
    coverage = (
        frame.assign(fp_valid=valid)
        .groupby(["cohort", "execution_minutes", "split"], dropna=False, observed=False, sort=True)
        .agg(events=("trade_event_id", "size"), covered=("fp_valid", "sum"), coverage=("fp_valid", "mean"))
        .reset_index()
    )
    matched = frame.loc[valid].copy()
    net = "target_htf240_net_return_cost2x"
    rows = []
    for name, mask in (
        ("covered_all", pd.Series(True, index=matched.index)),
        ("fp_absorption", matched.get("fp_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1)),
        ("fp_delta_recovery", matched.get("fp_delta_recovery_flag", 0).fillna(0).astype(int).eq(1)),
    ):
        summ = _group_metric(matched.loc[mask], ["cohort", "execution_minutes", "split"], net)
        if not summ.empty:
            summ.insert(0, "mechanism", name)
            rows.append(summ)
    return coverage, pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _first_threshold_stages(stage_path: Path, threshold: int) -> pd.DataFrame:
    cols = [
        "stage_id", "episode_id", "trade_direction", "sweep_pos_1m", "sweep_bar_time_1m",
        "episode_start_pos_1m", "episode_start_time_1m",
        "price_pools_10p0bp_cum", "max_source_timeframe_min_cum",
    ]
    stages = pd.read_csv(stage_path, usecols=lambda c: c in set(cols), low_memory=False)
    stages["sweep_bar_time_1m"] = pd.to_datetime(stages["sweep_bar_time_1m"], errors="coerce")
    if "episode_start_time_1m" in stages.columns:
        stages["episode_start_time_1m"] = pd.to_datetime(stages["episode_start_time_1m"], errors="coerce")
    part = stages.loc[
        pd.to_numeric(stages["trade_direction"], errors="coerce").eq(1)
        & pd.to_numeric(stages["price_pools_10p0bp_cum"], errors="coerce").fillna(0).ge(int(threshold))
    ].copy()
    part = part.sort_values(["episode_id", "sweep_pos_1m", "stage_id"], kind="stable")
    return part.drop_duplicates(["episode_id"], keep="first").reset_index(drop=True)


def _execution_overlay_summary(overlay: pd.DataFrame, hybrid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if not overlay.empty:
        frame = overlay.copy()
        frame["year"] = pd.to_datetime(frame["signal_available_time"], errors="coerce").dt.year
        for kind, part in frame.groupby(["execution_minutes", "trigger_type"], dropna=False, observed=False, sort=True):
            minutes, trigger = kind
            row = {"execution_minutes": minutes, "execution_mode": trigger}
            row["attempts"] = int(len(part))
            row["fill_rate"] = float(pd.to_numeric(part["entry_fill_flag"], errors="coerce").fillna(0).mean())
            row.update(_metric(part, "target_htf240_net_execution_cost2x"))
            rows.append(row)
    if not hybrid.empty:
        for minutes, part in hybrid.groupby("execution_minutes", dropna=False, observed=False, sort=True):
            row = {"execution_minutes": minutes, "execution_mode": "stack_first_fvg_hybrid_50_50"}
            row["attempts"] = int(len(part))
            row["fill_rate"] = float(pd.to_numeric(part["limit_filled_before_market_exit"], errors="coerce").fillna(0).mean())
            row.update(_metric(part, "hybrid_net_execution_cost2x"))
            rows.append(row)
    overall = pd.DataFrame(rows)

    year_rows = []
    if not overlay.empty:
        temp = overlay.copy()
        temp["year"] = pd.to_datetime(temp["signal_available_time"], errors="coerce").dt.year
        summ = _group_metric(temp, ["execution_minutes", "trigger_type", "year"], "target_htf240_net_execution_cost2x")
        if not summ.empty:
            summ = summ.rename(columns={"trigger_type": "execution_mode"})
            year_rows.append(summ)
    if not hybrid.empty:
        temp = hybrid.copy()
        # episode timing is not carried into hybrid; year can be joined upstream if desired.
        if "year" in temp.columns:
            summ = _group_metric(temp, ["execution_minutes", "year"], "hybrid_net_execution_cost2x")
            if not summ.empty:
                summ["execution_mode"] = "stack_first_fvg_hybrid_50_50"
                year_rows.append(summ)
    return overall, pd.concat(year_rows, ignore_index=True, sort=False) if year_rows else pd.DataFrame()


def _exit_target_comparison(candidates: pd.DataFrame, execution_minutes: int = 5) -> pd.DataFrame:
    part = candidates.loc[
        candidates["cohort"].eq("core_ge4")
        & pd.to_numeric(candidates["execution_minutes"], errors="coerce").eq(int(execution_minutes))
    ].copy()
    rows = []
    for target in ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r2p0", "r3p0", "r5p0"):
        net = f"target_{target}_net_return_cost2x"
        row = {"target": target}
        row.update(_metric(part, net))
        hold = pd.to_numeric(part.get(f"target_{target}_holding_minutes"), errors="coerce")
        row["median_holding_minutes"] = float(hold.median()) if hold.notna().any() else np.nan
        row["over_1d_rate"] = float((hold > 1440).mean()) if hold.notna().any() else np.nan
        row["median_target_r"] = float(pd.to_numeric(part.get(f"target_{target}_r_multiple"), errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R03Config(
        market_roundtrip_cost=float(args.market_roundtrip_cost),
        limit_roundtrip_cost=float(args.limit_roundtrip_cost),
        fvg_signal_wait_minutes=int(args.fvg_signal_wait_minutes),
        fvg_limit_wait_minutes=int(args.fvg_limit_wait_minutes),
        footprint_chunk_days=int(args.footprint_chunk_days),
    ).validate()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    r02_dir = PROJECT_ROOT / args.r02_report_dir
    progress = not bool(args.no_progress)
    r02_manifest = _read_r02_manifest(r02_dir)

    print("[r03] load R02 causal trade/outcome report", flush=True)
    r02, _ = _read_r02_report(r02_dir)
    candidates = _candidate_sets(r02, cfg)
    overall, yearly, htf = _candidate_performance(candidates)
    overall.to_csv(out_dir / "03_candidate_cohort_performance.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "04_candidate_year_performance.csv", index=False, encoding="utf-8-sig")
    htf.to_csv(out_dir / "05_candidate_htf_involvement.csv", index=False, encoding="utf-8-sig")

    checkpoints = candidates.loc[candidates["cohort"].eq("expand_ge3")].copy()
    # One checkpoint per concrete R02 trade ID. The >=4 cohort is a subset of
    # these same rows, so microstructure is extracted once and reused.
    checkpoints = checkpoints.drop_duplicates("trade_event_id", keep="first")
    checkpoint_frame = pd.DataFrame(
        {
            "checkpoint_id": checkpoints["trade_event_id"].astype(str),
            "decision_time": pd.to_datetime(checkpoints["signal_available_time"], errors="coerce"),
            "episode_start_time": pd.to_datetime(checkpoints["episode_start_time_1m"], errors="coerce"),
        }
    )

    print("[r03] causal 1m trade-bar microstructure", flush=True)
    tb_features, tb_audit = build_tradebar_microstructure_features(
        checkpoint_frame,
        symbol=args.symbol,
        data_dir=args.data_dir,
        db_name=args.tradebar_db_name,
        config=cfg,
        show_progress=progress,
    )
    tb_features.to_csv(out_dir / "06_tradebar_features_causal.csv.gz", index=False, compression="gzip")
    tb_audit.to_csv(out_dir / "07_tradebar_build_audit.csv", index=False, encoding="utf-8-sig")
    tb_join = candidates.merge(tb_features, left_on="trade_event_id", right_on="checkpoint_id", how="left", validate="many_to_one")
    tb_mech, tb_strata, tb_freq, tb_thresholds = _tradebar_summaries(tb_join)
    tb_mech.to_csv(out_dir / "08_tradebar_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    tb_strata.to_csv(out_dir / "09_tradebar_frozen_quartile_summary.csv", index=False, encoding="utf-8-sig")
    tb_freq.to_csv(out_dir / "10_tradebar_frequency_recovery.csv", index=False, encoding="utf-8-sig")
    tb_thresholds.to_csv(out_dir / "10b_tradebar_train_quartile_thresholds.csv", index=False, encoding="utf-8-sig")

    fp_features = pd.DataFrame()
    fp_audit = pd.DataFrame()
    if not args.skip_footprint:
        print("[r03] causal r0020/step1 footprint incremental context", flush=True)
        fp_features, fp_audit = attach_footprint_microstructure_features(
            checkpoint_frame[["checkpoint_id", "decision_time"]],
            symbol=args.symbol,
            data_dir=args.data_dir,
            range_db_name=args.range_db_name,
            footprint_db_name=args.footprint_db_name,
            config=cfg,
            show_progress=progress,
        )
        fp_features.to_csv(out_dir / "11_footprint_features_causal.csv.gz", index=False, compression="gzip")
        fp_audit.to_csv(out_dir / "12_footprint_build_audit.csv", index=False, encoding="utf-8-sig")
        fp_join = candidates.merge(fp_features, left_on="trade_event_id", right_on="checkpoint_id", how="left", validate="many_to_one")
        fp_cov, fp_mech = _footprint_summaries(fp_join)
        fp_cov.to_csv(out_dir / "13_footprint_coverage.csv", index=False, encoding="utf-8-sig")
        fp_mech.to_csv(out_dir / "14_footprint_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"status": "disabled_by_cli"}]).to_csv(out_dir / "13_footprint_coverage.csv", index=False, encoding="utf-8-sig")

    overlay = pd.DataFrame()
    hybrid = pd.DataFrame()
    position_audit = pd.DataFrame()
    if not args.skip_execution_overlay:
        print("[r03] load bare 1m K for secondary FVG execution overlay", flush=True)
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        # R02 position-bearing lifecycle/stage artifacts were built on the 1m
        # frame beginning at the R02 warmup start.  Reusing only research_start
        # shifts every persisted integer position by ~one year and corrupts both
        # structural stops and the dynamic target book.  Prefer the manifest as
        # the source of truth; CLI warmup is only a fallback for legacy reports.
        overlay_warmup_start = str(r02_manifest.get("warmup_start_date") or args.warmup_start_date)
        manifest_symbol = str(r02_manifest.get("symbol") or args.symbol)
        if manifest_symbol != str(args.symbol):
            raise RuntimeError(
                f"R02/R03 symbol mismatch: R02 manifest={manifest_symbol!r}, R03 --symbol={args.symbol!r}"
            )
        bars = loader.fetch_data_by_date_range(overlay_warmup_start, args.end_date)
        if bars.empty:
            raise RuntimeError("no 1m bare K data for R03 execution overlay")
        lifecycle_path = r02_dir / "01_liquidity_lifecycle_causal.csv"
        stage_path = r02_dir / "04_sweep_episode_stages_causal.csv"
        if not lifecycle_path.exists() or not stage_path.exists():
            raise FileNotFoundError("R03 execution overlay requires R02 lifecycle and episode-stage files")
        lifecycle_cols = [
            "pivot_side", "active_pos_1m", "initial_available_time", "level_price",
            "source_timeframe_min", "sweep_pos_1m", "sweep_bar_time_1m",
        ]
        lifecycle = pd.read_csv(lifecycle_path, usecols=lambda c: c in set(lifecycle_cols), low_memory=False)
        for c in ("initial_available_time", "sweep_bar_time_1m"):
            if c in lifecycle.columns:
                lifecycle[c] = pd.to_datetime(lifecycle[c], errors="coerce")
        threshold_stages = _first_threshold_stages(stage_path, cfg.pool_threshold_core)
        position_audit = _position_alignment_audit(bars, threshold_stages, lifecycle)
        position_audit.to_csv(out_dir / "15a_execution_position_alignment_audit.csv", index=False, encoding="utf-8-sig")
        position_bad = int(pd.to_numeric(position_audit.get("violations"), errors="coerce").fillna(0).sum())
        if position_bad:
            raise RuntimeError(
                "R02/R03 1m position alignment failed; refusing to run execution overlay because "
                f"structural stops/targets would be shifted. violations={position_bad}, "
                f"R02 warmup={overlay_warmup_start}. See 15a_execution_position_alignment_audit.csv"
            )
        overlays = []
        for minutes in cfg.fvg_execution_minutes:
            attempts = build_fvg_execution_overlay_attempts(
                bars,
                threshold_stages,
                execution_minutes=int(minutes),
                config=cfg,
                show_progress=progress,
            )
            if not attempts.empty:
                overlays.append(attach_overlay_structural_outcomes(bars, lifecycle, attempts, config=cfg, show_progress=progress))
        overlay = pd.concat(overlays, ignore_index=True, sort=False) if overlays else pd.DataFrame()
        if not overlay.empty:
            hybrid = build_hybrid_5050_outcomes(bars, overlay, config=cfg)
        overlay.to_csv(out_dir / "15_fvg_execution_overlay_trades.csv.gz", index=False, compression="gzip")
        hybrid.to_csv(out_dir / "16_fvg_hybrid_5050_outcomes.csv.gz", index=False, compression="gzip")
        ex_summary, ex_year = _execution_overlay_summary(overlay, hybrid)
        ex_summary.to_csv(out_dir / "17_fvg_execution_overlay_summary.csv", index=False, encoding="utf-8-sig")
        ex_year.to_csv(out_dir / "18_fvg_execution_year_summary.csv", index=False, encoding="utf-8-sig")

    exit_compare = _exit_target_comparison(candidates, cfg.baseline_execution_minutes)
    exit_compare.to_csv(out_dir / "19_exit_target_comparison_5m_core.csv", index=False, encoding="utf-8-sig")

    audit = r03_causal_audit(tb_features, fp_features, overlay)
    audit.to_csv(out_dir / "20_causal_audit.csv", index=False, encoding="utf-8-sig")
    engineering = pd.DataFrame(
        [
            {"check": "r02_trade_event_id_unique_after_repair", "passed": int(not r02["trade_event_id"].duplicated().any())},
            {"check": "candidate_trade_event_id_unique_within_cohort", "passed": int(not candidates.duplicated(["cohort", "trade_event_id"]).any())},
            {"check": "ny_open_used_as_gate", "passed": 1, "value": "NO"},
            {"check": "time_profit_exit_used", "passed": 1, "value": "NO"},
            {"check": "footprint_missing_is_not_zero_signal", "passed": 1, "value": "matched-covered subset only"},
            {
                "check": "r02_execution_position_alignment",
                "passed": int(position_audit.empty or pd.to_numeric(position_audit.get("violations"), errors="coerce").fillna(0).sum() == 0),
                "value": "R02 position-bearing files are checked against naked-1m timestamps before overlay",
            },
        ]
    )
    engineering.to_csv(out_dir / "01_engineering_audit.csv", index=False, encoding="utf-8-sig")

    design = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "config": asdict(cfg),
        "frozen_core": "long + 10bp independent pools + first >=4 pool crossing + episode reclaim + structural stop + opposing 4H+ liquidity target",
        "frequency_expansion": "first >=3 pool crossing; order-flow may stratify but cannot alter historical liquidity admission",
        "tradebar_mechanism": "high sell activity vs prior 60m + lower downside impact per sell million + 5m delta improvement",
        "footprint_mechanism": "more low-3-bin sell flow vs previous down range + lower price impact + delta and close-off-low improvement",
        "execution_overlay": "after frozen >=4 stack, first same-direction FVG; compare market, proximal limit, and 50/50 hybrid",
        "session_policy": "sessions retained only as descriptive columns inherited from R02; no NY-open gate",
        "exit_policy": "structural stop vs frozen opposing 4H+ liquidity; 7d is censor only",
        "caution": "R02 >=4/4H findings were discovered on the same historical corpus; R03 can replicate across definitions and forward microstructure strata but is not a brand-new untouched external holdout.",
    }
    (out_dir / "02_frozen_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {TITLE}",
        "",
        "R03 keeps crypto-native liquidity-stack logic. NYSE/NY-open timing is not an admission rule.",
        "",
        "## What to judge first",
        "1. Does >=4 pool Long remain positive across 1m/2m/5m and years under 2x cost?",
        "2. Can the predeclared >=3 frequency expansion recover more trades when the fixed trade-bar absorption/recovery mechanism is present?",
        "3. On footprint-covered dates only, does the fixed absorption mechanism improve the matched covered baseline?",
        "4. Does first-FVG market beat waiting for proximal limit after the same frozen stack event, and does 50/50 improve the fill/price tradeoff?",
        "5. Do not promote a quartile just because it is the best historical bin. Quartiles are diagnostics only.",
        "",
        "## Cost semantics for execution overlay",
        f"- market round-trip convention: {cfg.market_roundtrip_cost:.6f}",
        f"- FVG limit-entry + taker-exit convention: {cfg.limit_roundtrip_cost:.6f}",
        "- 2x/3x rows multiply the corresponding execution cost, not the gross path.",
        "",
        "## Run",
        f"`python research\\ict\\mss2\\03_liquidity_stack_orderflow_execution.py --symbol {args.symbol} --start-date {args.start_date} --end-date \"{args.end_date}\"`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "rows_r02": int(len(r02)),
        "candidate_rows": int(len(candidates)),
        "tradebar_rows": int(len(tb_features)),
        "footprint_rows": int(len(fp_features)),
        "execution_overlay_rows": int(len(overlay)),
        "hybrid_rows": int(len(hybrid)),
        "causal_violations": int(pd.to_numeric(audit.get("violations"), errors="coerce").fillna(0).sum()) if len(audit) else 0,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] R03 report -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
