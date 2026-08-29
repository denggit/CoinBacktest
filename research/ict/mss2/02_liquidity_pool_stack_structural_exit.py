#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 - ETH causal liquidity-pool stack exhaustion + structural exits.

R01 showed that treating each swing as an independent statistical event badly
inflates the apparent sample size when one 1m impulse consumes several levels.
It also showed that a complete MSS->FVG entry is not profitable as a universal
ETH rule. R02 therefore changes both the event unit and trade-management unit:

1. One 1m sweep bar = one ``sweep_stage`` regardless of how many levels it hits.
2. Nearby swept levels are merged into 5/10/20bp price pools.
3. Consecutive same-direction stages that keep extending the extreme form a
   causal ``sweep_episode``. Every stage only sees current/past episode state.
4. 1m/2m/5m execution is compared without assuming NY-open or 15m->1m is best.
5. Entries compare immediate reclaim, episode reclaim, structural MSS market,
   and structural MSS + FVG limit.
6. There is NO time-profit exit. A structural stop competes against an opposing
   liquidity target frozen at entry. Seven days is only right-censoring.
7. Targets compare nearest active level, nearest >=2-level pool, nearest >=2-TF
   pool, nearest 1H+/4H+/1D+ liquidity and fixed-R diagnostics.

All HTF levels, sweep stages, references, entry signals and frozen targets are
causal. Long-horizon path/exit outcomes are physically separated from features.
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

try:
    from config.loader import TIMEZONE  # type: ignore[attr-defined]  # noqa: E402
except ImportError:
    TIMEZONE = "+8"

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2 import (  # noqa: E402
    MSS2Config,
    R02Config,
    attach_stage_forward_paths,
    attach_structural_exit_outcomes,
    build_first_sweep_lifecycle,
    build_liquidity_levels,
    build_stack_execution_triggers,
    build_sweep_episodes,
    build_sweep_stages,
    classify_liquidity,
    normalize_1m_bars,
    r02_causal_audit,
    split_r02_features_and_labels,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "2.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_LIQUIDITY_POOL_STACK_STRUCTURAL_EXIT_R02"
EDGE_ID = "RESEARCH_ONLY_ICT_MSS2_LIQUIDITY_STACK_EXHAUSTION"
TITLE = "ETH ICT MSS2 Liquidity Pool / Stack Exhaustion + Structural Exit R02"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
BASELINE_ROUNDTRIP_COST = 0.0011


def _comma_ints(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(v.strip()) for v in str(text).split(",") if v.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _comma_floats(text: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(v.strip()) for v in str(text).split(",") if v.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return values


def _parse_tf_spec(text: str) -> tuple[tuple[str, int], ...]:
    mapping = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "1H": 60,
        "4h": 240,
        "4H": 240,
        "1d": 1440,
        "1D": 1440,
    }
    canonical = {5: "5m", 15: "15m", 30: "30m", 60: "1H", 240: "4H", 1440: "1D"}
    out: list[tuple[str, int]] = []
    for token in [v.strip() for v in str(text).split(",") if v.strip()]:
        if token not in mapping:
            raise argparse.ArgumentTypeError(f"unsupported liquidity timeframe: {token}")
        pair = (canonical[mapping[token]], mapping[token])
        if pair not in out:
            out.append(pair)
    if not out:
        raise argparse.ArgumentTypeError("no liquidity timeframes")
    return tuple(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal liquidity-pool stack exhaustion, 1m/2m/5m execution, and structural-liquidity exits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-08-15 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--liquidity-timeframes",
        type=_parse_tf_spec,
        default=(("15m", 15), ("30m", 30), ("1H", 60), ("4H", 240), ("1D", 1440)),
        help="Use 5m explicitly if you want the separate lower-context sensitivity; default keeps R01-comparable 15m+ pools.",
    )
    parser.add_argument("--liquidity-orders", type=_comma_ints, default=(1, 2, 3, 5))
    parser.add_argument("--execution-orders", type=_comma_ints, default=(1, 2, 3))
    parser.add_argument("--execution-minutes", type=_comma_ints, default=(1, 2, 5))
    parser.add_argument("--reference-modes", default="structural")
    parser.add_argument("--pool-tolerances-bps", type=_comma_floats, default=(5.0, 10.0, 20.0))
    parser.add_argument("--episode-gap-minutes", type=int, default=15)
    parser.add_argument("--episode-gap-sensitivity", type=_comma_ints, default=(5, 15, 30))
    parser.add_argument("--stack-thresholds", type=_comma_ints, default=(1, 2, 3, 4))
    parser.add_argument("--target-pool-tolerance-bps", type=float, default=10.0)
    parser.add_argument("--max-confirmation-minutes", type=int, default=180)
    parser.add_argument("--max-fvg-wait-minutes", type=int, default=180)
    parser.add_argument("--exit-censor-minutes", type=int, default=10_080)
    parser.add_argument("--path-horizons-minutes", type=_comma_ints, default=(60, 360, 720, 1440, 2880, 4320, 10_080))
    parser.add_argument("--confluence-tolerance-bps", type=float, default=10.0)
    parser.add_argument("--touch-tolerance-bps", type=float, default=5.0)
    parser.add_argument("--approach-tolerance-bps", type=float, default=25.0)
    parser.add_argument("--sweep-epsilon-bps", type=float, default=0.01)
    parser.add_argument("--mss-break-epsilon-bps", type=float, default=0.01)
    parser.add_argument("--stop-buffer-bps", type=float, default=2.0)
    parser.add_argument("--roundtrip-cost", type=float, default=BASELINE_ROUNDTRIP_COST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--review-sample-size", type=int, default=30_000)
    parser.add_argument("--disable-reclaims", action="store_true")
    parser.add_argument("--disable-mss-market", action="store_true")
    parser.add_argument("--disable-mss-fvg", action="store_true")
    return parser.parse_args(argv)


def _load_ohlcv(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] official OKX naked 1m OHLCV | {args.symbol} | {args.warmup_start_date} -> {args.end_date}",
        flush=True,
    )
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    bars = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    keep = [name for name in ("open", "high", "low", "close", "volume") if name in bars.columns]
    bars = normalize_1m_bars(bars.loc[:, keep])
    print(f"[load] rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _window(frame: pd.DataFrame, time_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ts = pd.to_datetime(frame[time_col], errors="coerce")
    return frame.loc[(ts >= start) & (ts <= end)].reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = frame.copy()
    for col in [c for c in safe.columns if isinstance(c, str) and c.startswith("_consumed_level_prices")]:
        safe[col] = safe[col].map(lambda x: "|".join(f"{float(v):.8f}" for v in x) if isinstance(x, (tuple, list, np.ndarray)) else "")
    safe.to_csv(path, index=False, encoding="utf-8-sig")


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(x.loc[x > 0].sum())
    losses = float(-x.loc[x < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def _path_metric(frame: pd.DataFrame, horizons: Sequence[int]) -> dict[str, object]:
    row: dict[str, object] = {"events": int(len(frame))}
    if frame.empty:
        return row
    for h in horizons:
        ret = pd.to_numeric(frame.get(f"path_close_return_{int(h)}m"), errors="coerce")
        mfe = pd.to_numeric(frame.get(f"path_mfe_{int(h)}m"), errors="coerce")
        mae = pd.to_numeric(frame.get(f"path_mae_{int(h)}m"), errors="coerce")
        valid = ret.dropna()
        row[f"mean_return_{h}m_bps"] = float(valid.mean() * 10_000.0) if len(valid) else np.nan
        row[f"median_return_{h}m_bps"] = float(valid.median() * 10_000.0) if len(valid) else np.nan
        row[f"positive_rate_{h}m"] = float((valid > 0).mean()) if len(valid) else np.nan
        row[f"mean_mfe_{h}m_bps"] = float(mfe.mean() * 10_000.0) if mfe.notna().any() else np.nan
        row[f"mean_mae_{h}m_bps"] = float(mae.mean() * 10_000.0) if mae.notna().any() else np.nan
    return row


def _group_path(frame: pd.DataFrame, group_cols: Sequence[str], horizons: Sequence[int]) -> pd.DataFrame:
    available = [c for c in group_cols if c in frame.columns]
    if frame.empty:
        return pd.DataFrame()
    if not available:
        return pd.DataFrame([_path_metric(frame, horizons)])
    rows: list[dict[str, object]] = []
    for key, part in frame.groupby(available, dropna=False, observed=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(available, keys)}
        row.update(_path_metric(part, horizons))
        rows.append(row)
    return pd.DataFrame(rows)


def _target_metric(frame: pd.DataFrame, target: str) -> dict[str, object]:
    row: dict[str, object] = {"trades": int(len(frame)), "target_type": target}
    if frame.empty:
        return row
    outcome_col = f"target_{target}_outcome"
    if outcome_col not in frame.columns:
        return row
    outcome = frame[outcome_col].astype(str)
    row["no_target_rate"] = float(outcome.eq("no_target").mean())
    available = frame.loc[~outcome.isin(["no_target", "invalid", "invalid_target", ""])].copy()
    row["target_available"] = int(len(available))
    if available.empty:
        return row
    available_outcome = available[outcome_col].astype(str)
    row["target_hit_rate"] = float(available_outcome.eq("target").mean())
    row["stop_rate"] = float(available_outcome.eq("stop").mean())
    row["censored_rate"] = float(available_outcome.eq("censored").mean())
    row["median_target_r"] = float(pd.to_numeric(available.get(f"target_{target}_r_multiple"), errors="coerce").median())
    resolved = available.loc[available_outcome.isin(["target", "stop"])].copy()
    row["resolved"] = int(len(resolved))
    if resolved.empty:
        return row
    hold = pd.to_numeric(resolved.get(f"target_{target}_holding_minutes"), errors="coerce")
    row["median_hold_minutes"] = float(hold.median())
    row["mean_hold_minutes"] = float(hold.mean())
    for suffix in ("base", "cost2x", "cost3x"):
        net = pd.to_numeric(resolved.get(f"target_{target}_net_return_{suffix}"), errors="coerce").dropna()
        row[f"mean_net_{suffix}"] = float(net.mean()) if len(net) else np.nan
        row[f"win_rate_{suffix}"] = float((net > 0).mean()) if len(net) else np.nan
        row[f"pf_{suffix}"] = _pf(net)
    return row


def _group_targets(frame: pd.DataFrame, group_cols: Sequence[str], targets: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    available_cols = [c for c in group_cols if c in frame.columns]
    rows: list[dict[str, object]] = []
    grouped = [((), frame)] if not available_cols else frame.groupby(available_cols, dropna=False, observed=False, sort=True)
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        base = {name: value for name, value in zip(available_cols, keys)}
        for target in targets:
            row = dict(base)
            row.update(_target_metric(part, target))
            rows.append(row)
    return pd.DataFrame(rows)


def _threshold_crossing_rows(
    frame: pd.DataFrame,
    *,
    tolerance_bps: float,
    thresholds: Sequence[int],
    unit: str,
) -> pd.DataFrame:
    """Select first causal stage/trade after each cumulative pool threshold is known."""
    if frame.empty:
        return pd.DataFrame()
    token = str(float(tolerance_bps)).replace(".", "p")
    col = f"price_pools_{token}bp_cum"
    if col not in frame.columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for threshold in thresholds:
        part = frame.loc[pd.to_numeric(frame[col], errors="coerce").fillna(0).ge(int(threshold))].copy()
        if part.empty:
            continue
        if unit == "path":
            keys = [c for c in ("episode_id", "trade_direction") if c in part.columns]
            sort_cols = [c for c in ("sweep_pos_1m", "stage_id") if c in part.columns]
        elif unit == "trade":
            keys = [c for c in ("episode_id", "execution_minutes", "trigger_type", "trade_direction") if c in part.columns]
            sort_cols = [c for c in ("entry_pos_1m", "trade_event_id") if c in part.columns]
        else:
            raise ValueError("unit must be path/trade")
        if sort_cols:
            part = part.sort_values(sort_cols, kind="stable")
        if keys:
            part = part.drop_duplicates(keys, keep="first")
        part["pool_tolerance_bps"] = float(tolerance_bps)
        part["pool_threshold"] = int(threshold)
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _attach_calendar(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    ts = pd.to_datetime(out[time_col], errors="coerce")
    out["year"] = ts.dt.year.astype("Int16")
    out["quarter"] = ts.dt.to_period("Q").astype("string")
    out["month"] = ts.dt.month.astype("Int8")
    return out


def _holding_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        [-np.inf, 60, 360, 720, 1440, 2880, 4320, np.inf],
        labels=["<=1h", "1-6h", "6-12h", "12-24h", "1-2d", "2-3d", ">3d"],
        right=True,
    ).astype("string")


def _write_report_readme(
    out_dir: Path,
    *,
    audit: dict[str, int],
    cfg: R02Config,
    base_cfg: MSS2Config,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    lines = [
        f"# {TITLE}",
        "",
        "## R02 research decision question",
        "R01 rejected a universal sweep->MSS->FVG strategy but found that simultaneous consumption of multiple independent liquidity pools was much more informative than a single swing sweep. R02 tests that hypothesis without treating one level as one independent event.",
        "",
        "## Event semantics",
        "- A sweep stage is one unique 1m bar/direction, regardless of how many swing levels it consumes.",
        "- Swept prices are clustered at fixed 5/10/20bp sensitivities; these are not optimized entry parameters.",
        "- A sweep episode only continues when a same-direction stage arrives inside the fixed gap and extends the episode extreme.",
        "- Cumulative episode fields are current/past only. The eventual future size of an episode is never backfilled.",
        "- 1m/2m/5m execution is compared on the same causal 1m liquidity lifecycle.",
        "",
        "## Exit semantics",
        "- There is no time-profit exit.",
        "- Stop is structural: beyond the complete episode-to-signal extreme plus a small fixed buffer.",
        "- Opposing liquidity targets are frozen from the active book at entry time.",
        "- Targets: nearest level, nearest >=2-level pool, nearest >=2-level >=2-timeframe pool, nearest 1H+, 4H+, 1D+, plus fixed-R diagnostics.",
        "- Seven days is right-censoring only. Unresolved trades remain censored and are not force-closed or assigned zero return.",
        "- FVG limit fill-bar ambiguity is pessimistic: stop may count on fill bar, target may not.",
        "",
        "## Anti-lookahead",
        "- HTF swing order only upgrades when its right-side confirmation is actually available.",
        "- MSS reference must already be confirmed before the execution bar containing the sweep begins.",
        "- Dynamic opposing-liquidity books remove a swept target at sweep_pos+1, never at the start of the bar that will later sweep it.",
        "- Forward path and target outcomes are physically split from causal feature files.",
        "",
        "## Counts",
    ]
    for key, value in counts.items():
        lines.append(f"- `{key}`: {value:,}")
    lines += ["", "## Causal audit"]
    for key, value in audit.items():
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## R02 config",
        "```json",
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Base structure config",
        "```json",
        json.dumps(asdict(base_cfg), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation rule",
        "Do not promote a subgroup because it has the single best PF. Prefer a broad parameter plateau across pool tolerances, multiple years, independent episodes, and realistic costs. 2026 degradation must be called out rather than optimized away.",
        "",
        "## Run command",
        f"`python research\\ict\\mss2\\02_liquidity_pool_stack_structural_exit.py --symbol {args.symbol} --warmup-start-date {args.warmup_start_date} --start-date {args.start_date} --end-date \"{args.end_date}\"`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if any(int(v) not in {1, 2, 5} for v in args.execution_minutes):
        raise ValueError("R02 default engine supports execution-minutes 1,2,5 only")
    reference_modes = tuple(v.strip().lower() for v in str(args.reference_modes).split(",") if v.strip())
    if not reference_modes or any(v not in {"recent", "structural"} for v in reference_modes):
        raise ValueError("reference modes must be recent and/or structural")
    if int(args.exit_censor_minutes) < max(args.path_horizons_minutes):
        raise ValueError("exit-censor-minutes must be >= max path horizon")

    base_cfg = MSS2Config(
        liquidity_timeframes=tuple(args.liquidity_timeframes),
        liquidity_confirmation_orders=tuple(args.liquidity_orders),
        execution_confirmation_orders=tuple(args.execution_orders),
        confluence_tolerance_bps=float(args.confluence_tolerance_bps),
        touch_tolerance_bps=float(args.touch_tolerance_bps),
        approach_tolerance_bps=float(args.approach_tolerance_bps),
        sweep_epsilon_bps=float(args.sweep_epsilon_bps),
        mss_break_epsilon_bps=float(args.mss_break_epsilon_bps),
        max_mss_minutes=int(args.max_confirmation_minutes),
        max_entry_wait_minutes=int(args.max_fvg_wait_minutes),
        max_outcome_minutes=min(180, int(args.max_confirmation_minutes)),
        stop_buffer_bps=float(args.stop_buffer_bps),
    ).validate()
    cfg = R02Config(
        pool_tolerances_bps=tuple(float(v) for v in args.pool_tolerances_bps),
        episode_gap_minutes=int(args.episode_gap_minutes),
        max_confirmation_minutes=int(args.max_confirmation_minutes),
        max_fvg_wait_minutes=int(args.max_fvg_wait_minutes),
        exit_censor_minutes=int(args.exit_censor_minutes),
        path_horizons_minutes=tuple(int(v) for v in args.path_horizons_minutes),
        target_pool_tolerance_bps=float(args.target_pool_tolerance_bps),
        stop_buffer_bps=float(args.stop_buffer_bps),
        mss_break_epsilon_bps=float(args.mss_break_epsilon_bps),
    ).validate()
    show_progress = not bool(args.no_progress)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    bars = _load_ohlcv(args)
    print(f"[stage] causal liquidity candidates: {base_cfg.liquidity_timeframes}", flush=True)
    levels = build_liquidity_levels(bars, base_cfg)
    print(f"[liquidity] candidates={len(levels):,}", flush=True)
    print("[stage] first true 1m sweep lifecycle + causal taxonomy", flush=True)
    lifecycle = build_first_sweep_lifecycle(bars, levels, base_cfg, show_progress=show_progress)
    lifecycle = classify_liquidity(lifecycle, base_cfg)

    future_cols = [c for c in lifecycle.columns if c.startswith("future_eventual_order_")]
    causal_cols = [c for c in lifecycle.columns if c not in future_cols]
    causal_lifecycle = lifecycle.loc[:, causal_cols].copy()
    _write_csv(causal_lifecycle, out_dir / "01_liquidity_lifecycle_causal.csv")
    if future_cols:
        _write_csv(lifecycle.loc[:, ["level_id", *future_cols]], out_dir / "02_liquidity_future_order_audit_labels.csv")

    print("[stage] collapse level events -> unique 1m sweep stages", flush=True)
    stages = build_sweep_stages(
        bars, causal_lifecycle, config=cfg, project_timezone=TIMEZONE, show_progress=show_progress
    )
    stages = _window(stages, "sweep_available_time_1m", start, end)
    _write_csv(stages, out_dir / "03_sweep_stages_causal.csv")
    print(f"[stages] independent sweep bars={len(stages):,}", flush=True)

    sensitivity_parts: list[pd.DataFrame] = []
    primary_episode_stages = pd.DataFrame()
    primary_paths = pd.DataFrame()
    for gap in args.episode_gap_sensitivity:
        local_cfg = R02Config(**{**asdict(cfg), "episode_gap_minutes": int(gap)}).validate()
        print(f"[stage] episode gap sensitivity={gap}m", flush=True)
        episode_stages = build_sweep_episodes(stages, config=local_cfg)
        paths = attach_stage_forward_paths(bars, episode_stages, config=local_cfg, show_progress=show_progress)
        paths["episode_gap_minutes"] = int(gap)
        crossings = []
        for tolerance in cfg.pool_tolerances_bps:
            crossing = _threshold_crossing_rows(
                paths,
                tolerance_bps=tolerance,
                thresholds=args.stack_thresholds,
                unit="path",
            )
            if not crossing.empty:
                crossings.append(crossing)
        if crossings:
            sensitivity_parts.append(pd.concat(crossings, ignore_index=True, sort=False))
        if int(gap) == int(cfg.episode_gap_minutes):
            primary_episode_stages = episode_stages
            primary_paths = paths
    if primary_episode_stages.empty:
        # Permit custom primary gap not included in sensitivity list.
        primary_episode_stages = build_sweep_episodes(stages, config=cfg)
        primary_paths = attach_stage_forward_paths(bars, primary_episode_stages, config=cfg, show_progress=show_progress)
        primary_paths["episode_gap_minutes"] = int(cfg.episode_gap_minutes)

    _write_csv(primary_episode_stages, out_dir / "04_sweep_episode_stages_causal.csv")
    path_label_cols = [c for c in primary_paths.columns if c.startswith(("path_close_return_", "path_mfe_", "path_mae_", "path_entry_"))]
    path_ids = [c for c in ("stage_id", "episode_id") if c in primary_paths.columns]
    _write_csv(primary_paths.loc[:, list(dict.fromkeys([*path_ids, *path_label_cols]))], out_dir / "05_sweep_long_horizon_labels.csv")

    print("[stage] execution triggers 1m/2m/5m", flush=True)
    trade_parts: list[pd.DataFrame] = []
    for execution_minutes in args.execution_minutes:
        print(f"[execution] {execution_minutes}m", flush=True)
        trades = build_stack_execution_triggers(
            bars,
            primary_episode_stages,
            execution_minutes=int(execution_minutes),
            base_config=base_cfg,
            config=cfg,
            reference_modes=reference_modes,
            include_reclaims=not bool(args.disable_reclaims),
            include_mss_market=not bool(args.disable_mss_market),
            include_mss_fvg=not bool(args.disable_mss_fvg),
            project_timezone=TIMEZONE,
            show_progress=show_progress,
        )
        trades = _window(trades, "entry_time", start, end)
        print(f"[execution] {execution_minutes}m trigger rows={len(trades):,}", flush=True)
        if trades.empty:
            continue
        outcomes = attach_structural_exit_outcomes(
            bars,
            causal_lifecycle,
            trades,
            config=cfg,
            roundtrip_cost=float(args.roundtrip_cost),
            show_progress=show_progress,
        )
        trade_parts.append(outcomes)

    all_trades = pd.concat(trade_parts, ignore_index=True, sort=False) if trade_parts else pd.DataFrame()
    if not all_trades.empty:
        all_trades = _attach_calendar(all_trades, "entry_time")
        features, labels = split_r02_features_and_labels(all_trades)
        _write_csv(features, out_dir / "10_trade_features_causal.csv")
        _write_csv(labels, out_dir / "11_trade_structural_exit_labels.csv")
        _write_csv(all_trades.head(max(0, int(args.review_sample_size))), out_dir / "12_trade_review_sample.csv")
    else:
        _write_csv(all_trades, out_dir / "10_trade_features_causal.csv")
        _write_csv(all_trades, out_dir / "11_trade_structural_exit_labels.csv")

    audit = r02_causal_audit(primary_episode_stages, all_trades)
    audit_fail_keys = (
        "stage_available_before_sweep_close",
        "signal_before_sweep_exec_available",
        "entry_before_signal_available",
        "mss_reference_available_after_known_cutoff",
        "episode_start_after_stage",
    )
    if any(int(audit.get(key, 0)) for key in audit_fail_keys):
        raise RuntimeError(f"R02 causal audit failed: {audit}")

    print("[stage] summaries: independent pool-threshold crossings, context, calendar and exits", flush=True)
    summaries: dict[str, pd.DataFrame] = {}
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True, sort=False) if sensitivity_parts else pd.DataFrame()
    sensitivity = _attach_calendar(sensitivity, "sweep_available_time_1m") if not sensitivity.empty else sensitivity
    horizons = tuple(int(v) for v in cfg.path_horizons_minutes)
    summaries["episode_gap_pool_threshold_paths"] = _group_path(
        sensitivity,
        ["episode_gap_minutes", "pool_tolerance_bps", "pool_threshold", "trade_direction"],
        horizons,
    )
    summaries["pool_threshold_year_paths"] = _group_path(
        sensitivity,
        ["episode_gap_minutes", "pool_tolerance_bps", "pool_threshold", "year", "trade_direction"],
        horizons,
    )
    summaries["pool_threshold_context_paths"] = _group_path(
        sensitivity,
        ["episode_gap_minutes", "pool_tolerance_bps", "pool_threshold", "max_source_timeframe_min_cum", "distinct_timeframes_cum", "trade_direction"],
        horizons,
    )

    primary_paths = _attach_calendar(primary_paths, "sweep_available_time_1m")
    summaries["stage_pool_count"] = _group_path(
        primary_paths,
        ["price_pools_10p0bp_stage", "trade_direction"],
        horizons,
    )
    summaries["cumulative_pool_count"] = _group_path(
        primary_paths,
        ["price_pools_10p0bp_cum", "trade_direction"],
        horizons,
    )
    summaries["source_tf_context"] = _group_path(
        primary_paths,
        ["max_source_timeframe_min_cum", "distinct_timeframes_cum", "trade_direction"],
        horizons,
    )
    summaries["weekday_weekend_paths"] = _group_path(
        primary_paths,
        ["is_weekend_ny", "trade_direction"],
        horizons,
    )
    summaries["session_paths"] = _group_path(
        primary_paths,
        ["session_primary", "trade_direction"],
        horizons,
    )
    summaries["year_paths"] = _group_path(primary_paths, ["year", "trade_direction"], horizons)

    targets = ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r1p0", "r2p0", "r3p0", "r5p0")
    if not all_trades.empty:
        independent = all_trades.loc[pd.to_numeric(all_trades.get("episode_first_entry_flag"), errors="coerce").fillna(0).eq(1)].copy()
        summaries["exit_overall_independent"] = _group_targets(
            independent,
            ["execution_minutes", "trigger_type", "trade_direction"],
            targets,
        )
        summaries["exit_year_independent"] = _group_targets(
            independent,
            ["execution_minutes", "trigger_type", "year", "trade_direction"],
            targets,
        )
        summaries["exit_session_independent"] = _group_targets(
            independent,
            ["execution_minutes", "trigger_type", "session_primary", "trade_direction"],
            targets,
        )
        summaries["exit_context_independent"] = _group_targets(
            independent,
            ["execution_minutes", "trigger_type", "max_source_timeframe_min_cum", "distinct_timeframes_cum", "trade_direction"],
            targets,
        )

        threshold_trade_parts: list[pd.DataFrame] = []
        for tolerance in cfg.pool_tolerances_bps:
            crossed = _threshold_crossing_rows(
                all_trades,
                tolerance_bps=tolerance,
                thresholds=args.stack_thresholds,
                unit="trade",
            )
            if not crossed.empty:
                threshold_trade_parts.append(crossed)
        threshold_trades = pd.concat(threshold_trade_parts, ignore_index=True, sort=False) if threshold_trade_parts else pd.DataFrame()
        summaries["exit_pool_threshold"] = _group_targets(
            threshold_trades,
            ["pool_tolerance_bps", "pool_threshold", "execution_minutes", "trigger_type", "trade_direction"],
            targets,
        )
        summaries["exit_pool_threshold_year"] = _group_targets(
            threshold_trades,
            ["pool_tolerance_bps", "pool_threshold", "execution_minutes", "trigger_type", "year", "trade_direction"],
            targets,
        )

        # Holding-time distribution is descriptive only; it never changes the exit.
        hold_rows: list[pd.DataFrame] = []
        for target in ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440"):
            outcome_col = f"target_{target}_outcome"
            hold_col = f"target_{target}_holding_minutes"
            if outcome_col not in independent.columns or hold_col not in independent.columns:
                continue
            resolved = independent.loc[independent[outcome_col].astype(str).isin(["target", "stop"])].copy()
            if resolved.empty:
                continue
            resolved["target_type"] = target
            resolved["holding_bucket"] = _holding_bucket(resolved[hold_col])
            hold_rows.append(
                resolved.groupby(["target_type", "holding_bucket", "trade_direction"], dropna=False, observed=False)
                .size().rename("trades").reset_index()
            )
        summaries["holding_distribution"] = pd.concat(hold_rows, ignore_index=True, sort=False) if hold_rows else pd.DataFrame()

        # Exact 1m/2m/5m overlap by stage + trigger; useful to learn whether slower
        # execution confirms the same exhaustion episodes or a different subset.
        overlap_base = independent[[c for c in ("episode_id", "stage_id", "execution_minutes", "trigger_type", "trade_direction", "entry_time") if c in independent.columns]].copy()
        if not overlap_base.empty:
            overlap_base["present"] = 1
            summaries["execution_overlap"] = (
                overlap_base.pivot_table(
                    index=["episode_id", "stage_id", "trigger_type", "trade_direction"],
                    columns="execution_minutes",
                    values="present",
                    aggfunc="max",
                    fill_value=0,
                )
                .reset_index()
            )
    else:
        for name in (
            "exit_overall_independent", "exit_year_independent", "exit_session_independent",
            "exit_context_independent", "exit_pool_threshold", "exit_pool_threshold_year",
            "holding_distribution", "execution_overlap",
        ):
            summaries[name] = pd.DataFrame()

    for name, frame in summaries.items():
        _write_csv(frame, out_dir / f"20_summary_{name}.csv")

    counts = {
        "liquidity_candidates": int(len(levels)),
        "causal_lifecycle_levels": int(len(causal_lifecycle)),
        "unique_sweep_stages": int(len(stages)),
        "primary_episode_stage_rows": int(len(primary_episode_stages)),
        "primary_unique_episodes": int(primary_episode_stages["episode_id"].nunique()) if "episode_id" in primary_episode_stages.columns else 0,
        "trade_trigger_rows": int(len(all_trades)),
        "independent_episode_first_entries": int(pd.to_numeric(all_trades.get("episode_first_entry_flag"), errors="coerce").fillna(0).sum()) if not all_trades.empty else 0,
    }
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "script_version": SCRIPT_VERSION,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "research_start_date": args.start_date,
        "research_end_date": args.end_date,
        "roundtrip_cost": float(args.roundtrip_cost),
        "project_timezone": str(TIMEZONE),
        "base_config": asdict(base_cfg),
        "r02_config": asdict(cfg),
        "episode_gap_sensitivity": list(args.episode_gap_sensitivity),
        "stack_thresholds": list(args.stack_thresholds),
        "causal_audit": audit,
        "counts": counts,
        "important_semantics": [
            "statistical unit is a unique sweep stage / causal sweep episode, not one row per swept swing",
            "nearby swept levels are clustered at fixed 5/10/20bp tolerances only for robustness sensitivity",
            "episode cumulative fields never include future stages",
            "time is never a profit-taking exit; exit_censor_minutes is right-censoring only",
            "structural stop is beyond the observed episode-to-signal extreme",
            "opposing-liquidity targets are selected from the active book and frozen at entry",
            "a level swept during a bar is removed from the active target book only at the next bar start",
            "MSS references are confirmed before the sweep-containing execution bar starts",
            "FVG limit fill-bar ambiguity is pessimistic: stop can count on fill bar and target cannot",
            "session/weekend/hour fields are diagnostics only and never admission filters",
            "forward path and exit labels are physically separated from causal trade features",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_report_readme(out_dir, audit=audit, cfg=cfg, base_cfg=base_cfg, args=args, counts=counts)
    print(f"[audit] {audit}", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
