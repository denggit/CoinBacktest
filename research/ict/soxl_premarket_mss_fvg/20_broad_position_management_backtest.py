#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R20: broad-entry position-management backtest.

This is deliberately a strategy/backtest pass, not another narrowing atlas.
R20 starts from the broad first-visible causal MSS/structure-break universe: the
earliest visible 1m or 2m break per liquidity path, including wick-only breaks
as a weaker setup rather than deleting them.  The entry is the next available
1m open after the break becomes known.  R20 then asks whether causal position
management can turn that near-daily opportunity stream into a stable strategy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.research_common.ict.broad_position_management import (
    BroadPositionManagementConfig,
    SCENARIOS,
    replay_position_scenarios,
    select_discovery_policy,
    summarize_management,
    summarize_periods,
)
from src.research_common.ict.premarket_mss_fvg import NY_TZ
from src.research_common.ict.spot_perp_overlap import densify_equity_minutes_causally
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_RANGE_MODEL = "prominent_15m_pair_0830"
DEFAULT_ENTRY_ARCHETYPE = "mss_first_visible_any_break_next_open_market"
DEFAULT_VISIBLE_SWING_PERCENTILE = 0.50
DEFAULT_R16_CACHE = "data/reports/research/ict/soxl/mss/r16_entry_archetype_survival_atlas_alpaca_2023_2026_08"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r20_broad_position_management_backtest_alpaca_2023_2026_08"


def _csv_numbers(text: str, *, cast=float) -> tuple:
    return tuple(cast(x.strip()) for x in str(text).split(",") if x.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R20 broad position-management backtest")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--r16-cache-dir", default=DEFAULT_R16_CACHE)
    p.add_argument("--r15-cache-dir", default="", help="Optional explicit R15 cache; otherwise resolved from the R16 manifest")
    p.add_argument("--visible-swing-percentile", type=float, default=DEFAULT_VISIBLE_SWING_PERCENTILE)
    p.add_argument("--range-model", default=DEFAULT_RANGE_MODEL)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1,1.5,2")
    p.add_argument("--entry-delay-minutes", default="0,1,2")
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=1.0)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--minimum-trades-per-session", type=float, default=0.5)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < float(args.round_trip_cost) < 0.05:
        raise ValueError("round_trip_cost must be in (0, 0.05)")
    if not 0 < float(args.risk_fraction) <= 0.05:
        raise ValueError("risk_fraction must be in (0, 0.05]")
    if not 0 < float(args.max_notional_multiple) <= 5:
        raise ValueError("max_notional_multiple must be in (0, 5]")
    if float(args.minimum_trades_per_session) < 0.5:
        raise ValueError("R20 hard rule: minimum_trades_per_session cannot be below 0.5")
    if not 0.0 <= float(args.visible_swing_percentile) <= 1.0:
        raise ValueError("visible_swing_percentile must be in [0, 1]")


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
    end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
    loader = AlpacaStockLoader(
        symbol=args.alpaca_symbol,
        timeframe="1Min",
        feed=args.alpaca_feed,
        adjustment=args.alpaca_adjustment,
        data_dir=args.data_dir,
    )
    raw = loader.fetch_data_by_date_range(
        start_ny.tz_convert("UTC"),
        end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1),
        local_only=bool(args.local_only),
    )
    if raw.empty:
        raise RuntimeError("Alpaca loader returned no data")
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    bars = raw.copy()
    bars.index = idx.tz_convert(NY_TZ)
    bars.index.name = "bar_start_ny"
    bars = densify_equity_minutes_causally(bars)
    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 510) & (mins < 990)].copy()  # 08:30 -> 16:30 ET only.
    print(f"[load] Alpaca rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _normalize_cache_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def _load_r16_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    cache = Path(args.r16_cache_dir)
    mf_path = cache / "13_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"R16 manifest missing: {mf_path}")
    manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    if str(manifest.get("start_date")) != str(args.start_date) or str(manifest.get("end_date")) != str(args.end_date):
        raise RuntimeError(f"R16 cache window {manifest.get('start_date')}->{manifest.get('end_date')} does not match requested window")
    return cache, manifest


def _resolve_r15_cache(args: argparse.Namespace, r16_cache: Path, r16_manifest: dict[str, object]) -> Path:
    candidates: list[Path] = []
    if str(args.r15_cache_dir).strip():
        candidates.append(_normalize_cache_path(args.r15_cache_dir))
    raw = str(r16_manifest.get("r15_cache", "")).strip()
    if raw:
        candidates.append(_normalize_cache_path(raw))
    candidates.append(r16_cache.parent / "r15_daily_liquidity_traversal_path_atlas_alpaca_2023_2026_08")
    for candidate in candidates:
        if (candidate / "06_causal_mss_narratives.csv").exists() and (candidate / "03_daily_path_outcomes.csv").exists():
            return candidate
    raise FileNotFoundError("R15 cache could not be resolved from --r15-cache-dir, R16 manifest, or sibling default")


def _valid_sessions_from_cache(r16_cache: Path, r15_cache: Path, manifest: dict[str, object]) -> list[str]:
    for path in (r16_cache / "03_daily_paths_with_approach_features.csv", r15_cache / "03_daily_path_outcomes.csv"):
        if path.exists():
            dates = pd.read_csv(path, usecols=["ny_date"])["ny_date"].dropna().astype(str).drop_duplicates().sort_values().tolist()
            if dates:
                return dates
    idx = pd.bdate_range(str(manifest.get("start_date")), str(manifest.get("end_date")))
    return idx.strftime("%Y-%m-%d").tolist()


def _load_broad_signals(args: argparse.Namespace, r15_cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_file = r15_cache / "03_daily_path_outcomes.csv"
    narrative_file = r15_cache / "06_causal_mss_narratives.csv"
    path_header = pd.read_csv(path_file, nrows=0).columns.tolist()
    narrative_header = pd.read_csv(narrative_file, nrows=0).columns.tolist()
    path_cols = [c for c in ["ny_date", "range_model", "path_event_id", "first_raid_side", "first_raid_time", "target_price", "source_level_price"] if c in path_header]
    narrative_cols = [c for c in [
        "ny_date", "range_model", "path_event_id", "event_id", "trade_side", "execution_tf", "execution_tf_minutes",
        "break_available_time", "break_bar_start", "break_wick_cross", "break_close_cross",
        "terminal_extreme_time", "terminal_extreme_price", "mss_reference_time", "mss_reference_available_time",
        "causal_visibility_percentile", "directional_bar_fraction", "path_efficiency", "break_overshoot_abs",
        "first_raid_time", "target_price", "source_level_price",
    ] if c in narrative_header]
    paths = pd.read_csv(path_file, usecols=path_cols, low_memory=False)
    narratives = pd.read_csv(narrative_file, usecols=narrative_cols, low_memory=False)
    paths = paths.loc[paths["range_model"].astype(str).eq(str(args.range_model))].copy()
    q = narratives.loc[narratives["range_model"].astype(str).eq(str(args.range_model))].copy()
    vis = pd.to_numeric(q.get("causal_visibility_percentile"), errors="coerce")
    q = q.loc[vis.ge(float(args.visible_swing_percentile))].copy()
    if "break_wick_cross" in q.columns:
        wick = q["break_wick_cross"].astype(str).str.lower().isin(["true", "1", "yes"])
        q = q.loc[wick].copy()
    if q.empty:
        return q, paths
    q["break_available_time"] = pd.to_datetime(q["break_available_time"], errors="coerce", utc=True)
    q["execution_tf_minutes"] = pd.to_numeric(q.get("execution_tf_minutes"), errors="coerce")
    q = q.dropna(subset=["break_available_time", "path_event_id"]).copy()
    # Broad causal definition: retain the first visible structure break on either
    # 1m or 2m.  Close confirmation is a feature, not an admission filter.
    sort_cols = ["path_event_id", "break_available_time", "execution_tf_minutes"]
    if "mss_reference_available_time" in q.columns:
        q["mss_reference_available_time"] = pd.to_datetime(q["mss_reference_available_time"], errors="coerce", utc=True)
        sort_cols.append("mss_reference_available_time")
    q = q.sort_values(sort_cols, kind="mergesort").drop_duplicates("path_event_id", keep="first").reset_index(drop=True)
    # Canonical path metadata wins where a narrow narrative omits a field.
    meta_cols = [c for c in ["path_event_id", "target_price", "source_level_price", "first_raid_time", "first_raid_side"] if c in paths.columns]
    if meta_cols:
        meta = paths[meta_cols].drop_duplicates("path_event_id")
        for c in meta_cols:
            if c != "path_event_id" and c in q.columns:
                q = q.drop(columns=[c])
        q = q.merge(meta, on="path_event_id", how="left", validate="one_to_one")
    return q, paths


def _materialize_broad_market_entries(signals: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    q = signals.copy()
    signal = pd.to_datetime(q["break_available_time"], errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    idx = pd.DatetimeIndex(bars.index)
    pos = np.searchsorted(idx.asi8, pd.DatetimeIndex(signal).asi8, side="left")
    valid = pos < len(idx)
    fill_time = pd.Series(pd.NaT, index=q.index, dtype="datetime64[ns, America/New_York]")
    entry_price = pd.Series(np.nan, index=q.index, dtype=float)
    if valid.any():
        vi = np.flatnonzero(valid)
        p = pos[valid]
        fill_time.iloc[vi] = idx[p]
        entry_price.iloc[vi] = pd.to_numeric(bars["open"], errors="coerce").to_numpy(float)[p]
    same_day = fill_time.dt.strftime("%Y-%m-%d").eq(q["ny_date"].astype(str))
    filled = valid & same_day.to_numpy() & np.isfinite(entry_price.to_numpy(float))
    q["entry_archetype"] = DEFAULT_ENTRY_ARCHETYPE
    q["entry_order_type"] = "market_next_open"
    q["entry_available_time"] = signal
    q["entry_price"] = entry_price
    q["entry_price_replay"] = entry_price
    q["fill_time"] = fill_time
    q["filled"] = filled
    q["stop_price"] = pd.to_numeric(q.get("terminal_extreme_price"), errors="coerce")
    q["target_price"] = pd.to_numeric(q.get("target_price"), errors="coerce")
    q["close_confirmed_mss"] = q.get("break_close_cross", False).astype(str).str.lower().isin(["true", "1", "yes"]) if "break_close_cross" in q.columns else False
    q["broad_signal_source_tf"] = q.get("execution_tf", q.get("execution_tf_minutes", "")).astype(str)
    return q.reset_index(drop=True)

def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _funnel(paths: pd.DataFrame, signals: pd.DataFrame, base: pd.DataFrame, valid_sessions: list[str]) -> pd.DataFrame:
    n = max(1, len(valid_sessions))
    path_count = int(paths["path_event_id"].nunique()) if not paths.empty and "path_event_id" in paths else 0
    signal_count = int(signals["path_event_id"].nunique()) if not signals.empty else 0
    fill_count = int(base.loc[base["filled"].fillna(False).astype(bool), "path_event_id"].nunique()) if not base.empty else 0
    close_count = int(base.loc[base["filled"].fillna(False).astype(bool) & base["close_confirmed_mss"].fillna(False).astype(bool), "path_event_id"].nunique()) if not base.empty and "close_confirmed_mss" in base else 0
    return pd.DataFrame([
        {"stage": "valid_sessions", "count": len(valid_sessions), "per_session": 1.0, "retention_vs_paths": np.nan},
        {"stage": "prominent_15m_liquidity_paths", "count": path_count, "per_session": path_count / n, "retention_vs_paths": 1.0 if path_count else np.nan},
        {"stage": "first_visible_1m_or_2m_structure_break", "count": signal_count, "per_session": signal_count / n, "retention_vs_paths": signal_count / max(1, path_count)},
        {"stage": "next_open_market_fills", "count": fill_count, "per_session": fill_count / n, "retention_vs_paths": fill_count / max(1, path_count)},
        {"stage": "close_confirmed_subset_diagnostic_only", "count": close_count, "per_session": close_count / n, "retention_vs_paths": close_count / max(1, path_count)},
    ])

def _decision(selection: dict[str, object], period: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    policy = str(selection.get("selected_policy", ""))
    if not policy:
        return {**selection, "decision": "REJECT_NO_BROAD_POLICY", "reason": "no Discovery-positive management policy survived the >=0.5 trades/session gate"}
    p = period.loc[
        period["management_scenario"].astype(str).eq(policy)
        & pd.to_numeric(period["cost_multiple"], errors="coerce").eq(1.0)
        & pd.to_numeric(period["entry_delay_minutes"], errors="coerce").eq(0)
    ].copy()
    by_period = {str(r["period"]): r for _, r in p.iterrows()}
    required = ["validation_2025", "forward_2026"]
    oos_ok = all(
        k in by_period
        and float(by_period[k].get("trades_per_session", 0.0)) >= float(args.minimum_trades_per_session)
        and float(by_period[k].get("mean_net_return", np.nan)) > 0
        and float(by_period[k].get("profit_factor", np.nan)) > 1.0
        for k in required
    )
    stress = summary.loc[
        summary["management_scenario"].astype(str).eq(policy)
        & pd.to_numeric(summary["entry_delay_minutes"], errors="coerce").eq(0)
        & pd.to_numeric(summary["cost_multiple"], errors="coerce").eq(2.0)
    ]
    cost2x_ok = bool(len(stress) and float(stress.iloc[0].get("mean_net_return", np.nan)) > 0 and float(stress.iloc[0].get("profit_factor", np.nan)) > 1.0)
    decision = "PROMOTE_TO_EXECUTABLE_STRATEGY_BACKTEST" if oos_ok and cost2x_ok else "REJECT_OR_REWORK_POSITION_MANAGEMENT"
    return {
        **selection,
        "decision": decision,
        "oos_2025_2026_positive": bool(oos_ok),
        "cost_2x_positive_full_window": bool(cost2x_ok),
        "hard_frequency_floor": float(args.minimum_trades_per_session),
    }



def _selected_policy_detail(managed: pd.DataFrame, policy: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not policy:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    q = managed.loc[
        managed["management_scenario"].astype(str).eq(policy)
        & pd.to_numeric(managed["cost_multiple"], errors="coerce").eq(1.0)
        & pd.to_numeric(managed["entry_delay_minutes"], errors="coerce").eq(0)
        & managed["managed"].fillna(False).astype(bool)
    ].copy()
    if q.empty:
        return q, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    q["managed_fill_time"] = pd.to_datetime(q["managed_fill_time"], errors="coerce", utc=True)
    q = q.sort_values("managed_fill_time", kind="mergesort").reset_index(drop=True)
    entry = pd.to_numeric(q["managed_entry_price"], errors="coerce")
    stop = pd.to_numeric(q["managed_initial_stop"], errors="coerce")
    risk_pct = (entry - stop).abs() / entry
    q["notional_multiple"] = (float(args.risk_fraction) / risk_pct).clip(upper=float(args.max_notional_multiple))
    q["account_return"] = (pd.to_numeric(q["management_net_return"], errors="coerce").fillna(0.0) * q["notional_multiple"]).clip(lower=-0.999)
    q["capital"] = float(args.initial_capital) * (1.0 + q["account_return"]).cumprod()
    q["equity_peak"] = q["capital"].cummax()
    q["drawdown"] = q["capital"] / q["equity_peak"] - 1.0
    local = q["managed_fill_time"].dt.tz_convert(NY_TZ)
    q["year"] = local.dt.year
    q["month"] = local.dt.strftime("%Y-%m")
    yearly = q.groupby("year", sort=True).agg(
        trades=("path_event_id", "size"),
        account_return=("account_return", lambda x: float(np.prod(1.0 + x.to_numpy(float)) - 1.0)),
        mean_trade_return=("management_net_return", "mean"),
    ).reset_index()
    monthly = q.groupby("month", sort=True).agg(
        trades=("path_event_id", "size"),
        account_return=("account_return", lambda x: float(np.prod(1.0 + x.to_numpy(float)) - 1.0)),
        mean_trade_return=("management_net_return", "mean"),
    ).reset_index()
    stress = managed.loc[managed["management_scenario"].astype(str).eq(policy)].copy()
    return q, monthly, yearly, stress

def _design_text(args: argparse.Namespace) -> str:
    return f"""# SOXL ICT R20 — Broad Position-Management Backtest

- This is a strategy/backtest pass, not a new hard-filter atlas.
- Broad entry: earliest first-visible 1m/2m causal structure break per path -> next available 1m open, `{args.range_model}`.
- Visibility floor {float(args.visible_swing_percentile):.2f} defines a tradable swing; close confirmation is diagnostic only and never filters entry.
- Hard frequency floor: >= {float(args.minimum_trades_per_session):.2f} filled trades per valid session.
- No event-probability threshold, session filter, HTF filter, CVD filter, or profitability bucket can remove trades.
- All management policies see the same base entry universe.
- Policies: {', '.join(SCENARIOS)}.
- +1R partial/protection and +2R lock are predeclared lifecycle mechanisms, not a parameter grid.
- A stop change triggered by a 1m bar becomes active only on the next 1m bar.
- 2m structural trail uses only pivots whose confirmation_available_time is already known at the current 1m bar start.
- Same-minute stop vs partial/TP ambiguity resolves to stop.
- Main TP remains the opposite external liquidity. No 25/50/75 dealing-range target is used.
- Baseline round-trip cost: {float(args.round_trip_cost):.4%}; stress multipliers: {args.cost_multipliers}.
- Entry-delay stress: {args.entry_delay_minutes} minutes.
- Policy selection uses Discovery 2023H2-2024 only; 2025 and 2026 are evaluation only.
"""


def _synthetic_bars() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    idx = pd.date_range("2026-06-02 08:30", "2026-06-02 16:29", freq="1min", tz=NY_TZ)
    close = np.full(len(idx), 100.0)
    # 09:30 entry; rally above +1R, then retrace to entry, then later recover.
    pos = int(np.where(idx == pd.Timestamp("2026-06-02 09:30", tz=NY_TZ))[0][0])
    close[pos:pos+4] = [100.4, 101.2, 100.8, 99.9]
    close[pos+4:pos+20] = np.linspace(100.0, 104.0, 16)
    close[pos+20:] = 104.0
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) + 0.15
    lo = np.minimum(op, close) - 0.15
    bars = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": 1000.0}, index=idx)
    trade = pd.DataFrame([{
        "ny_date": "2026-06-02", "path_event_id": "p1", "event_id": "p1", "range_model": DEFAULT_RANGE_MODEL,
        "entry_archetype": DEFAULT_ENTRY_ARCHETYPE, "execution_tf": "1m", "execution_tf_minutes": 1,
        "trade_side": "LONG", "entry_order_type": "market_next_open", "entry_available_time": idx[pos],
        "entry_price": float(op[pos]), "entry_price_replay": float(op[pos]), "fill_time": idx[pos], "filled": True,
        "stop_price": 99.0, "target_price": 104.0,
    }])
    return bars, trade, ["2026-06-02"]


def run_self_test() -> int:
    bars, base, sessions = _synthetic_bars()
    managed = replay_position_scenarios(bars, base, round_trip_cost=0.0011, cost_multipliers=(1.0,), delays=(0,), scenarios=SCENARIOS)
    assert len(managed) == len(SCENARIOS)
    assert managed["managed"].all()
    assert set(managed["management_scenario"]) == set(SCENARIOS)
    summary = summarize_management(managed, valid_sessions=sessions, risk_fraction=0.01, max_notional_multiple=1.0, initial_capital=10_000.0)
    assert (summary["trades_per_session"] == 1.0).all()
    print("R20 self-test PASS")
    return 0


def run_research(args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("[stage 1/7] load R15/R16 causal broad structure universe", flush=True)
    r16_cache, r16_manifest = _load_r16_manifest(args)
    r15_cache = _resolve_r15_cache(args, r16_cache, r16_manifest)
    signals, paths = _load_broad_signals(args, r15_cache)
    valid_sessions = _valid_sessions_from_cache(r16_cache, r15_cache, r16_manifest)

    print("[stage 2/7] load causal Alpaca 1m bars + materialize next-open entries", flush=True)
    bars = _load_1m(args)
    base = _materialize_broad_market_entries(signals, bars)
    funnel = _funnel(paths, signals, base, valid_sessions)
    fill_count = int(base["filled"].fillna(False).astype(bool).sum()) if not base.empty else 0
    frequency = fill_count / max(1, len(valid_sessions))
    close_count = int(base.loc[base["filled"].fillna(False).astype(bool) & base["close_confirmed_mss"].fillna(False).astype(bool)].shape[0]) if not base.empty else 0
    print(f"[funnel] valid_sessions={len(valid_sessions):,} paths={len(paths):,} broad_signals={len(signals):,} fills={fill_count:,} trades/session={frequency:.3f} close_confirmed_subset={close_count:,}", flush=True)
    if frequency < float(args.minimum_trades_per_session):
        print(f"[frequency-warning] broad universe is still below hard target: {frequency:.3f} < {float(args.minimum_trades_per_session):.3f}; completing diagnostics instead of aborting before management", flush=True)
    costs = _csv_numbers(args.cost_multipliers, cast=float)
    delays = _csv_numbers(args.entry_delay_minutes, cast=int)
    total = fill_count * len(SCENARIOS) * len(delays)
    progress = ProgressReporter(label="[R20 management]", total=max(1, total), every=max(1, total // 100), enabled=not args.no_progress)
    print(f"[stage 3/7] replay management trades={fill_count:,} policies={len(SCENARIOS)} delays={delays}", flush=True)
    managed = replay_position_scenarios(
        bars,
        base,
        round_trip_cost=float(args.round_trip_cost),
        cost_multipliers=costs,
        delays=delays,
        scenarios=SCENARIOS,
        config=BroadPositionManagementConfig(),
        progress=progress,
    )
    progress.close()

    print("[stage 4/7] account + period scorecards", flush=True)
    summary = summarize_management(
        managed,
        valid_sessions=valid_sessions,
        risk_fraction=float(args.risk_fraction),
        max_notional_multiple=float(args.max_notional_multiple),
        initial_capital=float(args.initial_capital),
    )
    period = summarize_periods(
        managed,
        valid_sessions=valid_sessions,
        risk_fraction=float(args.risk_fraction),
        max_notional_multiple=float(args.max_notional_multiple),
        initial_capital=float(args.initial_capital),
    )
    selection = select_discovery_policy(period, minimum_trades_per_session=float(args.minimum_trades_per_session))
    decision = _decision(selection, period, summary, args)

    print("[stage 5/7] write strategy/backtest outputs", flush=True)
    (out / "00_strategy_rules.md").write_text(_design_text(args), encoding="utf-8")
    _write(funnel, out / "01_frequency_funnel.csv")
    _write(base, out / "02_frozen_broad_entry_universe.csv")
    _write(managed, out / "03_position_management_lifecycle.csv")
    _write(summary, out / "04_management_account_scorecard.csv")
    _write(period, out / "05_management_period_stability.csv")
    (out / "06_frozen_policy_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "07_strategy_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    selected_policy = str(selection.get("selected_policy", ""))
    equity, monthly, yearly, selected_stress = _selected_policy_detail(managed, selected_policy, args)
    _write(equity, out / "08_selected_policy_equity_and_trades.csv")
    _write(monthly, out / "09_selected_policy_monthly.csv")
    _write(yearly, out / "10_selected_policy_yearly.csv")
    _write(selected_stress, out / "11_selected_policy_stress_lifecycle.csv")
    manifest = {
        "research_id": "R20",
        "purpose": "broad-entry executable position-management backtest",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "valid_sessions": len(valid_sessions),
        "broad_entry_trades": fill_count,
        "base_trades_per_session": frequency,
        "hard_minimum_trades_per_session": float(args.minimum_trades_per_session),
        "range_model": args.range_model,
        "entry_archetype": DEFAULT_ENTRY_ARCHETYPE,
        "visible_swing_percentile": float(args.visible_swing_percentile),
        "broad_entry_definition": "earliest first-visible 1m/2m wick-or-close structure break per path -> next available 1m open",
        "r15_cache": str(r15_cache),
        "management_scenarios": list(SCENARIOS),
        "cost_multipliers": list(costs),
        "entry_delay_minutes": list(delays),
        "round_trip_cost": float(args.round_trip_cost),
        "risk_fraction": float(args.risk_fraction),
        "max_notional_multiple": float(args.max_notional_multiple),
        "r16_cache": str(args.r16_cache_dir),
        "r16_manifest": r16_manifest,
        "selection": selection,
        "decision": decision,
    }
    (out / "12_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("[stage 6/7] concise console result", flush=True)
    cols = [c for c in ["management_scenario", "cost_multiple", "entry_delay_minutes", "trades", "trades_per_session", "win_rate", "profit_factor", "mean_net_return", "max_consecutive_losses", "max_drawdown", "cagr"] if c in summary.columns]
    print(summary.loc[(summary["cost_multiple"] == 1.0) & (summary["entry_delay_minutes"] == 0), cols].to_string(index=False), flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str), flush=True)

    print("[stage 7/7] finalize review pack", flush=True)
    if not args.skip_review_pack:
        try:
            finalize_research_report(out)
        except Exception as exc:
            print(f"[review-pack] warning: {exc}", flush=True)
    return {"out_dir": out}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    run_research(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
