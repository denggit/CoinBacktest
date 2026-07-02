#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Lead Candidate Replay Audit.

Purpose
-------
Focused candidate audit for the strong ETH Lead Flywheel candidates discovered
in `eth_lead_flywheel_focused_research.py`.

This is NOT another strategy search. It replays a fixed shortlist of candidate
specs and writes audit artefacts for:

1) no-lookahead / context timestamp alignment;
2) fast event-driven vs slow bar-by-bar exactness;
3) candidate trades with entry/exit context;
4) OHLC intrabar ambiguity and same-bar exit risk;
5) stronger fee/slippage/delay/risk stress matrix;
6) parameter-neighbourhood checks for swing_window stability.

Assumptions are inherited from the focused research script:
- closed-bar signal, next-bar open entry;
- add / time-stop / fail-fast decisions are closed-bar decisions executed next
  bar open;
- same-bar TP/SL collision is conservative: SL first;
- default fee per side 0.00055, round trip 0.11%; books are not used.

Typical command from CoinBacktest root
--------------------------------------
python research/eth_lead_candidate_replay_audit.py --local-only --no-build-missing-cache --include-range-context --include-footprint-context --out-dir data/reports/research/eth_lead_candidate_replay_audit

If the focused research file has a different path:
python research/eth_lead_candidate_replay_audit.py --base-script research/eth_lead_flywheel_focused_research.py --out-dir data/reports/research/eth_lead_candidate_replay_audit
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_NAME = "eth_lead_candidate_replay_audit"

DEFAULT_CANDIDATE_IDS = [
    "F00222",
    "F00240",
    "F00231",
    "F00230",
    "F00237",
    "F00088",
    "F00829",
    "F00832",
]

DEFAULT_NEIGHBOR_SWINGS = [60, 120, 240, 480]


# =============================================================================
# Base module loading
# =============================================================================


def _load_base_module(base_script: str | None):
    candidates: list[Path] = []
    if base_script:
        candidates.append((PROJECT_ROOT / base_script).resolve())
        candidates.append(Path(base_script).resolve())
    candidates.append((PROJECT_ROOT / "research" / "eth_lead_flywheel_focused_research.py").resolve())
    candidates.append((CURRENT_FILE.parent / "eth_lead_flywheel_focused_research.py").resolve())

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        checked = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Cannot find eth_lead_flywheel_focused_research.py. "
            "Put this audit file next to it or pass --base-script. Checked:\n" + checked
        )
    spec = importlib.util.spec_from_file_location("eth_lead_flywheel_focused_research_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base research script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


# =============================================================================
# Helpers
# =============================================================================


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _candidate_prefix(spec_id: str) -> str:
    return str(spec_id).split("_", 1)[0]


def _parse_mixed_dt(values: Any) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        s = pd.Series(values).astype(str)
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.isna().any():
            stripped = s.str.replace(r"\.\d+$", "", regex=True)
            parsed2 = pd.to_datetime(stripped, errors="coerce")
            parsed = parsed.fillna(parsed2)
        return parsed


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if not math.isfinite(a) or not math.isfinite(b) or abs(b) <= 1e-12:
            return default
        return a / b
    except Exception:
        return default


def _to_records(items: list[Any]) -> list[dict[str, Any]]:
    return [asdict(x) for x in items]


def _spec_identity(spec: Any) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "system": getattr(spec, "system", ""),
        "entry_model": getattr(spec, "entry_model", ""),
        "signal_frame": getattr(spec, "signal_frame", ""),
        "regime": getattr(spec, "regime", ""),
        "structure": getattr(spec, "structure", ""),
        "swing_window": getattr(spec, "swing_window", ""),
        "confirmation": getattr(spec, "confirmation", ""),
        "layer": getattr(spec, "layer", ""),
    }


def _find_candidate_specs(base: Any, cfg: Any, wanted_ids: list[str], max_specs: int | None) -> list[Any]:
    frames = base._signal_frames_from_cfg(cfg)
    specs = base.generate_specs(
        "core",
        max_specs,
        signal_frames=frames,
        include_sanity_in_core=False,
    )
    wanted = set(wanted_ids)
    out = [sp for sp in specs if _candidate_prefix(sp.spec_id) in wanted or sp.spec_id in wanted]
    found = {_candidate_prefix(sp.spec_id) for sp in out}
    missing = [x for x in wanted_ids if _candidate_prefix(x) not in found and x not in {sp.spec_id for sp in out}]
    if missing:
        raise RuntimeError(
            "Missing candidate spec ids in generated focused core spec list: " + ",".join(missing) +
            ". Increase --spec-generation-max or check that you are using the same focused research script."
        )
    # Preserve input order.
    order = {x: i for i, x in enumerate(wanted_ids)}
    out.sort(key=lambda sp: order.get(_candidate_prefix(sp.spec_id), 9999))
    return out


def _make_candidate_neighbours(base: Any, specs: list[Any], swings: list[int]) -> list[Any]:
    neighbours: list[Any] = []
    seen: set[tuple[str, int]] = set()
    for sp in specs:
        for sw in swings:
            key = (_candidate_prefix(sp.spec_id), int(sw))
            if key in seen:
                continue
            seen.add(key)
            nsp = replace(sp, swing_window=int(sw), spec_id=f"{_candidate_prefix(sp.spec_id)}_NEIGHBOR_sw{int(sw)}_{sp.entry_model}_{sp.regime}_{sp.structure}")
            neighbours.append(nsp)
    return neighbours


# =============================================================================
# Context timestamp tracing
# =============================================================================


def _context_prefix(tf: str) -> str:
    return "tf" + str(tf).replace("m", "m").replace("H", "H")


def _load_context_timestamp_trace(cfg: Any, base_index: pd.DatetimeIndex, timeframes: list[str]) -> pd.DataFrame:
    """Trace which higher-timeframe timestamp was available at each primary row.

    This is independent from the base feature code. It is only for audit.  The
    key check is simple: used context timestamp must be <= signal_time.  Anything
    in the future is a hard lookahead violation.
    """
    from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: WPS433

    base_index = pd.DatetimeIndex(pd.to_datetime(base_index).tz_localize(None))
    out = pd.DataFrame(index=base_index)
    for tf in timeframes:
        if not tf or (getattr(cfg, "primary_frame", "time") == "time" and tf == getattr(cfg, "timeframe", "1m")):
            continue
        prefix = _context_prefix(tf)
        try:
            loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe=tf, data_dir=cfg.data_dir)
            if cfg.local_only or not cfg.build_missing_cache:
                ctx = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
            else:
                ctx = loader.fetch_data_by_date_range(
                    cfg.warmup_start_date,
                    cfg.end_date,
                    chunksize=cfg.chunksize,
                    force_rebuild=False,
                    cvd_mode="range",
                )
        except Exception as exc:  # audit must keep running even if optional trace fails
            out[f"trace_{prefix}_error"] = str(exc)
            continue
        if ctx is None or ctx.empty:
            out[f"trace_{prefix}_ts"] = pd.NaT
            continue
        if not isinstance(ctx.index, pd.DatetimeIndex):
            if "timestamp" in ctx.columns:
                ctx.index = pd.to_datetime(ctx["timestamp"])
            elif "end_ts" in ctx.columns:
                ctx.index = pd.to_datetime(ctx["end_ts"])
            else:
                out[f"trace_{prefix}_ts"] = pd.NaT
                continue
        ctx.index = pd.to_datetime(ctx.index).tz_localize(None)
        ctx = ctx[~ctx.index.duplicated(keep="last")].sort_index()
        ts_series = pd.Series(ctx.index, index=ctx.index)
        aligned = ts_series.reindex(base_index, method="ffill")
        out[f"trace_{prefix}_ts"] = aligned.to_numpy()
    return out


# =============================================================================
# Candidate audit metrics
# =============================================================================


def _index_position_map(index: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    # For time-primary candidates index should be unique. If it is not, keep the
    # first occurrence for audit lookup and expose index_is_unique in diagnostics.
    out: dict[pd.Timestamp, int] = {}
    for i, ts in enumerate(pd.DatetimeIndex(index)):
        out.setdefault(pd.Timestamp(ts), i)
    return out


def _lookup_pos(ts: Any, pos_map: dict[pd.Timestamp, int]) -> int | None:
    parsed = _parse_mixed_dt(pd.Series([ts])).iloc[0]
    if pd.isna(parsed):
        return None
    return pos_map.get(pd.Timestamp(parsed))


def _candidate_trade_audit_rows(
    base: Any,
    df: pd.DataFrame,
    cfg: Any,
    spec: Any,
    trades: list[Any],
    arrays: Any,
    context_trace: pd.DataFrame,
) -> list[dict[str, Any]]:
    idx = pd.DatetimeIndex(arrays.index)
    pos_map = _index_position_map(idx)
    bt = base.SingleSpecBacktester(df, cfg, spec, arrays=arrays)
    rows: list[dict[str, Any]] = []

    for trade_no, tr in enumerate(trades, start=1):
        signal_i = _lookup_pos(tr.signal_time, pos_map)
        entry_i = _lookup_pos(tr.entry_time, pos_map)
        exit_i = _lookup_pos(tr.exit_time, pos_map)
        signal_ts = pd.NaT if signal_i is None else idx[signal_i]
        entry_ts = pd.NaT if entry_i is None else idx[entry_i]
        exit_ts = pd.NaT if exit_i is None else idx[exit_i]
        side_int = 1 if str(tr.side).upper() == "LONG" else -1

        expected_entry_i = None if signal_i is None else signal_i + 1 + int(cfg.entry_delay_bars)
        expected_entry_open = np.nan
        entry_open_match = False
        if expected_entry_i is not None and 0 <= expected_entry_i < len(df):
            expected_entry_open = float(arrays.open[expected_entry_i])
            entry_open_match = abs(float(tr.entry_price) - expected_entry_open) <= max(1e-7, abs(expected_entry_open) * 1e-9)

        stop_dist = np.nan
        initial_stop = np.nan
        full_tp = np.nan
        partial_tp = np.nan
        entry_bar_stop_hit = False
        entry_bar_tp_hit = False
        entry_bar_partial_hit = False
        entry_bar_both_stop_tp = False
        if signal_i is not None and entry_i is not None:
            try:
                stop_dist = float(bt._stop_distance(signal_i, float(tr.entry_price)))
                initial_stop = float(tr.entry_price) - side_int * stop_dist
                full_tp = float(tr.entry_price) + side_int * stop_dist * float(spec.tp_r)
                partial_tp = float(tr.entry_price) + side_int * stop_dist * float(spec.partial_tp_r)
                h = float(arrays.high[entry_i])
                l = float(arrays.low[entry_i])
                entry_bar_stop_hit = bool(l <= initial_stop if side_int == 1 else h >= initial_stop)
                entry_bar_tp_hit = bool(h >= full_tp if side_int == 1 else l <= full_tp)
                entry_bar_partial_hit = bool(h >= partial_tp if side_int == 1 else l <= partial_tp)
                entry_bar_both_stop_tp = bool(entry_bar_stop_hit and entry_bar_tp_hit)
            except Exception:
                pass

        used_tf5m_ts = pd.NaT
        tf5m_not_future = True
        tf5m_age_min = np.nan
        if signal_i is not None and "trace_tf5m_ts" in context_trace.columns:
            used_tf5m_ts = pd.to_datetime(context_trace.iloc[signal_i]["trace_tf5m_ts"], errors="coerce")
            if pd.notna(used_tf5m_ts) and pd.notna(signal_ts):
                tf5m_not_future = bool(pd.Timestamp(used_tf5m_ts) <= pd.Timestamp(signal_ts))
                tf5m_age_min = (pd.Timestamp(signal_ts) - pd.Timestamp(used_tf5m_ts)).total_seconds() / 60.0

        used_tf15m_ts = pd.NaT
        tf15m_not_future = True
        tf15m_age_min = np.nan
        if signal_i is not None and "trace_tf15m_ts" in context_trace.columns:
            used_tf15m_ts = pd.to_datetime(context_trace.iloc[signal_i]["trace_tf15m_ts"], errors="coerce")
            if pd.notna(used_tf15m_ts) and pd.notna(signal_ts):
                tf15m_not_future = bool(pd.Timestamp(used_tf15m_ts) <= pd.Timestamp(signal_ts))
                tf15m_age_min = (pd.Timestamp(signal_ts) - pd.Timestamp(used_tf15m_ts)).total_seconds() / 60.0

        lookahead_flag = False
        if signal_i is None or entry_i is None:
            lookahead_flag = True
        elif entry_i <= signal_i:
            lookahead_flag = True
        elif expected_entry_i is not None and entry_i != expected_entry_i:
            lookahead_flag = True
        elif not entry_open_match:
            lookahead_flag = True
        elif not bool(tf5m_not_future and tf15m_not_future):
            lookahead_flag = True

        row = {
            **_spec_identity(spec),
            "trade_no": trade_no,
            "side": tr.side,
            "signal_time": tr.signal_time,
            "entry_time": tr.entry_time,
            "exit_time": tr.exit_time,
            "signal_i": signal_i if signal_i is not None else -1,
            "entry_i": entry_i if entry_i is not None else -1,
            "exit_i": exit_i if exit_i is not None else -1,
            "expected_entry_i": expected_entry_i if expected_entry_i is not None else -1,
            "signal_before_entry": bool(entry_i is not None and signal_i is not None and entry_i > signal_i),
            "entry_is_expected_next_open": bool(entry_i == expected_entry_i if expected_entry_i is not None else False),
            "entry_open_match": bool(entry_open_match),
            "expected_entry_open": expected_entry_open,
            "entry_price": float(tr.entry_price),
            "exit_price": float(tr.exit_price),
            "exit_reason": tr.exit_reason,
            "hold_bars": int(tr.hold_bars),
            "adds": int(tr.adds),
            "partial_taken": bool(tr.partial_taken),
            "mfe_r": float(tr.mfe_r),
            "mae_r": float(tr.mae_r),
            "net_pnl_frac": float(tr.net_pnl_frac),
            "return_pct": float(tr.return_pct),
            "cost_frac": float(tr.cost_frac),
            "stop_dist": stop_dist,
            "initial_stop": initial_stop,
            "initial_full_tp": full_tp,
            "initial_partial_tp": partial_tp,
            "entry_bar_stop_hit": bool(entry_bar_stop_hit),
            "entry_bar_full_tp_hit": bool(entry_bar_tp_hit),
            "entry_bar_partial_tp_hit": bool(entry_bar_partial_hit),
            "entry_bar_both_stop_full_tp": bool(entry_bar_both_stop_tp),
            "same_bar_exit": bool(int(tr.hold_bars) <= 1),
            "used_tf5m_ts": str(used_tf5m_ts) if pd.notna(used_tf5m_ts) else "",
            "tf5m_context_not_future": bool(tf5m_not_future),
            "tf5m_context_age_min": tf5m_age_min,
            "used_tf15m_ts": str(used_tf15m_ts) if pd.notna(used_tf15m_ts) else "",
            "tf15m_context_not_future": bool(tf15m_not_future),
            "tf15m_context_age_min": tf15m_age_min,
            "lookahead_flag": bool(lookahead_flag),
        }

        # Add a small feature snapshot at the signal bar.  Missing columns remain blank.
        if signal_i is not None:
            for col in [
                "open", "high", "low", "close", "volume", "session_vwap", "atr_60",
                "tf5m_close", "tf5m_ema_20", "tf5m_ema_60", "tf5m_trend_up", "tf5m_trend_down",
                "tf15m_close", "tf15m_trend_up", "tf15m_trend_down",
                "rf_bar_count", "rf_direction_sum", "rf_imbalance", "range_dir_sum_3",
                "fp_absorption_hint", "fp_delta_sum",
            ]:
                row[f"signal_{col}"] = df.iloc[signal_i][col] if col in df.columns else ""
        rows.append(row)
    return rows


def _event_window_rows(
    df: pd.DataFrame,
    arrays: Any,
    spec: Any,
    trades: list[Any],
    *,
    trades_per_spec: int,
    pre_bars: int,
    post_bars: int,
) -> list[dict[str, Any]]:
    idx = pd.DatetimeIndex(arrays.index)
    pos_map = _index_position_map(idx)
    rows: list[dict[str, Any]] = []
    for trade_no, tr in enumerate(trades[: max(0, trades_per_spec)], start=1):
        entry_i = _lookup_pos(tr.entry_time, pos_map)
        signal_i = _lookup_pos(tr.signal_time, pos_map)
        if entry_i is None:
            continue
        start = max(0, entry_i - pre_bars)
        end = min(len(df) - 1, entry_i + post_bars)
        for j in range(start, end + 1):
            r = {
                **_spec_identity(spec),
                "trade_no": trade_no,
                "rel_bar": j - entry_i,
                "timestamp": str(idx[j]),
                "is_signal_bar": bool(signal_i == j),
                "is_entry_bar": bool(entry_i == j),
                "side": tr.side,
                "entry_price": float(tr.entry_price),
                "exit_time": tr.exit_time,
                "exit_reason": tr.exit_reason,
            }
            for col in [
                "open", "high", "low", "close", "volume", "buy_notional", "sell_notional", "delta_notional",
                "session_vwap", "ema_20", "ema_60", "atr_60", "bar_range_pct",
                "tf5m_open", "tf5m_high", "tf5m_low", "tf5m_close", "tf5m_ema_20", "tf5m_ema_60", "tf5m_trend_up", "tf5m_trend_down",
                "tf15m_close", "tf15m_trend_up", "tf15m_trend_down",
                "rf_bar_count", "rf_direction_sum", "rf_imbalance", "range_dir_sum_3", "fp_absorption_hint",
            ]:
                r[col] = df.iloc[j][col] if col in df.columns else ""
            rows.append(r)
    return rows


def _audit_summary_from_enriched(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for spec_id, g in enriched.groupby("spec_id", sort=False):
        rows.append({
            "spec_id": spec_id,
            "trades": int(len(g)),
            "lookahead_flags": int(g["lookahead_flag"].sum()) if "lookahead_flag" in g else 0,
            "entry_not_next_open": int((~g["entry_is_expected_next_open"].astype(bool)).sum()) if "entry_is_expected_next_open" in g else 0,
            "entry_price_mismatch": int((~g["entry_open_match"].astype(bool)).sum()) if "entry_open_match" in g else 0,
            "tf5m_future_flags": int((~g["tf5m_context_not_future"].astype(bool)).sum()) if "tf5m_context_not_future" in g else 0,
            "tf15m_future_flags": int((~g["tf15m_context_not_future"].astype(bool)).sum()) if "tf15m_context_not_future" in g else 0,
            "same_bar_exit_rate": float(g["same_bar_exit"].mean()) if "same_bar_exit" in g else 0.0,
            "entry_bar_both_stop_tp_rate": float(g["entry_bar_both_stop_full_tp"].mean()) if "entry_bar_both_stop_full_tp" in g else 0.0,
            "entry_bar_stop_hit_rate": float(g["entry_bar_stop_hit"].mean()) if "entry_bar_stop_hit" in g else 0.0,
            "entry_bar_full_tp_hit_rate": float(g["entry_bar_full_tp_hit"].mean()) if "entry_bar_full_tp_hit" in g else 0.0,
            "avg_tf5m_age_min": float(pd.to_numeric(g.get("tf5m_context_age_min", pd.Series(dtype=float)), errors="coerce").mean()),
            "median_hold_bars": float(pd.to_numeric(g["hold_bars"], errors="coerce").median()) if "hold_bars" in g else 0.0,
            "avg_mae_r": float(pd.to_numeric(g["mae_r"], errors="coerce").mean()) if "mae_r" in g else 0.0,
            "p95_adverse_r": float(pd.to_numeric(g["mae_r"], errors="coerce").quantile(0.05)) if "mae_r" in g else 0.0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Stress and neighbourhood checks
# =============================================================================


def _stress_scenarios(base_cfg: Any, base: Any) -> list[tuple[str, Any]]:
    scenarios = [
        ("base", base_cfg),
        ("fee_1_5x", replace(base_cfg, fee_rate_per_side=float(base_cfg.fee_rate_per_side) * 1.5)),
        ("fee_2x", replace(base_cfg, fee_rate_per_side=float(base_cfg.fee_rate_per_side) * 2.0)),
        ("slippage_3bp", replace(base_cfg, slippage_pct=0.00030)),
        ("slippage_5bp", replace(base_cfg, slippage_pct=0.00050)),
        ("slippage_10bp", replace(base_cfg, slippage_pct=0.00100)),
        ("delay_1bar", replace(base_cfg, entry_delay_bars=1)),
        ("delay_2bar", replace(base_cfg, entry_delay_bars=2)),
        ("delay_3bar", replace(base_cfg, entry_delay_bars=3)),
        ("delay_5bar", replace(base_cfg, entry_delay_bars=5)),
        ("risk_half", replace(base_cfg, risk_per_trade=float(base_cfg.risk_per_trade) * 0.5)),
        ("strict_copy_1", replace(base_cfg, fee_rate_per_side=float(base_cfg.fee_rate_per_side) * 1.5, slippage_pct=0.00050, entry_delay_bars=1)),
        ("strict_copy_3", replace(base_cfg, fee_rate_per_side=float(base_cfg.fee_rate_per_side) * 1.5, slippage_pct=0.00050, entry_delay_bars=3)),
    ]
    return scenarios


def _run_specs_summary(
    base: Any,
    df: pd.DataFrame,
    cfg: Any,
    specs: list[Any],
    arrays: Any,
    signal_cache: dict[Any, np.ndarray],
    index_cache: dict[Any, np.ndarray],
    *,
    scenario: str,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    rows: list[dict[str, Any]] = []
    trade_map: dict[str, list[Any]] = {}
    for sp in specs:
        key = base.signal_full_key(sp)
        sig = signal_cache[key]
        idxs = index_cache[key]
        bt = base.SingleSpecBacktester(df, cfg, sp, arrays=arrays)
        trades = bt.run(sig, idxs)
        trade_map[sp.spec_id] = trades
        row = base.summarize_trades(sp, trades, cfg)
        row["scenario"] = scenario
        rows.append(row)
    return rows, trade_map


def _stress_matrix(
    base: Any,
    df: pd.DataFrame,
    cfg: Any,
    specs: list[Any],
    arrays: Any,
    signal_cache: dict[Any, np.ndarray],
    index_cache: dict[Any, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, scfg in _stress_scenarios(cfg, base):
        print(f"      stress scenario: {scenario}", flush=True)
        srows, _ = _run_specs_summary(base, df, scfg, specs, arrays, signal_cache, index_cache, scenario=scenario)
        rows.extend(srows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["stress_pass_basic"] = (
            (pd.to_numeric(out.get("total_return_pct", 0), errors="coerce") > 0)
            & (pd.to_numeric(out.get("profit_factor", 0), errors="coerce") > 1.05)
            & (pd.to_numeric(out.get("max_drawdown_pct", -999), errors="coerce") > -25.0)
        )
    return out


def _neighbourhood_check(
    base: Any,
    df: pd.DataFrame,
    cfg: Any,
    target_specs: list[Any],
    arrays: Any,
    swings: list[int],
) -> pd.DataFrame:
    neigh = _make_candidate_neighbours(base, target_specs, swings)
    if not neigh:
        return pd.DataFrame()
    signal_cache, index_cache, count_cache, raw_count_cache = base.build_signal_caches(df, neigh, cfg)
    rows, _ = _run_specs_summary(base, df, cfg, neigh, arrays, signal_cache, index_cache, scenario="base_neighbor")
    out = pd.DataFrame(rows)
    if not out.empty:
        out["candidate_prefix"] = out["spec_id"].astype(str).str.extract(r"^(F\d{5})", expand=False)
    return out


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay/audit strong ETH Lead Flywheel candidates.")
    p.add_argument("--base-script", default="research/eth_lead_flywheel_focused_research.py")
    p.add_argument("--candidate-ids", default=",".join(DEFAULT_CANDIDATE_IDS))
    p.add_argument("--spec-generation-max", type=int, default=1200)
    p.add_argument("--neighbour-swings", default=",".join(str(x) for x in DEFAULT_NEIGHBOR_SWINGS))

    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--primary-frame", choices=["time", "range"], default="time")
    p.add_argument("--context-timeframes", default="5m,15m,1H")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.0020)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.00015)
    p.add_argument("--entry-delay-bars", type=int, default=0)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--build-missing-cache", dest="build_missing_cache", action="store_true", default=True)
    p.add_argument("--no-build-missing-cache", dest="build_missing_cache", action="store_false")
    p.add_argument("--include-range-context", action="store_true")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--include-footprint-context", action="store_true")
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--min-signal-gap-bars", type=int, default=10)
    p.add_argument("--max-signals-per-spec", type=int, default=0)
    p.add_argument("--window-trades-per-spec", type=int, default=40)
    p.add_argument("--window-pre-bars", type=int, default=5)
    p.add_argument("--window-post-bars", type=int, default=8)
    p.add_argument("--skip-neighbourhood", action="store_true")
    p.add_argument("--skip-stress", action="store_true")
    p.add_argument("--out-dir", default="data/reports/research/eth_lead_candidate_replay_audit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base, base_path = _load_base_module(args.base_script)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = base.RunConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        primary_frame=args.primary_frame,
        context_timeframes=args.context_timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        data_dir=args.data_dir,
        mode="core",
        max_specs=args.spec_generation_max,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_notional_mult=args.max_notional_mult,
        fee_rate_per_side=args.fee_rate_per_side,
        slippage_pct=args.slippage_pct,
        entry_delay_bars=args.entry_delay_bars,
        local_only=args.local_only,
        build_missing_cache=args.build_missing_cache,
        include_range_context=args.include_range_context,
        range_pct=args.range_pct,
        include_footprint_context=args.include_footprint_context,
        price_step=args.price_step,
        chunksize=args.chunksize,
        min_signal_gap_bars=args.min_signal_gap_bars,
        max_signals_per_spec=args.max_signals_per_spec,
        include_sanity_in_core=False,
        robustness_top_n=0,
        write_trades=True,
        verify_fast_exactness=False,
        verify_fast_exactness_specs=0,
        fail_on_empty_signals=True,
        fail_on_empty_audit=True,
        out_dir=str(out_dir),
    )

    candidate_ids = _parse_csv(args.candidate_ids)
    neighbour_swings = [int(x) for x in _parse_csv(args.neighbour_swings)]

    print(f"[1/7] Loading data with base research script: {base_path}", flush=True)
    bars = base.load_primary_bars(cfg)
    range_ctx = base.load_range_context(cfg, bars.index)
    footprint_ctx = base.load_footprint_context(cfg, bars.index)
    time_ctx = base.load_timeframe_contexts(cfg, bars.index)

    print("[2/7] Building feature frame and immutable arrays", flush=True)
    df = base.build_features(bars, range_ctx, footprint_ctx, time_ctx)
    arrays = base.build_backtest_arrays(df, cfg)
    base.write_data_diagnostics(bars, df, cfg, arrays, out_dir)
    base.write_signal_condition_breakdown(df, cfg, out_dir)

    print("[3/7] Selecting candidate specs", flush=True)
    specs = _find_candidate_specs(base, cfg, candidate_ids, args.spec_generation_max)
    pd.DataFrame([asdict(sp) for sp in specs]).to_csv(out_dir / "01_candidate_specs.csv", index=False)
    print("      candidates:", ", ".join(sp.spec_id for sp in specs), flush=True)

    print("[4/7] Building candidate signal caches", flush=True)
    signal_cache, index_cache, count_cache, raw_count_cache = base.build_signal_caches(df, specs, cfg)
    base.write_raw_signal_diagnostics(raw_count_cache, out_dir)
    pd.DataFrame([
        {**_spec_identity(sp), **count_cache.get(base.signal_full_key(sp), {})}
        for sp in specs
    ]).to_csv(out_dir / "03_candidate_signal_counts.csv", index=False)

    # Fast-vs-slow exactness for exactly these candidates, not top-by-signal-count.
    exact_cfg = replace(cfg, verify_fast_exactness=True, verify_fast_exactness_specs=len(specs), fail_on_empty_audit=True)
    base.verify_fast_exactness(
        df,
        specs,
        exact_cfg,
        arrays=arrays,
        signal_cache=signal_cache,
        index_cache=index_cache,
        count_cache=count_cache,
        out_dir=out_dir,
    )

    print("[5/7] Replaying candidate trades and writing enriched audit rows", flush=True)
    summary_rows, trade_map = _run_specs_summary(base, df, cfg, specs, arrays, signal_cache, index_cache, scenario="base")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "02_candidate_summary.csv", index=False)

    all_trade_rows: list[dict[str, Any]] = []
    all_enriched_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    ctx_frames = _parse_csv(args.context_timeframes)
    context_trace = _load_context_timestamp_trace(cfg, arrays.index, ctx_frames)

    for sp in specs:
        trades = trade_map.get(sp.spec_id, [])
        if trades:
            tdf = pd.DataFrame(_to_records(trades))
            all_trade_rows.extend(tdf.to_dict("records"))
            y = base.yearly_stats(tdf, cfg)
            if not y.empty:
                yearly_rows.extend(y.to_dict("records"))
        all_enriched_rows.extend(_candidate_trade_audit_rows(base, df, cfg, sp, trades, arrays, context_trace))
        all_window_rows.extend(_event_window_rows(
            df,
            arrays,
            sp,
            trades,
            trades_per_spec=int(args.window_trades_per_spec),
            pre_bars=int(args.window_pre_bars),
            post_bars=int(args.window_post_bars),
        ))

    trades_df = pd.DataFrame(all_trade_rows)
    trades_df.to_csv(out_dir / "07_candidate_trades.csv", index=False)
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(out_dir / "04_candidate_yearly.csv", index=False)
    enriched = pd.DataFrame(all_enriched_rows)
    enriched.to_csv(out_dir / "11_candidate_trade_audit.csv", index=False)
    audit_summary = _audit_summary_from_enriched(enriched)
    audit_summary.to_csv(out_dir / "12_candidate_audit_summary.csv", index=False)
    windows = pd.DataFrame(all_window_rows)
    windows.to_csv(out_dir / "13_candidate_event_windows.csv", index=False)

    stress = pd.DataFrame()
    if not args.skip_stress:
        print("[6/7] Running stronger cost/delay/risk stress matrix", flush=True)
        stress = _stress_matrix(base, df, cfg, specs, arrays, signal_cache, index_cache)
        stress.to_csv(out_dir / "05_candidate_stress_matrix.csv", index=False)

    if not args.skip_neighbourhood:
        print("[7/7] Running swing_window neighbourhood checks", flush=True)
        neighbours = _neighbourhood_check(base, df, cfg, specs, arrays, neighbour_swings)
        neighbours.to_csv(out_dir / "14_parameter_neighbourhood.csv", index=False)
    else:
        print("[7/7] Skipping neighbourhood checks", flush=True)

    hard_flags = []
    if not audit_summary.empty:
        for _, r in audit_summary.iterrows():
            flags = []
            if int(r.get("lookahead_flags", 0)) > 0:
                flags.append("LOOKAHEAD_FLAG")
            if int(r.get("entry_not_next_open", 0)) > 0:
                flags.append("ENTRY_NOT_NEXT_OPEN")
            if int(r.get("entry_price_mismatch", 0)) > 0:
                flags.append("ENTRY_PRICE_MISMATCH")
            if int(r.get("tf5m_future_flags", 0)) > 0 or int(r.get("tf15m_future_flags", 0)) > 0:
                flags.append("CONTEXT_FUTURE")
            if float(r.get("entry_bar_both_stop_tp_rate", 0.0)) > 0.01:
                flags.append("HIGH_INTRABAR_AMBIGUITY")
            hard_flags.append({"spec_id": r["spec_id"], "audit_flags": ";".join(flags), **r.to_dict()})
    pd.DataFrame(hard_flags).to_csv(out_dir / "15_audit_flags.csv", index=False)

    config_json = {
        "script": SCRIPT_NAME,
        "base_script": str(base_path),
        "candidate_ids": candidate_ids,
        "neighbour_swings": neighbour_swings,
        "run_config": asdict(cfg),
        "default_fee_note": "fee_rate_per_side=0.00055 means round-trip fee about 0.11% before slippage.",
        "purpose": "Candidate replay audit: future-function, context alignment, intrabar ambiguity, stress and neighbourhood checks.",
    }
    (out_dir / "00_audit_config.json").write_text(json.dumps(config_json, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    readme = f"""# ETH Lead Candidate Replay Audit

Generated by `{SCRIPT_NAME}`.

## What this run checks
- fixed shortlist candidate replay, not a new strategy search;
- fast-vs-slow exactness on the candidate list itself;
- signal time < entry time and entry price equals expected next-bar open;
- context timestamp tracing for higher timeframes, especially tf5m and tf15m;
- OHLC intrabar ambiguity: same-bar exits and entry-bar stop/TP collision risk;
- stronger fee/slippage/delay/risk stress matrix;
- swing-window neighbourhood robustness.

## Output files
- `00_audit_config.json`: config and candidate IDs
- `00_data_diagnostics.csv`: data/window diagnostics from base script
- `01_candidate_specs.csv`: selected candidate specs
- `02_candidate_summary.csv`: base summary for candidates
- `03_candidate_signal_counts.csv`: signal counts after regime filters
- `04_candidate_yearly.csv`: yearly stats from candidate trades
- `05_candidate_stress_matrix.csv`: stronger cost/delay/risk scenarios
- `07_candidate_trades.csv`: full candidate trade list
- `08_fast_exactness_check.csv`: fast vs slow exactness audit
- `09_raw_signal_diagnostics.csv`: raw signal counts before regime filter
- `10_signal_condition_breakdown.csv`: independent feature/signal sanity counts
- `11_candidate_trade_audit.csv`: enriched per-trade audit rows
- `12_candidate_audit_summary.csv`: per-candidate audit aggregates
- `13_candidate_event_windows.csv`: small OHLC/feature windows around sampled entries
- `14_parameter_neighbourhood.csv`: swing-window neighbourhood check
- `15_audit_flags.csv`: hard audit flags by candidate

## Hard fail interpretation
If `15_audit_flags.csv` contains `LOOKAHEAD_FLAG`, `ENTRY_NOT_NEXT_OPEN`,
`ENTRY_PRICE_MISMATCH`, or `CONTEXT_FUTURE`, treat that candidate as invalid
until the underlying implementation is fixed.  A high intrabar ambiguity rate
means the candidate needs trade-level replay before trust.

## Safety assumptions
- closed-bar signal;
- next-bar open entry;
- add/exit decisions scheduled to next-bar open;
- SL-first if TP and SL both happen inside one OHLC bar;
- books are not used.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\nDONE", flush=True)
    print(f"Report dir: {out_dir.resolve()}", flush=True)
    if not summary.empty:
        cols = [c for c in ["spec_id", "trades", "total_return_pct", "win_rate", "profit_factor", "max_drawdown_pct", "green_touch_rate", "max_days_without_trade", "score"] if c in summary.columns]
        print(summary[cols].to_string(index=False, max_rows=20), flush=True)
    if not audit_summary.empty:
        print("\nAudit flags summary:", flush=True)
        print(pd.DataFrame(hard_flags)[["spec_id", "audit_flags", "same_bar_exit_rate", "entry_bar_both_stop_tp_rate"]].to_string(index=False, max_rows=20), flush=True)


if __name__ == "__main__":
    main()
