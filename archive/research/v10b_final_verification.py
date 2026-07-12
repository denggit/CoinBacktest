#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10B Final Verification
=======================

Research/verification-only tool for ETH LF Portfolio V10B all-engine swing
structural stop candidate.

Goals:
  1. Compare V10A baseline vs V10B using percentage metrics, not absolute PnL.
  2. Produce a safe trade diff without NaN/cartesian matching bugs.
  3. Audit structural-stop footprint, code-level timing assumptions, top-winner
     dependency, yearly stability, and fee/slippage stress.
  4. Gate whether V10B is ready for AetherEdge parity/dry-run work.

This script does not modify official strategy logic and does not touch AetherEdge.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

V10A_NAME = "eth_lf_portfolio_v10a_momentum_micro_short_speed_filter"
V10B_NAME = "eth_lf_portfolio_v10b_all_swing_structural_stop"
OUT_NAME = "v10b_final_verification"

V10A_SCRIPT = "backtest/lf/eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest.py"
V10B_SCRIPT = "backtest/lf/eth_lf_portfolio_v10b_all_swing_structural_stop_backtest.py"

DEFAULT_V10A_DIR = f"data/reports/lf/{V10A_NAME}"
DEFAULT_V10B_DIR = f"data/reports/lf/{V10B_NAME}/turbo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final V10A vs V10B verification before AetherEdge migration.")
    p.add_argument("--out-dir", default=f"data/reports/research/{OUT_NAME}")
    p.add_argument("--v10a-dir", default=DEFAULT_V10A_DIR, help="Existing V10A report dir.")
    p.add_argument("--v10b-dir", default=DEFAULT_V10B_DIR, help="Existing V10B report dir.")
    p.add_argument("--from-reports", action="store_true", help="Read existing reports and build verification outputs.")
    p.add_argument("--rerun-formal", action="store_true", help="Rerun V10A and V10B baseline reports first.")
    p.add_argument("--stress", action="store_true", help="Run fee/slippage stress for V10A/V10B.")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--initial-capital", default="1000")
    p.add_argument("--range-pct", default="0.002")
    p.add_argument("--price-step", default="1.0")
    p.add_argument("--fee-grid", default="0.00055,0.00075,0.001", help="Comma-separated fee_rate values.")
    p.add_argument("--slippage-grid", default="0.0002,0.0005,0.001", help="Comma-separated slippage values.")
    p.add_argument("--min-return-ratio", type=float, default=1.0)
    p.add_argument("--max-dd-delta", type=float, default=0.0, help="Require V10B max drawdown <= V10A + this value.")
    p.add_argument("--min-pf-delta", type=float, default=0.0, help="Require V10B PF >= V10A + this value.")
    p.add_argument("--max-top3-dependency-delta", type=float, default=5.0)
    p.add_argument("--write-combined-trades", action="store_true")
    args = p.parse_args()
    if not args.from_reports and not args.rerun_formal and not args.stress:
        args.from_reports = True
    return args


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(PROJECT_ROOT) / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_file(report_dir: Path, suffix: str) -> Path:
    direct = report_dir / suffix
    if direct.exists():
        return direct
    matches = sorted(report_dir.glob(f"*{suffix}"))
    if matches:
        return matches[0]
    matches = sorted(report_dir.rglob(f"*{suffix}"))
    return matches[0] if matches else direct


def load_report(report_dir: Path, strategy_name: str) -> dict[str, Any]:
    summary = read_json(find_file(report_dir, f"{strategy_name}_summary.json"))
    trades = read_csv(find_file(report_dir, f"{strategy_name}_trades.csv"))
    equity = read_csv(find_file(report_dir, f"{strategy_name}_equity.csv"))
    audit = read_csv(find_file(report_dir, f"{strategy_name}_signal_audit.csv"))
    return {"summary": summary, "trades": trades, "equity": equity, "audit": audit, "dir": report_dir}


def nfloat(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def metric(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    return nfloat(summary.get(key), default)


def build_summary_compare(v10a: dict[str, Any], v10b: dict[str, Any]) -> pd.DataFrame:
    a = v10a["summary"]
    b = v10b["summary"]
    rows = []
    keys = [
        "total_trades", "long_trades", "short_trades", "total_return_pct", "win_rate",
        "profit_factor", "expectancy_pct", "max_drawdown_pct", "avg_mfe_r", "avg_mae_r",
        "avg_units", "avg_risk_mult", "avg_holding_hours", "mfe_ge_1r_ended_loss",
        "mfe_ge_2r_ended_loss", "total_fees", "structural_stop_trade_count",
        "structural_stop_total_updates", "structural_stop_exit_count",
    ]
    for key in keys:
        av = metric(a, key, float("nan"))
        bv = metric(b, key, float("nan"))
        rows.append({
            "metric": key,
            "v10a": av,
            "v10b": bv,
            "delta": bv - av if math.isfinite(av) and math.isfinite(bv) else float("nan"),
            "ratio": bv / av if math.isfinite(av) and abs(av) > 1e-12 and math.isfinite(bv) else float("nan"),
        })
    rows.append({
        "metric": "summary_source_note",
        "v10a": "summary_json",
        "v10b": "summary_json",
        "delta": "percentage metrics only; absolute pnl ignored for decision",
        "ratio": "",
    })
    return pd.DataFrame(rows)


def _side_col(df: pd.DataFrame) -> pd.Series:
    if "type" in df.columns:
        return df["type"].astype(str).str.upper().map({"LONG": 1, "SHORT": -1}).fillna(0).astype(int)
    if "side" in df.columns:
        return pd.to_numeric(df["side"], errors="coerce").fillna(0).astype(int)
    return pd.Series(0, index=df.index)


def prepare_trades(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["_row_id"] = range(len(out))
    out["entry_time_ts"] = pd.to_datetime(out.get("entry_time"), errors="coerce")
    out["exit_time_ts"] = pd.to_datetime(out.get("exit_time"), errors="coerce")
    out["side_int"] = _side_col(out)
    out["engine_key"] = out.get("engine", "UNKNOWN").astype(str).fillna("UNKNOWN")
    out["entry_price_key"] = pd.to_numeric(out.get("first_entry"), errors="coerce").round(3)
    out["avg_entry_key"] = pd.to_numeric(out.get("avg_entry"), errors="coerce").round(3)
    # Stable exact-enough key. Occurrence guards against duplicate same-bar entries.
    out["trade_key_base"] = (
        out["entry_time_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("NA")
        + "|" + out["side_int"].astype(str)
        + "|" + out["engine_key"]
        + "|" + out["entry_price_key"].astype(str)
    )
    out["trade_key_occ"] = out.groupby("trade_key_base").cumcount()
    out["trade_key"] = out["trade_key_base"] + "|" + out["trade_key_occ"].astype(str)
    if out["trade_key_base"].str.contains("NA", na=False).any():
        # fallback to row-order key for rows without timestamps; never allow NaN cartesian join.
        missing = out["trade_key_base"].str.contains("NA", na=False)
        out.loc[missing, "trade_key"] = label + "|ROW|" + out.loc[missing, "_row_id"].astype(str)
    out["return_pct_num"] = pd.to_numeric(out.get("return_pct"), errors="coerce")
    out["mfe_r_num"] = pd.to_numeric(out.get("mfe_r"), errors="coerce")
    out["mae_r_num"] = pd.to_numeric(out.get("mae_r"), errors="coerce")
    return out


def build_trade_diff(v10a_trades: pd.DataFrame, v10b_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = prepare_trades(v10a_trades, "v10a")
    b = prepare_trades(v10b_trades, "v10b")
    if a.empty or b.empty:
        return pd.DataFrame(), pd.DataFrame([{"issue": "missing trades", "severity": "FAIL"}])
    a_pref = a.add_prefix("v10a_")
    b_pref = b.add_prefix("v10b_")
    merged = a_pref.merge(
        b_pref,
        left_on="v10a_trade_key",
        right_on="v10b_trade_key",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    merged["matched"] = merged["_merge"].eq("both")
    merged["return_delta_pct"] = pd.to_numeric(merged.get("v10b_return_pct_num"), errors="coerce") - pd.to_numeric(merged.get("v10a_return_pct_num"), errors="coerce")
    merged["exit_time_changed"] = merged.get("v10a_exit_time_ts").astype(str) != merged.get("v10b_exit_time_ts").astype(str)
    merged["note_changed"] = merged.get("v10a_note").astype(str) != merged.get("v10b_note").astype(str)
    merged["structural_update_count"] = pd.to_numeric(merged.get("v10b_structure_updates"), errors="coerce").fillna(0)
    merged["v10b_has_struct_update"] = merged["structural_update_count"].gt(0)
    merged["v10a_win"] = pd.to_numeric(merged.get("v10a_return_pct_num"), errors="coerce").gt(0)
    merged["v10b_win"] = pd.to_numeric(merged.get("v10b_return_pct_num"), errors="coerce").gt(0)
    merged["loss_to_win"] = merged["matched"] & (~merged["v10a_win"]) & merged["v10b_win"]
    merged["win_to_loss"] = merged["matched"] & merged["v10a_win"] & (~merged["v10b_win"])

    matched = merged[merged["matched"]]
    detail_cols = [
        "_merge", "matched", "return_delta_pct", "exit_time_changed", "note_changed",
        "v10a_entry_time", "v10a_exit_time", "v10a_type", "v10a_engine", "v10a_return_pct", "v10a_mfe_r", "v10a_note",
        "v10b_entry_time", "v10b_exit_time", "v10b_type", "v10b_engine", "v10b_return_pct", "v10b_mfe_r", "v10b_note",
        "v10b_structure_updates", "v10b_structural_stop_source",
        "loss_to_win", "win_to_loss",
    ]
    detail = merged[[c for c in detail_cols if c in merged.columns]].copy()
    a_key_nulls = int(a["trade_key"].isna().sum())
    b_key_nulls = int(b["trade_key"].isna().sum())
    a_key_dupes = int(a["trade_key"].duplicated().sum())
    b_key_dupes = int(b["trade_key"].duplicated().sum())
    row_conservation = int(merged["matched"].sum()) + int(merged["_merge"].eq("left_only").sum()) == len(a) and int(merged["matched"].sum()) + int(merged["_merge"].eq("right_only").sum()) == len(b)
    key_quality_pass = (a_key_nulls == 0 and b_key_nulls == 0 and a_key_dupes == 0 and b_key_dupes == 0 and bool(row_conservation))
    summary = pd.DataFrame([{
        "baseline_trades": len(a),
        "candidate_trades": len(b),
        "matched_trades": int(merged["matched"].sum()),
        "baseline_only_trades": int(merged["_merge"].eq("left_only").sum()),
        "candidate_only_trades": int(merged["_merge"].eq("right_only").sum()),
        "matched_return_delta_sum_pct": float(matched["return_delta_pct"].sum(skipna=True)),
        "matched_return_delta_avg_pct": float(matched["return_delta_pct"].mean(skipna=True)) if not matched.empty else float("nan"),
        "improved_matched_trades": int(matched["return_delta_pct"].gt(0).sum()),
        "worsened_matched_trades": int(matched["return_delta_pct"].lt(0).sum()),
        "loss_to_win_count": int(merged["loss_to_win"].sum()),
        "win_to_loss_count": int(merged["win_to_loss"].sum()),
        "exit_time_changed_count": int((matched["exit_time_changed"]).sum()) if not matched.empty else 0,
        "note_changed_count": int((matched["note_changed"]).sum()) if not matched.empty else 0,
        "structural_update_matched_trades": int(matched["v10b_has_struct_update"].sum()) if not matched.empty else 0,
        "v10a_trade_key_null_count": a_key_nulls,
        "v10b_trade_key_null_count": b_key_nulls,
        "v10a_trade_key_duplicate_count": a_key_dupes,
        "v10b_trade_key_duplicate_count": b_key_dupes,
        "row_conservation_pass": bool(row_conservation),
        "key_quality_pass": bool(key_quality_pass),
        "diff_join_validation": "pandas_validate_one_to_one_passed",
    }])
    return detail, summary


def build_engine_compare(v10a_trades: pd.DataFrame, v10b_trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("v10a", v10a_trades), ("v10b", v10b_trades)]:
        if df.empty or "engine" not in df.columns:
            continue
        x = df.copy()
        x["return_pct"] = pd.to_numeric(x.get("return_pct"), errors="coerce")
        x["win"] = x["return_pct"].gt(0)
        for engine, g in x.groupby("engine", dropna=False):
            gross_profit = g.loc[g["return_pct"] > 0, "return_pct"].sum()
            gross_loss = g.loc[g["return_pct"] < 0, "return_pct"].sum()
            rows.append({
                "version": label,
                "engine": engine,
                "trades": len(g),
                "win_rate": float(g["win"].mean() * 100.0) if len(g) else 0.0,
                "sum_return_pct": float(g["return_pct"].sum(skipna=True)),
                "avg_return_pct": float(g["return_pct"].mean(skipna=True)),
                "pf_pct_based": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf"),
                "avg_mfe_r": float(pd.to_numeric(g.get("mfe_r"), errors="coerce").mean(skipna=True)),
                "avg_mae_r": float(pd.to_numeric(g.get("mae_r"), errors="coerce").mean(skipna=True)),
            })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    a = raw[raw["version"] == "v10a"].set_index("engine")
    b = raw[raw["version"] == "v10b"].set_index("engine")
    out = []
    for engine in sorted(set(a.index).union(set(b.index))):
        row = {"engine": engine}
        for col in ["trades", "win_rate", "sum_return_pct", "avg_return_pct", "pf_pct_based", "avg_mfe_r", "avg_mae_r"]:
            av = a.loc[engine, col] if engine in a.index else float("nan")
            bv = b.loc[engine, col] if engine in b.index else float("nan")
            row[f"v10a_{col}"] = av
            row[f"v10b_{col}"] = bv
            row[f"delta_{col}"] = bv - av if pd.notna(av) and pd.notna(bv) else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def build_yearly_compare(v10a_trades: pd.DataFrame, v10b_trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("v10a", v10a_trades), ("v10b", v10b_trades)]:
        if df.empty:
            continue
        x = df.copy()
        x["entry_time_ts"] = pd.to_datetime(x.get("entry_time"), errors="coerce")
        x["year"] = x["entry_time_ts"].dt.year
        x["return_pct"] = pd.to_numeric(x.get("return_pct"), errors="coerce")
        for year, g in x.groupby("year", dropna=False):
            if pd.isna(year):
                continue
            rows.append({
                "version": label,
                "year": int(year),
                "trades": len(g),
                "sum_return_pct": float(g["return_pct"].sum(skipna=True)),
                "win_rate": float(g["return_pct"].gt(0).mean() * 100.0),
                "avg_return_pct": float(g["return_pct"].mean(skipna=True)),
            })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    piv = raw.pivot(index="year", columns="version", values=["trades", "sum_return_pct", "win_rate", "avg_return_pct"])
    piv.columns = [f"{v}_{m}" for m, v in piv.columns]
    piv = piv.reset_index()
    if "v10a_sum_return_pct" in piv and "v10b_sum_return_pct" in piv:
        piv["delta_sum_return_pct"] = piv["v10b_sum_return_pct"] - piv["v10a_sum_return_pct"]
    if "v10a_win_rate" in piv and "v10b_win_rate" in piv:
        piv["delta_win_rate"] = piv["v10b_win_rate"] - piv["v10a_win_rate"]
    return piv


def top_dependency(df: pd.DataFrame, version: str) -> dict[str, Any]:
    if df.empty:
        return {"version": version}
    x = df.copy()
    x["return_pct"] = pd.to_numeric(x.get("return_pct"), errors="coerce")
    total = float(x["return_pct"].sum(skipna=True))
    winners = x[x["return_pct"] > 0].sort_values("return_pct", ascending=False)
    top1 = float(winners.head(1)["return_pct"].sum()) if not winners.empty else 0.0
    top3 = float(winners.head(3)["return_pct"].sum()) if not winners.empty else 0.0
    return {
        "version": version,
        "total_return_sum_pct": total,
        "top1_return_pct": top1,
        "top3_return_pct": top3,
        "top1_dependency_pct": top1 / total * 100.0 if abs(total) > 1e-12 else float("nan"),
        "top3_dependency_pct": top3 / total * 100.0 if abs(total) > 1e-12 else float("nan"),
        "return_sum_without_top1_pct": total - top1,
        "return_sum_without_top3_pct": total - top3,
    }


def build_structural_audit(v10b: dict[str, Any]) -> pd.DataFrame:
    trades = v10b["trades"].copy()
    summary = v10b["summary"]
    rows = []
    if trades.empty:
        return pd.DataFrame([{"check": "trades_present", "status": "FAIL", "detail": "no V10B trades"}])
    updates = pd.to_numeric(trades.get("structure_updates"), errors="coerce").fillna(0)
    src = trades.get("structural_stop_source", pd.Series("", index=trades.index)).astype(str)
    rows.extend([
        {"check": "structural_stop_enabled", "status": "PASS" if summary.get("structural_stop_enabled") is True else "FAIL", "detail": summary.get("structural_stop_enabled")},
        {"check": "lookback_is_21", "status": "PASS" if int(metric(summary, "structural_stop_lookback_bars", -1)) == 21 else "FAIL", "detail": summary.get("structural_stop_lookback_bars")},
        {"check": "scope_all", "status": "PASS" if str(summary.get("structural_stop_scope")).upper() == "ALL" else "FAIL", "detail": summary.get("structural_stop_scope")},
        {"check": "buffer_zero", "status": "PASS" if abs(metric(summary, "structural_stop_buffer_atr", 999)) < 1e-12 else "FAIL", "detail": summary.get("structural_stop_buffer_atr")},
        {"check": "trigger_zero", "status": "PASS" if abs(metric(summary, "structural_stop_trigger_mfe_r", 999)) < 1e-12 else "FAIL", "detail": summary.get("structural_stop_trigger_mfe_r")},
        {"check": "has_structural_updates", "status": "PASS" if int(updates.sum()) > 0 else "FAIL", "detail": int(updates.sum())},
        {"check": "structural_source_names", "status": "PASS" if src.str.contains("STRUCT_SWING|NONE", regex=True).all() else "WARN", "detail": src.value_counts().to_dict()},
        {"check": "no_position_sizing_change_claim", "status": "MANUAL", "detail": "V10B should not change initial stop/sizing; verify code and trade diff."},
    ])
    return pd.DataFrame(rows)


def build_no_lookahead_audit() -> pd.DataFrame:
    path = project_path(V10B_SCRIPT)
    full_text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = full_text.find("def run_v10b_backtest(")
    end = full_text.find("def write_outputs(", start if start >= 0 else 0)
    text = full_text[start:end] if start >= 0 and end > start else full_text
    checks = []

    def order(a: str, b: str) -> bool:
        ia = text.find(a)
        ib = text.find(b)
        return ia >= 0 and ib >= 0 and ia < ib

    def full_order(*parts: str) -> bool:
        positions = [text.find(p) for p in parts]
        return all(x >= 0 for x in positions) and positions == sorted(positions)

    checks.append({
        "check": "full_21_bar_structural_window",
        "status": "PASS" if "min_periods=lookback_bars" in full_text else "FAIL",
        "detail": "Promoted V10B must not use min_periods=3; no structural candidate until full lookback is available.",
    })
    checks.append({
        "check": "structural_columns_before_trade_slice",
        "status": "PASS" if full_text.find("features = add_structural_columns(features") < full_text.find("features = features.loc[trade_start") else "FAIL",
        "detail": "Compute rolling structural columns on warmup-inclusive features before slicing to trade_start.",
    })
    checks.append({
        "check": "delayed_range_exit_before_bar_high_low",
        "status": "PASS" if full_order("if pending_range_exit_i is not None and i >= pending_range_exit_i", "exit_price = v10a.apply_exit_slippage(float(row.open)", "else:\n                high = float(row.high)") else "WARN",
        "detail": "Non-default delayed range exit should execute at bar open before reading completed high/low.",
    })
    checks.append({
        "check": "active_stop_snapshotted_before_stop_touch",
        "status": "PASS" if order("active_stop = stop_price", "touched_stop") else "FAIL",
        "detail": "Current bar stop touch should use active_stop captured at bar start.",
    })
    checks.append({
        "check": "stop_touch_uses_active_stop",
        "status": "PASS" if ("low <= active_stop" in text and "high >= active_stop" in text) else "FAIL",
        "detail": "Current bar stop touch must not use freshly computed next_stop.",
    })
    checks.append({
        "check": "structural_update_after_exit_decision",
        "status": "PASS" if order("if exit_now:", "candidate, source = _structural_stop_candidate") else "FAIL",
        "detail": "Do not record or commit structural updates on bars that already exit.",
    })
    checks.append({
        "check": "structural_update_not_above_long_close_or_below_short_close",
        "status": "PASS" if ("candidate < close" in text and "candidate > close" in text) else "FAIL",
        "detail": "Avoid impossible stop beyond current close on update bar.",
    })
    checks.append({
        "check": "add_on_blocked_when_pending_range_exit",
        "status": "PASS" if "in_pos and pending_range_exit_i is None and units < active_cfg.max_units" in text else "FAIL",
        "detail": "V10B should preserve V10A add-on guard while a delayed range exit is pending.",
    })
    checks.append({
        "check": "current_completed_bar_comment",
        "status": "PASS" if "completed" in full_text.lower() and "future" in full_text.lower() else "WARN",
        "detail": "Documentation should state completed-bar only timing.",
    })
    checks.append({
        "check": "manual_review_required",
        "status": "MANUAL",
        "detail": "Static audit is still not proof. Run report-level audit, parity, and dry-run before live.",
    })
    return pd.DataFrame(checks)

def run_cmd(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def common_backtest_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--warmup-start-date", args.warmup_start_date,
        "--initial-capital", str(args.initial_capital),
        "--range-pct", str(args.range_pct),
        "--price-step", str(args.price_step),
    ]
    return out


def rerun_formal(args: argparse.Namespace) -> None:
    v10a_out = project_path(args.v10a_dir)
    v10b_out = project_path(args.v10b_dir)
    ensure_dir(v10a_out)
    ensure_dir(v10b_out)
    base = common_backtest_args(args)
    rc1 = run_cmd([sys.executable, V10A_SCRIPT, *base, "--out-dir", str(v10a_out)])
    if rc1 != 0:
        raise RuntimeError(f"V10A rerun failed: {rc1}")
    rc2 = run_cmd([sys.executable, V10B_SCRIPT, *base, "--out-dir", str(v10b_out)])
    if rc2 != 0:
        raise RuntimeError(f"V10B rerun failed: {rc2}")


def _grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_stress(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rows = []
    stress_root = out_dir / "stress_runs"
    ensure_dir(stress_root)
    base = common_backtest_args(args)
    for fee in _grid(args.fee_grid):
        for slip in _grid(args.slippage_grid):
            for version, script, name in [
                ("v10a", V10A_SCRIPT, V10A_NAME),
                ("v10b", V10B_SCRIPT, V10B_NAME),
            ]:
                subdir = stress_root / f"{version}_fee{str(fee).replace('.', 'p')}_slip{str(slip).replace('.', 'p')}"
                cmd = [
                    sys.executable, script, *base,
                    "--fee-rate", str(fee),
                    "--slippage-pct", str(slip),
                    "--out-dir", str(subdir),
                ]
                rc = run_cmd(cmd)
                if rc != 0:
                    rows.append({"version": version, "fee_rate": fee, "slippage_pct": slip, "status": "FAIL", "returncode": rc})
                    continue
                rep = load_report(subdir, name)
                rows.append({
                    "version": version,
                    "fee_rate": fee,
                    "slippage_pct": slip,
                    "status": "PASS",
                    "total_return_pct": metric(rep["summary"], "total_return_pct", float("nan")),
                    "max_drawdown_pct": metric(rep["summary"], "max_drawdown_pct", float("nan")),
                    "profit_factor": metric(rep["summary"], "profit_factor", float("nan")),
                    "win_rate": metric(rep["summary"], "win_rate", float("nan")),
                    "total_trades": metric(rep["summary"], "total_trades", float("nan")),
                })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    a = raw[raw["version"] == "v10a"].set_index(["fee_rate", "slippage_pct"])
    b = raw[raw["version"] == "v10b"].set_index(["fee_rate", "slippage_pct"])
    compare_rows = []
    for key in sorted(set(a.index).intersection(set(b.index))):
        ar = a.loc[key]
        br = b.loc[key]
        compare_rows.append({
            "fee_rate": key[0],
            "slippage_pct": key[1],
            "v10a_total_return_pct": ar.get("total_return_pct"),
            "v10b_total_return_pct": br.get("total_return_pct"),
            "return_ratio_v10b_vs_v10a": br.get("total_return_pct") / ar.get("total_return_pct") if abs(ar.get("total_return_pct", 0)) > 1e-12 else float("nan"),
            "v10a_max_drawdown_pct": ar.get("max_drawdown_pct"),
            "v10b_max_drawdown_pct": br.get("max_drawdown_pct"),
            "drawdown_delta": br.get("max_drawdown_pct") - ar.get("max_drawdown_pct"),
            "v10a_profit_factor": ar.get("profit_factor"),
            "v10b_profit_factor": br.get("profit_factor"),
            "pf_delta": br.get("profit_factor") - ar.get("profit_factor"),
            "v10a_win_rate": ar.get("win_rate"),
            "v10b_win_rate": br.get("win_rate"),
            "win_rate_delta": br.get("win_rate") - ar.get("win_rate"),
        })
    return pd.DataFrame(compare_rows)


def decision_matrix(args: argparse.Namespace, summary_compare: pd.DataFrame, top_dep: pd.DataFrame, no_look: pd.DataFrame, stress_df: pd.DataFrame) -> pd.DataFrame:
    def get(metric_name: str, col: str) -> float:
        row = summary_compare[summary_compare["metric"] == metric_name]
        if row.empty:
            return float("nan")
        return nfloat(row.iloc[0].get(col), float("nan"))
    total_ratio = get("total_return_pct", "ratio")
    dd_delta = get("max_drawdown_pct", "delta")
    pf_delta = get("profit_factor", "delta")
    win_delta = get("win_rate", "delta")
    top3_delta = float("nan")
    if not top_dep.empty and {"version", "top3_dependency_pct"}.issubset(top_dep.columns):
        vals = top_dep.set_index("version")["top3_dependency_pct"]
        if "v10a" in vals.index and "v10b" in vals.index:
            top3_delta = float(vals.loc["v10b"] - vals.loc["v10a"])
    no_look_fail = int(no_look["status"].eq("FAIL").sum()) if not no_look.empty and "status" in no_look.columns else 999
    stress_min_ratio = float("nan")
    stress_min_pf_delta = float("nan")
    stress_max_dd_delta = float("nan")
    if not stress_df.empty:
        stress_min_ratio = float(pd.to_numeric(stress_df.get("return_ratio_v10b_vs_v10a"), errors="coerce").min(skipna=True))
        stress_min_pf_delta = float(pd.to_numeric(stress_df.get("pf_delta"), errors="coerce").min(skipna=True))
        stress_max_dd_delta = float(pd.to_numeric(stress_df.get("drawdown_delta"), errors="coerce").max(skipna=True))
    rows = [
        {"gate": "return_ratio", "value": total_ratio, "threshold": f">= {args.min_return_ratio}", "pass": bool(total_ratio >= args.min_return_ratio)},
        {"gate": "max_drawdown_delta", "value": dd_delta, "threshold": f"<= {args.max_dd_delta}", "pass": bool(dd_delta <= args.max_dd_delta)},
        {"gate": "profit_factor_delta", "value": pf_delta, "threshold": f">= {args.min_pf_delta}", "pass": bool(pf_delta >= args.min_pf_delta)},
        {"gate": "top3_dependency_delta", "value": top3_delta, "threshold": f"<= {args.max_top3_dependency_delta}", "pass": bool(pd.notna(top3_delta) and top3_delta <= args.max_top3_dependency_delta)},
        {"gate": "no_lookahead_static_fail_count", "value": no_look_fail, "threshold": "== 0", "pass": bool(no_look_fail == 0)},
        {"gate": "stress_min_return_ratio", "value": stress_min_ratio, "threshold": ">= 1.0 if stress run", "pass": bool(pd.isna(stress_min_ratio) or stress_min_ratio >= 1.0)},
        {"gate": "stress_min_pf_delta", "value": stress_min_pf_delta, "threshold": ">= 0 if stress run", "pass": bool(pd.isna(stress_min_pf_delta) or stress_min_pf_delta >= 0)},
        {"gate": "stress_max_drawdown_delta", "value": stress_max_dd_delta, "threshold": "<= 0 if stress run", "pass": bool(pd.isna(stress_max_dd_delta) or stress_max_dd_delta <= 0)},
        {"gate": "win_rate_delta_info", "value": win_delta, "threshold": "informational; V10B is not win-rate strategy", "pass": True},
    ]
    df = pd.DataFrame(rows)
    final = bool(df.loc[~df["gate"].eq("win_rate_delta_info"), "pass"].all())
    df.loc[len(df)] = {"gate": "overall_research_ready_for_aetheredge_parity", "value": final, "threshold": "all hard gates pass; then still requires AetherEdge parity/dry-run", "pass": final}
    return df


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    ensure_dir(out_dir)

    if args.rerun_formal:
        rerun_formal(args)

    v10a = load_report(project_path(args.v10a_dir), V10A_NAME)
    v10b = load_report(project_path(args.v10b_dir), V10B_NAME)
    if not v10a["summary"] or v10a["trades"].empty:
        raise RuntimeError(f"Missing V10A report under {project_path(args.v10a_dir)}. Run --rerun-formal first or pass --v10a-dir.")
    if not v10b["summary"] or v10b["trades"].empty:
        raise RuntimeError(f"Missing V10B report under {project_path(args.v10b_dir)}. Run --rerun-formal first or pass --v10b-dir.")

    summary_df = build_summary_compare(v10a, v10b)
    engine_df = build_engine_compare(v10a["trades"], v10b["trades"])
    yearly_df = build_yearly_compare(v10a["trades"], v10b["trades"])
    trade_detail, trade_summary = build_trade_diff(v10a["trades"], v10b["trades"])
    top_df = pd.DataFrame([top_dependency(v10a["trades"], "v10a"), top_dependency(v10b["trades"], "v10b")])
    struct_df = build_structural_audit(v10b)
    no_look_df = build_no_lookahead_audit()

    stress_df = pd.DataFrame()
    if args.stress:
        stress_df = run_stress(args, out_dir)

    decision_df = decision_matrix(args, summary_df, top_df, no_look_df, stress_df)

    summary_df.to_csv(out_dir / "01_summary_compare.csv", index=False)
    engine_df.to_csv(out_dir / "02_engine_compare.csv", index=False)
    yearly_df.to_csv(out_dir / "03_yearly_compare.csv", index=False)
    top_df.to_csv(out_dir / "04_top_trade_dependency.csv", index=False)
    trade_summary.to_csv(out_dir / "05_trade_diff_summary.csv", index=False)
    trade_detail.to_csv(out_dir / "06_trade_diff_detail.csv", index=False)
    struct_df.to_csv(out_dir / "07_structural_stop_audit.csv", index=False)
    no_look_df.to_csv(out_dir / "08_no_lookahead_static_audit.csv", index=False)
    stress_df.to_csv(out_dir / "09_fee_slippage_stress.csv", index=False)
    decision_df.to_csv(out_dir / "10_decision_matrix.csv", index=False)
    if args.write_combined_trades:
        a = v10a["trades"].copy(); a["version"] = "v10a"
        b = v10b["trades"].copy(); b["version"] = "v10b"
        pd.concat([a, b], ignore_index=True, sort=False).to_csv(out_dir / "11_combined_trades.csv", index=False)

    meta = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "project_root": PROJECT_ROOT,
        "v10a_dir": str(project_path(args.v10a_dir)),
        "v10b_dir": str(project_path(args.v10b_dir)),
        "out_dir": str(out_dir),
        "percentage_metric_policy": "Decision uses total_return_pct/return_pct/PF/win_rate/drawdown, not absolute pnl.",
        "no_lookahead_policy": "Static audit only; manual review and AetherEdge parity are still required before live.",
        "next_steps": [
            "If gates pass, create AetherEdge eth_lf_portfolio_v10b plugin.",
            "Run backtest-live parity on closed 4H bars.",
            "Run V10B dry-run/shadow-run before live trading.",
        ],
        "args": vars(args),
    }
    with (out_dir / "99_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 96)
    print("V10B Final Verification outputs written to:", out_dir.resolve())
    print("Key files: 01_summary_compare.csv, 05_trade_diff_summary.csv, 08_no_lookahead_static_audit.csv, 10_decision_matrix.csv")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
