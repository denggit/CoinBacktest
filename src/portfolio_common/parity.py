#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parity comparison between refactored and legacy Portfolio V1 reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio_common.allocator import edge_attribution, standardize_trades

SUMMARY_FIELDS = ["trades", "total_return", "final_capital", "max_drawdown", "win_rate", "profit_factor"]
TRADES_FIELDS = ["entry_time", "exit_time", "side", "pnl", "fee", "return", "display_exit_reason"]
EQUITY_FIELDS = ["timestamp", "equity", "drawdown"]
EDGE_FIELDS = ["edge_id", "trades", "pnl", "return", "win_rate", "profit_factor"]
EXACT_FIELDS = {"entry_time", "exit_time", "timestamp", "side", "display_exit_reason", "edge_id"}


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    report: pd.DataFrame
    first_diff: dict[str, object] | None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _primary_summary(df: pd.DataFrame, primary_scenario: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "scenario" in df.columns:
        part = df.loc[df["scenario"].astype(str).eq(str(primary_scenario))].copy()
        if not part.empty:
            return part.head(1).reset_index(drop=True)
    return df.head(1).reset_index(drop=True)


def _normalize_summary(df: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "total_trades" in out.columns and "trades" not in out.columns:
        out["trades"] = out["total_trades"]
    if "return_total" in out.columns and "total_return" not in out.columns:
        out["total_return"] = out["return_total"]
    if "total_fee" not in out.columns:
        if trades is not None and not trades.empty and "fee" in trades.columns:
            out["total_fee"] = float(pd.to_numeric(trades["fee"], errors="coerce").fillna(0.0).sum())
        else:
            out["total_fee"] = 0.0
    return out


def _legacy_trades(report_dir: Path, primary_scenario: str) -> pd.DataFrame:
    combined = _read_csv(report_dir / "03_combined_trades.csv")
    if combined.empty:
        return pd.DataFrame()
    if "scenario" in combined.columns:
        combined = combined.loc[combined["scenario"].astype(str).eq(str(primary_scenario))].copy()
    if combined.empty:
        return pd.DataFrame()
    for col in ["entry_time", "exit_time"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    return standardize_trades(combined)


def load_report(report_dir: str | Path, primary_scenario: str) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    standard_trades = _read_csv(root / "02_trades.csv")
    if not standard_trades.empty:
        summary = _normalize_summary(_primary_summary(_read_csv(root / "01_summary.csv"), primary_scenario), standard_trades)
        equity = _read_csv(root / "03_equity.csv")
        edge = _read_csv(root / "04_edge_attribution.csv")
        return {"summary": summary, "trades": standard_trades, "equity": equity, "edge": edge}

    trades = _legacy_trades(root, primary_scenario)
    summary = _normalize_summary(_primary_summary(_read_csv(root / "04_scenario_summary.csv"), primary_scenario), trades)
    equity = _read_csv(root / "03_equity.csv")
    if equity.empty:
        combined = _read_csv(root / "03_combined_trades.csv")
        if not combined.empty and "scenario" in combined.columns:
            combined = combined.loc[combined["scenario"].astype(str).eq(str(primary_scenario))].copy()
        if not combined.empty and "portfolio_capital" in combined.columns:
            combined["exit_time"] = pd.to_datetime(combined["exit_time"], errors="coerce")
            combined = combined.dropna(subset=["exit_time"]).sort_values(["exit_time", "strategy_leg", "entry_time"]).reset_index(drop=True)
            eq = pd.to_numeric(combined["portfolio_capital"], errors="coerce")
            peak = eq.cummax()
            equity = pd.DataFrame({"timestamp": combined["exit_time"], "equity": eq, "drawdown": eq / peak - 1.0})
    edge = edge_attribution(trades) if not trades.empty else pd.DataFrame(columns=EDGE_FIELDS)
    return {"summary": summary, "trades": trades, "equity": equity, "edge": edge}


def _text_value(value: Any, *, is_time: bool = False) -> str:
    if pd.isna(value):
        return ""
    if is_time:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _num_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _classify_reason(field: str, new_value: Any, old_value: Any) -> str:
    if field in {"entry_time", "exit_time", "timestamp"}:
        return "time_alignment_mismatch"
    if field == "fee":
        return "fee_model_mismatch"
    if field in {"entry_price", "exit_price"}:
        return "execution_price_mismatch"
    if field in {"exit_reason", "display_exit_reason"}:
        return "exit_reason_mismatch"
    if field == "side":
        return "signal_priority_mismatch"
    if field in {"pnl", "return", "total_return", "equity", "drawdown", "avg_trade_return"}:
        return "position_sizing_mismatch"
    if field == "_row_count":
        return "data_load_mismatch"
    return "unknown"


def _compare_frames(
    file_name: str,
    new: pd.DataFrame,
    old: pd.DataFrame,
    fields: list[str],
    *,
    tolerance: float,
    key_field: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(new) != len(old):
        rows.append(
            {
                "file": file_name,
                "row": 0,
                "key": "_row_count",
                "field": "_row_count",
                "new_value": len(new),
                "old_value": len(old),
                "abs_diff": abs(len(new) - len(old)),
                "tolerance": tolerance,
                "reason": _classify_reason("_row_count", len(new), len(old)),
            }
        )
    n = min(len(new), len(old))
    for i in range(n):
        key = ""
        if key_field and key_field in new.columns:
            key = _text_value(new.iloc[i].get(key_field), is_time=key_field in {"timestamp", "entry_time", "exit_time"})
        elif "trade_id" in new.columns:
            key = str(new.iloc[i].get("trade_id", i))
        else:
            key = str(i)
        for field in fields:
            if field not in new.columns and field not in old.columns:
                continue
            if field not in new.columns:
                rows.append(
                    {
                        "file": file_name,
                        "row": i,
                        "key": key,
                        "field": field,
                        "new_value": "<missing>",
                        "old_value": old.iloc[i].get(field) if field in old.columns else "<missing>",
                        "abs_diff": np.nan,
                        "tolerance": tolerance,
                        "reason": "missing_legacy_logic",
                    }
                )
                continue
            if field not in old.columns:
                rows.append(
                    {
                        "file": file_name,
                        "row": i,
                        "key": key,
                        "field": field,
                        "new_value": new.iloc[i].get(field),
                        "old_value": "<missing>",
                        "abs_diff": np.nan,
                        "tolerance": tolerance,
                        "reason": "missing_legacy_logic",
                    }
                )
                continue
            nv = new.iloc[i].get(field)
            ov = old.iloc[i].get(field)
            if field in EXACT_FIELDS:
                is_time = field in {"entry_time", "exit_time", "timestamp"}
                nv_str = _text_value(nv, is_time=is_time)
                ov_str = _text_value(ov, is_time=is_time)
                if nv_str != ov_str:
                    rows.append(
                        {
                            "file": file_name,
                            "row": i,
                            "key": key,
                            "field": field,
                            "new_value": nv_str,
                            "old_value": ov_str,
                            "abs_diff": np.nan,
                            "tolerance": "exact",
                            "reason": _classify_reason(field, nv, ov),
                        }
                    )
                continue
            nf = _num_value(nv)
            of = _num_value(ov)
            diff = abs(nf - of) if np.isfinite(nf) and np.isfinite(of) else (0.0 if pd.isna(nv) and pd.isna(ov) else np.nan)
            same = bool(np.isfinite(diff) and diff <= tolerance)
            if not same:
                rows.append(
                    {
                        "file": file_name,
                        "row": i,
                        "key": key,
                        "field": field,
                        "new_value": nv,
                        "old_value": ov,
                        "abs_diff": diff,
                        "tolerance": tolerance,
                        "reason": _classify_reason(field, nv, ov),
                    }
                )
    return pd.DataFrame(rows)


def _build_scenario_summary_parity(
    new_summary: pd.DataFrame,
    old_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare all time48 scenario summaries between new and old reports."""
    rows: list[dict[str, object]] = []
    if new_summary.empty and old_summary.empty:
        return pd.DataFrame(rows)
    if "scenario" not in new_summary.columns or "scenario" not in old_summary.columns:
        rows.append({"section": "scenario_summary", "check": "scenario_column", "passed": False, "detail": "scenario column missing"})
        return pd.DataFrame(rows)

    new_scenarios = set(new_summary["scenario"].astype(str))
    old_scenarios = set(old_summary["scenario"].astype(str))

    # Check for MFE variant presence (should NOT be in new)
    mfe_in_new = [s for s in new_scenarios if "mfe_lock" in s.lower()]
    mfe_in_old = [s for s in old_scenarios if "mfe_lock" in s.lower()]

    rows.append(
        {
            "section": "scenario_summary",
            "check": "mfe_lock_absent_in_new",
            "passed": len(mfe_in_new) == 0,
            "detail": f"MFE scenarios in new: {len(mfe_in_new)}; in old: {len(mfe_in_old)}",
        }
    )

    # Compare time48 scenarios only
    time48_new = {s for s in new_scenarios if "time48" in s}
    time48_old = {s for s in old_scenarios if "time48" in s}

    rows.append(
        {
            "section": "scenario_summary",
            "check": "time48_scenario_count",
            "passed": len(time48_new) == len(time48_old),
            "detail": f"new={len(time48_new)} old={len(time48_old)}",
        }
    )

    only_new = time48_new - time48_old
    only_old = time48_old - time48_new
    common = time48_new & time48_old

    if only_new:
        rows.append(
            {"section": "scenario_summary", "check": "time48_only_new", "passed": False, "detail": str(only_new)}
        )
    if only_old:
        rows.append(
            {"section": "scenario_summary", "check": "time48_only_old", "passed": False, "detail": str(only_old)}
        )

    # Compare metrics for common scenarios
    compare_fields = ["trades", "total_return", "final_capital", "max_drawdown", "win_rate", "profit_factor"]
    new_idx = new_summary.set_index("scenario")
    old_idx = old_summary.set_index("scenario")
    mismatches = 0
    for s in sorted(common):
        for f in compare_fields:
            if f not in new_idx.columns or f not in old_idx.columns:
                continue
            nv = _num_value(new_idx.loc[s, f]) if s in new_idx.index else float("nan")
            ov = _num_value(old_idx.loc[s, f]) if s in old_idx.index else float("nan")
            diff = abs(nv - ov) if np.isfinite(nv) and np.isfinite(ov) else (0.0 if pd.isna(nv) and pd.isna(ov) else np.nan)
            if not np.isfinite(diff) or diff > 1e-8:
                mismatches += 1
                if mismatches <= 5:
                    rows.append(
                        {
                            "section": "scenario_summary",
                            "check": f"metric:{s}:{f}",
                            "passed": False,
                            "detail": f"new={nv} old={ov} diff={diff}",
                        }
                    )

    all_scenarios_match = (mismatches == 0) and (len(time48_new) == len(time48_old)) and (len(only_new) == 0) and (len(only_old) == 0)
    rows.append(
        {
            "section": "scenario_summary",
            "check": "all_time48_scenarios_match",
            "passed": all_scenarios_match,
            "detail": f"common={len(common)} mismatches={mismatches}",
        }
    )
    return pd.DataFrame(rows)


def _build_parity_report(
    summary_passed: bool,
    summary_diff_count: int,
    trades_passed: bool,
    trades_diff_count: int,
    trades_entry_time_diffs: int,
    trades_exit_time_diffs: int,
    trades_side_diffs: int,
    trades_pnl_max_abs_diff: float,
    trades_fee_max_abs_diff: float,
    trades_return_max_abs_diff: float,
    trades_display_exit_reason_diffs: int,
    equity_passed: bool,
    equity_diff_count: int,
    edge_passed: bool,
    edge_diff_count: int,
    scenario_summary_passed: bool,
    scenario_summary_detail: str,
    mfe_absent_in_new: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"section": "primary_summary_passed", "value": summary_passed},
            {"section": "primary_trades_passed", "value": trades_passed},
            {"section": "trades_entry_time_diffs", "value": trades_entry_time_diffs},
            {"section": "trades_exit_time_diffs", "value": trades_exit_time_diffs},
            {"section": "trades_side_diffs", "value": trades_side_diffs},
            {"section": "trades_pnl_max_abs_diff", "value": trades_pnl_max_abs_diff},
            {"section": "trades_fee_max_abs_diff", "value": trades_fee_max_abs_diff},
            {"section": "trades_return_max_abs_diff", "value": trades_return_max_abs_diff},
            {"section": "trades_display_exit_reason_diffs", "value": trades_display_exit_reason_diffs},
            {"section": "primary_equity_passed", "value": equity_passed},
            {"section": "primary_edge_passed", "value": edge_passed},
            {"section": "scenario_summary_all_time48_match", "value": scenario_summary_passed},
            {"section": "mfe_lock_absent_in_new", "value": mfe_absent_in_new},
            {"section": "scenario_summary_detail", "value": scenario_summary_detail},
        ]
    )


def run_parity(
    *,
    new_report_dir: str | Path,
    old_report_dir: str | Path,
    primary_scenario: str,
    tolerance: float = 1e-9,
) -> ParityResult:
    new = load_report(new_report_dir, primary_scenario)
    old = load_report(old_report_dir, primary_scenario)

    # --- Primary summary parity ---
    summary_diff = _compare_frames("01_summary.csv", new["summary"], old["summary"], SUMMARY_FIELDS, tolerance=tolerance)

    # --- Trades parity ---
    # For old trades, compute display_exit_reason for comparison if not present.
    old_trades = old["trades"].copy()
    if "display_exit_reason" not in old_trades.columns and "exit_reason" in old_trades.columns:
        old_trades["display_exit_reason"] = old_trades["exit_reason"].astype(str)
    new_trades = new["trades"].copy()
    if "display_exit_reason" not in new_trades.columns and "exit_reason" in new_trades.columns:
        # synthesize from strategy_leg + exit_reason
        def _synth_display(row: pd.Series) -> str:
            leg = str(row.get("strategy_leg", ""))
            reason = str(row.get("exit_reason", ""))
            if leg == "MF_LOW_SWEEP_TIME48":
                return f"MF_LOW_SWEEP_TIME48:{reason}"
            return f"LF_V10B:{reason}"
        new_trades["display_exit_reason"] = new_trades.apply(_synth_display, axis=1)

    trades_diff = _compare_frames("02_trades.csv", new_trades, old_trades, TRADES_FIELDS, tolerance=tolerance, key_field="trade_id")

    # --- Equity parity ---
    equity_diff = _compare_frames("03_equity.csv", new["equity"], old["equity"], EQUITY_FIELDS, tolerance=tolerance, key_field="timestamp")

    # --- Edge attribution parity ---
    edge_diff = _compare_frames("04_edge_attribution.csv", new["edge"], old["edge"], EDGE_FIELDS, tolerance=tolerance, key_field="edge_id")

    # --- Scenario summary parity ---
    new_scenario_summary = _read_csv(Path(new_report_dir) / "07_scenario_summary.csv")
    old_scenario_summary = _read_csv(Path(old_report_dir) / "04_scenario_summary.csv")
    if old_scenario_summary.empty:
        old_scenario_summary = _read_csv(Path(old_report_dir) / "01_summary.csv")
    scenario_parity = _build_scenario_summary_parity(new_scenario_summary, old_scenario_summary)

    # --- Aggregate into 91_parity_report.csv ---
    summary_passed = bool(summary_diff.empty)
    trades_passed = bool(trades_diff.empty)

    # Count trade-level diffs by field
    if trades_diff.empty:
        entry_time_diffs = 0
        exit_time_diffs = 0
        side_diffs = 0
        pnl_max_abs = 0.0
        fee_max_abs = 0.0
        ret_max_abs = 0.0
        display_exit_reason_diffs = 0
    else:
        entry_time_diffs = int((trades_diff["field"] == "entry_time").sum())
        exit_time_diffs = int((trades_diff["field"] == "exit_time").sum())
        side_diffs = int((trades_diff["field"] == "side").sum())
        pnl_vals = trades_diff.loc[trades_diff["field"] == "pnl", "abs_diff"]
        pnl_max_abs = float(pnl_vals.max()) if len(pnl_vals) else 0.0
        fee_vals = trades_diff.loc[trades_diff["field"] == "fee", "abs_diff"]
        fee_max_abs = float(fee_vals.max()) if len(fee_vals) else 0.0
        ret_vals = trades_diff.loc[trades_diff["field"] == "return", "abs_diff"]
        ret_max_abs = float(ret_vals.max()) if len(ret_vals) else 0.0
        display_exit_reason_diffs = int((trades_diff["field"] == "display_exit_reason").sum())

    equity_passed = bool(equity_diff.empty)
    edge_passed = bool(edge_diff.empty)

    scenario_passed = bool(scenario_parity.loc[scenario_parity["check"] == "all_time48_scenarios_match", "passed"].values[0]) if not scenario_parity.empty else False
    scenario_detail = ""
    if not scenario_parity.empty:
        detail_parts = []
        for _, r in scenario_parity.iterrows():
            detail_parts.append(f"{r['check']}={r['passed']}")
        scenario_detail = "; ".join(detail_parts)

    mfe_absent = True
    if not scenario_parity.empty:
        mfe_row = scenario_parity.loc[scenario_parity["check"] == "mfe_lock_absent_in_new", "passed"]
        mfe_absent = bool(mfe_row.values[0]) if len(mfe_row) else True

    report = _build_parity_report(
        summary_passed=summary_passed,
        summary_diff_count=int(len(summary_diff)),
        trades_passed=trades_passed,
        trades_diff_count=int(len(trades_diff)),
        trades_entry_time_diffs=entry_time_diffs,
        trades_exit_time_diffs=exit_time_diffs,
        trades_side_diffs=side_diffs,
        trades_pnl_max_abs_diff=pnl_max_abs,
        trades_fee_max_abs_diff=fee_max_abs,
        trades_return_max_abs_diff=ret_max_abs,
        trades_display_exit_reason_diffs=display_exit_reason_diffs,
        equity_passed=equity_passed,
        equity_diff_count=int(len(equity_diff)),
        edge_passed=edge_passed,
        edge_diff_count=int(len(edge_diff)),
        scenario_summary_passed=scenario_passed,
        scenario_summary_detail=scenario_detail,
        mfe_absent_in_new=mfe_absent,
    )

    # Determine overall pass: primary summary + trades + scenario summary must all pass.
    overall_passed = bool(summary_passed and trades_passed and scenario_passed)

    all_diffs = pd.concat([summary_diff, trades_diff, equity_diff, edge_diff], ignore_index=True, sort=False)
    first_diff = None if all_diffs.empty else all_diffs.iloc[0].to_dict()

    out_dir = Path(new_report_dir)
    report.to_csv(out_dir / "91_parity_report.csv", index=False)
    print(f"[parity] wrote 91_parity_report.csv", flush=True)
    if not trades_diff.empty:
        trades_diff.to_csv(out_dir / "91_parity_diff_trades.csv", index=False)
    if not equity_diff.empty:
        equity_diff.to_csv(out_dir / "91_parity_diff_equity.csv", index=False)

    return ParityResult(passed=overall_passed, report=report, first_diff=first_diff)
