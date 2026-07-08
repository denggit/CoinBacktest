#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parity comparison between refactored and legacy Portfolio V1 reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio_common.allocator import edge_attribution, standardize_trades

SUMMARY_FIELDS = ["trades", "total_return", "max_drawdown", "win_rate", "profit_factor", "avg_trade_return", "total_fee"]
TRADES_FIELDS = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl", "fee", "exit_reason"]
EQUITY_FIELDS = ["timestamp", "equity", "drawdown"]
EDGE_FIELDS = ["edge_id", "trades", "pnl", "return", "win_rate", "profit_factor"]
EXACT_FIELDS = {"entry_time", "exit_time", "timestamp", "side", "exit_reason", "edge_id"}
ALLOWED_REASONS = {
    "data_load_mismatch",
    "time_alignment_mismatch",
    "fee_model_mismatch",
    "slippage_model_mismatch",
    "execution_price_mismatch",
    "exit_priority_mismatch",
    "signal_priority_mismatch",
    "position_sizing_mismatch",
    "floating_point_mismatch",
    "missing_legacy_logic",
    "unknown",
}


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    report: pd.DataFrame
    first_diff: dict[str, object] | None
    trades_diff: pd.DataFrame
    equity_diff: pd.DataFrame


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
        if trades is not None and not trades.empty:
            out["total_fee"] = float(pd.to_numeric(trades.get("fee", 0.0), errors="coerce").fillna(0.0).sum())
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


def _legacy_equity(report_dir: Path, primary_scenario: str) -> pd.DataFrame:
    combined = _read_csv(report_dir / "03_combined_trades.csv")
    if combined.empty:
        return pd.DataFrame(columns=EQUITY_FIELDS)
    if "scenario" in combined.columns:
        combined = combined.loc[combined["scenario"].astype(str).eq(str(primary_scenario))].copy()
    if combined.empty or "portfolio_capital" not in combined.columns:
        return pd.DataFrame(columns=EQUITY_FIELDS)
    combined["exit_time"] = pd.to_datetime(combined["exit_time"], errors="coerce")
    combined = combined.dropna(subset=["exit_time"]).sort_values(["exit_time", "strategy_leg", "entry_time"]).reset_index(drop=True)
    equity = pd.to_numeric(combined["portfolio_capital"], errors="coerce")
    peak = equity.cummax()
    return pd.DataFrame(
        {
            "timestamp": combined["exit_time"],
            "equity": equity,
            "drawdown": equity / peak - 1.0,
        }
    )


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
    equity = _legacy_equity(root, primary_scenario)
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


def classify_reason(file_name: str, field: str, new_value: Any, old_value: Any, tol: float) -> str:
    if field in {"entry_time", "exit_time", "timestamp"}:
        return "time_alignment_mismatch"
    if field == "fee":
        return "fee_model_mismatch"
    if field in {"entry_price", "exit_price"}:
        return "execution_price_mismatch"
    if field == "exit_reason":
        return "exit_priority_mismatch"
    if field == "side":
        return "signal_priority_mismatch"
    if field in {"pnl", "return", "total_return", "equity", "drawdown", "avg_trade_return"}:
        try:
            if abs(float(new_value) - float(old_value)) <= max(tol * 10.0, 1e-8):
                return "floating_point_mismatch"
        except (TypeError, ValueError):
            pass
        return "position_sizing_mismatch"
    if field == "_row_count":
        return "missing_legacy_logic" if "trades" in file_name else "data_load_mismatch"
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
                "reason": classify_reason(file_name, "_row_count", len(new), len(old), tolerance),
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
            if field not in new.columns or field not in old.columns:
                rows.append(
                    {
                        "file": file_name,
                        "row": i,
                        "key": key,
                        "field": field,
                        "new_value": "<missing>" if field not in new.columns else new.iloc[i].get(field),
                        "old_value": "<missing>" if field not in old.columns else old.iloc[i].get(field),
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
                if _text_value(nv, is_time=is_time) != _text_value(ov, is_time=is_time):
                    rows.append(
                        {
                            "file": file_name,
                            "row": i,
                            "key": key,
                            "field": field,
                            "new_value": _text_value(nv, is_time=is_time),
                            "old_value": _text_value(ov, is_time=is_time),
                            "abs_diff": np.nan,
                            "tolerance": "exact",
                            "reason": classify_reason(file_name, field, nv, ov, tolerance),
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
                        "reason": classify_reason(file_name, field, nv, ov, tolerance),
                    }
                )
    return pd.DataFrame(rows)


def run_parity(
    *,
    new_report_dir: str | Path,
    old_report_dir: str | Path,
    primary_scenario: str,
    tolerance: float = 1e-9,
) -> ParityResult:
    new = load_report(new_report_dir, primary_scenario)
    old = load_report(old_report_dir, primary_scenario)

    summary_diff = _compare_frames("01_summary.csv", new["summary"], old["summary"], SUMMARY_FIELDS, tolerance=tolerance)
    trades_diff = _compare_frames("02_trades.csv", new["trades"], old["trades"], TRADES_FIELDS, tolerance=tolerance, key_field="trade_id")
    equity_diff = _compare_frames("03_equity.csv", new["equity"], old["equity"], EQUITY_FIELDS, tolerance=tolerance, key_field="timestamp")
    edge_diff = _compare_frames("04_edge_attribution.csv", new["edge"], old["edge"], EDGE_FIELDS, tolerance=tolerance, key_field="edge_id")

    report_rows = []
    for name, diff, new_df, old_df in [
        ("summary", summary_diff, new["summary"], old["summary"]),
        ("trades", trades_diff, new["trades"], old["trades"]),
        ("equity", equity_diff, new["equity"], old["equity"]),
        ("edge_attribution", edge_diff, new["edge"], old["edge"]),
    ]:
        report_rows.append(
            {
                "section": name,
                "passed": bool(diff.empty),
                "new_rows": int(len(new_df)),
                "old_rows": int(len(old_df)),
                "diff_count": int(len(diff)),
                "tolerance": tolerance,
            }
        )
    report = pd.DataFrame(report_rows)
    all_diffs = pd.concat([summary_diff, trades_diff, equity_diff, edge_diff], ignore_index=True, sort=False)
    first_diff = None if all_diffs.empty else all_diffs.iloc[0].to_dict()
    passed = bool(all_diffs.empty)

    out_dir = Path(new_report_dir)
    report.to_csv(out_dir / "08_parity_report.csv", index=False)
    if not trades_diff.empty:
        trades_diff.to_csv(out_dir / "08_parity_diff_trades.csv", index=False)
    if not equity_diff.empty:
        equity_diff.to_csv(out_dir / "08_parity_diff_equity.csv", index=False)
    return ParityResult(passed=passed, report=report, first_diff=first_diff, trades_diff=trades_diff, equity_diff=equity_diff)

