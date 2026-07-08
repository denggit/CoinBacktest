#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standard Portfolio V1 artifact writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.portfolio_common.allocator import PORTFOLIO_ID
from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack


def write_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {label} -> {path}", flush=True)


def write_json(data: dict[str, Any], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[write] {label} -> {path}", flush=True)


def write_standard_artifacts(
    out_dir: str | Path,
    *,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    edge_attribution: pd.DataFrame,
    daily_returns: pd.DataFrame,
    stress: pd.DataFrame | None,
) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = [
        "00_manifest.json",
        "01_summary.csv",
        "02_trades.csv",
        "03_equity.csv",
        "04_edge_attribution.csv",
        "05_daily_returns.csv",
        "09_decision_draft.json",
    ]
    if stress is not None:
        artifacts.insert(6, "06_stress.csv")
    manifest = dict(manifest)
    manifest["artifacts"] = artifacts
    write_json(manifest, out / "00_manifest.json", "manifest")
    write_csv(summary, out / "01_summary.csv", "summary")
    write_csv(trades, out / "02_trades.csv", "trades")
    write_csv(equity, out / "03_equity.csv", "equity")
    write_csv(edge_attribution, out / "04_edge_attribution.csv", "edge_attribution")
    write_csv(daily_returns, out / "05_daily_returns.csv", "daily_returns")
    if stress is not None:
        write_csv(stress, out / "06_stress.csv", "stress")
    decision = {
        "portfolio_id": PORTFOLIO_ID,
        "decision": "review_required",
        "candidate": "ETH Portfolio V1 refactor parity validation",
        "notes": [
            "Strategy logic is intended to match the legacy Portfolio V1 source of truth.",
            "Do not promote changes unless parity and import-boundary checks are reviewed.",
        ],
    }
    write_json(decision, out / "09_decision_draft.json", "decision_draft")
    return artifacts


def finalize_review_pack(out_dir: str | Path) -> None:
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=Path(out_dir),
            experiment_id="ETH_PORTFOLIO_V1",
            edge_id="ETH_PORTFOLIO_V1",
            stage="portfolio_backtest",
            title="ETH Portfolio V1",
            decision_focus="portfolio_parity_review",
            print_log=True,
        )
    )

