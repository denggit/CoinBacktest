#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic R02 sleeve-registry artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

from .registry import SLEEVE_SPECS


def write_sleeve_framework_artifacts(report_dir: str | Path) -> dict[str, Path]:
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    registry_path = target / "00_sleeve_registry.json"
    summary_path = target / "01_framework_summary.md"
    payload = {
        "schema_version": 1,
        "sleeves": [spec.to_dict() for spec in SLEEVE_SPECS.values()],
        "shared_contract": {
            "research_output": "TradeCandidate",
            "portfolio_output": "TargetPositionDecision",
            "direct_exchange_orders": False,
            "independent_labels": True,
            "independent_exit_policies": True,
        },
    }
    registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# R02 ETH AI 三-Sleeve框架",
        "",
        "三个Sleeve共享数据接口和输出合同，但标签、训练、退出与验收相互独立。任何研究只能输出候选和目标仓位意图，不能直接调用交易所。",
        "",
        "| Sleeve | 决策频率 | 目标持仓 | 目标波幅 | 方向周期 | 入场周期 | 退出 |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for spec in SLEEVE_SPECS.values():
        hold_text = (
            f"无最低持仓，最长{spec.intended_hold_max_minutes}分钟"
            if spec.intended_hold_min_minutes == 0
            else f"{spec.intended_hold_min_minutes}–{spec.intended_hold_max_minutes}分钟"
        )
        lines.append(
            f"| {spec.sleeve_id} | {spec.decision_cadence} | "
            f"{hold_text} | "
            f"{spec.target_move_min:.1%}–{spec.target_move_max:.1%} | "
            f"{', '.join(spec.direction_timeframes)} | {', '.join(spec.entry_timeframes)} | {spec.exit_style} |"
        )
    lines.extend(
        [
            "",
            "## 强制边界",
            "",
            "1. Swing和Intraday的最长持仓只作安全上限，禁止作为主退出逻辑。",
            "2. Short-horizon可以有时间上限，但仍必须结合止损、MFE保护和模型/订单流失效。",
            "3. 三个Sleeve不能各自直接下单，最终统一为一个ETH目标净仓位。",
            "4. CoinBacktest负责研究；AetherEdge以后只加载版本化模型包并执行标准信号。",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"registry": registry_path, "summary": summary_path}
