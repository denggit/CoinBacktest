from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def write_review_pack(root: Path) -> Path:
    pack = root / "gpt_review_pack.zip"
    with ZipFile(pack, "w", compression=ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p != pack:
                z.write(p, p.relative_to(root.parent))
    return pack


def write_decision(root: Path, episodes: pd.DataFrame, stats: pd.DataFrame, checkpoint_stats: pd.DataFrame) -> None:
    discovery = episodes[episodes["split"] == "DISCOVERY_2023_2024"] if not episodes.empty else pd.DataFrame()
    validation = episodes[episodes["split"] == "VALIDATION_2025"] if not episodes.empty else pd.DataFrame()
    lines = [
        "# R04 Turtle Path Atlas — Decision",
        "",
        "R04 does **not** change the source-locked Turtle entry/exit rules. It reconstructs the minute-by-minute path of every R03 Turtle episode to learn position-management behavior.",
        "",
        f"- Total episodes: **{len(episodes)}**.",
        f"- Discovery episodes (2023-2024): **{len(discovery)}**.",
        f"- Validation episodes (2025): **{len(validation)}**.",
        "- 2026 sealed holdout opened: **NO**.",
        "- Path labels are retrospective diagnostics only; they are forbidden as live features until converted into causal rules and validated on a later period.",
        "",
        "## What to inspect",
        "",
        "1. Whether profitable episodes prove themselves by reaching Unit 2/3/4 faster than failed episodes.",
        "2. Whether losses cluster in MAX_UNIT_1 (no follow-through) or PYRAMID_THEN_FAIL episodes.",
        "3. How much favorable excursion is typically given back before the 20D exit.",
        "4. Whether an early-path distinction seen in 2023-2024 repeats in 2025 before any position overlay is proposed.",
        "",
        "## Gate for R05",
        "",
        "Only a simple causal position-management rule whose qualitative pattern appears in discovery **and** validation may advance. Do not tune thresholds on 2025. Do not alter 55D entry, 20D exit, N, or 0.5N add spacing in R04.",
    ]
    (root / "99_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
