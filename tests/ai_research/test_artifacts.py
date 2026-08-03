from __future__ import annotations

import csv
import json
from pathlib import Path

from src.ai_research.artifacts import initial_stage_status, write_framework_artifacts
from src.ai_research.plan import DEFAULT_RESEARCH_PLAN


def test_initial_status_unblocks_only_r00() -> None:
    rows = initial_stage_status(DEFAULT_RESEARCH_PLAN)
    assert rows[0]["stage_id"] == "R00"
    assert rows[0]["status"] == "READY"
    assert all(row["status"] == "BLOCKED" for row in rows[1:])


def test_framework_artifacts_are_written_and_status_is_preserved(tmp_path: Path) -> None:
    paths = write_framework_artifacts(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    assert plan["plan_id"] == "ETH_AI_TRADING"
    assert len(plan["stages"]) == 15

    with paths["status"].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "READY"

    custom = paths["status"].read_text(encoding="utf-8-sig").replace("READY", "PASSED", 1)
    paths["status"].write_text(custom, encoding="utf-8-sig")
    write_framework_artifacts(tmp_path)
    assert "PASSED" in paths["status"].read_text(encoding="utf-8-sig")


def test_explicit_status_reset_is_supported(tmp_path: Path) -> None:
    paths = write_framework_artifacts(tmp_path)
    paths["status"].write_text("custom\n", encoding="utf-8")
    write_framework_artifacts(tmp_path, overwrite_status=True)
    assert "stage_id" in paths["status"].read_text(encoding="utf-8-sig")
