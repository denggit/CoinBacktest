#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import zipfile

from src.research_common.review_pack import finalize_research_report


def test_finalize_research_report_writes_review_pack(tmp_path):
    report_dir = tmp_path / "ETH_RESEARCH_TEST_V1"
    report_dir.mkdir()
    (report_dir / "00_manifest.json").write_text(
        json.dumps(
            {
                "edge_id": "ETH_EDGE_TEST",
                "experiment_id": "ETH_RESEARCH_TEST_V1",
                "title": "Test edge",
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "01_summary.csv").write_text("metric,value\ntrades,10\n", encoding="utf-8")

    result = finalize_research_report(report_dir, print_log=False)

    assert result.zip_path.exists()
    assert result.prompt_path.exists()
    assert result.manifest_path.exists()
    with zipfile.ZipFile(result.zip_path) as zf:
        names = set(zf.namelist())
    assert "GPT_REVIEW_PROMPT.md" in names
    assert "REVIEW_PACK_MANIFEST.json" in names
    assert "00_manifest.json" in names
    assert "01_summary.csv" in names
