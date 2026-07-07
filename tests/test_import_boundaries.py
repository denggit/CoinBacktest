from __future__ import annotations

from pathlib import Path

from src.architecture.import_boundaries import unexpected_violations


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_new_research_backtest_import_coupling() -> None:
    unexpected = unexpected_violations(
        REPO_ROOT,
        REPO_ROOT / "config" / "import_boundary_legacy_allowlist.json",
    )
    assert not unexpected, [
        f"{item.file}: {item.module} ({item.reason})"
        for item in unexpected
    ]
