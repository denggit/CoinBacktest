#!/usr/bin/env python
"""Discover the actual local pre-embargo source catalog before assigning R27."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_feed.local_market_catalog import catalog_local_market_data  # noqa: E402
from src.research_common.ict_mss2.source_readiness import (  # noqa: E402
    SourceReadinessConfig,
    assert_pre_embargo_catalog,
    build_archive_inventory,
    build_mechanism_readiness_gate,
    build_sqlite_inventory,
    r27_gate_decision,
    source_readiness_markdown,
)


DEFAULT_REPORT_DIR = (
    _PROJECT_ROOT / "data" / "reports" / "research" / "ict" / "mss2" / "pre_r27_source_readiness_audit"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SourceReadinessConfig()
    catalog = catalog_local_market_data(
        args.data_dir,
        window_start=config.warmup_start,
        end_exclusive=config.validation_end_exclusive,
    )
    seal_checks = assert_pre_embargo_catalog(
        catalog, end_exclusive=config.validation_end_exclusive
    )
    sqlite_inventory = build_sqlite_inventory(catalog, config=config)
    sqlite_metadata = catalog.metadata_frame()
    archive_inventory = build_archive_inventory(catalog)
    mechanism_gate = build_mechanism_readiness_gate(catalog, config=config)
    decision = r27_gate_decision(mechanism_gate)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    sqlite_inventory.to_csv(args.report_dir / "01_sqlite_series_catalog.csv", index=False)
    sqlite_metadata.to_csv(args.report_dir / "02_sqlite_table_metadata.csv", index=False)
    archive_inventory.to_csv(args.report_dir / "03_archive_series_catalog.csv", index=False)
    mechanism_gate.to_csv(args.report_dir / "04_mechanism_readiness_gate.csv", index=False)
    seal_checks.to_csv(args.report_dir / "05_pre_embargo_seal_checks.csv", index=False)

    report_text = source_readiness_markdown(
        sqlite_inventory, archive_inventory, mechanism_gate, seal_checks
    )
    (args.report_dir / "SOURCE_READINESS_AUDIT.md").write_text(report_text, encoding="utf-8")
    manifest = {
        "study": "pre_r27_source_readiness_audit",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "warmup_start": config.warmup_start.isoformat(),
        "discovery_start": config.discovery_start.isoformat(),
        "physical_end_exclusive": config.validation_end_exclusive.isoformat(),
        "sqlite_series": int(len(sqlite_inventory)),
        "sqlite_metadata_tables": int(len(sqlite_metadata)),
        "archive_series": int(len(archive_inventory)),
        "eligible_mechanisms": int(
            mechanism_gate["decision"].eq("ELIGIBLE_FOR_PRECOMMITMENT").sum()
        ),
        "decision": decision,
        "r27_assigned": decision == "R27_PRECOMMITMENT_ALLOWED",
        "july_or_holdout_outcomes_loaded": 0,
    }
    (args.report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[catalog] sqlite_series={len(sqlite_inventory)} archive_series={len(archive_inventory)}")
    print(f"[seal] checks={len(seal_checks)} passed={int(seal_checks['passed'].sum())}")
    print(f"[decision] {decision}")
    print(f"[report] {args.report_dir / 'SOURCE_READINESS_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

