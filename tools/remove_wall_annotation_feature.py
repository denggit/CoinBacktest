#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove obsolete Liquidity Wall Human Annotation V1 files.

The research patch already restores Analyze Tool server/static files to the
pre-annotation version. This cleanup only deletes orphan implementation/test/doc
files left behind by earlier zip overlays. Existing user annotation data under
``data/analyze_tool/wall_annotations`` is intentionally preserved.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBSOLETE = (
    "analyze_tool/static/wall_annotation.js",
    "analyze_tool/wall_annotation_store.py",
    "tests/analyze_tool/test_wall_annotation_store.py",
    "docs/LIQUIDITY_WALL_ANNOTATION_V1.md",
    "docs/LIQUIDITY_WALL_ANNOTATION_V1_0_1_DRAG_FIX.md",
    "PATCH_MANIFEST_WALL_ANNOTATION_V1.md",
    "PATCH_MANIFEST_WALL_ANNOTATION_V1_0_1.md",
)


def main() -> int:
    deleted = 0
    for relative in OBSOLETE:
        path = PROJECT_ROOT / relative
        if path.exists():
            path.unlink()
            deleted += 1
            print(f"[deleted] {relative}")
        else:
            print(f"[skip] {relative}")
    print(f"[done] removed={deleted}; preserved=data/analyze_tool/wall_annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
