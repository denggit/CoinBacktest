#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Static import-boundary checks for CoinBacktest entry scripts.

Rules for new code:
- ``src`` modules must not import ``research``, ``backtest`` or ``tools``.
- ``research`` scripts must not import ``research`` or ``backtest`` scripts.
- ``backtest`` scripts must not import ``research`` scripts.
- ``backtest`` scripts should not import other ``backtest`` scripts unless the
  current legacy allowlist says that import already existed.

The allowlist freezes known legacy coupling so new coupling fails tests without
requiring a large one-shot refactor.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ALLOWLIST = Path("config") / "import_boundary_legacy_allowlist.json"


@dataclass(frozen=True)
class ImportViolation:
    file: str
    module: str
    reason: str

    @property
    def key(self) -> str:
        return f"{self.file}|{self.module}|{self.reason}"

    def to_dict(self) -> dict[str, str]:
        return {"file": self.file, "module": self.module, "reason": self.reason}


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts or ".pytest_cache" in parts:
            continue
        yield path


def _imported_modules(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                yield node.module


def _classify(file: str, module: str) -> str | None:
    if file.startswith("src/") and module.split(".", 1)[0] in {"research", "backtest", "tools"}:
        return "src_imports_entrypoint_layer"
    if file.startswith("research/") and module.startswith("research."):
        return "research_imports_research"
    if file.startswith("research/") and module.startswith("backtest."):
        return "research_imports_backtest"
    if file.startswith("backtest/") and module.startswith("research."):
        return "backtest_imports_research"
    if file.startswith("backtest/") and module.startswith("backtest."):
        return "backtest_imports_backtest"
    return None


def scan_import_boundaries(root: str | Path) -> list[ImportViolation]:
    repo_root = Path(root)
    violations: list[ImportViolation] = []
    for path in _iter_python_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            violations.append(
                ImportViolation(
                    file=rel,
                    module="<parse-error>",
                    reason=f"syntax_error:{exc.lineno}",
                )
            )
            continue
        for module in _imported_modules(tree):
            reason = _classify(rel, module)
            if reason:
                violations.append(
                    ImportViolation(file=rel, module=module, reason=reason)
                )
    return sorted(violations, key=lambda item: item.key)


def load_allowlist(path: str | Path = DEFAULT_ALLOWLIST) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    data = json.loads(target.read_text(encoding="utf-8"))
    rows = data.get("allowed_legacy_imports", [])
    return {
        f"{row['file']}|{row['module']}|{row['reason']}"
        for row in rows
    }


def unexpected_violations(
    root: str | Path,
    allowlist_path: str | Path = DEFAULT_ALLOWLIST,
) -> list[ImportViolation]:
    allowed = load_allowlist(allowlist_path)
    return [
        violation
        for violation in scan_import_boundaries(root)
        if violation.key not in allowed
    ]
