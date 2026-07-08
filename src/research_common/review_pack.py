#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPT review-pack helpers for CoinBacktest research reports."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INCLUDE_SUFFIXES = {".csv", ".json", ".md", ".txt", ".png", ".html"}
DEFAULT_EXCLUDE_NAMES = {
    "gpt_review_pack.zip",
    "GPT_REVIEW_PROMPT.md",
    "REVIEW_PACK_MANIFEST.json",
}
DEFAULT_META_CANDIDATES = (
    "00_manifest.json",
    "10_meta.json",
    "99_research_meta.json",
    "12_lab_meta.json",
    "08_lab_meta.json",
    "00_config.json",
)


@dataclass(frozen=True)
class ReviewPackResult:
    """Result returned after writing a GPT review pack."""

    zip_path: Path
    prompt_path: Path
    manifest_path: Path
    included_files: tuple[str, ...]
    skipped_files: tuple[str, ...]


@dataclass(frozen=True)
class ReviewPackConfig:
    """Configuration for a GPT review pack."""

    report_dir: Path
    experiment_id: str | None = None
    edge_id: str | None = None
    stage: str = "research"
    title: str | None = None
    decision_focus: str = "edge_review"
    zip_name: str = "gpt_review_pack.zip"
    max_file_bytes: int = 8 * 1024 * 1024
    max_total_bytes: int = 40 * 1024 * 1024
    include_suffixes: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_INCLUDE_SUFFIXES))
    exclude_names: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_EXCLUDE_NAMES))
    print_log: bool = True


def finalize_research_report(
    report_dir: str | Path,
    *,
    experiment_id: str | None = None,
    edge_id: str | None = None,
    title: str | None = None,
    print_log: bool = True,
) -> ReviewPackResult:
    """Finalize a research report by writing a GPT review pack.

    New research scripts should call this once after all CSV/JSON artifacts have
    been written. The helper only packages completed outputs; it never changes
    research data or strategy timing.
    """

    return write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=Path(report_dir),
            experiment_id=experiment_id,
            edge_id=edge_id,
            stage="research",
            title=title,
            print_log=print_log,
        )
    )


def write_gpt_review_pack(config: ReviewPackConfig) -> ReviewPackResult:
    """Write GPT_REVIEW_PROMPT.md, REVIEW_PACK_MANIFEST.json, and a zip pack."""

    report_dir = Path(config.report_dir)
    if not report_dir.exists():
        raise FileNotFoundError(f"report_dir does not exist: {report_dir}")
    if not report_dir.is_dir():
        raise NotADirectoryError(f"report_dir is not a directory: {report_dir}")

    metadata = _load_report_metadata(report_dir)
    experiment_id = config.experiment_id or _first_string(metadata, "experiment_id", "experiment", "run_id")
    edge_id = config.edge_id or _first_string(metadata, "edge_id", "edge")
    title = config.title or _first_string(metadata, "title", "name", "strategy_name") or report_dir.name

    prompt = build_gpt_review_prompt(
        report_dir=report_dir,
        experiment_id=experiment_id,
        edge_id=edge_id,
        title=title,
        stage=config.stage,
        decision_focus=config.decision_focus,
    )
    prompt_path = report_dir / "GPT_REVIEW_PROMPT.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    candidates, skipped = _collect_report_files(report_dir, config)
    manifest = {
        "pack_type": "gpt_review_pack",
        "stage": config.stage,
        "decision_focus": config.decision_focus,
        "report_dir": str(report_dir),
        "experiment_id": experiment_id,
        "edge_id": edge_id,
        "title": title,
        "included_files": [p.relative_to(report_dir).as_posix() for p in candidates],
        "skipped_files": skipped,
        "notes": [
            "Upload this zip to GPT for edge review.",
            "GPT should judge rejected / research_continue / promote_to_backtest.",
            "Skipped files are usually too large for review and should be inspected locally if needed.",
        ],
    }
    manifest_path = report_dir / "REVIEW_PACK_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    zip_path = report_dir / config.zip_name
    files_to_zip = _dedupe_paths([prompt_path, manifest_path, *candidates])
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files_to_zip:
            zf.write(path, path.relative_to(report_dir).as_posix())

    result = ReviewPackResult(
        zip_path=zip_path,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        included_files=tuple(p.relative_to(report_dir).as_posix() for p in candidates),
        skipped_files=tuple(skipped),
    )
    if config.print_log:
        print(f"[review-pack] wrote {zip_path}", flush=True)
        if skipped:
            print(f"[review-pack] skipped {len(skipped)} large/unsupported files; see REVIEW_PACK_MANIFEST.json", flush=True)
    return result


def build_gpt_review_prompt(
    *,
    report_dir: str | Path,
    experiment_id: str | None,
    edge_id: str | None,
    title: str,
    stage: str,
    decision_focus: str,
) -> str:
    """Build the prompt bundled into each GPT review pack."""

    exp = experiment_id or "<unknown_experiment_id>"
    edge = edge_id or "<unknown_edge_id>"
    return f"""# GPT Review Prompt

You are reviewing a CoinBacktest ETH perpetual {stage} report.

## IDs
- Edge ID: `{edge}`
- Experiment ID: `{exp}`
- Title: `{title}`
- Report dir: `{Path(report_dir).as_posix()}`
- Decision focus: `{decision_focus}`

## Task
Please judge whether this research has a real, tradable edge for ETH perpetuals.

Focus on:
1. Whether the forward returns or trade results show stable positive expectancy.
2. Whether the edge survives costs, slippage, delay, and reasonable parameter changes if those files are present.
3. Whether performance is concentrated in one year/regime or looks robust.
4. Whether sample size is enough.
5. Whether there may be lookahead bias, timestamp mistakes, duplicated events, or other causal issues.
6. Whether the idea should be rejected, upgraded with another research pass, or promoted to backtest.

## Required Decision
Return exactly one primary decision:

- `rejected`: no usable edge, or too weak after costs.
- `research_continue`: some signal exists, but it needs another research version before backtest.
- `promote_to_backtest`: research evidence is strong enough to build a backtest candidate.

Also include:
- Key evidence from the files.
- Main risks or failure modes.
- Concrete next experiments if decision is `research_continue`.
- Backtest design requirements if decision is `promote_to_backtest`.
"""


def _load_report_metadata(report_dir: Path) -> dict[str, Any]:
    for name in DEFAULT_META_CANDIDATES:
        path = report_dir / name
        if not path.exists() or path.stat().st_size <= 0:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _first_string(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _collect_report_files(report_dir: Path, config: ReviewPackConfig) -> tuple[list[Path], list[str]]:
    candidates: list[Path] = []
    skipped: list[str] = []
    total_bytes = 0
    for path in sorted(report_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(report_dir).as_posix()
        if "__pycache__" in path.parts:
            continue
        if path.name in config.exclude_names:
            continue
        if path.suffix.lower() not in config.include_suffixes:
            skipped.append(f"{rel} unsupported_suffix")
            continue
        size = path.stat().st_size
        if size > config.max_file_bytes:
            skipped.append(f"{rel} too_large_bytes={size}")
            continue
        if total_bytes + size > config.max_total_bytes:
            skipped.append(f"{rel} total_limit_bytes={size}")
            continue
        candidates.append(path)
        total_bytes += size
    return candidates, skipped


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out
