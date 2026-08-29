"""Import bridge for the numerically prefixed 10s pinned-flow script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "27_pinned_10s_release_model.py"
_SPEC = importlib.util.spec_from_file_location("eth_pinned_10s_release_model", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_release_events = _MODULE.build_release_events
build_side_positions = _MODULE.build_side_positions
filter_pinned_candidates = _MODULE.filter_pinned_candidates
