"""Import bridge for the numerically prefixed daily reclaim script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "29_daily_liquidity_reclaim.py"
_SPEC = importlib.util.spec_from_file_location("eth_daily_liquidity_reclaim", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_daily_reclaim_events = _MODULE.build_daily_reclaim_events
build_side_positions = _MODULE.build_side_positions
