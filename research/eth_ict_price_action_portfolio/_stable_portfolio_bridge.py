"""Import bridge for the numerically prefixed stable-portfolio research script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "06_stable_portfolio_research.py"
_SPEC = importlib.util.spec_from_file_location("eth_stable_portfolio_research", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_daily_core = _MODULE.build_daily_core
hold_events = _MODULE._hold_events
load_inputs = _MODULE.load_inputs
metrics = _MODULE.metrics
simulate = _MODULE.simulate
