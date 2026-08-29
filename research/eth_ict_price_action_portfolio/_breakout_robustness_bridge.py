"""Import bridge for the numerically prefixed breakout robustness script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "07_breakout_robustness.py"
_SPEC = importlib.util.spec_from_file_location("eth_breakout_robustness", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

aggregate_four_hour = _MODULE.aggregate_four_hour
definition = _MODULE.definition
align = _MODULE.align
bootstrap_daily = _MODULE.bootstrap_daily
