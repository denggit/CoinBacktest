"""Import bridge for the numerically prefixed NR7 script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "30_nr7_breakout_model.py"
_SPEC = importlib.util.spec_from_file_location("eth_nr7_breakout_model", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_nr7_setups = _MODULE.build_nr7_setups
build_breakout_events = _MODULE.build_breakout_events
