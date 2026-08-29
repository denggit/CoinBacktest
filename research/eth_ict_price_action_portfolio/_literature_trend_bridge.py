"""Import bridge for the numerically prefixed literature trend script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "12_literature_trend_core.py"
_SPEC = importlib.util.spec_from_file_location("eth_literature_trend", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

daily_bos_positions = _MODULE.daily_bos_positions
monthly_tsmom_positions = _MODULE.monthly_tsmom_positions
