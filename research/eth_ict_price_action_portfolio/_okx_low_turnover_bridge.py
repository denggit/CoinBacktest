"""Import bridge for the numerically prefixed low-turnover research."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "18_okx_low_turnover_model.py"
_SPEC = importlib.util.spec_from_file_location("eth_okx_low_turnover_model", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

RESULTS = _MODULE.RESULTS
RETURN_GATE = _MODULE.RETURN_GATE
TACTICAL_SIZE = _MODULE.TACTICAL_SIZE
align_events = _MODULE.align_events
build_samples = _MODULE.build_samples
model_events = _MODULE.model_events

