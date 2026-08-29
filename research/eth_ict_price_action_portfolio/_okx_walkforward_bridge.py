"""Import bridge for the numerically prefixed OKX walk-forward research."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "17_okx_walkforward_model.py"
_SPEC = importlib.util.spec_from_file_location("eth_okx_walkforward_model", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

END = _MODULE.END
FEATURE_COLUMNS = _MODULE.FEATURE_COLUMNS
ONE_WAY_COST = _MODULE.ONE_WAY_COST
START = _MODULE.START
build_hourly_features = _MODULE.build_hourly_features
core_state = _MODULE.core_state
load_inputs = _MODULE.load_inputs
metrics = _MODULE.metrics
simulate_minute = _MODULE.simulate_minute

