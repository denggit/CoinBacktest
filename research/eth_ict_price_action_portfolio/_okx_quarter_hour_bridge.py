"""Import bridge for quarter-hour OKX research."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "20_okx_quarter_hour_model.py"
_SPEC = importlib.util.spec_from_file_location("eth_okx_quarter_hour_model", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

QH_FEATURES = _MODULE.QH_FEATURES
build_samples = _MODULE.build_samples

