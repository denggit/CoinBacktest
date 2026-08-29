"""Import bridge for the numerically prefixed multi-scale BOS script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "28_multiscale_bos_portfolio.py"
_SPEC = importlib.util.spec_from_file_location("eth_multiscale_bos_portfolio", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_daily_bos_features = _MODULE.build_daily_bos_features
positions_from_features = _MODULE.positions_from_features
