"""Import bridge for the numerically prefixed multi-speed EMA script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "31_multispeed_ema_trend.py"
_SPEC = importlib.util.spec_from_file_location("eth_multispeed_ema_trend", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_daily_ema_features = _MODULE.build_daily_ema_features
positions_from_features = _MODULE.positions_from_features
