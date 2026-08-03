"""R03.4.2.5 high-confidence persistent-failure exit overlay."""

from .config import DEFAULT_FAILURE_OVERLAY_CONFIG, FailureOverlayConfig
from .pipeline import FailureOverlayResult, run_failure_overlay_research

__all__ = [
    "DEFAULT_FAILURE_OVERLAY_CONFIG",
    "FailureOverlayConfig",
    "FailureOverlayResult",
    "run_failure_overlay_research",
]
