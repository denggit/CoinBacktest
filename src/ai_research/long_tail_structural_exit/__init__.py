"""R03.4.2.7 causal non-time structural exit research."""

from .config import DEFAULT_STRUCTURAL_EXIT_CONFIG, StructuralExitConfig
from .pipeline import StructuralExitResult, run_structural_exit_audit

__all__ = [
    "DEFAULT_STRUCTURAL_EXIT_CONFIG",
    "StructuralExitConfig",
    "StructuralExitResult",
    "run_structural_exit_audit",
]
