"""R03.4.2.8B account-level dual risk-slot research."""

from .config import DEFAULT_TRANCHE_ACCOUNT_CONFIG, TrancheAccountConfig, TranchePolicy
from .pipeline import TrancheAccountResult, run_tranche_account_audit

__all__ = [
    "DEFAULT_TRANCHE_ACCOUNT_CONFIG",
    "TrancheAccountConfig",
    "TranchePolicy",
    "TrancheAccountResult",
    "run_tranche_account_audit",
]
