"""ETH external-strategy tournament research package."""

from .config import TournamentConfig
from .runner import run_tournament

__all__ = ["TournamentConfig", "run_tournament"]
