from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]


@dataclass(frozen=True)
class SignalThresholds:
    fedwatch_15m_pct: float = 5.0
    fedwatch_60m_pct: float = 10.0
    expected_rate_15m_bp: float = 2.0
    expected_rate_60m_bp: float = 4.0
    us2y_5m_bp: float = 3.0
    us2y_15m_bp: float = 5.0
    us2y_60m_bp: float = 8.0
    us10y_5m_bp: float = 3.0
    us10y_15m_bp: float = 5.0
    us10y_60m_bp: float = 8.0
    dxy_5m_pct: float = 0.15
    dxy_15m_pct: float = 0.25
    dxy_60m_pct: float = 0.40


@dataclass(frozen=True)
class ResearchConfig:
    macro_db: Path = REPOSITORY_ROOT / "data" / "macro_monitor" / "macro_monitor.sqlite"
    fred_yields_csv: Path = REPOSITORY_ROOT / "data" / "macro_free" / "raw" / "fred_dgs2_dgs10_daily.csv"
    existing_market_db: Path = REPOSITORY_ROOT / "data" / "crypto_history.db"
    research_data_dir: Path = PACKAGE_DIR / "data"
    raw_dir: Path = PACKAGE_DIR / "data" / "raw"
    output_dir: Path = PACKAGE_DIR / "outputs"
    event_cooldown_minutes: int = 30
    bootstrap_samples: int = 2_000
    random_seed: int = 20260829
    intraday_horizons_minutes: tuple[int, ...] = (5, 15, 30, 60)
    daily_horizons_sessions: tuple[int, ...] = (1, 3, 5)
    scheduled_signal_delays_minutes: tuple[int, ...] = (5, 10, 15)
    execution_delays_minutes: tuple[int, ...] = (0, 5, 10)
    scheduled_horizons_minutes: tuple[int, ...] = (5, 15, 30, 60)
    intraday_history_range: str = "60d"
    intraday_history_interval: str = "5m"
    equity_symbols: tuple[str, ...] = ("SOXX", "SOXL", "QQQ")
    thresholds: SignalThresholds = field(default_factory=SignalThresholds)

    def ensure_directories(self) -> None:
        self.research_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
