from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without importing config/env_loader.py."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Thresholds:
    fedwatch_15m_pct: float = 5.0
    fedwatch_60m_pct: float = 10.0
    us2y_5m_bp: float = 3.0
    us2y_15m_bp: float = 5.0
    us2y_60m_bp: float = 8.0
    us10y_5m_bp: float = 3.0
    us10y_15m_bp: float = 5.0
    us10y_60m_bp: float = 8.0
    dxy_5m_pct: float = 0.15
    dxy_15m_pct: float = 0.25
    dxy_60m_pct: float = 0.40
    curve_15m_bp: float = 2.0
    curve_60m_bp: float = 4.0

    @classmethod
    def from_env(cls) -> "Thresholds":
        return cls(
            fedwatch_15m_pct=_env_float("FEDWATCH_15M_ALERT_PCT", 5.0),
            fedwatch_60m_pct=_env_float("FEDWATCH_60M_ALERT_PCT", 10.0),
            us2y_5m_bp=_env_float("US2Y_5M_ALERT_BP", 3.0),
            us2y_15m_bp=_env_float("US2Y_15M_ALERT_BP", 5.0),
            us2y_60m_bp=_env_float("US2Y_60M_ALERT_BP", 8.0),
            us10y_5m_bp=_env_float("US10Y_5M_ALERT_BP", 3.0),
            us10y_15m_bp=_env_float("US10Y_15M_ALERT_BP", 5.0),
            us10y_60m_bp=_env_float("US10Y_60M_ALERT_BP", 8.0),
            dxy_5m_pct=_env_float("DXY_5M_ALERT_PCT", 0.15),
            dxy_15m_pct=_env_float("DXY_15M_ALERT_PCT", 0.25),
            dxy_60m_pct=_env_float("DXY_60M_ALERT_PCT", 0.40),
            curve_15m_bp=_env_float("CURVE_15M_ALERT_BP", 2.0),
            curve_60m_bp=_env_float("CURVE_60M_ALERT_BP", 4.0),
        )


@dataclass(frozen=True)
class EmailConfig:
    enabled: bool
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    sender: str
    recipients: tuple[str, ...]
    use_ssl: bool
    cooldown_seconds: int
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls, cli_enabled: bool = False) -> "EmailConfig":
        sender = os.getenv("EMAIL_FROM") or os.getenv("EMAIL_SENDER", "")
        username = os.getenv("EMAIL_USERNAME") or sender
        receiver_text = os.getenv("EMAIL_TO") or os.getenv("EMAIL_RECEIVER", "")
        recipients = tuple(x.strip() for x in receiver_text.replace(";", ",").split(",") if x.strip())
        host = os.getenv("EMAIL_SMTP_HOST", "").strip()
        inferred_host, inferred_port, inferred_ssl = infer_smtp(sender)
        host = host or inferred_host
        port = _env_int("EMAIL_SMTP_PORT", inferred_port)
        use_ssl = _parse_bool(os.getenv("EMAIL_SMTP_SSL"), inferred_ssl)
        enabled = cli_enabled or _parse_bool(os.getenv("EMAIL_ENABLED"), False)
        return cls(
            enabled=enabled,
            smtp_host=host,
            smtp_port=port,
            username=username,
            password=os.getenv("EMAIL_PASSWORD", ""),
            sender=sender,
            recipients=recipients,
            use_ssl=use_ssl,
            cooldown_seconds=_env_int("EMAIL_COOLDOWN_SECONDS", 1800),
            timeout_seconds=_env_float("EMAIL_TIMEOUT_SECONDS", 10.0),
        )

    @property
    def configured(self) -> bool:
        return bool(self.smtp_host and self.sender and self.recipients and self.password)


def infer_smtp(sender: str) -> tuple[str, int, bool]:
    domain = sender.rsplit("@", 1)[-1].lower() if "@" in sender else ""
    known = {
        "qq.com": ("smtp.qq.com", 465, True),
        "163.com": ("smtp.163.com", 465, True),
        "126.com": ("smtp.126.com", 465, True),
        "gmail.com": ("smtp.gmail.com", 465, True),
        "outlook.com": ("smtp.office365.com", 587, False),
        "hotmail.com": ("smtp.office365.com", 587, False),
    }
    return known.get(domain, ("", 465, True))


@dataclass(frozen=True)
class MonitorConfig:
    db_path: Path
    fedwatch_poll_seconds: float
    treasury_poll_seconds: float
    dxy_poll_seconds: float
    off_hours_poll_seconds: float
    weekend_poll_seconds: float
    headed: bool
    verbose: bool
    retention_days: int
    source_retries: int
    thresholds: Thresholds
    email: EmailConfig

    @classmethod
    def from_env(cls, root: Path, *, headed: bool = False, verbose: bool = False, email: bool = False) -> "MonitorConfig":
        db_raw = os.getenv("MACRO_MONITOR_DB", "data/macro_monitor/macro_monitor.sqlite")
        db_path = Path(db_raw)
        if not db_path.is_absolute():
            db_path = root / db_path
        return cls(
            db_path=db_path,
            fedwatch_poll_seconds=_env_float("FEDWATCH_POLL_SECONDS", 60.0),
            treasury_poll_seconds=_env_float("TREASURY_POLL_SECONDS", 15.0),
            dxy_poll_seconds=_env_float("DXY_POLL_SECONDS", 15.0),
            off_hours_poll_seconds=_env_float("MACRO_OFF_HOURS_POLL_SECONDS", 300.0),
            weekend_poll_seconds=_env_float("MACRO_WEEKEND_POLL_SECONDS", 3600.0),
            headed=headed or _parse_bool(os.getenv("MACRO_MONITOR_HEADED"), False),
            verbose=verbose,
            retention_days=_env_int("MACRO_MONITOR_RETENTION_DAYS", 365),
            source_retries=max(1, _env_int("MACRO_SOURCE_RETRIES", 3)),
            thresholds=Thresholds.from_env(),
            email=EmailConfig.from_env(cli_enabled=email),
        )

    def with_overrides(self, **changes: object) -> "MonitorConfig":
        return replace(self, **changes)
