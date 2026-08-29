from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


NEW_YORK_TIMEZONE_NAME = "America/New_York"
NORMAL_SESSION_START = time(7, 0)
NORMAL_SESSION_END = time(19, 0)

try:
    NEW_YORK_TIMEZONE = ZoneInfo(NEW_YORK_TIMEZONE_NAME)
except ZoneInfoNotFoundError as exc:  # pragma: no cover - depends on host installation
    raise RuntimeError(
        "America/New_York timezone data is unavailable; install the 'tzdata' package"
    ) from exc


def _aware_utc(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def new_york_time(instant: datetime) -> datetime:
    return _aware_utc(instant).astimezone(NEW_YORK_TIMEZONE)


def is_normal_polling_session(instant: datetime) -> bool:
    local = new_york_time(instant)
    local_clock = local.time().replace(tzinfo=None)
    return local.weekday() < 5 and NORMAL_SESSION_START <= local_clock < NORMAL_SESSION_END


def polling_mode(instant: datetime) -> str:
    local = new_york_time(instant)
    if local.weekday() >= 5:
        return "weekend"
    return "normal" if is_normal_polling_session(instant) else "off-hours"


def seconds_until_next_normal_session(instant: datetime) -> float:
    now_utc = _aware_utc(instant)
    local = now_utc.astimezone(NEW_YORK_TIMEZONE)
    candidate: date = local.date()
    local_clock = local.time().replace(tzinfo=None)
    if local.weekday() >= 5 or local_clock >= NORMAL_SESSION_START:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    opening = datetime.combine(candidate, NORMAL_SESSION_START, tzinfo=NEW_YORK_TIMEZONE)
    return max(0.0, (opening.astimezone(timezone.utc) - now_utc).total_seconds())


def poll_delay_seconds(
    normal_interval_seconds: float,
    off_hours_interval_seconds: float,
    weekend_interval_seconds: float | None = None,
    *,
    elapsed_seconds: float = 0.0,
    now: datetime | None = None,
) -> float:
    instant = _aware_utc(now or datetime.now(timezone.utc))
    mode = polling_mode(instant)
    weekend_interval = off_hours_interval_seconds if weekend_interval_seconds is None else weekend_interval_seconds
    interval = {
        "normal": normal_interval_seconds,
        "off-hours": off_hours_interval_seconds,
        "weekend": weekend_interval,
    }[mode]
    delay = max(0.1, interval - max(0.0, elapsed_seconds))
    if mode != "normal":
        until_open = seconds_until_next_normal_session(instant)
        if until_open > 0:
            delay = min(delay, max(0.1, until_open))
    return delay
