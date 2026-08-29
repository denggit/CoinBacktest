from __future__ import annotations

from datetime import datetime, timezone

import pytest

from macro_monitor.schedule import (
    is_normal_polling_session,
    new_york_time,
    poll_delay_seconds,
    polling_mode,
    seconds_until_next_normal_session,
)


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_new_york_winter_session_boundaries() -> None:
    assert not is_normal_polling_session(utc(2026, 1, 5, 11, 59))  # 06:59 EST
    assert is_normal_polling_session(utc(2026, 1, 5, 12, 0))       # 07:00 EST
    assert is_normal_polling_session(utc(2026, 1, 5, 23, 59))      # 18:59 EST
    assert not is_normal_polling_session(utc(2026, 1, 6, 0, 0))    # 19:00 EST


def test_new_york_summer_session_uses_daylight_saving_time() -> None:
    assert not is_normal_polling_session(utc(2026, 7, 6, 10, 59))  # 06:59 EDT
    assert is_normal_polling_session(utc(2026, 7, 6, 11, 0))       # 07:00 EDT
    assert is_normal_polling_session(utc(2026, 7, 6, 22, 59))      # 18:59 EDT
    assert not is_normal_polling_session(utc(2026, 7, 6, 23, 0))   # 19:00 EDT


def test_weekends_are_always_off_hours() -> None:
    saturday = utc(2026, 8, 29, 16, 0)
    assert new_york_time(saturday).weekday() == 5
    assert polling_mode(saturday) == "weekend"


def test_poll_delay_uses_normal_or_five_minute_weekday_frequency() -> None:
    normal = utc(2026, 8, 28, 16, 0)       # Friday 12:00 EDT
    after_hours = utc(2026, 8, 29, 3, 0)   # Friday 23:00 EDT
    assert poll_delay_seconds(15, 300, 3600, elapsed_seconds=2, now=normal) == pytest.approx(13)
    assert poll_delay_seconds(15, 300, 3600, elapsed_seconds=2, now=after_hours) == pytest.approx(298)


def test_weekend_polling_is_hourly() -> None:
    saturday = utc(2026, 8, 29, 16, 0)     # Saturday 12:00 EDT
    assert poll_delay_seconds(15, 300, 3600, elapsed_seconds=2, now=saturday) == pytest.approx(3598)


def test_off_hours_delay_wakes_at_weekday_session_open() -> None:
    before_open = utc(2026, 8, 28, 10, 59)  # Friday 06:59 EDT
    assert seconds_until_next_normal_session(before_open) == pytest.approx(60)
    assert poll_delay_seconds(60, 300, 3600, now=before_open) == pytest.approx(60)
