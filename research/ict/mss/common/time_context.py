#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calendar/session labels for ICT MSS research.

Input timestamps in CoinBacktest's legacy OHLC database are naive timestamps in
``config.loader.TIMEZONE``.  We first attach that fixed offset, then convert to
real UTC/London/New-York clocks so DST is handled for London and New York.
"""

from __future__ import annotations

from datetime import timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def parse_project_timezone_offset_hours(value: object, default: int = 8) -> int:
    text = str(value).strip()
    if not text:
        return int(default)
    if text.upper() in {"UTC", "Z", "+0", "0"}:
        return 0
    try:
        if text.startswith("+"):
            return int(text[1:])
        if text.startswith("-"):
            return -int(text[1:])
        return int(text)
    except ValueError:
        return int(default)


def _hour_decimal(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0


def add_calendar_session_context(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    project_offset_hours: int = 8,
) -> pd.DataFrame:
    """Attach overlapping session flags without forcing one exclusive session.

    The windows are fixed before outcome inspection:
    - Asia: 08:00-16:00 Asia/Shanghai
    - London active: 07:00-12:00 Europe/London
    - New York active: 08:00-16:00 America/New_York
    - ICT London KZ: 02:00-05:00 America/New_York
    - ICT NY KZ: 07:00-10:00 America/New_York
    - US cash open: 09:30-11:00 America/New_York (plus a 30m opening flag)

    Flags intentionally overlap; overlap is economically real and avoids a
    fragile arbitrary priority rule.
    """

    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    local_tz = timezone(timedelta(hours=int(project_offset_hours)))
    aware = pd.DatetimeIndex(ts).tz_localize(local_tz)
    utc = aware.tz_convert("UTC")
    sh = aware.tz_convert(ZoneInfo("Asia/Shanghai"))
    lon = aware.tz_convert(ZoneInfo("Europe/London"))
    ny = aware.tz_convert(ZoneInfo("America/New_York"))

    utc_h = _hour_decimal(utc)
    sh_h = _hour_decimal(sh)
    lon_h = _hour_decimal(lon)
    ny_h = _hour_decimal(ny)

    out["utc_timestamp"] = utc.tz_localize(None)
    out["new_york_timestamp"] = ny.tz_localize(None)
    out["london_timestamp"] = lon.tz_localize(None)
    out["utc_day_of_week"] = utc.day_name()
    out["ny_day_of_week"] = ny.day_name()
    out["is_weekend_utc"] = utc.dayofweek.to_numpy() >= 5
    out["is_weekday_utc"] = ~out["is_weekend_utc"]
    out["utc_hour"] = utc_h
    out["shanghai_hour"] = sh_h
    out["london_hour"] = lon_h
    out["new_york_hour"] = ny_h

    out["session_asia"] = (sh_h >= 8.0) & (sh_h < 16.0)
    out["session_london"] = (lon_h >= 7.0) & (lon_h < 12.0)
    out["session_new_york"] = (ny_h >= 8.0) & (ny_h < 16.0)
    out["ict_london_kill_zone"] = (ny_h >= 2.0) & (ny_h < 5.0)
    out["ict_new_york_kill_zone"] = (ny_h >= 7.0) & (ny_h < 10.0)
    out["us_cash_open_30m"] = (ny_h >= 9.5) & (ny_h < 10.0)
    out["us_cash_open_90m"] = (ny_h >= 9.5) & (ny_h < 11.0)
    out["new_york_am"] = (ny_h >= 8.0) & (ny_h < 12.0)
    return out
