#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Official Binance Vision daily archive client and parser."""

from __future__ import annotations

import hashlib
import io
import re
import threading
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .models import (
    BINANCE_VISION_BASE_URL,
    EXPECTED_ROWS_PER_DAY,
    METRIC_COLUMNS,
    METRICS_PERIOD,
    BinanceMetricsDayResult,
    BinanceMetricsDownloadError,
    normalize_symbol,
    parse_archive_day,
    timezone_offset,
)


class BinanceMetricsArchiveClient:
    """Fetch, validate, retain, and normalize official daily metrics ZIPs."""

    def __init__(
        self,
        symbol: str,
        *,
        raw_root: Path,
        session: Any | None = None,
        base_url: str = BINANCE_VISION_BASE_URL,
        timeout: int = 30,
    ) -> None:
        self.symbol = normalize_symbol(symbol)
        self.raw_root = Path(raw_root)
        self.session = session
        self.base_url = str(base_url).rstrip("/")
        self.timeout = max(1, int(timeout))
        self._thread_local = threading.local()

    def archive_url(self, day_utc: date | str | pd.Timestamp) -> str:
        day = parse_archive_day(day_utc).isoformat()
        filename = f"{self.symbol}-metrics-{day}.zip"
        return f"{self.base_url}/data/futures/um/daily/metrics/{self.symbol}/{filename}"

    def checksum_url(self, day_utc: date | str | pd.Timestamp) -> str:
        return self.archive_url(day_utc) + ".CHECKSUM"

    def raw_archive_path(self, day_utc: date | str | pd.Timestamp) -> Path:
        day = parse_archive_day(day_utc)
        return self.raw_root / f"{day.year:04d}" / f"{day.month:02d}" / f"{self.symbol}-metrics-{day.isoformat()}.zip"

    def fetch_day(
        self,
        day: date,
        *,
        verify_checksum: bool,
        require_checksum: bool,
        keep_raw: bool,
        force_download: bool,
        retries: int,
        retry_backoff_seconds: float,
    ) -> BinanceMetricsDayResult:
        url = self.archive_url(day)
        raw_path = self.raw_archive_path(day)
        used_local = raw_path.exists() and not force_download
        if used_local:
            archive_bytes = raw_path.read_bytes()
        else:
            archive_bytes = self._request_bytes(
                url,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            if archive_bytes is None:
                return BinanceMetricsDayResult(
                    day_utc=day,
                    status="missing",
                    rows=0,
                    frame=None,
                    source_url=url,
                    error="archive returned HTTP 404",
                )

        expected_sha = ""
        checksum_verified = False
        if verify_checksum:
            local_checksum_path = Path(str(raw_path) + ".CHECKSUM")
            if used_local and local_checksum_path.exists():
                checksum_bytes = local_checksum_path.read_bytes()
            else:
                checksum_bytes = self._request_bytes(
                    self.checksum_url(day),
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
            if checksum_bytes is None:
                if require_checksum:
                    raise BinanceMetricsDownloadError(f"checksum missing for {day.isoformat()}")
            else:
                expected_sha = self._parse_checksum(checksum_bytes.decode("utf-8", errors="replace"))
                actual_sha = hashlib.sha256(archive_bytes).hexdigest()
                if expected_sha.lower() != actual_sha.lower():
                    raise BinanceMetricsDownloadError(
                        f"checksum mismatch for {day.isoformat()}: expected={expected_sha} actual={actual_sha}"
                    )
                checksum_verified = True

        frame = self._parse_archive(archive_bytes, day)
        rows = len(frame)
        if rows <= 0:
            raise BinanceMetricsDownloadError(f"archive contains no valid metric rows for {day.isoformat()}")
        status = "complete" if rows == EXPECTED_ROWS_PER_DAY else "partial"

        if keep_raw and not used_local:
            self._persist_raw(raw_path, archive_bytes, expected_sha)

        return BinanceMetricsDayResult(
            day_utc=day,
            status=status,
            rows=rows,
            frame=frame,
            source_url=url,
            checksum_sha256=expected_sha,
            checksum_verified=checksum_verified,
            used_local_archive=used_local,
        )

    def _http_session(self) -> Any:
        if self.session is not None:
            return self.session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _request_bytes(
        self,
        url: str,
        *,
        retries: int,
        retry_backoff_seconds: float,
    ) -> bytes | None:
        attempts = max(0, int(retries)) + 1
        for attempt in range(attempts):
            try:
                response = self._http_session().get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt + 1 >= attempts:
                    raise BinanceMetricsDownloadError(f"request failed for {url}: {exc}") from exc
                time.sleep(max(0.0, retry_backoff_seconds) * (2**attempt))
                continue

            status = int(getattr(response, "status_code", 200))
            if status == 404:
                return None
            if status == 429 or 500 <= status < 600:
                if attempt + 1 >= attempts:
                    raise BinanceMetricsDownloadError(f"HTTP {status} after {attempts} attempts: {url}")
                retry_after = getattr(response, "headers", {}).get("Retry-After")
                try:
                    sleep_for = float(retry_after) if retry_after is not None else retry_backoff_seconds * (2**attempt)
                except (TypeError, ValueError):
                    sleep_for = retry_backoff_seconds * (2**attempt)
                time.sleep(max(0.0, sleep_for))
                continue
            if status >= 400:
                text = str(getattr(response, "text", ""))[:300]
                raise BinanceMetricsDownloadError(f"HTTP {status} for {url}: {text}")
            return bytes(getattr(response, "content", b""))
        raise BinanceMetricsDownloadError(f"unreachable request state for {url}")

    def _parse_archive(self, archive_bytes: bytes, day: date) -> pd.DataFrame:
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not candidates:
                    raise BinanceMetricsDownloadError("ZIP contains no CSV file")
                preferred = [name for name in candidates if "metrics" in Path(name).name.lower()]
                member = sorted(preferred or candidates)[0]
                with archive.open(member) as handle:
                    frame = pd.read_csv(handle)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise BinanceMetricsDownloadError(f"invalid metrics ZIP: {exc}") from exc

        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required = {"create_time", "symbol", *METRIC_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise BinanceMetricsDownloadError(f"metrics CSV missing columns: {missing}")

        frame = frame.assign(source_timestamp_utc=self._parse_source_time(frame["create_time"]))
        frame = frame.loc[frame["source_timestamp_utc"].notna()].copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame = frame.loc[frame["symbol"] == self.symbol].copy()
        for column in METRIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        utc_day_start = pd.Timestamp(day, tz="UTC")
        utc_day_end = utc_day_start + pd.Timedelta(days=1)
        source_utc = pd.to_datetime(frame["source_timestamp_utc"], utc=True)
        frame = frame.loc[(source_utc >= utc_day_start) & (source_utc < utc_day_end)].copy()
        source_utc = pd.to_datetime(frame["source_timestamp_utc"], utc=True)
        frame["source_timestamp_utc"] = source_utc.dt.tz_convert(None)
        frame["timestamp"] = frame["source_timestamp_utc"] + timezone_offset()
        frame["period"] = METRICS_PERIOD
        frame["source_day_utc"] = day.isoformat()
        frame["source"] = "binance_vision_metrics"

        columns = [
            "symbol",
            "timestamp",
            "source_timestamp_utc",
            "period",
            *METRIC_COLUMNS,
            "source_day_utc",
            "source",
        ]
        return (
            frame[columns]
            .sort_values("timestamp")
            .drop_duplicates(["symbol", "timestamp"], keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _parse_source_time(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(values) else 0.0
        if numeric_ratio >= 0.95:
            finite = numeric.dropna().abs()
            magnitude = float(finite.median()) if not finite.empty else 0.0
            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        return pd.to_datetime(values, utc=True, errors="coerce")

    @staticmethod
    def _parse_checksum(text: str) -> str:
        match = re.search(r"\b([0-9a-fA-F]{64})\b", str(text))
        if not match:
            raise BinanceMetricsDownloadError("invalid checksum file")
        return match.group(1).lower()

    @staticmethod
    def _persist_raw(raw_path: Path, archive_bytes: bytes, expected_sha: str) -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = raw_path.with_suffix(raw_path.suffix + ".part")
        part_path.write_bytes(archive_bytes)
        part_path.replace(raw_path)
        if expected_sha:
            checksum_path = Path(str(raw_path) + ".CHECKSUM")
            checksum_part = Path(str(checksum_path) + ".part")
            checksum_part.write_text(f"{expected_sha}  {raw_path.name}\n", encoding="utf-8")
            checksum_part.replace(checksum_path)
