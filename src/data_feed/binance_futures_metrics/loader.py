#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local-first facade for Binance USD-M futures metrics."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import requests

from .archive import BinanceMetricsArchiveClient
from .features import add_derived_ratio_columns, build_relative_features, set_index_mode
from .models import (
    BINANCE_VISION_BASE_URL,
    DEFAULT_RELATIVE_WINDOWS,
    BinanceMetricsCoverage,
    BinanceMetricsDayResult,
    BinanceMetricsDownloadSummary,
    iter_days,
    normalize_symbol,
    parse_archive_day,
    parse_timedelta,
)
from .store import BinanceMetricsStore


class BinanceFuturesMetricsLoader:
    """Download and load official Binance USD-M 5-minute metrics."""

    def __init__(
        self,
        symbol: str = "ETHUSDT",
        *,
        data_dir: str | Path | None = None,
        session: requests.Session | None = None,
        base_url: str = BINANCE_VISION_BASE_URL,
        timeout: int = 30,
        db_name: str = "binance_futures_metrics.db",
    ) -> None:
        self.symbol = normalize_symbol(symbol)
        if not self.symbol.endswith("USDT"):
            raise ValueError(f"USD-M metrics symbol must end with USDT, got {self.symbol!r}")
        project_root = Path(__file__).resolve().parents[3]
        self.data_dir = Path(data_dir) if data_dir else project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / db_name
        self.raw_root = self.data_dir / "binance" / "raw" / "futures_metrics" / self.symbol
        self.store = BinanceMetricsStore(self.symbol, self.db_path)
        self.archive = BinanceMetricsArchiveClient(
            self.symbol,
            raw_root=self.raw_root,
            session=session,
            base_url=base_url,
            timeout=timeout,
        )

    def archive_url(self, day_utc: Any) -> str:
        return self.archive.archive_url(day_utc)

    def checksum_url(self, day_utc: Any) -> str:
        return self.archive.checksum_url(day_utc)

    def raw_archive_path(self, day_utc: Any) -> Path:
        return self.archive.raw_archive_path(day_utc)

    def download_history(
        self,
        start_date: Any,
        end_date: Any,
        *,
        workers: int = 6,
        force_rebuild: bool = False,
        verify_checksum: bool = True,
        require_checksum: bool = False,
        keep_raw: bool = True,
        retries: int = 4,
        retry_backoff_seconds: float = 0.75,
        progress: Callable[[int, int, BinanceMetricsDayResult], None] | None = None,
    ) -> BinanceMetricsDownloadSummary:
        days = iter_days(start_date, end_date)
        started = time.perf_counter()
        pending: list[date] = []
        skipped = 0
        for day in days:
            if not force_rebuild and self.store.day_is_complete(day):
                skipped += 1
            else:
                pending.append(day)

        counters = {"downloaded": 0, "partial": 0, "missing": 0, "error": 0, "rows": 0}
        done = skipped
        total = len(days)
        if progress and skipped:
            progress(
                done,
                total,
                BinanceMetricsDayResult(days[0], "skipped", 0, None, ""),
            )

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="binance-metrics") as executor:
                futures = {
                    executor.submit(
                        self.archive.fetch_day,
                        day,
                        verify_checksum=verify_checksum,
                        require_checksum=require_checksum,
                        keep_raw=keep_raw,
                        force_download=force_rebuild,
                        retries=retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    ): day
                    for day in pending
                }
                for future in as_completed(futures):
                    day = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = BinanceMetricsDayResult(
                            day,
                            "error",
                            0,
                            None,
                            self.archive_url(day),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    if result.status in {"complete", "partial"} and result.frame is not None:
                        self.store.save_day(result)
                        counters["downloaded"] += 1
                        counters["rows"] += int(result.rows)
                        counters["partial"] += int(result.status == "partial")
                    else:
                        self.store.save_coverage_result(result)
                        counters["missing" if result.status == "missing" else "error"] += 1
                    done += 1
                    if progress:
                        progress(done, total, result)

        return BinanceMetricsDownloadSummary(
            symbol=self.symbol,
            requested_days=total,
            downloaded_days=counters["downloaded"],
            skipped_days=skipped,
            partial_days=counters["partial"],
            missing_days=counters["missing"],
            error_days=counters["error"],
            rows_written=counters["rows"],
            elapsed_seconds=time.perf_counter() - started,
            db_path=self.db_path,
        )

    def inspect_archive_day(
        self,
        day_utc: Any,
        *,
        verify_checksum: bool = True,
        require_checksum: bool = False,
        retries: int = 2,
    ) -> BinanceMetricsDayResult:
        return self.archive.fetch_day(
            parse_archive_day(day_utc),
            verify_checksum=verify_checksum,
            require_checksum=require_checksum,
            keep_raw=False,
            force_download=True,
            retries=retries,
            retry_backoff_seconds=0.5,
        )

    def load_metrics(
        self,
        start: Any,
        end: Any,
        *,
        publication_lag: str | pd.Timedelta = "1min",
        index_mode: str = "timestamp",
    ) -> pd.DataFrame:
        start_ts, end_ts = self._validate_range(start, end)
        frame = self.store.load_timestamp_range(start_ts, end_ts)
        return self._finalize_raw(frame, publication_lag=publication_lag, index_mode=index_mode)

    def load_archive_days(
        self,
        start_date: Any,
        end_date: Any,
        *,
        publication_lag: str | pd.Timedelta = "1min",
        index_mode: str = "timestamp",
    ) -> pd.DataFrame:
        frame = self.store.load_archive_days(start_date, end_date)
        return self._finalize_raw(frame, publication_lag=publication_lag, index_mode=index_mode)

    def load_relative_features(
        self,
        start: Any,
        end: Any,
        *,
        windows: Sequence[str | pd.Timedelta] = DEFAULT_RELATIVE_WINDOWS,
        publication_lag: str | pd.Timedelta = "1min",
        baseline_tolerance: str | pd.Timedelta = "5min",
        index_mode: str = "available_time",
    ) -> pd.DataFrame:
        start_ts, end_ts = self._validate_range(start, end)
        parsed_windows = tuple(parse_timedelta(value) for value in windows)
        if not parsed_windows:
            raise ValueError("windows must not be empty")
        lag = parse_timedelta(publication_lag)
        tolerance = parse_timedelta(baseline_tolerance)
        raw = self.store.load_timestamp_range(start_ts - max(parsed_windows) - tolerance - lag, end_ts)
        if raw.empty:
            return raw
        raw = add_derived_ratio_columns(raw)
        raw = build_relative_features(raw, windows=parsed_windows, baseline_tolerance=tolerance)
        raw["available_time"] = pd.to_datetime(raw["timestamp"]) + lag
        out = raw.loc[(raw["available_time"] >= start_ts) & (raw["available_time"] <= end_ts)].copy()
        return set_index_mode(out, index_mode=index_mode)

    def coverage(self) -> BinanceMetricsCoverage:
        return self.store.coverage()

    def coverage_by_day(self, start_date: Any | None = None, end_date: Any | None = None) -> pd.DataFrame:
        return self.store.coverage_by_day(start_date, end_date)

    @staticmethod
    def _validate_range(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end must be greater than or equal to start")
        return start_ts, end_ts

    @staticmethod
    def _finalize_raw(
        frame: pd.DataFrame,
        *,
        publication_lag: str | pd.Timedelta,
        index_mode: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = add_derived_ratio_columns(frame)
        out["available_time"] = pd.to_datetime(out["timestamp"]) + parse_timedelta(publication_lag)
        return set_index_mode(out, index_mode=index_mode)
