#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKX books/order-book data loader for backtests.

Usage in backtest:

    loader = OKXBooksLoader(symbol="ETH-USDT-SWAP", depth=400)

    for day, chunk in loader.iter_books("2026-06-01", "2026-06-10", chunksize=100_000):
        backtest_on_books_chunk(chunk)
        del chunk

Design:
- Local first: read local daily sqlite db if complete.
- Lazy fetch: if a completed day is missing, request OKX historical-data export
  links, download official books file, stream raw rows into sqlite, then yield it.
- Memory safe: never loads more than one day; read/write uses chunks.
"""

from __future__ import annotations

import email.utils
import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

UTC = timezone.utc
OKX_EXPORT_ENDPOINT = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"
OKX_HISTORICAL_DATA_REFERER = "https://www.okx.com/en-us/historical-data"
OKX_BOOK_MODULE_400 = "4"
OKX_BOOK_MODULE_5000 = "5"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
OKX_TOO_MANY_REQUESTS_CODE = "50011"
ISO_DATE_RE = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")
COMPACT_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
TS_MS_RE = re.compile(r"(?<!\d)(1[5-9]\d{11}|2\d{12})(?!\d)")


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: Path
    status: str
    size_bytes: int = 0
    sha256: str = ""


class OKXBooksLoader:
    def __init__(self, symbol: str = "ETH-USDT-SWAP", depth: int = 400, data_dir: str | os.PathLike[str] | None = None):
        self.symbol = symbol
        self.depth = int(depth)
        self.timeframe = f"l2_{self.depth}"
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.raw_dir = self.data_dir / "okx" / "raw" / "books" / self._safe(symbol)
        self.db_root = self.data_dir / "okx" / "books" / self._safe(symbol) / self.timeframe

    def iter_books(self, start_date: str | date, end_date: str | date, *, chunksize: int = 100_000) -> Iterator[tuple[date, pd.DataFrame]]:
        """Yield books raw rows by day/chunk; auto-fetch missing completed days."""

        for day in self._date_range(start_date, end_date):
            if not self.has_complete_day(day):
                if not self._is_completed_day(day):
                    logger.info("skip caching incomplete/current books day: %s", day)
                    continue
                self.download_and_save_day(day, line_batch_size=chunksize)

            for chunk in self.read_day(day, chunksize=chunksize):
                yield day, chunk

    def has_complete_day(self, day: str | date) -> bool:
        db = self.db_path(day)
        if not db.exists() or db.stat().st_size <= 0:
            return False
        try:
            with sqlite3.connect(db) as conn:
                row = conn.execute("SELECT value FROM meta WHERE key='complete'").fetchone()
                return bool(row and json.loads(row[0]) is True)
        except Exception:
            return False

    def read_day(self, day: str | date, *, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
        db = self.db_path(day)
        if not db.exists():
            return
        conn = sqlite3.connect(db)
        try:
            query = "SELECT row_id, source_file, line_no, ts_ms, raw_text FROM book_rows ORDER BY COALESCE(ts_ms, 0), row_id"
            for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
                yield chunk
        finally:
            conn.close()

    def download_and_save_day(self, day: str | date, *, line_batch_size: int = 100_000) -> list[DownloadResult]:
        d = self._parse_date(day)
        items = self._request_export_items(d)
        if not items:
            raise RuntimeError(f"OKX books export returned no links for {d}")

        results: list[DownloadResult] = []
        for idx, item in enumerate(items, start=1):
            url = item["url"]
            filename = item.get("file_name") or Path(urlparse(url).path).name or f"{self.symbol}_books_{d.isoformat()}_{idx:04d}.zip"
            raw_file = self.raw_dir / Path(filename).name
            result = self._download_one(url, raw_file)
            results.append(result)
            self._ingest_raw_file_to_db(d, result, line_batch_size=line_batch_size)
        return results

    def fetch_snapshot(self, *, timeout: int = 20) -> dict[str, Any]:
        """Current order-book snapshot only. This is not historical full-day data."""

        url = f"https://www.okx.com/api/v5/market/books?instId={self.symbol}&sz={self.depth}"
        req = urllib.request.Request(url, headers={"User-Agent": "CoinBacktest/okx-books-loader"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if str(payload.get("code", "0")) != "0":
            raise RuntimeError(f"OKX books snapshot error: {payload}")
        return {"arg": {"channel": "books", "instId": self.symbol}, "data": payload.get("data") or [], "fetched_at_utc": datetime.now(UTC).isoformat()}

    def db_path(self, day: str | date) -> Path:
        d = self._parse_date(day)
        return self.db_root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.db"

    def _request_export_items(self, day: date) -> list[dict[str, str]]:
        module = OKX_BOOK_MODULE_400 if self.depth <= 400 else OKX_BOOK_MODULE_5000
        inst_type, selector = self._infer_export_instrument()
        start_ms = self._date_start_ms(day)
        end_ms = self._date_start_ms(day + timedelta(days=1))
        payload = self._request_okx_export_links(
            module=module,
            inst_type=inst_type,
            inst_selector=selector,
            begin_ms=str(start_ms),
            end_ms=str(end_ms),
            date_aggr="daily",
        )
        return self._filter_items_by_date(self._extract_download_items(payload), day, day)

    def _ingest_raw_file_to_db(self, day: date, result: DownloadResult, *, line_batch_size: int) -> None:
        db = self.db_path(day)
        db.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with self._connect(db) as conn:
            self._init_db(conn)
            conn.execute(
                "INSERT OR REPLACE INTO book_files(file_path, source_url, status, size_bytes, sha256, added_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                (str(result.path), result.url, result.status, result.size_bytes, result.sha256, datetime.now(UTC).isoformat()),
            )
            batch: list[tuple[str, int, int | None, str]] = []
            for line_no, line in enumerate(self._iter_text_lines(result.path), start=1):
                text = line.rstrip("\n\r")
                if not text:
                    continue
                batch.append((str(result.path), line_no, self._infer_ts_ms(text), text))
                if len(batch) >= line_batch_size:
                    conn.executemany("INSERT OR IGNORE INTO book_rows(source_file, line_no, ts_ms, raw_text) VALUES (?, ?, ?, ?)", batch)
                    total += len(batch)
                    batch.clear()
            if batch:
                conn.executemany("INSERT OR IGNORE INTO book_rows(source_file, line_no, ts_ms, raw_text) VALUES (?, ?, ?, ?)", batch)
                total += len(batch)
            self._set_meta(conn, complete=total > 0, rows=total, symbol=self.symbol, date=day.isoformat(), depth=self.depth, source=str(result.path), sha256=result.sha256)
        logger.info("saved books db: %s rows=%s", db, total)

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_files (
                file_path TEXT PRIMARY KEY,
                source_url TEXT,
                status TEXT,
                size_bytes INTEGER,
                sha256 TEXT,
                added_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_rows (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                ts_ms INTEGER,
                raw_text TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_book_rows_ts_ms ON book_rows(ts_ms)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_book_rows_unique ON book_rows(source_file, line_no)")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def _connect(self, db: Path) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-65536")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _download_one(self, url: str, output_path: Path, *, timeout: int = 300, retries: int = 3) -> DownloadResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            return DownloadResult(url=url, path=output_path, status="skipped", size_bytes=output_path.stat().st_size, sha256=self._sha256(output_path))

        tmp = output_path.with_suffix(output_path.suffix + ".part")
        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "CoinBacktest/okx-books-loader"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
                    while True:
                        block = resp.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                tmp.replace(output_path)
                return DownloadResult(url=url, path=output_path, status="downloaded", size_bytes=output_path.stat().st_size, sha256=self._sha256(output_path))
            except Exception as exc:
                last_error = repr(exc)
                logger.warning("download retry %s/%s url=%s error=%s", attempt, retries, url, last_error)
                time.sleep(min(2 ** attempt, 30))
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download OKX books file: {last_error}")

    def _request_okx_export_links(
        self,
        *,
        module: str,
        inst_type: str,
        inst_selector: Mapping[str, list[str]],
        begin_ms: str,
        end_ms: str,
        date_aggr: str,
        timeout: int = 300,
        retries: int = 6,
        backoff_sec: float = 10.0,
    ) -> dict[str, Any]:
        body = {
            "module": str(module),
            "instType": str(inst_type),
            "instQueryParam": dict(inst_selector),
            "dateQuery": {"dateAggrType": str(date_aggr), "begin": str(begin_ms), "end": str(end_ms)},
        }
        headers = {
            "User-Agent": "Mozilla/5.0 CoinBacktest",
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://www.okx.com",
            "Referer": OKX_HISTORICAL_DATA_REFERER,
        }
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        last_error = ""
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(OKX_EXPORT_ENDPOINT, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = resp.read().decode("utf-8", errors="replace")
                result = json.loads(payload)
                if str(result.get("code")) != "0":
                    if str(result.get("code")) == OKX_TOO_MANY_REQUESTS_CODE and attempt < retries:
                        sleep = min(backoff_sec * (2 ** (attempt - 1)), 300)
                        logger.warning("OKX export retry code=%s msg=%s sleep=%s", result.get("code"), result.get("msg"), sleep)
                        time.sleep(sleep)
                        continue
                    raise RuntimeError(f"OKX export error: {result}")
                data_obj = result.get("data")
                return data_obj if isinstance(data_obj, dict) else {"data": data_obj}
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
                if exc.code in RETRYABLE_HTTP_CODES and attempt < retries:
                    retry_after = self._parse_retry_after(exc.headers.get("Retry-After"))
                    sleep = retry_after if retry_after is not None else min(backoff_sec * (2 ** (attempt - 1)), 300)
                    logger.warning("OKX export HTTP retry %s/%s sleep=%s error=%s", attempt, retries, sleep, last_error)
                    time.sleep(sleep)
                    continue
                raise RuntimeError(f"OKX export failed: {last_error}") from exc
            except Exception as exc:
                last_error = repr(exc)
                if attempt < retries:
                    sleep = min(backoff_sec * (2 ** (attempt - 1)), 300)
                    logger.warning("OKX export retry %s/%s sleep=%s error=%s", attempt, retries, sleep, last_error)
                    time.sleep(sleep)
                    continue
                raise RuntimeError(f"OKX export failed: {last_error}") from exc
        raise RuntimeError(f"OKX export failed: {last_error}")

    def _extract_download_items(self, payload: Mapping[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(url: str, file_name: str = "") -> None:
            clean = str(url or "").strip()
            if not clean.startswith(("http://", "https://")) or clean in seen:
                return
            seen.add(clean)
            items.append({"url": clean, "file_name": str(file_name or "").strip()})

        def walk(obj: Any, parent: Mapping[str, Any] | None = None) -> None:
            if isinstance(obj, Mapping):
                maybe_file = obj.get("fileName") or obj.get("filename") or obj.get("name") or ""
                for key, value in obj.items():
                    if isinstance(value, str) and key.lower() in {"url", "downloadurl", "download_url", "link"}:
                        add(value, str(maybe_file or ""))
                    walk(value, obj)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item, parent)
            elif isinstance(obj, str) and obj.startswith(("http://", "https://")):
                maybe_file = ""
                if parent:
                    maybe_file = str(parent.get("fileName") or parent.get("filename") or parent.get("name") or "")
                add(obj, maybe_file)

        walk(payload)
        return items

    def _filter_items_by_date(self, items: Sequence[Mapping[str, str]], start: date, end: date) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for item in items:
            item_date = self._infer_item_date(item)
            if item_date is None or start <= item_date <= end:
                out.append({"url": str(item.get("url", "")), "file_name": str(item.get("file_name", ""))})
        return out

    def _infer_export_instrument(self) -> tuple[str, dict[str, list[str]]]:
        if self.symbol.endswith("-SWAP"):
            return "SWAP", {"instFamilyList": [self.symbol[:-5]]}
        return "SPOT", {"instIdList": [self.symbol]}

    def _iter_text_lines(self, path: str | Path) -> Iterator[str]:
        p = Path(path)
        name = p.name.lower()
        if name.endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    with zf.open(member) as f:
                        for raw in f:
                            yield raw.decode("utf-8", errors="replace")
            return
        if name.endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                yield from f
            return
        with p.open("r", encoding="utf-8", errors="replace") as f:
            yield from f

    def _infer_item_date(self, item: Mapping[str, str]) -> date | None:
        text = " ".join([str(item.get("file_name", "")), str(item.get("url", ""))])
        for pattern in (ISO_DATE_RE, COMPACT_DATE_RE):
            for match in pattern.finditer(text):
                try:
                    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                except ValueError:
                    continue
        return None

    def _infer_ts_ms(self, text: str) -> int | None:
        match = TS_MS_RE.search(text)
        return int(match.group(1)) if match else None

    def _set_meta(self, conn: sqlite3.Connection, **kwargs) -> None:
        for key, value in kwargs.items():
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False, default=str)))

    def _date_range(self, start: str | date, end: str | date) -> Iterator[date]:
        cur = self._parse_date(start)
        last = self._parse_date(end)
        if last < cur:
            raise ValueError("end_date must be >= start_date")
        while cur <= last:
            yield cur
            cur += timedelta(days=1)

    def _parse_date(self, value: str | date) -> date:
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()

    def _is_completed_day(self, day: str | date) -> bool:
        d = self._parse_date(day)
        end = datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(days=1)
        return datetime.now(UTC) >= end

    def _date_start_ms(self, day: date) -> int:
        return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)

    def _parse_retry_after(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            try:
                dt = email.utils.parsedate_to_datetime(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return max(0.0, (dt - datetime.now(UTC)).total_seconds())
            except Exception:
                return None

    def _safe(self, value: str) -> str:
        return str(value).replace("/", "_").replace("\\", "_").strip() or "default"

    def _sha256(self, path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
