#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKX order-book historical ZIP loader for backtests.

Design:
- Local first: use downloaded official OKX books ZIP/export files.
- Lazy fetch: if a completed day is missing, request OKX export links and download.
- ZIP-only cache: no sqlite/db duplicate; raw ZIP files are the cache.
- Memory safe: streams raw rows from ZIP/text files in chunks.

The historical books export schema can vary, so this loader exposes raw rows with
best-effort timestamp inference instead of expanding every book row into a giant
in-memory structure.
"""

from __future__ import annotations

import ast
import csv
import email.utils
import io
import itertools
import gzip
import hashlib
import json
import logging
import os
import re
import time
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from src.liquidity_map.models import BookEvent, BookLevel

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
        # Existing CoinBacktest archives are stored directly under the symbol
        # directory, e.g.
        # data/okx/raw/books/ETH-USDT-SWAP/ETH-USDT-SWAP-L2orderbook-400lv-2026-06-01.tar.gz
        #
        # Older/newer loader-managed downloads may live under l2_<depth>/YYYY/MM.
        # Keep both layouts readable; raw_dir points at the symbol root so error
        # messages describe the user's real data location.
        self.raw_dir = self.data_dir / "okx" / "raw" / "books" / self._safe(symbol)
        self.depth_raw_dir = self.raw_dir / self.timeframe

    def iter_books(self, start_date: str | date, end_date: str | date, *, chunksize: int = 100_000) -> Iterator[tuple[date, pd.DataFrame]]:
        """Yield book rows by day/chunk; auto-fetch missing completed days."""
        for day in self._date_range(start_date, end_date):
            if not self.has_complete_day(day):
                if not self._is_completed_day(day):
                    logger.info("skip missing incomplete/current books day: %s", day)
                    continue
                self.download_day(day)

            for chunk in self.read_day(day, chunksize=chunksize):
                yield day, chunk

    def has_complete_day(self, day: str | date) -> bool:
        manifest = self.manifest_path(day)
        if not manifest.exists() or manifest.stat().st_size <= 0:
            return False
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("complete") is not True:
                return False
            files = data.get("files") or []
            return bool(files) and all(Path(item["path"]).exists() and Path(item["path"]).stat().st_size > 0 for item in files)
        except Exception:
            return False

    def read_day(self, day: str | date, *, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
        """Stream one local books day from raw ZIP/export files. Does not download."""
        manifest = self.manifest_path(day)
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            files = [Path(item["path"]) for item in data.get("files", [])]
        else:
            files = self.find_local_book_files(day)

        batch: list[dict[str, Any]] = []
        for file_path in files:
            for row in self._iter_book_rows(file_path):
                batch.append(row)
                if len(batch) >= chunksize:
                    yield pd.DataFrame(batch)
                    batch.clear()
            if batch:
                yield pd.DataFrame(batch)
                batch.clear()

    def iter_book_events(
        self,
        day: str | date,
        *,
        files: Sequence[str | Path] | None = None,
    ) -> Iterator[BookEvent]:
        """Yield normalized snapshot/update events from one local books day.

        The official export format has changed across releases.  This method
        intentionally supports the three forms seen in local archives:

        * JSON/JSONL websocket-like messages with ``action/data/asks/bids``;
        * CSV rows with nested JSON ``asks`` and ``bids`` columns;
        * CSV price-level rows with ``timestamp/side/price/size`` columns.

        It does not download missing data.  Unsupported schemas fail with a
        diagnostic containing the detected header and first data line.
        """

        selected_files = [Path(item) for item in files] if files is not None else self._book_files_for_day(day)
        if not selected_files:
            raise FileNotFoundError(
                f"no local OKX books file found for {self.symbol} {self._parse_date(day)} under {self.raw_dir}"
            )
        emitted = 0
        for file_path in selected_files:
            for source_name, lines in self._iter_text_sources(file_path):
                for event in self._parse_book_stream(lines, source_name=source_name):
                    emitted += 1
                    yield event
        if emitted == 0:
            sample = self.inspect_day_schema(day)
            raise ValueError(
                "OKX books files were found but no normalized events were parsed. "
                f"schema sample={sample}"
            )

    def inspect_day_schema(self, day: str | date, *, max_lines: int = 3) -> dict[str, Any]:
        """Return a small non-destructive schema sample for diagnostics."""

        out: dict[str, Any] = {"symbol": self.symbol, "day": str(self._parse_date(day)), "files": []}
        for file_path in self._book_files_for_day(day):
            for source_name, lines in self._iter_text_sources(file_path):
                sample: list[str] = []
                for line in lines:
                    text = line.strip()
                    if text:
                        sample.append(text[:1000])
                    if len(sample) >= max_lines:
                        break
                out["files"].append({"source": source_name, "sample": sample})
        return out

    def probe_day_events(self, day: str | date, *, max_events: int = 200) -> dict[str, Any]:
        """Parse a small event prefix and summarize cadence/schema safely.

        This is intentionally bounded.  It helps distinguish high-frequency
        incremental 400-level exports from slower repeated 5000-level snapshots
        without scanning the whole archive.
        """
        action_counts: dict[str, int] = {}
        timestamps: list[int] = []
        bid_levels: list[int] = []
        ask_levels: list[int] = []
        sources: set[str] = set()
        for event in self.iter_book_events(day):
            action_counts[event.action] = action_counts.get(event.action, 0) + 1
            timestamps.append(int(event.ts_ms))
            bid_levels.append(len(event.bids))
            ask_levels.append(len(event.asks))
            if event.source_file:
                sources.add(event.source_file)
            if len(timestamps) >= max(1, int(max_events)):
                break
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b >= a]
        intervals_sorted = sorted(intervals)
        median_interval = intervals_sorted[len(intervals_sorted) // 2] if intervals_sorted else None
        return {
            "requested_depth": self.depth,
            "events_sampled": len(timestamps),
            "action_counts": action_counts,
            "first_ts_ms": timestamps[0] if timestamps else None,
            "last_ts_ms": timestamps[-1] if timestamps else None,
            "median_interval_ms": median_interval,
            "bid_levels": {
                "min": min(bid_levels) if bid_levels else 0,
                "max": max(bid_levels) if bid_levels else 0,
                "last": bid_levels[-1] if bid_levels else 0,
            },
            "ask_levels": {
                "min": min(ask_levels) if ask_levels else 0,
                "max": max(ask_levels) if ask_levels else 0,
                "last": ask_levels[-1] if ask_levels else 0,
            },
            "sources": sorted(sources),
        }

    def _book_files_for_day(self, day: str | date) -> list[Path]:
        manifest = self.manifest_path(day)
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                files = [Path(item["path"]) for item in data.get("files", [])]
                existing = [path for path in files if path.exists() and path.stat().st_size > 0]
                if existing:
                    return existing
            except Exception:
                pass
        return self.find_local_book_files(day)

    def _iter_text_sources(self, path: str | Path) -> Iterator[tuple[str, Iterator[str]]]:
        p = Path(path)
        name = p.name.lower()
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(p, mode="r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    raw = tf.extractfile(member)
                    if raw is None:
                        continue
                    with raw:
                        wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                        yield f"{p}!{member.name}", iter(wrapper)
            return
        if name.endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    with zf.open(member) as raw:
                        wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                        yield f"{p}!{member}", iter(wrapper)
            return
        if name.endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8-sig", errors="replace", newline="") as f:
                yield str(p), iter(f)
            return
        with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            yield str(p), iter(f)

    def _parse_book_stream(self, lines: Iterator[str], *, source_name: str) -> Iterator[BookEvent]:
        iterator = iter(lines)
        prefix: list[str] = []
        first = ""
        for line in iterator:
            prefix.append(line)
            if line.strip():
                first = line.lstrip("\ufeff \t\r\n")
                break
        if not first:
            return
        chained = itertools.chain(prefix, iterator)
        if first.startswith(("{", "[")):
            yield from self._parse_json_lines(chained, source_name=source_name)
            return
        yield from self._parse_csv_lines(chained, source_name=source_name)

    def _parse_json_lines(self, lines: Iterator[str], *, source_name: str) -> Iterator[BookEvent]:
        for line_no, line in enumerate(lines, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid books JSON at {source_name}:{line_no}: {exc}") from exc
            yield from self._events_from_payload(payload, source_name=source_name, line_no=line_no)

    def _parse_csv_lines(self, lines: Iterator[str], *, source_name: str) -> Iterator[BookEvent]:
        buffered = iter(lines)
        try:
            header_line = next(buffered)
        except StopIteration:
            return
        while not header_line.strip():
            try:
                header_line = next(buffered)
            except StopIteration:
                return
        try:
            dialect = csv.Sniffer().sniff(header_line, delimiters=",\t;|")
            delimiter = dialect.delimiter
            skipinitialspace = bool(getattr(dialect, "skipinitialspace", False))
        except csv.Error:
            delimiter = ","
            skipinitialspace = False
        # Sniffing a header-only line can incorrectly set ``doublequote=False``.
        # Official/nested exports contain JSON arrays escaped with doubled
        # quotes, so use standard CSV quote semantics while keeping the
        # detected delimiter.
        reader = csv.DictReader(
            itertools.chain([header_line], buffered),
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
            skipinitialspace=skipinitialspace,
        )
        if not reader.fieldnames:
            raise ValueError(f"books CSV has no header: {source_name}")
        normalized = {self._normalize_key(name): name for name in reader.fieldnames if name is not None}
        has_nested = "asks" in normalized or "bids" in normalized
        has_level = (
            any(key in normalized for key in {"side", "bookside"})
            and any(key in normalized for key in {"price", "px"})
            and any(key in normalized for key in {"size", "sz", "qty", "quantity", "amount"})
        )
        if has_nested:
            for line_no, row in enumerate(reader, start=2):
                payload = {self._normalize_key(key): value for key, value in row.items() if key is not None}
                yield from self._events_from_payload(payload, source_name=source_name, line_no=line_no)
            return
        if has_level:
            yield from self._parse_level_rows(reader, normalized=normalized, source_name=source_name)
            return
        raise ValueError(
            f"unsupported OKX books CSV schema in {source_name}; columns={reader.fieldnames}. "
            "Expected asks/bids message columns or timestamp/side/price/size level rows."
        )

    def _parse_level_rows(
        self,
        reader: csv.DictReader,
        *,
        normalized: Mapping[str, str],
        source_name: str,
    ) -> Iterator[BookEvent]:
        ts_col = self._first_column(normalized, "ts", "timestamp", "time", "datetime", "exchange_ts", "exchangets")
        side_col = self._first_column(normalized, "side", "bookside")
        price_col = self._first_column(normalized, "price", "px")
        size_col = self._first_column(normalized, "size", "sz", "qty", "quantity", "amount")
        order_col = self._first_column(normalized, "ordercount", "orders", "numorders", "count")
        action_col = self._first_column(normalized, "action", "type", "eventtype", "snapshot")
        seq_col = self._first_column(normalized, "seqid", "seq", "sequenceid")
        prev_col = self._first_column(normalized, "prevseqid", "prevseq", "previoussequenceid")
        if not all([ts_col, side_col, price_col, size_col]):
            raise ValueError(f"incomplete price-level books schema in {source_name}: {reader.fieldnames}")

        current_key: tuple[Any, ...] | None = None
        bids: list[BookLevel] = []
        asks: list[BookLevel] = []
        first_line = 2

        def emit(key: tuple[Any, ...] | None) -> BookEvent | None:
            if key is None:
                return None
            ts_ms, action, seq_id, prev_seq_id = key
            return BookEvent(
                ts_ms=int(ts_ms),
                action=str(action),
                bids=tuple(bids),
                asks=tuple(asks),
                seq_id=seq_id,
                prev_seq_id=prev_seq_id,
                source_file=source_name,
                source_line=first_line,
            )

        for line_no, row in enumerate(reader, start=2):
            ts_ms = self._coerce_ts_ms(row.get(ts_col))
            # A price-level export without an action column represents full
            # snapshots at each timestamp.  WebSocket-style incremental CSVs
            # include an explicit action/sequence column.
            action = self._coerce_action(row.get(action_col)) if action_col else "snapshot"
            seq_id = self._coerce_int(row.get(seq_col)) if seq_col else None
            prev_seq_id = self._coerce_int(row.get(prev_col)) if prev_col else None
            key = (ts_ms, action, seq_id, prev_seq_id)
            if current_key is not None and key != current_key:
                event = emit(current_key)
                if event is not None:
                    yield event
                bids.clear()
                asks.clear()
                first_line = line_no
            current_key = key
            side = str(row.get(side_col, "")).strip().lower()
            level = BookLevel(
                price=float(row.get(price_col) or 0),
                size_contracts=float(row.get(size_col) or 0),
                order_count=int(float(row.get(order_col) or 0)) if order_col else 0,
            )
            if side in {"bid", "bids", "buy"}:
                bids.append(level)
            elif side in {"ask", "asks", "sell"}:
                asks.append(level)
            else:
                raise ValueError(f"unknown book side {side!r} at {source_name}:{line_no}")
        event = emit(current_key)
        if event is not None:
            yield event

    def _events_from_payload(self, payload: Any, *, source_name: str, line_no: int) -> Iterator[BookEvent]:
        if isinstance(payload, list):
            for item in payload:
                yield from self._events_from_payload(item, source_name=source_name, line_no=line_no)
            return
        if not isinstance(payload, Mapping):
            return
        normalized = {self._normalize_key(str(key)): value for key, value in payload.items()}
        data = normalized.get("data")
        if isinstance(data, list) and data and isinstance(data[0], Mapping):
            outer_action = normalized.get("action") or normalized.get("type") or normalized.get("snapshot")
            for item in data:
                merged = dict(item)
                if outer_action is not None and "action" not in merged and "snapshot" not in merged:
                    merged["action"] = outer_action
                yield from self._events_from_payload(merged, source_name=source_name, line_no=line_no)
            return
        asks = self._parse_levels(normalized.get("asks"))
        bids = self._parse_levels(normalized.get("bids"))
        if not asks and not bids and not any(key in normalized for key in {"asks", "bids"}):
            return
        ts_value = self._first_value(normalized, "ts", "timestamp", "time", "datetime", "exchange_ts", "exchangets")
        if ts_value is None:
            raise ValueError(f"books event has no timestamp at {source_name}:{line_no}")
        action_value = self._first_value(normalized, "action", "type", "eventtype", "snapshot", "issnapshot")
        if action_value is None:
            # Historical export rows commonly contain complete asks+bids but no
            # action field.  Treat those as snapshots.  A one-sided payload
            # without an action remains an update.
            action_value = "snapshot" if asks and bids else "update"
        seq_value = self._first_value(normalized, "seqid", "seq", "sequenceid")
        prev_value = self._first_value(normalized, "prevseqid", "prevseq", "previoussequenceid")
        yield BookEvent(
            ts_ms=self._coerce_ts_ms(ts_value),
            action=self._coerce_action(action_value),
            bids=tuple(bids),
            asks=tuple(asks),
            seq_id=self._coerce_int(seq_value),
            prev_seq_id=self._coerce_int(prev_value),
            source_file=source_name,
            source_line=line_no,
        )

    def _parse_levels(self, value: Any) -> list[BookLevel]:
        if value is None or value == "":
            return []
        obj = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(text)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"cannot parse book levels: {text[:200]!r}") from exc
        if isinstance(obj, Mapping):
            obj = obj.get("levels") or obj.get("data") or []
        if not isinstance(obj, (list, tuple)):
            raise ValueError(f"book levels must be an array, got {type(obj).__name__}")
        out: list[BookLevel] = []
        for row in obj:
            if isinstance(row, Mapping):
                normalized = {self._normalize_key(str(key)): item for key, item in row.items()}
                price = self._first_value(normalized, "price", "px")
                size = self._first_value(normalized, "size", "sz", "qty", "quantity", "amount")
                orders = self._first_value(normalized, "ordercount", "orders", "numorders", "count")
                if price is None or size is None:
                    continue
                out.append(BookLevel(float(price), float(size), int(float(orders or 0))))
            else:
                out.append(BookLevel.from_sequence(row))
        return out

    def _coerce_ts_ms(self, value: Any) -> int:
        if value is None:
            raise ValueError("timestamp cannot be null")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return int(ts.timestamp() * 1000)
        if numeric < 10_000_000_000:
            numeric *= 1000
        return int(numeric)

    def _coerce_action(self, value: Any) -> str:
        if isinstance(value, bool):
            return "snapshot" if value else "update"
        text = str(value or "update").strip().lower()
        if text in {"1", "true", "yes", "snapshot", "full", "partial"}:
            return "snapshot"
        return "update"

    def _coerce_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def _first_column(self, mapping: Mapping[str, str], *keys: str) -> str | None:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    def _first_value(self, mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    def download_day(self, day: str | date) -> list[DownloadResult]:
        """Download official OKX books files for one completed day and write a tiny manifest."""
        d = self._parse_date(day)
        if not self._is_completed_day(d):
            raise RuntimeError(f"refuse to cache incomplete/current books day: {d}")

        items = self._request_export_items(d)
        if not items:
            raise RuntimeError(f"OKX books export returned no links for {d}")

        results: list[DownloadResult] = []
        for idx, item in enumerate(items, start=1):
            url = item["url"]
            filename = item.get("file_name") or Path(urlparse(url).path).name or f"{self.symbol}_books_{d.isoformat()}_{idx:04d}.zip"
            raw_file = self.raw_path(d, filename)
            results.append(self._download_one(url, raw_file))

        self._write_manifest(d, results)
        return results

    def fetch_snapshot(self, *, timeout: int = 20) -> dict[str, Any]:
        """Current snapshot only; 5000 depth uses OKX ``books-full``."""
        endpoint = "books-full" if self.depth > 400 else "books"
        url = f"https://www.okx.com/api/v5/market/{endpoint}?instId={self.symbol}&sz={self.depth}"
        req = urllib.request.Request(url, headers={"User-Agent": "CoinBacktest/okx-books-loader"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if str(payload.get("code", "0")) != "0":
            raise RuntimeError(f"OKX books snapshot error: {payload}")
        return {"arg": {"channel": "books", "instId": self.symbol}, "data": payload.get("data") or [], "fetched_at_utc": datetime.now(UTC).isoformat()}

    def manifest_path(self, day: str | date) -> Path:
        d = self._parse_date(day)
        return self.depth_raw_dir / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.manifest.json"

    def raw_path(self, day: str | date, filename: str) -> Path:
        d = self._parse_date(day)
        return self.depth_raw_dir / f"{d.year:04d}" / f"{d.month:02d}" / Path(filename).name

    def find_local_book_files(self, day: str | date) -> list[Path]:
        """Find one raw UTC day for the requested depth only.

        Official 400-level and 5000-level archives can coexist in the same
        symbol directory.  Mixing them would replay two incompatible feeds for
        one day, so root-level files carrying a ``<depth>lv`` marker are matched
        strictly to ``self.depth``.  Files without a depth marker are accepted
        only from the loader-managed ``l2_<depth>`` directory.
        """
        d = self._parse_date(day)
        search_dirs = [
            (self.raw_dir, False),
            (self.raw_dir / f"{d.year:04d}" / f"{d.month:02d}", False),
            (self.depth_raw_dir, True),
            (self.depth_raw_dir / f"{d.year:04d}" / f"{d.month:02d}", True),
        ]
        patterns = [f"*{d.isoformat()}*", f"*{d.strftime('%Y%m%d')}*"]
        depth_re = re.compile(r"(?:^|[-_])(?:l2orderbook[-_])?(\d+)lv(?:[-_.]|$)", re.IGNORECASE)
        out: list[Path] = []
        for base, depth_scoped in search_dirs:
            if not base.exists():
                continue
            for pattern in patterns:
                for path in base.glob(pattern):
                    if not path.is_file() or path.stat().st_size <= 0 or path.name.endswith(".manifest.json"):
                        continue
                    match = depth_re.search(path.name)
                    if match is not None:
                        if int(match.group(1)) != self.depth:
                            continue
                    elif not depth_scoped:
                        # A root-level file without an explicit depth marker is
                        # ambiguous when both official feeds are present.
                        continue
                    out.append(path)
        return sorted(set(out))

    def _write_manifest(self, day: date, results: Sequence[DownloadResult]) -> Path:
        manifest = self.manifest_path(day)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "complete": bool(results),
            "symbol": self.symbol,
            "depth": self.depth,
            "date": day.isoformat(),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": [
                {
                    "url": item.url,
                    "path": str(item.path),
                    "status": item.status,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in results
            ],
        }
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("saved books manifest: %s files=%s", manifest, len(results))
        return manifest

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
                result = DownloadResult(url=url, path=output_path, status="downloaded", size_bytes=output_path.stat().st_size, sha256=self._sha256(output_path))
                logger.info("downloaded OKX books file: %s", output_path)
                return result
            except Exception as exc:
                last_error = repr(exc)
                logger.warning("download retry %s/%s url=%s error=%s", attempt, retries, url, last_error)
                time.sleep(min(2 ** attempt, 30))
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download OKX books file: {last_error}")

    def _iter_book_rows(self, path: str | Path) -> Iterator[dict[str, Any]]:
        source = str(path)
        for line_no, line in enumerate(self._iter_text_lines(path), start=1):
            text = line.rstrip("\n\r")
            if not text:
                continue
            yield {
                "source_file": source,
                "line_no": line_no,
                "ts_ms": self._infer_ts_ms(text),
                "raw_text": text,
            }

    def _iter_text_lines(self, path: str | Path) -> Iterator[str]:
        p = Path(path)
        name = p.name.lower()
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(p, mode="r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    raw = tf.extractfile(member)
                    if raw is None:
                        continue
                    with raw:
                        wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                        yield from wrapper
            return
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
