#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKX tick/trades ZIP loader for backtests.

Design:
- Local first: read the local official OKX trades ZIP if it exists.
- Lazy fetch: if a completed day is missing locally, download the official ZIP.
- ZIP-only cache: no sqlite/db copy; the downloaded ZIP is the cache.
- Flat raw layout: data/okx/raw/trades/<symbol>/<official-file>.zip.
- Memory safe: yields pandas chunks from the ZIP; never loads multiple days at once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

UTC = timezone.utc
DEFAULT_OKX_TRADES_URL_TEMPLATE = "https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip"


class OKXTickLoader:
    def __init__(
        self,
        symbol: str = "ETH-USDT-SWAP",
        data_dir: str | os.PathLike[str] | None = None,
        trades_url_template: str = "",
    ):
        self.symbol = symbol
        self.trades_url_template = trades_url_template
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.raw_dir = self.data_dir / "okx" / "raw" / "trades" / self._safe(symbol)

    def iter_trades(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        chunksize: int = 100_000,
        trades_url_template: str | None = None,
    ) -> Iterator[tuple[date, pd.DataFrame]]:
        """Yield trade/tick chunks by day.

        The loader reads local ZIP files first. If a completed UTC day is missing,
        it downloads the official OKX ZIP, keeps that ZIP as the only cache, and
        streams chunks directly from it.
        """
        template = trades_url_template if trades_url_template is not None else self.trades_url_template

        for day in self._date_range(start_date, end_date):
            raw_file = self.find_local_trade_file(day, template=template)
            if raw_file is None:
                if not self._is_completed_day(day):
                    logger.info("skip missing incomplete/current tick day: %s", day)
                    continue
                if not template:
                    raise RuntimeError(
                        "local OKX tick ZIP missing and trades_url_template is empty. "
                        "Set OKX official trade file URL template before running tick backtest."
                    )
                raw_file = self.download_official_trade_file(day, template)

            for chunk in self.read_zip(raw_file, chunksize=chunksize):
                yield day, chunk

    def has_complete_day(self, day: str | date, *, trades_url_template: str | None = None) -> bool:
        """Return True if the local official ZIP exists for the day."""
        template = trades_url_template if trades_url_template is not None else self.trades_url_template
        raw_file = self.find_local_trade_file(day, template=template)
        return bool(raw_file and raw_file.exists() and raw_file.stat().st_size > 0)

    def read_day(self, day: str | date, *, chunksize: int = 100_000, trades_url_template: str | None = None) -> Iterator[pd.DataFrame]:
        """Read one local day from ZIP. Does not download missing data."""
        template = trades_url_template if trades_url_template is not None else self.trades_url_template
        raw_file = self.find_local_trade_file(day, template=template)
        if raw_file is None:
            return
        yield from self.read_zip(raw_file, chunksize=chunksize)

    def read_zip(self, path: str | Path, *, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
        """Stream normalized trade chunks directly from a ZIP/CSV file."""
        yield from self._iter_trade_file(path, chunksize=chunksize)

    def find_local_trade_file(self, day: str | date, *, template: str | None = None) -> Path | None:
        """Find a local raw trades file for the day.

        New canonical files are stored directly under raw_dir. The old YYYY/MM
        partition is still checked as a backward-compatible fallback.
        """
        d = self._parse_date(day)
        candidates: list[Path] = []
        if template:
            url = self._format_trade_url(d, template)
            filename = Path(urlparse(url).path).name or f"{self.symbol}-trades-{d.isoformat()}.zip"
            # New canonical layout: flat raw_dir, matching tools/download_okx_historical_data.py.
            candidates.append(self.raw_dir / filename)
            # Backward-compatible fallback for older partitioned files.
            candidates.append(self.partitioned_raw_path(d, filename))

        patterns = [f"*{d.isoformat()}*", f"*{d.strftime('%Y%m%d')}*"]
        for pattern in patterns:
            # Prefer flat layout because the standalone downloader stores files here.
            candidates.extend(self.raw_dir.glob(pattern))
            # Keep old YYYY/MM layout readable for compatibility.
            candidates.extend((self.raw_dir / f"{d.year:04d}" / f"{d.month:02d}").glob(pattern))

        for path in candidates:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return path
        return None

    def download_official_trade_file(self, day: str | date, url_template: str, *, timeout: int = 60, retries: int = 3) -> Path:
        d = self._parse_date(day)
        url = self._format_trade_url(d, url_template)
        filename = Path(urlparse(url).path).name or f"{self.symbol}-trades-{d.isoformat()}.zip"
        out = self.raw_path(d, filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            return out

        tmp = out.with_suffix(out.suffix + ".part")
        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "CoinBacktest/okx-tick-loader"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
                    while True:
                        block = resp.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                tmp.replace(out)
                logger.info("downloaded OKX tick ZIP: %s sha256=%s", out, self._sha256(out))
                return out
            except Exception as exc:
                last_error = repr(exc)
                logger.warning("download retry %s/%s url=%s error=%s", attempt, retries, url, last_error)
                time.sleep(min(2 ** attempt, 30))
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download OKX tick ZIP: {last_error}")

    def raw_path(self, day: str | date, filename: str) -> Path:
        """Canonical tick raw path: flat directory, no YYYY/MM partition."""
        return self.raw_dir / Path(filename).name

    def partitioned_raw_path(self, day: str | date, filename: str) -> Path:
        """Backward-compatible path for older generated YYYY/MM tick files."""
        d = self._parse_date(day)
        return self.raw_dir / f"{d.year:04d}" / f"{d.month:02d}" / Path(filename).name

    def _iter_trade_file(self, path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(p) as zf:
                members = [name for name in zf.namelist() if not name.endswith("/")]
                if not members:
                    raise RuntimeError(f"empty OKX trade ZIP: {p}")
                for name in members:
                    with zf.open(name) as f:
                        for raw in pd.read_csv(f, chunksize=chunksize):
                            chunk = self._normalize_trades(raw)
                            if not chunk.empty:
                                yield chunk
            return

        for raw in pd.read_csv(p, chunksize=chunksize):
            chunk = self._normalize_trades(raw)
            if not chunk.empty:
                yield chunk

    def _normalize_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rename = {}
        for col in out.columns:
            low = str(col).strip().lower()
            if low in {"ts", "timestamp", "time", "datetime", "created_time", "createdtime", "create_time", "created_at", "createdat"}:
                rename[col] = "ts_ms"
            elif low in {"tradeid", "trade_id", "id", "trade_id_str"}:
                rename[col] = "trade_id"
            elif low in {"px", "price"}:
                rename[col] = "price"
            elif low in {"sz", "size", "qty", "amount"}:
                rename[col] = "size"
            elif low == "side":
                rename[col] = "side"
        out = out.rename(columns=rename)
        if "ts_ms" not in out.columns:
            raise ValueError(f"trade file has no timestamp column: {list(df.columns)}")

        numeric_ts = pd.to_numeric(out["ts_ms"], errors="coerce")
        if numeric_ts.notna().mean() > 0.9:
            out = out.loc[numeric_ts.notna()].copy()
            numeric_ts = numeric_ts.loc[out.index]
            if numeric_ts.max() < 10_000_000_000:
                numeric_ts = numeric_ts * 1000
            out["ts_ms"] = numeric_ts.astype("int64")
            out["timestamp"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)
        else:
            ts = pd.to_datetime(out["ts_ms"], utc=True, errors="coerce")
            out = out.loc[ts.notna()].copy()
            ts = ts.loc[out.index]
            out["timestamp"] = ts
            out["ts_ms"] = (ts.astype("int64") // 1_000_000).astype("int64")

        for col in ["trade_id", "price", "size", "side"]:
            if col not in out.columns:
                out[col] = None
        out["symbol"] = self.symbol
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out["size"] = pd.to_numeric(out["size"], errors="coerce")
        out["side"] = out["side"].astype(str).str.lower()

        base = {"ts_ms", "timestamp", "symbol", "trade_id", "price", "size", "side"}
        extra_cols = [c for c in out.columns if c not in base]
        if extra_cols:
            out["raw_json"] = out[extra_cols].apply(
                lambda row: json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str), axis=1
            )
        else:
            out["raw_json"] = None

        out = out[["ts_ms", "timestamp", "symbol", "trade_id", "price", "size", "side", "raw_json"]]
        out = out.dropna(subset=["ts_ms", "price", "size"]).sort_values("ts_ms")
        return out.reset_index(drop=True)

    def _format_trade_url(self, day: date, url_template: str) -> str:
        return url_template.format(date=day.isoformat(), yyyymmdd=day.strftime("%Y%m%d"), symbol=self.symbol, kind="trades")

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

    def _safe(self, value: str) -> str:
        return str(value).replace("/", "_").replace("\\", "_").strip() or "default"

    def _sha256(self, path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
