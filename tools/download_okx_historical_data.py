#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download OKX official historical data files to local storage.

Supported workflows:

1) Trades CDN template mode

   python tools/download_okx_historical_data.py \
     --kind trades \
     --symbol ETH-USDT-SWAP \
     --start-date 2025-05-01 \
     --end-date 2025-05-31 \
     --url-template 'https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip'

2) Books export-link mode

   python tools/download_okx_historical_data.py \
     --kind books \
     --symbol ETH-USDT-SWAP \
     --start-date 2025-05-01 \
     --end-date 2025-05-31

   This requests official OKX historical-data export links and downloads the
   returned files. If OKX returns no links for the selected range, export the
   book files from the OKX historical-data page and pass the resulting official
   URLs with --url or --manifest.

3) Direct URL / manifest mode

   python tools/download_okx_historical_data.py \
     --kind books \
     --symbol ETH-USDT-SWAP \
     --url 'https://.../one-okx-file.zip'

Downloaded files are saved under:
    data/okx/raw/<kind>/<symbol>/
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.log import get_logger

logger = get_logger("OKXHistoricalDownloader")

DEFAULT_OUT_ROOT = ROOT / "data" / "okx" / "raw"
VALID_KINDS = {"trades", "books", "books_l2"}
BOOK_KINDS = {"books", "books_l2"}
OKX_EXPORT_ENDPOINT = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"
OKX_HISTORICAL_DATA_REFERER = "https://www.okx.com/en-us/historical-data"
OKX_BOOK_MODULE_400 = "4"
OKX_BOOK_MODULE_5000 = "5"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
OKX_TOO_MANY_REQUESTS_CODE = "50011"
ISO_DATE_RE = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")
COMPACT_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


@dataclass
class DownloadTask:
    url: str
    output_path: Path
    date_tag: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OKX official historical data files for local backtests.")
    p.add_argument("--kind", required=True, choices=sorted(VALID_KINDS), help="Data kind: trades/books/books_l2.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP", help="OKX instrument id.")
    p.add_argument("--start-date", help="Start date, YYYY-MM-DD. Required for --url-template and books export mode.")
    p.add_argument("--end-date", help="End date, YYYY-MM-DD. Required for --url-template and books export mode.")
    p.add_argument("--url-template", help="Official OKX URL template. Supports {date}, {yyyymmdd}, {symbol}, {kind}.")
    p.add_argument("--url", action="append", default=[], help="One official OKX download URL. Can be repeated multiple times.")
    p.add_argument("--manifest", type=Path, help="Text/JSONL manifest containing official OKX download URLs.")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Default: data/okx/raw")
    p.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    p.add_argument("--dry-run", action="store_true", help="Print tasks without downloading. Books export mode still requests links.")
    p.add_argument("--timeout", type=int, default=60, help="File download timeout seconds.")
    p.add_argument("--retries", type=int, default=3, help="File download retries.")
    p.add_argument("--sleep-sec", type=float, default=2.0, help="Sleep between file downloads. Higher values reduce Windows port-exhaustion risk.")

    p.add_argument("--export-timeout", type=int, default=300, help="OKX export-link request timeout seconds.")
    p.add_argument("--export-retries", type=int, default=6, help="Retries for OKX export-link requests, especially HTTP 429.")
    p.add_argument("--export-backoff-sec", type=float, default=10.0, help="Base backoff seconds for OKX export-link retries.")
    p.add_argument("--export-sleep-sec", type=float, default=3.0, help="Sleep after each OKX export-link request.")
    p.add_argument("--chunk-days", type=int, default=1, help="Days per OKX export request for books mode. Default: 1.")
    p.add_argument("--books-depth", type=int, choices=[400, 5000], default=400, help="Requested OKX order book depth. Default: 400.")
    p.add_argument("--books-module", default="", help="Override OKX export module code. Default: 4 for depth 400, 5 for depth 5000.")
    p.add_argument("--inst-type", choices=["AUTO", "SWAP", "SPOT"], default="AUTO", help="OKX instrument type for books export. Default: AUTO.")
    p.add_argument("--inst-family", default="", help="OKX instrument family for SWAP books export. Default inferred from symbol, e.g. ETH-USDT-SWAP -> ETH-USDT.")
    p.add_argument("--date-aggr", choices=["daily", "monthly"], default="daily", help="OKX export date aggregation. Default: daily.")
    p.add_argument("--allow-missing-book-days", action="store_true", help="Do not fail if a requested books date range returns zero export links.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_out = args.out_root / args.kind / args.symbol / "download_manifest.jsonl"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[OKX-DOWNLOAD-START] kind=%s symbol=%s out=%s", args.kind, args.symbol, manifest_out.parent)
    session = build_http_session()
    ok = skipped = failed = total = 0
    with manifest_out.open("a", encoding="utf-8") as mf:
        for task in build_tasks(args):
            total += 1
            if args.dry_run:
                print(f"DRY-RUN {task.url} -> {task.output_path}")
                continue
            result = download_one(task, overwrite=bool(args.overwrite), timeout=int(args.timeout), retries=int(args.retries), session=session)
            if result["status"] == "downloaded":
                ok += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
            mf.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            mf.flush()
            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

    if total <= 0:
        raise SystemExit(
            "No download tasks generated. Provide --url, --manifest, --url-template, "
            "or use --kind books with --start-date/--end-date for OKX export mode."
        )
    logger.info("[OKX-DOWNLOAD-DONE] tasks=%d downloaded=%d skipped=%d failed=%d manifest=%s", total, ok, skipped, failed, manifest_out)
    return 1 if failed else 0


def build_tasks(args: argparse.Namespace) -> Iterable[DownloadTask]:
    out_dir = args.out_root / args.kind / args.symbol

    for idx, url in enumerate(args.url or [], start=1):
        clean_url = clean_url_arg(url)
        if clean_url:
            yield DownloadTask(
                url=clean_url,
                output_path=out_dir / filename_from_url(clean_url, fallback=f"{args.symbol}_{args.kind}_url_{idx:06d}.dat"),
            )

    if args.manifest:
        for idx, url in enumerate(read_manifest_urls(args.manifest), start=1):
            yield DownloadTask(url=clean_url_arg(url), output_path=out_dir / filename_from_url(clean_url_arg(url), fallback=f"{args.symbol}_{args.kind}_manifest_{idx:06d}.dat"))

    if args.url_template:
        if not args.start_date or not args.end_date:
            raise SystemExit("--start-date and --end-date are required with --url-template")
        url_template = clean_url_arg(args.url_template)
        for d in date_range(parse_date(args.start_date), parse_date(args.end_date)):
            url = url_template.format(
                date=d.isoformat(),
                yyyymmdd=d.strftime("%Y%m%d"),
                symbol=args.symbol,
                kind=args.kind,
            )
            yield DownloadTask(url=url, output_path=out_dir / filename_from_url(url, fallback=f"{args.symbol}_{args.kind}_{d.isoformat()}.dat"), date_tag=d.isoformat())

    if should_use_books_export_mode(args):
        yield from build_books_export_tasks(args, out_dir=out_dir)


def should_use_books_export_mode(args: argparse.Namespace) -> bool:
    return args.kind in BOOK_KINDS and not args.url and not args.manifest and not args.url_template


def build_books_export_tasks(args: argparse.Namespace, out_dir: Path) -> Iterable[DownloadTask]:
    if not args.start_date or not args.end_date:
        raise SystemExit("--start-date and --end-date are required for --kind books export mode")

    inst_type, inst_selector = infer_export_instrument(args)
    module = str(args.books_module or (OKX_BOOK_MODULE_400 if int(args.books_depth) <= 400 else OKX_BOOK_MODULE_5000))
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")

    chunk_days = max(1, int(args.chunk_days))
    total_links = 0
    missing_ranges: list[str] = []
    for chunk_start, chunk_end in date_chunks(start, end, chunk_days=chunk_days):
        begin_ms = date_start_ms(chunk_start)
        end_ms = date_start_ms(chunk_end)
        date_tag = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
        logger.info(
            "[OKX-BOOKS-EXPORT-LINKS] symbol=%s inst_type=%s selector=%s module=%s range=%s begin=%s end=%s",
            args.symbol,
            inst_type,
            inst_selector,
            module,
            date_tag,
            begin_ms,
            end_ms,
        )
        response = request_okx_export_links(
            module=module,
            inst_type=inst_type,
            inst_selector=inst_selector,
            begin_ms=str(begin_ms),
            end_ms=str(end_ms),
            date_aggr=str(args.date_aggr),
            timeout=int(args.export_timeout),
            retries=int(args.export_retries),
            backoff_sec=float(args.export_backoff_sec),
        )
        raw_items = extract_download_items(response)
        items = filter_download_items_by_date(raw_items, start=chunk_start, end=chunk_end)
        if len(items) < len(raw_items):
            logger.info(
                "[OKX-BOOKS-EXPORT-FILTERED] range=%s kept=%d dropped=%d",
                date_tag,
                len(items),
                len(raw_items) - len(items),
            )
        if not items:
            missing_ranges.append(date_tag)
            logger.warning("[OKX-BOOKS-EXPORT-EMPTY] range=%s response_keys=%s", date_tag, sorted(response.keys()))
        else:
            for idx, item in enumerate(items, start=1):
                total_links += 1
                url = item["url"]
                fallback = f"{args.symbol}_{args.kind}_{date_tag}_{idx:04d}.dat"
                file_name = item.get("file_name") or filename_from_url(url, fallback=fallback)
                yield DownloadTask(url=url, output_path=out_dir / safe_output_filename(file_name, fallback=fallback), date_tag=date_tag)

        if args.export_sleep_sec > 0:
            time.sleep(float(args.export_sleep_sec))

    if missing_ranges and not args.allow_missing_book_days:
        raise SystemExit(
            "OKX books export returned zero download links for requested range(s): "
            + ", ".join(missing_ranges)
            + ". If this is expected, rerun with --allow-missing-book-days."
        )

    if total_links <= 0:
        raise SystemExit(
            "OKX books export generated zero download links. This can mean the requested date/instrument has no data, "
            "or OKX did not make links available through this request. Export links from the OKX website and pass them "
            "with --url or --manifest."
        )


def infer_export_instrument(args: argparse.Namespace) -> tuple[str, dict[str, list[str]]]:
    symbol = str(args.symbol)
    if args.inst_type == "AUTO":
        inst_type = "SWAP" if symbol.endswith("-SWAP") else "SPOT"
    else:
        inst_type = str(args.inst_type)

    if inst_type == "SWAP":
        inst_family = str(args.inst_family or (symbol[:-5] if symbol.endswith("-SWAP") else symbol))
        return inst_type, {"instFamilyList": [inst_family]}
    return inst_type, {"instIdList": [symbol]}


def request_okx_export_links(
    module: str,
    inst_type: str,
    inst_selector: Mapping[str, list[str]],
    begin_ms: str,
    end_ms: str,
    date_aggr: str,
    timeout: int,
    retries: int,
    backoff_sec: float,
) -> dict[str, Any]:
    body = {
        "module": str(module),
        "instType": str(inst_type),
        "instQueryParam": dict(inst_selector),
        "dateQuery": {
            "dateAggrType": str(date_aggr),
            "begin": str(begin_ms),
            "end": str(end_ms),
        },
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.okx.com",
        "Referer": OKX_HISTORICAL_DATA_REFERER,
    }
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    last_error = ""
    max_attempts = max(1, int(retries))
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(OKX_EXPORT_ENDPOINT, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
            result = json.loads(payload)
            if str(result.get("code")) != "0":
                if str(result.get("code")) == OKX_TOO_MANY_REQUESTS_CODE and attempt < max_attempts:
                    sleep_sec = export_retry_sleep(attempt, backoff_sec, retry_after=None)
                    logger.warning(
                        "[OKX-BOOKS-EXPORT-RETRY] attempt=%d/%d code=%s msg=%s sleep=%.1fs",
                        attempt,
                        max_attempts,
                        result.get("code"),
                        result.get("msg"),
                        sleep_sec,
                    )
                    time.sleep(sleep_sec)
                    continue
                raise RuntimeError(f"OKX export request returned error: {result}")
            data_obj = result.get("data")
            if isinstance(data_obj, dict):
                return data_obj
            return {"data": data_obj}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code in RETRYABLE_HTTP_CODES and attempt < max_attempts:
                retry_after = parse_retry_after(exc.headers.get("Retry-After"))
                sleep_sec = export_retry_sleep(attempt, backoff_sec, retry_after=retry_after)
                logger.warning(
                    "[OKX-BOOKS-EXPORT-RETRY] attempt=%d/%d http=%s sleep=%.1fs detail=%s",
                    attempt,
                    max_attempts,
                    exc.code,
                    sleep_sec,
                    detail,
                )
                time.sleep(sleep_sec)
                continue
            raise RuntimeError(f"OKX export request failed {last_error}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            if attempt < max_attempts:
                sleep_sec = export_retry_sleep(attempt, backoff_sec, retry_after=None)
                logger.warning(
                    "[OKX-BOOKS-EXPORT-RETRY] attempt=%d/%d sleep=%.1fs error=%s",
                    attempt,
                    max_attempts,
                    sleep_sec,
                    last_error,
                )
                time.sleep(sleep_sec)
                continue
            raise RuntimeError(f"OKX export request failed: {last_error}") from exc
    raise RuntimeError(f"OKX export request failed after {max_attempts} attempts: {last_error}")


def export_retry_sleep(attempt: int, backoff_sec: float, retry_after: float | None) -> float:
    if retry_after is not None and retry_after > 0:
        return min(float(retry_after), 300.0)
    return min(max(float(backoff_sec), 1.0) * (2 ** max(0, attempt - 1)), 300.0)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


def extract_download_items(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(url: str, file_name: str = "") -> None:
        clean_url = str(url or "").strip()
        if not clean_url.startswith(("http://", "https://")) or clean_url in seen:
            return
        seen.add(clean_url)
        items.append({"url": clean_url, "file_name": str(file_name or "").strip()})

    details = payload.get("details") if isinstance(payload, Mapping) else None
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            group_details = detail.get("groupDetails") or detail.get("group_details") or []
            if isinstance(group_details, list):
                for group_detail in group_details:
                    if not isinstance(group_detail, Mapping):
                        continue
                    url = group_detail.get("url") or group_detail.get("downloadUrl") or group_detail.get("download_url")
                    file_name = group_detail.get("fileName") or group_detail.get("filename") or group_detail.get("name") or ""
                    if url:
                        add(str(url), str(file_name or ""))

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


def filter_download_items_by_date(items: Sequence[Mapping[str, str]], start: date, end: date) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for item in items:
        item_date = infer_download_item_date(item)
        if item_date is None:
            logger.warning("[OKX-BOOKS-EXPORT-DATE-UNKNOWN] url=%s file_name=%s", item.get("url", ""), item.get("file_name", ""))
            filtered.append({"url": str(item.get("url", "")), "file_name": str(item.get("file_name", ""))})
            continue
        if start <= item_date <= end:
            filtered.append({"url": str(item.get("url", "")), "file_name": str(item.get("file_name", ""))})
    return filtered


def infer_download_item_date(item: Mapping[str, str]) -> date | None:
    text = " ".join([str(item.get("file_name", "")), str(item.get("url", ""))])
    for pattern in (ISO_DATE_RE, COMPACT_DATE_RE):
        for match in pattern.finditer(text):
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                continue
    return None


def build_http_session():
    """Build a persistent HTTP session to avoid creating one TCP connection per file.

    On Windows, rapidly opening thousands of short-lived HTTPS connections can
    exhaust local ephemeral ports and raise WinError 10048.  requests.Session
    reuses one keep-alive connection where the server allows it and is much more stable for long date ranges.
    """
    if requests is None:
        logger.warning("requests is not installed; falling back to urllib without connection pooling")
        return None
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "CoinBacktest/okx-historical-downloader", "Connection": "keep-alive"})
    return session


def download_one(task: DownloadTask, overwrite: bool, timeout: int, retries: int, session=None) -> dict[str, object]:
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    url = clean_url_arg(task.url)
    base = {
        "url": url,
        "output_path": str(task.output_path),
        "date_tag": task.date_tag,
        "downloaded_at_utc": utc_now(),
    }
    if task.output_path.exists() and task.output_path.stat().st_size > 0 and not overwrite:
        return {**base, "status": "skipped", "size_bytes": task.output_path.stat().st_size, "sha256": sha256_file(task.output_path)}

    tmp = task.output_path.with_suffix(task.output_path.suffix + ".part")
    last_error = ""
    max_attempts = max(1, int(retries))
    for attempt in range(1, max_attempts + 1):
        try:
            if session is not None:
                with session.get(url, stream=True, timeout=timeout) as resp:
                    resp.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_content(1024 * 1024):
                            if chunk:
                                f.write(chunk)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "CoinBacktest/okx-historical-downloader"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(task.output_path)
            return {**base, "status": "downloaded", "size_bytes": task.output_path.stat().st_size, "sha256": sha256_file(task.output_path), "attempts": attempt}
        except Exception as exc:
            last_error = repr(exc)
            logger.warning("[OKX-DOWNLOAD-RETRY] attempt=%d/%d url=%s error=%s", attempt, max_attempts, url, last_error)
            sleep_sec = download_retry_sleep(attempt, last_error)
            time.sleep(sleep_sec)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return {**base, "status": "failed", "error": last_error, "attempts": max_attempts}


def download_retry_sleep(attempt: int, error_text: str) -> float:
    """Backoff for downloads; wait longer for Windows local port exhaustion."""
    base = min(2 ** max(1, attempt), 60)
    if "10048" in error_text or "Only one usage of each socket address" in error_text:
        return max(base, 30.0)
    if "10054" in error_text or "ConnectionResetError" in error_text:
        return max(base, 10.0)
    if "timed out" in error_text.lower() or "timeout" in error_text.lower():
        return max(base, 10.0)
    return float(base)


def read_manifest_urls(path: Path) -> list[str]:
    urls: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                obj = json.loads(text)
                url = str(obj.get("url") or obj.get("download_url") or "").strip()
            else:
                url = text
            if url:
                urls.append(url)
    return urls


def clean_url_arg(value: str) -> str:
    """Strip accidental shell quotes, especially Windows CMD single quotes."""
    return str(value or "").strip().strip("'\"")


def filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def safe_output_filename(name: str, fallback: str) -> str:
    candidate = Path(str(name or fallback)).name.strip()
    if not candidate or candidate in {".", ".."}:
        candidate = fallback
    return candidate.replace("/", "_").replace("\\", "_")


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def date_chunks(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def date_start_ms(d: date) -> int:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
