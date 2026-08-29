from __future__ import annotations

import html as html_module
import re
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Iterable

from .models import FedWatchSnapshot, TargetProbability, utc_now_iso


_SPACE_RE = re.compile(r"\s+")
_RANGE_RE = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d{1,3})?)\s*[-–—]\s*(\d{1,4}(?:\.\d{1,3})?)(?!\d)")
_PCT_RE = re.compile(r"(?<!\d)(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table" and self._table is None:
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if not value:
            return
        self.text_parts.append(value)
        if self._cell_parts is not None:
            self._cell_parts.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_parts is not None and self._row is not None:
            self._row.append(_clean(" ".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None

    @property
    def text(self) -> str:
        return "\n".join(self.text_parts)


class _PriceParser(HTMLParser):
    SELECTOR_VALUES = {"instrument-price-last", "last_last"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture_depth = 0
        self._parts: list[str] = []
        self.candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): (v or "") for k, v in attrs}
        data_test = attr.get("data-test", "").lower()
        element_id = attr.get("id", "").lower()
        classes = attr.get("class", "").lower()
        if self._capture_depth:
            self._capture_depth += 1
        elif data_test == "instrument-price-last" or element_id == "last_last" or "instrument-price-last" in classes:
            self._capture_depth = 1
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if not self._capture_depth:
            return
        self._capture_depth -= 1
        if self._capture_depth == 0:
            value = _clean(" ".join(self._parts))
            if value:
                self.candidates.append(value)
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_depth and data.strip():
            self._parts.append(data.strip())


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", html_module.unescape(value)).strip()


def _bounds(value: str) -> tuple[float, float] | None:
    match = _RANGE_RE.search(value)
    if not match:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    if low < 25 and high < 25:
        return low * 100.0, high * 100.0
    return low, high


def normalize_target_range(value: str) -> str:
    bounds = _bounds(value)
    if bounds is None:
        raise ValueError(f"Invalid target-rate range: {value!r}")
    return f"{bounds[0] / 100:.2f}-{bounds[1] / 100:.2f}"


def _parse_probability(value: str) -> float | None:
    match = _PCT_RE.search(value)
    return float(match.group(1)) if match else None


def _summary_from_tables(tables: Iterable[list[list[str]]]) -> tuple[float | None, float | None, float | None]:
    for table in tables:
        for index, row in enumerate(table):
            upper = [cell.upper() for cell in row]
            joined = " ".join(upper)
            if ("EASE" in joined or "降息" in joined) and ("NO CHANGE" in joined or "不变" in joined):
                values = table[index + 1] if index + 1 < len(table) else []
                parsed = [_parse_probability(cell) for cell in values]
                parsed = [x for x in parsed if x is not None]
                if len(parsed) >= 3:
                    return parsed[0], parsed[1], parsed[2]
            labels: dict[str, float] = {}
            for item in row:
                pct = _parse_probability(item)
                key = item.upper()
                if pct is None:
                    continue
                if "EASE" in key or "降息" in key:
                    labels["cut"] = pct
                elif "NO CHANGE" in key or "不变" in key:
                    labels["hold"] = pct
                elif "HIKE" in key or "加息" in key:
                    labels["hike"] = pct
            if "cut" in labels and "hold" in labels:
                return labels.get("cut"), labels.get("hold"), labels.get("hike")
    return None, None, None


def _extract_distribution(tables: Iterable[list[list[str]]], text: str) -> tuple[TargetProbability, ...]:
    found: dict[str, float] = {}
    for table in tables:
        table_text = " ".join(cell for row in table for cell in row).upper()
        if "TARGET RATE" not in table_text and "目标利率" not in table_text:
            continue
        for row in table:
            range_index = next((i for i, cell in enumerate(row) if _RANGE_RE.search(cell)), None)
            if range_index is None:
                continue
            probability = next((_parse_probability(cell) for cell in row[range_index + 1:] if _parse_probability(cell) is not None), None)
            if probability is not None:
                found[normalize_target_range(row[range_index])] = probability

    if not found:
        for line in text.splitlines():
            range_match = _RANGE_RE.search(line)
            if not range_match:
                continue
            probability = _parse_probability(line[range_match.end():])
            if probability is not None:
                found[normalize_target_range(range_match.group(0))] = probability
    return tuple(TargetProbability(key, value) for key, value in sorted(found.items(), key=lambda x: _bounds(x[0]) or (0, 0)))


def parse_date_text(value: str, *, today: date | None = None) -> str | None:
    today = today or date.today()
    candidates: list[date] = []
    patterns = (
        (r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", lambda m: date(int(m[3]), int(m[1]), int(m[2]))),
    )
    for pattern, builder in patterns:
        for match in re.finditer(pattern, value):
            try:
                candidates.append(builder(match))
            except ValueError:
                pass

    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
        "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    for match in re.finditer(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s*,?\s*(20\d{2})\b", value):
        month = month_names.get(match.group(2).lower())
        if month:
            try:
                candidates.append(date(int(match.group(3)), month, int(match.group(1))))
            except ValueError:
                pass
    for match in re.finditer(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d{2})\b", value):
        month = month_names.get(match.group(1).lower())
        if month:
            try:
                candidates.append(date(int(match.group(3)), month, int(match.group(2))))
            except ValueError:
                pass
    for match in re.finditer(r"\b(\d{1,2})\s*(\d{1,2})月\s*(20\d{2})\b", value):
        try:
            candidates.append(date(int(match.group(3)), int(match.group(2)), int(match.group(1))))
        except ValueError:
            pass

    if not candidates:
        return None
    future = sorted({item for item in candidates if item >= today})
    return (future[0] if future else max(candidates)).isoformat()


def _meeting_from_tables(tables: Iterable[list[list[str]]], text: str, today: date | None) -> str | None:
    for table in tables:
        for row_index, row in enumerate(table):
            for cell_index, cell in enumerate(row):
                upper = cell.upper()
                if "MEETING DATE" not in upper and "会议日期" not in cell:
                    continue
                for candidate in row[cell_index + 1:]:
                    parsed = parse_date_text(candidate, today=today)
                    if parsed:
                        return parsed
                if row_index + 1 < len(table) and cell_index < len(table[row_index + 1]):
                    parsed = parse_date_text(table[row_index + 1][cell_index], today=today)
                    if parsed:
                        return parsed
    return parse_date_text(text, today=today)


def _current_target(text: str) -> str | None:
    patterns = (
        r"current\s+target\s+rate(?:\s+is)?[^\d]{0,20}(\d{1,4}(?:\.\d+)?)\s*[-–—]\s*(\d{1,4}(?:\.\d+)?)",
        r"当前目标利率(?:为)?[^\d]{0,20}(\d{1,4}(?:\.\d+)?)\s*[-–—]\s*(\d{1,4}(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_target_range(match.group(0))
    return None


def _derive_summary(distribution: tuple[TargetProbability, ...], current_target: str | None) -> tuple[float | None, float | None, float | None]:
    if not distribution or not current_target:
        return None, None, None
    current = _bounds(current_target)
    if current is None:
        return None, None, None
    cut = hold = hike = 0.0
    for item in distribution:
        bounds = _bounds(item.target_range)
        if bounds is None:
            continue
        if bounds[1] <= current[0]:
            cut += item.probability
        elif bounds == current or (bounds[0] < current[1] and bounds[1] > current[0]):
            hold += item.probability
        else:
            hike += item.probability
    return cut, hold, hike


def parse_fedwatch_html(
    html: str,
    *,
    source: str = "cme_fedwatch",
    timestamp_utc: str | None = None,
    today: date | None = None,
) -> FedWatchSnapshot:
    parser = _TableParser()
    parser.feed(html)
    meeting_date = _meeting_from_tables(parser.tables, parser.text, today)
    distribution = _extract_distribution(parser.tables, parser.text)
    if not meeting_date:
        raise ValueError("FedWatch meeting date not found")
    if not distribution:
        raise ValueError("FedWatch target-rate probability table not found")
    current_target = _current_target(parser.text)
    cut, hold, hike = _summary_from_tables(parser.tables)
    if cut is None or hold is None:
        cut, hold, hike = _derive_summary(distribution, current_target)
    return FedWatchSnapshot(
        timestamp_utc=timestamp_utc or utc_now_iso(),
        source=source,
        meeting_date=meeting_date,
        probabilities=distribution,
        cut_probability=cut,
        hold_probability=hold,
        hike_probability=hike,
        current_target_range=current_target,
    )


def parse_numeric_yield(value: str) -> float:
    match = _NUMBER_RE.search(value.replace("\u2212", "-"))
    if not match:
        raise ValueError(f"Treasury yield value not found in {value!r}")
    number = float(match.group(0).replace(",", ""))
    if not 0.0 <= number <= 25.0:
        raise ValueError(f"Treasury yield outside plausible percent range: {number}")
    return number


def parse_numeric_index(value: str) -> float:
    match = _NUMBER_RE.search(value.replace("\u2212", "-"))
    if not match:
        raise ValueError(f"DXY value not found in {value!r}")
    number = float(match.group(0).replace(",", ""))
    if not 40.0 <= number <= 200.0:
        raise ValueError(f"DXY outside plausible index range: {number}")
    return number


def parse_treasury_yield_html(html: str, *, term: str) -> float:
    parser = _PriceParser()
    parser.feed(html)
    for candidate in parser.candidates:
        try:
            return parse_numeric_yield(candidate)
        except ValueError:
            continue
    plain = _clean(re.sub(r"<[^>]+>", " ", html))
    term_pattern = "2" if str(term).lower() in {"2", "2y", "us2y"} else "10"
    contextual = re.search(
        rf"(?:U\.?S\.?\s*)?{term_pattern}[ -]Year\s+(?:Bond\s+)?Yield.{{0,180}}?(\d{{1,2}}\.\d{{2,4}})\s*%?",
        plain,
        re.IGNORECASE,
    )
    if contextual:
        return parse_numeric_yield(contextual.group(1))
    raise ValueError(f"US{term_pattern}Y yield not found")


def parse_dxy_index_html(html: str) -> float:
    parser = _PriceParser()
    parser.feed(html)
    for candidate in parser.candidates:
        try:
            return parse_numeric_index(candidate)
        except ValueError:
            continue
    plain = _clean(re.sub(r"<[^>]+>", " ", html))
    contextual = re.search(
        r"(?:U\.?S\.?\s+)?Dollar\s+Index.{0,180}?((?:[4-9]\d|1\d{2})(?:\.\d{1,4})?)",
        plain,
        re.IGNORECASE,
    )
    if contextual:
        return parse_numeric_index(contextual.group(1))
    raise ValueError("DXY index value not found")
