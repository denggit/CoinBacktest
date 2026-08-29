from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from .browser import BrowserSession
from .models import DxySnapshot, FedWatchSnapshot, TreasurySnapshot, utc_now_iso
from .parsers import parse_date_text, parse_fedwatch_html, parse_numeric_index, parse_numeric_yield


class SourceUnavailable(RuntimeError):
    def __init__(self, source: str, detail: str) -> None:
        super().__init__(detail)
        self.source = source
        self.detail = detail


@dataclass(frozen=True)
class FedWatchEndpoint:
    name: str
    url: str


FEDWATCH_ENDPOINTS = (
    FedWatchEndpoint("cme_fedwatch_en", "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"),
    FedWatchEndpoint("cme_fedwatch_cn", "https://www.cmegroup.cn/fed-watch/"),
)

TREASURY_ENDPOINTS = {
    "2y": (
        ("investing_us2y", "https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield"),
        ("cnbc_us2y", "https://www.cnbc.com/quotes/US2Y"),
        ("marketwatch_us2y", "https://www.marketwatch.com/investing/bond/tmubmusd02y?countrycode=bx"),
        ("tradingview_us2y", "https://www.tradingview.com/symbols/TVC-US02Y/"),
    ),
    "10y": (
        ("investing_us10y", "https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield"),
        ("cnbc_us10y", "https://www.cnbc.com/quotes/US10Y"),
        ("marketwatch_us10y", "https://www.marketwatch.com/investing/bond/tmubmusd10y?countrycode=bx"),
        ("tradingview_us10y", "https://www.tradingview.com/symbols/TVC-US10Y/"),
    ),
}

DXY_ENDPOINTS = (
    ("cnbc_dxy", "https://www.cnbc.com/quotes/.DXY"),
    ("investing_dxy", "https://www.investing.com/indices/usdollar"),
    ("marketwatch_dxy", "https://www.marketwatch.com/investing/index/dxy"),
    ("tradingview_dxy", "https://www.tradingview.com/symbols/TVC-DXY/"),
)


class FedWatchSource:
    def __init__(self, browser: BrowserSession, logger: logging.Logger, retries: int = 3) -> None:
        self.browser = browser
        self.logger = logger
        self.retries = retries
        self._preferred_source: str | None = None

    async def fetch(self) -> FedWatchSnapshot:
        errors: list[str] = []
        endpoints = sorted(FEDWATCH_ENDPOINTS, key=lambda item: item.name != self._preferred_source) if self._preferred_source else FEDWATCH_ENDPOINTS
        for endpoint in endpoints:
            for attempt in range(1, self.retries + 1):
                try:
                    result = await self._fetch_endpoint(endpoint)
                    self._preferred_source = endpoint.name
                    return result
                except Exception as exc:
                    await self.browser.reset_page("fedwatch")
                    errors.append(f"{endpoint.name} attempt {attempt}: {exc}")
                    self.logger.warning(
                        "[fedwatch] source unavailable source=%s attempt=%d/%d error=%s",
                        endpoint.name,
                        attempt,
                        self.retries,
                        exc,
                    )
                    if attempt < self.retries:
                        await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise SourceUnavailable("cme_fedwatch", "; ".join(errors[-4:]))

    async def _fetch_endpoint(self, endpoint: FedWatchEndpoint) -> FedWatchSnapshot:
        page = await self.browser.page("fedwatch")
        await page.goto(endpoint.url, wait_until="domcontentloaded")
        await self.browser.dismiss_overlays(page)
        try:
            await page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight, 900))")
        except Exception:
            pass
        frame = await self._wait_for_quikstrike(page)
        await self._select_nearest_meeting(frame)
        html = await frame.content()
        snapshot = parse_fedwatch_html(html, source=endpoint.name, timestamp_utc=utc_now_iso())
        total = sum(item.probability for item in snapshot.probabilities)
        if not 99.0 <= total <= 101.0:
            raise ValueError(f"FedWatch probabilities sum to {total:.2f}, expected approximately 100")
        return snapshot

    async def _wait_for_quikstrike(self, page: Any) -> Any:
        deadline = asyncio.get_running_loop().time() + 50.0
        last_summary = ""
        while asyncio.get_running_loop().time() < deadline:
            await self.browser.dismiss_overlays(page)
            for frame in page.frames:
                try:
                    info = await frame.evaluate(
                        """() => ({
                            tabs: document.querySelectorAll('a[id*="lbMeeting"]').length,
                            rates: Array.from(document.querySelectorAll('table')).some(
                                t => /Target Rate|目标利率/i.test(t.innerText)
                            ),
                            text: (document.body && document.body.innerText || '').slice(0, 120)
                        })"""
                    )
                    if info["tabs"] > 0 or info["rates"]:
                        return frame
                    last_summary = f"url={frame.url} text={info['text']!r}"
                except Exception:
                    continue
            await asyncio.sleep(0.5)
        raise TimeoutError(f"QuikStrike iframe did not render ({last_summary})")

    async def _select_nearest_meeting(self, frame: Any) -> None:
        tabs = await frame.eval_on_selector_all(
            'a[id*="lbMeeting"]',
            "els => els.map(el => ({id: el.id, text: (el.textContent || '').trim()}))",
        )
        if not tabs:
            return
        today = date.today()
        dated: list[tuple[str, dict[str, str]]] = []
        for tab in tabs:
            parsed = parse_date_text(tab.get("text", ""), today=today)
            if parsed:
                dated.append((parsed, tab))
        tab = min((item for item in dated if item[0] >= today.isoformat()), default=dated[0] if dated else ("", tabs[0]))[1]
        before = await self._current_meeting_text(frame)
        await frame.locator(f"#{tab['id']}").click(timeout=10_000)
        try:
            await frame.wait_for_function(
                """previous => {
                    const tables = Array.from(document.querySelectorAll('table'));
                    const info = tables.find(t => /Meeting Date|会议日期/i.test(t.innerText));
                    const loading = document.querySelector('.throbber, [class*="loading"]');
                    if (loading && loading.offsetParent !== null) return false;
                    return !previous || (info && info.innerText !== previous);
                }""",
                arg=before,
                timeout=15_000,
            )
        except Exception:
            # A click on the already-selected nearest tab legitimately changes nothing.
            await asyncio.sleep(0.5)

    async def _current_meeting_text(self, frame: Any) -> str:
        return await frame.evaluate(
            """() => {
                const table = Array.from(document.querySelectorAll('table')).find(
                    t => /Meeting Date|会议日期/i.test(t.innerText)
                );
                return table ? table.innerText : '';
            }"""
        )


class TreasurySource:
    def __init__(self, browser: BrowserSession, logger: logging.Logger, retries: int = 3) -> None:
        self.browser = browser
        self.logger = logger
        self.retries = retries
        self._preferred_sources: dict[str, str] = {}

    async def fetch(self) -> TreasurySnapshot:
        results = await asyncio.gather(self._fetch_term("2y"), self._fetch_term("10y"), return_exceptions=True)
        failures = [item for item in results if isinstance(item, Exception)]
        if len(failures) == 2:
            detail = "; ".join(str(item) for item in failures)
            raise SourceUnavailable("treasury_yields", detail)
        if isinstance(results[0], Exception):
            value_2y, source_2y = None, getattr(results[0], "source", "us2y_yield")
        else:
            value_2y, source_2y = results[0]
        if isinstance(results[1], Exception):
            value_10y, source_10y = None, getattr(results[1], "source", "us10y_yield")
        else:
            value_10y, source_10y = results[1]
        return TreasurySnapshot(utc_now_iso(), source_2y, source_10y, value_2y, value_10y)

    async def _fetch_term(self, term: str) -> tuple[float, str]:
        errors: list[str] = []
        preferred = self._preferred_sources.get(term)
        endpoints = sorted(TREASURY_ENDPOINTS[term], key=lambda item: item[0] != preferred) if preferred else TREASURY_ENDPOINTS[term]
        for source, url in endpoints:
            for attempt in range(1, self.retries + 1):
                try:
                    page = await self.browser.page(f"treasury_{term}")
                    if page.url != url:
                        await page.goto(url, wait_until="domcontentloaded")
                    await self.browser.dismiss_overlays(page)
                    value = await self._read_price(page)
                    self._preferred_sources[term] = source
                    return value, source
                except Exception as exc:
                    errors.append(f"{source} attempt {attempt}: {exc}")
                    self.logger.warning(
                        "[treasury] source unavailable term=%s source=%s attempt=%d/%d error=%s",
                        term.upper(),
                        source,
                        attempt,
                        self.retries,
                        exc,
                    )
                    if attempt < self.retries:
                        await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise SourceUnavailable(f"us{term}_yield", "; ".join(errors[-4:]))

    async def _read_price(self, page: Any) -> float:
        selectors = (
            '[data-test="instrument-price-last"]',
            '#last_last',
            '.instrument-price-last',
            '[class*="instrument-price_last"]',
            '[class*="QuoteStrip-lastPrice"]',
            '[class*="Summary-last"]',
            '.intraday__price .value',
            '[class*="intraday__price"] [class*="value"]',
            '[class*="last-JWoJqCpY"]',
        )
        deadline = asyncio.get_running_loop().time() + 25.0
        last_values: list[str] = []
        while asyncio.get_running_loop().time() < deadline:
            try:
                body_now = (await page.locator("body").inner_text(timeout=700)).strip()
                if body_now in {"403", "401"} or "Access Denied" in body_now[:500]:
                    raise ValueError(f"page blocked with {body_now[:80]!r}")
            except ValueError:
                raise
            except Exception:
                pass
            for selector in selectors:
                try:
                    locator = page.locator(selector).first
                    text = (await locator.text_content(timeout=600) or "").strip()
                    if text:
                        last_values.append(text)
                        return parse_numeric_yield(text)
                except Exception:
                    continue
            await asyncio.sleep(0.35)
        try:
            title = await page.title()
            body = (await page.locator("body").inner_text(timeout=2_000)).replace("\n", " ")[:240]
        except Exception as exc:
            title, body = "", f"diagnostic unavailable: {exc}"
        raise ValueError(
            f"yield DOM value not found; url={page.url!r} title={title!r} "
            f"body={body!r} candidates={last_values[-3:]}"
        )


class DxySource:
    def __init__(self, browser: BrowserSession, logger: logging.Logger, retries: int = 3) -> None:
        self.browser = browser
        self.logger = logger
        self.retries = retries
        self._preferred_source: str | None = None

    async def fetch(self) -> DxySnapshot:
        errors: list[str] = []
        endpoints = (
            sorted(DXY_ENDPOINTS, key=lambda item: item[0] != self._preferred_source)
            if self._preferred_source
            else DXY_ENDPOINTS
        )
        for source, url in endpoints:
            for attempt in range(1, self.retries + 1):
                try:
                    page = await self.browser.page("dxy")
                    if page.url != url:
                        await page.goto(url, wait_until="domcontentloaded")
                    await self.browser.dismiss_overlays(page)
                    value = await self._read_price(page)
                    self._preferred_source = source
                    return DxySnapshot(utc_now_iso(), source, value)
                except Exception as exc:
                    errors.append(f"{source} attempt {attempt}: {exc}")
                    self.logger.warning(
                        "[dxy] source unavailable source=%s attempt=%d/%d error=%s",
                        source,
                        attempt,
                        self.retries,
                        exc,
                    )
                    if attempt < self.retries:
                        await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise SourceUnavailable("dxy_index", "; ".join(errors[-4:]))

    async def _read_price(self, page: Any) -> float:
        selectors = (
            '[data-test="instrument-price-last"]',
            '#last_last',
            '.instrument-price-last',
            '[class*="instrument-price_last"]',
            '[class*="QuoteStrip-lastPrice"]',
            '[class*="Summary-last"]',
            '.intraday__price .value',
            '[class*="intraday__price"] [class*="value"]',
            '[class*="last-JWoJqCpY"]',
        )
        deadline = asyncio.get_running_loop().time() + 25.0
        last_values: list[str] = []
        while asyncio.get_running_loop().time() < deadline:
            try:
                body_now = (await page.locator("body").inner_text(timeout=700)).strip()
                if body_now in {"403", "401"} or "Access Denied" in body_now[:500]:
                    raise ValueError(f"page blocked with {body_now[:80]!r}")
            except ValueError:
                raise
            except Exception:
                pass
            for selector in selectors:
                try:
                    locator = page.locator(selector).first
                    text = (await locator.text_content(timeout=600) or "").strip()
                    if text:
                        last_values.append(text)
                        return parse_numeric_index(text)
                except Exception:
                    continue
            await asyncio.sleep(0.35)
        try:
            title = await page.title()
            body = (await page.locator("body").inner_text(timeout=2_000)).replace("\n", " ")[:240]
        except Exception as exc:
            title, body = "", f"diagnostic unavailable: {exc}"
        raise ValueError(
            f"DXY DOM value not found; url={page.url!r} title={title!r} "
            f"body={body!r} candidates={last_values[-3:]}"
        )
