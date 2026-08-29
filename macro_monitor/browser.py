from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any


class BrowserSession:
    """Own one reusable Chromium instance and one reusable browser context."""

    def __init__(self, *, headed: bool, verbose: bool, logger: logging.Logger) -> None:
        self.headed = headed
        self.verbose = verbose
        self.logger = logger
        self._playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self._pages: dict[str, Any] = {}

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required. Run: python -m pip install playwright && "
                "python -m playwright install chromium"
            ) from exc
        self._playwright = await async_playwright().start()
        launch_options: dict[str, Any] = {
            "headless": not self.headed,
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        executable = self._windows_executable()
        if executable:
            launch_options["executable_path"] = executable
            if self.verbose:
                mode = "headed" if self.headed else "headless"
                self.logger.debug("[browser] mode=%s executable=%s", mode, executable)
        try:
            self.browser = await self._playwright.chromium.launch(
                **launch_options,
            )
        except Exception as exc:
            try:
                await self._playwright.stop()
            finally:
                self._playwright = None
            if "Executable doesn't exist" in str(exc):
                raise RuntimeError(
                    "Chromium executable was not found. On Windows install Chrome/Edge, or run: "
                    "python -m playwright install chromium"
                ) from exc
            raise
        self.context = await self.browser.new_context(
            locale="en-US",
            viewport={"width": 1440, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )
        self.context.set_default_timeout(12_000)
        self.context.set_default_navigation_timeout(60_000)
        self.logger.info("[fedwatch] browser ready mode=%s", "headed" if self.headed else "headless")

    def _windows_executable(self) -> str | None:
        """Use an installed Windows browser for both headed and headless runs.

        This keeps headless collection working even when Playwright's optional
        chrome-headless-shell package has not been downloaded in the active
        Python environment.
        """
        if os.name != "nt":
            return None
        override = os.getenv("MACRO_BROWSER_EXECUTABLE", "").strip()
        candidates = (
            override,
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        )
        return next((value for value in candidates if value and Path(value).is_file()), None)

    async def page(self, name: str) -> Any:
        existing = self._pages.get(name)
        if existing is not None and not existing.is_closed():
            return existing
        page = await self.context.new_page()
        page.on("popup", lambda popup: asyncio.create_task(self._close_popup(popup)))
        self._pages[name] = page
        return page

    async def reset_page(self, name: str) -> None:
        page = self._pages.pop(name, None)
        if page is not None and not page.is_closed():
            try:
                await page.close()
            except Exception:
                pass

    async def _close_popup(self, page: Any) -> None:
        try:
            await page.close()
            if self.verbose:
                self.logger.debug("[browser] closed popup page")
        except Exception:
            pass

    async def dismiss_overlays(self, page: Any) -> None:
        selectors = (
            "#onetrust-accept-btn-handler",
            "button#onetrust-accept-btn-handler",
            "button[data-test='accept-all-cookies']",
            "button:has-text('Accept All')",
            "button:has-text('I Accept')",
            "button:has-text('接受全部')",
            "button[aria-label='Close']",
            "button[aria-label='关闭']",
            "[data-test='sign-up-dialog-close-button']",
        )
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if await locator.is_visible(timeout=350):
                    await locator.click(timeout=1_500)
            except Exception:
                continue

    async def close(self) -> None:
        if self.context is not None:
            try:
                await self.context.close()
            except Exception as exc:
                if self.verbose:
                    self.logger.debug("[browser] context already closed: %s", exc)
            self.context = None
        if self.browser is not None:
            try:
                await self.browser.close()
            except Exception as exc:
                if self.verbose:
                    self.logger.debug("[browser] browser already closed: %s", exc)
            self.browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                if self.verbose:
                    self.logger.debug("[browser] Playwright already stopped: %s", exc)
            self._playwright = None
