#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dependency-free progress helpers for CoinBacktest research scripts.

The goal is to keep long-running research readable without requiring tqdm.
Use this from research entrypoints and reusable research modules instead of
copying bespoke progress-bar code into every strategy or lab script.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def format_seconds(seconds: float) -> str:
    """Format elapsed/ETA seconds as a compact human-readable string."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?:??"
    seconds_i = int(seconds)
    if seconds_i < 60:
        return f"{seconds_i}s"
    minutes, sec = divmod(seconds_i, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


@dataclass
class ProgressReporter:
    """Small progress reporter that works in terminals and redirected logs.

    Parameters
    ----------
    label:
        Prefix printed before the progress bar.
    total:
        Total number of units. If <= 0, updates are suppressed.
    every:
        Emit an update every N completed units. The final update is always
        printed when ``close`` is called.
    enabled:
        Set False to disable all output.
    stream:
        Output stream. Defaults to stdout so it appears with normal research
        logs on both Windows and Unix shells.
    """

    label: str
    total: int
    every: int = 25
    enabled: bool = True
    stream: object = sys.stdout
    width: int = 28
    started_at: float = field(default_factory=time.perf_counter)
    last_done: int = 0
    closed: bool = False

    def update(self, done: int, *, force: bool = False) -> None:
        if not self.enabled or self.total <= 0 or self.closed:
            return
        done_i = max(0, min(int(done), int(self.total)))
        every_i = max(1, int(self.every))
        if not force and done_i < self.total and done_i % every_i != 0:
            return
        if not force and done_i == self.last_done:
            return
        self.last_done = done_i
        elapsed = max(0.0, time.perf_counter() - self.started_at)
        rate = done_i / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done_i) / rate if rate > 0 else float("nan")
        frac = min(1.0, max(0.0, done_i / self.total))
        filled = int(round(self.width * frac))
        bar = "#" * filled + "." * (self.width - filled)
        msg = (
            f"{self.label} [{bar}] {done_i:,}/{self.total:,} "
            f"({frac * 100:5.1f}%) elapsed={format_seconds(elapsed)} "
            f"eta={format_seconds(eta)} rate={rate:.2f}/s"
        )
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if is_tty:
            print("\r" + msg, end="\n" if done_i >= self.total else "", file=self.stream, flush=True)
        else:
            print(msg, file=self.stream, flush=True)

    def step(self, inc: int = 1, *, force: bool = False) -> None:
        self.update(self.last_done + int(inc), force=force)

    def close(self) -> None:
        if self.closed:
            return
        self.update(self.total, force=True)
        self.closed = True

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if exc_type is None:
            self.close()
        else:
            self.closed = True


def progress_iter(
    iterable: Iterable[T],
    *,
    label: str,
    total: int | None = None,
    every: int = 25,
    enabled: bool = True,
) -> Iterator[T]:
    """Yield items from ``iterable`` while reporting progress."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = 0
    reporter = ProgressReporter(label=label, total=int(total or 0), every=every, enabled=enabled)
    for i, item in enumerate(iterable, start=1):
        yield item
        reporter.update(i)
    reporter.close()
