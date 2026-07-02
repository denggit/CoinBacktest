#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import io

from src.research_common.progress import ProgressReporter, format_seconds, progress_iter


def test_format_seconds_compact() -> None:
    assert format_seconds(7) == "7s"
    assert format_seconds(65) == "1m05s"
    assert format_seconds(3661) == "1h01m"


def test_progress_reporter_writes_final_line_to_stream() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(label="test", total=3, every=2, enabled=True, stream=stream)
    reporter.update(1)
    reporter.update(2)
    reporter.close()

    text = stream.getvalue()
    assert "test" in text
    assert "2/3" in text
    assert "3/3" in text


def test_progress_iter_yields_all_items() -> None:
    out = list(progress_iter([1, 2, 3], label="items", every=1, enabled=False))

    assert out == [1, 2, 3]
