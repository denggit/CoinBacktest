#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility import for the research-owned panic episode detector.

New code should import from ``.common.panic_episode``.  This module remains only
so existing local tests or notebooks using the old research path keep working.
"""

from .common.panic_episode import (  # noqa: F401
    PanicEpisode,
    PanicEpisodeConfig,
    PanicEpisodeResult,
    PanicNode,
    detect_panic_episodes,
)

__all__ = [
    "PanicEpisode",
    "PanicEpisodeConfig",
    "PanicEpisodeResult",
    "PanicNode",
    "detect_panic_episodes",
]
