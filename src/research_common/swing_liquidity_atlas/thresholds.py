#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast first-threshold search structures used by the lifecycle builder."""

from __future__ import annotations

import numpy as np


class SegmentThresholdIndex:
    """Iterative min/max segment tree with first-match queries in O(log n)."""

    def __init__(self, values: np.ndarray):
        arr = np.asarray(values, dtype=float)
        self.n = int(len(arr))
        size = 1
        while size < self.n:
            size <<= 1
        self.size = size
        self.min_tree = np.full(size * 2, np.inf, dtype=float)
        self.max_tree = np.full(size * 2, -np.inf, dtype=float)
        finite = np.where(np.isfinite(arr), arr, np.nan)
        self.min_tree[size : size + self.n] = np.where(np.isfinite(finite), finite, np.inf)
        self.max_tree[size : size + self.n] = np.where(np.isfinite(finite), finite, -np.inf)
        for node in range(size - 1, 0, -1):
            self.min_tree[node] = min(self.min_tree[node * 2], self.min_tree[node * 2 + 1])
            self.max_tree[node] = max(self.max_tree[node * 2], self.max_tree[node * 2 + 1])

    def first_leq(self, start: int, end: int, threshold: float) -> int:
        if self.n == 0 or start > end or start >= self.n or end < 0 or not np.isfinite(threshold):
            return -1
        return self._first_leq(1, 0, self.size - 1, max(0, int(start)), min(self.n - 1, int(end)), float(threshold))

    def _first_leq(self, node: int, left: int, right: int, ql: int, qr: int, threshold: float) -> int:
        if right < ql or left > qr or self.min_tree[node] > threshold:
            return -1
        if left == right:
            return left if left < self.n else -1
        mid = (left + right) // 2
        found = self._first_leq(node * 2, left, mid, ql, qr, threshold)
        if found >= 0:
            return found
        return self._first_leq(node * 2 + 1, mid + 1, right, ql, qr, threshold)

    def first_geq(self, start: int, end: int, threshold: float) -> int:
        if self.n == 0 or start > end or start >= self.n or end < 0 or not np.isfinite(threshold):
            return -1
        return self._first_geq(1, 0, self.size - 1, max(0, int(start)), min(self.n - 1, int(end)), float(threshold))

    def _first_geq(self, node: int, left: int, right: int, ql: int, qr: int, threshold: float) -> int:
        if right < ql or left > qr or self.max_tree[node] < threshold:
            return -1
        if left == right:
            return left if left < self.n else -1
        mid = (left + right) // 2
        found = self._first_geq(node * 2, left, mid, ql, qr, threshold)
        if found >= 0:
            return found
        return self._first_geq(node * 2 + 1, mid + 1, right, ql, qr, threshold)


class FenwickTree:
    def __init__(self, size: int):
        self.values = np.zeros(int(size) + 1, dtype=np.int64)

    def add(self, index: int, delta: int) -> None:
        i = int(index) + 1
        while i < len(self.values):
            self.values[i] += int(delta)
            i += i & -i

    def prefix_sum(self, end_exclusive: int) -> int:
        i = int(end_exclusive)
        total = 0
        while i > 0:
            total += int(self.values[i])
            i -= i & -i
        return total

    def range_sum(self, left: int, right_exclusive: int) -> int:
        if right_exclusive <= left:
            return 0
        return self.prefix_sum(right_exclusive) - self.prefix_sum(left)
