#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JSON-backed ETH edge library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.experiment.models import normalize_id, utc_now_iso

from .models import EdgeRecord


DEFAULT_EDGE_LIBRARY_PATH = Path("edge_library") / "registry.json"


class EdgeLibrary:
    """Read and update the repository-level edge library."""

    def __init__(self, path: str | Path = DEFAULT_EDGE_LIBRARY_PATH) -> None:
        self.path = Path(path)

    def load(self) -> list[EdgeRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        items = data.get("edges", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"invalid edge library format: {self.path}")
        return [EdgeRecord.from_dict(item) for item in items]

    def save(self, records: Iterable[EdgeRecord]) -> None:
        rows = sorted((record.to_dict() for record in records), key=lambda x: x["id"])
        payload = {
            "schema_version": 1,
            "updated_at": utc_now_iso(),
            "edges": rows,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get(self, edge_id: str) -> EdgeRecord | None:
        target = normalize_id(edge_id)
        for record in self.load():
            if record.id == target:
                return record
        return None

    def upsert(self, record: EdgeRecord) -> EdgeRecord:
        records = self.load()
        out: list[EdgeRecord] = []
        replaced = False
        for current in records:
            if current.id == record.id:
                out.append(record)
                replaced = True
            else:
                out.append(current)
        if not replaced:
            out.append(record)
        self.save(out)
        return record

    def update(self, edge_id: str, **changes: object) -> EdgeRecord:
        current = self.get(edge_id)
        if current is None:
            raise KeyError(f"edge not found: {edge_id}")
        updated = current.with_update(**changes)
        return self.upsert(updated)
