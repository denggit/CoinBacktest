#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JSON-backed experiment registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import ExperimentRecord, normalize_id, utc_now_iso


DEFAULT_REGISTRY_PATH = Path("experiments") / "registry.json"


class ExperimentRegistry:
    """Read and update the repository-level experiment registry."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path)

    def load(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        items = data.get("experiments", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"invalid experiment registry format: {self.path}")
        return [ExperimentRecord.from_dict(item) for item in items]

    def save(self, records: Iterable[ExperimentRecord]) -> None:
        rows = sorted((record.to_dict() for record in records), key=lambda x: x["id"])
        payload = {
            "schema_version": 1,
            "updated_at": utc_now_iso(),
            "experiments": rows,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        target = normalize_id(experiment_id)
        for record in self.load():
            if record.id == target:
                return record
        return None

    def upsert(self, record: ExperimentRecord) -> ExperimentRecord:
        records = self.load()
        out: list[ExperimentRecord] = []
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

    def update(self, experiment_id: str, **changes: object) -> ExperimentRecord:
        current = self.get(experiment_id)
        if current is None:
            raise KeyError(f"experiment not found: {experiment_id}")
        updated = current.with_update(**changes)
        return self.upsert(updated)
