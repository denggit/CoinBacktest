from __future__ import annotations

import pandas as pd
import pytest

from src.ai_research.rl_market_agent.dataset import DatasetCatalog
from src.ai_research.rl_market_agent.shards import ShardStore


def test_sealed_shard_requires_explicit_opt_in(tmp_path):
    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    f = pd.DataFrame({"a": [1.0, 2.0]}, index=idx)
    y = pd.DataFrame({"y": [0.1, 0.2]}, index=idx)
    flags = pd.DataFrame({"sealed_holdout": [1, 1]}, index=idx)
    store = ShardStore(tmp_path / "cache", project_root=tmp_path)
    store.write(
        shard_id="2026-01", features=f, labels=y, flags=flags, sealed_holdout=True,
        extra_metadata={"feature_names":["a"], "label_names":["y"], "flag_names":["sealed_holdout"]},
    )
    catalog = DatasetCatalog(tmp_path / "cache", project_root=tmp_path)
    with pytest.raises(PermissionError):
        catalog.load("2026-01")
    final_audit = DatasetCatalog(tmp_path / "cache", project_root=tmp_path, allow_sealed=True)
    assert final_audit.load("2026-01").sealed_holdout is True


def test_whole_shard_training_iterator_is_blocked(tmp_path):
    catalog = DatasetCatalog(tmp_path / "cache", project_root=tmp_path)
    with pytest.raises(RuntimeError, match="unsafe"):
        next(catalog.iter_training_shards())
