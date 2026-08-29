from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.rl_market_agent.shards import ShardStore


def test_shard_round_trip_is_mmap_compatible(tmp_path):
    idx = pd.date_range("2026-01-01", periods=3, freq="5min")
    f = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    y = pd.DataFrame({"y": [0.1, 0.2, 0.3]}, index=idx)
    flags = pd.DataFrame({"sealed": [0, 0, 1]}, index=idx)
    store = ShardStore(tmp_path / "cache", project_root=tmp_path)
    record = store.write(shard_id="2026-01", features=f, labels=y, flags=flags, sealed_holdout=True, extra_metadata={})
    assert store.exists("2026-01")
    arr = np.load(tmp_path / record.features_path, mmap_mode="r")
    assert arr.dtype == np.float32
    assert arr.shape == (3, 1)
