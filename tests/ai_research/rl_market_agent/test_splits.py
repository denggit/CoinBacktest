from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai_research.rl_market_agent.dataset import DatasetCatalog
from src.ai_research.rl_market_agent.shards import ShardStore
from src.ai_research.rl_market_agent.splits import load_purged_window, make_purged_window


def _write_month(tmp_path, shard_id: str, idx: pd.DatetimeIndex, sealed: bool = False):
    store = ShardStore(tmp_path / "cache", project_root=tmp_path)
    features = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    labels = pd.DataFrame({"h360__final_return": np.arange(len(idx), dtype=float)}, index=idx)
    flags = pd.DataFrame({"sealed_holdout": int(sealed)}, index=idx)
    store.write(
        shard_id=shard_id, features=features, labels=labels, flags=flags,
        sealed_holdout=sealed,
        extra_metadata={"feature_names": ["f"], "label_names": ["h360__final_return"], "flag_names": ["sealed_holdout"]},
    )


def test_horizon_aware_purge_blocks_labels_crossing_right_boundary(tmp_path):
    idx = pd.date_range("2025-12-31 17:55", "2025-12-31 23:55", freq="5min")
    _write_month(tmp_path, "2025-12", idx)
    catalog = DatasetCatalog(tmp_path / "cache", project_root=tmp_path)
    window = make_purged_window("TRAIN", "2025-12-31 17:55", "2026-01-01 00:00", 360)
    loaded = load_purged_window(catalog, window, sealed_holdout_start="2026-01-01 00:00")
    ts = pd.to_datetime(loaded.timestamps_ns, unit="ns")
    assert ts.max() == pd.Timestamp("2025-12-31 18:00")
    assert loaded.rows_before_purge > loaded.rows_after_purge
    assert ((ts + pd.Timedelta(minutes=359)) < pd.Timestamp("2026-01-01 00:00")).all()


def test_unsealed_catalog_refuses_window_beyond_seal(tmp_path):
    idx = pd.date_range("2025-12-01", periods=2, freq="5min")
    _write_month(tmp_path, "2025-12", idx)
    catalog = DatasetCatalog(tmp_path / "cache", project_root=tmp_path)
    window = make_purged_window("BAD", "2025-12-01", "2026-01-02", 60)
    with pytest.raises(PermissionError):
        load_purged_window(catalog, window, sealed_holdout_start="2026-01-01")
