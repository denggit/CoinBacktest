from __future__ import annotations

from pathlib import Path

from src.edge_library import EdgeLibrary, EdgeRecord
from src.experiment import ExperimentRecord, ExperimentRegistry


def test_edge_library_upsert_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "edge_registry.json"
    library = EdgeLibrary(path)
    record = EdgeRecord(
        id="eth_edge_test",
        name="Test Edge",
        family="unit_test",
        status="edge_found",
        data_required=("1m_trade_bar",),
    )

    library.upsert(record)
    loaded = library.get("ETH_EDGE_TEST")

    assert loaded is not None
    assert loaded.id == "ETH_EDGE_TEST"
    assert loaded.status == "edge_found"
    assert loaded.data_required == ("1m_trade_bar",)


def test_experiment_registry_upsert_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "experiment_registry.json"
    registry = ExperimentRegistry(path)
    record = ExperimentRecord(
        id="eth_research_test",
        title="Research Test",
        stage="research",
        status="researching",
        family="unit_test",
    )

    registry.upsert(record)
    loaded = registry.get("ETH_RESEARCH_TEST")

    assert loaded is not None
    assert loaded.id == "ETH_RESEARCH_TEST"
    assert loaded.stage == "research"
    assert loaded.status == "researching"
