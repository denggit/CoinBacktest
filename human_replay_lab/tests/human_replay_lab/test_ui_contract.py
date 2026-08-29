from pathlib import Path


def test_ui_defaults_to_soxl_0730_workflow_and_magnet() -> None:
    root = Path(__file__).resolve().parents[2]
    html = (root / "human_replay_lab" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "human_replay_lab" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'value="SOXL" readonly' in html
    assert 'id="magnetToggle" type="checkbox" checked' in html
    assert 'type="date" value="2025-01-15"' in html
    assert "snap_field" in js
    assert "raw_clicked_price" in js
    assert "appendIncrementalBars" in js
