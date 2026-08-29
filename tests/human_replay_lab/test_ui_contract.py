from pathlib import Path


def test_ui_defaults_to_okx_soxl_0730_workflow_and_magnet() -> None:
    root = Path(__file__).resolve().parents[2]
    html = (root / "human_replay_lab" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "human_replay_lab" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'value="SOXL-USDT-SWAP" readonly' in html
    assert 'OKX 本地 1m' in html
    assert 'id="magnetToggle" type="checkbox" checked' in html
    assert 'id="sourceInfo"' in html
    assert "loadHealth" in js
    assert "snap_field" in js
    assert "raw_clicked_price" in js
    assert "appendIncrementalBars" in js
