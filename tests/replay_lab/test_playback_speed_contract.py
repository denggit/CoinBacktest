from pathlib import Path


def test_playback_speed_is_remapped_and_batches_fast_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    html = (root / "human_replay_lab" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "human_replay_lab" / "static" / "app.js").read_text(encoding="utf-8")
    assert '<option value="slow">慢</option>' in html
    assert '<option value="normal" selected>正常</option>' in html
    assert '<option value="fast">快</option>' in html
    assert '<option value="veryfast">很快</option>' in html
    assert 'slow: {minutes: 1, delayMs: 300}' in js
    assert 'normal: {minutes: 1, delayMs: 45}' in js
    assert 'fast: {minutes: 2, delayMs: 20}' in js
    assert 'veryfast: {minutes: 5, delayMs: 5}' in js
    assert 'await step(cfg.minutes)' in js
