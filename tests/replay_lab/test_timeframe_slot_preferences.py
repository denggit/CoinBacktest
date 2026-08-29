from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "human_replay_lab" / "static" / "index.html"
APP = ROOT / "human_replay_lab" / "static" / "app.js"


def test_six_editable_timeframe_slots_are_rendered():
    html = INDEX.read_text(encoding="utf-8")
    assert html.count('class="tf-slot-select"') == 6
    assert 'data-slot="0"' in html
    assert 'data-slot="5"' in html
    assert 'resetTimeframeSlotsBtn' in html


def test_default_slots_and_supported_timeframes_are_explicit():
    js = APP.read_text(encoding="utf-8")
    assert "const DEFAULT_TIMEFRAME_SLOTS = ['30m', '15m', '5m', '2m', '1m', '4H'];" in js
    assert "const SUPPORTED_TIMEFRAMES_UI = ['1m', '2m', '5m', '15m', '30m', '1H', '4H', '1D'];" in js


def test_slot_preferences_persist_and_fit_does_not_reset_timeframe():
    js = APP.read_text(encoding="utf-8")
    assert "humanReplayLab.timeframeSlots.v1" in js
    assert "window.localStorage?.setItem(TIMEFRAME_SLOT_STORAGE_KEY" in js
    assert "window.localStorage?.getItem(TIMEFRAME_SLOT_STORAGE_KEY" in js
    assert "pane.visibleCount = PANE_DEFAULTS[pane.id].visibleCount;" in js
    fit_block = js[js.index("for (const btn of document.querySelectorAll('.pane-fit'))"):]
    fit_block = fit_block[: fit_block.index("for (const b of document.querySelectorAll('.bias'))")]
    assert "pane.timeframe =" not in fit_block
    assert "DEFAULT_TIMEFRAME_SLOTS" not in fit_block


def test_editing_slot_updates_saved_slot_not_only_current_chart():
    js = APP.read_text(encoding="utf-8")
    assert "async function configureTimeframeSlot(index, tf)" in js
    assert "state.timeframeSlots[index] = tf;" in js
    assert "persistTimeframeSlotPreferences();" in js
    assert "await changeMainTimeframe(tf, {slotIndex: index});" in js
