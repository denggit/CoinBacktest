from pathlib import Path


def test_ui_defaults_to_okx_soxl_0730_workflow_and_magnet() -> None:
    root = Path(__file__).resolve().parents[2]
    html = (root / "human_replay_lab" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "human_replay_lab" / "static" / "app.js").read_text(encoding="utf-8")
    assert '<select id="symbol">' in html
    assert 'readonly' not in html[html.index('Symbol'):html.index('随机', html.index('Symbol'))]
    assert 'crypto_history.db' in html
    assert 'id="magnetToggle" type="checkbox" checked' in html
    assert 'id="sourceInfo"' in html
    assert 'id="orderType"' in html
    assert '<option value="limit" selected>' in html
    assert 'id="cancelOrderBtn"' in html
    assert "loadHealth" in js
    assert "snap_field" in js
    assert "raw_clicked_price" in js
    assert "appendIncrementalBars" in js
    assert 'data-pane="main"' in html
    assert html.count('class="chart-card') == 1
    assert 'id="timeframeSlots"' in html
    assert html.count('class="tf-slot') >= 6
    assert "DEFAULT_TIMEFRAME_SLOTS = ['30m', '15m', '5m', '2m', '1m', '4H']" in js
    assert "is_partial" in js
    assert "形成中" in html
    assert "cancelLatestLimitOrder" in js
    assert "activeLimitOrders" in js
    assert "limit_price" in js
    assert 'id="entryPriceInput"' in html
    assert 'id="slPriceInput"' in html
    assert 'id="tpPriceInput"' in html
    assert 'id="fillEntryBtn"' in html
    assert 'id="rewind1Btn"' in html and 'id="rewind5Btn"' in html and 'id="rewind15Btn"' in html
    assert "async function rewind" in js
    assert "stop_loss" in js and "take_profit" in js
    assert '界面全部显示北京时间' in html
    assert 'event_time_bjt' in js
    assert 'time_bjt' in js
    assert 'market_open_bjt' in js
    assert 'annotation-delete-btn' in js
    assert 'delete-annotation' in js
    assert 'Liquidity / Target / 共享线可在这里直接删除' in html
    assert 'ETH' in html

