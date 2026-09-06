from pathlib import Path


def test_tradingview_style_replay_ui_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    html = (root / "human_replay_lab" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "human_replay_lab" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'value="ETH-USDT-SWAP"' in html
    for timeframe in ("1m", "2m", "5m", "15m", "30m", "1H", "4H", "1D"):
        assert f'data-timeframe="{timeframe}"' in html
    for tool in ("trend", "horizontal-ray", "ray", "rectangle", "vertical", "ruler", "long-position", "short-position", "callout"):
        assert f'data-tool="{tool}"' in html
    assert 'id="magnetToggle"' in html
    assert 'data-magnet-mode="weak"' in html
    assert 'id="magnetModeBadge">弱<' in html
    assert 'id="lockToggle"' in html
    assert 'id="trashBtn"' in html
    assert 'id="drawingEditBar"' in html
    assert 'id="autoScaleBtn"' in html
    assert 'id="colorPalette"' in html
    assert 'id="accountInput"' in html
    assert "LIMIT_FEE = 0.0002" in js
    assert "MARKET_FEE = 0.0005" in js
    assert "MARKET_SLIPPAGE = 0.0002" in js
    assert "localStorage" in js
    assert "event.shiftKey" in js
    assert "loadMoreHistory" in js
    assert "update-order" in js
    assert "rectangleHandles" in js
    assert "function rayEndPoint" in js
    assert "function calloutGeometry" in js
    assert "drawing.type==='ray'?rayEndPoint" in js
    assert "timeframeMinutes" in js
    assert 'id="tradeTicket"' in html
    assert 'data-open-side="LONG"' in html
    assert 'data-open-side="SHORT"' in html
    assert 'id="riskPct" type="number" min="0.01" max="100" step="any" value="1"' in html
    assert 'id="positionForm" novalidate' in html
    assert "function validatePositionTicket" in js
    assert "reportValidity" not in js
    assert "!drawing.preview&&!state.drawingInteraction" in js
    assert "MAGNET_MODES = ['weak','strong','off']" in js
    assert "MAGNET_WEAK_THRESHOLD_PX = 12" in js
    assert "canvasPointFromEvent(event,{snap:state.magnetMode})" in js
    assert "magnetMode: 'weak'" in js
    assert "prefs.magnet === false ? 'off' : 'weak'" in js
    assert "data-dialog-close" in html
    assert '<option value="80" selected>1×</option>' in html
    assert '<option value="16">5×</option>' in html
    assert "value || 80" in js
    assert "function drawTradeOverlays" in js
    assert "function drawTradeMarkers" in js
    assert "function drawEntryMarker" in js
    assert "function drawExitMarker" in js
    assert "drawCrosshair(bars,range,geo);drawTradeMarkers" in js
    crosshair = js[js.index("function drawCrosshair") : js.index("function drawChart")]
    assert "state.hoverPoint.x" in crosshair
    assert "state.hoverPoint.snapX" in crosshair
    assert "const hoverBars=visibleBars(),rawIndex=" in js
    assert "hoverBars[state.hoverIndex]" in js
    assert "state.hoverIndex=null;state.hoverPoint=null;state.hoverDrawingId=null" in js
    assert "state.hoverPoint={x,y:point.y,snapX:point.x,snapField:point.snapField}" in js
    assert "const direct=bars.findIndex(bar=>String(bar.time)===String(time))" in js
    assert "const snappedToBar=Boolean(snapField)&&Boolean(bar)" in js
    assert "pointPixels(drawing.a,bars,range,geo,draft)" in js
    assert "lastX=xForIndex(lastIndex,bars,plot)" in js
    assert "xForIndex(lastIndex,bars,plot)+(target-last)/step*slot" in js
    assert "text=isLong?'B':'S'" in js
    assert "label=isTp?'TP':isAmbiguous?'SL*':'SL'" in js
    draw_chart = js[js.index("function drawChart()") : js.index("function resizeCanvas()")]
    assert draw_chart.index("drawDrawings(bars,range,geo)") < draw_chart.index("drawTradeMarkers(bars,range,geo)")
    assert "function syncAccountInputs" in js
    assert "function positionPricesChanged" in js
    assert "function plannedTradeMetrics" in js
    assert 'id="stopNetLoss"' in html
    assert "riskPerUnit=Math.abs(entryExec-stopLevel)" in js
    assert "stopNetLoss=stopLossPerUnit*qty" in js
    assert "Replay 继续，可准备下一笔" in js
    assert "1R 仅按 Entry → SL 价差计算仓位" in html
    assert "totalSlippage" in js
    assert "手续费 / 滑点" in html
    assert "超出 1R" in js
    assert "function syncMarketTicketToExecution" in js
    assert "state.executionPrice" in js
    assert 'id="executionPreview"' in html
    assert "净目标 R:R" in html
    assert "Entry 价格已固定" in js
    assert "Position 显示位置已保存；交易价格没有变化" in js
    assert "ticketAccountOverridden" in js
    assert "下一笔 Account Size 已跟随最新权益" in js
    assert "function positionHandles" in js
    assert "take: {x:p.x1,y:p.takeY}" in js
    assert "entry: {x:p.x1,y:p.entryY}" in js
    assert "stop: {x:p.x1,y:p.stopY}" in js
    assert "width: {x:p.x2,y:p.entryY}" in js
    assert "part:'activate'" in js
    assert "Position 计划已创建" in js
    assert "take_profit_before_entry" in js
    assert "挂单失效" in js
    assert "Replay 自动撤单" in js
    assert "data-close-trade" not in html
    assert "data-close-trade" not in js
    assert "function updateTradeTicketAvailability" in js
    assert "function updateEndEpisodeAvailability" in js
    assert "持仓等待 TP / SL" in js
    assert "state.episode.status!=='active'" in js
    assert "本次训练已结束；请开始新训练后再下单" in js
    assert "MANUAL_CLOSE" not in js
